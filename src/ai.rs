use crate::fsm_tables::*;
use std::cell::RefCell;

// ---------------------------------------------------------------------------
// AI player: tracks elimination state per row and per column, for each
// ship size (4, 3, 2), plus a simple candidate-set for submarines (size 1).
//
// Coordinates: row 0..9, col 0..9. The FSM tables operate on the inner
// columns 1..8 (0-indexed 0..7 within the table). Outer ring (row/col 0 or 9)
// cells can never be hit by a ship of size >= 2, so they are excluded from
// those FSMs entirely, but can still hold a submarine.
// ---------------------------------------------------------------------------

const SHIP_SIZES: [usize; 3] = [4, 3, 2];

#[derive(Clone, Copy, Debug)]
struct LineState {
    s4: usize,
    s3: usize,
    s2: usize,
}

impl LineState {
    fn new() -> Self {
        LineState {
            s4: INITIAL_STATE_SIZE4,
            s3: INITIAL_STATE_SIZE3,
            s2: INITIAL_STATE_SIZE2,
        }
    }
}

/// One salvo whose result bag held at least one 3 (a Cruiser hit). Cruiser
/// exact-cell identification via candidate-WINDOW enumeration (pinpointing
/// which specific cells a Cruiser occupies by narrowing a set of straight-3
/// placements to one) is deliberately not attempted — see 35e8c16, which
/// removed it for a soundness bug: that combinatorial search could declare
/// a phantom Cruiser found by mixing cells from 2 different real ships. What
/// IS derived — `coord_confirmed_cruiser_hit`, see `derive_confirmed_
/// cruiser_hits_by_elimination` — is a structurally different, narrower
/// rule that never reasons about window shapes at all, only literal
/// per-salvo bag arithmetic plus coordinate identity across salvos, so it
/// doesn't reintroduce that failure mode. Mirrors `Cross2Entry`.
#[derive(Clone)]
pub struct Cross3Entry {
    pub coords: [(usize, usize); 3],
    pub values: [usize; 3],
    /// Per-coordinate flag, parallel to `coords`: true once that specific
    /// fired cell has been proven impossible to be a Cruiser cell — either
    /// because its alive value for size 3 has dropped to zero (or it's on
    /// the outer ring, which never holds a ship of size >=2), or because
    /// `derive_confirmed_cruiser_hits_by_elimination` proved this salvo's
    /// "3" is already explained by a different, confirmed coordinate. I.e.
    /// this salvo's ambiguous "3" result can no longer have come from that
    /// particular cell. Recomputed at the end of every round by
    /// `refresh_cross3_entry_flags`. Monotonic: once true, always true.
    pub coord_ruled_out: [bool; 3],
    /// True for a coordinate proven to be a genuine Cruiser hit: this bag
    /// contains a "3", and every OTHER coordinate in it has been ruled out,
    /// so this one is the only cell left that could possibly have produced
    /// it. See `derive_confirmed_cruiser_hits_by_elimination`.
    pub coord_confirmed_cruiser_hit: [bool; 3],
}

/// One salvo whose result bag held at least one 2 (a Frigate hit) — same
/// shape as `Cross3Entry`, one level down in ship size.
#[derive(Clone)]
pub struct Cross2Entry {
    pub coords: [(usize, usize); 3],
    pub values: [usize; 3],
    /// Per-coordinate flag, parallel to `coords`: true once that specific
    /// fired cell has been proven impossible to be a Frigate cell — either
    /// its alive value for size 2 has dropped to zero (or it's on the outer
    /// ring, which never holds a ship of size >=2), or `derive_confirmed_
    /// frigate_hits_by_elimination` proved this salvo's "2" is already
    /// explained by a different, confirmed coordinate. Recomputed at the
    /// end of every round by `refresh_cross2_entry_flags`. Monotonic: once
    /// true, always true.
    pub coord_ruled_out: [bool; 3],
    /// True for a coordinate proven to be a genuine Frigate hit — same
    /// elimination rule as `Cross3Entry::coord_confirmed_cruiser_hit`, one
    /// size down. See `derive_confirmed_frigate_hits_by_elimination`.
    pub coord_confirmed_frigate_hit: [bool; 3],
}

/// One salvo whose result bag held at least one 4 (a Battleship hit),
/// recorded the same way `Cross3Entry` records a Cruiser hit — kept as a
/// history purely for the debug UI's per-salvo view (the actual deduction
/// still runs off the single merged `battleship_candidates` mask, since
/// there's only one Battleship to narrow down, unlike the two Cruisers'
/// separately-tracked cross-3 bags).
#[derive(Clone)]
pub struct Cross4Entry {
    pub coords: [(usize, usize); 3],
    pub values: [usize; 3],
    /// Per-coordinate flag, parallel to `coords` — same convention as
    /// `Cross3Entry::coord_ruled_out`: true once that specific fired cell's
    /// combined size-4 alive value has dropped to zero (or it's on the
    /// outer ring, which never holds a Battleship at all), meaning this
    /// salvo's "4" can no longer have come from that particular cell.
    /// Recomputed every round by `refresh_cross4_entry_flags`.
    pub coord_ruled_out: [bool; 3],
    /// True for a coordinate proven to be a genuine Battleship hit: this
    /// bag contains a "4", and every OTHER coordinate has been ruled out,
    /// so this one is the only cell left that could possibly have produced
    /// it — see `derive_confirmed_battleship_hits_by_elimination`, which
    /// mirrors `derive_confirmed_cruiser_hits_by_elimination`/
    /// `derive_confirmed_frigate_hits_by_elimination` one size up.
    pub coord_confirmed_battleship_hit: [bool; 3],
}

pub struct AiPlayer {
    /// FSM state per row, indexed 0..9 (row 0 and 9 stay at initial state, unused since
    /// inner placements never start/extend there, but harmless to keep uniform).
    row_state: [LineState; 10],
    /// FSM state per column, indexed 0..9.
    col_state: [LineState; 10],
    /// Submarine candidate cells: true if still possible. Includes outer ring.
    sub_candidates: [[bool; 10]; 10],
    /// Cells already fired at.
    fired: [[bool; 10]; 10],
    /// Ships already confirmed sunk, by size, so we stop targeting that size.
    sunk_sizes: [usize; 5], // count of sunk ships per size (index by size 1..4)
    /// Total ships remaining per size (from SHIP_DEFS: 1x4, 2x3, 3x2, 4x1).
    remaining_sizes: [usize; 5],
    /// Running "could the Battleship still be here" mask, narrowed each time a
    /// salvo comes back with a 4 in its result bag. Starts as "every inner cell
    /// is possible" (true), and only ever narrows (cells go from true to false,
    /// never back). Outer-ring cells are always false — the Battleship can't be
    /// there regardless of any cross logic.
    battleship_candidates: [[bool; 10]; 10],
    /// The Battleship's exact 4-cell layout, captured permanently the
    /// moment it's confirmed sunk (see `apply_full_battleship_elimination`)
    /// — unlike
    /// `battleship_candidates`/`battleship_identified` (both live,
    /// hunting-only concepts that get cleared once sunk, since there's
    /// nothing left to hunt for), this is a permanent record for the
    /// board/debug UI to keep rendering even after the live candidate
    /// state is gone. `None` if the ship sank via ordinary fire before
    /// `battleship_identified` ever narrowed things down to a single
    /// window — the exact layout genuinely isn't knowable from fog-of-war
    /// information alone in that case.
    found_battleship: Option<[(usize, usize); 4]>,
    /// Whether at least one salvo with a 4 has been seen yet. Until then,
    /// `battleship_candidates` is a meaningless "everything's possible" default,
    /// not an actual deduction — so we keep this flag to know when it's real.
    battleship_cross_seen: bool,
    /// Cells already fed into the size-4 FSM via cross-deduced elimination, so we
    /// don't redundantly re-drive the same transition every time the running
    /// candidate mask is intersected again.
    battleship_cross_processed: [[bool; 10]; 10],
    /// Count of salvos processed so far whose result bag held at least one 4 (i.e.
    /// contributed a cross-elimination pass). Gates the "exact layout" deduction
    /// in `battleship_identified` — a single salvo's cross union essentially never
    /// collapses to a bare 4-cell line, but this makes that requirement explicit
    /// rather than relying on it being true by construction.
    four_bearing_salvo_count: usize,
    /// Every 4-bearing salvo processed so far, in order — see `Cross4Entry`.
    cross4_entries: Vec<Cross4Entry>,
    /// Cells already fed into the size-3/size-2 FSMs via the "Battleship's
    /// neighbours can't hold another ship" deduction, so repeated calls once the
    /// ship is identified don't redrive the same transition twice.
    battleship_adjacency_processed: [[bool; 10]; 10],
    /// Every 3-bearing salvo processed so far, in order. Kept as a running
    /// history — there are 2 Cruisers, so two different salvos might be hits
    /// on two entirely different ships.
    cross3_entries: Vec<Cross3Entry>,
    /// Every 2-bearing salvo processed so far, in order — see `Cross2Entry`.
    /// Kept as a running history for the same reason as `cross3_entries`,
    /// just with 3 Frigates instead of 2 Cruisers.
    cross2_entries: Vec<Cross2Entry>,
    /// Per-size toggle (indexed by size 1..4, size 0 unused): when true for
    /// the size currently being hunted, `choose_shots`/`ai_suggest` may
    /// recommend a cell that's already been fired at instead of always
    /// excluding it. A debug/experimentation switch — manual firing via
    /// `Game::fire` still rejects already-fired cells regardless, since this
    /// only relaxes what the AI is willing to *suggest*.
    refire_allowed: [bool; 5],
    /// Debug/experimentation switch: when true, `current_target_size` never
    /// advances to 2 (Frigates) once every Cruiser is sunk — it keeps
    /// reporting 3 instead, indefinitely, even though nothing is left to
    /// eliminate at that size any more (so `choose_shots` effectively
    /// degrades to an arbitrary unfired-cell picker while frozen). Has no
    /// effect while size 4 or size 3 still has ships left to find; if every
    /// Frigate happens to get sunk anyway (e.g. via incidental decoy hits)
    /// despite never being deliberately targeted, `current_target_size`
    /// still moves on to 1 normally.
    freeze_before_frigates: bool,
    /// Every salvo ever processed, in order — coordinates and result values
    /// exactly as fired, regardless of what they contained (unlike
    /// `cross3_entries`/`cross2_entries`, which only keep salvos with at
    /// least one 3 or 2 respectively). Needed for `cruiser_heatmap`/
    /// `frigate_heatmap`'s full-history consistency check: a salvo with
    /// NEITHER a 3 nor a 2 is still real evidence (it proves none of its 3
    /// cells hold that ship size), and skipping it would understate how
    /// much is actually known.
    salvo_history: Vec<([(usize, usize); 3], [usize; 3])>,
    /// The Cruisers' exact 6-cell layout, once `update_fsm_and_resolve` has
    /// locked it in — `None` until then. Unlike the old always-automatic
    /// elimination this replaced, this is only ever set by that explicit,
    /// user-triggered call (see its doc comment for why), so a cell can be
    /// permanently excluded from `consistent_frigate_candidates` even
    /// while the RAW `consistent_cruiser_candidates` hasn't collapsed to a
    /// single hypothesis on its own — only the cross-reasoned identification
    /// has. See `cells_confirmed_cruiser_or_adjacent`.
    cruiser_layout_locked: Option<Vec<(usize, usize)>>,
    /// The Frigates' exact 6-cell layout (all 3 ships), once
    /// `update_fsm_and_resolve` has locked it in — mirrors
    /// `cruiser_layout_locked` one size down. `None` until then.
    frigate_layout_locked: Option<Vec<(usize, usize)>>,
    /// Per-cell: whether this cell has already used its one-time "extra"
    /// refire granted by `update_fsm_and_resolve`'s companion feature,
    /// disambiguation-with-refire (see `disambiguation_shots`'
    /// `allow_refire` parameter and `Game::fire`'s
    /// `is_disambiguation_extra_refire` gate). Independent of the general
    /// `refire_allowed` debug toggle and the Battleship/anchored-isolation
    /// refire allowances — this one is capped at exactly one use per cell,
    /// since (unlike those) it exists specifically to let the player pay
    /// for new information with an extra shot, not to run indefinitely.
    disambiguation_extra_refire_used: [[bool; 10]; 10],
    /// Memoized `consistent_cruiser_candidates`/`consistent_frigate_candidates`,
    /// keyed by `salvo_history.len()` — every JSON accessor the frontend
    /// polls (`resolution_status_json`, `cruiser_heatmap_json`,
    /// `cruiser_heatmap_fraction_json`, `cruiser_identified_json`, their
    /// Frigate counterparts, ...) independently calls these, and the
    /// cross-reasoning path (`jointly_consistent_hypothesis_pairs`) calls
    /// them again on top of that. Recomputing the full O(n^2)/O(n^3) window
    /// enumeration from scratch on every one of those calls made a single
    /// early-game move take well over a minute; caching by salvo count
    /// (the only thing these lists depend on) means each is computed once
    /// per real game state and reused everywhere else.
    cruiser_candidates_cache: RefCell<Option<(usize, Vec<std::collections::HashSet<(usize, usize)>>)>>,
    frigate_candidates_cache: RefCell<Option<(usize, Vec<std::collections::HashSet<(usize, usize)>>)>>,
}

const INNER_LO: usize = 1;
const INNER_HI: usize = 8; // inclusive

impl AiPlayer {
    pub fn new() -> Self {
        let mut remaining_sizes = [0usize; 5];
        remaining_sizes[4] = 1;
        remaining_sizes[3] = 2;
        remaining_sizes[2] = 3;
        remaining_sizes[1] = 4;

        AiPlayer {
            row_state: [LineState::new(); 10],
            col_state: [LineState::new(); 10],
            sub_candidates: [[true; 10]; 10],
            fired: [[false; 10]; 10],
            sunk_sizes: [0; 5],
            remaining_sizes,
            battleship_candidates: Self::initial_battleship_candidates(),
            found_battleship: None,
            battleship_cross_seen: false,
            battleship_cross_processed: [[false; 10]; 10],
            four_bearing_salvo_count: 0,
            cross4_entries: Vec::new(),
            battleship_adjacency_processed: [[false; 10]; 10],
            cross3_entries: Vec::new(),
            cross2_entries: Vec::new(),
            cruiser_layout_locked: None,
            frigate_layout_locked: None,
            disambiguation_extra_refire_used: [[false; 10]; 10],
            refire_allowed: [false; 5],
            freeze_before_frigates: false,
            salvo_history: Vec::new(),
            cruiser_candidates_cache: RefCell::new(None),
            frigate_candidates_cache: RefCell::new(None),
        }
    }

    /// Every inner cell starts "possible"; the outer ring never is, regardless of
    /// any cross deduction, since the Battleship can never occupy it.
    fn initial_battleship_candidates() -> [[bool; 10]; 10] {
        let mut grid = [[false; 10]; 10];
        for row in INNER_LO..=INNER_HI {
            for col in INNER_LO..=INNER_HI {
                grid[row][col] = true;
            }
        }
        grid
    }

    fn line_state_for_size(state: &LineState, size: usize) -> usize {
        match size {
            4 => state.s4,
            3 => state.s3,
            2 => state.s2,
            _ => unreachable!(),
        }
    }

    fn set_line_state_for_size(state: &mut LineState, size: usize, val: usize) {
        match size {
            4 => state.s4 = val,
            3 => state.s3 = val,
            2 => state.s2 = val,
            _ => unreachable!(),
        }
    }

    /// Eliminate placements of `size` passing through (row, col) in both the row-FSM
    /// and column-FSM, by driving the relevant transition. No-op for cells on the
    /// outer ring (size>=2 ships never occupy those) and no-op for size==1 (submarines
    /// handled separately).
    fn eliminate_size_at(&mut self, row: usize, col: usize, size: usize) {
        if size == 1 {
            return; // submarines handled via sub_candidates directly
        }
        // A size>=2 ship placement lies entirely within the inner 8x8 grid, so a
        // fired cell can only be relevant to either FSM if BOTH its row and column
        // are inner — if either is on the outer ring, this cell can never be part
        // of any such placement at all, and `row_state`/`col_state` for outer-ring
        // lines (0 and 9) must stay untouched at their initial state forever.
        if !(INNER_LO..=INNER_HI).contains(&row) || !(INNER_LO..=INNER_HI).contains(&col) {
            return;
        }
        let table_col = col - INNER_LO; // 0..7
        let cur = Self::line_state_for_size(&self.row_state[row], size);
        let next = match size {
            4 => TRANSITIONS_SIZE4[cur][table_col] as usize,
            3 => TRANSITIONS_SIZE3[cur][table_col] as usize,
            2 => TRANSITIONS_SIZE2[cur][table_col] as usize,
            _ => unreachable!(),
        };
        Self::set_line_state_for_size(&mut self.row_state[row], size, next);

        let table_row = row - INNER_LO;
        let cur = Self::line_state_for_size(&self.col_state[col], size);
        let next = match size {
            4 => TRANSITIONS_SIZE4[cur][table_row] as usize,
            3 => TRANSITIONS_SIZE3[cur][table_row] as usize,
            2 => TRANSITIONS_SIZE2[cur][table_row] as usize,
            _ => unreachable!(),
        };
        Self::set_line_state_for_size(&mut self.col_state[col], size, next);
    }

    /// Apply a miss at (row, col): eliminate all ship sizes >=2 through this cell
    /// (in both row and column FSMs) and remove it as a submarine candidate.
    fn apply_miss(&mut self, row: usize, col: usize) {
        for &size in &SHIP_SIZES {
            self.eliminate_size_at(row, col, size);
        }
        self.sub_candidates[row][col] = false;
    }

    /// Apply a hit of given `size` at (row, col): eliminate all LARGER sizes through
    /// this cell (since a real ship occupies it, only that exact size can be true here),
    /// and remove as submarine candidate (since it's occupied by a non-submarine, unless
    /// size==1).
    fn apply_hit(&mut self, row: usize, col: usize, size: usize) {
        if size == 1 {
            // It's the submarine itself; nothing larger to eliminate through this cell
            // beyond the standard "this is occupied" fact. Still, ships >=2 cannot also
            // occupy this cell (board cells are exclusive), so eliminate those too.
            for &s in &SHIP_SIZES {
                self.eliminate_size_at(row, col, s);
            }
            self.sub_candidates[row][col] = false; // resolved, no longer "candidate to fire at"
            return;
        }
        // Eliminate sizes larger than `size` (since they didn't match).
        for &s in &SHIP_SIZES {
            if s > size {
                self.eliminate_size_at(row, col, s);
            }
        }
        self.sub_candidates[row][col] = false;
    }

    pub fn mark_fired(&mut self, row: usize, col: usize) {
        self.fired[row][col] = true;
    }

    pub fn is_fired(&self, row: usize, col: usize) -> bool {
        self.fired[row][col]
    }

    /// Whether (row, col) is still considered a possible Submarine cell.
    pub fn is_submarine_candidate(&self, row: usize, col: usize) -> bool {
        self.sub_candidates[row][col]
    }

    /// Record a ship of `size` as sunk; used to stop scoring that size once all are found.
    pub fn mark_sunk(&mut self, size: usize) {
        if size >= 1 && size <= 4 {
            self.sunk_sizes[size] += 1;
        }
        if size == 3 {
            // Cruiser discovery (pinpointing exact cells) is deliberately
            // not attempted — see `refresh_cross3_entry_flags`. Re-running
            // this here closes the same one-round gap as the Frigate/
            // Battleship branches: `Game::fire` calls `apply_salvo` (which
            // also refreshes cross-3 flags) BEFORE calling `mark_sunk`, so
            // the exact salvo that sinks a Cruiser would otherwise see a
            // stale (pre-sinking) sunk count during its own apply_salvo call.
            self.refresh_cross3_entry_flags();
        }
        if size == 2 {
            // Frigate discovery (pinpointing exact cells) is deliberately
            // not attempted — see `refresh_cross2_entry_flags`. Re-running
            // it here closes the same one-round gap as the Cruiser/
            // Battleship branches: `Game::fire` calls `apply_salvo` (which
            // also refreshes cross-2 flags) BEFORE calling `mark_sunk`, so
            // the exact salvo that sinks a Frigate would otherwise see a
            // stale (pre-sinking) sunk count during its own apply_salvo call.
            self.refresh_cross2_entry_flags();
        }
        if size == 4 {
            // There is exactly 1 Battleship, so the moment this fires it's
            // unconditionally `size_fully_found(4)` — unlike Cruisers/
            // Frigates, there's no "wait for the LAST one" case to gate on.
            // Whether or not the cross-4 deduction ever narrowed the
            // candidate cross down to a single straight-4 window (a ship
            // can perfectly well sink via ordinary fire before 2+
            // intersecting salvos ever resolve that ambiguity), hunting for
            // size 4 is now entirely over — there is nothing left to search
            // for, so nothing should keep looking. Without this, a
            // still-ambiguous candidate cross from before the ship sank
            // would otherwise linger forever: the board/Cross-4 Bag UI kept
            // showing "possible" cells for a mystery that's already solved.
            self.refresh_cross4_entry_flags();
            self.apply_full_battleship_elimination();
        }
    }

    fn size_fully_found(&self, size: usize) -> bool {
        self.sunk_sizes[size] >= self.remaining_sizes[size]
    }

    /// Called the moment the (single) Battleship is confirmed sunk. The ship's own cells — whether that's
    /// the exact 4-cell layout if `battleship_identified` succeeded, or the
    /// still-broader candidate set if the ship sank via ordinary fire
    /// before the cross-4 deduction ever narrowed a multi-window ambiguity
    /// down to one — must NOT have their own size-4 alive value eliminated
    /// here. A previous version did exactly that (eliminating size 4
    /// unconditionally everywhere), which — combined with those same cells
    /// already correctly having size 2/3 eliminated (a cell can only hold
    /// one ship) — made the just-identified Battleship's own cells satisfy
    /// "no ship of size >=2 possible" and render as dead water, as if the
    /// ship had simply vanished the instant it sank. Only cells that were
    /// NEVER a candidate get eliminated; the candidate set itself is then
    /// cleared (a UI-only concern, unrelated to the FSM) so the board/
    /// Cross-4 Bag stop showing a "still possible" region for a mystery
    /// that's already solved.
    ///
    /// Before that clearing happens, snapshot `battleship_identified` (if
    /// it succeeded) into `found_battleship` — a SEPARATE, permanent record.
    /// Without this, clearing
    /// `battleship_candidates` also empties `battleship_identified`'s own
    /// output (both read the same live mask), so the board loses every
    /// trace of "this was the Battleship" the instant it sinks — no more
    /// candidate outline, but also no permanent "found" marker to replace
    /// it with, unlike a found Cruiser/Frigate's lasting green/purple cells.
    fn apply_full_battleship_elimination(&mut self) {
        self.found_battleship = self.battleship_identified();

        let candidate = self.battleship_candidates;
        for row in INNER_LO..=INNER_HI {
            for col in INNER_LO..=INNER_HI {
                if !candidate[row][col] {
                    self.eliminate_size_at(row, col, 4);
                }
            }
        }
        for row in INNER_LO..=INNER_HI {
            for col in INNER_LO..=INNER_HI {
                self.battleship_candidates[row][col] = false;
            }
        }
    }

    /// Process a salvo result: 3 coordinates and their result values as an unordered
    /// multiset, e.g. coords = [(r1,c1),(r2,c2),(r3,c3)], values = {3,2,0} (any order).
    ///
    /// We do NOT know which coordinate produced which value. For each cell, the only
    /// thing we know for certain is the set of values it COULD have taken across all
    /// valid permutations of `values` onto `coords` (with 3 free cells and no extra
    /// constraints, that's simply "any value in the multiset"). The safe elimination
    /// per cell is then: "no ship size greater than the MAXIMUM value this cell could
    /// possibly have taken passes through here" — using the max keeps every legitimate
    /// possibility open while still letting us eliminate sizes that are impossible
    /// under every permutation.
    pub fn apply_salvo(&mut self, coords: [(usize, usize); 3], values: [usize; 3]) {
        self.salvo_history.push((coords, values));
        for &(r, c) in &coords {
            self.mark_fired(r, c);
        }

        // Because the result is an unordered multiset, any of the 3 cells could have
        // produced any of the 3 values. The only per-cell guarantee that holds across
        // ALL permutations is: "this cell's true result <= max(values)". So the safe
        // uniform bound for elimination is max(values), applied to every fired cell.
        //
        // Special case: if max == 0, all three cells are guaranteed misses (0 0 0 salvo).
        let bound = *values.iter().max().unwrap_or(&0);

        for &(r, c) in &coords {
            if bound == 0 {
                self.apply_miss(r, c);
            } else {
                // Eliminate all ship sizes strictly greater than `bound` through this cell.
                self.apply_hit(r, c, bound);
            }
        }

        // Beyond that ">bound" elimination, every cell's true value is
        // guaranteed to be EXACTLY one of the 3 actual values in this bag —
        // so any ship size that doesn't appear anywhere in `values` at all
        // is impossible at every one of these 3 cells, even sizes SMALLER
        // than `bound` (e.g. bag [3, 1, 0] has no 2 anywhere, ruling out a
        // Frigate at all 3 cells despite 2 < bound == 3). Battleship (4)
        // doesn't need handling here: whenever it's absent, bound < 4
        // already eliminated it via the ">bound" rule above, and the
        // dedicated "no 4 in bag" branch below folds in its own stronger
        // cross-tracking on top.
        if bound > 0 {
            for &size in &[3usize, 2] {
                if !values.contains(&size) {
                    for &(r, c) in &coords {
                        self.eliminate_size_at(r, c, size);
                    }
                }
            }
        }

        // A 4 in the bag means one of these 3 cells is a genuine Battleship hit —
        // we just don't know which. Fold in the cross-elimination trick below.
        //
        // ...unless the Battleship is already fully confirmed (identified via
        // 2+ intersecting cross-4 salvos, or permanently recorded once sunk):
        // then this "4" can only be a deliberate re-fire of an already-known
        // cell (see `anchored_isolation_shot`, which refires a confirmed
        // Battleship cell as a safe anchor for isolating a Cruiser/Frigate
        // cell elsewhere on the board). Running the "don't know which of
        // these 3 cells is the hit" cross logic on that would wrongly treat
        // the OTHER 2 (likely far-apart, unrelated) cells as still-live
        // Battleship candidates for a brand new cross, and intersecting that
        // against history can wipe out the real, already-confirmed window
        // entirely. There's nothing left to deduce about its location, so
        // skip it.
        if bound == 4 && self.battleship_identified().is_none() && self.found_battleship.is_none() {
            self.apply_battleship_cross_elimination(coords, values);
        } else if bound != 4 {
            // No 4 anywhere in this bag means NONE of these 3 cells can hold the
            // Battleship (if one did, its value would be 4, forcing bound == 4).
            // That's a certain exclusion, independent of any cross-deduction —
            // unlike the cross trick, it doesn't need a 4 to reason from, so it
            // applies to the vast majority of ordinary salvos. Without this, the
            // candidate mask only ever narrows via 4-bearing salvos and keeps
            // "possible" cells that plain misses/lesser hits have already ruled
            // out, which can leave it well short of the true 4-cell layout even
            // once the Battleship is fully sunk.
            //
            // The size-4 row/col FSM was already updated for these cells above
            // (via apply_miss/apply_hit eliminating sizes > bound), so just mark
            // them processed to stop a later cross/room pass from re-driving
            // that same FSM transition a second time.
            for &(r, c) in &coords {
                self.battleship_candidates[r][c] = false;
                self.battleship_cross_processed[r][c] = true;
            }

            // Clearing even one cell can break a phantom run some other candidate
            // was relying on for room (e.g. a leftover ambiguous line from an
            // earlier cross-intersection) — re-run the room check so that fallout
            // cascades immediately, not just whenever the next 4-bearing salvo
            // happens to trigger it (which may be never, once the Battleship's
            // real cells are all already found).
            if self.battleship_cross_seen {
                self.prune_candidates_without_room();
            }
        }

        // Once the Battleship's exact cells are pinned down, its neighbours (incl.
        // diagonal) can't hold a Cruiser or Frigate — fold that in every call (it's
        // idempotent per cell, so this is a no-op once already applied).
        if let Some(cells) = self.battleship_identified() {
            self.apply_battleship_adjacency_elimination(cells);
        }

        // A 3 anywhere in the bag (regardless of `bound` — it could be [4,3,0])
        // means one of these 3 cells is a genuine Cruiser hit. Track it alongside
        // every other 3-bearing salvo seen so far.
        if values.contains(&3) {
            self.apply_cruiser_cross_tracking(coords, values);
        }
        // Same idea, one size down: a 2 anywhere in the bag means one of
        // these 3 cells is a genuine Frigate hit.
        if values.contains(&2) {
            self.apply_frigate_cross_tracking(coords, values);
        }

        // Refresh flags so the debug UI reflects anything this salvo proved
        // impossible.
        self.refresh_cross3_entry_flags();
        self.refresh_cross2_entry_flags();
        self.refresh_cross4_entry_flags();
    }

    /// The Battleship forbids orthogonal *and* diagonal adjacency to any other
    /// ship (see `try_place` in lib.rs), so once its 4 cells are known, every cell
    /// touching them rules out a Cruiser (size 3) or Frigate (size 2) — whether or
    /// not that cell has ever been fired at. Submarines are deliberately left
    /// alone at *neighbouring* cells: they only forbid *orthogonal* adjacency, so
    /// a diagonal neighbour of the Battleship could still legitimately hold one.
    ///
    /// The 4 ship cells themselves are a separate, even more direct case: a board
    /// cell can only ever hold one ship, so a cell definitely occupied by the
    /// Battleship definitely isn't a Cruiser, Frigate, or Submarine cell either —
    /// eliminated below regardless of adjacency rules, and regardless of whether
    /// that cell has actually been fired at yet (identified doesn't mean sunk).
    ///
    /// Eliminates unconditionally, even for already-fired cells: a salvo whose
    /// *bound* is 3 or 4 never eliminates size 3 (and a bound of 4 never
    /// eliminates size 2 either) for any of its 3 cells via the normal
    /// apply_hit path, since any of them could ambiguously be the exact match —
    /// see `apply_salvo`. So a ship cell or neighbour that happened to be fired
    /// as part of such an ambiguous salvo (e.g. as a decoy) would otherwise
    /// never get size 3/2 eliminated at all. This is safe to call more than
    /// once on the same cell — the FSM transition tables are idempotent per
    /// column, so re-eliminating an already-excluded one is a no-op — and
    /// `battleship_adjacency_processed` still avoids the redundant work.
    fn apply_battleship_adjacency_elimination(&mut self, ship_cells: [(usize, usize); 4]) {
        for &(row, col) in &ship_cells {
            if self.battleship_adjacency_processed[row][col] {
                continue;
            }
            self.battleship_adjacency_processed[row][col] = true;
            self.eliminate_size_at(row, col, 3);
            self.eliminate_size_at(row, col, 2);
            self.sub_candidates[row][col] = false;
        }

        for &(row, col) in &ship_cells {
            for dr in -1isize..=1 {
                for dc in -1isize..=1 {
                    if dr == 0 && dc == 0 {
                        continue;
                    }
                    let nr = row as isize + dr;
                    let nc = col as isize + dc;
                    if !(INNER_LO as isize..=INNER_HI as isize).contains(&nr)
                        || !(INNER_LO as isize..=INNER_HI as isize).contains(&nc)
                    {
                        continue;
                    }
                    let (nr, nc) = (nr as usize, nc as usize);
                    if ship_cells.contains(&(nr, nc)) {
                        continue; // part of the Battleship itself, not a neighbour
                    }
                    if self.battleship_adjacency_processed[nr][nc] {
                        continue;
                    }
                    self.battleship_adjacency_processed[nr][nc] = true;
                    self.eliminate_size_at(nr, nc, 3);
                    self.eliminate_size_at(nr, nc, 2);
                }
            }
        }
    }

    /// Manual counterpart to the old always-automatic layout elimination —
    /// triggered from the UI's "Update FSM and resolve" button once the
    /// player judges the heatmap has stopped evolving (see
    /// `Game::update_fsm_and_resolve`), rather than firing silently on
    /// every salvo. Attempts to lock in BOTH the Cruisers' and the
    /// Frigates' exact layout — symmetric, since either one identified
    /// first can narrow the other (see `lock_in_cruiser_layout`/
    /// `lock_in_frigate_layout`, which do the actual work and explain the
    /// elimination each performs). Locking one can immediately unlock the
    /// other in the same call: e.g. locking the Cruisers narrows
    /// `cells_confirmed_cruiser_or_adjacent`, which
    /// `consistent_frigate_candidates` excludes from its own window
    /// search, so the Frigate attempt below deliberately runs against
    /// freshly-cleared caches rather than ones left over from before the
    /// Cruiser lock.
    ///
    /// Idempotent and safe to call repeatedly — each side is individually
    /// a no-op once already locked, or if that class isn't cross-reasoning-
    /// identified yet. Returns whether EITHER side actually locked
    /// anything in this call.
    pub fn update_fsm_and_resolve(&mut self) -> bool {
        let cruiser_locked = self.lock_in_cruiser_layout();
        if cruiser_locked {
            // Clear now, not just at the end — the Frigate attempt right
            // below must see the narrower `cells_confirmed_cruiser_or_
            // adjacent` exclusion this just produced, not a stale
            // pre-lock candidate list cached under the same salvo count.
            *self.cruiser_candidates_cache.borrow_mut() = None;
            *self.frigate_candidates_cache.borrow_mut() = None;
        }

        let frigate_locked = self.lock_in_frigate_layout();

        if cruiser_locked || frigate_locked {
            self.refresh_cross3_entry_flags();
            self.refresh_cross2_entry_flags();
            *self.cruiser_candidates_cache.borrow_mut() = None;
            *self.frigate_candidates_cache.borrow_mut() = None;
        }
        cruiser_locked || frigate_locked
    }

    /// Uses `cruiser_identified_cells_refined` (the cross-reasoning-aware
    /// identification), so it can lock in a layout the RAW, non-cross-
    /// reasoned `consistent_cruiser_candidates` hasn't collapsed to a
    /// single hypothesis on its own yet.
    ///
    /// Once locked in (`cruiser_layout_locked`, checked by
    /// `cells_confirmed_cruiser_or_adjacent`):
    /// - Every inner cell that ISN'T one of the 6 real Cruiser cells can
    ///   never hold a Cruiser either (there are only 2, and we now know
    ///   exactly where both are) — eliminate size 3 there. This alone
    ///   already covers every Cruiser-adjacent cell too (they're not among
    ///   the 6 real cells either), so there's no separate neighbour-only
    ///   size-3 pass below — it would be pure redundant idempotent work.
    /// - Every cell adjacent to a real Cruiser cell (including diagonally)
    ///   can never hold a Frigate OR a Battleship either — eliminate size
    ///   2 and size 4 there. THIS genuinely needs the neighbour-only pass:
    ///   unlike size 3, there's no broader "every non-Cruiser-cell loses
    ///   size 2/4" step, since plenty of eligible cells for both exist far
    ///   from any Cruiser. Safe to eliminate size 4 (unlike size 3 or 2 at
    ///   a single unconfirmed cell — see `apply_adjacency_elimination_
    ///   around`) because the full 6-cell layout is known here, so a
    ///   neighbour is provably NOT one of the Cruisers' own cells, and
    ///   there's only 1 Battleship anyway — no risk of mistaking a ship's
    ///   own sibling cell for a rule violation.
    ///
    /// Doesn't itself refresh cross-3/cross-2 entries or clear the
    /// candidate caches — `update_fsm_and_resolve` (the only caller) does
    /// that once, after attempting both this and `lock_in_frigate_layout`.
    fn lock_in_cruiser_layout(&mut self) -> bool {
        if self.cruiser_layout_locked.is_some() {
            return false;
        }
        let cells = self.cruiser_identified_cells_refined();
        if cells.is_empty() {
            return false;
        }
        self.cruiser_layout_locked = Some(cells.clone());

        let real: std::collections::HashSet<(usize, usize)> = cells.iter().copied().collect();
        for row in INNER_LO..=INNER_HI {
            for col in INNER_LO..=INNER_HI {
                if !real.contains(&(row, col)) {
                    self.eliminate_size_at(row, col, 3);
                }
            }
        }

        for &(row, col) in &cells {
            for dr in -1isize..=1 {
                for dc in -1isize..=1 {
                    if dr == 0 && dc == 0 {
                        continue;
                    }
                    let nr = row as isize + dr;
                    let nc = col as isize + dc;
                    if !(INNER_LO as isize..=INNER_HI as isize).contains(&nr) || !(INNER_LO as isize..=INNER_HI as isize).contains(&nc) {
                        continue;
                    }
                    let (nr, nc) = (nr as usize, nc as usize);
                    if real.contains(&(nr, nc)) {
                        continue; // part of a Cruiser itself, not a neighbour
                    }
                    self.eliminate_size_at(nr, nc, 2);
                    self.eliminate_size_at(nr, nc, 4);
                }
            }
        }
        true
    }

    /// Mirrors `lock_in_cruiser_layout` one size down, using
    /// `frigate_identified_cells_refined`. Unlike the Cruiser side, EVERY
    /// elimination here needs an explicit neighbour-only pass: knowing all
    /// 6 real Frigate cells only proves size 2 is dead everywhere else
    /// (the broad, non-neighbour-specific step, mirroring Cruiser's own),
    /// but says nothing on its own about size 3 or size 4 anywhere —
    /// Frigates being smaller than Cruisers, "not a Frigate cell" is a
    /// much weaker statement than "not a Cruiser cell" was on the Cruiser
    /// side, so the size-3/size-4 exclusion has to come specifically from
    /// adjacency (no Cruiser or Battleship may sit next to a confirmed
    /// Frigate), not from a broader "everywhere but these 6 cells" sweep
    /// the way it did one size up.
    fn lock_in_frigate_layout(&mut self) -> bool {
        if self.frigate_layout_locked.is_some() {
            return false;
        }
        let cells = self.frigate_identified_cells_refined();
        if cells.is_empty() {
            return false;
        }
        self.frigate_layout_locked = Some(cells.clone());

        let real: std::collections::HashSet<(usize, usize)> = cells.iter().copied().collect();
        for row in INNER_LO..=INNER_HI {
            for col in INNER_LO..=INNER_HI {
                if !real.contains(&(row, col)) {
                    self.eliminate_size_at(row, col, 2);
                }
            }
        }

        for &(row, col) in &cells {
            for dr in -1isize..=1 {
                for dc in -1isize..=1 {
                    if dr == 0 && dc == 0 {
                        continue;
                    }
                    let nr = row as isize + dr;
                    let nc = col as isize + dc;
                    if !(INNER_LO as isize..=INNER_HI as isize).contains(&nr) || !(INNER_LO as isize..=INNER_HI as isize).contains(&nc) {
                        continue;
                    }
                    let (nr, nc) = (nr as usize, nc as usize);
                    if real.contains(&(nr, nc)) {
                        continue; // part of a Frigate itself, not a neighbour
                    }
                    self.eliminate_size_at(nr, nc, 3);
                    self.eliminate_size_at(nr, nc, 2);
                    self.eliminate_size_at(nr, nc, 4);
                }
            }
        }
        true
    }

    /// Mark a "beam" in both directions through (row, col), reaching `reach` cells
    /// either way (clipped to the inner 8x8 — the only place a ship of size >= 2
    /// can be): every cell that could be part of a ship of length `reach + 1`
    /// which passes through (row, col) — since a hit could land on any cell of
    /// the ship, its other cells can be at most `reach` away along the same line.
    fn mark_cross_reach(mask: &mut [[bool; 10]; 10], row: usize, col: usize, reach: isize) {
        for d in -reach..=reach {
            let cc = col as isize + d;
            if (INNER_LO as isize..=INNER_HI as isize).contains(&cc) {
                mask[row][cc as usize] = true;
            }
            let rr = row as isize + d;
            if (INNER_LO as isize..=INNER_HI as isize).contains(&rr) {
                mask[rr as usize][col] = true;
            }
        }
    }

    /// Mark a length-7 "beam" in both directions through (row, col) — every cell
    /// that could be part of a 4-length ship (Battleship) passing through it.
    fn mark_cross(mask: &mut [[bool; 10]; 10], row: usize, col: usize) {
        Self::mark_cross_reach(mask, row, col, 3);
    }

    /// Given a salvo whose result bag contained a 4, narrow the running Battleship
    /// candidate set: build the union of the 3 crosses centred on each fired
    /// coordinate (the true hit is one of them, we just don't know which), then
    /// intersect that union into `battleship_candidates`. Any cell that falls out
    /// of the running candidate set is now known to be Battleship-free, so we feed
    /// it into the size-4 FSM the same way a real miss would be — even though we
    /// never actually fired there.
    fn apply_battleship_cross_elimination(&mut self, coords: [(usize, usize); 3], values: [usize; 3]) {
        self.cross4_entries.push(Cross4Entry {
            coords,
            values,
            coord_ruled_out: [false; 3],
            coord_confirmed_battleship_hit: [false; 3],
        });

        let mut salvo_union = [[false; 10]; 10];
        for &(r, c) in &coords {
            Self::mark_cross(&mut salvo_union, r, c);
        }

        for row in INNER_LO..=INNER_HI {
            for col in INNER_LO..=INNER_HI {
                if !salvo_union[row][col] {
                    self.battleship_candidates[row][col] = false;
                }
            }
        }
        self.battleship_cross_seen = true;
        self.four_bearing_salvo_count += 1;

        self.drop_candidates(|ai| {
            let mut dead = Vec::new();
            for row in INNER_LO..=INNER_HI {
                for col in INNER_LO..=INNER_HI {
                    if !ai.battleship_candidates[row][col] {
                        dead.push((row, col));
                    }
                }
            }
            dead
        });

        self.prune_candidates_without_room();
    }

    /// Given a salvo whose result bag contained a 3, record the entry — the
    /// true Cruiser hit is one of its 3 coordinates, we just don't know
    /// which yet (see `Cross3Entry::coord_ruled_out`, refreshed every round
    /// by `refresh_cross3_entry_flags`).
    fn apply_cruiser_cross_tracking(&mut self, coords: [(usize, usize); 3], values: [usize; 3]) {
        self.cross3_entries.push(Cross3Entry {
            coords,
            values,
            coord_ruled_out: [false; 3],
            coord_confirmed_cruiser_hit: [false; 3],
        });
    }

    /// Given a salvo whose result bag contained a 2, record the entry — the
    /// true Frigate hit is one of its 3 coordinates, we just don't know which
    /// yet (see `Cross2Entry::coord_ruled_out`, refreshed every round by
    /// `refresh_cross2_entry_flags`).
    fn apply_frigate_cross_tracking(&mut self, coords: [(usize, usize); 3], values: [usize; 3]) {
        self.cross2_entries.push(Cross2Entry {
            coords,
            values,
            coord_ruled_out: [false; 3],
            coord_confirmed_frigate_hit: [false; 3],
        });
    }

    /// Combined "alive" value for `size` at (row, col): the row's horizontal
    /// elimination value at this column, plus the column's vertical elimination
    /// value at this row (see `line_state_score`/the VALUES tables). A value
    /// here is exactly "how many currently-alive placements of this size would
    /// be affected by firing here" — so zero means no alive placement,
    /// horizontal *or* vertical, passes through this cell at all, regardless of
    /// whether the cell itself was ever individually fired or excluded. This is
    /// what the "Ship alive grids" debug view shows (see `alive_grids`), and
    /// what `refresh_cross3_entry_flags` uses to decide a cell is dead for size 3.
    fn alive_value(&self, row: usize, col: usize, size: usize) -> u32 {
        let horizontal = Self::line_state_score(Self::line_state_for_size(&self.row_state[row], size), size, col - INNER_LO);
        let vertical = Self::line_state_score(Self::line_state_for_size(&self.col_state[col], size), size, row - INNER_LO);
        horizontal + vertical
    }

    /// The 3 debug grids for `size` (4, 3, or 2): horizontal alive value,
    /// vertical alive value, and their sum, one entry per inner cell (8x8,
    /// indexed 0..8 for board rows/cols 1..8). For size 3 the combined grid is
    /// exactly the criterion `refresh_cross3_entry_flags` uses.
    pub fn alive_grids(&self, size: usize) -> (Vec<Vec<u32>>, Vec<Vec<u32>>, Vec<Vec<u32>>) {
        let mut horizontal = vec![vec![0u32; 8]; 8];
        let mut vertical = vec![vec![0u32; 8]; 8];
        let mut combined = vec![vec![0u32; 8]; 8];
        for row in INNER_LO..=INNER_HI {
            for col in INNER_LO..=INNER_HI {
                let h = Self::line_state_score(Self::line_state_for_size(&self.row_state[row], size), size, col - INNER_LO);
                let v = Self::line_state_score(Self::line_state_for_size(&self.col_state[col], size), size, row - INNER_LO);
                horizontal[row - INNER_LO][col - INNER_LO] = h;
                vertical[row - INNER_LO][col - INNER_LO] = v;
                combined[row - INNER_LO][col - INNER_LO] = h + v;
            }
        }
        (horizontal, vertical, combined)
    }

    /// Mirrors `try_place`'s own adjacency check in lib.rs: overlap or any
    /// orthogonal/diagonal neighbour (dr<=1 && dc<=1) between any cell of
    /// `a` and any cell of `b`. Ship-size-agnostic — a Cruiser window and a
    /// Frigate window are just as mutually exclusive as 2 windows of the
    /// same size.
    fn windows_overlap_or_adjacent(a: &[(usize, usize)], b: &[(usize, usize)]) -> bool {
        a.iter().any(|&(ar, ac)| {
            b.iter().any(|&(br, bc)| {
                let dr = (ar as isize - br as isize).unsigned_abs();
                let dc = (ac as isize - bc as isize).unsigned_abs();
                dr <= 1 && dc <= 1
            })
        })
    }

    /// Is the hypothesis "the cells in `window_union` are exactly every
    /// real cell of size `ship_value`, and nothing else is" consistent with
    /// every salvo fired so far? See `salvo_history`'s doc comment for why
    /// this needs the FULL raw history, not just `cross3_entries`/
    /// `cross2_entries`. Necessary because a "3" (or "2") can only ever
    /// come from a Cruiser (or Frigate) cell; sufficient because it pins
    /// down, for every fired cell, whether it does or doesn't belong to
    /// this hypothesis, without needing anything else about what's really
    /// at the salvo's other 2 cells. This is the exact same, independently
    /// self-validated algorithm as `measure_identifiability.rs` — see there
    /// for the fuller derivation and the empirical confirmation that it
    /// never disagrees with ground truth.
    fn consistent_with_salvo_history(window_union: &std::collections::HashSet<(usize, usize)>, history: &[([(usize, usize); 3], [usize; 3])], ship_value: usize) -> bool {
        history.iter().all(|(coords, values)| {
            let hits_in_hypothesis = coords.iter().filter(|c| window_union.contains(c)).count();
            let matches_in_bag = values.iter().filter(|&&v| v == ship_value).count();
            hits_in_hypothesis == matches_in_bag
        })
    }

    /// Every straight-3 window in the inner 8x8 grid (48 horizontal + 48
    /// vertical = 96 total).
    fn all_cruiser_windows() -> Vec<[(usize, usize); 3]> {
        let mut windows = Vec::new();
        for r in INNER_LO..=INNER_HI {
            for c in INNER_LO..=(INNER_HI - 2) {
                windows.push([(r, c), (r, c + 1), (r, c + 2)]);
            }
        }
        for c in INNER_LO..=INNER_HI {
            for r in INNER_LO..=(INNER_HI - 2) {
                windows.push([(r, c), (r + 1, c), (r + 2, c)]);
            }
        }
        windows
    }

    /// Every straight-2 window in the inner 8x8 grid (56 horizontal + 56
    /// vertical = 112 total).
    fn all_frigate_windows() -> Vec<[(usize, usize); 2]> {
        let mut windows = Vec::new();
        for r in INNER_LO..=INNER_HI {
            for c in INNER_LO..=(INNER_HI - 1) {
                windows.push([(r, c), (r, c + 1)]);
            }
        }
        for c in INNER_LO..=INNER_HI {
            for r in INNER_LO..=(INNER_HI - 1) {
                windows.push([(r, c), (r + 1, c)]);
            }
        }
        windows
    }

    /// A cell is provably NOT a cell of size `ship_value` if it was ever
    /// fired as part of a salvo whose result bag contains no `ship_value`
    /// at all. Cheap, always-sound necessary condition, used purely to
    /// shrink the candidate pool before the combinatorial search below (an
    /// imperfect/too-permissive filter would just mean more work, never a
    /// wrong answer — the full consistency check is what actually decides
    /// correctness).
    fn cells_possibly_size(history: &[([(usize, usize); 3], [usize; 3])], ship_value: usize) -> [[bool; 10]; 10] {
        let mut possible = [[true; 10]; 10];
        for (coords, values) in history {
            if !values.contains(&ship_value) {
                for &(r, c) in coords {
                    possible[r][c] = false;
                }
            }
        }
        possible
    }

    /// Every cell proven, via the SEPARATE cross-4 bag deduction, to
    /// definitely hold a Battleship — either currently identified
    /// (`battleship_identified`) or permanently recorded once sunk
    /// (`found_battleship_cells`, which stays populated even after the
    /// live identified-state clears). A cell holds exactly one ship, so
    /// these can never also be a Cruiser or Frigate cell — but nothing in
    /// `consistent_with_salvo_history` enforces that on its own (it only
    /// checks aggregate per-salvo counts of the ONE value it's asked
    /// about, so a window can wrongly "explain" a salvo's evidence by
    /// including a confirmed Battleship cell instead of the real Cruiser/
    /// Frigate cell that salvo actually contained). Used to exclude those
    /// cells from window generation below, rather than relying on the
    /// aggregate check to rule them out by coincidence.
    fn cells_confirmed_battleship(&self) -> [[bool; 10]; 10] {
        let mut confirmed = [[false; 10]; 10];
        for (r, c) in self.battleship_identified_cells() {
            confirmed[r][c] = true;
        }
        for (r, c) in self.found_battleship_cells() {
            confirmed[r][c] = true;
        }
        confirmed
    }

    /// The Cruisers' own confirmed cells, plus every cell adjacent to them
    /// (including diagonally — no ship may ever be placed adjacent to
    /// another), once EITHER `consistent_cruiser_candidates` has narrowed
    /// to a single remaining hypothesis on its own, OR `cruiser_layout_locked`
    /// has been set by `update_fsm_and_resolve` (the cross-reasoned
    /// identification, which can resolve a layout the raw candidate list
    /// alone hasn't). Empirically validated at scale (a 1000-game
    /// self-play run, ~2M per-cell checks) to never be wrong when the raw
    /// list reaches that point on its own — see
    /// `self_play_discovers_every_ship_of_size_at_least_2_by_game_end`'s
    /// heatmap-soundness assertion — so this is used as ground truth to
    /// prune the Frigate search, exactly like `cells_confirmed_battleship`
    /// one size down.
    fn cells_confirmed_cruiser_or_adjacent(&self) -> [[bool; 10]; 10] {
        let mut out = [[false; 10]; 10];
        let candidates = self.consistent_cruiser_candidates();
        let real: Option<Vec<(usize, usize)>> = if let Some(locked) = &self.cruiser_layout_locked {
            Some(locked.clone())
        } else if candidates.len() == 1 {
            Some(candidates[0].iter().copied().collect())
        } else {
            None
        };
        if let Some(cells) = real {
            for &(r, c) in &cells {
                for dr in -1isize..=1 {
                    for dc in -1isize..=1 {
                        let nr = r as isize + dr;
                        let nc = c as isize + dc;
                        if (0..10).contains(&nr) && (0..10).contains(&nc) {
                            out[nr as usize][nc as usize] = true;
                        }
                    }
                }
            }
        }
        out
    }

    /// Every coordinate individually confirmed a real Cruiser hit by
    /// `derive_confirmed_cruiser_hits_by_elimination`'s bag arithmetic,
    /// across every Cross-3 entry — independent of whether the FULL
    /// 2-Cruiser layout is known. Any candidate window-pair whose union
    /// doesn't contain one of these is provably wrong (that cell is
    /// definitely part of SOME real Cruiser, not just possibly one), not
    /// merely "less likely" — see its use in
    /// `consistent_cruiser_candidates_uncached`.
    fn cells_confirmed_individually_cruiser(&self) -> std::collections::HashSet<(usize, usize)> {
        self.cross3_entries
            .iter()
            .flat_map(|e| e.coords.iter().zip(e.coord_confirmed_cruiser_hit.iter()).filter(|(_, &c)| c).map(|(&coord, _)| coord))
            .collect()
    }

    /// Every distinct pair of non-overlapping Cruiser windows currently
    /// consistent with the full salvo history — one entry per hypothesis
    /// "these 2 windows, and nothing else, are the real Cruisers". Shared
    /// by `cruiser_heatmap` (which marginalizes over this list) and
    /// `cruiser_disambiguation_shots` (which reasons about the individual
    /// hypotheses directly, to find a shot that best tells them apart).
    ///
    /// Memoized in `cruiser_candidates_cache`, keyed by `salvo_history.len()`.
    /// This list also depends on `cells_confirmed_individually_cruiser`
    /// (see `consistent_cruiser_candidates_uncached`), but that never
    /// invalidates the cache on its own: every path that can newly confirm
    /// a cell either extends `salvo_history` first (`apply_salvo`, which
    /// changes the cache key itself) or explicitly clears this cache
    /// already for its own reasons (`update_fsm_and_resolve`). Expensive
    /// enough (`consistent_frigate_candidates` one size down even more so)
    /// that recomputing it on every one of the several independent JSON
    /// accessors that call it per game state would otherwise be wasteful.
    fn consistent_cruiser_candidates(&self) -> Vec<std::collections::HashSet<(usize, usize)>> {
        let key = self.salvo_history.len();
        if let Some((cached_key, cached)) = self.cruiser_candidates_cache.borrow().as_ref() {
            if *cached_key == key {
                return cached.clone();
            }
        }
        let computed = self.consistent_cruiser_candidates_uncached();
        *self.cruiser_candidates_cache.borrow_mut() = Some((key, computed.clone()));
        computed
    }

    fn consistent_cruiser_candidates_uncached(&self) -> Vec<std::collections::HashSet<(usize, usize)>> {
        let confirmed_battleship = self.cells_confirmed_battleship();
        let confirmed_cruiser_cells = self.cells_confirmed_individually_cruiser();
        let windows: Vec<[(usize, usize); 3]> = Self::all_cruiser_windows()
            .into_iter()
            .filter(|w| w.iter().all(|&(r, c)| !confirmed_battleship[r][c]))
            .collect();
        let mut out = Vec::new();
        for i in 0..windows.len() {
            for j in (i + 1)..windows.len() {
                if Self::windows_overlap_or_adjacent(&windows[i], &windows[j]) {
                    continue;
                }
                let union: std::collections::HashSet<(usize, usize)> = windows[i].iter().chain(windows[j].iter()).copied().collect();
                // A hypothesis whose union doesn't cover every
                // individually-confirmed Cruiser cell is provably wrong,
                // not just less likely — the same "must include every
                // confirmed cell" pruning `prune_candidates_not_through_
                // confirmed` already does for Battleship's own window
                // search, one size up.
                if confirmed_cruiser_cells.iter().all(|c| union.contains(c)) && Self::consistent_with_salvo_history(&union, &self.salvo_history, 3) {
                    out.push(union);
                }
            }
        }
        out
    }

    /// Every coordinate individually confirmed a real Frigate hit by
    /// `derive_confirmed_frigate_hits_by_elimination`'s bag arithmetic —
    /// mirrors `cells_confirmed_individually_cruiser` one size down.
    fn cells_confirmed_individually_frigate(&self) -> std::collections::HashSet<(usize, usize)> {
        self.cross2_entries
            .iter()
            .flat_map(|e| e.coords.iter().zip(e.coord_confirmed_frigate_hit.iter()).filter(|(_, &c)| c).map(|(&coord, _)| coord))
            .collect()
    }

    /// Every distinct TRIPLE of non-overlapping Frigate windows (3
    /// Frigates, not 2) currently consistent with the full salvo history —
    /// see `consistent_cruiser_candidates`. Unfiltered triple enumeration
    /// over all 112 windows would be ~227,920 combinations —
    /// `cells_possibly_size` narrows the candidate pool first (usually
    /// drastically) before the O(n^3) search. Before any salvo, though,
    /// there's nothing yet to narrow with — memoized in
    /// `frigate_candidates_cache` for the same reason as
    /// `consistent_cruiser_candidates`, including the same "depends on
    /// `cells_confirmed_individually_frigate` too, but that never
    /// invalidates the cache on its own" reasoning.
    fn consistent_frigate_candidates(&self) -> Vec<std::collections::HashSet<(usize, usize)>> {
        let key = self.salvo_history.len();
        if let Some((cached_key, cached)) = self.frigate_candidates_cache.borrow().as_ref() {
            if *cached_key == key {
                return cached.clone();
            }
        }
        let computed = self.consistent_frigate_candidates_uncached();
        *self.frigate_candidates_cache.borrow_mut() = Some((key, computed.clone()));
        computed
    }

    fn consistent_frigate_candidates_uncached(&self) -> Vec<std::collections::HashSet<(usize, usize)>> {
        let windows = Self::all_frigate_windows();
        let possible = Self::cells_possibly_size(&self.salvo_history, 2);
        let confirmed_battleship = self.cells_confirmed_battleship();
        let cruiser_exclusion = self.cells_confirmed_cruiser_or_adjacent();
        let confirmed_frigate_cells = self.cells_confirmed_individually_frigate();
        let candidates: Vec<[(usize, usize); 2]> = windows
            .into_iter()
            .filter(|w| w.iter().all(|&(r, c)| possible[r][c] && !confirmed_battleship[r][c] && !cruiser_exclusion[r][c]))
            .collect();

        let mut out = Vec::new();
        let n = candidates.len();
        for i in 0..n {
            for j in (i + 1)..n {
                if Self::windows_overlap_or_adjacent(&candidates[i], &candidates[j]) {
                    continue;
                }
                for k in (j + 1)..n {
                    if Self::windows_overlap_or_adjacent(&candidates[i], &candidates[k]) || Self::windows_overlap_or_adjacent(&candidates[j], &candidates[k]) {
                        continue;
                    }
                    let union: std::collections::HashSet<(usize, usize)> = candidates[i].iter().chain(candidates[j].iter()).chain(candidates[k].iter()).copied().collect();
                    // Same "must include every individually-confirmed
                    // cell" pruning as the Cruiser side — a hypothesis
                    // whose union misses one is provably wrong.
                    if !confirmed_frigate_cells.iter().all(|c| union.contains(c)) {
                        continue;
                    }
                    if Self::consistent_with_salvo_history(&union, &self.salvo_history, 2) {
                        out.push(union);
                    }
                }
            }
        }
        out
    }

    /// Whether two ship-placement hypotheses (each a full cell-set for one
    /// ship type) can coexist on the same board under the no-adjacent-ships
    /// rule — same Chebyshev-distance-1 check as `windows_overlap_or_adjacent`,
    /// generalized to arbitrary-sized sets so a Cruiser hypothesis can be
    /// checked against a Frigate hypothesis.
    fn hypotheses_compatible(a: &std::collections::HashSet<(usize, usize)>, b: &std::collections::HashSet<(usize, usize)>) -> bool {
        a.iter().all(|&(ar, ac)| {
            b.iter().all(|&(br, bc)| {
                let dr = (ar as isize - br as isize).unsigned_abs();
                let dc = (ac as isize - bc as isize).unsigned_abs();
                dr > 1 || dc > 1
            })
        })
    }

    /// Every (Cruiser hypothesis, Frigate hypothesis) pair from
    /// `consistent_cruiser_candidates`/`consistent_frigate_candidates` that
    /// is mutually consistent under `hypotheses_compatible`. Each of those
    /// two lists is already internally consistent (no window overlaps or
    /// touches another window of the SAME type), but neither ever checks
    /// itself against the OTHER type — so a Cruiser hypothesis and a
    /// Frigate hypothesis that each independently survive can still collide
    /// with each other. This is the cross-ship-type analysis a human can do
    /// by eye once the per-type heatmaps stop finding anything further on
    /// their own: pairing the two candidate lists against each other can
    /// eliminate hypotheses neither heatmap rules out alone (e.g. a Frigate
    /// arm that happens to touch every remaining Cruiser line, even though
    /// no single Cruiser line has yet won out on its own).
    ///
    /// Early game, before any salvo has narrowed things down, each of
    /// `consistent_cruiser_candidates`/`consistent_frigate_candidates` can
    /// itself run into the thousands of hypotheses (see the comment on
    /// `consistent_frigate_candidates`), making their cross product far too
    /// large to check pairwise. Cross-reasoning only ever matters once both
    /// lists are already small from salvo evidence narrowing them down, so
    /// bailing out early above this budget costs nothing real — same as the
    /// callers' existing "pairs.is_empty() -> fall back to the un-cross-
    /// checked list" handling.
    fn jointly_consistent_hypothesis_pairs(&self) -> Vec<(std::collections::HashSet<(usize, usize)>, std::collections::HashSet<(usize, usize)>)> {
        const CROSS_REASONING_PAIR_BUDGET: usize = 100_000;
        let cruiser = self.consistent_cruiser_candidates();
        let frigate = self.consistent_frigate_candidates();
        if cruiser.len().saturating_mul(frigate.len()) > CROSS_REASONING_PAIR_BUDGET {
            return Vec::new();
        }
        let mut out = Vec::new();
        for c in &cruiser {
            for f in &frigate {
                if Self::hypotheses_compatible(c, f) {
                    out.push((c.clone(), f.clone()));
                }
            }
        }
        out
    }

    /// `consistent_cruiser_candidates`, narrowed further by cross-checking
    /// every hypothesis against every Frigate hypothesis (see
    /// `jointly_consistent_hypothesis_pairs`) — a Cruiser hypothesis with no
    /// remaining compatible Frigate partner is dropped. Each surviving
    /// hypothesis appears once per compatible Frigate partner, so
    /// marginalizing this (via `heatmap_from_candidates`) weighs it by how
    /// many Frigate arrangements it's still consistent with, rather than
    /// just by raw presence — the correct marginal under a joint-uniform
    /// prior over (Cruiser hypothesis, Frigate hypothesis) pairs.
    ///
    /// Falls back to the un-cross-checked list once it's down to 0 or 1
    /// entries already (nothing left to narrow), or if cross-checking would
    /// eliminate every hypothesis outright — the true layout is always
    /// jointly consistent with itself, so that would mean a bug elsewhere,
    /// not genuine new information, and showing the un-refined heatmap is
    /// the safer fallback over showing a false blank slate.
    fn cross_reasoned_cruiser_candidates(&self) -> Vec<std::collections::HashSet<(usize, usize)>> {
        let cruiser = self.consistent_cruiser_candidates();
        if cruiser.len() <= 1 {
            return cruiser;
        }
        let pairs = self.jointly_consistent_hypothesis_pairs();
        if pairs.is_empty() {
            return cruiser;
        }
        pairs.into_iter().map(|(c, _)| c).collect()
    }

    /// Same idea as `cross_reasoned_cruiser_candidates`, one size down.
    fn cross_reasoned_frigate_candidates(&self) -> Vec<std::collections::HashSet<(usize, usize)>> {
        let frigate = self.consistent_frigate_candidates();
        if frigate.len() <= 1 {
            return frigate;
        }
        let pairs = self.jointly_consistent_hypothesis_pairs();
        if pairs.is_empty() {
            return frigate;
        }
        pairs.into_iter().map(|(_, f)| f).collect()
    }

    /// Deduplicates a list of hypotheses by full cell-set equality — after
    /// cross-reasoning, a hypothesis appears once per compatible partner in
    /// the other ship type's candidate list (see
    /// `cross_reasoned_cruiser_candidates`), so "exactly one hypothesis
    /// remains" means exactly one DISTINCT cell-set, not exactly one list
    /// entry.
    ///
    /// `hyps` can be the un-cross-reasoned fallback list too (see the
    /// callers), which on an early/fresh board runs into the tens or
    /// hundreds of thousands of entries — an O(n^2) `Vec::contains` scan
    /// there is exactly the wrong complexity at exactly the size where it
    /// matters most. Dedup via a sorted canonical key through a `HashSet`
    /// instead, for O(n log k) (k = hypothesis cell count, always small).
    fn distinct_hypotheses(hyps: &[std::collections::HashSet<(usize, usize)>]) -> Vec<&std::collections::HashSet<(usize, usize)>> {
        let mut seen: std::collections::HashSet<Vec<(usize, usize)>> = std::collections::HashSet::new();
        let mut out: Vec<&std::collections::HashSet<(usize, usize)>> = Vec::new();
        for h in hyps {
            let mut key: Vec<(usize, usize)> = h.iter().copied().collect();
            key.sort_unstable();
            if seen.insert(key) {
                out.push(h);
            }
        }
        out
    }

    /// Marginalize a list of consistent candidate hypotheses (each the
    /// full cell-set of one hypothesis) into a per-cell probability grid —
    /// shared body for `cruiser_heatmap`/`frigate_heatmap`. Same 8x8 grid
    /// convention as `alive_grids` (indexed 0..8 for board rows/cols 1..8).
    /// All zeros before any salvo has been fired (nothing to condition on
    /// yet — every hypothesis is equally "consistent" so no single cell
    /// stands out; returning a flat 0 rather than a flat nonzero avoids
    /// implying false precision).
    fn heatmap_from_candidates(&self, candidates: &[std::collections::HashSet<(usize, usize)>]) -> Vec<Vec<f64>> {
        let mut counts = [[0u32; 10]; 10];
        for cand in candidates {
            for &(r, c) in cand {
                counts[r][c] += 1;
            }
        }
        let total = candidates.len();
        let mut grid = vec![vec![0.0f64; 8]; 8];
        if total > 0 && !self.salvo_history.is_empty() {
            for row in INNER_LO..=INNER_HI {
                for col in INNER_LO..=INNER_HI {
                    grid[row - INNER_LO][col - INNER_LO] = counts[row][col] as f64 / total as f64;
                }
            }
        }
        grid
    }

    /// Per-cell probability (0.0-1.0) that a Cruiser occupies it, under a
    /// uniform prior over every currently-consistent pair of Cruiser
    /// windows. See `heatmap_from_candidates`.
    pub fn cruiser_heatmap(&self) -> Vec<Vec<f64>> {
        self.heatmap_from_candidates(&self.consistent_cruiser_candidates())
    }

    /// Same idea as `cruiser_heatmap`, one size down: per-cell probability
    /// that a Frigate occupies it, under a uniform prior over every
    /// currently-consistent TRIPLE of Frigate windows. See
    /// `heatmap_from_candidates`.
    pub fn frigate_heatmap(&self) -> Vec<Vec<f64>> {
        self.heatmap_from_candidates(&self.consistent_frigate_candidates())
    }

    /// Same marginalization as `heatmap_from_candidates`, but returning the
    /// raw (count, total) pair behind each cell's probability instead of
    /// the divided float — for displaying the underlying fraction directly
    /// (e.g. "1/3") rather than a percentage. Same all-zero-pair
    /// convention before any salvo has been fired.
    fn heatmap_fraction_from_candidates(&self, candidates: &[std::collections::HashSet<(usize, usize)>]) -> Vec<Vec<(u32, u32)>> {
        let mut counts = [[0u32; 10]; 10];
        for cand in candidates {
            for &(r, c) in cand {
                counts[r][c] += 1;
            }
        }
        let total = candidates.len() as u32;
        let mut grid = vec![vec![(0u32, 0u32); 8]; 8];
        if total > 0 && !self.salvo_history.is_empty() {
            for row in INNER_LO..=INNER_HI {
                for col in INNER_LO..=INNER_HI {
                    grid[row - INNER_LO][col - INNER_LO] = (counts[row][col], total);
                }
            }
        }
        grid
    }

    /// See `heatmap_fraction_from_candidates`.
    pub fn cruiser_heatmap_fraction(&self) -> Vec<Vec<(u32, u32)>> {
        self.heatmap_fraction_from_candidates(&self.consistent_cruiser_candidates())
    }

    /// See `heatmap_fraction_from_candidates`.
    pub fn frigate_heatmap_fraction(&self) -> Vec<Vec<(u32, u32)>> {
        self.heatmap_fraction_from_candidates(&self.consistent_frigate_candidates())
    }

    /// Same idea as `cruiser_heatmap`, but marginalized over
    /// `cross_reasoned_cruiser_candidates` instead of the raw
    /// `consistent_cruiser_candidates` — i.e. after also cross-checking
    /// every remaining hypothesis against every remaining Frigate
    /// hypothesis for mutual adjacency. Strictly at least as informative as
    /// `cruiser_heatmap` (can only narrow probabilities further, never
    /// widen them); this is the one the UI shows.
    pub fn cruiser_heatmap_refined(&self) -> Vec<Vec<f64>> {
        self.heatmap_from_candidates(&self.cross_reasoned_cruiser_candidates())
    }

    /// Same idea as `cruiser_heatmap_refined`, one size down.
    pub fn frigate_heatmap_refined(&self) -> Vec<Vec<f64>> {
        self.heatmap_from_candidates(&self.cross_reasoned_frigate_candidates())
    }

    /// See `cruiser_heatmap_refined` — same idea, returning the raw
    /// (count, total) pair instead of the divided float. See
    /// `heatmap_fraction_from_candidates`.
    pub fn cruiser_heatmap_fraction_refined(&self) -> Vec<Vec<(u32, u32)>> {
        self.heatmap_fraction_from_candidates(&self.cross_reasoned_cruiser_candidates())
    }

    /// See `cruiser_heatmap_fraction_refined`, one size down.
    pub fn frigate_heatmap_fraction_refined(&self) -> Vec<Vec<(u32, u32)>> {
        self.heatmap_fraction_from_candidates(&self.cross_reasoned_frigate_candidates())
    }

    /// Given the currently consistent candidate hypotheses for a ship
    /// class, pick 3 unfired cells whose combined salvo result narrows the
    /// candidate set as much as possible in the worst case — a minimax
    /// "20 questions" search. `None` if there's nothing left to
    /// disambiguate (0 or 1 consistent hypotheses).
    ///
    /// The search is scoped to "informative" cells — ones where the
    /// hypotheses disagree (some include it, some don't) — since firing
    /// anywhere else can't distinguish between any surviving hypothesis:
    /// every consistent candidate already agrees on the rest of the board,
    /// so the result there is 100% predictable and teaches nothing. Cells
    /// already fired are excluded even if they still show a fractional
    /// split (an earlier salvo whose bag didn't reveal which of its 3
    /// cells was the hit) — we already know their true value from the
    /// log, so re-firing teaches nothing new.
    ///
    /// The informative pool is capped for combinatorial feasibility,
    /// keeping the individually most-discriminating cells (closest to an
    /// even split) — a pragmatic approximation of the true joint-optimal
    /// search, not guaranteed globally optimal, but always sound: any
    /// salvo returned is guaranteed to strictly narrow the candidate set
    /// on the worst-case outcome, never to leave it unchanged.
    ///
    /// Also gives up (returns `None`) when the candidate set is still huge
    /// — right after a class is sunk, most of the board can still be
    /// completely untested, so literally any untouched region could host
    /// an entirely phantom alternate window, and "consistent" candidate
    /// pairs/triples can number in the thousands. Chasing that down to a
    /// unique answer via dedicated disambiguation shots alone could take
    /// dozens of turns, monopolizing every salvo and completely starving
    /// Frigate/Submarine hunting in the meantime. Ordinary hunting shots
    /// narrow this same candidate set for free as a side effect (every
    /// shot updates `salvo_history`, regardless of what it was aimed at) —
    /// so it's better to defer and let that happen first, then step in
    /// with dedicated effort once the residual ambiguity has already
    /// shrunk to something a handful of shots can realistically resolve.
    /// `allow_refire`: when true, a cell already fired is still treated as
    /// eligible for `informative` (see below) provided it hasn't already
    /// spent its one-time bonus refire (`disambiguation_extra_refire_used`)
    /// — see `disambiguation_shots_with_refire`. `false` reproduces the
    /// original, always-fired-cells-excluded behaviour.
    /// `ignore_refire_cap`: when true (and `allow_refire` is also true),
    /// a cell that has ALREADY spent its one-time bonus refire is still
    /// eligible — see `disambiguation_shots_last_resort`. This exists for
    /// the genuine dead end one tier further than the ordinary bonus
    /// refire: a "cluster of 3" Frigate/Cruiser ambiguity (a certain pivot
    /// cell plus 2 mutually-exclusive end cells) whose 2 end cells both
    /// already spent their bonus refire earlier resolving some OTHER,
    /// now-settled ambiguity, leaving this specific tie permanently
    /// unbreakable under the normal one-bonus-per-cell rule even though a
    /// single clean refire (paired with 2 known-neutral fillers) would
    /// fully resolve it. Should only ever be reached once
    /// `disambiguation_shots(..., true, false)` has already come back
    /// empty — see `disambiguation_shots_last_resort`'s doc comment.
    fn disambiguation_shots(&self, candidates: &[std::collections::HashSet<(usize, usize)>], allow_refire: bool, ignore_refire_cap: bool) -> Option<[(usize, usize); 3]> {
        // Was 80 — measured empirically to be far too conservative: the cost
        // of this search is dominated by the O(pool^3) triple loop below
        // (pool capped at MAX_POOL=14 regardless of candidates.len()), with
        // candidates.len() only a linear factor inside that — 112,872
        // candidates (a real saved board that plateaued forever under the
        // old cap) resolves in ~0.1ms once actually allowed through. The old
        // 80-cap meant this search — including the with-refire and
        // last-resort tiers, which all share it — never even ran once a
        // board had more than 80 surviving hypotheses, silently falling
        // back to non-disambiguating filler shots for the rest of the game
        // no matter how many salvos were spent. 200,000 keeps real headroom
        // while comfortably covering boards like that one.
        const MAX_CANDIDATES_TO_ATTEMPT: usize = 200_000;
        if candidates.len() <= 1 || candidates.len() > MAX_CANDIDATES_TO_ATTEMPT {
            return None;
        }

        // BTreeMap (not HashMap): iteration order feeds directly into
        // `informative`'s ordering below, which in turn decides tie-breaks
        // whenever multiple cells score equally — a HashMap's iteration
        // order isn't guaranteed stable across separate instances, which
        // made 2 calls against the identical AiPlayer state able to return
        // different salvos for a tied scenario. The advisory must be
        // deterministic for a given board state.
        let mut counts: std::collections::BTreeMap<(usize, usize), usize> = std::collections::BTreeMap::new();
        for cand in candidates {
            for &cell in cand {
                *counts.entry(cell).or_insert(0) += 1;
            }
        }
        let total = candidates.len();
        let mut informative: Vec<(usize, usize)> = counts
            .iter()
            .filter(|&(&(r, c), &n)| {
                n > 0
                    && n < total
                    && (!self.fired[r][c]
                        || (allow_refire && (ignore_refire_cap || !self.disambiguation_extra_refire_used[r][c])))
            })
            .map(|(&cell, _)| cell)
            .collect();

        if informative.is_empty() {
            // Every cell the remaining candidates disagree on is already
            // fired — e.g. 2 hypotheses differing only in which cell of an
            // already-completed salvo held the real hit (its bag's
            // aggregate count matches either way; see the module-level
            // reasoning on `consistent_with_salvo_history`). Re-firing an
            // already-known cell teaches nothing, so this is a genuine,
            // permanent limit of the unordered-bag observation model, not
            // a puzzle any future shot could ever crack — report it as
            // resolved-as-far-as-possible instead of looping forever on a
            // directionless filler salvo.
            return None;
        }

        const MAX_POOL: usize = 14;
        if informative.len() > MAX_POOL {
            informative.sort_by_key(|cell| {
                let n = counts[cell];
                (n as isize - (total as isize - n as isize)).unsigned_abs()
            });
            informative.truncate(MAX_POOL);
        }

        let mut pool = informative;
        // Always pad in at least 2 filler cells, even when the informative
        // set alone is already >= 3 — never just enough to reach the bare
        // minimum of 3. Padding to exactly 3 total is a trap whenever
        // `informative` itself has exactly 2 (or 3) cells that are mutually
        // exclusive alternatives across the surviving hypotheses (e.g. 2
        // hypotheses differing in exactly one cell each, like "B4 or C5
        // completes this Frigate"): the O(n^3) search below can only ever
        // try whichever single triple the pool happens to contain, so
        // padding to exactly 3 forces BOTH alternatives into that one
        // triple with no other combination ever compared against it. That
        // specific combo is provably the worst possible choice here —
        // firing both alternatives together always yields the identical
        // bag regardless of which hypothesis is real (exactly one of them
        // is always the hit), teaching nothing — but the search never gets
        // a chance to discover that firing just ONE of them plus 2 neutral
        // fillers would fully resolve it instead, because that
        // strictly-better triple was never actually a candidate. Fillers
        // carry no discriminating power of their own; they only need to
        // exist so genuine alternative triples are available to compare.
        let mut fillers_added = 0;
        'pad: for r in 0..10 {
            for c in 0..10 {
                if fillers_added >= 2 {
                    break 'pad;
                }
                if !self.fired[r][c] && !pool.contains(&(r, c)) {
                    pool.push((r, c));
                    fillers_added += 1;
                }
            }
        }
        if pool.len() < 3 {
            return None;
        }

        let mut best: Option<([(usize, usize); 3], usize)> = None;
        for i in 0..pool.len() {
            for j in (i + 1)..pool.len() {
                for k in (j + 1)..pool.len() {
                    let salvo = [pool[i], pool[j], pool[k]];
                    // For every hypothesis, how many of these 3 cells it
                    // predicts as hits — the only thing the resulting
                    // bag's count of matching values can ever reveal about
                    // which hypothesis is real (see
                    // `consistent_with_salvo_history`).
                    let mut buckets = [0usize; 4];
                    for cand in candidates {
                        let hits = salvo.iter().filter(|cell| cand.contains(cell)).count();
                        buckets[hits] += 1;
                    }
                    let worst = *buckets.iter().max().unwrap();
                    if best.as_ref().is_none_or(|(_, best_worst)| worst < *best_worst) {
                        best = Some((salvo, worst));
                    }
                }
            }
        }
        best.map(|(salvo, _)| salvo)
    }

    /// "Anchor-and-isolate" cleanup shot, tried before the general minimax
    /// `disambiguation_shots` search: if the Cruiser/Frigate heatmaps show
    /// one cell that's already provably possible ONLY as a Cruiser (its
    /// Frigate probability is exactly 0) and a different cell that's
    /// provably possible ONLY as a Frigate (Cruiser probability exactly
    /// 0), firing both together alongside any cell whose true value is
    /// already known with total certainty (currently: a confirmed
    /// Battleship cell) lets the resulting bag be decoded by elimination
    /// instead of probability. The known cell's contribution is always
    /// identifiable (a 4 is the board's only possible "4", and can never
    /// come from anywhere else in this salvo, since confirmed Battleship
    /// cells are excluded from every Cruiser/Frigate window by
    /// construction — see `cells_confirmed_battleship`). Since neither of
    /// the other 2 cells could ever produce the OTHER ship's value,
    /// whichever count actually shows up in the bag must have come from
    /// the one cell that could possibly produce it — fully resolving BOTH
    /// cells from a single salvo, rather than merely narrowing the worst
    /// case the way `disambiguation_shots` does. `None` if no confirmed
    /// Battleship cell exists yet, or no such cross-exclusive pair exists.
    fn anchored_isolation_shot(&self) -> Option<[(usize, usize); 3]> {
        let confirmed_battleship = self.cells_confirmed_battleship();
        let mut anchor = None;
        'find_anchor: for r in INNER_LO..=INNER_HI {
            for c in INNER_LO..=INNER_HI {
                if confirmed_battleship[r][c] {
                    anchor = Some((r, c));
                    break 'find_anchor;
                }
            }
        }
        let anchor = anchor?;

        let cruiser_grid = self.cruiser_heatmap();
        let frigate_grid = self.frigate_heatmap();

        let mut cruiser_only = None;
        let mut frigate_only = None;
        for r in INNER_LO..=INNER_HI {
            for c in INNER_LO..=INNER_HI {
                if (r, c) == anchor {
                    continue;
                }
                let cp = cruiser_grid[r - INNER_LO][c - INNER_LO];
                let fp = frigate_grid[r - INNER_LO][c - INNER_LO];
                // Strictly between 0 and 1, not just nonzero: a cell
                // already at 1.0 is already fully resolved, so "isolating"
                // it again would waste the salvo re-confirming something
                // already certain — and, worse, since firing the exact
                // same cells always reproduces the exact same bag, picking
                // an already-resolved cell here forever would never
                // change anything, looping indefinitely instead of making
                // progress.
                if cruiser_only.is_none() && cp > 0.0 && cp < 1.0 && fp == 0.0 {
                    cruiser_only = Some((r, c));
                }
                if frigate_only.is_none() && fp > 0.0 && fp < 1.0 && cp == 0.0 {
                    frigate_only = Some((r, c));
                }
            }
        }

        match (cruiser_only, frigate_only) {
            (Some(a), Some(b)) => Some([anchor, a, b]),
            _ => None,
        }
    }

    /// Best next salvo to disambiguate the Cruisers' exact layout, once
    /// both are sunk but more than one placement remains consistent with
    /// the evidence so far. See `disambiguation_shots`. `None` before any
    /// salvo has been fired at all — same "nothing to condition on yet"
    /// reasoning as `cruiser_heatmap`'s all-zero grid: with zero evidence,
    /// literally every non-overlapping window pair is equally
    /// "consistent", so there's nothing genuine to disambiguate yet, only
    /// an arbitrary first guess (and `choose_shots` never reaches this
    /// before both Cruisers are sunk in practice anyway, which itself
    /// requires a non-empty history).
    pub fn cruiser_disambiguation_shots(&self) -> Option<[(usize, usize); 3]> {
        if self.salvo_history.is_empty() {
            return None;
        }
        self.disambiguation_shots(&self.consistent_cruiser_candidates(), false, false)
    }

    /// Same idea as `cruiser_disambiguation_shots`, one size down.
    pub fn frigate_disambiguation_shots(&self) -> Option<[(usize, usize); 3]> {
        if self.salvo_history.is_empty() {
            return None;
        }
        self.disambiguation_shots(&self.consistent_frigate_candidates(), false, false)
    }

    /// Same idea as `cruiser_disambiguation_shots`, but allows the salvo to
    /// include an already-fired cell that hasn't yet spent its one-time
    /// bonus refire (see `disambiguation_extra_refire_used`) — for the
    /// "heatmap fully evolved" dead end where every cell the remaining
    /// hypotheses disagree on has already been fired (see `disambiguation_
    /// shots`' own `informative.is_empty()` case): pairing one of those
    /// cells with 2 fresh cells whose own value the AI already knows for
    /// certain (e.g. cells proven to hold nothing of any relevant size)
    /// isolates that cell's true value in the new salvo's bag, resolving
    /// an ambiguity no ordinary (never-refire) shot ever could. `Game::fire`
    /// only allows the refire through when it matches the specific cell(s)
    /// this returns — see `is_disambiguation_extra_refire`.
    pub fn cruiser_disambiguation_shots_with_refire(&self) -> Option<[(usize, usize); 3]> {
        if self.salvo_history.is_empty() {
            return None;
        }
        self.disambiguation_shots(&self.consistent_cruiser_candidates(), true, false)
    }

    /// Same idea as `cruiser_disambiguation_shots_with_refire`, one size down.
    pub fn frigate_disambiguation_shots_with_refire(&self) -> Option<[(usize, usize); 3]> {
        if self.salvo_history.is_empty() {
            return None;
        }
        self.disambiguation_shots(&self.consistent_frigate_candidates(), true, false)
    }

    /// Combined entry point for the UI's "disambiguate (allow 1 refire)"
    /// button — same Cruiser-before-Frigate priority as `choose_shots`'
    /// ordinary (no-refire) disambiguation block, since narrowing the
    /// Cruisers first also shrinks the Frigate search.
    pub fn disambiguation_shots_with_refire(&self) -> Option<[(usize, usize); 3]> {
        self.cruiser_disambiguation_shots_with_refire()
            .or_else(|| self.frigate_disambiguation_shots_with_refire())
    }

    /// Same idea as `cruiser_disambiguation_shots_with_refire`, but ALSO
    /// eligible for a cell that has already spent its one-time bonus
    /// refire — the last-resort tier, one further than that: a "cluster of
    /// 3" ambiguity (certain pivot cell, 2 mutually-exclusive end cells)
    /// can end up with BOTH end cells' bonus refire already spent
    /// resolving some earlier, now-settled ambiguity, leaving this
    /// specific tie permanently unbreakable under the normal cap even
    /// though a single clean refire would fully resolve it. Only ever
    /// worth trying once `cruiser_disambiguation_shots_with_refire` itself
    /// has already come back empty — see `is_last_resort_refire`, which
    /// gates `Game::fire` letting this through, for why this doesn't just
    /// replace the capped version outright: it's deliberately a narrower,
    /// harder-to-reach fallback, not a blanket relaxation of the cap.
    pub fn cruiser_disambiguation_shots_last_resort(&self) -> Option<[(usize, usize); 3]> {
        if self.salvo_history.is_empty() {
            return None;
        }
        self.disambiguation_shots(&self.consistent_cruiser_candidates(), true, true)
    }

    /// Same idea as `cruiser_disambiguation_shots_last_resort`, one size down.
    pub fn frigate_disambiguation_shots_last_resort(&self) -> Option<[(usize, usize); 3]> {
        if self.salvo_history.is_empty() {
            return None;
        }
        self.disambiguation_shots(&self.consistent_frigate_candidates(), true, true)
    }

    /// Combined entry point for the last-resort tier — same Cruiser-before-
    /// Frigate priority as `disambiguation_shots_with_refire`. The UI should
    /// only ever reach for this once `disambiguation_shots_with_refire`
    /// itself has returned `None` (see `is_last_resort_refire`).
    pub fn disambiguation_shots_last_resort(&self) -> Option<[(usize, usize); 3]> {
        self.cruiser_disambiguation_shots_last_resort()
            .or_else(|| self.frigate_disambiguation_shots_last_resort())
    }

    /// True if (row, col) is fired, hasn't already spent its one-time bonus
    /// refire, and is part of the exact salvo `disambiguation_shots_with_
    /// refire` would currently suggest — used by `Game::fire` to let that
    /// specific refire through even with the general refire-allowed toggle
    /// off, mirroring `is_battleship_discriminating_refire`/`is_anchored_
    /// isolation_refire` one size up.
    pub fn is_disambiguation_extra_refire(&self, row: usize, col: usize) -> bool {
        self.fired[row][col]
            && !self.disambiguation_extra_refire_used[row][col]
            && self.disambiguation_shots_with_refire().is_some_and(|shots| shots.contains(&(row, col)))
    }

    /// Marks (row, col) as having spent its one-time bonus refire — called
    /// by `Game::fire` once a salvo `is_disambiguation_extra_refire` let
    /// through has actually been fired, so it can never be granted a
    /// second bonus refire.
    pub fn mark_disambiguation_extra_refire_used(&mut self, row: usize, col: usize) {
        self.disambiguation_extra_refire_used[row][col] = true;
    }

    /// True if (row, col) is fired and is part of the exact salvo
    /// `disambiguation_shots_last_resort` would currently suggest —
    /// deliberately requires `disambiguation_shots_with_refire` (the
    /// capped tier) to ALREADY be `None` before this can ever be true, so
    /// the one-time-bonus cap can never be bypassed just because this
    /// function exists — only reachable once the capped tier has
    /// genuinely nothing left to offer. Unlike `is_disambiguation_extra_
    /// refire`, not capped at one use per cell: by the time this is even
    /// checked, that cap has already done its job once (that's WHY the
    /// capped tier came back empty) — see `disambiguation_shots`'
    /// `ignore_refire_cap` doc comment for the exact scenario this exists
    /// for.
    pub fn is_last_resort_refire(&self, row: usize, col: usize) -> bool {
        self.fired[row][col]
            && self.disambiguation_shots_with_refire().is_none()
            && self.disambiguation_shots_last_resort().is_some_and(|shots| shots.contains(&(row, col)))
    }

    /// Whether `choose_shots` currently has a genuine Cruiser-disambiguating
    /// salvo to offer — i.e. whether the Cruisers' exact layout is still
    /// ambiguous. Exposed so callers (the "fleet cleared" popup/auto-play
    /// stop condition in the UI) can wait for disambiguation to actually
    /// finish, rather than declaring victory the instant both Cruisers are
    /// literally sunk — which, per `consistent_cruiser_candidates`, doesn't
    /// by itself guarantee their exact cells are uniquely determined.
    pub fn cruiser_disambiguation_pending(&self) -> bool {
        self.cruiser_disambiguation_shots().is_some()
    }

    /// Same idea as `cruiser_disambiguation_pending`, one size down.
    pub fn frigate_disambiguation_pending(&self) -> bool {
        self.frigate_disambiguation_shots().is_some()
    }

    /// Re-check, for every cross-3 entry, whether each of its 3 ORIGINAL fired
    /// coordinates (not the derived bag — the actual salvo cells) could still
    /// possibly be the real Cruiser hit that produced that salvo's "3". A cell
    /// on the outer ring never holds a ship of size >=2, so it's ruled out
    /// immediately; an inner cell is ruled out once its combined alive value
    /// for size 3 has dropped to zero (see `alive_value`) — proof that no
    /// placement, horizontal or vertical, still passes through it. Called at
    /// the end of every round so `coord_ruled_out` always reflects everything
    /// deduced so far, not just what was known when the entry was created.
    fn refresh_cross3_entry_flags(&mut self) {
        let is_ruled_out = |ai: &Self, row: usize, col: usize| {
            if (INNER_LO..=INNER_HI).contains(&row) && (INNER_LO..=INNER_HI).contains(&col) {
                ai.alive_value(row, col, 3) == 0
            } else {
                true // outer ring: never a Cruiser cell in the first place
            }
        };
        let flags: Vec<[bool; 3]> = self
            .cross3_entries
            .iter()
            .map(|entry| {
                let mut flags = [false; 3];
                for (i, &(r, c)) in entry.coords.iter().enumerate() {
                    flags[i] = is_ruled_out(self, r, c);
                }
                flags
            })
            .collect();
        // OR-merge, never overwrite: `derive_confirmed_cruiser_hits_by_
        // elimination` below can also set `coord_ruled_out` bits (from bag
        // arithmetic, not the FSM), and those must survive next round's
        // refresh too, same monotonic guarantee as the FSM-derived ones.
        for (entry, entry_flags) in self.cross3_entries.iter_mut().zip(flags) {
            for i in 0..3 {
                entry.coord_ruled_out[i] |= entry_flags[i];
            }
        }
        self.derive_confirmed_cruiser_hits_by_elimination();
    }

    /// Once a Cross-3 entry's bag holds exactly N copies of "3" and only N
    /// of its coordinates remain un-ruled-out, those N are certain Cruiser
    /// hits — mirrors `derive_confirmed_battleship_hits_by_elimination` one
    /// size down, generalized from N=1 (Battleship only ever has 1 "4" per
    /// bag, since there's only 1 Battleship) to any N up to 3, since 2 (or
    /// even 3) of a salvo's cells can genuinely be different Cruiser cells
    /// at once. Deliberately NOT the removed candidate-window search (see
    /// 35e8c16): this never asks which straight-3 placement a cell belongs
    /// to, only this literal salvo's own bag arithmetic, so it can't
    /// reproduce that bug's phantom-Cruiser failure mode — but it MUST
    /// count how many "3"s a bag actually holds, not just whether it holds
    /// at least one, or it reproduces a different version of the exact same
    /// failure class: an earlier version of this function treated any bag
    /// containing a "3" as fully explained by a single confirmed
    /// coordinate, which silently assumed N=1 and wrongly ruled out a
    /// second, genuinely real Cruiser cell whenever 2 landed in the same
    /// salvo (caught by `self_play_discovers_every_ship_of_size_at_least_2_
    /// by_game_end`'s "no real Cruiser cell ever wrongly ruled out"
    /// assertion — exactly the kind of self-play regression test 35e8c16
    /// added for the original bug, and it did its job here too).
    ///
    /// 2 counting rules, applied per entry:
    ///   1. If exactly N coordinates remain un-ruled-out, all N must be real
    ///      (there's nowhere else the N needed hits could be) — confirm them.
    ///   2. If exactly N coordinates are already confirmed, every other
    ///      coordinate must NOT be real (there's no room left) — rule them
    ///      out.
    ///
    /// Confirmed status also propagates across entries by coordinate
    /// identity: a cell confirmed via one salvo is the SAME physical board
    /// cell wherever else it was fired (e.g. a refire, or 2 different
    /// salvos sharing a coordinate), so rule 2 above can fire in another
    /// entry using a coordinate confirmed elsewhere. Iterates to a fixed
    /// point, since one round's confirmation/elimination can enable the
    /// next, in either entry.
    ///
    /// A coordinate ruled out this way is fed into the REAL size-3 FSM via
    /// `eliminate_size_at`, not just recorded on the entry — the reasoning
    /// above proves it Cruiser-free globally, the same strength as an
    /// ordinary miss, not just for this one salvo. Without this, the
    /// deduction would be debug-display-only and never actually change
    /// `choose_shots`/the heatmaps, which read the FSM directly and know
    /// nothing about Cross-3 entries.
    fn derive_confirmed_cruiser_hits_by_elimination(&mut self) {
        loop {
            let mut changed = false;
            // Every coordinate whose `coord_confirmed_cruiser_hit` flips to
            // true THIS round (either rule below) — not just re-derived
            // from an earlier round. See the adjacency-elimination pass at
            // the end of this iteration for why this is tracked separately
            // from `newly_ruled_out`.
            let mut newly_confirmed: std::collections::HashSet<(usize, usize)> = std::collections::HashSet::new();

            // Rule 1: exactly N remaining open coordinates, for a bag
            // holding N "3"s, must all be real.
            for entry in &mut self.cross3_entries {
                let n = entry.values.iter().filter(|&&v| v == 3).count();
                if n == 0 {
                    continue;
                }
                let open: Vec<usize> = (0..3).filter(|&i| !entry.coord_ruled_out[i]).collect();
                if open.len() == n {
                    for &i in &open {
                        if !entry.coord_confirmed_cruiser_hit[i] {
                            entry.coord_confirmed_cruiser_hit[i] = true;
                            changed = true;
                            newly_confirmed.insert(entry.coords[i]);
                        }
                    }
                }
            }

            let confirmed_coords: std::collections::HashSet<(usize, usize)> = self
                .cross3_entries
                .iter()
                .flat_map(|e| e.coords.iter().zip(e.coord_confirmed_cruiser_hit.iter()).filter(|(_, &c)| c).map(|(&coord, _)| coord))
                .collect();

            // Propagate confirmed status across entries by coordinate
            // identity BEFORE rule 2, so a coordinate confirmed via a
            // different entry counts toward THIS entry's own confirmed
            // total too.
            for entry in &mut self.cross3_entries {
                if !entry.values.contains(&3) {
                    continue;
                }
                for i in 0..3 {
                    if !entry.coord_confirmed_cruiser_hit[i] && confirmed_coords.contains(&entry.coords[i]) {
                        entry.coord_confirmed_cruiser_hit[i] = true;
                        changed = true;
                        newly_confirmed.insert(entry.coords[i]);
                    }
                }
            }

            // Collected rather than applied inline: `eliminate_size_at` needs
            // `&mut self` as a whole (it drives `row_state`/`col_state`),
            // which can't happen while `self.cross3_entries` is itself
            // mutably borrowed by the loop below. A HashSet also naturally
            // dedupes a coordinate that gets newly ruled out via more than
            // one entry in the same pass (harmless to feed twice either way
            // — `eliminate_size_at` is idempotent per cell — but pointless).
            let mut newly_ruled_out: std::collections::HashSet<(usize, usize)> = std::collections::HashSet::new();

            // Rule 2: exactly N confirmed, for a bag holding N "3"s, means
            // every other coordinate is definitely not real.
            for entry in &mut self.cross3_entries {
                let n = entry.values.iter().filter(|&&v| v == 3).count();
                if n == 0 {
                    continue;
                }
                let confirmed_count = (0..3).filter(|&i| entry.coord_confirmed_cruiser_hit[i]).count();
                if confirmed_count != n {
                    continue;
                }
                for i in 0..3 {
                    if !entry.coord_confirmed_cruiser_hit[i] && !entry.coord_ruled_out[i] {
                        entry.coord_ruled_out[i] = true;
                        changed = true;
                        newly_ruled_out.insert(entry.coords[i]);
                    }
                }
            }

            for (row, col) in newly_ruled_out {
                self.eliminate_size_at(row, col, 3);
            }

            // A cell just confirmed a genuine Cruiser hit forbids any OTHER
            // ship from being Chebyshev-adjacent to it — same rule
            // `try_place` enforces, applied around a single confirmed cell
            // rather than waiting for the whole 2-Cruiser layout to lock
            // via `lock_in_cruiser_layout` (which may never happen; a
            // permanently ambiguous layout shouldn't cost this deduction
            // too). Also explicitly strips size 2 at the confirmed cell
            // itself — unlike the neighbours, the cell's own ordinary
            // `apply_hit` never does this on its own: its bag's bound was
            // exactly 3 (not >2), so only sizes >3 got stripped there.
            //
            // `apply_adjacency_elimination_around` is passed `3` as this
            // cell's own size specifically so it DOESN'T also eliminate
            // size 3 at the neighbours — a single confirmed cell, on its
            // own, doesn't yet know where the REST of its own Cruiser is;
            // a straight-line neighbour could easily be that very cell,
            // which is not a rule violation at all. Only eliminating every
            // OTHER size is safe without the full layout (contrast
            // `lock_in_cruiser_layout`, which DOES know all 6 real cells
            // and so can safely eliminate size 3 too, explicitly excluding
            // them from that pass).
            for (row, col) in &newly_confirmed {
                self.eliminate_size_at(*row, *col, 2);
            }
            for (row, col) in newly_confirmed {
                self.apply_adjacency_elimination_around(row, col, 3);
            }

            if !changed {
                break;
            }
        }
    }

    /// Every 3-bearing salvo processed so far, in order. Exposed for the
    /// debug/inspector UI.
    pub fn cross3_entries(&self) -> &[Cross3Entry] {
        &self.cross3_entries
    }

    /// Re-check, for every cross-2 entry, whether each of its 3 ORIGINAL fired
    /// coordinates could still possibly be the real Frigate hit that produced
    /// that salvo's "2" — mirrors `refresh_cross3_entry_flags` one size down,
    /// but deliberately stops at this single "traffic light" read-out: purely
    /// the size-2 FSM's own alive value fed straight back into the Cross-2
    /// Bag's coordinate list, with no combination search, same-Frigate
    /// pairing, or "found" identification layered on top (Frigate discovery
    /// is intentionally not attempted — only sinking is tracked). A cell on
    /// the outer ring never holds a ship of size >=2, so it's ruled out
    /// immediately; an inner cell is ruled out once its own alive value for
    /// size 2 has dropped to zero (see `alive_value`) — whether a Cruiser or
    /// Battleship might ALSO still be possible there is irrelevant to
    /// whether THIS cell could be a Frigate. Called at the end of every
    /// round so `coord_ruled_out` always reflects everything the FSM has
    /// deduced so far, not just what was known when the entry was created.
    fn refresh_cross2_entry_flags(&mut self) {
        let is_ruled_out = |ai: &Self, row: usize, col: usize| {
            if (INNER_LO..=INNER_HI).contains(&row) && (INNER_LO..=INNER_HI).contains(&col) {
                // Whether this coordinate could still be THIS bag's Frigate
                // hit depends only on size 2's own alive value — whether a
                // Cruiser or Battleship might ALSO still be possible here is
                // irrelevant to that question (mirrors
                // `refresh_cross3_entry_flags`, which only ever checks its
                // own size 3 for the same reason).
                ai.alive_value(row, col, 2) == 0
            } else {
                true // outer ring: never holds a ship of size >=2 in the first place
            }
        };
        let flags: Vec<[bool; 3]> = self
            .cross2_entries
            .iter()
            .map(|entry| {
                let mut flags = [false; 3];
                for (i, &(r, c)) in entry.coords.iter().enumerate() {
                    flags[i] = is_ruled_out(self, r, c);
                }
                flags
            })
            .collect();
        // OR-merge, never overwrite — see `refresh_cross3_entry_flags`'s
        // identical reasoning, one size up.
        for (entry, entry_flags) in self.cross2_entries.iter_mut().zip(flags) {
            for i in 0..3 {
                entry.coord_ruled_out[i] |= entry_flags[i];
            }
        }
        self.derive_confirmed_frigate_hits_by_elimination();
    }

    /// Mirrors `derive_confirmed_cruiser_hits_by_elimination` one size
    /// down — same 2 counting rules (N "2"s in the bag need exactly N
    /// confirmed/open coordinates, not just "at least one"; see that
    /// function's doc comment for why the naive "any 2 present" version was
    /// unsound), same coordinate-identity propagation across entries, same
    /// fixed-point iteration, same reason it doesn't reintroduce the
    /// soundness bug 35e8c16 removed, and same feed into the real size-2
    /// FSM via `eliminate_size_at` for a newly-ruled-out coordinate, so this
    /// actually changes `choose_shots`/the heatmaps rather than only the
    /// debug display. Frigate exact-cell/layout identification via
    /// candidate-window enumeration remains untouched (still not
    /// attempted) — this only ever reasons about literal coordinates and
    /// bag counts, never window shapes.
    fn derive_confirmed_frigate_hits_by_elimination(&mut self) {
        loop {
            let mut changed = false;
            // See `derive_confirmed_cruiser_hits_by_elimination`'s identical
            // comment on why this is tracked separately from `newly_ruled_out`.
            let mut newly_confirmed: std::collections::HashSet<(usize, usize)> = std::collections::HashSet::new();

            for entry in &mut self.cross2_entries {
                let n = entry.values.iter().filter(|&&v| v == 2).count();
                if n == 0 {
                    continue;
                }
                let open: Vec<usize> = (0..3).filter(|&i| !entry.coord_ruled_out[i]).collect();
                if open.len() == n {
                    for &i in &open {
                        if !entry.coord_confirmed_frigate_hit[i] {
                            entry.coord_confirmed_frigate_hit[i] = true;
                            changed = true;
                            newly_confirmed.insert(entry.coords[i]);
                        }
                    }
                }
            }

            let confirmed_coords: std::collections::HashSet<(usize, usize)> = self
                .cross2_entries
                .iter()
                .flat_map(|e| e.coords.iter().zip(e.coord_confirmed_frigate_hit.iter()).filter(|(_, &c)| c).map(|(&coord, _)| coord))
                .collect();

            for entry in &mut self.cross2_entries {
                if !entry.values.contains(&2) {
                    continue;
                }
                for i in 0..3 {
                    if !entry.coord_confirmed_frigate_hit[i] && confirmed_coords.contains(&entry.coords[i]) {
                        entry.coord_confirmed_frigate_hit[i] = true;
                        changed = true;
                        newly_confirmed.insert(entry.coords[i]);
                    }
                }
            }

            // See `derive_confirmed_cruiser_hits_by_elimination`'s identical
            // comment on why this is collected rather than applied inline.
            let mut newly_ruled_out: std::collections::HashSet<(usize, usize)> = std::collections::HashSet::new();

            for entry in &mut self.cross2_entries {
                let n = entry.values.iter().filter(|&&v| v == 2).count();
                if n == 0 {
                    continue;
                }
                let confirmed_count = (0..3).filter(|&i| entry.coord_confirmed_frigate_hit[i]).count();
                if confirmed_count != n {
                    continue;
                }
                for i in 0..3 {
                    if !entry.coord_confirmed_frigate_hit[i] && !entry.coord_ruled_out[i] {
                        entry.coord_ruled_out[i] = true;
                        changed = true;
                        newly_ruled_out.insert(entry.coords[i]);
                    }
                }
            }

            for (row, col) in newly_ruled_out {
                self.eliminate_size_at(row, col, 2);
            }

            // A cell just confirmed a genuine Frigate hit forbids any OTHER
            // ship from being Chebyshev-adjacent to it — mirrors
            // `derive_confirmed_cruiser_hits_by_elimination`'s identical
            // step one size up, including passing `2` as this cell's own
            // size so size 2 itself is left untouched at the neighbours
            // (same reasoning: a straight-line neighbour could easily be
            // this very Frigate's own other cell). No extra own-cell
            // elimination needed here, unlike the Cruiser side: this
            // cell's own ordinary `apply_hit` already had bound == 2,
            // which strips every size >2 (3 and 4) there on its own.
            for (row, col) in newly_confirmed {
                self.apply_adjacency_elimination_around(row, col, 2);
            }

            if !changed {
                break;
            }
        }
    }

    /// Eliminates every ship size EXCEPT `own_size` at every cell
    /// orthogonally or diagonally adjacent to (row, col) — the same "no
    /// OTHER ship may be Chebyshev-adjacent" rule `try_place` enforces,
    /// applied around a single confirmed ship cell rather than a
    /// fully-known ship's whole layout (contrast `lock_in_cruiser_layout`/
    /// `lock_in_frigate_layout`, which know the complete 6-cell layout and
    /// so can safely eliminate that class's own size too, explicitly
    /// excluding the ship's own cells from that pass).
    ///
    /// Deliberately does NOT eliminate `own_size` itself: a cell confirmed
    /// as (for example) a Cruiser hit doesn't by itself prove which of its
    /// neighbours are or aren't the SAME Cruiser's own other cells — a
    /// straight-line neighbour could easily be exactly that, which isn't a
    /// rule violation at all (2 cells of the SAME ship are of course
    /// adjacent to each other). Every OTHER size, though, is always safe
    /// to eliminate regardless: whether a neighbour turns out to be part
    /// of the SAME ship, water, or (impossibly) a different ship, it's
    /// never a DIFFERENT-sized ship in any of those cases.
    fn apply_adjacency_elimination_around(&mut self, row: usize, col: usize, own_size: usize) {
        for dr in -1isize..=1 {
            for dc in -1isize..=1 {
                if dr == 0 && dc == 0 {
                    continue;
                }
                let nr = row as isize + dr;
                let nc = col as isize + dc;
                if !(INNER_LO as isize..=INNER_HI as isize).contains(&nr) || !(INNER_LO as isize..=INNER_HI as isize).contains(&nc) {
                    continue;
                }
                let (nr, nc) = (nr as usize, nc as usize);
                for &size in &[4usize, 3, 2] {
                    if size != own_size {
                        self.eliminate_size_at(nr, nc, size);
                    }
                }
            }
        }
    }

    /// Every 2-bearing salvo processed so far, in order. Exposed for the
    /// debug/inspector UI.
    pub fn cross2_entries(&self) -> &[Cross2Entry] {
        &self.cross2_entries
    }

    /// Re-check, for every cross-4 entry, whether each of its 3 original
    /// fired coordinates could still possibly be the real Battleship hit
    /// that produced that salvo's "4" — mirrors `refresh_cross3_entry_flags`/
    /// `refresh_cross2_entry_flags` one size up, but simpler: there's only
    /// one Battleship, so no found-ship entry overrides or same-ship pairing
    /// are needed here, just the generic "has the combined size-4 alive
    /// value at this coordinate dropped to zero" check.
    fn refresh_cross4_entry_flags(&mut self) {
        let is_ruled_out = |ai: &Self, row: usize, col: usize| {
            if (INNER_LO..=INNER_HI).contains(&row) && (INNER_LO..=INNER_HI).contains(&col) {
                ai.alive_value(row, col, 4) == 0
            } else {
                true // outer ring: never a Battleship cell in the first place
            }
        };
        let flags: Vec<[bool; 3]> = self
            .cross4_entries
            .iter()
            .map(|entry| {
                let mut flags = [false; 3];
                for (i, &(r, c)) in entry.coords.iter().enumerate() {
                    flags[i] = is_ruled_out(self, r, c);
                }
                flags
            })
            .collect();
        // OR-merge, never overwrite: `derive_confirmed_battleship_hits_by_
        // elimination` below can also set `coord_ruled_out` bits (from bag
        // arithmetic via cross-entry propagation, not the FSM), and those
        // must survive next round's refresh too — mirrors `refresh_cross3_
        // entry_flags`/`refresh_cross2_entry_flags`'s identical reasoning.
        for (entry, entry_flags) in self.cross4_entries.iter_mut().zip(flags) {
            for i in 0..3 {
                entry.coord_ruled_out[i] |= entry_flags[i];
            }
        }
        // If an entry's own OTHER 2 candidates are already ruled out, its
        // one remaining candidate is confirmed with total certainty — see
        // `derive_confirmed_battleship_hits_by_elimination`. Runs before
        // the window-pruning step below so a cell confirmed this round
        // feeds it immediately.
        self.derive_confirmed_battleship_hits_by_elimination();
        // Once at least one cell is a confirmed Battleship hit, the real
        // ship must be one of the straight-4 windows passing through it —
        // see `prune_candidates_not_through_confirmed`.
        self.prune_candidates_not_through_confirmed();
    }

    /// Once a Cross-4 entry's bag contains a "4" and every coordinate but
    /// one has been ruled out, that one is a certain Battleship hit —
    /// mirrors `derive_confirmed_cruiser_hits_by_elimination`/
    /// `derive_confirmed_frigate_hits_by_elimination` one size up, with one
    /// simplification: a bag can never need more than 1 confirmed
    /// coordinate here (there's only 1 Battleship, so N is always exactly
    /// 1) — the "how many does this bag actually need explained" counting
    /// bug those 2 functions had to guard against structurally can't recur
    /// here, since a second real "4" in the same bag is impossible.
    ///
    /// Also propagates across entries by coordinate identity, same as the
    /// Cruiser/Frigate versions: a cell confirmed here is the SAME
    /// physical board cell wherever else it was fired (e.g. a refire), so
    /// any OTHER entry containing that exact coordinate and also holding a
    /// "4" has its own "4" already explained by it — its other 2
    /// coordinates can be ruled out too, fed into the real size-4 FSM via
    /// `eliminate_size_at`. Iterates to a fixed point.
    ///
    /// Every newly confirmed coordinate also:
    /// - has size 3 and size 2 explicitly stripped at itself — ordinary
    ///   `apply_hit` doesn't do this for a bound-4 hit on its own (there's
    ///   no size larger than 4 to trigger the ">bound" rule, and the
    ///   "value missing from bag entirely" rule is bag-wide, not specific
    ///   to which of the 3 cells actually held the 4).
    /// - gets `apply_adjacency_elimination_around(row, col, 4)` applied —
    ///   the same per-cell adjacency elimination Cruiser/Frigate confirmed
    ///   cells get, eliminating every OTHER size at its 8 neighbours
    ///   without needing to know the rest of the Battleship's own layout
    ///   (which may not resolve to a single straight-4 window for a while
    ///   yet, or ever, the same way a Cruiser/Frigate layout might not).
    fn derive_confirmed_battleship_hits_by_elimination(&mut self) {
        loop {
            let mut changed = false;
            let mut newly_confirmed: std::collections::HashSet<(usize, usize)> = std::collections::HashSet::new();

            for entry in &mut self.cross4_entries {
                if !entry.values.contains(&4) {
                    continue;
                }
                let open: Vec<usize> = (0..3).filter(|&i| !entry.coord_ruled_out[i]).collect();
                if let [only] = open[..] {
                    if !entry.coord_confirmed_battleship_hit[only] {
                        entry.coord_confirmed_battleship_hit[only] = true;
                        changed = true;
                        newly_confirmed.insert(entry.coords[only]);
                    }
                }
            }

            let confirmed_coords: std::collections::HashSet<(usize, usize)> = self
                .cross4_entries
                .iter()
                .flat_map(|e| e.coords.iter().zip(e.coord_confirmed_battleship_hit.iter()).filter(|(_, &c)| c).map(|(&coord, _)| coord))
                .collect();

            for entry in &mut self.cross4_entries {
                if !entry.values.contains(&4) {
                    continue;
                }
                for i in 0..3 {
                    if !entry.coord_confirmed_battleship_hit[i] && confirmed_coords.contains(&entry.coords[i]) {
                        entry.coord_confirmed_battleship_hit[i] = true;
                        changed = true;
                        newly_confirmed.insert(entry.coords[i]);
                    }
                }
            }

            // See `derive_confirmed_cruiser_hits_by_elimination`'s identical
            // comment on why this is collected rather than applied inline.
            let mut newly_ruled_out: std::collections::HashSet<(usize, usize)> = std::collections::HashSet::new();

            for entry in &mut self.cross4_entries {
                if !entry.values.contains(&4) {
                    continue;
                }
                // N is always exactly 1 here (only 1 Battleship exists) —
                // once ANY coordinate is confirmed, every other coordinate
                // in this same entry is definitely not the Battleship hit.
                if !(0..3).any(|i| entry.coord_confirmed_battleship_hit[i]) {
                    continue;
                }
                for i in 0..3 {
                    if !entry.coord_confirmed_battleship_hit[i] && !entry.coord_ruled_out[i] {
                        entry.coord_ruled_out[i] = true;
                        changed = true;
                        newly_ruled_out.insert(entry.coords[i]);
                    }
                }
            }

            for (row, col) in newly_ruled_out {
                self.eliminate_size_at(row, col, 4);
            }

            for (row, col) in &newly_confirmed {
                self.eliminate_size_at(*row, *col, 3);
                self.eliminate_size_at(*row, *col, 2);
            }
            for (row, col) in newly_confirmed {
                self.apply_adjacency_elimination_around(row, col, 4);
            }

            if !changed {
                break;
            }
        }
    }

    /// Once at least one cell is confirmed a genuine Battleship hit (see
    /// `derive_confirmed_battleship_hits_by_elimination`), the real ship
    /// must be exactly one of the straight-4 windows passing through EVERY
    /// confirmed cell — any candidate that isn't part of at least one such
    /// window can be eliminated outright, the same way
    /// `apply_battleship_cross_elimination`'s union-of-crosses trick
    /// already eliminates candidates outside the running cross
    /// intersection.
    fn prune_candidates_not_through_confirmed(&mut self) {
        let confirmed: Vec<(usize, usize)> = self
            .cross4_entries
            .iter()
            .flat_map(|e| {
                e.coords
                    .iter()
                    .zip(e.coord_confirmed_battleship_hit.iter())
                    .filter(|(_, &c)| c)
                    .map(|(&coord, _)| coord)
            })
            .collect();
        if confirmed.is_empty() {
            return;
        }
        let windows = self.battleship_candidate_windows();
        let surviving: Vec<&[(usize, usize); 4]> =
            windows.iter().filter(|w| confirmed.iter().all(|c| w.contains(c))).collect();
        if surviving.is_empty() {
            return; // shouldn't happen on a consistent board; don't guess
        }
        let mut safe = [[false; 10]; 10];
        for window in &surviving {
            for &(r, c) in window.iter() {
                safe[r][c] = true;
            }
        }
        self.drop_candidates(|ai| {
            let mut dead = Vec::new();
            for row in INNER_LO..=INNER_HI {
                for col in INNER_LO..=INNER_HI {
                    if ai.battleship_candidates[row][col] && !safe[row][col] {
                        dead.push((row, col));
                    }
                }
            }
            dead
        });
    }

    /// Every 4-bearing salvo processed so far, in order. Exposed for the
    /// debug/inspector UI.
    pub fn cross4_entries(&self) -> &[Cross4Entry] {
        &self.cross4_entries
    }

    /// Snapshot of every cross-3 entry's `coord_ruled_out` flags, in entry
    /// order — take one of these before a round, then pass it to
    /// `newly_ruled_out_since` afterwards to find out what changed.
    pub fn cross3_ruled_out_snapshot(&self) -> Vec<[bool; 3]> {
        self.cross3_entries.iter().map(|e| e.coord_ruled_out).collect()
    }

    /// Coordinates that flipped from "not ruled out" to "ruled out" since
    /// `before` was captured. Entries are only ever appended, never reordered
    /// or removed, so entry index `i` in `before` still refers to the same
    /// entry it did when the snapshot was taken — an entry created since
    /// then (`before.get(i)` -> `None`) is correctly treated as "wasn't ruled
    /// out before" for every one of its coordinates.
    pub fn newly_ruled_out_since(&self, before: &[[bool; 3]]) -> Vec<(usize, usize)> {
        let mut result = Vec::new();
        for (i, entry) in self.cross3_entries.iter().enumerate() {
            for j in 0..3 {
                let was_ruled_out = before.get(i).map(|f| f[j]).unwrap_or(false);
                if entry.coord_ruled_out[j] && !was_ruled_out {
                    result.push(entry.coords[j]);
                }
            }
        }
        result
    }

    /// If at least 2 four-bearing salvos have happened, check whether exactly one
    /// straight 4-long placement (out of all 80 horizontal/vertical placements on
    /// the inner 8x8) still has every one of its cells marked possible in
    /// `battleship_candidates`. If so, that placement can't be anything BUT the
    /// Battleship — no other placement has room to exist any more.
    ///
    /// This deliberately does NOT require the *whole* candidate mask to have
    /// shrunk to exactly those 4 cells. Stray cells can survive elsewhere that
    /// don't happen to complete any full straight-4 window of their own (e.g. an
    /// odd cell left over from an intersection that never lines up with 3 more
    /// neighbours) — such cells are harmless noise, not a competing placement,
    /// and requiring the mask to be pixel-perfect would make the deduction far
    /// more fragile than the underlying logic actually is.
    fn battleship_identified(&self) -> Option<[(usize, usize); 4]> {
        if self.four_bearing_salvo_count < 2 {
            return None;
        }
        let windows = self.battleship_candidate_windows();
        if let [only] = windows[..] {
            Some(only)
        } else {
            None
        }
    }

    /// Every valid straight-4 window still consistent with the current
    /// Battleship candidate mask (`battleship_candidates`) — the full list
    /// `battleship_identified` collapses down to a single result (or none)
    /// for, exposed here so a genuinely discriminating test shot can be
    /// chosen when more than one window survives (see
    /// `battleship_discriminating_test_cell`).
    fn battleship_candidate_windows(&self) -> Vec<[(usize, usize); 4]> {
        let is_possible = |r: usize, c: usize| self.battleship_candidates[r][c];
        let mut windows = Vec::new();

        // Horizontal placements: every row, every starting column that keeps all
        // 4 cells within the inner 1..=8 range.
        for row in INNER_LO..=INNER_HI {
            for start in INNER_LO..=(INNER_HI - 3) {
                let cells = [(row, start), (row, start + 1), (row, start + 2), (row, start + 3)];
                if cells.iter().all(|&(r, c)| is_possible(r, c)) {
                    windows.push(cells);
                }
            }
        }
        // Vertical placements: symmetric, varying row instead of column.
        for col in INNER_LO..=INNER_HI {
            for start in INNER_LO..=(INNER_HI - 3) {
                let cells = [(start, col), (start + 1, col), (start + 2, col), (start + 3, col)];
                if cells.iter().all(|&(r, c)| is_possible(r, c)) {
                    windows.push(cells);
                }
            }
        }
        windows
    }

    /// A coordinate present in SOME but not EVERY still-possible Battleship
    /// window — firing there teaches something (a miss rules out every
    /// window containing it; a hit rules out every window that does NOT),
    /// unlike a coordinate common to every surviving window, which is a
    /// guaranteed hit but resolves nothing about which exact window is
    /// real. `None` when
    /// fewer than 2 windows survive (nothing left to discriminate
    /// between), or in the degenerate case where every remaining
    /// candidate cell happens to be common to every window.
    ///
    /// Requires at least 2 cross-4 salvos, same as `battleship_identified` —
    /// after just 1, `battleship_candidates` is still a single raw cross,
    /// not yet narrowed by any actual intersection, so it trivially
    /// contains dozens of straight-4 sub-windows that don't reflect
    /// genuine remaining ambiguity about the real ship. "Discriminating"
    /// between those is noise, not signal: it can pick a coordinate with
    /// almost no elimination value at all over one sitting in the middle
    /// of the busiest, most-informative part of the cross, purely because
    /// it happens to not appear in every one of those largely-arbitrary
    /// sub-windows. Once a 2nd salvo's intersection has actually narrowed
    /// the candidate set, the survivors are few enough that discriminating
    /// between them is meaningful again.
    fn battleship_discriminating_test_cell(&self) -> Option<(usize, usize)> {
        if self.four_bearing_salvo_count < 2 {
            return None;
        }
        let windows = self.battleship_candidate_windows();
        if windows.len() < 2 {
            return None;
        }
        let is_discriminating = |r: usize, c: usize| !windows.iter().all(|w| w.contains(&(r, c)));

        // Prefer a fresh, never-fired discriminating cell when one exists.
        for window in &windows {
            for &(r, c) in window {
                if !self.fired[r][c] && is_discriminating(r, c) {
                    return Some((r, c));
                }
            }
        }

        // Every discriminating cell may already be fired — e.g. all of them
        // were ambiguous decoys in earlier cross-4 salvos (a "5 adjacent
        // candidates, 2 overlapping windows" cluster where both outer cells
        // were already part of some salvo). Re-firing one is still the ONLY
        // way to ever resolve which window is real: every OTHER surviving
        // candidate is common to every window, so hitting one just confirms
        // a hit without discriminating anything — the ambiguity would
        // otherwise never close. See `is_battleship_discriminating_refire`,
        // which lets this specific refire through even with the general
        // refire-allowed toggle off.
        for window in &windows {
            for &(r, c) in window {
                if is_discriminating(r, c) {
                    return Some((r, c));
                }
            }
        }
        None
    }

    /// True if (row, col) is the coordinate `choose_shots` deliberately
    /// picked via `battleship_discriminating_test_cell` — used by
    /// `Game::fire` to let that one specific refire through even when the
    /// general refire-allowed toggle is off.
    pub fn is_battleship_discriminating_refire(&self, row: usize, col: usize) -> bool {
        self.fired[row][col] && self.battleship_discriminating_test_cell() == Some((row, col))
    }

    /// Same idea as `is_battleship_discriminating_refire`, for
    /// `anchored_isolation_shot`: the anchor cell is a confirmed Battleship
    /// cell, almost always already fired by the time Cruiser/Frigate
    /// disambiguation runs (the class can't be "fully sunk" otherwise) —
    /// and the cross-exclusive Cruiser/Frigate cell it's paired with is
    /// frequently already fired too (that's exactly what made its
    /// probability resolvable in the first place). Without this, `Game::
    /// fire` would reject the whole deliberate salvo as an ordinary
    /// refire.
    pub fn is_anchored_isolation_refire(&self, row: usize, col: usize) -> bool {
        self.fired[row][col] && self.anchored_isolation_shot().is_some_and(|shots| shots.contains(&(row, col)))
    }

    /// The Battleship's exact 4-cell layout, once `battleship_identified` has
    /// deduced it — for the UI to render solid rather than the tentative
    /// "candidate" outline. Empty until then.
    pub fn battleship_identified_cells(&self) -> Vec<(usize, usize)> {
        match self.battleship_identified() {
            Some(cells) => cells.to_vec(),
            None => Vec::new(),
        }
    }

    /// The Battleship's exact 4-cell layout, permanently, once confirmed
    /// sunk (see `found_battleship`) — for the UI to keep rendering it even
    /// after the live candidate/identified state is cleared. Empty if the
    /// ship hasn't sunk yet, or if it sank via ordinary fire before
    /// `battleship_identified` ever narrowed things down to one window.
    pub fn found_battleship_cells(&self) -> Vec<(usize, usize)> {
        match self.found_battleship {
            Some(cells) => cells.to_vec(),
            None => Vec::new(),
        }
    }

    /// The Cruisers' exact 6-cell layout (both ships combined), once
    /// `consistent_cruiser_candidates` has narrowed to a single remaining
    /// hypothesis — treated as ground truth (see
    /// `cells_confirmed_cruiser_or_adjacent`'s doc comment for the
    /// empirical validation behind that). Empty until then. Unlike
    /// `found_battleship_cells`, no separate permanent snapshot is needed:
    /// `consistent_cruiser_candidates` is a pure function of
    /// `salvo_history` alone, and once narrowed to 1 it can only ever stay
    /// at that same single candidate (more evidence can only shrink the
    /// consistent set further, never revive an eliminated hypothesis —
    /// and the true layout, by construction, is never eliminated).
    pub fn cruiser_identified_cells(&self) -> Vec<(usize, usize)> {
        let candidates = self.consistent_cruiser_candidates();
        if candidates.len() == 1 {
            candidates[0].iter().copied().collect()
        } else {
            Vec::new()
        }
    }

    /// Same idea as `cruiser_identified_cells`, one size down: the
    /// Frigates' exact 6-cell layout (all 3 ships combined), once
    /// `consistent_frigate_candidates` has narrowed to a single remaining
    /// hypothesis. Empty until then.
    pub fn frigate_identified_cells(&self) -> Vec<(usize, usize)> {
        let candidates = self.consistent_frigate_candidates();
        if candidates.len() == 1 {
            candidates[0].iter().copied().collect()
        } else {
            Vec::new()
        }
    }

    /// Same idea as `cruiser_identified_cells`, but checked against
    /// `cross_reasoned_cruiser_candidates` instead of the raw
    /// `consistent_cruiser_candidates` — resolves cases where the Cruiser
    /// heatmap alone still shows more than one hypothesis, but every
    /// hypothesis except one collides with every remaining Frigate
    /// hypothesis (see `jointly_consistent_hypothesis_pairs`). Since a
    /// hypothesis can appear more than once in the cross-reasoned list (one
    /// entry per compatible partner), "exactly one remains" means exactly
    /// one DISTINCT cell-set, not exactly one list entry — see
    /// `distinct_hypotheses`.
    pub fn cruiser_identified_cells_refined(&self) -> Vec<(usize, usize)> {
        let candidates = self.cross_reasoned_cruiser_candidates();
        let distinct = Self::distinct_hypotheses(&candidates);
        if distinct.len() == 1 {
            distinct[0].iter().copied().collect()
        } else {
            Vec::new()
        }
    }

    /// Same idea as `cruiser_identified_cells_refined`, one size down.
    pub fn frigate_identified_cells_refined(&self) -> Vec<(usize, usize)> {
        let candidates = self.cross_reasoned_frigate_candidates();
        let distinct = Self::distinct_hypotheses(&candidates);
        if distinct.len() == 1 {
            distinct[0].iter().copied().collect()
        } else {
            Vec::new()
        }
    }

    /// Length of the maximal run of contiguous `true` cells in `mask`, along `row`,
    /// that includes `col` (inner 8x8 only).
    fn max_contiguous_run_horizontal(mask: &[[bool; 10]; 10], row: usize, col: usize) -> usize {
        let mut left = col;
        while left > INNER_LO && mask[row][left - 1] {
            left -= 1;
        }
        let mut right = col;
        while right < INNER_HI && mask[row][right + 1] {
            right += 1;
        }
        right - left + 1
    }

    /// Same as `max_contiguous_run_horizontal`, but along `col` instead of `row`.
    fn max_contiguous_run_vertical(mask: &[[bool; 10]; 10], row: usize, col: usize) -> usize {
        let mut top = row;
        while top > INNER_LO && mask[top - 1][col] {
            top -= 1;
        }
        let mut bottom = row;
        while bottom < INNER_HI && mask[bottom + 1][col] {
            bottom += 1;
        }
        bottom - top + 1
    }

    /// A size-4 Battleship needs 4 consecutive candidate cells in a straight line to
    /// physically fit. Any candidate cell that isn't part of *some* horizontal or
    /// vertical run of at least 4 (measured against the current candidate mask
    /// itself) can never actually host the ship, no matter which of its neighbours
    /// turn out true — so it's a contradiction and can be eliminated outright.
    ///
    /// This IS the "feed the row/col FSM's eliminations back into the cross-4 bag"
    /// step: a contiguous run of true cells is exactly the set of cells the size-4
    /// FSM would consider still-alive for a placement through that row/column, so
    /// "no run >= 4 in either direction" and "the FSM has no surviving placement
    /// through this cell" are the same statement. Every caller (both the cross-
    /// intersection path and the plain-miss path in `apply_salvo`) re-invokes this
    /// after any change to the mask, so newly-dead cells are re-fed the moment they
    /// appear, not just when the next 4-bearing salvo happens to trigger it.
    ///
    /// Dropping a cell can shrink the run a neighbouring cell was relying on, so
    /// this re-scans until a full pass removes nothing further (fixed point).
    fn prune_candidates_without_room(&mut self) {
        loop {
            let removed = self.drop_candidates(|ai| {
                let mut dead = Vec::new();
                for row in INNER_LO..=INNER_HI {
                    for col in INNER_LO..=INNER_HI {
                        if !ai.battleship_candidates[row][col] {
                            continue;
                        }
                        let h = Self::max_contiguous_run_horizontal(&ai.battleship_candidates, row, col);
                        let v = Self::max_contiguous_run_vertical(&ai.battleship_candidates, row, col);
                        if h < 4 && v < 4 {
                            dead.push((row, col));
                        }
                    }
                }
                dead
            });
            if removed == 0 {
                break;
            }
        }
    }

    /// Shared plumbing for both narrowing passes above: run `find_dead` to collect
    /// cells that should no longer be Battleship candidates, clear them from
    /// `battleship_candidates` (which is what drives the yellow "cross candidate"
    /// border in the UI, so those borders disappear on the next redraw), and feed
    /// each newly-dead cell into the size-4 FSM exactly as a real miss would be —
    /// while `battleship_cross_processed` ensures we never redrive the same
    /// cell's FSM transition twice. Returns how many cells were dropped.
    ///
    /// Eliminates unconditionally, even for already-fired cells: a salvo whose
    /// bound is 4 never eliminates size 4 for any of its 3 cells via the normal
    /// apply_hit path (any of them could ambiguously be the real hit) — see
    /// `apply_salvo`. So a decoy cell fired as part of a 4-bearing salvo would
    /// otherwise never get size 4 eliminated at all, even once it's later
    /// proven not to be the Battleship. Safe to call more than once on the same
    /// cell — the FSM transition tables are idempotent per column.
    fn drop_candidates(&mut self, find_dead: impl FnOnce(&Self) -> Vec<(usize, usize)>) -> usize {
        let dead = find_dead(self);
        let count = dead.len();
        for &(row, col) in &dead {
            self.battleship_candidates[row][col] = false;
        }
        for (row, col) in dead {
            if !self.battleship_cross_processed[row][col] {
                self.battleship_cross_processed[row][col] = true;
                self.eliminate_size_at(row, col, 4);
            }
        }
        count
    }

    /// Cells the Battleship could still occupy, per the cross-deduction trick.
    /// Returns an empty vec until at least one salvo with a 4 has been seen —
    /// before that there's no real constraint to show.
    pub fn battleship_candidate_cells(&self) -> Vec<(usize, usize)> {
        if !self.battleship_cross_seen {
            return Vec::new();
        }
        let mut cells = Vec::new();
        for row in INNER_LO..=INNER_HI {
            for col in INNER_LO..=INNER_HI {
                if self.battleship_candidates[row][col] {
                    cells.push((row, col));
                }
            }
        }
        cells
    }

    /// Score a single cell: sum of elimination "value" across row-FSM and col-FSM for
    /// each ship size not yet fully found, plus a bonus if it's still a submarine
    /// candidate (size 1).
    ///
    /// Not currently used by `choose_shots` — shot selection is scoped to the
    /// Battleship (size 4) FSM only for now, via `best_size4_cell` below. Kept
    /// here for when size 3 / size 2 / submarine get folded into the same
    /// sequential-selection approach.
    #[allow(dead_code)]
    fn score_cell(&self, row: usize, col: usize) -> u32 {
        if self.fired[row][col] {
            return 0;
        }
        let mut score = 0u32;

        let cell_is_inner = (INNER_LO..=INNER_HI).contains(&row) && (INNER_LO..=INNER_HI).contains(&col);
        if cell_is_inner {
            let table_col = col - INNER_LO;
            let table_row = row - INNER_LO;
            for &size in &SHIP_SIZES {
                if self.size_fully_found(size) {
                    continue;
                }
                let cur_row = Self::line_state_for_size(&self.row_state[row], size);
                let cur_col = Self::line_state_for_size(&self.col_state[col], size);
                let (v_row, v_col) = match size {
                    4 => (VALUES_SIZE4[cur_row][table_col], VALUES_SIZE4[cur_col][table_row]),
                    3 => (VALUES_SIZE3[cur_row][table_col], VALUES_SIZE3[cur_col][table_row]),
                    2 => (VALUES_SIZE2[cur_row][table_col], VALUES_SIZE2[cur_col][table_row]),
                    _ => unreachable!(),
                };
                score += v_row as u32 + v_col as u32;
            }
        }

        if !self.size_fully_found(1) && self.sub_candidates[row][col] {
            score += 1;
        }

        score
    }

    /// Elimination "value" for `size` in a given FSM state/table-index, per the
    /// pre-generated tables — the single per-size lookup `size_cell_score` and
    /// `apply_hypothetical_miss` build on.
    fn line_state_score(state: usize, size: usize, table_index: usize) -> u32 {
        match size {
            4 => VALUES_SIZE4[state][table_index] as u32,
            3 => VALUES_SIZE3[state][table_index] as u32,
            2 => VALUES_SIZE2[state][table_index] as u32,
            _ => 0,
        }
    }

    /// FSM transition for `size` in a given state/table-index. Companion to
    /// `line_state_score` for the hypothetical-miss folding below.
    fn line_state_transition(state: usize, size: usize, table_index: usize) -> usize {
        match size {
            4 => TRANSITIONS_SIZE4[state][table_index] as usize,
            3 => TRANSITIONS_SIZE3[state][table_index] as usize,
            2 => TRANSITIONS_SIZE2[state][table_index] as usize,
            _ => state,
        }
    }

    /// Elimination score for a single cell under `size`'s FSM, given a
    /// *hypothetical* working copy of that size's row/col FSM states (as opposed
    /// to `self.row_state`/`self.col_state`, which reflect only confirmed info).
    fn size_cell_score(row_line: &[usize; 10], col_line: &[usize; 10], row: usize, col: usize, size: usize) -> u32 {
        if !(INNER_LO..=INNER_HI).contains(&row) || !(INNER_LO..=INNER_HI).contains(&col) {
            return 0;
        }
        let table_col = col - INNER_LO;
        let table_row = row - INNER_LO;
        Self::line_state_score(row_line[row], size, table_col) + Self::line_state_score(col_line[col], size, table_row)
    }

    /// Fold a *hypothetical* miss at (row, col) into a working copy of `size`'s
    /// row/col FSM states — i.e. "if this shot comes back as a miss, what would
    /// that size's FSM look like afterwards". Mirrors `eliminate_size_at`, but
    /// operates on local scratch state rather than `self`.
    fn apply_hypothetical_miss(row_line: &mut [usize; 10], col_line: &mut [usize; 10], row: usize, col: usize, size: usize) {
        if !(INNER_LO..=INNER_HI).contains(&row) || !(INNER_LO..=INNER_HI).contains(&col) {
            return;
        }
        let table_col = col - INNER_LO;
        let table_row = row - INNER_LO;
        row_line[row] = Self::line_state_transition(row_line[row], size, table_col);
        col_line[col] = Self::line_state_transition(col_line[col], size, table_row);
    }

    /// Find the unfired, not-yet-chosen-this-salvo cell with the highest score
    /// under `score_fn` — shared search/fallback logic for both single-size
    /// scoring (`best_cell_for_size`) and the combined size-4 + size-3 scoring
    /// used once the Battleship's rough vicinity is known but not its exact cell
    /// (see `choose_shots`).
    ///
    /// Only searches the inner 8x8 — the only place a size>=2 ship can ever be,
    /// and therefore the only coordinates any of these FSMs have an opinion
    /// about. Outer-ring cells always score 0 and were previously eligible as a
    /// last-resort tie-break (since the search started at row/col 0), which
    /// could surface a useless outer-ring suggestion once the inner board
    /// saturated. Falls back to any unfired cell anywhere on the board only in
    /// the (normally unreachable) case where every inner cell is already fired
    /// or chosen.
    ///
    /// If `forbid_candidates` is set, cells currently inside the cross-deduced
    /// Battleship candidate set are excluded from consideration entirely — used
    /// to cap how many of a salvo's 3 shots may land in that region (see
    /// `choose_shots`). Falls back to allowing candidate cells after all if no
    /// eligible non-candidate cell remains, rather than picking nothing.
    ///
    /// If `require_candidates` is set instead (mutually exclusive with
    /// `forbid_candidates` — no caller sets both), only cells INSIDE that same
    /// candidate set are considered — the opposite restriction, for when a shot
    /// should specifically dig into the Battleship's already-narrowed region
    /// rather than the raw per-cell score wandering off to some untouched row
    /// or column elsewhere on the board that merely hasn't been narrowed by
    /// anything yet (see `choose_shots`'s first-pick fallback). Falls back to
    /// the ordinary unrestricted search if no eligible candidate cell remains.
    fn best_cell_by_score(
        &self,
        chosen_so_far: &[(usize, usize)],
        forbid_candidates: bool,
        require_candidates: bool,
        allow_refired: bool,
        score_fn: impl Fn(usize, usize) -> u32,
    ) -> (usize, usize) {
        let search = |forbid: bool, require: bool| {
            let mut best_score: i64 = -1;
            let mut best_cell: Option<(usize, usize)> = None;
            for r in INNER_LO..=INNER_HI {
                for c in INNER_LO..=INNER_HI {
                    if (self.fired[r][c] && !allow_refired) || chosen_so_far.contains(&(r, c)) {
                        continue;
                    }
                    if forbid && self.battleship_candidates[r][c] {
                        continue;
                    }
                    if require && !self.battleship_candidates[r][c] {
                        continue;
                    }
                    let score = score_fn(r, c) as i64;
                    if score > best_score {
                        best_score = score;
                        best_cell = Some((r, c));
                    }
                }
            }
            best_cell
        };

        if let Some(cell) = search(forbid_candidates, require_candidates) {
            return cell;
        }

        if forbid_candidates || require_candidates {
            // No eligible cell satisfying the restriction — fall back to the
            // ordinary unrestricted search rather than failing outright.
            if let Some(cell) = search(false, false) {
                return cell;
            }
        }

        // Every inner cell is spoken for. Fall back to any remaining unfired cell
        // anywhere on the board rather than returning a bogus default.
        for r in 0..10 {
            for c in 0..10 {
                if !self.fired[r][c] && !chosen_so_far.contains(&(r, c)) {
                    return (r, c);
                }
            }
        }
        (0, 0) // Unreachable in practice: would mean all 100 cells are spoken for.
    }

    /// Single-size wrapper around `best_cell_by_score` — used both for hunting
    /// whichever size `current_target_size` reports, and for the Battleship-only
    /// "test shot" in `choose_shots` (see there for when only 1 of the 3 shots
    /// gets scored this way instead of the combined size-4 + size-3 score).
    fn best_cell_for_size(
        &self,
        row_line: &[usize; 10],
        col_line: &[usize; 10],
        chosen_so_far: &[(usize, usize)],
        forbid_candidates: bool,
        require_candidates: bool,
        allow_refired: bool,
        size: usize,
    ) -> (usize, usize) {
        self.best_cell_by_score(chosen_so_far, forbid_candidates, require_candidates, allow_refired, |r, c| {
            Self::size_cell_score(row_line, col_line, r, c, size)
        })
    }

    /// Per-row and per-column FSM state for a given ship size (4, 3, or 2), one
    /// entry per line index 0..9. Exposed for the debug/inspector UI. Indices 0
    /// and 9 are outer-ring lines and don't correspond to any real placement —
    /// included for transparency, but callers should treat them as informational
    /// only (see `best_cell_for_size`, which never queries them for any size).
    pub fn line_states(&self, size: usize) -> (Vec<usize>, Vec<usize>) {
        let rows = self.row_state.iter().map(|s| Self::line_state_for_size(s, size)).collect();
        let cols = self.col_state.iter().map(|s| Self::line_state_for_size(s, size)).collect();
        (rows, cols)
    }

    /// Number of still-possible placements for `size` in a given FSM state index.
    pub fn alive_count(size: usize, state: usize) -> u8 {
        match size {
            4 => ALIVE_COUNT_SIZE4.get(state).copied().unwrap_or(0),
            3 => ALIVE_COUNT_SIZE3.get(state).copied().unwrap_or(0),
            2 => ALIVE_COUNT_SIZE2.get(state).copied().unwrap_or(0),
            _ => 0,
        }
    }

    /// Which ship size `choose_shots` is currently trying to eliminate: the
    /// largest size — in priority order Battleship (4) > Cruiser (3) > Frigate
    /// (2) — that isn't fully sunk yet. Falls back to 1 (submarines) once all
    /// three bigger classes are found; `choose_shots` doesn't target submarines
    /// itself, but this still reports "nothing bigger is left" accurately for
    /// the debug UI.
    pub fn current_target_size(&self) -> usize {
        for &size in &SHIP_SIZES {
            if !self.size_fully_found(size) {
                if size == 2 && self.freeze_before_frigates {
                    return 3;
                }
                return size;
            }
        }
        1
    }

    /// Set whether `choose_shots` may recommend an already-fired cell while
    /// hunting `size` (2, 3, or 4). Manual firing via `Game::fire` is
    /// unaffected either way — this only relaxes what the AI suggests.
    pub fn set_refire_allowed(&mut self, size: usize, allowed: bool) {
        if size < self.refire_allowed.len() {
            self.refire_allowed[size] = allowed;
        }
    }

    /// Whether `choose_shots` may currently recommend an already-fired cell
    /// while hunting `size`. See `set_refire_allowed`.
    pub fn is_refire_allowed(&self, size: usize) -> bool {
        self.refire_allowed.get(size).copied().unwrap_or(false)
    }

    /// Set whether `current_target_size` should freeze at 3 (Cruiser) rather
    /// than ever advancing to 2 (Frigate) once every Cruiser is sunk. See the
    /// `freeze_before_frigates` field doc for exactly when this does and
    /// doesn't take effect.
    pub fn set_freeze_before_frigates(&mut self, freeze: bool) {
        self.freeze_before_frigates = freeze;
    }

    /// Whether the freeze-before-Frigates toggle is currently on. See
    /// `set_freeze_before_frigates`.
    pub fn is_freeze_before_frigates(&self) -> bool {
        self.freeze_before_frigates
    }

    /// Choose the next 3 cells to fire at, scoped to whichever size
    /// `current_target_size` reports.
    ///
    /// Approach: find the shot with the most eliminations for that size. Then,
    /// *assuming that shot comes back as a miss* (i.e. every elimination it
    /// implies actually happens), fold that hypothetical result into a working
    /// copy of the FSM state and recompute the best remaining shot against the
    /// updated state. Repeat once more for the third shot. This never repeats a
    /// coordinate, since each pick is excluded from the next search.
    ///
    /// The rest of this only applies while still hunting the Battleship
    /// (size 4) — the cross-deduction machinery below is specific to it:
    ///
    /// Once the cross-deduction trick has narrowed things down (`battleship_cross_seen`)
    /// but the exact cell isn't pinned down yet, only the *first* of the 3 shots gets an
    /// unrestricted, Battleship-only look — including at cells inside the current
    /// candidate region. It specifically tries to test a coordinate that distinguishes
    /// between the surviving candidate windows (see `battleship_discriminating_test_cell`,
    /// which will refire an already-fired candidate if every discriminating cell happens
    /// to have been fired already — that's still the only way the ambiguity ever closes).
    /// Reasoning for why only one shot may ever touch the region: if two or more
    /// candidate cells were fired in the same ambiguous salvo, a 4 in the result bag
    /// still wouldn't tell us *which* of them was the hit — we'd be back to
    /// cross-intersecting again, having wasted the whole point of the deliberate,
    /// isolating test. Restricting the candidate region to that first shot means a 4 can
    /// only have come from it, collapsing the ambiguity immediately instead of costing
    /// another round of deduction. (An earlier version of this let the other 2 shots also
    /// compete for the region on raw score — reverted: it let them land on OTHER
    /// candidate cells in the very same salvo as the deliberate discriminating test,
    /// destroying that test's whole reason for existing.)
    ///
    /// The other 2 shots are forced away from the candidate region regardless — they
    /// can't help test it further this salvo — so rather than dig further into a
    /// size-4 line that's already narrowed as far as this round's information allows,
    /// they're scored purely against the Cruiser (size-3) FSM instead. A shot that
    /// can't land near the Battleship anyway might as well make progress hunting the
    /// Cruisers, rather than only ever re-confirming the same already-established
    /// Battleship candidates.
    ///
    /// If `battleship_identified` has pinned down the exact 4-cell layout, none of that
    /// applies any more — there's nothing left to protect or switch scoring for. Instead, this fills
    /// as many of the 3 shots as possible with that placement's still-unfired cells first
    /// (finishing the ship off directly), then falls back to the ordinary FSM search for
    /// any slots left over.
    pub fn choose_shots(&self) -> [(usize, usize); 3] {
        // Cruiser and Frigate discovery (pinpointing exact cells, and so
        // also disambiguating between candidate layouts) are deliberately
        // not attempted — see `refresh_cross3_entry_flags`/
        // `refresh_cross2_entry_flags`. `chosen`/`avoid_as_filler` stay
        // empty here; the loop below fills all 3 slots from the ordinary
        // FSM search.
        let mut chosen: Vec<(usize, usize)> = Vec::with_capacity(3);
        let avoid_as_filler: Vec<(usize, usize)> = Vec::new();

        let size = self.current_target_size();

        // Disambiguation takes priority over ever moving on to a smaller
        // class, but only once the Battleship and both Cruisers are fully
        // sunk — Battleship hunting always wins first regardless (matching
        // its priority everywhere else in this function), and Cruiser
        // disambiguation is deliberately checked and resolved before
        // Frigate disambiguation is even attempted: pinning down the
        // Cruisers first also shrinks the Frigate candidate search once
        // its turn comes (fewer cells left that could still be a Frigate,
        // since a confirmed Cruiser cell can never also be a Frigate
        // cell). Independent of `current_target_size`/`freeze_before_frigates`
        // (a debug-only toggle for the ordinary hunting FSM) — this is a
        // separate, higher-priority activity that only ever fires once
        // there's nothing bigger left to actually hunt.
        if self.size_fully_found(4) && self.size_fully_found(3) {
            // Try the cheap, fully-conclusive cleanup shot first — see
            // `anchored_isolation_shot`. It can resolve a Cruiser cell and
            // a Frigate cell in a single salvo whenever the heatmaps
            // already show that exact cross-exclusive pattern, which is
            // strictly better than the general minimax search's
            // worst-case narrowing.
            if let Some(shots) = self.anchored_isolation_shot() {
                return shots;
            }
            if let Some(shots) = self.cruiser_disambiguation_shots() {
                return shots;
            }
            if self.size_fully_found(2) {
                if let Some(shots) = self.frigate_disambiguation_shots() {
                    return shots;
                }
            }
        }

        // Once every size >=2 ship is sunk, only submarines are left — the
        // line-FSM tables above only cover sizes 4/3/2, so this must branch off
        // before touching them (line_state_for_size panics on anything else).
        if size == 1 {
            return self.choose_submarine_shots(chosen, &avoid_as_filler);
        }

        let mut row_line: [usize; 10] = [0; 10];
        let mut col_line: [usize; 10] = [0; 10];
        for r in 0..10 {
            row_line[r] = Self::line_state_for_size(&self.row_state[r], size);
        }
        for c in 0..10 {
            col_line[c] = Self::line_state_for_size(&self.col_state[c], size);
        }

        // The Battleship-specific cross machinery only makes sense while size 4
        // is still the target.
        let identified = if size == 4 { self.battleship_identified() } else { None };
        if let Some(cells) = identified {
            for (r, c) in cells {
                if chosen.len() >= 3 {
                    break;
                }
                if !self.fired[r][c] && !chosen.contains(&(r, c)) {
                    chosen.push((r, c));
                }
            }
        }

        // Still hunting the Battleship, hit it at least once, but haven't pinned
        // down its exact cell: the first shot targets the cross-4 bag itself
        // (see below), but every shot after that is scored purely on the
        // Cruiser FSM instead — deliberately searching OUTSIDE the cross-4
        // bag (via `forbid_candidates`) rather than digging further into a
        // region already narrowed as far as this round's information
        // allows, so these 2 shots aren't wasted only ever re-confirming
        // the same Battleship-candidate cells.
        let hunt_with_cruiser_fsm = size == 4 && identified.is_none() && self.battleship_cross_seen;
        let mut row3: [usize; 10] = [0; 10];
        let mut col3: [usize; 10] = [0; 10];
        if hunt_with_cruiser_fsm {
            for r in 0..10 {
                row3[r] = Self::line_state_for_size(&self.row_state[r], 3);
            }
            for c in 0..10 {
                col3[c] = Self::line_state_for_size(&self.col_state[c], 3);
            }
        }

        let allow_refired = self.is_refire_allowed(size);

        while chosen.len() < 3 {
            let is_first_pick = chosen.is_empty();
            // Only the first pick is allowed anywhere, including the candidate
            // region; every pick after that is forced away from it, whenever the
            // Battleship's candidate region is a live concern at all.
            let forbid_candidates = size == 4 && identified.is_none() && self.battleship_cross_seen && !is_first_pick;

            // Scoring must skip both the cells already chosen this round AND
            // every cell reserved as "unsafe filler" above (see `avoid_as_filler`) —
            // combined here since `best_cell_for_size`/`best_cell_by_score`
            // only take a single exclusion list.
            let exclude: Vec<(usize, usize)> = chosen.iter().chain(avoid_as_filler.iter()).copied().collect();

            let next = if is_first_pick && size == 4 && identified.is_none() && self.battleship_cross_seen {
                // The first shot, while the Battleship's exact cell is still
                // unknown, is the one chance to test a coordinate that
                // actually distinguishes between the surviving candidate
                // windows — see `battleship_discriminating_test_cell`. Raw
                // alive-value scoring below would otherwise happily settle
                // for a coordinate common to every window (a guaranteed
                // hit, but one that teaches nothing about which window is
                // real), so try that first. If no such cell exists (e.g.
                // early on, before enough cross-4 salvos have narrowed
                // things down to well-formed windows at all), fall back to
                // the generic scorer — but restricted to the candidate
                // region already established by the cross-4 bag
                // (`require_candidates`), rather than letting the raw
                // per-cell score wander off to some untouched row or column
                // elsewhere that merely hasn't been narrowed by anything
                // yet and so looks deceptively "more alive". Digging
                // further into the region we already have real information
                // about is strictly more valuable than a blind guess
                // outside it.
                self.battleship_discriminating_test_cell().unwrap_or_else(|| {
                    self.best_cell_for_size(&row_line, &col_line, &exclude, forbid_candidates, true, allow_refired, size)
                })
            } else if hunt_with_cruiser_fsm && !is_first_pick {
                self.best_cell_for_size(&row3, &col3, &exclude, forbid_candidates, false, allow_refired, 3)
            } else {
                self.best_cell_for_size(&row_line, &col_line, &exclude, forbid_candidates, false, allow_refired, size)
            };

            Self::apply_hypothetical_miss(&mut row_line, &mut col_line, next.0, next.1, size);
            if hunt_with_cruiser_fsm {
                Self::apply_hypothetical_miss(&mut row3, &mut col3, next.0, next.1, 3);
            }
            chosen.push(next);
        }

        [chosen[0], chosen[1], chosen[2]]
    }

    /// Fallback shot selection once every ship size >=2 is fully sunk and only
    /// submarines remain. Submarines are single cells with no line-FSM notion
    /// of "alive placements", so this doesn't try to reuse that machinery — it
    /// just prefers cells still marked as viable submarine candidates (see
    /// `sub_candidates`), then fills any remaining slots with whatever unfired
    /// cells are left.
    fn choose_submarine_shots(&self, mut chosen: Vec<(usize, usize)>, avoid: &[(usize, usize)]) -> [(usize, usize); 3] {
        'candidates: for r in 0..10 {
            for c in 0..10 {
                if chosen.len() >= 3 {
                    break 'candidates;
                }
                if !self.fired[r][c] && self.sub_candidates[r][c] && !chosen.contains(&(r, c)) && !avoid.contains(&(r, c)) {
                    chosen.push((r, c));
                }
            }
        }

        'fallback: for r in 0..10 {
            for c in 0..10 {
                if chosen.len() >= 3 {
                    break 'fallback;
                }
                if !self.fired[r][c] && !chosen.contains(&(r, c)) && !avoid.contains(&(r, c)) {
                    chosen.push((r, c));
                }
            }
        }

        // Last-resort failsafe (should be unreachable in practice — `avoid`
        // is only ever a small handful of coordinates): if steering clear of
        // them left the salvo incomplete, fill remaining slots from them
        // anyway rather than leaving `chosen` short.
        'last_resort: for r in 0..10 {
            for c in 0..10 {
                if chosen.len() >= 3 {
                    break 'last_resort;
                }
                if !self.fired[r][c] && !chosen.contains(&(r, c)) {
                    chosen.push((r, c));
                }
            }
        }

        // Absolute last resort — genuinely unreachable in ordinary play (a
        // 100-cell board only ever needs 20 real hits to win, so this would
        // require nearly the entire board fired without winning first), but
        // if the board is ever THIS exhausted, refiring already-fired cells
        // is still better than panicking with an incomplete salvo. `Game::
        // fire` rejects a salvo containing the same cell twice, so these
        // must still be distinct from each other and from `chosen` so far.
        'absolute_last_resort: for r in 0..10 {
            for c in 0..10 {
                if chosen.len() >= 3 {
                    break 'absolute_last_resort;
                }
                if !chosen.contains(&(r, c)) {
                    chosen.push((r, c));
                }
            }
        }

        [chosen[0], chosen[1], chosen[2]]
    }

}
