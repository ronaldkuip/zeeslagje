use crate::fsm_tables::*;
use std::cell::RefCell;

mod battleship;
mod fsm;
mod heatmap_gen;
mod heatmap_ops;
mod hypothetical;

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
pub(crate) struct LineState {
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
    /// Memoized `jointly_consistent_hypothesis_pairs`, same `salvo_history.
    /// len()` key as `cruiser_candidates_cache`/`frigate_candidates_cache`.
    /// This one was the actual gap: every caller of the cross-reasoned
    /// lists (`cross_reasoned_cruiser_candidates`, `cross_reasoned_frigate_
    /// candidates` — in turn used by the refined heatmaps, the refined
    /// identified-cell checks, and `resolution_status_json`) independently
    /// recomputed the full O(cruiser_len * frigate_len) pairwise adjacency
    /// check from scratch, even within a single call (the cruiser-view and
    /// frigate-view calls redo the exact same pairs list). Cleared
    /// alongside the other two at both invalidation points in
    /// `update_fsm_and_resolve` — those clear on a lock-in event that
    /// doesn't change `salvo_history.len()`, so keying alone isn't enough.
    joint_pairs_cache: RefCell<Option<(usize, Vec<(std::collections::HashSet<(usize, usize)>, std::collections::HashSet<(usize, usize)>)>)>>,
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
            joint_pairs_cache: RefCell::new(None),
        }
    }

    /// Clears all 3 memoized candidate-list caches at once. Replaces what
    /// used to be 3 separate raw-field writes duplicated at 2 call sites
    /// inside `fsm::update_fsm_and_resolve` — same effect, just named and
    /// centralized here next to the fields themselves, so a bucket module
    /// invalidating the caches doesn't need to know their exact field
    /// names. See the refactor plan's cross-dependency item #4.
    pub(crate) fn invalidate_candidate_caches(&mut self) {
        *self.cruiser_candidates_cache.borrow_mut() = None;
        *self.frigate_candidates_cache.borrow_mut() = None;
        *self.joint_pairs_cache.borrow_mut() = None;
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























































    /// Every 3-bearing salvo processed so far, in order. Exposed for the
    /// debug/inspector UI.
    pub fn cross3_entries(&self) -> &[Cross3Entry] {
        &self.cross3_entries
    }




    /// Every 2-bearing salvo processed so far, in order. Exposed for the
    /// debug/inspector UI.
    pub fn cross2_entries(&self) -> &[Cross2Entry] {
        &self.cross2_entries
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
