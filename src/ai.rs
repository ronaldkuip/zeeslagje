use crate::fsm_tables::*;

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

/// One salvo whose result bag held at least one 3 (a Cruiser hit), together
/// with its derived "cross-3" bag: the union of a reach-2 cross around each of
/// the salvo's 3 coordinates — analogous to the Battleship's cross-4 bag, but
/// reach 2 instead of 3, since a Cruiser is only 3 cells long (so any two
/// cells of the *same* Cruiser are at most 2 apart along its line).
#[derive(Clone)]
pub struct Cross3Entry {
    pub coords: [(usize, usize); 3],
    pub values: [usize; 3],
    pub bag: Vec<(usize, usize)>,
    /// Per-coordinate flag, parallel to `coords`: true once that specific
    /// fired cell has been proven impossible to be a Cruiser cell (its alive
    /// value for size 3 has dropped to zero — or it's on the outer ring,
    /// which never holds a ship of size >=2) — i.e. this salvo's ambiguous
    /// "3" result can no longer have come from that particular cell.
    /// Recomputed at the end of every round by `refresh_cross3_entry_flags`.
    /// Monotonic: once true, always true, since alive values only ever
    /// shrink toward zero.
    pub coord_ruled_out: [bool; 3],
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
    /// `Cross3Entry::coord_ruled_out`. Currently always false (green): the
    /// rule for when a Battleship salvo's coordinate should flip red is not
    /// yet defined.
    pub coord_ruled_out: [bool; 3],
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
    /// Every 3-bearing salvo processed so far, in order, with its derived
    /// cross-3 bag. Kept as a running history rather than intersected together
    /// like the Battleship's single cross-4 bag — there are 2 Cruisers, so two
    /// different salvos might be hits on two entirely different ships, and
    /// blindly intersecting everything would wrongly narrow down to nothing.
    cross3_entries: Vec<Cross3Entry>,
    /// Once two entries above turn out to share zero cells, that's proof they're
    /// hits on two *different* Cruisers (a single Cruiser's cross-3 bag always
    /// contains any other hit belonging to the same ship). This becomes the
    /// union of that first disjoint pair — everywhere else on the board can
    /// then be ruled out for size 3.
    discovered_3_bag: Option<[[bool; 10]; 10]>,
    /// Cells already fed into the size-3 FSM as misses because they fell
    /// outside `discovered_3_bag`, so we don't redrive the same transition twice.
    discovered_3_processed: [[bool; 10]; 10],
    /// Cells proven to hold no ship at all: part of a salvo whose whole result
    /// bag was zero, so every one of its 3 cells is a guaranteed miss (unlike
    /// an ambiguous bound=3/4 salvo, there's no "which of the 3" uncertainty
    /// here). Used by `prune_discovered_3_bag` to strip out bag cells that
    /// can't possibly be a Cruiser after all.
    confirmed_miss: [[bool; 10]; 10],
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
    /// Cells already fed into the size-3 FSM as misses because both Cruisers
    /// are confirmed sunk and they fell outside the union of every cross-3
    /// salvo's raw coordinates, so we don't redrive the same transition
    /// twice. See `apply_full_cruiser_elimination`.
    cruiser_fully_sunk_processed: [[bool; 10]; 10],
    /// Straight-3 Cruiser layouts confirmed via `cruiser_combination_candidates`
    /// narrowing to exactly one surviving combination — recorded so each is
    /// only acted on (adjacency elimination) once, and so the debug UI can
    /// show them as "found" (green) rather than merely candidates.
    found_cruisers: Vec<[(usize, usize); 3]>,
    /// Cells already fed into the size-3/size-2 FSMs via a found Cruiser's
    /// "its neighbours can't hold another ship" deduction, so repeated calls
    /// don't redrive the same transition twice. See
    /// `apply_found_cruiser_adjacency_elimination`.
    found_cruiser_adjacency_processed: [[bool; 10]; 10],
    /// Set when `cruiser_combination_candidates` narrows to exactly 2
    /// possible layouts and a disambiguating coordinate (present in exactly
    /// one of the two) has been picked as the next salvo's first shot. `.0`
    /// is that coordinate; `.1` is the combo it belongs to (confirmed if the
    /// salvo that fires it comes back with a 3 in the bag); `.2` is the
    /// other combo (confirmed if it doesn't). See `choose_shots` and
    /// `refresh_cruiser_disambiguation`.
    pending_cruiser_disambiguation: Option<((usize, usize), [(usize, usize); 3], [(usize, usize); 3])>,
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
            battleship_cross_seen: false,
            battleship_cross_processed: [[false; 10]; 10],
            four_bearing_salvo_count: 0,
            cross4_entries: Vec::new(),
            battleship_adjacency_processed: [[false; 10]; 10],
            cross3_entries: Vec::new(),
            discovered_3_bag: None,
            discovered_3_processed: [[false; 10]; 10],
            confirmed_miss: [[false; 10]; 10],
            refire_allowed: [false; 5],
            freeze_before_frigates: false,
            cruiser_fully_sunk_processed: [[false; 10]; 10],
            found_cruisers: Vec::new(),
            found_cruiser_adjacency_processed: [[false; 10]; 10],
            pending_cruiser_disambiguation: None,
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
        // Row FSM: only meaningful if col is within inner range (a size>=2 ship lies
        // horizontally within row `row`, using inner columns). We drive the row's FSM
        // using `col` as the fired column, IF col is inner.
        if (INNER_LO..=INNER_HI).contains(&col) {
            let table_col = col - INNER_LO; // 0..7
            let cur = Self::line_state_for_size(&self.row_state[row], size);
            let next = match size {
                4 => TRANSITIONS_SIZE4[cur][table_col] as usize,
                3 => TRANSITIONS_SIZE3[cur][table_col] as usize,
                2 => TRANSITIONS_SIZE2[cur][table_col] as usize,
                _ => unreachable!(),
            };
            Self::set_line_state_for_size(&mut self.row_state[row], size, next);
        }
        // Column FSM: symmetric, using `row` as the fired row within column `col`.
        if (INNER_LO..=INNER_HI).contains(&row) {
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
    }

    /// Apply a miss at (row, col): eliminate all ship sizes >=2 through this cell
    /// (in both row and column FSMs) and remove it as a submarine candidate.
    fn apply_miss(&mut self, row: usize, col: usize) {
        for &size in &SHIP_SIZES {
            self.eliminate_size_at(row, col, size);
        }
        self.sub_candidates[row][col] = false;
        self.confirmed_miss[row][col] = true;
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
            // Once this call is the one that confirms both Cruisers sunk, every
            // real Cruiser cell is now guaranteed to be among the cross-3 salvos
            // seen so far — eliminate everywhere else immediately.
            if self.size_fully_found(3) {
                self.apply_full_cruiser_elimination();
            }
            // `Game::fire` calls `apply_salvo` (which also runs this same
            // check) BEFORE calling `mark_sunk` for whatever just sank — so
            // the exact salvo that sinks a Cruiser would otherwise see a
            // stale (pre-sinking) sunk count during its own apply_salvo call
            // and miss a combination that only becomes checkable once this
            // call's sunk_sizes bump has landed. Re-checking here closes
            // that one-round gap; the found_cruisers guard makes repeating
            // the check from both places harmless.
            self.check_and_apply_found_cruisers();
            self.refresh_cruiser_disambiguation();
        }
    }

    fn size_fully_found(&self, size: usize) -> bool {
        self.sunk_sizes[size] >= self.remaining_sizes[size]
    }

    /// Once both Cruisers are confirmed sunk, all 6 of their cells are
    /// guaranteed to have been hit at some point — and any salvo that hits a
    /// Cruiser cell reports a "3" in its result bag, which is exactly what
    /// makes `apply_cruiser_cross_tracking` record that salvo's 3 raw fired
    /// coordinates as a `Cross3Entry` (see there). So the union of every
    /// cross-3 salvo's coordinates seen over the whole game is guaranteed to
    /// contain all 6 real Cruiser cells, with total certainty — no more
    /// Cruisers exist anywhere else, so every OTHER inner cell can be
    /// eliminated for size 3 outright. This is strictly stronger than the
    /// discovered-3 bag (a geometric heuristic that can narrow things down
    /// earlier, before both ships are sunk, but only ever proves a superset
    /// of the truth) — once this fires, it settles size 3 completely.
    fn apply_full_cruiser_elimination(&mut self) {
        let mut candidate = [[false; 10]; 10];
        for entry in &self.cross3_entries {
            for &(r, c) in &entry.coords {
                candidate[r][c] = true;
            }
        }
        for row in INNER_LO..=INNER_HI {
            for col in INNER_LO..=INNER_HI {
                if !candidate[row][col] && !self.cruiser_fully_sunk_processed[row][col] {
                    self.cruiser_fully_sunk_processed[row][col] = true;
                    self.eliminate_size_at(row, col, 3);
                }
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

        // A 4 in the bag means one of these 3 cells is a genuine Battleship hit —
        // we just don't know which. Fold in the cross-elimination trick below.
        if bound == 4 {
            self.apply_battleship_cross_elimination(coords, values);
        } else {
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

        // Refresh flags now (not just at the end of the round) so the
        // combination search below sees anything this salvo already proved
        // impossible — a stale (too-green) view could only under-narrow the
        // search, never misidentify a wrong combination as the unique one
        // (a real Cruiser's true hit is never flagged red), so this is safe
        // even though the full end-of-round refresh runs again below.
        self.refresh_cross3_entry_flags();

        // If this salvo is the deliberate disambiguation shot `choose_shots`
        // queued up (see `pending_cruiser_disambiguation`), interpret its
        // result: the other 2 coordinates in that salvo were chosen to be
        // already-proven-impossible for a Cruiser, so a 3 anywhere in this
        // bag can only have come from the disambiguating coordinate itself —
        // confirming whichever of the 2 layouts contains it. No 3 confirms
        // the other layout instead (and would also already be settled by the
        // normal apply_hit/apply_miss elimination above, since a bag with no
        // 3 means bound < 3, which eliminates size 3 at every fired cell
        // including this one via the ordinary path — but resolving it
        // explicitly here is simpler than relying on that side effect).
        if let Some((coord, combo_if_hit, combo_if_miss)) = self.pending_cruiser_disambiguation {
            if coords.contains(&coord) {
                let confirmed = if values.contains(&3) { combo_if_hit } else { combo_if_miss };
                if !self.found_cruisers.contains(&confirmed) {
                    self.found_cruisers.push(confirmed);
                    self.apply_found_cruiser_adjacency_elimination(confirmed);
                }
                self.pending_cruiser_disambiguation = None;
            }
        }

        // See `check_and_apply_found_cruisers` — also re-run from `mark_sunk`
        // to close a one-round gap around the exact salvo that sinks a ship.
        self.check_and_apply_found_cruisers();

        // Every salvo can prove cells impossible for a Cruiser that a stored
        // cross-3 bag doesn't know about yet (an ordinary miss elsewhere, the
        // Battleship's own elimination above, etc.) — sweep those out, then
        // re-check for a disjoint pair, since pruning can reveal one just as
        // well as adding a new entry can.
        self.prune_cross3_bags();
        if self.discovered_3_bag.is_none() {
            self.recheck_cross3_disjoint_pairs();
        }
        self.prune_discovered_3_bag();
        if let Some(discovered) = self.discovered_3_bag {
            self.apply_discovered_3_elimination(discovered);
        }

        // End-of-round check: every cross-3 entry's 3 original fired
        // coordinates may have had their alive value for size 3 driven to
        // zero by anything above (or by an unrelated salvo elsewhere) — flag
        // any that can no longer possibly be the real Cruiser hit.
        self.refresh_cross3_entry_flags();

        // Now that everything above has settled, recompute the disambiguation
        // target for next round (using the fully up-to-date combination
        // search) — a no-op if nothing is ambiguous, or if a target is still
        // pending and hasn't been fired yet (find_cruiser_disambiguation is
        // deterministic given the same combos, so this just reselects the
        // same one).
        self.refresh_cruiser_disambiguation();
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

    /// A found Cruiser (see `cruiser_combination_candidates`) forbids
    /// orthogonal *and* diagonal adjacency to any other ship, exactly like
    /// the Battleship (see `apply_battleship_adjacency_elimination`, which
    /// this mirrors) — once its 3 cells are confirmed, every touching cell
    /// rules out a Cruiser or Frigate, and the 3 cells themselves rule out
    /// everything else (a board cell can only ever hold one ship). Submarines
    /// are left alone at neighbouring cells for the same reason as there:
    /// they only forbid orthogonal adjacency, so a diagonal neighbour could
    /// still legitimately hold one.
    ///
    /// Eliminates unconditionally, even for already-fired cells, for the same
    /// ambiguous-salvo reason `apply_battleship_adjacency_elimination` does —
    /// safe to call repeatedly since the FSM tables are idempotent per column
    /// and `found_cruiser_adjacency_processed` avoids the redundant work.
    fn apply_found_cruiser_adjacency_elimination(&mut self, ship_cells: [(usize, usize); 3]) {
        for &(row, col) in &ship_cells {
            if self.found_cruiser_adjacency_processed[row][col] {
                continue;
            }
            self.found_cruiser_adjacency_processed[row][col] = true;
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
                        continue; // part of the found Cruiser itself, not a neighbour
                    }
                    if self.found_cruiser_adjacency_processed[nr][nc] {
                        continue;
                    }
                    self.found_cruiser_adjacency_processed[nr][nc] = true;
                    self.eliminate_size_at(nr, nc, 3);
                    self.eliminate_size_at(nr, nc, 2);
                }
            }
        }
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

    /// Given a salvo whose result bag contained a 3, build its cross-3 bag (union
    /// of a reach-2 cross around each of the 3 fired coordinates — the true
    /// Cruiser hit is one of them, we just don't know which) and record the
    /// entry. Cells already proven impossible for a Cruiser (see
    /// `alive_value`) are left out of the bag from the start — no point
    /// recording a candidate that's already known dead.
    ///
    /// Disjointness against every other entry is (re)checked separately by
    /// `recheck_cross3_disjoint_pairs`, called once per salvo after both this
    /// and `prune_cross3_bags` have run — see there for why a single combined
    /// check is more robust than checking only right after this one entry.
    fn apply_cruiser_cross_tracking(&mut self, coords: [(usize, usize); 3], values: [usize; 3]) {
        let mut bag_mask = [[false; 10]; 10];
        for &(r, c) in &coords {
            Self::mark_cross_reach(&mut bag_mask, r, c, 2);
        }

        let mut bag_cells = Vec::new();
        for row in INNER_LO..=INNER_HI {
            for col in INNER_LO..=INNER_HI {
                if bag_mask[row][col] && self.alive_value(row, col, 3) > 0 {
                    bag_cells.push((row, col));
                }
            }
        }
        self.cross3_entries.push(Cross3Entry {
            coords,
            values,
            bag: bag_cells,
            coord_ruled_out: [false; 3],
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
    /// what `prune_cross3_bags` uses to decide a cell is dead for size 3.
    fn alive_value(&self, row: usize, col: usize, size: usize) -> u32 {
        let horizontal = Self::line_state_score(Self::line_state_for_size(&self.row_state[row], size), size, col - INNER_LO);
        let vertical = Self::line_state_score(Self::line_state_for_size(&self.col_state[col], size), size, row - INNER_LO);
        horizontal + vertical
    }

    /// The 3 debug grids for `size` (4, 3, or 2): horizontal alive value,
    /// vertical alive value, and their sum, one entry per inner cell (8x8,
    /// indexed 0..8 for board rows/cols 1..8). For size 3 the combined grid is
    /// exactly the criterion `prune_cross3_bags` uses.
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

    /// Strip any cell now proven impossible for a Cruiser out of every stored
    /// cross-3 bag — a cell is dead once `alive_value(.., 3)` is zero,
    /// regardless of whether it was ever individually fired. Bags are built
    /// once, from raw geometry, when their salvo is processed — a *later*
    /// salvo (an ordinary miss elsewhere, the Battleship being identified and
    /// ruling out its neighbours, or the discovered-3 bag ruling out
    /// everything outside it) can prove some of those cells impossible after
    /// the fact — including cells that were never themselves fired, if enough
    /// of their row or column has been eliminated that no placement can
    /// possibly reach them any more. Without this, stale cells would linger
    /// forever, which could make two bags that are ACTUALLY disjoint — once
    /// you account for everything we now know — still look like they overlap.
    fn prune_cross3_bags(&mut self) {
        let mut alive = [[0u32; 10]; 10];
        for row in INNER_LO..=INNER_HI {
            for col in INNER_LO..=INNER_HI {
                alive[row][col] = self.alive_value(row, col, 3);
            }
        }
        for entry in &mut self.cross3_entries {
            entry.bag.retain(|&(r, c)| alive[r][c] > 0);
        }
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
        for (entry, entry_flags) in self.cross3_entries.iter_mut().zip(flags) {
            entry.coord_ruled_out = entry_flags;
        }
    }

    /// Check every pair of cross-3 entries for disjointness (see
    /// `apply_cruiser_cross_tracking`'s doc comment for why disjoint proves two
    /// different Cruisers). Re-derives from scratch every time it's called
    /// rather than only checking the newest entry, because pruning stale cells
    /// out of *existing* bags — not just adding a new one — can just as easily
    /// be what makes a pair newly disjoint.
    fn recheck_cross3_disjoint_pairs(&mut self) {
        for i in 0..self.cross3_entries.len() {
            for j in (i + 1)..self.cross3_entries.len() {
                let disjoint = self.cross3_entries[i]
                    .bag
                    .iter()
                    .all(|cell| !self.cross3_entries[j].bag.contains(cell));
                if disjoint {
                    let mut union = [[false; 10]; 10];
                    for &(r, c) in &self.cross3_entries[i].bag {
                        union[r][c] = true;
                    }
                    for &(r, c) in &self.cross3_entries[j].bag {
                        union[r][c] = true;
                    }
                    self.discovered_3_bag = Some(union);
                    return;
                }
            }
        }
    }

    /// Every cell outside the discovered-3 bag can't hold either Cruiser — feed
    /// it into the size-3 FSM the same way a real miss would be, guarded by
    /// `discovered_3_processed` so this never redrives the same transition twice.
    ///
    /// Eliminates unconditionally, even for already-fired cells: a salvo whose
    /// bound is 3 or 4 never eliminates size 3 for any of its 3 cells via the
    /// normal apply_hit path (any of them could ambiguously be the real hit) —
    /// see `apply_salvo`. So a decoy cell fired as part of such an ambiguous
    /// salvo would otherwise never get size 3 eliminated at all, even once it's
    /// proven to fall outside the discovered-3 region. Safe to call more than
    /// once on the same cell — the FSM transition tables are idempotent per
    /// column.
    fn apply_discovered_3_elimination(&mut self, discovered: [[bool; 10]; 10]) {
        for row in INNER_LO..=INNER_HI {
            for col in INNER_LO..=INNER_HI {
                if !discovered[row][col] && !self.discovered_3_processed[row][col] {
                    self.discovered_3_processed[row][col] = true;
                    self.eliminate_size_at(row, col, 3);
                }
            }
        }
    }

    /// Tighten the discovered-3 bag itself, removing cells that can no longer
    /// actually be a Cruiser cell even though they're still inside the union:
    ///
    /// - it's since been proven to hold the Battleship instead (a board cell
    ///   can only ever hold one ship, so a confirmed Battleship cell is
    ///   definitely not a Cruiser cell);
    /// - it's a confirmed miss (part of an all-zero salvo, so it's plain
    ///   water, not part of *either* ship);
    /// - it can't physically fit a straight run of 3 within the bag itself —
    ///   every real Cruiser cell's whole 3-length run must lie entirely
    ///   inside the bag (see `apply_discovered_3_elimination`'s doc comment:
    ///   everywhere outside is already ruled out), so a bag cell with no
    ///   run of >=3 through it, horizontal or vertical, measured against the
    ///   bag mask itself, is a contradiction.
    ///
    /// Re-scans to a fixed point, since shrinking the mask for one reason can
    /// break the room another cell was relying on. Cells dropped here are
    /// picked up by the next `apply_discovered_3_elimination` call the same
    /// way any other newly-outside cell would be.
    fn prune_discovered_3_bag(&mut self) {
        let Some(mut discovered) = self.discovered_3_bag else {
            return;
        };
        let battleship_cells = self.battleship_identified();
        loop {
            let mut removed = false;
            for row in INNER_LO..=INNER_HI {
                for col in INNER_LO..=INNER_HI {
                    if !discovered[row][col] {
                        continue;
                    }
                    let is_battleship_cell =
                        battleship_cells.is_some_and(|cells| cells.contains(&(row, col)));
                    let is_confirmed_miss = self.confirmed_miss[row][col];
                    // The direct, most general check: has ANYTHING (a real
                    // hit whose bound is small enough to eliminate size 3 at
                    // its own cell, e.g. a bound=2 Frigate hit; the Battleship
                    // adjacency sweep; an unrelated row/column narrowing —
                    // anything at all) already driven this cell's combined
                    // alive value to zero? The other three checks below each
                    // catch one *specific* route to that same conclusion, but
                    // this is the ground truth the FSM itself tracks, and the
                    // only one of the four that catches every route.
                    let is_dead_for_size3 = self.alive_value(row, col, 3) == 0;
                    let h = Self::max_contiguous_run_horizontal(&discovered, row, col);
                    let v = Self::max_contiguous_run_vertical(&discovered, row, col);
                    let lacks_room = h < 3 && v < 3;
                    if is_battleship_cell || is_confirmed_miss || is_dead_for_size3 || lacks_room {
                        discovered[row][col] = false;
                        removed = true;
                    }
                }
            }
            if !removed {
                break;
            }
        }
        self.discovered_3_bag = Some(discovered);
    }

    /// Cells in the discovered-3 bag (see `apply_cruiser_cross_tracking`) — empty
    /// until two mutually-disjoint cross-3 bags have been found.
    pub fn discovered_3_cells(&self) -> Vec<(usize, usize)> {
        match &self.discovered_3_bag {
            Some(mask) => {
                let mut cells = Vec::new();
                for row in INNER_LO..=INNER_HI {
                    for col in INNER_LO..=INNER_HI {
                        if mask[row][col] {
                            cells.push((row, col));
                        }
                    }
                }
                cells
            }
            None => Vec::new(),
        }
    }

    /// Every 3-bearing salvo processed so far, in order, with its derived
    /// cross-3 bag. Exposed for the debug/inspector UI.
    pub fn cross3_entries(&self) -> &[Cross3Entry] {
        &self.cross3_entries
    }

    /// If exactly one straight-3 line survives `cruiser_combination_candidates`,
    /// that's provably a sunk Cruiser's true layout — the real combination is
    /// always among the candidates, so if it's the only survivor, it must be
    /// it. Records it (once — `found_cruisers` de-dupes) and treats it
    /// exactly like a found Battleship: its own cells and every neighbour
    /// rule out a Cruiser or Frigate. Called from both `apply_salvo` (so
    /// later salvos that organically complete the picture are caught) and
    /// `mark_sunk` (to catch it immediately on the exact salvo that sinks a
    /// ship, before `apply_salvo`'s own view of the sunk count goes stale).
    fn check_and_apply_found_cruisers(&mut self) {
        let found = self.cruiser_combination_candidates();
        if found.len() == 1 && !self.found_cruisers.contains(&found[0]) {
            self.found_cruisers.push(found[0]);
            self.apply_found_cruiser_adjacency_elimination(found[0]);
        }
    }

    /// When `cruiser_combination_candidates` narrows to exactly 2 possible
    /// layouts, find a coordinate that belongs to only one of them (the two
    /// combos are never identical, so at least one such cell always exists).
    /// Firing it — alongside 2 cells already proven impossible to be a
    /// Cruiser, so any "3" in the result can only be attributed to this one
    /// — settles which of the two layouts is real: a 3 confirms the combo
    /// containing it, no 3 confirms the other one (see `apply_salvo`'s
    /// handling of `pending_cruiser_disambiguation`). Returns `None` unless
    /// exactly 2 combos currently survive.
    fn find_cruiser_disambiguation(
        &self,
    ) -> Option<((usize, usize), [(usize, usize); 3], [(usize, usize); 3])> {
        let combos = self.cruiser_combination_candidates();
        if combos.len() != 2 {
            return None;
        }
        let (a, b) = (combos[0], combos[1]);
        for &cell in &a {
            if !b.contains(&cell) {
                return Some((cell, a, b));
            }
        }
        for &cell in &b {
            if !a.contains(&cell) {
                return Some((cell, b, a));
            }
        }
        None // unreachable if a != b, which cruiser_combination_candidates guarantees
    }

    /// Recompute `pending_cruiser_disambiguation` from the current
    /// combination search. Called at the end of `apply_salvo` and from
    /// `mark_sunk`, mirroring `check_and_apply_found_cruisers`'s two call
    /// sites for the same reason (catching the exact round something
    /// changes, not just eventually on some later round).
    fn refresh_cruiser_disambiguation(&mut self) {
        self.pending_cruiser_disambiguation = self.find_cruiser_disambiguation();
    }

    /// True if (row, col) is the coordinate `choose_shots` deliberately fired
    /// to disambiguate between 2 possible Cruiser layouts — used by
    /// `Game::fire` to let that one specific refire through even when the
    /// general refire-allowed toggle is off, since this is a deliberate
    /// internal strategy, not the debug/experimentation relaxation.
    pub fn is_pending_cruiser_disambiguation(&self, row: usize, col: usize) -> bool {
        self.pending_cruiser_disambiguation
            .is_some_and(|(coord, _, _)| coord == (row, col))
    }

    /// Once at least one Cruiser is sunk, there's an extra deduction
    /// available regardless of how many cross-3 salvos have piled up by
    /// then (could be exactly 3, could be 4, 5, or more if some salvos'
    /// decoys never panned out): every sunk Cruiser's 3 real cells must each
    /// have shown up as the (unknown-which-one) hit in some cross-3 salvo,
    /// and — critically — no two of those 3 real cells can ever have come
    /// from the SAME salvo entry, since each entry is one salvo's worth of
    /// ambiguity contributing at most one real Cruiser hit to reason about
    /// here (two real hits landing in the same salvo would show as two
    /// separate 3s in that one salvo's bag, which is a different, simpler
    /// case not handled by this combinatorial search).
    ///
    /// So: pick any 3 DISTINCT cross-3 entries (every possible such triple,
    /// not just "the first 3"), take one still-possible (non-red-flagged —
    /// see `Cross3Entry::coord_ruled_out`) coordinate from each of the 3
    /// chosen entries, and keep the combination only if those 3 coordinates
    /// form a valid straight, contiguous 3-cell line. Every survivor is a
    /// candidate for what some sunk Cruiser's real layout could be. Purely a
    /// reporting/debug aid for now — returns every surviving combination
    /// (deduplicated), doesn't eliminate anything on its own. Empty unless
    /// at least one Cruiser is sunk.
    pub fn cruiser_combination_candidates(&self) -> Vec<[(usize, usize); 3]> {
        if self.sunk_sizes[3] == 0 {
            return Vec::new();
        }

        let candidates: Vec<Vec<(usize, usize)>> = self
            .cross3_entries
            .iter()
            .map(|entry| {
                entry
                    .coords
                    .iter()
                    .zip(entry.coord_ruled_out.iter())
                    .filter(|(_, &ruled_out)| !ruled_out)
                    .map(|(&c, _)| c)
                    .collect()
            })
            .collect();

        let n = candidates.len();
        let mut combinations: Vec<[(usize, usize); 3]> = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                for k in (j + 1)..n {
                    for &a in &candidates[i] {
                        for &b in &candidates[j] {
                            for &c in &candidates[k] {
                                let mut combo = [a, b, c];
                                if !Self::is_straight_run_of_3(&combo) {
                                    continue;
                                }
                                combo.sort();
                                if !combinations.contains(&combo) {
                                    combinations.push(combo);
                                }
                            }
                        }
                    }
                }
            }
        }
        combinations
    }

    /// Every Cruiser layout confirmed so far via `cruiser_combination_candidates`
    /// narrowing to exactly one surviving combination — for the debug UI to
    /// render as "found" (e.g. coloured green), distinct from the merely
    /// still-possible combinations reported elsewhere.
    pub fn found_cruisers(&self) -> &[[(usize, usize); 3]] {
        &self.found_cruisers
    }

    /// True if the 3 given cells, regardless of order, form a contiguous
    /// straight line of exactly 3 cells — either horizontal (same row,
    /// consecutive columns) or vertical (same column, consecutive rows).
    fn is_straight_run_of_3(cells: &[(usize, usize); 3]) -> bool {
        let mut sorted = *cells;
        sorted.sort();
        if sorted[0].0 == sorted[1].0 && sorted[1].0 == sorted[2].0 {
            return sorted[1].1 == sorted[0].1 + 1 && sorted[2].1 == sorted[1].1 + 1;
        }
        if sorted[0].1 == sorted[1].1 && sorted[1].1 == sorted[2].1 {
            return sorted[1].0 == sorted[0].0 + 1 && sorted[2].0 == sorted[1].0 + 1;
        }
        false
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

        let is_possible = |r: usize, c: usize| self.battleship_candidates[r][c];

        let mut found: Option<[(usize, usize); 4]> = None;
        let mut count = 0usize;

        // Horizontal placements: every row, every starting column that keeps all
        // 4 cells within the inner 1..=8 range.
        for row in INNER_LO..=INNER_HI {
            for start in INNER_LO..=(INNER_HI - 3) {
                let cells = [(row, start), (row, start + 1), (row, start + 2), (row, start + 3)];
                if cells.iter().all(|&(r, c)| is_possible(r, c)) {
                    count += 1;
                    found = Some(cells);
                }
            }
        }
        // Vertical placements: symmetric, varying row instead of column.
        for col in INNER_LO..=INNER_HI {
            for start in INNER_LO..=(INNER_HI - 3) {
                let cells = [(start, col), (start + 1, col), (start + 2, col), (start + 3, col)];
                if cells.iter().all(|&(r, c)| is_possible(r, c)) {
                    count += 1;
                    found = Some(cells);
                }
            }
        }

        if count == 1 {
            found
        } else {
            None
        }
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

        for &size in &SHIP_SIZES {
            if self.size_fully_found(size) {
                continue;
            }
            if (INNER_LO..=INNER_HI).contains(&col) {
                let table_col = col - INNER_LO;
                let cur = Self::line_state_for_size(&self.row_state[row], size);
                let v = match size {
                    4 => VALUES_SIZE4[cur][table_col],
                    3 => VALUES_SIZE3[cur][table_col],
                    2 => VALUES_SIZE2[cur][table_col],
                    _ => unreachable!(),
                };
                score += v as u32;
            }
            if (INNER_LO..=INNER_HI).contains(&row) {
                let table_row = row - INNER_LO;
                let cur = Self::line_state_for_size(&self.col_state[col], size);
                let v = match size {
                    4 => VALUES_SIZE4[cur][table_row],
                    3 => VALUES_SIZE3[cur][table_row],
                    2 => VALUES_SIZE2[cur][table_row],
                    _ => unreachable!(),
                };
                score += v as u32;
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
        let mut score = 0u32;
        if (INNER_LO..=INNER_HI).contains(&col) {
            let table_col = col - INNER_LO;
            score += Self::line_state_score(row_line[row], size, table_col);
        }
        if (INNER_LO..=INNER_HI).contains(&row) {
            let table_row = row - INNER_LO;
            score += Self::line_state_score(col_line[col], size, table_row);
        }
        score
    }

    /// Fold a *hypothetical* miss at (row, col) into a working copy of `size`'s
    /// row/col FSM states — i.e. "if this shot comes back as a miss, what would
    /// that size's FSM look like afterwards". Mirrors `eliminate_size_at`, but
    /// operates on local scratch state rather than `self`.
    fn apply_hypothetical_miss(row_line: &mut [usize; 10], col_line: &mut [usize; 10], row: usize, col: usize, size: usize) {
        if (INNER_LO..=INNER_HI).contains(&col) {
            let table_col = col - INNER_LO;
            row_line[row] = Self::line_state_transition(row_line[row], size, table_col);
        }
        if (INNER_LO..=INNER_HI).contains(&row) {
            let table_row = row - INNER_LO;
            col_line[col] = Self::line_state_transition(col_line[col], size, table_row);
        }
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
    fn best_cell_by_score(
        &self,
        chosen_so_far: &[(usize, usize)],
        forbid_candidates: bool,
        allow_refired: bool,
        score_fn: impl Fn(usize, usize) -> u32,
    ) -> (usize, usize) {
        let search = |forbid: bool| {
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
                    let score = score_fn(r, c) as i64;
                    if score > best_score {
                        best_score = score;
                        best_cell = Some((r, c));
                    }
                }
            }
            best_cell
        };

        if let Some(cell) = search(forbid_candidates) {
            return cell;
        }

        if forbid_candidates {
            // No eligible non-candidate cell left — better to use up our one
            // candidate-region allowance than to fail outright.
            if let Some(cell) = search(false) {
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
        allow_refired: bool,
        size: usize,
    ) -> (usize, usize) {
        self.best_cell_by_score(chosen_so_far, forbid_candidates, allow_refired, |r, c| {
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
    /// candidate region. Reasoning: if two or more candidate cells were fired in the same
    /// ambiguous salvo, a 4 in the result bag still wouldn't tell us *which* of them was
    /// the hit — we'd be back to cross-intersecting again. Restricting the candidate
    /// region to that first shot means a 4 can only have come from it, collapsing the
    /// ambiguity immediately instead of costing another round of deduction.
    ///
    /// The other 2 shots are forced away from the candidate region regardless — they
    /// can't help test it further this salvo — so rather than spend them purely hunting
    /// the Battleship's line, they're scored against size-4 *and* size-3 (Cruiser)
    /// combined. A shot that can't land near the Battleship anyway might as well also
    /// make progress on the Cruisers instead of "wasting" its elimination value on a ship
    /// whose region is already staked out.
    ///
    /// If `battleship_identified` has pinned down the exact 4-cell layout, none of that
    /// applies any more — there's nothing left to protect or blend. Instead, this fills
    /// as many of the 3 shots as possible with that placement's still-unfired cells first
    /// (finishing the ship off directly), then falls back to the ordinary FSM search for
    /// any slots left over.
    pub fn choose_shots(&self) -> [(usize, usize); 3] {
        // Resolving which of 2 possible Cruiser layouts is real is valuable
        // enough to take priority over whatever size is currently being
        // hunted — see `pending_cruiser_disambiguation`.
        if let Some((coord, _, _)) = self.pending_cruiser_disambiguation {
            return self.choose_disambiguation_shots(coord);
        }

        let size = self.current_target_size();

        // Once every size >=2 ship is sunk, only submarines are left — the
        // line-FSM tables above only cover sizes 4/3/2, so this must branch off
        // before touching them (line_state_for_size panics on anything else).
        if size == 1 {
            return self.choose_submarine_shots();
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
        let mut chosen: Vec<(usize, usize)> = Vec::with_capacity(3);
        if let Some(cells) = identified {
            for (r, c) in cells {
                if chosen.len() >= 3 {
                    break;
                }
                if !self.fired[r][c] {
                    chosen.push((r, c));
                }
            }
        }

        // Still hunting the Battleship, hit it at least once, but haven't pinned
        // down its exact cell: blend size-4 and size-3 scoring for every shot
        // after the first (see doc comment above).
        let blend_with_size3 = size == 4 && identified.is_none() && self.battleship_cross_seen;
        let mut row3: [usize; 10] = [0; 10];
        let mut col3: [usize; 10] = [0; 10];
        if blend_with_size3 {
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

            let next = if blend_with_size3 && !is_first_pick {
                self.best_cell_by_score(&chosen, forbid_candidates, allow_refired, |r, c| {
                    Self::size_cell_score(&row_line, &col_line, r, c, 4)
                        + Self::size_cell_score(&row3, &col3, r, c, 3)
                })
            } else {
                self.best_cell_for_size(&row_line, &col_line, &chosen, forbid_candidates, allow_refired, size)
            };

            Self::apply_hypothetical_miss(&mut row_line, &mut col_line, next.0, next.1, size);
            if blend_with_size3 {
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
    fn choose_submarine_shots(&self) -> [(usize, usize); 3] {
        let mut chosen: Vec<(usize, usize)> = Vec::with_capacity(3);

        'candidates: for r in 0..10 {
            for c in 0..10 {
                if chosen.len() >= 3 {
                    break 'candidates;
                }
                if !self.fired[r][c] && self.sub_candidates[r][c] {
                    chosen.push((r, c));
                }
            }
        }

        'fallback: for r in 0..10 {
            for c in 0..10 {
                if chosen.len() >= 3 {
                    break 'fallback;
                }
                if !self.fired[r][c] && !chosen.contains(&(r, c)) {
                    chosen.push((r, c));
                }
            }
        }

        [chosen[0], chosen[1], chosen[2]]
    }

    /// Builds the salvo that resolves a pending Cruiser-layout ambiguity:
    /// `target` (the coordinate present in only one of the 2 candidate
    /// layouts) goes first — deliberately re-fired regardless of whether
    /// it's already been fired before, since this is a specific internal
    /// strategy call, not the general refire-allowed relaxation (see
    /// `Game::fire`'s `is_pending_cruiser_disambiguation` check). The other
    /// 2 slots are filled with cells already proven impossible to be a
    /// Cruiser (alive value 0, or outer ring, which never holds one either
    /// way) — chosen so a 3 anywhere in the resulting bag can only have come
    /// from `target` itself, making the result unambiguous.
    fn choose_disambiguation_shots(&self, target: (usize, usize)) -> [(usize, usize); 3] {
        let mut chosen: Vec<(usize, usize)> = vec![target];

        'safe_unfired: for r in 0..10 {
            for c in 0..10 {
                if chosen.len() >= 3 {
                    break 'safe_unfired;
                }
                if (r, c) == target || self.fired[r][c] || chosen.contains(&(r, c)) {
                    continue;
                }
                let is_safe_for_cruiser = if (INNER_LO..=INNER_HI).contains(&r) && (INNER_LO..=INNER_HI).contains(&c) {
                    self.alive_value(r, c, 3) == 0
                } else {
                    true // outer ring never holds a Cruiser
                };
                if is_safe_for_cruiser {
                    chosen.push((r, c));
                }
            }
        }

        // Fallback (should be unreachable in practice — the outer ring alone
        // is 36 cells, always safe): fill any remaining slot with whatever
        // unfired cell is left, rather than failing to complete the salvo.
        if chosen.len() < 3 {
            for r in 0..10 {
                for c in 0..10 {
                    if chosen.len() >= 3 {
                        break;
                    }
                    if (r, c) != target && !self.fired[r][c] && !chosen.contains(&(r, c)) {
                        chosen.push((r, c));
                    }
                }
            }
        }

        [chosen[0], chosen[1], chosen[2]]
    }
}
