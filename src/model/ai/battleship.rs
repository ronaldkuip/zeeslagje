//! "Other" bucket: the self-contained Battleship cross-4
//! candidate-mask deduction subsystem. FSM-like in spirit (drives
//! eliminate_size_at, same as the fsm-operations bucket) but operates on
//! its own `battleship_candidates: [[bool; 10]; 10]` grid rather than the
//! row/col LineState FSM, with no probabilistic-heatmap analogue the way
//! Cruiser/Frigate have — kept as its own file rather than folded into
//! fsm.rs for that reason. Extracted from the old ai.rs verbatim (Stage 3
//! of the refactor plan); a second `impl AiPlayer` block in a sibling
//! module of where `AiPlayer` is defined, so it keeps the exact same
//! field/method access an ordinary impl block would have — no call-site
//! or method-body changes anywhere else in the crate were needed for
//! this move.

use super::*;

impl AiPlayer {
    /// Every inner cell starts "possible"; the outer ring never is, regardless of
    /// any cross deduction, since the Battleship can never occupy it.
    pub(crate) fn initial_battleship_candidates() -> [[bool; 10]; 10] {
        let mut grid = [[false; 10]; 10];
        for row in INNER_LO..=INNER_HI {
            for col in INNER_LO..=INNER_HI {
                grid[row][col] = true;
            }
        }
        grid
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
    pub(crate) fn apply_full_battleship_elimination(&mut self) {
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
    pub(crate) fn apply_battleship_cross_elimination(&mut self, coords: [(usize, usize); 3], values: [usize; 3]) {
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
    /// Once at least one cell is confirmed a genuine Battleship hit (see
    /// `derive_confirmed_battleship_hits_by_elimination`), the real ship
    /// must be exactly one of the straight-4 windows passing through EVERY
    /// confirmed cell — any candidate that isn't part of at least one such
    /// window can be eliminated outright, the same way
    /// `apply_battleship_cross_elimination`'s union-of-crosses trick
    /// already eliminates candidates outside the running cross
    /// intersection.
    pub(crate) fn prune_candidates_not_through_confirmed(&mut self) {
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
    pub(crate) fn battleship_identified(&self) -> Option<[(usize, usize); 4]> {
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
    pub(crate) fn battleship_discriminating_test_cell(&self) -> Option<(usize, usize)> {
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
    pub(crate) fn prune_candidates_without_room(&mut self) {
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
}
