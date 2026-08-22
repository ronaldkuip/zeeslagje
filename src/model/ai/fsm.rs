//! "Fsm operations" (bucket A): the core per-row/per-column line-state
//! machine — transitions, hit/miss entry points, confirmed-hit-by-
//! elimination derivation (Cross-3/Cross-2/Cross-4 bag arithmetic),
//! adjacency elimination, and the alive-value/alive-count readouts
//! everything else (heatmaps, shot scoring) is built on. Extracted from
//! the old ai.rs verbatim (Stage 4 of the refactor plan); a second
//! `impl AiPlayer` block in a sibling module of where `AiPlayer` is
//! defined, so it keeps the exact same field/method access an ordinary
//! impl block would have.

use super::*;

impl AiPlayer {
    pub(crate) fn line_state_for_size(state: &LineState, size: usize) -> usize {
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
    pub(crate) fn eliminate_size_at(&mut self, row: usize, col: usize, size: usize) {
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
    pub(crate) fn apply_miss(&mut self, row: usize, col: usize) {
        for &size in &SHIP_SIZES {
            self.eliminate_size_at(row, col, size);
        }
        self.sub_candidates[row][col] = false;
    }

    /// Apply a hit of given `size` at (row, col): eliminate all LARGER sizes through
    /// this cell (since a real ship occupies it, only that exact size can be true here),
    /// and remove as submarine candidate (since it's occupied by a non-submarine, unless
    /// size==1).
    pub(crate) fn apply_hit(&mut self, row: usize, col: usize, size: usize) {
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
    pub(crate) fn apply_battleship_adjacency_elimination(&mut self, ship_cells: [(usize, usize); 4]) {
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
            self.invalidate_candidate_caches();
        }

        let frigate_locked = self.lock_in_frigate_layout();

        if cruiser_locked || frigate_locked {
            self.refresh_cross3_entry_flags();
            self.refresh_cross2_entry_flags();
            self.invalidate_candidate_caches();
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

    /// Combined "alive" value for `size` at (row, col): the row's horizontal
    /// elimination value at this column, plus the column's vertical elimination
    /// value at this row (see `line_state_score`/the VALUES tables). A value
    /// here is exactly "how many currently-alive placements of this size would
    /// be affected by firing here" — so zero means no alive placement,
    /// horizontal *or* vertical, passes through this cell at all, regardless of
    /// whether the cell itself was ever individually fired or excluded. This is
    /// what the "Ship alive grids" debug view shows (see `alive_grids`), and
    /// what `refresh_cross3_entry_flags` uses to decide a cell is dead for size 3.
    pub(crate) fn alive_value(&self, row: usize, col: usize, size: usize) -> u32 {
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

    /// Every inner cell where all 3 combined "alive" values (Battleship,
    /// Cruiser, Frigate — see `alive_grids`) have dropped to zero: nothing
    /// of size >=2 can occupy it any more, only a Submarine or nothing.
    /// Board-coordinate `(row, col)` pairs (1-indexed, matching every other
    /// `*_cells` accessor's convention), not flat indices — flattening is
    /// the caller's job. A Model-layer derived fact (combines 3 already-
    /// Model-owned grids), not a display-formatting concern, even though
    /// its only current caller is `controller::game::fully_eliminated_
    /// cells_json`. See the refactor plan's Stage 7 notes on this one.
    pub(crate) fn fully_eliminated_cells(&self) -> Vec<(usize, usize)> {
        let (_, _, combined4) = self.alive_grids(4);
        let (_, _, combined3) = self.alive_grids(3);
        let (_, _, combined2) = self.alive_grids(2);
        let mut cells = Vec::new();
        for r in 0..8 {
            for c in 0..8 {
                if combined4[r][c] == 0 && combined3[r][c] == 0 && combined2[r][c] == 0 {
                    cells.push((r + 1, c + 1));
                }
            }
        }
        cells
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
    pub(crate) fn refresh_cross3_entry_flags(&mut self) {
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
    pub(crate) fn refresh_cross2_entry_flags(&mut self) {
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

    /// Re-check, for every cross-4 entry, whether each of its 3 original
    /// fired coordinates could still possibly be the real Battleship hit
    /// that produced that salvo's "4" — mirrors `refresh_cross3_entry_flags`/
    /// `refresh_cross2_entry_flags` one size up, but simpler: there's only
    /// one Battleship, so no found-ship entry overrides or same-ship pairing
    /// are needed here, just the generic "has the combined size-4 alive
    /// value at this coordinate dropped to zero" check.
    pub(crate) fn refresh_cross4_entry_flags(&mut self) {
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

    /// Elimination "value" for `size` in a given FSM state/table-index, per the
    /// pre-generated tables — the single per-size lookup `size_cell_score` and
    /// `apply_hypothetical_miss` build on.
    pub(crate) fn line_state_score(state: usize, size: usize, table_index: usize) -> u32 {
        match size {
            4 => VALUES_SIZE4[state][table_index] as u32,
            3 => VALUES_SIZE3[state][table_index] as u32,
            2 => VALUES_SIZE2[state][table_index] as u32,
            _ => 0,
        }
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

    /// Given a salvo whose result bag contained a 3, record the entry — the
    /// true Cruiser hit is one of its 3 coordinates, we just don't know
    /// which yet (see `Cross3Entry::coord_ruled_out`, refreshed every round
    /// by `refresh_cross3_entry_flags`).
    pub(crate) fn apply_cruiser_cross_tracking(&mut self, coords: [(usize, usize); 3], values: [usize; 3]) {
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
    pub(crate) fn apply_frigate_cross_tracking(&mut self, coords: [(usize, usize); 3], values: [usize; 3]) {
        self.cross2_entries.push(Cross2Entry {
            coords,
            values,
            coord_ruled_out: [false; 3],
            coord_confirmed_frigate_hit: [false; 3],
        });
    }
}
