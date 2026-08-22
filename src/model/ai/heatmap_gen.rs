//! "Generate heatmap" (bucket B): enumerates every straight-line window
//! for Cruiser/Frigate, filters to those still consistent with the full
//! salvo history and the FSM's alive_value, pairs them into full-layout
//! hypotheses, cross-checks Cruiser hypotheses against Frigate ones for
//! mutual adjacency, and marginalizes the survivors into the per-cell
//! probability heatmaps. Extracted from the old ai.rs verbatim (Stage 5
//! of the refactor plan); a second `impl AiPlayer` block in a sibling
//! module of where `AiPlayer` is defined, so it keeps the exact same
//! field/method access an ordinary impl block would have.

use super::*;

impl AiPlayer {

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
    pub(crate) fn cells_confirmed_battleship(&self) -> [[bool; 10]; 10] {
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
    pub(crate) fn consistent_cruiser_candidates(&self) -> Vec<std::collections::HashSet<(usize, usize)>> {
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
            // `alive_value(_, _, 3) == 0` is the FSM's live, comprehensive
            // "provably not a Cruiser cell" signal — unlike the aggregate
            // salvo-history check below, it also sees cells eliminated via
            // adjacency/propagation that were never individually fired
            // (invisible to salvo_history entirely). Without this, a
            // window through such a cell could still survive purely
            // because nothing in the raw fired-coordinate bags happens to
            // contradict it.
            .filter(|w| w.iter().all(|&(r, c)| !confirmed_battleship[r][c] && self.alive_value(r, c, 3) > 0))
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
    pub(crate) fn consistent_frigate_candidates(&self) -> Vec<std::collections::HashSet<(usize, usize)>> {
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
            // `alive_value(_, _, 2) == 0` is the FSM's live, comprehensive
            // "provably not a Frigate cell" signal — a strict superset of
            // `possible` (which only sees cells that were directly fired):
            // it also catches cells eliminated via adjacency/propagation
            // that were never individually fired at all, invisible to
            // `possible` entirely. `possible` is kept alongside it rather
            // than replaced — redundant given the superset relationship,
            // but harmless, and this is exactly the kind of code where
            // "definitely still correct" beats "slightly less redundant".
            .filter(|w| w.iter().all(|&(r, c)| possible[r][c] && !confirmed_battleship[r][c] && !cruiser_exclusion[r][c] && self.alive_value(r, c, 2) > 0))
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
    /// Memoized by `salvo_history.len()` in `joint_pairs_cache` — every
    /// caller of the cross-reasoned lists hits this, and without caching it
    /// was being recomputed from scratch on each one, including twice over
    /// within a single `resolution_status_json`/`is_fully_resolved` check
    /// (once via the cruiser-refined path, once via the frigate-refined
    /// path — both redo this exact same pairs list independently). See
    /// `jointly_consistent_hypothesis_pairs_uncached` for the actual O(n^2)
    /// work; this wrapper mirrors `consistent_cruiser_candidates`'s own
    /// cache exactly, including relying on `update_fsm_and_resolve` to
    /// clear it on a lock-in event that leaves `salvo_history.len()`
    /// unchanged.
    fn jointly_consistent_hypothesis_pairs(&self) -> Vec<(std::collections::HashSet<(usize, usize)>, std::collections::HashSet<(usize, usize)>)> {
        let key = self.salvo_history.len();
        if let Some((cached_key, cached)) = self.joint_pairs_cache.borrow().as_ref() {
            if *cached_key == key {
                return cached.clone();
            }
        }
        let computed = self.jointly_consistent_hypothesis_pairs_uncached();
        *self.joint_pairs_cache.borrow_mut() = Some((key, computed.clone()));
        computed
    }

    fn jointly_consistent_hypothesis_pairs_uncached(&self) -> Vec<(std::collections::HashSet<(usize, usize)>, std::collections::HashSet<(usize, usize)>)> {
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
    pub(crate) fn cross_reasoned_cruiser_candidates(&self) -> Vec<std::collections::HashSet<(usize, usize)>> {
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
    pub(crate) fn cross_reasoned_frigate_candidates(&self) -> Vec<std::collections::HashSet<(usize, usize)>> {
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
    pub(crate) fn distinct_hypotheses(hyps: &[std::collections::HashSet<(usize, usize)>]) -> Vec<&std::collections::HashSet<(usize, usize)>> {
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
}
