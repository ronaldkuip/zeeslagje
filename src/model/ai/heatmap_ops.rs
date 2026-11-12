//! "Heatmap operations" (bucket C): consumes bucket B's heatmaps/
//! candidate lists (and, for `anchored_isolation_shot`, the battleship
//! subsystem's confirmed cells too) to actually pick or evaluate a shot
//! - the minimax disambiguation search, the anchored-isolation cleanup
//! shot, every "is this fully identified" check, and `choose_shots`
//! itself. This module reads `fsm`'s `LineState` layout directly (via
//! `fsm::line_state_for_size`) for shot scoring - that's intentional,
//! not a leak to fix; C's scoring genuinely needs A's per-line FSM
//! state. Extracted from the old ai.rs verbatim (Stage 5 of the
//! refactor plan); a second `impl AiPlayer` block in a sibling module
//! of where `AiPlayer` is defined.

use super::*;

impl AiPlayer {

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
    fn disambiguation_shots(&self, candidates: &[exact::Mask], allow_refire: bool, ignore_refire_cap: bool) -> Option<[(usize, usize); 3]> {
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

        let mut counts = [[0usize; 10]; 10];
        for &cand in candidates {
            for (r, c) in exact::cells(cand) {
                counts[r][c] += 1;
            }
        }
        let total = candidates.len();
        // Row-major order (what the BTreeMap this replaced iterated in):
        // it feeds the tie-breaks below, so the advisory for a given board
        // state stays deterministic.
        let mut informative: Vec<(usize, usize)> = (0..10)
            .flat_map(|r| (0..10).map(move |c| (r, c)))
            .filter(|&(r, c)| {
                let n = counts[r][c];
                n > 0
                    && n < total
                    && (!self.fired[r][c]
                        || (allow_refire && (ignore_refire_cap || !self.disambiguation_extra_refire_used[r][c])))
            })
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
                let n = counts[cell.0][cell.1];
                (n as isize - (total as isize - n as isize)).unsigned_abs()
            });
            informative.truncate(MAX_POOL);
        }

        let mut pool = informative;
        // The identified Battleship's unfired cells first (see
        // `pending_battleship_cells`): as neutral as any filler — their
        // value is known — but they have to be fired anyway, so a triple
        // using them is preferred over an equally good one that doesn't.
        let pending = self.pending_battleship_cells();
        for &cell in &pending {
            if !pool.contains(&cell) {
                pool.push(cell);
            }
        }
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

        let mut best: Option<([(usize, usize); 3], (usize, usize, std::cmp::Reverse<usize>))> = None;
        for i in 0..pool.len() {
            for j in (i + 1)..pool.len() {
                for k in (j + 1)..pool.len() {
                    let salvo = [pool[i], pool[j], pool[k]];
                    // For every hypothesis, how many of these 3 cells it
                    // predicts as hits — the only thing the resulting
                    // bag's count of matching values can ever reveal about
                    // which hypothesis is real (see
                    // `consistent_with_salvo_history`).
                    let salvo_mask = salvo.iter().fold(0u128, |m, &cell| m | exact::bit(cell));
                    let mut buckets = [0usize; 4];
                    for &cand in candidates {
                        buckets[(cand & salvo_mask).count_ones() as usize] += 1;
                    }
                    let worst = *buckets.iter().max().unwrap();
                    // Expected number of hypotheses left (times `total`):
                    // breaks minimax ties toward salvos that more often
                    // settle it outright — e.g. 4 layouts {D4|E5}x{E4|D5}:
                    // D4+E4+D5 always leaves 2 (E4/D5 is bag-blind), while
                    // D4+E4+filler resolves half the time, same worst case.
                    let spread: usize = buckets.iter().map(|b| b * b).sum();
                    let owed = salvo.iter().filter(|cell| pending.contains(cell)).count();
                    let score = (worst, spread, std::cmp::Reverse(owed));
                    if best.as_ref().is_none_or(|(_, best_score)| score < *best_score) {
                        best = Some((salvo, score));
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

        let cruiser_grid = self.cruiser_heatmap_refined();
        let frigate_grid = self.frigate_heatmap_refined();

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
        self.disambiguation_shots(&self.cruiser_masks(), false, false)
    }

    /// Same idea as `cruiser_disambiguation_shots`, one size down.
    pub fn frigate_disambiguation_shots(&self) -> Option<[(usize, usize); 3]> {
        if self.salvo_history.is_empty() {
            return None;
        }
        self.disambiguation_shots(&self.frigate_masks(), false, false)
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
        self.disambiguation_shots(&self.cruiser_masks(), true, false)
    }

    /// Same idea as `cruiser_disambiguation_shots_with_refire`, one size down.
    pub fn frigate_disambiguation_shots_with_refire(&self) -> Option<[(usize, usize); 3]> {
        if self.salvo_history.is_empty() {
            return None;
        }
        self.disambiguation_shots(&self.frigate_masks(), true, false)
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
        self.disambiguation_shots(&self.cruiser_masks(), true, true)
    }

    /// Same idea as `cruiser_disambiguation_shots_last_resort`, one size down.
    pub fn frigate_disambiguation_shots_last_resort(&self) -> Option<[(usize, usize); 3]> {
        if self.salvo_history.is_empty() {
            return None;
        }
        self.disambiguation_shots(&self.frigate_masks(), true, true)
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
        match Self::distinct_masks(&self.cross_reasoned_cruiser_masks()).as_slice() {
            [only] => exact::cells(*only).collect(),
            _ => Vec::new(),
        }
    }

    /// Same idea as `cruiser_identified_cells_refined`, one size down.
    pub fn frigate_identified_cells_refined(&self) -> Vec<(usize, usize)> {
        match Self::distinct_masks(&self.cross_reasoned_frigate_masks()).as_slice() {
            [only] => exact::cells(*only).collect(),
            _ => Vec::new(),
        }
    }

    /// Whether Battleship, Cruiser, and Frigate are each fully identified —
    /// the exact same 3-flag computation `controller::game::Game`'s
    /// `resolution_status_json` (which needs all 3 individual flags for its
    /// JSON output) and `is_fully_resolved` (which only needs their AND)
    /// used to duplicate independently. Centralized here — a Model-layer
    /// query, since it reads only `AiPlayer` state — so both callers stay
    /// in sync automatically instead of by convention. See the refactor
    /// plan's "fake View methods" fix for `resolution_status_json`.
    pub(crate) fn identification_flags(&self) -> (bool, bool, bool) {
        let battleship_identified = !self.battleship_identified_cells().is_empty() || !self.found_battleship_cells().is_empty();
        let cruiser_identified = !self.cruiser_identified_cells_refined().is_empty();
        let frigate_identified = !self.frigate_identified_cells_refined().is_empty();
        (battleship_identified, cruiser_identified, frigate_identified)
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

    /// Same idea as `best_cell_for_size`, but the score is a blend of the
    /// size actually being hunted with a small tie-breaking contribution
    /// from the next size down's current line state (`secondary`) — e.g.
    /// while hunting Battleship, also weigh in how useful a cell still is
    /// for Cruiser; while hunting Cruiser, weigh in Frigate. `PRIMARY_
    /// WEIGHT` is far larger than either score's own natural range
    /// (`VALUES_SIZE{4,3,2}` entries top out in the single digits), so
    /// this can only ever break a tie in the primary size's own ranking —
    /// it can never cause a cell that's worse for the size actually being
    /// hunted to be preferred over one that's better for it. `secondary:
    /// None` (Frigate hunting has no smaller size left to blend in)
    /// reproduces `best_cell_for_size`'s plain score exactly.
    fn best_cell_for_size_blended(
        &self,
        row_line: &[usize; 10],
        col_line: &[usize; 10],
        secondary: Option<(&[usize; 10], &[usize; 10], usize)>,
        chosen_so_far: &[(usize, usize)],
        forbid_candidates: bool,
        require_candidates: bool,
        allow_refired: bool,
        size: usize,
    ) -> (usize, usize) {
        const PRIMARY_WEIGHT: u32 = 1000;
        self.best_cell_by_score(chosen_so_far, forbid_candidates, require_candidates, allow_refired, |r, c| {
            let primary = Self::size_cell_score(row_line, col_line, r, c, size);
            match secondary {
                Some((row2, col2, size2)) => primary * PRIMARY_WEIGHT + Self::size_cell_score(row2, col2, r, c, size2),
                None => primary,
            }
        })
    }

    /// Which ship size `choose_shots` is currently trying to eliminate: the
    /// largest size — in priority order Battleship (4) > Cruiser (3) > Frigate
    /// (2) — that isn't fully sunk yet. Falls back to 1 (submarines) once all
    /// three bigger classes are found; `choose_shots` doesn't target submarines
    /// itself, but this still reports "nothing bigger is left" accurately for
    /// the debug UI.
    pub fn current_target_size(&self) -> usize {
        for &size in &SHIP_SIZES {
            if !self.size_found_for_hunting(size) {
                if size == 2 && self.freeze_before_frigates {
                    return 3;
                }
                return size;
            }
        }
        1
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
        self.with_pending_battleship_cells(self.choose_shots_before_battleship_anchors())
    }

    fn choose_shots_before_battleship_anchors(&self) -> [(usize, usize); 3] {
        self.choose_shots_avoiding(Vec::new())
    }

    fn choose_shots_avoiding(&self, avoid_as_filler: Vec<(usize, usize)>) -> [(usize, usize); 3] {
        // Cruiser and Frigate discovery (pinpointing exact cells, and so
        // also disambiguating between candidate layouts) are deliberately
        // not attempted — see `refresh_cross3_entry_flags`/
        // `refresh_cross2_entry_flags`. `chosen`/`avoid_as_filler` stay
        // empty here; the loop below fills all 3 slots from the ordinary
        // FSM search.
        let mut chosen: Vec<(usize, usize)> = Vec::with_capacity(3);

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
        if self.size_found_for_hunting(4) && self.size_fully_found(3) {
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

        // The hunted size's layout already pinned down to a single
        // hypothesis: its unfired cells are certain hits, so fire them (and
        // any identified Battleship's pending cells) before anything else.
        // The line FSM below never sees this cross-salvo deduction and would
        // otherwise keep probing cells it still thinks could hold the ship.
        if size == 2 || size == 3 {
            for cell in self.known_unfired_cells(size) {
                if chosen.len() < 3 && !chosen.contains(&cell) {
                    chosen.push(cell);
                }
            }
            if !chosen.is_empty() {
                for cell in self.pending_battleship_cells() {
                    if chosen.len() < 3 && !chosen.contains(&cell) {
                        chosen.push(cell);
                    }
                }
            }
            if chosen.len() == 3 {
                return [chosen[0], chosen[1], chosen[2]];
            }
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
        // Cruiser's current line state, kept up to date across every pick
        // this call makes (see the hypothetical-miss refolding below) —
        // computed whenever size 4 is being hunted at all (not just once
        // `hunt_with_cruiser_fsm` kicks in), since it's also the "secondary"
        // blended into the ordinary per-cell score below (see
        // `best_cell_for_size_blended`), which applies from the very first
        // Battleship shot, well before the cross-4 bag narrows anything.
        let mut row3: [usize; 10] = [0; 10];
        let mut col3: [usize; 10] = [0; 10];
        if size == 4 {
            for r in 0..10 {
                row3[r] = Self::line_state_for_size(&self.row_state[r], 3);
            }
            for c in 0..10 {
                col3[c] = Self::line_state_for_size(&self.col_state[c], 3);
            }
        }
        // Same idea, one size down: Frigate's line state, blended into
        // Cruiser hunting's score.
        let mut row2: [usize; 10] = [0; 10];
        let mut col2: [usize; 10] = [0; 10];
        if size == 3 {
            for r in 0..10 {
                row2[r] = Self::line_state_for_size(&self.row_state[r], 2);
            }
            for c in 0..10 {
                col2[c] = Self::line_state_for_size(&self.col_state[c], 2);
            }
        }
        let allow_refired = self.is_refire_allowed(size);

        // Never hunt a cell the salvo history already proves empty of every
        // ship of size >= 2 (see `exact`); the line FSM can't see those
        // cross-salvo deductions and keeps scoring such cells as alive.
        // Only from the Cruiser hunt on: during the Battleship hunt the
        // audit found few provably-empty shots (0.3 per game), and checking
        // there too bought ~0.007 salvo for ~10% more solve time.
        let mut avoid_as_filler = avoid_as_filler;
        for r in if size <= 3 { INNER_LO..=INNER_HI } else { 1..=0 } {
            for c in INNER_LO..=INNER_HI {
                if !self.fired[r][c] && self.provably_no_big_ship(r, c) && !avoid_as_filler.contains(&(r, c)) {
                    avoid_as_filler.push((r, c));
                }
            }
        }

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
                // Blend in the next size down's current line state as a
                // tie-breaker — see `best_cell_for_size_blended`. `None`
                // for Frigate hunting (size 2), which has nothing smaller
                // left to blend in; reproduces the plain unblended score.
                let secondary: Option<(&[usize; 10], &[usize; 10], usize)> = match size {
                    4 => Some((&row3, &col3, 3)),
                    3 => Some((&row2, &col2, 2)),
                    _ => None,
                };
                self.best_cell_for_size_blended(&row_line, &col_line, secondary, &exclude, forbid_candidates, false, allow_refired, size)
            };

            Self::apply_hypothetical_miss(&mut row_line, &mut col_line, next.0, next.1, size);
            if size == 4 {
                Self::apply_hypothetical_miss(&mut row3, &mut col3, next.0, next.1, 3);
            }
            if size == 3 {
                Self::apply_hypothetical_miss(&mut row2, &mut col2, next.0, next.1, 2);
            }
            chosen.push(next);
        }

        let greedy = [chosen[0], chosen[1], chosen[2]];
        if size == 2 || size == 3 {
            if let Some(redundant) = self.bag_blind_cell(&greedy, size) {
                if avoid_as_filler.iter().filter(|&&(r, c)| !self.provably_no_big_ship(r, c)).count() < 6 {
                    let mut avoid = avoid_as_filler.clone();
                    avoid.push(redundant);
                    return self.choose_shots_avoiding(avoid);
                }
            }
        }
        if size == 3 {
            if let Some(better) = self.cruiser_hunt_salvo(greedy) {
                return better;
            }
        }
        if size == 2 {
            if let Some(better) = self.frigate_endgame_salvo(greedy) {
                return better;
            }
        }
        greedy
    }

    /// One-step lookahead for the Cruiser hunt — the Cruiser-phase
    /// counterpart of `frigate_endgame_salvo`. The greedy FSM pick scores
    /// cells one at a time by line eliminations and can't see which
    /// salvos actually split the surviving Cruiser layouts (nor bag
    /// ambiguity between its own 3 cells). This scores a salvo by the
    /// expected number of salvos still needed to pin down AND sink both
    /// Cruisers, averaged over every consistent Cruiser placement grouped
    /// by the bag it would produce, and replaces `greedy` only when some
    /// salvo is strictly better. Pool: the greedy cells, the cells closest
    /// to splitting the layouts in half, certain-hit cells, and pending
    /// Battleship cells (hits owed, known value, never split anything).
    fn cruiser_hunt_salvo(&self, greedy: [(usize, usize); 3]) -> Option<[(usize, usize); 3]> {
        // Beyond this many layouts no 3-cell salvo splits them much and the
        // greedy pick is as good (measured: 200/1000/4000 caps identical),
        // so the planner only pays its cost late in the phase.
        const MAX_HYPOTHESES: usize = 300;
        const INFORMATIVE: usize = 10;
        const CERTAIN: usize = 2;
        // Small credit per salvo cell that can still hold a Frigate: the
        // estimate below only counts Cruiser work, and without this the
        // planner traded Frigate information away for tiny Cruiser gains.
        const FRIGATE_CREDIT: f64 = 0.3;

        let hyps = self.cruiser_masks();
        if hyps.len() <= 1 || hyps.len() > MAX_HYPOTHESES {
            return None;
        }
        let bit = |(r, c): (usize, usize)| 1u128 << (r * 10 + c);
        let total = hyps.len();

        let mut fired = 0u128;
        for r in 0..10 {
            for c in 0..10 {
                if self.fired[r][c] {
                    fired |= bit((r, c));
                }
            }
        }
        let pending = self.pending_battleship_cells();
        let pending_mask = pending.iter().fold(0u128, |m, &cell| m | bit(cell));

        // Per-cell hypothesis counts, for ranking pool cells.
        let mut counts = [[0usize; 10]; 10];
        for &h in &hyps {
            let mut m = h;
            while m != 0 {
                let i = m.trailing_zeros() as usize;
                counts[i / 10][i % 10] += 1;
                m &= m - 1;
            }
        }
        let mut pool: Vec<(usize, usize)> = greedy.to_vec();
        for &cell in &pending {
            if !pool.contains(&cell) {
                pool.push(cell);
            }
        }
        let mut informative: Vec<((usize, usize), usize)> = Vec::new();
        let mut certain: Vec<(usize, usize)> = Vec::new();
        for r in INNER_LO..=INNER_HI {
            for c in INNER_LO..=INNER_HI {
                if self.fired[r][c] || pool.contains(&(r, c)) {
                    continue;
                }
                let n = counts[r][c];
                if n == total {
                    certain.push((r, c));
                } else if n > 0 {
                    informative.push(((r, c), (2 * n).abs_diff(total)));
                }
            }
        }
        informative.sort_by_key(|&(_, d)| d);
        pool.extend(informative.iter().take(INFORMATIVE).map(|&(cell, _)| cell));
        pool.extend(certain.iter().take(CERTAIN));
        if pool.len() < 3 {
            return None;
        }

        // Cells still worth probing for Frigates: unfired inner cells the
        // salvo history hasn't proven empty.
        let mut frigate_probe = 0u128;
        for r in INNER_LO..=INNER_HI {
            for c in INNER_LO..=INNER_HI {
                if !self.fired[r][c] && !self.provably_no_big_ship(r, c) {
                    frigate_probe |= bit((r, c));
                }
            }
        }

        #[derive(Clone, Copy)]
        struct Group { n: usize, rem_max: u32, or: u128, and: u128 }
        let evaluate = |salvo: [(usize, usize); 3]| -> (f64, usize) {
            let s = salvo.iter().fold(0u128, |m, &cell| m | bit(cell));
            let fired_after = fired | s;
            let frig_value = (s & frigate_probe).count_ones() as f64 / 3.0;
            let mut groups = [Group { n: 0, rem_max: 0, or: 0, and: !0 }; 4];
            for &h in &hyps {
                let g = &mut groups[(s & h).count_ones() as usize];
                g.n += 1;
                g.rem_max = g.rem_max.max((h & !fired_after).count_ones());
                g.or |= h;
                g.and &= h;
            }
            let pending_left = (pending_mask & !fired_after).count_ones();
            let mut weighted = 0.0;
            let mut worst = 0;
            for g in groups.iter().filter(|g| g.n > 0) {
                let hits_left = (g.rem_max + pending_left).div_ceil(3) as f64;
                let estimate = if g.n == 1 {
                    hits_left
                } else if (g.or ^ g.and) & !fired_after == 0 {
                    // Only a refire can still separate these.
                    hits_left + 1.0
                } else {
                    Self::salvos_to_resolve(g.n, hits_left)
                };
                weighted += estimate * g.n as f64;
                worst = worst.max(g.n);
            }
            (weighted / total as f64 - FRIGATE_CREDIT * frig_value, worst)
        };

        let greedy_score = evaluate(greedy);
        let mut best: Option<([(usize, usize); 3], (f64, usize))> = None;
        for i in 0..pool.len() {
            for j in (i + 1)..pool.len() {
                for k in (j + 1)..pool.len() {
                    let salvo = [pool[i], pool[j], pool[k]];
                    let score = evaluate(salvo);
                    let better = match &best {
                        None => true,
                        Some((_, (cost, worst))) => score.0 < cost - 1e-9 || ((score.0 - cost).abs() <= 1e-9 && score.1 < *worst),
                    };
                    if better {
                        best = Some((salvo, score));
                    }
                }
            }
        }
        let (salvo, (cost, worst)) = best?;
        if cost < greedy_score.0 - 1e-9 || ((cost - greedy_score.0).abs() <= 1e-9 && worst < greedy_score.1) {
            Some(salvo)
        } else {
            None
        }
    }

    /// Estimated salvos still needed when `n` (> 1) layouts remain that
    /// some unfired cell can still separate, and `hits_left` salvos' worth
    /// of hits are owed regardless. Disambiguating and sinking overlap
    /// (kill shots anchor probes and vice versa), so the two are not
    /// added; resolving `n` layouts costs about log2(n) further salvos (a
    /// bag has 4 outcomes for one class, so a good salvo cuts the layouts
    /// by a factor of 2-3), plus half a salvo of slack. Replaced a flat
    /// "+0.5 whatever n is", which made both planners prefer kill shots
    /// over splitting shots: 14.69 -> 14.07 on the 20,000-board set, the
    /// divisor on log2 swept from 2.0 to 0.8 with 1.0-1.2 flat at the
    /// optimum.
    fn salvos_to_resolve(n: usize, hits_left: f64) -> f64 {
        hits_left.max((n as f64).log2()) + 0.5
    }

    /// One-step lookahead for the Frigate endgame (Battleship and both
    /// Cruisers sunk, few joint hypotheses left): scores a salvo by the
    /// expected number of salvos still needed afterwards — to both pin the
    /// layout down AND hit every remaining Frigate cell — averaged over
    /// every surviving (Cruiser, Frigate) hypothesis pair, grouped by the
    /// bag each would produce. Only replaces `greedy` if some salvo is
    /// strictly better under that estimate.
    ///
    /// Exists because the greedy hunt's hypothetical-miss folding can't see
    /// bag ambiguity: with "B4 + (B3 or C4)" left, a miss assumed at C4
    /// makes B3 a certain hit, so greedy fires B3 and C4 together — which
    /// always yields the same bag whichever is real (exactly one of them
    /// hits), forcing an extra refire salvo afterwards. Firing just one of
    /// them alongside other useful cells resolves it outright.
    fn frigate_endgame_salvo(&self, greedy: [(usize, usize); 3]) -> Option<[(usize, usize); 3]> {
        const MAX_HYPOTHESES: usize = 1000;
        const INFORMATIVE: usize = 10;

        let hyps = self.joint_pairs_masks();
        if hyps.len() <= 1 || hyps.len() > MAX_HYPOTHESES {
            return None;
        }
        let bit = |(r, c): (usize, usize)| 1u128 << (r * 10 + c);

        let mut fired = 0u128;
        for r in 0..10 {
            for c in 0..10 {
                if self.fired[r][c] {
                    fired |= bit((r, c));
                }
            }
        }
        let any_cell = hyps.iter().fold(0u128, |m, &(c, f)| m | c | f);
        // An identified Battleship's still-unfired cells (see
        // `pending_battleship_cells`): hits still owed, same as unhit
        // Frigate cells, but with a value known in advance — they never
        // split hypotheses, so the planner can slot them in wherever a
        // salvo has room without losing information.
        let pending = self.pending_battleship_cells();
        let pending_mask = pending.iter().fold(0u128, |m, &cell| m | bit(cell));

        // Pool: the greedy cells, pending Battleship cells, the cells that
        // split the hypotheses most evenly (top `INFORMATIVE`; 8-30 all
        // measure the same, 6 starts to lose, and the triple loop is cubic
        // in the pool), and 2 known-empty fillers so a salvo can hold fewer
        // than 3 informative cells (see `disambiguation_shots`' padding
        // comment for why that matters).
        let mut pool: Vec<(usize, usize)> = greedy.iter().copied().filter(|&(r, c)| !self.fired[r][c]).collect();
        for &cell in &pending {
            if !pool.contains(&cell) {
                pool.push(cell);
            }
        }
        let total = hyps.len();
        let mut informative: Vec<((usize, usize), usize)> = Vec::new();
        let mut fillers = 0;
        for r in 0..10 {
            for c in 0..10 {
                let b = bit((r, c));
                if fired & b != 0 || pool.contains(&(r, c)) {
                    continue;
                }
                if any_cell & b != 0 {
                    let n = hyps.iter().filter(|&&(cm, fm)| (cm | fm) & b != 0).count();
                    informative.push(((r, c), (2 * n).abs_diff(total)));
                } else if fillers < 2 {
                    pool.push((r, c));
                    fillers += 1;
                }
            }
        }
        informative.sort_by_key(|&(_, d)| d);
        pool.extend(informative.iter().take(INFORMATIVE).map(|&(cell, _)| cell));
        if pool.len() < 3 {
            return None;
        }

        // Per-bag-outcome aggregates, enough to estimate what's left
        // without materializing the groups themselves.
        #[derive(Clone, Copy)]
        struct Group { n: usize, rem_max: u32, or_c: u128, and_c: u128, or_f: u128, and_f: u128 }
        let evaluate = |salvo: [(usize, usize); 3]| -> (f64, usize) {
            let s = salvo.iter().fold(0u128, |m, &cell| m | bit(cell));
            let fired_after = fired | s;
            let mut groups = [Group { n: 0, rem_max: 0, or_c: 0, and_c: !0, or_f: 0, and_f: !0 }; 16];
            for &(c, f) in &hyps {
                let (c, f): (u128, u128) = (c, f);
                let g = &mut groups[((s & c).count_ones() * 4 + (s & f).count_ones()) as usize];
                g.n += 1;
                g.rem_max = g.rem_max.max((f & !fired_after).count_ones());
                g.or_c |= c;
                g.and_c &= c;
                g.or_f |= f;
                g.and_f &= f;
            }
            let mut weighted = 0.0;
            let mut worst = 0;
            for g in groups.iter().filter(|g| g.n > 0) {
                let hits_left = (g.rem_max + (pending_mask & !fired_after).count_ones()).div_ceil(3) as f64;
                let estimate = if g.n == 1 {
                    hits_left
                } else if ((g.or_c ^ g.and_c) | (g.or_f ^ g.and_f)) & !fired_after == 0 {
                    // Every cell these hypotheses disagree on is already
                    // fired: only a refire can still separate them.
                    hits_left + 1.0
                } else {
                    Self::salvos_to_resolve(g.n, hits_left)
                };
                weighted += estimate * g.n as f64;
                worst = worst.max(g.n);
            }
            (weighted / hyps.len() as f64, worst)
        };

        let greedy_score = evaluate(greedy);
        let mut best: Option<([(usize, usize); 3], (f64, usize))> = None;
        for i in 0..pool.len() {
            for j in (i + 1)..pool.len() {
                for k in (j + 1)..pool.len() {
                    let salvo = [pool[i], pool[j], pool[k]];
                    let score = evaluate(salvo);
                    let better = match &best {
                        None => true,
                        Some((_, (cost, worst))) => score.0 < cost - 1e-9 || ((score.0 - cost).abs() <= 1e-9 && score.1 < *worst),
                    };
                    if better {
                        best = Some((salvo, score));
                    }
                }
            }
        }
        let (salvo, (cost, worst)) = best?;
        if cost < greedy_score.0 - 1e-9 || ((cost - greedy_score.0).abs() <= 1e-9 && worst < greedy_score.1) {
            Some(salvo)
        } else {
            None
        }
    }

    /// A cell of `salvo` the bag can't attribute: some subset of 2 or 3
    /// salvo cells, each of which is a `size`-ship cell in some surviving
    /// hypotheses but not all, whose combined hit count is nevertheless the
    /// same in every hypothesis (e.g. both possible ends C4/F4 of a Cruiser
    /// already hit at D4+E4). Firing them together guarantees hits but
    /// says nothing about which cell scored, forcing a refire later — the
    /// greedy hunt's hypothetical-miss folding can't see that (a miss
    /// assumed at C4 makes F4 look like a sure hit). Returns the last such
    /// cell in pick order, so the hunt can re-pick without it.
    fn bag_blind_cell(&self, salvo: &[(usize, usize); 3], size: usize) -> Option<(usize, usize)> {
        const MAX_HYPOTHESES: usize = 20_000;
        let close = |a: (usize, usize), b: (usize, usize)| a.0.abs_diff(b.0).max(a.1.abs_diff(b.1)) <= size;
        let mut subsets: Vec<Vec<usize>> = Vec::new();
        for i in 0..3 {
            for j in (i + 1)..3 {
                if close(salvo[i], salvo[j]) {
                    subsets.push(vec![i, j]);
                }
            }
        }
        if subsets.is_empty() {
            return None;
        }
        if subsets.len() >= 2 {
            subsets.push(vec![0, 1, 2]);
        }
        let hyps = if size == 3 { self.cruiser_masks() } else { self.frigate_masks() };
        if hyps.len() < 2 || hyps.len() > MAX_HYPOTHESES {
            return None;
        }
        let varies = |cell: &(usize, usize)| {
            let b = exact::bit(*cell);
            let n = hyps.iter().filter(|&&h| h & b != 0).count();
            n > 0 && n < hyps.len()
        };
        let varying: Vec<bool> = salvo.iter().map(varies).collect();
        for subset in subsets {
            if !subset.iter().all(|&i| varying[i]) {
                continue;
            }
            let sub_mask = subset.iter().fold(0u128, |m, &i| m | exact::bit(salvo[i]));
            let count = |h: u128| (h & sub_mask).count_ones();
            let first = count(hyps[0]);
            if hyps.iter().all(|&h| count(h) == first) {
                return Some(salvo[*subset.last().unwrap()]);
            }
        }
        None
    }

    /// Unfired cells of `size`'s ships when the layout for that size is down
    /// to a single hypothesis; empty otherwise (or when the hypothesis list
    /// is too large to be worth checking). Frigates use the cross-reasoned
    /// list — cheap by then, since the Cruisers are sunk. Cruisers only use
    /// the raw list: cross-checking it against every Frigate hypothesis
    /// mid-hunt would mean enumerating Frigate triples while the Frigates are
    /// still wide open, roughly quadrupling solve time for ~0.1 salvo.
    fn known_unfired_cells(&self, size: usize) -> Vec<(usize, usize)> {
        const MAX_HYPOTHESES: usize = 20_000;
        let raw = if size == 3 { self.cruiser_masks() } else { self.frigate_masks() };
        if raw.is_empty() || raw.len() > MAX_HYPOTHESES {
            return Vec::new();
        }
        let hyps = if size == 3 { raw } else { self.cross_reasoned_frigate_masks() };
        match Self::distinct_masks(&hyps).as_slice() {
            // Row-major, as `exact::cells` yields.
            [only] => exact::cells(*only).filter(|&(r, c)| !self.fired[r][c]).collect(),
            _ => Vec::new(),
        }
    }

    /// Whether `size` is done as far as HUNTING goes: fully sunk, or — for
    /// the Battleship — its exact 4 cells already identified, even if some
    /// are still unfired (see `pending_battleship_cells`).
    pub(crate) fn size_found_for_hunting(&self, size: usize) -> bool {
        self.size_fully_found(size) || (size == 4 && self.battleship_identified().is_some())
    }

    /// The identified Battleship's cells that still have to be fired to
    /// sink it. Empty once sunk or while it isn't identified yet.
    pub(crate) fn pending_battleship_cells(&self) -> Vec<(usize, usize)> {
        if self.size_fully_found(4) {
            return Vec::new();
        }
        match self.battleship_identified() {
            Some(cells) => cells.into_iter().filter(|&(r, c)| !self.fired[r][c]).collect(),
            None => Vec::new(),
        }
    }

    /// Fires the identified Battleship's remaining cells only in slots
    /// that would otherwise teach nothing about any ship of size >= 2:
    /// known-empty filler, or a refire of an already-hit Battleship cell
    /// used as an anchor. Their "4" is known in advance, so they anchor a
    /// bag just as well as a miss — and they have to be fired eventually
    /// anyway, so this saves the slots spent sinking it straight away.
    fn with_pending_battleship_cells(&self, mut shots: [(usize, usize); 3]) -> [(usize, usize); 3] {
        let mut pending: Vec<(usize, usize)> = self.pending_battleship_cells().into_iter().filter(|cell| !shots.contains(cell)).collect();
        if pending.is_empty() {
            return shots;
        }
        let battleship = self.battleship_identified().unwrap_or_default();
        for slot in shots.iter_mut() {
            let Some(&next) = pending.first() else { break };
            let (r, c) = *slot;
            let wasted = if battleship.contains(slot) {
                self.fired[r][c]
            } else {
                // The outer ring can only ever hold Submarines.
                let inner = (INNER_LO..=INNER_HI).contains(&r) && (INNER_LO..=INNER_HI).contains(&c);
                !inner || (self.alive_value(r, c, 2) == 0 && self.alive_value(r, c, 3) == 0) || self.provably_no_big_ship(r, c)
            };
            if wasted {
                *slot = next;
                pending.remove(0);
            }
        }
        shots
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Last Frigate somewhere in the D4/E4/D5/E5 square, all four cells
    /// already fired: one of {D4, E5} plus one of {E4, D5}. Firing E4 and D5
    /// together is bag-blind (exactly one of them always hits), so the
    /// refire must not include both; D4+E4(+filler) resolves half the time.
    #[test]
    fn disambiguation_avoids_bag_blind_refire_pair() {
        let (d4, e4, d5, e5) = ((3, 3), (3, 4), (4, 3), (4, 4));
        let mut ai = AiPlayer::new();
        for &(r, c) in &[d4, e4, d5, e5] {
            ai.fired[r][c] = true;
        }
        let set = |a: (usize, usize), b: (usize, usize)| exact::bit(a) | exact::bit(b);
        let candidates = vec![set(d4, e4), set(d4, d5), set(e5, e4), set(e5, d5)];
        let salvo = ai.disambiguation_shots(&candidates, true, true).expect("a refire salvo");
        let informative = salvo.iter().filter(|cell| [d4, e4, d5, e5].contains(cell)).count();
        assert!(!(salvo.contains(&e4) && salvo.contains(&d5)), "E4+D5 together is bag-blind: {salvo:?}");
        assert!(!(salvo.contains(&d4) && salvo.contains(&e5)), "D4+E5 together is bag-blind: {salvo:?}");
        assert_eq!(informative, 2, "two non-blind refires resolve half the time: {salvo:?}");
    }
}
