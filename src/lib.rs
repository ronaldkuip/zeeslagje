use wasm_bindgen::prelude::*;
use serde::Serialize;

mod fsm_tables;
mod model;

use model::ai::AiPlayer;
use model::fleet::{cell_to_str, generate_board, try_place};
pub use model::fleet::{BoardLayout, Cell, GameState, ResolutionStatus, SalvoResult, Ship};

// ---------------------------------------------------------------------------
// WASM-exported Game struct
// ---------------------------------------------------------------------------

#[wasm_bindgen]
pub struct Game {
    state: GameState,
    ai: AiPlayer,
}

#[wasm_bindgen]
impl Game {
    /// Create a new game with a freshly generated random board.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Game {
        Game { state: generate_board(), ai: AiPlayer::new() }
    }

    /// Reset to a new game.
    pub fn reset(&mut self) {
        self.state = generate_board();
        self.ai = AiPlayer::new();
    }

    /// Restart play from scratch on the SAME board (ship placement
    /// unchanged) — for replaying an identical layout to see whether the
    /// AI's deduction plays out the same way or diverges, without waiting
    /// for a fresh board to happen to reproduce the same situation.
    /// Clears every hit/fired/log/AI-deduction state, exactly like a new
    /// game, but leaves `state.board`/`state.ships` (and therefore
    /// `total_hits`) untouched.
    pub fn restart_same_board(&mut self) {
        for ship in &mut self.state.ships {
            ship.hits = 0;
            ship.sunk = false;
        }
        self.state.fired = vec![vec![false; 10]; 10];
        self.state.log = Vec::new();
        self.state.turn = 1;
        self.state.won = false;
        self.state.hit_count = 0;
        self.ai = AiPlayer::new();
    }

    /// The current board's ship placement only (no fired/hit/turn state) —
    /// for saving a board to replay later. See `load_board_layout_json`.
    pub fn board_layout_json(&self) -> String {
        let layout = BoardLayout { board: self.state.board.clone(), ships: self.state.ships.clone() };
        serde_json::to_string(&layout).unwrap_or_else(|_| "{}".to_string())
    }

    /// Start a fresh game on a previously-saved board layout (see
    /// `board_layout_json`) instead of a randomly generated one — same
    /// idea as `restart_same_board`, but for an externally supplied
    /// layout. Any hits/sunk flags in the JSON are ignored; the loaded
    /// game always starts completely fresh. Returns a JSON error object
    /// if the layout doesn't parse.
    pub fn load_board_layout_json(&mut self, json: &str) -> String {
        let layout: BoardLayout = match serde_json::from_str(json) {
            Ok(l) => l,
            Err(e) => return format!("{{\"error\":\"invalid board layout: {e}\"}}"),
        };
        let total_hits: usize = layout.ships.iter().map(|s| s.size).sum();
        let mut ships = layout.ships;
        for ship in &mut ships {
            ship.hits = 0;
            ship.sunk = false;
        }
        self.state = GameState {
            board: layout.board,
            ships,
            fired: vec![vec![false; 10]; 10],
            log: Vec::new(),
            turn: 1,
            won: false,
            total_hits,
            hit_count: 0,
        };
        self.ai = AiPlayer::new();
        r#"{"ok":true}"#.to_string()
    }

    /// Whether the AI's own deduction currently has every identifiable
    /// ship class (Battleship, Cruiser, Frigate) pinned down with full
    /// certainty, plus the Cruiser/Frigate probability grids when it
    /// doesn't. Cruiser/Frigate certainty is checked after cross-reasoning
    /// each one's remaining hypotheses against the other's for mutual
    /// adjacency (see `AiPlayer::cruiser_identified_cells_refined`) — this
    /// can resolve a board neither heatmap resolves on its own. See
    /// `ResolutionStatus`.
    pub fn resolution_status_json(&self) -> String {
        let battleship_identified = !self.ai.battleship_identified_cells().is_empty() || !self.ai.found_battleship_cells().is_empty();
        let cruiser_identified = !self.ai.cruiser_identified_cells_refined().is_empty();
        let frigate_identified = !self.ai.frigate_identified_cells_refined().is_empty();
        let resolved = battleship_identified && cruiser_identified && frigate_identified;
        let status = ResolutionStatus {
            resolved,
            battleship_identified,
            cruiser_identified,
            frigate_identified,
            cruiser_odds: if resolved { None } else { Some(self.ai.cruiser_heatmap_refined()) },
            frigate_odds: if resolved { None } else { Some(self.ai.frigate_heatmap_refined()) },
        };
        serde_json::to_string(&status).unwrap_or_else(|_| "{}".to_string())
    }

    /// Same 3-part check as `resolution_status_json`'s `resolved` field,
    /// without the JSON/odds overhead — used by `fire` to decide whether
    /// "won" should actually stop further firing (see its doc comment).
    fn is_fully_resolved(&self) -> bool {
        let battleship_identified = !self.ai.battleship_identified_cells().is_empty() || !self.ai.found_battleship_cells().is_empty();
        let cruiser_identified = !self.ai.cruiser_identified_cells_refined().is_empty();
        let frigate_identified = !self.ai.frigate_identified_cells_refined().is_empty();
        battleship_identified && cruiser_identified && frigate_identified
    }

    /// Fire a salvo of exactly 3 cells. Coordinates are flat indices: row * 10 + col.
    /// Returns a JSON-serialised SalvoResult, or an error string.
    ///
    /// `won` (every real ship cell hit at least once) is deliberately NOT
    /// enough on its own to stop firing — a disambiguation salvo's filler
    /// cell can incidentally BE the very last unfound real cell, so `won`
    /// can flip true mid-disambiguation, before the Cruiser/Frigate exact
    /// layout is actually pinned down. Locking out all further firing at
    /// that instant would permanently strand a resolvable ambiguity: the
    /// refire/last-resort salvo needed to finish it would itself now be
    /// rejected as "game already won", with no way to ever submit it. Only
    /// once the layout is ALSO fully resolved (see `is_fully_resolved`) is
    /// there genuinely nothing further to learn.
    pub fn fire(&mut self, indices: &[usize]) -> String {
        if self.state.won && self.is_fully_resolved() {
            return r#"{"error":"game already won"}"#.to_string();
        }
        if indices.len() != 3 {
            return r#"{"error":"exactly 3 cells required"}"#.to_string();
        }

        // Validate: no duplicates, none already fired — unless the refire-allowed
        // toggle is currently on for whatever size the AI is hunting, in which
        // case an already-fired cell is permitted (but never a cell repeated
        // twice within this SAME salvo — that's a different, always-invalid case).
        let refire_ok = self.ai.is_refire_allowed(self.ai.current_target_size());
        // Genuine board exhaustion: fewer than 3 unfired cells remain anywhere
        // on the 100-cell board. A 3-cell salvo can't be won that only ever
        // needs 20 real hits, but every fired cell that turns out to be a
        // miss still consumes one of the 80 non-ship cells, so a long enough
        // run of misses can legitimately exhaust the board down to its last
        // couple of cells before the final ship cell is ever found — at that
        // point there simply aren't 3 distinct never-fired cells left to
        // offer, and refiring is the only way to submit a salvo at all. Not
        // a debug relaxation like `refire_ok` — a structural necessity, so
        // it's unconditional.
        let unfired_count: usize = self.state.fired.iter().flatten().filter(|&&f| !f).count();
        let board_exhausted = unfired_count < 3;
        let mut cells_to_fire: Vec<(usize, usize)> = Vec::new();
        let mut extra_refire_cells: Vec<(usize, usize)> = Vec::new();
        for &idx in indices {
            let r = idx / 10;
            let c = idx % 10;
            if r > 9 || c > 9 {
                return r#"{"error":"index out of range"}"#.to_string();
            }
            // The Battleship discriminating-cell refire (see
            // `AiPlayer::battleship_discriminating_test_cell`) is always let
            // through, independent of the general refire-allowed toggle —
            // it's a specific internal strategy, not the debug relaxation:
            // when every coordinate that would distinguish between 2+
            // surviving candidate windows has already been fired as an
            // ambiguous decoy in some earlier cross-4 salvo, re-firing one
            // is the ONLY way the ambiguity ever closes — every other
            // surviving candidate is common to every window and would just
            // confirm a hit without discriminating anything.
            let is_disambiguation_refire = self.ai.is_battleship_discriminating_refire(r, c);
            // Same reasoning, for `AiPlayer::anchored_isolation_shot`: its
            // known-Battleship anchor is almost always already fired (the
            // class can't be sunk otherwise), and its cross-exclusive
            // Cruiser/Frigate partner cell is frequently already fired too.
            let is_anchored_isolation_refire = self.ai.is_anchored_isolation_refire(r, c);
            // Same idea one size down, for `AiPlayer::disambiguation_shots_
            // with_refire`'s "heatmap fully evolved" dead end — but unlike
            // the 2 refires above (which are always allowed whenever the
            // underlying strategy calls for them), this one is capped at
            // one bonus use per cell (see `disambiguation_extra_refire_
            // used`), so it's tracked separately below to be marked spent
            // only once the whole salvo actually goes through.
            let is_disambiguation_extra_refire = self.ai.is_disambiguation_extra_refire(r, c);
            // One tier further than the capped bonus refire above — see
            // `AiPlayer::is_last_resort_refire`: only ever true once the
            // capped tier has already come back with nothing, so this can
            // never bypass the one-bonus-per-cell cap while it's still
            // doing useful work, only once it genuinely can't help anymore.
            let is_last_resort_refire = self.ai.is_last_resort_refire(r, c);
            if self.state.fired[r][c] && !refire_ok && !is_disambiguation_refire && !is_anchored_isolation_refire && !is_disambiguation_extra_refire && !is_last_resort_refire && !board_exhausted {
                return r#"{"error":"cell already fired"}"#.to_string();
            }
            if cells_to_fire.iter().any(|&(pr, pc)| pr == r && pc == c) {
                return r#"{"error":"duplicate cell in salvo"}"#.to_string();
            }
            if is_disambiguation_extra_refire {
                extra_refire_cells.push((r, c));
            }
            cells_to_fire.push((r, c));
        }

        let mut results: Vec<usize> = Vec::new();
        let mut sunk_names: Vec<String> = Vec::new();
        let mut sunk_sizes: Vec<usize> = Vec::new();

        for &(r, c) in &cells_to_fire {
            // A cell fired for the first time drives hit-count/sunk bookkeeping;
            // a refire of an already-known cell must NOT do that again — the
            // ship's hit tally and the win condition (hit_count >= total_hits)
            // would otherwise double-count a cell that was only ever hit once.
            let already_fired = self.state.fired[r][c];
            self.state.fired[r][c] = true;
            if let Some(ship_id) = self.state.board[r][c] {
                let ship = &mut self.state.ships[ship_id];
                let size = ship.size;
                if !already_fired {
                    ship.register_hit();
                    if ship.sunk {
                        sunk_names.push(ship.name.clone());
                        sunk_sizes.push(size);
                    }
                    self.state.hit_count += 1;
                }
                results.push(size);
            } else {
                results.push(0);
            }
        }

        // Sort first — the player only sees an unordered bag like "3 2 0".
        // Feed that same sorted bag to the AI so it plays fair, working only
        // from the information the player has.
        results.sort_by(|a, b| b.cmp(a));
        let result_str = results.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(" ");

        let was_battleship_identified = !self.ai.battleship_identified_cells().is_empty();
        let before_ruled_out = self.ai.cross3_ruled_out_snapshot();

        let coords_arr: [(usize, usize); 3] = [cells_to_fire[0], cells_to_fire[1], cells_to_fire[2]];
        let values_arr: [usize; 3] = [results[0], results[1], results[2]];
        self.ai.apply_salvo(coords_arr, values_arr);
        for size in sunk_sizes {
            self.ai.mark_sunk(size);
        }
        for (r, c) in extra_refire_cells {
            self.ai.mark_disambiguation_extra_refire_used(r, c);
        }

        let battleship_discovered =
            !was_battleship_identified && !self.ai.battleship_identified_cells().is_empty();

        let newly_ruled_out_coords: Vec<String> = self
            .ai
            .newly_ruled_out_since(&before_ruled_out)
            .iter()
            .map(|&(r, c)| cell_to_str(&Cell { row: r, col: c }))
            .collect();

        let coords: Vec<String> = indices.iter()
            .map(|&idx| cell_to_str(&Cell { row: idx / 10, col: idx % 10 }))
            .collect();

        let salvo = SalvoResult {
            coords,
            result: result_str,
            sunk_names,
            turn: self.state.turn,
            battleship_discovered,
            newly_ruled_out_coords,
        };

        self.state.log.push(salvo.clone());
        self.state.turn += 1;

        if self.state.hit_count >= self.state.total_hits {
            self.state.won = true;
        }

        serde_json::to_string(&salvo).unwrap_or_else(|_| r#"{"error":"serialization failed"}"#.to_string())
    }

    /// Ask the AI for its next recommended 3 shots. Returns a JSON array of flat
    /// indices (row*10+col), e.g. [23, 45, 67].
    pub fn ai_suggest(&self) -> String {
        let shots = self.ai.choose_shots();
        let indices: Vec<usize> = shots.iter().map(|&(r, c)| r * 10 + c).collect();
        serde_json::to_string(&indices).unwrap_or_else(|_| "[]".to_string())
    }

    /// Debug/inspector: which ship size (4, 3, 2, or 1) `choose_shots` is
    /// currently trying to eliminate — the largest class not yet fully sunk, in
    /// priority order Battleship > Cruiser > Frigate > Submarine.
    pub fn ai_target_size(&self) -> usize {
        self.ai.current_target_size()
    }

    /// Whether the Cruisers are sunk but their exact 2-window layout is
    /// still ambiguous — i.e. `choose_shots` still has genuine
    /// disambiguating work to do for them. See
    /// `AiPlayer::cruiser_disambiguation_pending`.
    pub fn ai_cruiser_disambiguation_pending(&self) -> bool {
        self.ai.cruiser_disambiguation_pending()
    }

    /// Same idea as `ai_cruiser_disambiguation_pending`, one size down.
    pub fn ai_frigate_disambiguation_pending(&self) -> bool {
        self.ai.frigate_disambiguation_pending()
    }

    /// The manual "heatmap fully evolved" workflow's first step: once the
    /// Cruisers are cross-reasoning-identified (`AiPlayer::cruiser_
    /// identified_cells_refined`), lock that layout in, feed it into the
    /// size-3/size-2 FSM, refresh the cross-3/cross-2 bags against it, and
    /// clear the memoized heatmap candidate lists so the next read
    /// reflects the change. See `AiPlayer::update_fsm_and_resolve`. Returns
    /// whether it actually locked anything in (false if already locked, or
    /// if the Cruisers aren't cross-reasoning-identified yet).
    pub fn update_fsm_and_resolve(&mut self) -> bool {
        self.ai.update_fsm_and_resolve()
    }

    /// The manual workflow's second step, for the dead end where every
    /// cell the remaining Cruiser/Frigate hypotheses disagree on has
    /// already been fired: the best disambiguating salvo if one specific
    /// already-fired cell were allowed exactly one bonus refire. Returns a
    /// JSON array of flat indices, same shape as `ai_suggest`, or `[]` if
    /// no such salvo exists (either nothing is ambiguous, or ambiguity
    /// remains but no refire could ever resolve it). See `AiPlayer::
    /// disambiguation_shots_with_refire`; `Game::fire` only accepts an
    /// already-fired cell if it matches this exact suggestion, and only
    /// once per cell.
    pub fn ai_suggest_disambiguation_refire(&self) -> String {
        let shots = self.ai.disambiguation_shots_with_refire();
        let indices: Vec<usize> = match shots {
            Some(cells) => cells.iter().map(|&(r, c)| r * 10 + c).collect(),
            None => Vec::new(),
        };
        serde_json::to_string(&indices).unwrap_or_else(|_| "[]".to_string())
    }

    /// One tier further than `ai_suggest_disambiguation_refire`: for the
    /// genuine dead end where the discriminating cell(s) ALSO already spent
    /// their one-time bonus refire (typically resolving some earlier,
    /// now-settled ambiguity) — a "cluster of 3" Frigate/Cruiser ambiguity
    /// (certain pivot cell, 2 mutually-exclusive end cells) can leave both
    /// ends permanently tied under the normal cap even though a single
    /// clean refire would resolve it outright. Only meaningfully different
    /// from `ai_suggest_disambiguation_refire` once that one has already
    /// returned `[]` — see `AiPlayer::disambiguation_shots_last_resort`;
    /// `Game::fire` only accepts the already-fired cell(s) this suggests
    /// once the capped tier genuinely has nothing left to offer (see
    /// `AiPlayer::is_last_resort_refire`).
    pub fn ai_suggest_disambiguation_last_resort(&self) -> String {
        let shots = self.ai.disambiguation_shots_last_resort();
        let indices: Vec<usize> = match shots {
            Some(cells) => cells.iter().map(|&(r, c)| r * 10 + c).collect(),
            None => Vec::new(),
        };
        serde_json::to_string(&indices).unwrap_or_else(|_| "[]".to_string())
    }

    /// Toggle whether the AI's advisory (`ai_suggest`/`choose_shots`) may
    /// recommend an already-fired cell while hunting `size` (2, 3, or 4).
    /// Manual firing via `fire` is unaffected either way.
    pub fn set_refire_allowed(&mut self, size: usize, allowed: bool) {
        self.ai.set_refire_allowed(size, allowed);
    }

    /// Whether the refire-allowed toggle is currently on for `size`.
    pub fn is_refire_allowed(&self, size: usize) -> bool {
        self.ai.is_refire_allowed(size)
    }

    /// Toggle whether `ai_target_size`/`choose_shots` should freeze at 3
    /// (Cruiser) rather than ever advancing to 2 (Frigate) once every
    /// Cruiser is sunk.
    pub fn set_freeze_before_frigates(&mut self, freeze: bool) {
        self.ai.set_freeze_before_frigates(freeze);
    }

    /// Whether the freeze-before-Frigates toggle is currently on.
    pub fn is_freeze_before_frigates(&self) -> bool {
        self.ai.is_freeze_before_frigates()
    }

    /// Cells that could still hold the Battleship, per the cross-deduction trick:
    /// once a salvo comes back with a 4 in its result bag, the ship must lie along
    /// a length-7 "beam" through one of that salvo's 3 coordinates, and the running
    /// candidate set narrows every time a new 4-bearing salvo intersects with it.
    /// Returns a JSON array of flat indices, or an empty array if no 4 has been
    /// seen yet. Once the Battleship is sunk, this naturally reflects wherever it
    /// was last narrowed down to (no separate "sunk" handling needed here).
    pub fn battleship_candidates_json(&self) -> String {
        let cells = self.ai.battleship_candidate_cells();
        let indices: Vec<usize> = cells.iter().map(|&(r, c)| r * 10 + c).collect();
        serde_json::to_string(&indices).unwrap_or_else(|_| "[]".to_string())
    }

    /// The Battleship's exact 4-cell layout, once 2 or more four-bearing salvos have
    /// narrowed the candidate cross down to a single straight run of exactly 4 cells —
    /// at that point there's nothing left to deduce, only cells left to sink. Returns
    /// a JSON array of flat indices, or an empty array until that deduction lands.
    pub fn battleship_identified_json(&self) -> String {
        let cells = self.ai.battleship_identified_cells();
        let indices: Vec<usize> = cells.iter().map(|&(r, c)| r * 10 + c).collect();
        serde_json::to_string(&indices).unwrap_or_else(|_| "[]".to_string())
    }

    /// The Battleship's exact 4-cell layout, permanently, once confirmed
    /// sunk — flat indices (a single array of 4, or empty if the ship
    /// hasn't sunk yet, or sank via ordinary fire before
    /// `battleship_identified` ever narrowed things down to one window).
    /// Unlike `battleship_identified_json` (a live, hunting-only view that
    /// goes empty the instant the ship sinks, since there's nothing left to
    /// hunt for), this persists so the board keeps showing where the
    /// Battleship was.
    pub fn found_battleship_json(&self) -> String {
        let cells = self.ai.found_battleship_cells();
        let indices: Vec<usize> = cells.iter().map(|&(r, c)| r * 10 + c).collect();
        serde_json::to_string(&indices).unwrap_or_else(|_| "[]".to_string())
    }

    /// The Cruisers' exact 6-cell layout, once the heatmap has narrowed to
    /// a single consistent hypothesis after also cross-checking against the
    /// Frigate candidates — see `AiPlayer::cruiser_identified_cells_refined`.
    /// Empty until then.
    pub fn cruiser_identified_json(&self) -> String {
        let cells = self.ai.cruiser_identified_cells_refined();
        let indices: Vec<usize> = cells.iter().map(|&(r, c)| r * 10 + c).collect();
        serde_json::to_string(&indices).unwrap_or_else(|_| "[]".to_string())
    }

    /// Same idea as `cruiser_identified_json`, one size down — see
    /// `AiPlayer::frigate_identified_cells_refined`. Empty until all 3
    /// Frigates' exact 6-cell layout is pinned down.
    pub fn frigate_identified_json(&self) -> String {
        let cells = self.ai.frigate_identified_cells_refined();
        let indices: Vec<usize> = cells.iter().map(|&(r, c)| r * 10 + c).collect();
        serde_json::to_string(&indices).unwrap_or_else(|_| "[]".to_string())
    }

    /// Debug/inspector: every 3-bearing salvo seen so far — same shape as
    /// `cross2_debug_json`, one ship size up.
    ///
    /// `values` is an unordered bag (see `fire`, which sorts it before handing
    /// it to the AI) — it never tells you *which* coordinate produced the 3.
    /// `true_cruiser_coords` answers that directly from the real board (the same
    /// ground truth `debug_ships_json` reveals), independent of the fog-of-war
    /// model the AI itself has to reason under. It's the subset of `coords` that
    /// actually holds a Cruiser cell — for the UI to highlight, not for any
    /// deduction logic to use.
    pub fn cross3_debug_json(&self) -> String {
        #[derive(Serialize)]
        struct Cross3EntryDebug {
            coords: Vec<usize>,
            values: [usize; 3],
            true_cruiser_coords: Vec<usize>,
            ruled_out_coords: Vec<usize>,
            confirmed_coords: Vec<usize>,
        }
        #[derive(Serialize)]
        struct Cross3Debug {
            entries: Vec<Cross3EntryDebug>,
        }

        let is_cruiser_cell = |r: usize, c: usize| {
            matches!(self.state.board[r][c], Some(id) if self.state.ships[id].size == 3)
        };

        let entries: Vec<Cross3EntryDebug> = self
            .ai
            .cross3_entries()
            .iter()
            .map(|e| Cross3EntryDebug {
                coords: e.coords.iter().map(|&(r, c)| r * 10 + c).collect(),
                values: e.values,
                true_cruiser_coords: e
                    .coords
                    .iter()
                    .filter(|&&(r, c)| is_cruiser_cell(r, c))
                    .map(|&(r, c)| r * 10 + c)
                    .collect(),
                ruled_out_coords: e
                    .coords
                    .iter()
                    .zip(e.coord_ruled_out.iter())
                    .filter(|(_, &ruled_out)| ruled_out)
                    .map(|(&(r, c), _)| r * 10 + c)
                    .collect(),
                confirmed_coords: e
                    .coords
                    .iter()
                    .zip(e.coord_confirmed_cruiser_hit.iter())
                    .filter(|(_, &confirmed)| confirmed)
                    .map(|(&(r, c), _)| r * 10 + c)
                    .collect(),
            })
            .collect();

        serde_json::to_string(&Cross3Debug { entries }).unwrap_or_else(|_| "{}".to_string())
    }

    /// Debug/inspector: every 2-bearing salvo seen so far — same shape as
    /// `cross3_debug_json`, one ship size down.
    pub fn cross2_debug_json(&self) -> String {
        #[derive(Serialize)]
        struct Cross2EntryDebug {
            coords: Vec<usize>,
            values: [usize; 3],
            true_frigate_coords: Vec<usize>,
            ruled_out_coords: Vec<usize>,
            confirmed_coords: Vec<usize>,
        }
        #[derive(Serialize)]
        struct Cross2Debug {
            entries: Vec<Cross2EntryDebug>,
        }

        let is_frigate_cell = |r: usize, c: usize| {
            matches!(self.state.board[r][c], Some(id) if self.state.ships[id].size == 2)
        };

        let entries: Vec<Cross2EntryDebug> = self
            .ai
            .cross2_entries()
            .iter()
            .map(|e| Cross2EntryDebug {
                coords: e.coords.iter().map(|&(r, c)| r * 10 + c).collect(),
                values: e.values,
                true_frigate_coords: e
                    .coords
                    .iter()
                    .filter(|&&(r, c)| is_frigate_cell(r, c))
                    .map(|&(r, c)| r * 10 + c)
                    .collect(),
                ruled_out_coords: e
                    .coords
                    .iter()
                    .zip(e.coord_ruled_out.iter())
                    .filter(|(_, &ruled_out)| ruled_out)
                    .map(|(&(r, c), _)| r * 10 + c)
                    .collect(),
                confirmed_coords: e
                    .coords
                    .iter()
                    .zip(e.coord_confirmed_frigate_hit.iter())
                    .filter(|(_, &confirmed)| confirmed)
                    .map(|(&(r, c), _)| r * 10 + c)
                    .collect(),
            })
            .collect();

        serde_json::to_string(&Cross2Debug { entries }).unwrap_or_else(|_| "{}".to_string())
    }

    /// Debug/inspector: every 4-bearing salvo seen so far — its 3 coordinates,
    /// raw result values, and per-coordinate elimination flag — the cross-4
    /// counterpart of `cross3_debug_json`. `ruled_out_coords` is currently
    /// always empty (every coordinate starts, and stays, green) until the
    /// red-flagging rule for the Battleship's cross-4 salvos is defined.
    pub fn cross4_debug_json(&self) -> String {
        #[derive(Serialize)]
        struct Cross4EntryDebug {
            coords: Vec<usize>,
            values: [usize; 3],
            ruled_out_coords: Vec<usize>,
            confirmed_coords: Vec<usize>,
        }
        #[derive(Serialize)]
        struct Cross4Debug {
            entries: Vec<Cross4EntryDebug>,
        }

        let entries: Vec<Cross4EntryDebug> = self
            .ai
            .cross4_entries()
            .iter()
            .map(|e| Cross4EntryDebug {
                coords: e.coords.iter().map(|&(r, c)| r * 10 + c).collect(),
                values: e.values,
                ruled_out_coords: e
                    .coords
                    .iter()
                    .zip(e.coord_ruled_out.iter())
                    .filter(|(_, &ruled_out)| ruled_out)
                    .map(|(&(r, c), _)| r * 10 + c)
                    .collect(),
                confirmed_coords: e
                    .coords
                    .iter()
                    .zip(e.coord_confirmed_battleship_hit.iter())
                    .filter(|(_, &confirmed)| confirmed)
                    .map(|(&(r, c), _)| r * 10 + c)
                    .collect(),
            })
            .collect();

        serde_json::to_string(&Cross4Debug { entries }).unwrap_or_else(|_| "{}".to_string())
    }

    /// Debug/inspector: the 3 "alive" grids for `size` (4, 3, or 2), each an
    /// 8x8 array (row-major, indices 0..8 corresponding to board rows/cols
    /// 1..8) — horizontal (row FSM value at this column), vertical (column FSM
    /// value at this row), and their sum. A combined value of 0 means no alive
    /// placement of this size, horizontal or vertical, passes through that
    /// cell. For size 3, that's exactly the criterion the cross-3 bags are
    /// pruned against after every salvo (see `AiPlayer::prune_cross3_bags`).
    pub fn alive_grids_json(&self, size: usize) -> String {
        #[derive(Serialize)]
        struct AliveGrids {
            horizontal: Vec<Vec<u32>>,
            vertical: Vec<Vec<u32>>,
            combined: Vec<Vec<u32>>,
        }
        let (horizontal, vertical, combined) = self.ai.alive_grids(size);
        serde_json::to_string(&AliveGrids { horizontal, vertical, combined }).unwrap_or_else(|_| "{}".to_string())
    }

    /// Per-cell probability (0.0-1.0) that a Cruiser occupies it, given
    /// every salvo fired so far, and after also cross-checking each
    /// remaining Cruiser hypothesis against every remaining Frigate
    /// hypothesis for mutual adjacency — see
    /// `AiPlayer::cruiser_heatmap_refined`. Same 8x8 grid convention as
    /// `alive_grids_json`.
    pub fn cruiser_heatmap_json(&self) -> String {
        serde_json::to_string(&self.ai.cruiser_heatmap_refined()).unwrap_or_else(|_| "[]".to_string())
    }

    /// Per-cell probability (0.0-1.0) that a Frigate occupies it — see
    /// `AiPlayer::frigate_heatmap_refined`. Same 8x8 grid convention as
    /// `alive_grids_json`.
    pub fn frigate_heatmap_json(&self) -> String {
        serde_json::to_string(&self.ai.frigate_heatmap_refined()).unwrap_or_else(|_| "[]".to_string())
    }

    /// Same 8x8 grid as `cruiser_heatmap_json`, but each cell is a
    /// `[count, total]` pair instead of the divided probability — for
    /// displaying the underlying fraction directly. See
    /// `AiPlayer::cruiser_heatmap_fraction_refined`.
    pub fn cruiser_heatmap_fraction_json(&self) -> String {
        serde_json::to_string(&self.ai.cruiser_heatmap_fraction_refined()).unwrap_or_else(|_| "[]".to_string())
    }

    /// Same idea as `cruiser_heatmap_fraction_json`, one size down.
    pub fn frigate_heatmap_fraction_json(&self) -> String {
        serde_json::to_string(&self.ai.frigate_heatmap_fraction_refined()).unwrap_or_else(|_| "[]".to_string())
    }

    /// Every inner cell where all 3 combined "alive" values — Battleship,
    /// Cruiser, and Frigate (see `alive_grids_json`) — have dropped to zero:
    /// nothing of size >=2 can occupy it any more, only a Submarine or
    /// nothing. Flat indices, for the main grid to outline distinctly.
    pub fn fully_eliminated_cells_json(&self) -> String {
        let (_, _, combined4) = self.ai.alive_grids(4);
        let (_, _, combined3) = self.ai.alive_grids(3);
        let (_, _, combined2) = self.ai.alive_grids(2);
        let mut cells: Vec<usize> = Vec::new();
        for r in 0..8 {
            for c in 0..8 {
                if combined4[r][c] == 0 && combined3[r][c] == 0 && combined2[r][c] == 0 {
                    // alive_grids is 0-indexed over the inner 8x8; board rows/cols are r+1/c+1.
                    cells.push((r + 1) * 10 + (c + 1));
                }
            }
        }
        serde_json::to_string(&cells).unwrap_or_else(|_| "[]".to_string())
    }

    /// Returns true if the cell (flat index) has already been fired at.
    pub fn is_fired(&self, idx: usize) -> bool {
        let r = idx / 10;
        let c = idx % 10;
        self.state.fired[r][c]
    }

    /// Returns true if the game has been won.
    pub fn is_won(&self) -> bool {
        self.state.won
    }

    /// Current turn number.
    pub fn turn(&self) -> usize {
        self.state.turn
    }

    /// Full JSON snapshot of the log (all SalvoResults).
    pub fn log_json(&self) -> String {
        serde_json::to_string(&self.state.log)
            .unwrap_or_else(|_| "[]".to_string())
    }

    /// JSON array of all ships with sunk status (for fleet tracker).
    pub fn ships_json(&self) -> String {
        #[derive(Serialize)]
        struct ShipInfo<'a> {
            id: usize,
            name: &'a str,
            size: usize,
            sunk: bool,
        }
        let info: Vec<ShipInfo> = self.state.ships.iter().map(|s| ShipInfo {
            id: s.id,
            name: &s.name,
            size: s.size,
            sunk: s.sunk,
        }).collect();
        serde_json::to_string(&info).unwrap_or_else(|_| "[]".to_string())
    }

    /// Debug/inspector view: the true generated board, independent of what's
    /// actually been fired. Reveals every ship's coordinates and sunk status.
    pub fn debug_ships_json(&self) -> String {
        #[derive(Serialize)]
        struct ShipDebugInfo<'a> {
            id: usize,
            name: &'a str,
            size: usize,
            cells: Vec<String>,
            sunk: bool,
        }
        let info: Vec<ShipDebugInfo> = self.state.ships.iter().map(|s| ShipDebugInfo {
            id: s.id,
            name: &s.name,
            size: s.size,
            cells: s.cells.iter().map(cell_to_str).collect(),
            sunk: s.sunk,
        }).collect();
        serde_json::to_string(&info).unwrap_or_else(|_| "[]".to_string())
    }

    /// Debug/inspector view: the AI's current FSM state for a given ship size
    /// (4, 3, or 2), broken out per direction. Each line entry carries its index
    /// (0..9), current FSM state, and the number of still-possible placements
    /// ("alive") in that state. Indices 0 and 9 are the outer ring and never
    /// correspond to a real placement — included for transparency, but the AI's
    /// own shot selection never queries them for size 4 (and by construction
    /// never scores anything for sizes 3/2 there either).
    pub fn fsm_status_json(&self, size: usize) -> String {
        #[derive(Serialize)]
        struct LineFsm {
            index: usize,
            state: usize,
            alive: u8,
        }
        #[derive(Serialize)]
        struct FsmStatus {
            size: usize,
            rows: Vec<LineFsm>,
            cols: Vec<LineFsm>,
        }
        let (row_states, col_states) = self.ai.line_states(size);
        let rows: Vec<LineFsm> = row_states.iter().enumerate()
            .map(|(index, &state)| LineFsm { index, state, alive: AiPlayer::alive_count(size, state) })
            .collect();
        let cols: Vec<LineFsm> = col_states.iter().enumerate()
            .map(|(index, &state)| LineFsm { index, state, alive: AiPlayer::alive_count(size, state) })
            .collect();
        let status = FsmStatus { size, rows, cols };
        serde_json::to_string(&status).unwrap_or_else(|_| "{}".to_string())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------


#[cfg(test)]
mod tests;
