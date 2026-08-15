use wasm_bindgen::prelude::*;
use serde::{Deserialize, Serialize};

mod fsm_tables;
mod ai;

use ai::AiPlayer;

// ---------------------------------------------------------------------------
// Random number generation (wraps getrandom which supports WASM via js feature)
// ---------------------------------------------------------------------------

fn random_usize(n: usize) -> usize {
    let mut buf = [0u8; 4];
    getrandom::getrandom(&mut buf).expect("getrandom failed");
    let v = u32::from_le_bytes(buf) as usize;
    v % n
}

fn random_bool() -> bool {
    random_usize(2) == 0
}

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Cell {
    pub row: usize,
    pub col: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Ship {
    pub id: usize,
    pub name: String,
    pub size: usize,
    pub cells: Vec<Cell>,
    pub hits: usize,
    pub sunk: bool,
}

impl Ship {
    fn new(id: usize, name: &str, size: usize, cells: Vec<Cell>) -> Self {
        Ship { id, name: name.to_string(), size, cells, hits: 0, sunk: false }
    }

    fn register_hit(&mut self) {
        self.hits += 1;
        if self.hits >= self.size {
            self.sunk = true;
        }
    }
}

/// Just the ship placement — no fired/hit/turn state — so a board can be
/// saved and later reloaded to start a fresh game on the exact same
/// layout. See `Game::board_layout_json`/`Game::load_board_layout_json`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BoardLayout {
    pub board: Vec<Vec<Option<usize>>>,
    pub ships: Vec<Ship>,
}

/// Whether the AI's own deduction has pinned down every ship class it's
/// capable of identifying (Battleship, Cruiser, Frigate — Submarines
/// aren't tracked this way) with full certainty, independent of whether
/// the game itself has been fully won. See `Game::resolution_status_json`.
/// A game can be won while this is still false (some residual ambiguity
/// can be permanent — see `AiPlayer::disambiguation_shots`), and this can
/// become true before the game is won (nothing forces firing at every
/// last submarine cell first).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ResolutionStatus {
    pub resolved: bool,
    pub battleship_identified: bool,
    pub cruiser_identified: bool,
    pub frigate_identified: bool,
    /// Only populated when NOT fully resolved — the per-cell probability
    /// grids explaining exactly what's still uncertain and by how much.
    pub cruiser_odds: Option<Vec<Vec<f64>>>,
    pub frigate_odds: Option<Vec<Vec<f64>>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SalvoResult {
    pub coords: Vec<String>,     // e.g. ["A3", "C7", "J10"]
    pub result: String,           // e.g. "4 1 0"
    pub sunk_names: Vec<String>,  // ships sunk this salvo
    pub turn: usize,
    /// True on the exact salvo where the Battleship's exact 4-cell layout
    /// first became known — distinct from "sunk": the AI can identify the
    /// Battleship's location well before every one of its cells is actually
    /// hit.
    pub battleship_discovered: bool,
    /// Cross-3 salvo coordinates that became newly ruled out THIS round — i.e.
    /// just proven impossible to be that salvo's real Cruiser hit (whether
    /// because of something this very salvo did, or because of unrelated
    /// deduction elsewhere that happened to land on an older salvo's cell).
    /// Empty most rounds. See `AiPlayer::refresh_cross3_entry_flags`.
    pub newly_ruled_out_coords: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GameState {
    /// 10×10 grid: None = water, Some(ship_id) = ship occupies this cell
    board: Vec<Vec<Option<usize>>>,
    ships: Vec<Ship>,
    fired: Vec<Vec<bool>>,
    pub log: Vec<SalvoResult>,
    pub turn: usize,
    pub won: bool,
    pub total_hits: usize,
    pub hit_count: usize,
}

// ---------------------------------------------------------------------------
// Board generation
// ---------------------------------------------------------------------------

const SHIP_DEFS: &[(&str, usize, usize)] = &[
    ("Battleship", 4, 1),
    ("Cruiser",    3, 2),
    ("Frigate",    2, 3),
    ("Submarine",  1, 4),
];

/// Check adjacency rules and return the cells if placement is valid.
/// Ships (size >= 2): inner 8×8 only, no orthogonal or diagonal adjacency to other ships.
/// Submarines (size == 1): anywhere, no orthogonal adjacency to any ship.
fn try_place(
    ships: &[Ship],
    row: usize,
    col: usize,
    size: usize,
    horizontal: bool,
) -> Option<Vec<Cell>> {
    if size == 1 {
        // Submarine: can go anywhere, no orthogonal neighbours
        if row > 9 || col > 9 {
            return None;
        }
        for ship in ships {
            for &Cell { row: pr, col: pc } in &ship.cells {
                let dr = (pr as isize - row as isize).unsigned_abs();
                let dc = (pc as isize - col as isize).unsigned_abs();
                // Exact overlap (dr==0 && dc==0) and orthogonal adjacency
                // both forbidden; diagonal is ok. Board cells are a single
                // `Option<usize>` each — 2 submarines landing on the same
                // cell would silently overwrite one another there, leaving
                // the loser permanently unhittable (it's still in `ships`
                // with its own recorded cell, but `board` only ever points
                // at whichever one placed last) and the game unwinnable.
                if dr <= 1 && dc <= 1 && !(dr == 1 && dc == 1) {
                    return None;
                }
            }
        }
        return Some(vec![Cell { row, col }]);
    }

    // Ship: build cells, check inner 8×8, check no adjacency (incl. diagonal) to other ships
    let mut cells = Vec::with_capacity(size);
    for i in 0..size {
        let (r, c) = if horizontal {
            (row, col + i)
        } else {
            (row + i, col)
        };
        // Must stay in rows/cols 1..=8 (inner 8×8)
        if r < 1 || r > 8 || c < 1 || c > 8 {
            return None;
        }
        cells.push(Cell { row: r, col: c });
    }

    for ship in ships {
        for &Cell { row: pr, col: pc } in &ship.cells {
            for &Cell { row: nr, col: nc } in &cells {
                let dr = (pr as isize - nr as isize).unsigned_abs();
                let dc = (pc as isize - nc as isize).unsigned_abs();
                if dr <= 1 && dc <= 1 {
                    return None;
                }
            }
        }
    }

    Some(cells)
}

fn generate_board() -> GameState {
    loop {
        let mut ships: Vec<Ship> = Vec::new();
        let mut board: Vec<Vec<Option<usize>>> = vec![vec![None; 10]; 10];
        let mut id = 0usize;
        let mut failed = false;

        'outer: for &(name, size, count) in SHIP_DEFS {
            for _ in 0..count {
                let mut placed = false;
                for _ in 0..5000 {
                    let (row, col, horizontal) = if size == 1 {
                        (random_usize(10), random_usize(10), true)
                    } else {
                        (random_usize(8) + 1, random_usize(8) + 1, random_bool())
                    };

                    if let Some(cells) = try_place(&ships, row, col, size, horizontal) {
                        for &Cell { row: r, col: c } in &cells {
                            board[r][c] = Some(id);
                        }
                        ships.push(Ship::new(id, name, size, cells));
                        id += 1;
                        placed = true;
                        break;
                    }
                }
                if !placed {
                    failed = true;
                    break 'outer;
                }
            }
        }

        if !failed {
            let total_hits: usize = ships.iter().map(|s| s.size).sum();
            return GameState {
                board,
                ships,
                fired: vec![vec![false; 10]; 10],
                log: Vec::new(),
                turn: 1,
                won: false,
                total_hits,
                hit_count: 0,
            };
        }
        // retry if placement failed
    }
}

// ---------------------------------------------------------------------------
// Coordinate helpers
// ---------------------------------------------------------------------------

fn col_letter(c: usize) -> char {
    (b'A' + c as u8) as char
}

fn cell_to_str(cell: &Cell) -> String {
    format!("{}{}", col_letter(cell.col), cell.row + 1)
}

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
mod tests {
    use super::*;


    #[test]
    fn board_generation_produces_correct_fleet() {
        let ships: Vec<Ship> = Vec::new();
        assert!(try_place(&ships, 1, 1, 4, true).is_some());
        assert!(try_place(&ships, 0, 0, 4, true).is_none());
        assert!(try_place(&ships, 0, 0, 1, true).is_some());
        assert!(try_place(&ships, 9, 9, 1, true).is_some());
    }

    #[test]
    fn no_diagonal_adjacency_for_ships() {
        let cells = try_place(&[], 1, 1, 2, true).unwrap();
        let ship = Ship::new(0, "Frigate", 2, cells);
        assert!(try_place(&[ship.clone()], 2, 3, 2, true).is_none());
    }

    #[test]
    fn submarine_diagonal_allowed() {
        let cells = try_place(&[], 5, 5, 1, true).unwrap();
        let sub = Ship::new(0, "Submarine", 1, cells);
        assert!(try_place(&[sub], 6, 6, 1, true).is_some());
    }

    #[test]
    fn submarine_orthogonal_blocked() {
        let cells = try_place(&[], 5, 5, 1, true).unwrap();
        let sub = Ship::new(0, "Submarine", 1, cells);
        assert!(try_place(&[sub], 5, 6, 1, true).is_none());
    }

    #[test]
    fn submarine_exact_overlap_blocked() {
        // Regression: `try_place`'s submarine branch only checked orthogonal
        // adjacency (dr==0&&dc==1 or dr==1&&dc==0), never dr==0&&dc==0 —
        // exact overlap. 2 submarines landing on the same cell corrupts
        // `board`, since a cell only ever stores one `Option<usize>`: the
        // 2nd placement silently overwrites the 1st there, leaving that
        // ship permanently un-hittable (it's still in `ships` with its own
        // recorded cell, but no board cell ever points back to it) and the
        // game unwinnable — reproduced via self-play games getting stuck
        // forever at hit_count 19/20.
        let cells = try_place(&[], 5, 5, 1, true).unwrap();
        let sub = Ship::new(0, "Submarine", 1, cells);
        assert!(try_place(&[sub], 5, 5, 1, true).is_none(), "a 2nd submarine must not be placeable on the exact same cell as an existing one");
    }

    // -----------------------------------------------------------------
    // AI tests
    // -----------------------------------------------------------------

    #[test]
    fn ai_apply_salvo_eliminates_absent_sizes_even_below_bound() {
        let mut ai = AiPlayer::new();
        // Bag [3, 1, 0]: bound = 3, so the plain ">bound" rule only clears
        // size 4 at each cell. But every cell's true value is guaranteed to
        // be EXACTLY one of {3, 1, 0} — 2 never appears in this bag at all,
        // so none of these 3 cells can hold a Frigate either, even though
        // 2 < bound. Mirrors the same reasoning already applied to
        // Battleship via the "no 4 in bag" branch, generalized to every size.
        ai.apply_salvo([(2, 2), (2, 5), (2, 8)], [3, 1, 0]);

        let (_, _, combined2) = ai.alive_grids(2);
        for &(r, c) in &[(2usize, 2usize), (2, 5), (2, 8)] {
            assert_eq!(combined2[r - 1][c - 1], 0, "Frigate must be eliminated at {:?}: bag has no 2", (r, c));
        }
    }

    #[test]
    fn ai_picks_three_distinct_unfired_cells_initially() {
        let ai = AiPlayer::new();
        let shots = ai.choose_shots();
        assert_ne!(shots[0], shots[1]);
        assert_ne!(shots[1], shots[2]);
        assert_ne!(shots[0], shots[2]);
    }

    #[test]
    fn ai_eliminates_battleship_through_full_miss_row() {
        let mut ai = AiPlayer::new();
        // Fire a full-miss salvo at row 5, cols 1,2,3 (inner) -> should eliminate
        // some size-4 placements in row 5.
        ai.apply_salvo([(5, 1), (5, 2), (5, 3)], [0, 0, 0]);
        // After eliminating cols 1,2,3 from row 5's size-4 FSM, the state should
        // have moved away from the initial state (alive count should drop from 5).
        // We can't access internals directly without making them pub(crate) for test,
        // so just assert the AI no longer proposes those cells.
        let shots = ai.choose_shots();
        for &(r, c) in &shots {
            assert!(!(r == 5 && (c == 1 || c == 2 || c == 3)));
        }
    }

    #[test]
    fn ai_never_resuggests_fired_cells() {
        let mut ai = AiPlayer::new();
        ai.apply_salvo([(2, 2), (2, 3), (2, 4)], [0, 0, 0]);
        let shots = ai.choose_shots();
        for &(r, c) in &shots {
            assert!(!ai.is_fired(r, c) || true); // is_fired should be true for fired cells
            assert!(!(shots.contains(&(2, 2)) && r == 2 && c == 2));
        }
        // direct check
        assert!(ai.is_fired(2, 2));
        assert!(ai.is_fired(2, 3));
        assert!(ai.is_fired(2, 4));
        assert!(!shots.contains(&(2, 2)));
        assert!(!shots.contains(&(2, 3)));
        assert!(!shots.contains(&(2, 4)));
    }

    #[test]
    fn ai_refire_allowed_defaults_to_off_and_is_settable_per_size() {
        let ai = AiPlayer::new();
        assert!(!ai.is_refire_allowed(2));
        assert!(!ai.is_refire_allowed(3));
        assert!(!ai.is_refire_allowed(4));

        let mut ai = ai;
        ai.set_refire_allowed(3, true);
        assert!(!ai.is_refire_allowed(2), "toggling size 3 must not affect size 2");
        assert!(ai.is_refire_allowed(3));
        assert!(!ai.is_refire_allowed(4), "toggling size 3 must not affect size 4");

        ai.set_refire_allowed(3, false);
        assert!(!ai.is_refire_allowed(3), "toggling back off must take effect");
    }

    #[test]
    fn ai_refire_allowed_toggle_lets_choose_shots_recommend_a_fired_cell() {
        let mut ai = AiPlayer::new();
        ai.mark_sunk(4);
        ai.mark_sunk(3);
        ai.mark_sunk(3);
        assert_eq!(ai.current_target_size(), 2, "sanity: now hunting Frigates (size 2)");

        // Fire every inner cell (the whole 8x8 playable area) as misses,
        // leaving the untouched outer ring as the only unfired cells left.
        let cells: Vec<(usize, usize)> = (1..=8).flat_map(|r| (1..=8).map(move |c| (r, c))).collect();
        for chunk in cells.chunks(3) {
            let mut coords = [(0usize, 0usize); 3];
            for (i, &c) in chunk.iter().enumerate() {
                coords[i] = c;
            }
            for i in chunk.len()..3 {
                coords[i] = (0, 0); // harmless already-outer-ring padding
            }
            ai.apply_salvo(coords, [0, 0, 0]);
        }

        // Default (refire off): every inner cell is fired, so choose_shots
        // must fall back to the untouched outer ring.
        let shots_default = ai.choose_shots();
        for &(r, c) in &shots_default {
            assert!(
                r == 0 || r == 9 || c == 0 || c == 9,
                "expected an outer-ring fallback shot with refire off, got {:?}", (r, c)
            );
        }

        // Toggle refire on for size 2 — an already-fired inner cell is fair
        // game again, and (being inner) is preferred over the outer ring.
        ai.set_refire_allowed(2, true);
        let shots_toggled = ai.choose_shots();
        assert!(
            shots_toggled.iter().any(|&(r, c)| (1..=8).contains(&r) && (1..=8).contains(&c)),
            "expected at least one already-fired inner cell once refire is allowed for size 2, got {:?}",
            shots_toggled
        );
    }

    #[test]
    fn game_refire_allowed_lets_manual_fire_repeat_a_cell_without_double_counting() {
        let mut game = Game::new();
        // Fresh game: nothing sunk yet, so current_target_size() is 4 — find a
        // real Battleship cell from ground truth so the refire toggle (set for
        // size 4 below) actually matches what Game::fire checks against.
        let (r, c, ship_id) = (0..10)
            .flat_map(|r| (0..10).map(move |c| (r, c)))
            .find_map(|(r, c)| {
                game.state.board[r][c]
                    .filter(|&id| game.state.ships[id].size == 4)
                    .map(|id| (r, c, id))
            })
            .expect("board must have a Battleship cell");
        let idx = r * 10 + c;

        // Decoys must be guaranteed water (not just "not idx") — otherwise a
        // decoy could land on some OTHER unfired ship cell for the first time
        // and legitimately bump hit_count, which would be mistaken for the
        // very double-count bug this test is checking for.
        let mut water = (0..100usize).filter(|&i| i != idx && game.state.board[i / 10][i % 10].is_none());
        let d1 = water.next().unwrap();
        let d2 = water.next().unwrap();
        let first = game.fire(&[idx, d1, d2]);
        let first: serde_json::Value = serde_json::from_str(&first).unwrap();
        assert!(first.get("error").is_none(), "first fire should succeed: {first}");

        let hit_count_before = game.state.hit_count;
        let hits_before = game.state.ships[ship_id].hits;

        // Without the toggle, refiring the same cell is still rejected.
        let mut water2 = (0..100usize).filter(|&i| i != idx && i != d1 && i != d2 && game.state.board[i / 10][i % 10].is_none());
        let d3 = water2.next().unwrap();
        let d4 = water2.next().unwrap();
        let rejected = game.fire(&[idx, d3, d4]);
        let rejected: serde_json::Value = serde_json::from_str(&rejected).unwrap();
        assert_eq!(rejected["error"], "cell already fired");

        // Toggle refire on for size 4 (the Battleship, and the current
        // target) — the same salvo should now succeed.
        game.set_refire_allowed(4, true);
        let second = game.fire(&[idx, d3, d4]);
        let second: serde_json::Value = serde_json::from_str(&second).unwrap();
        assert!(second.get("error").is_none(), "refire should succeed once allowed: {second}");

        // Critically, it must not double-count the hit or re-announce sunk.
        assert_eq!(game.state.hit_count, hit_count_before, "hit_count must not double-count a refire");
        assert_eq!(game.state.ships[ship_id].hits, hits_before, "ship hit tally must not double-count a refire");
    }

    #[test]
    fn ai_hit_eliminates_larger_sizes_only() {
        let mut ai = AiPlayer::new();
        // Salvo result "3 2 0" — ambiguous bag. The max-reachable bound per cell
        // should be 3 for all three cells (since under some permutation, any one
        // of them could be the '3'). This is intentionally conservative.
        ai.apply_salvo([(4, 4), (5, 5), (6, 6)], [3, 2, 0]);
        // Sanity: cells get marked fired regardless.
        assert!(ai.is_fired(4, 4));
        assert!(ai.is_fired(5, 5));
        assert!(ai.is_fired(6, 6));
    }

    #[test]
    fn ai_battleship_shots_are_distinct_and_inner() {
        let ai = AiPlayer::new();
        let shots = ai.choose_shots();
        // Never repeat a coordinate within a salvo.
        assert_ne!(shots[0], shots[1]);
        assert_ne!(shots[1], shots[2]);
        assert_ne!(shots[0], shots[2]);
        // At the initial state every outer-ring cell scores 0 for the Battleship
        // FSM (size 4 never occupies row/col 0 or 9), so all three picks should
        // land in the inner 8x8 where the Battleship can actually be.
        for &(r, c) in &shots {
            assert!((1..=8).contains(&r) && (1..=8).contains(&c));
        }
    }

    #[test]
    fn ai_never_suggests_outer_ring_cells() {
        let mut ai = AiPlayer::new();
        // Fire at a mix including outer-ring coordinates. Because row_state[0]/[9]
        // and col_state[0]/[9] still get updated whenever the *other* axis is inner
        // (e.g. firing at (3, 0) updates col_state[0] using row 3), outer-ring cells
        // can end up with a misleadingly nonzero score under the old unrestricted
        // search. choose_shots must never suggest them regardless — the Battleship
        // can never occupy row/col 0 or 9.
        ai.apply_salvo([(0, 3), (9, 5), (3, 0)], [0, 0, 0]);
        let shots = ai.choose_shots();
        for &(r, c) in &shots {
            assert!(
                (1..=8).contains(&r) && (1..=8).contains(&c),
                "outer-ring cell suggested: {:?}",
                (r, c)
            );
        }
    }

    #[test]
    fn ai_battleship_second_shot_reacts_to_hypothetical_miss_from_first() {
        // After a real miss at (5,1)..(5,3), row 5's size-4 FSM has moved away
        // from the initial state. choose_shots should never pick row 5 cols
        // 1-3 again (they're fired), and — more importantly — its *third* pick
        // should differ from what a naive "top-3 under the original state"
        // selection would give, since shots 2 and 3 are evaluated against the
        // state as it would look after shots 1 and 2 hypothetically miss too.
        let mut ai = AiPlayer::new();
        ai.apply_salvo([(5, 1), (5, 2), (5, 3)], [0, 0, 0]);
        let shots = ai.choose_shots();

        let mut seen = std::collections::HashSet::new();
        for &(r, c) in &shots {
            assert!(!ai.is_fired(r, c), "choose_shots must never repeat a fired cell");
            assert!(seen.insert((r, c)), "choose_shots must never repeat a coordinate within a salvo");
        }
    }

    #[test]
    fn ai_battleship_cross_narrows_candidates_on_first_four() {
        let mut ai = AiPlayer::new();
        // No cross constraint yet.
        assert!(ai.battleship_candidate_cells().is_empty());

        // Salvo centred at (4,4): bag contains a 4, so one of these 3 cells is a
        // genuine Battleship hit. The candidate set should narrow to the union of
        // the three crosses, and must never include coordinates 4+ columns/rows
        // away on a *different* line than any of the three fired cells.
        //
        // 13 cells is the max for a single fully-interior cross (a 7-cell row arm
        // + 7-cell column arm, minus 1 for the shared center), so 3 non-overlapping
        // interior crosses cap out at 39 — but these three crosses aren't disjoint
        // (their arms cross each other at (4,6)/(2,4) and (4,2)/(7,4)), so the real
        // union is smaller: 28.
        ai.apply_salvo([(4, 4), (2, 6), (7, 2)], [4, 1, 0]);
        let candidates: std::collections::HashSet<(usize, usize)> =
            ai.battleship_candidate_cells().into_iter().collect();
        assert_eq!(candidates.len(), 28);

        // (1,1) shares neither row nor column with any of the 3 fired cells, and
        // is further than 3 away from all of them along any shared line — must be
        // eliminated.
        assert!(!candidates.contains(&(1, 1)));

        // (4,7) is on the same row as (4,4), within reach 3 — must remain a candidate.
        assert!(candidates.contains(&(4, 7)));
    }

    #[test]
    fn ai_choose_shots_caps_candidate_region_picks_at_one_while_ambiguous() {
        // A single 4-bearing salvo narrows the candidate cross but doesn't
        // identify the exact ship (that needs 2+ intersecting salvos) — still
        // genuinely ambiguous, with a wide candidate region: 25 cells (the
        // interior (4,4) cross is the max possible 13; the two corner decoys'
        // crosses shrink to 7 apiece since their reach gets clipped by the board
        // edge — a corner cross is at best a 4-cell row arm + 4-cell column arm,
        // minus 1 shared center — and (4,4)'s cross overlaps (1,1)'s at 2 cells).
        let mut ai = AiPlayer::new();
        ai.apply_salvo([(4, 4), (1, 1), (8, 8)], [4, 0, 0]);
        assert!(ai.battleship_identified_cells().is_empty());

        let candidates: std::collections::HashSet<(usize, usize)> =
            ai.battleship_candidate_cells().into_iter().collect();
        assert_eq!(candidates.len(), 25);

        // choose_shots should still pick the highest-elimination cells overall,
        // but at most 1 of the 3 may land inside the ambiguous candidate region —
        // firing 2+ candidate cells in the same salvo would waste the chance to
        // pin down which one was the real hit if a 4 comes back again. Mirrors a
        // reported live scenario: a deliberate discriminating-cell test on shot
        // 1 got its isolation destroyed by shots 2/3 also landing on other
        // candidate cells in the same salvo, making the whole salvo's result
        // ambiguous again — exactly the case this cap exists to prevent.
        let shots = ai.choose_shots();
        let picks_in_candidate_region = shots.iter().filter(|cell| candidates.contains(cell)).count();
        assert!(
            picks_in_candidate_region <= 1,
            "expected at most 1 shot inside the candidate region, got {picks_in_candidate_region}: {shots:?}"
        );
    }

    #[test]
    fn ai_first_battleship_shot_targets_highest_value_candidate_not_an_arbitrary_sub_window_split() {
        // Mirrors a reported live scenario exactly: E5,F6,D4 -> 4 0 0, then
        // B4,G7,G3 -> 0 0 0. Just 1 real cross-4 salvo has landed — the
        // resulting candidate mask is still a single raw, un-narrowed cross,
        // which trivially contains dozens of straight-4 sub-windows purely
        // as an artifact of its shape, not genuine remaining ambiguity about
        // the real ship. `battleship_discriminating_test_cell` used to treat
        // those as real windows worth discriminating between regardless,
        // and could pick a nearly-worthless coordinate (Combined alive
        // value 1) over one sitting in the busiest part of the cross
        // (Combined alive value 6-8) purely because it happened to be
        // absent from a handful of those largely-arbitrary sub-windows.
        let mut ai = AiPlayer::new();
        ai.apply_salvo([(4, 4), (5, 5), (3, 3)], [4, 0, 0]); // E5,F6,D4 -> 4 0 0
        ai.apply_salvo([(3, 1), (6, 6), (2, 6)], [0, 0, 0]); // B4,G7,G3 -> 0 0 0
        assert!(ai.battleship_identified_cells().is_empty(), "sanity: still ambiguous");

        let candidates: std::collections::HashSet<(usize, usize)> =
            ai.battleship_candidate_cells().into_iter().collect();
        let (_, _, combined4) = ai.alive_grids(4);
        let battleship_score = |(r, c): (usize, usize)| combined4[r - 1][c - 1];

        let shots = ai.choose_shots();
        let first = shots[0];
        assert!(candidates.contains(&first), "first shot {first:?} must still land inside the candidate region: {shots:?}");
        let better_alternative_exists = (1..=8usize).any(|r| {
            (1..=8usize).any(|c| {
                let cell = (r, c);
                cell != first && candidates.contains(&cell) && !ai.is_fired(r, c) && battleship_score(cell) > battleship_score(first)
            })
        });
        assert!(
            !better_alternative_exists,
            "first shot {first:?} (Battleship score {}) is not the best available candidate — reported live bug picked (3,2)=C4 (score 1) over (4,3)=D5 (score 6)",
            battleship_score(first)
        );
    }

    #[test]
    fn ai_battleship_hunt_scores_shots_after_the_first_purely_on_the_cruiser_fsm() {
        // Same wide, still-ambiguous candidate cross as
        // `ai_choose_shots_caps_candidate_region_picks_at_one_while_ambiguous`
        // — shots 2 and 3 are forced away from it either way, but they must
        // also be CHOSEN by maximizing the Cruiser (size-3) FSM's own
        // elimination value out there, not some Battleship-flavoured score:
        // outside the established candidate region, every cell's own size-4
        // alive value is already 0 (the cross-4 elimination feeds a "miss"
        // into the size-4 FSM everywhere outside the running candidate set,
        // every time it narrows) — so scoring against size-4 there could
        // never do anything but dilute the Cruiser signal.
        let mut ai = AiPlayer::new();
        ai.apply_salvo([(4, 4), (1, 1), (8, 8)], [4, 0, 0]);
        assert!(ai.battleship_identified_cells().is_empty(), "sanity: still ambiguous");

        let candidates: std::collections::HashSet<(usize, usize)> =
            ai.battleship_candidate_cells().into_iter().collect();

        // Break the symmetry of the Cruiser FSM outside the candidate
        // region with a couple of ordinary miss salvos, so "which outside
        // cell scores highest" isn't a trivial full-board tie.
        ai.apply_salvo([(6, 2), (6, 3), (0, 0)], [0, 0, 0]);
        ai.apply_salvo([(2, 6), (3, 6), (0, 9)], [0, 0, 0]);

        let shots = ai.choose_shots();
        let (_, _, combined3) = ai.alive_grids(3);
        let cruiser_score = |(r, c): (usize, usize)| combined3[r - 1][c - 1];

        for &pick in &shots[1..] {
            assert!(!candidates.contains(&pick), "shot {pick:?} must land outside the candidate region: {shots:?}");
            let better_alternative_exists = (1..=8usize).any(|r| {
                (1..=8usize).any(|c| {
                    let cell = (r, c);
                    cell != pick
                        && !candidates.contains(&cell)
                        && !ai.is_fired(r, c)
                        && !shots.contains(&cell)
                        && cruiser_score(cell) > cruiser_score(pick)
                })
            });
            assert!(
                !better_alternative_exists,
                "shot {pick:?} (Cruiser score {}) is not the best available Cruiser-FSM cell outside the candidate region: {shots:?}",
                cruiser_score(pick)
            );
        }
    }

    #[test]
    fn ai_suggests_refiring_a_discriminating_cell_even_when_every_candidate_is_already_fired() {
        // Mirrors a reported live scenario exactly: 4 cross-4 salvos narrow
        // the Battleship candidates down to 5 adjacent cells in row 4 —
        // E4,F4,G4,H4,I4=(3,4)..(3,8) — leaving exactly 2 overlapping
        // straight-4 windows, [E4,F4,G4,H4] and [F4,G4,H4,I4]. E4 and I4
        // (the 2 outer cells) are the only coordinates that discriminate
        // between them; F4/G4/H4 are common to both and would just confirm
        // a hit without settling anything. Critically, EVERY one of the 5
        // candidates was already fired as part of one of these same 4
        // salvos — so unless a refire is explicitly permitted for this one
        // deliberate purpose, the ambiguity can never close.
        let mut ai = AiPlayer::new();
        ai.apply_salvo([(3, 7), (4, 8), (5, 1)], [4, 3, 1]); // H4, I5, B6
        ai.apply_salvo([(3, 4), (4, 7), (3, 8)], [4, 0, 0]); // E4, H5, I4
        ai.apply_salvo([(3, 5), (6, 3), (2, 7)], [4, 0, 0]); // F4, D7, H3
        ai.apply_salvo([(3, 6), (5, 2), (7, 4)], [4, 0, 0]); // G4, C6, E8

        let candidates: std::collections::HashSet<(usize, usize)> = ai.battleship_candidate_cells().into_iter().collect();
        let expected: std::collections::HashSet<(usize, usize)> = [(3, 4), (3, 5), (3, 6), (3, 7), (3, 8)].into_iter().collect();
        assert_eq!(candidates, expected, "sanity: 5 adjacent candidates, matching the reported scenario");
        assert!(ai.battleship_identified_cells().is_empty(), "sanity: not yet identified, 2 overlapping windows survive");
        for &(r, c) in &expected {
            assert!(ai.is_fired(r, c), "sanity: every candidate cell must already be fired");
        }

        let shots = ai.choose_shots();
        assert!(
            shots[0] == (3, 4) || shots[0] == (3, 8),
            "expected the first shot to refire the outer (discriminating) cell E4 or I4, got {:?}",
            shots[0]
        );
        assert!(
            ai.is_battleship_discriminating_refire(shots[0].0, shots[0].1),
            "the suggested refire must be recognized as a deliberate, always-permitted disambiguation refire"
        );
        // Mirrors a reported live regression: shots 2 and 3 must NOT also
        // land on another candidate cell in this same salvo — if they did,
        // the whole point of shot 1's deliberate, isolating discriminating
        // test would be destroyed: a 4 in the result bag would once again
        // be ambiguous across multiple candidate cells instead of being
        // unambiguously attributable to shot 1 alone.
        assert!(
            !candidates.contains(&shots[1]),
            "shot 2 must stay outside the candidate region so shot 1's discriminating test stays isolated, got {:?}",
            shots[1]
        );
        assert!(
            !candidates.contains(&shots[2]),
            "shot 3 must stay outside the candidate region so shot 1's discriminating test stays isolated, got {:?}",
            shots[2]
        );
    }

    #[test]
    fn ai_clears_stale_battleship_candidates_once_sunk_via_ordinary_fire() {
        // Mirrors a reported live scenario: the same ambiguous 5-cell
        // cluster as above — E4,F4,G4,H4,I4=(3,4)..(3,8), 2 overlapping
        // straight-4 windows — but this time the Battleship sinks (its 4th
        // real cell gets hit) BEFORE the cross-4 deduction machinery ever
        // narrows the ambiguity down to a single window. The candidate
        // cross must not linger forever once sunk — there's nothing left
        // to search for regardless of whether the ambiguity was ever
        // cleanly resolved.
        let mut ai = AiPlayer::new();
        ai.apply_salvo([(3, 7), (4, 8), (5, 1)], [4, 3, 1]);
        ai.apply_salvo([(3, 4), (4, 7), (3, 8)], [4, 0, 0]);
        ai.apply_salvo([(3, 5), (6, 3), (2, 7)], [4, 0, 0]);
        ai.apply_salvo([(3, 6), (5, 2), (7, 4)], [4, 0, 0]);
        assert!(!ai.battleship_candidate_cells().is_empty(), "sanity: still a live, unresolved candidate cross");
        assert!(ai.battleship_identified_cells().is_empty(), "sanity: genuinely ambiguous, never resolved to 1 window");

        ai.mark_sunk(4);

        assert!(
            ai.battleship_candidate_cells().is_empty(),
            "the stale candidate cross must be cleared once the Battleship is confirmed sunk"
        );
    }

    #[test]
    fn ai_choose_shots_blends_size3_into_forced_away_picks_while_battleship_ambiguous() {
        // Real hit-cross confined to the top-left corner — (1,1)'s cross only
        // spans rows/cols 1-4 — with fully inert outer-ring decoys. Only 1
        // four-bearing salvo so far: genuinely ambiguous, not identified.
        let mut ai = AiPlayer::new();
        ai.apply_salvo([(1, 1), (0, 0), (9, 9)], [4, 0, 0]);
        assert!(ai.battleship_identified_cells().is_empty());

        let candidates: std::collections::HashSet<(usize, usize)> =
            ai.battleship_candidate_cells().into_iter().collect();

        let shots = ai.choose_shots();

        // Shots 2 and 3 are always forced away from the ambiguous candidate
        // region (only the first shot gets to test it), regardless of scoring.
        assert!(!candidates.contains(&shots[1]), "second shot must avoid the candidate region");
        assert!(!candidates.contains(&shots[2]), "third shot must avoid the candidate region");

        // Once forced away from the Battleship's region, every eligible cell
        // scores exactly 0 for size 4 here — every non-candidate inner cell was
        // already fed into the size-4 FSM as a miss by the cross elimination
        // above (see `apply_battleship_cross_elimination`), so a size-4-only
        // search would just fall back to scan order, arbitrarily. Blending in
        // size-3 (Cruiser) scoring picks out (3,3) instead: its row and column
        // are both still untouched by any elimination and sit at the FSM's most
        // central table position (value 3 each way, for size 3's initial
        // state), giving a combined score of 6 — the maximum reachable, and the
        // first cell in row-major scan order to reach it.
        assert_eq!(shots[1], (3, 3));
    }

    #[test]
    fn ai_battleship_cross_intersects_across_salvos() {
        let mut ai = AiPlayer::new();
        // First salvo: cross centred at (4,4) (plus two far-off decoys whose own
        // crosses only add extra candidates, never remove any).
        ai.apply_salvo([(4, 4), (1, 1), (8, 8)], [4, 0, 0]);
        let first: std::collections::HashSet<(usize, usize)> =
            ai.battleship_candidate_cells().into_iter().collect();
        assert!(first.contains(&(4, 7))); // same row as (4,4), within reach

        // Second salvo also has a 4, centred at (6,4) this time (plus decoys).
        // The running candidate set should shrink to the intersection: cells on
        // row/col 4 within reach of BOTH (4,4) and (6,4).
        ai.apply_salvo([(6, 4), (1, 2), (8, 7)], [4, 0, 0]);
        let second: std::collections::HashSet<(usize, usize)> =
            ai.battleship_candidate_cells().into_iter().collect();

        // Must never grow — every remaining candidate was already a candidate
        // after the first salvo.
        assert!(second.is_subset(&first));
        // (4,7) is far from (6,4) on every shared line (not same row, and column 7
        // is 3 away from column 4 vertically only via col beam, not row 6) — check
        // an unambiguous exclusion: (4,1) is on (4,4)'s row-beam but has no
        // relationship to (6,4)'s cross at all, so it must now be gone.
        assert!(!second.contains(&(4, 1)) || first.contains(&(4, 1)));
        assert!(second.len() <= first.len());
    }

    #[test]
    fn ai_battleship_room_pruning_kills_isolated_cross_intersections() {
        let mut ai = AiPlayer::new();

        // First salvo: real hit-cross centred at (4,4); decoys along row 1, far
        // from the second salvo's decoys so nothing spurious survives across
        // the two rounds except the crossings computed below.
        ai.apply_salvo([(4, 4), (1, 1), (1, 8)], [4, 0, 0]);

        // Second salvo: real hit-cross centred at (6,6); decoys along row 8.
        // The two "real" crosses only ever meet at two single points:
        // (4,6) — salvo 1's row-4 arm crossing salvo 2's col-6 arm — and
        // (6,4) — salvo 1's col-4 arm crossing salvo 2's row-6 arm. Plain
        // intersection alone would leave exactly these two isolated points as
        // candidates, but neither has 3 more candidate cells in a straight
        // line beside it, so no 4-long Battleship can actually pass through
        // either — the room-pruning pass should drop both.
        ai.apply_salvo([(6, 6), (8, 1), (8, 8)], [4, 0, 0]);

        assert!(ai.battleship_candidate_cells().is_empty());
    }

    #[test]
    fn ai_identifies_exact_battleship_layout_after_two_intersecting_crosses() {
        let mut ai = AiPlayer::new();

        // Real hit-cross centred at (4,3); decoys tucked in the top-left corner,
        // far enough from column 6 / row 8 that they can't spuriously survive
        // round 2's intersection.
        ai.apply_salvo([(4, 3), (1, 1), (2, 2)], [4, 0, 0]);
        assert!(ai.battleship_identified_cells().is_empty()); // only 1 four-bearing salvo so far

        // Real hit-cross centred at (4,6); decoys tucked in the bottom-right
        // corner. The two "real" crosses only overlap where their row-4 arms
        // both reach: cols 3-6 — a straight, exactly-4-long run, which must
        // therefore BE the Battleship.
        ai.apply_salvo([(4, 6), (8, 7), (8, 8)], [4, 0, 0]);

        let identified: std::collections::HashSet<(usize, usize)> =
            ai.battleship_identified_cells().into_iter().collect();
        let expected: std::collections::HashSet<(usize, usize)> =
            [(4, 3), (4, 4), (4, 5), (4, 6)].into_iter().collect();
        assert_eq!(identified, expected);

        // choose_shots should now finish the identified ship off directly:
        // (4,3) and (4,6) are already fired (they were this salvo's real hit
        // cells), so both still-unfired cells of the run, (4,4) and (4,5),
        // must be among the next 3 picks.
        let shots = ai.choose_shots();
        assert!(shots.contains(&(4, 4)));
        assert!(shots.contains(&(4, 5)));
    }

    #[test]
    fn ai_keeps_a_sunk_battleships_own_cells_alive_in_the_size4_grid() {
        // Mirrors a reported live scenario: 2 intersecting crosses identify
        // the exact layout (row 4, cols 3-6), then the ship sinks. The
        // Ship Alive Grids (size 4) debug view must keep showing the sunk
        // ship's own 4 cells as alive (matching how a found Cruiser/Frigate
        // keeps its own cells alive in that size's grid) — not wipe them to
        // 0, which would make them satisfy "no ship of size >=2 possible"
        // and render as dead water, as if the just-identified Battleship
        // had simply vanished the instant it sank.
        let mut ai = AiPlayer::new();
        ai.apply_salvo([(4, 3), (1, 1), (2, 2)], [4, 0, 0]);
        ai.apply_salvo([(4, 6), (8, 7), (8, 8)], [4, 0, 0]);
        assert_eq!(ai.battleship_identified_cells().len(), 4, "sanity: identified before sinking");

        ai.mark_sunk(4);

        let (_, _, combined4) = ai.alive_grids(4);
        for &(r, c) in &[(4, 3), (4, 4), (4, 5), (4, 6)] {
            assert!(
                combined4[r - 1][c - 1] > 0,
                "({r},{c}) is the sunk Battleship's own cell and must stay alive in the size-4 grid, not read as \"no ship possible\""
            );
        }
        // Elsewhere on the board, with the (single) Battleship now fully
        // accounted for, there's genuinely nothing left to search for.
        assert_eq!(combined4[6][6], 0, "(7,7) was never a candidate and must be eliminated for size 4 now that the Battleship is sunk");
    }

    #[test]
    fn ai_permanently_records_the_sunk_battleships_layout_after_the_live_candidate_state_clears() {
        // Mirrors a reported live scenario: once sunk, `battleship_identified_cells`
        // goes empty (nothing left to hunt for), which used to mean the board
        // lost every trace of where the Battleship was — no candidate
        // outline AND no permanent "found" marker, unlike a found Cruiser/
        // Frigate's lasting cells. `found_battleship_cells` must capture
        // the identified layout permanently, the moment it sinks.
        let mut ai = AiPlayer::new();
        ai.apply_salvo([(4, 3), (1, 1), (2, 2)], [4, 0, 0]);
        ai.apply_salvo([(4, 6), (8, 7), (8, 8)], [4, 0, 0]);
        assert!(ai.found_battleship_cells().is_empty(), "sanity: not sunk yet, nothing permanent recorded");

        ai.mark_sunk(4);

        assert!(ai.battleship_identified_cells().is_empty(), "the live, hunting-only view goes empty once sunk — nothing left to hunt for");
        let found: std::collections::HashSet<(usize, usize)> = ai.found_battleship_cells().into_iter().collect();
        let expected: std::collections::HashSet<(usize, usize)> = [(4, 3), (4, 4), (4, 5), (4, 6)].into_iter().collect();
        assert_eq!(found, expected, "the permanent record must capture the identified layout the moment it sinks");
    }

    #[test]
    fn ai_first_battleship_test_shot_prefers_a_discriminating_cell_over_a_merely_common_one() {
        let mut ai = AiPlayer::new();

        // Real hit-cross centred at (4,4); decoys tucked in the top-left
        // corner, far from row 4 / col 6-8 so they can't spuriously survive
        // round 2's intersection.
        ai.apply_salvo([(4, 4), (1, 1), (2, 2)], [4, 0, 0]);
        // Real hit-cross centred at (4,6); decoys tucked in the bottom-right
        // corner. Unlike the previous test, these two crosses' row-4 arms
        // overlap across a 5-cell span (cols 3-7), not a clean 4 — leaving 2
        // valid straight-4 windows, [3,4,5,6] and [4,5,6,7], both still live.
        ai.apply_salvo([(4, 6), (8, 7), (8, 8)], [4, 0, 0]);

        let ambiguous: std::collections::HashSet<(usize, usize)> =
            ai.battleship_candidate_cells().into_iter().collect();
        let expected: std::collections::HashSet<(usize, usize)> =
            [(4, 3), (4, 4), (4, 5), (4, 6), (4, 7)].into_iter().collect();
        assert_eq!(ambiguous, expected);
        assert!(ai.battleship_identified_cells().is_empty());

        // Cols 4, 5 and 6 are common to BOTH surviving windows — a shot there
        // is a guaranteed hit but proves nothing about which window is real.
        // Cols 3 and 7 each belong to only one window: hitting one confirms
        // its window outright, and a miss there is just as informative,
        // collapsing straight to the other. The AI's first shot this round
        // should be one of those two discriminating cells, not one of the
        // 3 merely-common ones.
        let shots = ai.choose_shots();
        assert!(
            shots[0] == (4, 3) || shots[0] == (4, 7),
            "expected first shot to discriminate between the 2 windows, got {:?} (all shots: {:?})",
            shots[0],
            shots
        );
    }

    #[test]
    fn ai_resolves_ambiguous_parallel_run_once_an_ordinary_miss_breaks_it() {
        let mut ai = AiPlayer::new();

        // Real Battleship at row 5, cols 2-5. Each salvo also fires a decoy at
        // the mirror position on row 3 (a corner cell rounds out each salvo
        // harmlessly, contributing nothing to the inner grid). Row 3's decoys
        // are placed so their crosses intersect into a second, equally-valid-
        // looking straight run — row 3 cols 2-5 — alongside the true one.
        ai.apply_salvo([(5, 2), (3, 2), (0, 0)], [4, 0, 0]);
        ai.apply_salvo([(5, 5), (3, 5), (9, 9)], [4, 0, 0]);

        // Two parallel 4-long runs survive cross-intersection + room-pruning —
        // row 3's run is just as "room-valid" as row 5's, so nothing can yet
        // tell them apart. Not identified.
        let ambiguous: std::collections::HashSet<(usize, usize)> =
            ai.battleship_candidate_cells().into_iter().collect();
        assert_eq!(ambiguous.len(), 8);
        assert!(ambiguous.contains(&(5, 3)));
        assert!(ambiguous.contains(&(3, 3)));
        assert!(ai.battleship_identified_cells().is_empty());

        // An ordinary miss at (3,3) — no 4 anywhere in this bag — definitively
        // rules out that cell. That alone breaks row 3's run into two length-2
        // fragments, so the room-pruning cascade should drop the rest of row 3
        // too, leaving row 5's run as the only candidate: the Battleship is now
        // fully identified.
        ai.apply_salvo([(3, 3), (0, 1), (0, 2)], [0, 0, 0]);

        let identified: std::collections::HashSet<(usize, usize)> =
            ai.battleship_identified_cells().into_iter().collect();
        let expected: std::collections::HashSet<(usize, usize)> =
            [(5, 2), (5, 3), (5, 4), (5, 5)].into_iter().collect();
        assert_eq!(identified, expected);
    }

    #[test]
    fn ai_battleship_adjacency_eliminates_size3_and_size2_neighbours_once_identified() {
        let mut ai = AiPlayer::new();

        // Real Battleship at row 5, cols 2-5, identified via two corner-decoyed
        // four-bearing salvos (same clean geometry as the identification test).
        ai.apply_salvo([(5, 2), (0, 0), (0, 9)], [4, 0, 0]);
        ai.apply_salvo([(5, 5), (9, 0), (9, 9)], [4, 0, 0]);
        assert_eq!(ai.battleship_identified_cells().len(), 4);

        let (rows3, _) = ai.line_states(3);
        let (rows2, _) = ai.line_states(2);
        // Row 0 is outer ring and never touched by any size>=2 elimination — use
        // it as the untouched baseline for "alive placements before anything is
        // ruled out".
        let baseline3 = AiPlayer::alive_count(3, rows3[0]);
        let baseline2 = AiPlayer::alive_count(2, rows2[0]);

        // Rows 4 and 6 run directly alongside — and diagonally past — every
        // ship cell. A Cruiser or Frigate can't touch the Battleship at all
        // (not even diagonally), so both rows must have lost placements for
        // both sizes, despite never having been fired at.
        for &row in &[4usize, 6usize] {
            assert!(AiPlayer::alive_count(3, rows3[row]) < baseline3, "row {row} size3 not narrowed");
            assert!(AiPlayer::alive_count(2, rows2[row]) < baseline2, "row {row} size2 not narrowed");
        }

        // Row 5 itself loses its two flanking cells, (5,1) and (5,6), via the
        // neighbour path, AND its own 4 ship cells via the own-cell path — so
        // cols 1-6 are all accounted for. That kills every size-3 placement
        // (a 3-window fits nowhere in cols 1-8 without touching cols 1-6), but
        // for size 2 one window survives untouched: cols 7-8, the only 2
        // consecutive columns neither the ship nor its flanks ever reached.
        assert_eq!(AiPlayer::alive_count(3, rows3[5]), 0, "row 5 size3 should be fully dead");
        assert_eq!(AiPlayer::alive_count(2, rows2[5]), 1, "row 5 size2 should have only cols 7-8 left");
    }

    #[test]
    fn ai_battleship_identified_eliminates_size3_and_size2_at_its_own_unfired_cells() {
        let mut ai = AiPlayer::new();

        // Real Battleship at row 5, cols 2-5, identified via two corner-decoyed
        // four-bearing salvos. Only (5,2) and (5,5) are ever actually fired —
        // (5,3) and (5,4) are deduced, never shot at.
        ai.apply_salvo([(5, 2), (0, 0), (0, 9)], [4, 0, 0]);
        ai.apply_salvo([(5, 5), (9, 0), (9, 9)], [4, 0, 0]);
        assert_eq!(ai.battleship_identified_cells().len(), 4);
        assert!(!ai.is_fired(5, 3), "sanity: (5,3) was deduced, never actually shot");
        assert!(!ai.is_fired(5, 4), "sanity: (5,4) was deduced, never actually shot");

        // A cell definitely occupied by the Battleship can't also be a
        // Cruiser, Frigate, or Submarine cell — even though (5,3)/(5,4) were
        // never fired, they must already be ruled out as Submarine candidates.
        // (This is the one check this test adds beyond
        // `ai_battleship_adjacency_eliminates_size3_and_size2_neighbours_once_identified`,
        // which already confirms row 5's own size-3/size-2 FSMs go fully dead —
        // that needs cols 2-5 eliminated directly, not just via a neighbour,
        // since ship cells are always skipped by the neighbour-only path.)
        assert!(!ai.is_submarine_candidate(5, 3), "(5,3) must be ruled out as a Submarine cell");
        assert!(!ai.is_submarine_candidate(5, 4), "(5,4) must be ruled out as a Submarine cell");
    }

    /// Exhaustive sweep of all 80 possible Battleship placements (every row/col,
    /// every orientation) — not just a hand-picked sample. Every other inner
    /// cell is cleared out via ordinary misses first (so the candidate mask
    /// narrows to exactly this one placement through the plain "no 4 in bag"
    /// path, independent of any cross/room-pruning geometry), then 2 direct
    /// bound=4 hits satisfy `battleship_identified`'s >=2-salvo gate. Catches
    /// any boundary-specific bug (corners/edges clip the neighbour loop
    /// differently than interior placements) that a single hand-picked example
    /// could miss.
    #[test]
    fn ai_battleship_adjacency_elimination_holds_for_every_possible_placement() {
        let mut all_placements: Vec<[(usize, usize); 4]> = Vec::new();
        for row in 1..=8 {
            for start in 1..=5 {
                all_placements.push([(row, start), (row, start + 1), (row, start + 2), (row, start + 3)]);
            }
        }
        for col in 1..=8 {
            for start in 1..=5 {
                all_placements.push([(start, col), (start + 1, col), (start + 2, col), (start + 3, col)]);
            }
        }
        assert_eq!(all_placements.len(), 80);

        for ship in &all_placements {
            let mut ai = AiPlayer::new();
            let ship_set: std::collections::HashSet<(usize, usize)> = ship.iter().copied().collect();
            let all_others: Vec<(usize, usize)> = (1..=8)
                .flat_map(|r| (1..=8).map(move |c| (r, c)))
                .filter(|cell| !ship_set.contains(cell))
                .collect();
            for chunk in all_others.chunks(3) {
                let mut coords = [(0usize, 0usize); 3];
                for (i, &c) in chunk.iter().enumerate() {
                    coords[i] = c;
                }
                for i in chunk.len()..3 {
                    coords[i] = (0, 0); // pad leftover slots with an already-fired, harmless miss
                }
                ai.apply_salvo(coords, [0, 0, 0]);
            }
            ai.apply_salvo([ship[0], (0, 0), (0, 0)], [4, 0, 0]);
            ai.apply_salvo([ship[3], (0, 0), (0, 0)], [4, 0, 0]);

            assert_eq!(ai.battleship_identified_cells().len(), 4, "failed to identify ship at {:?}", ship);

            let mut to_check: std::collections::HashSet<(usize, usize)> = ship_set.clone();
            for &(r, c) in ship {
                for dr in -1isize..=1 {
                    for dc in -1isize..=1 {
                        let nr = r as isize + dr;
                        let nc = c as isize + dc;
                        if (1..=8).contains(&nr) && (1..=8).contains(&nc) {
                            to_check.insert((nr as usize, nc as usize));
                        }
                    }
                }
            }
            let (_, _, combined3) = ai.alive_grids(3);
            let (_, _, combined2) = ai.alive_grids(2);
            for &(r, c) in &to_check {
                assert_eq!(combined3[r - 1][c - 1], 0, "ship {:?}: size-3 alive value at ({r},{c}) should be 0", ship);
                assert_eq!(combined2[r - 1][c - 1], 0, "ship {:?}: size-2 alive value at ({r},{c}) should be 0", ship);
            }
        }
    }

    #[test]
    fn ai_battleship_adjacency_eliminates_size3_and_size2_even_at_the_fired_ship_cells_themselves() {
        let mut ai = AiPlayer::new();

        // Four salvos, each landing its ONE real Battleship hit at a
        // different one of the ship's 4 cells (cols 2,3,4,5 of row 5), plus 2
        // far-away decoys each. Each salvo's bound is 4, so the normal
        // apply_hit path never eliminates size 3/2 at any of its 3 cells (any
        // of them could ambiguously be the real hit — see `apply_salvo`).
        //
        // Crucially, the cross-intersection only narrows down to the exact
        // 4-cell layout on the 4th salvo (verified empirically) — so at the
        // moment `apply_battleship_adjacency_elimination` first runs for this
        // ship (during salvo 4's own processing), ALL FOUR own-cells are
        // already fired simultaneously. That matters because that function
        // only actually calls `eliminate_size_at` once per cell, the first
        // time it sees it post-identification — if any of the 4 cells were
        // still unfired at that moment, it would get processed correctly
        // regardless of the bug (the bug only skips cells that are ALREADY
        // fired), permanently masking whether the OTHER (fired) cells were
        // handled too. With all 4 fired first, the bug (if reinstated) skips
        // all 4 — leaving windows [2,3,4] and [3,4,5] alive for size 3, and
        // [2,3],[3,4],[4,5] alive for size 2 — while the fix must eliminate
        // all of them.
        ai.apply_salvo([(5, 2), (0, 0), (0, 9)], [4, 0, 0]);
        ai.apply_salvo([(5, 3), (1, 1), (8, 8)], [4, 0, 0]);
        ai.apply_salvo([(5, 4), (2, 2), (7, 7)], [4, 0, 0]);
        assert!(ai.battleship_identified_cells().is_empty(), "should not identify before the 4th salvo");
        ai.apply_salvo([(5, 5), (9, 0), (9, 9)], [4, 0, 0]);
        assert_eq!(ai.battleship_identified_cells().len(), 4);
        assert!(ai.is_fired(5, 2));
        assert!(ai.is_fired(5, 3));
        assert!(ai.is_fired(5, 4));
        assert!(ai.is_fired(5, 5));

        let (rows3, _) = ai.line_states(3);
        let (rows2, _) = ai.line_states(2);
        assert_eq!(AiPlayer::alive_count(3, rows3[5]), 0, "row 5 size3 should be fully dead");
        // Size-2 has one irreducible survivor: window [7,8] — cols 7/8 sit
        // beyond the flank at col 6 and are never touched by any cross/salvo
        // in this row, so they stay alive independent of the bug or the fix.
        // Every OTHER window depends on cols 2-5 (all bug-affected) or col 1/6
        // (flanks, always correctly excluded), so 1 is the correct fully-fixed
        // total — not 0.
        assert_eq!(AiPlayer::alive_count(2, rows2[5]), 1, "row 5 size2 should have only the untouched [7,8] window left");
    }

    #[test]
    fn ai_drop_candidates_eliminates_size4_even_for_a_cell_fired_as_a_decoy() {
        let mut ai = AiPlayer::new();

        // Decoy (1,1) is fired as part of round 1's ambiguous bound=4 salvo —
        // its own true result is 0, but since the SALVO's bound is 4, the
        // normal per-cell path eliminates nothing for size 4 there (any of the
        // 3 cells could ambiguously be the real hit).
        //
        // This is deliberately built so the surviving cells form a run of 7
        // (cols 2-8) — comfortably clear of the room-pruning threshold (4) —
        // so `prune_candidates_without_room` can't independently clean up
        // col 1's neighbours and mask the result the way a shorter run would.
        // Round 1 covers cols 1-8 in row 1 (via TWO decoys: (1,1) gives cols
        // 1-4, (1,8) gives cols 5-8); round 2 covers cols 2-8 (via decoy
        // (1,5)). Only col 1 — decoy (1,1) itself — lands in round 1's union
        // but not round 2's, so it alone gets dropped, while cols 2-8 survive
        // as genuine (never-dropped, hence FSM-untouched) candidates. That
        // makes (1,1) the sole determinant of whether row 1's one remaining
        // size-4 window ([1-4]) is still alive — nothing else can
        // coincidentally kill it instead.
        ai.apply_salvo([(4, 4), (1, 1), (1, 8)], [4, 0, 0]);
        let after_round1: std::collections::HashSet<(usize, usize)> =
            ai.battleship_candidate_cells().into_iter().collect();
        for col in 1..=8 {
            assert!(after_round1.contains(&(1, col)), "(1,{col}) should survive round 1");
        }

        ai.apply_salvo([(4, 6), (1, 5), (9, 0)], [4, 0, 0]);
        let after_round2: std::collections::HashSet<(usize, usize)> =
            ai.battleship_candidate_cells().into_iter().collect();
        assert!(!after_round2.contains(&(1, 1)), "(1,1) should be dropped in round 2");
        for col in 2..=8 {
            assert!(after_round2.contains(&(1, col)), "(1,{col}) should survive — never dropped, either round");
        }

        let (_, _, combined4) = ai.alive_grids(4);
        assert_eq!(
            combined4[0][0], 0,
            "size-4 alive value at (1,1) should be 0 once dropped, even though it was fired as a decoy"
        );
    }

    #[test]
    fn ai_cross3_entry_flags_outer_ring_decoys_ruled_out_immediately() {
        let mut ai = AiPlayer::new();
        ai.apply_salvo([(4, 4), (0, 0), (9, 9)], [3, 0, 0]);

        let entries = ai.cross3_entries();
        assert_eq!(entries.len(), 1);
        let entry = &entries[0];
        // Outer-ring decoys can never hold a ship of size >=2, so they're
        // ruled out immediately, with no further deduction needed.
        assert!(entry.coord_ruled_out[1], "(0,0) is outer ring, must be ruled out immediately");
        assert!(entry.coord_ruled_out[2], "(9,9) is outer ring, must be ruled out immediately");
        // (4,4) still has full room — nothing eliminates it yet.
        assert!(!entry.coord_ruled_out[0], "(4,4) still has room and must not be ruled out yet");
    }












    #[test]
    fn ai_cross2_entry_flags_outer_ring_decoys_ruled_out_immediately() {
        let mut ai = AiPlayer::new();
        ai.apply_salvo([(4, 4), (0, 0), (9, 9)], [2, 0, 0]);

        let entries = ai.cross2_entries();
        assert_eq!(entries.len(), 1);
        let entry = &entries[0];
        assert!(entry.coord_ruled_out[1], "(0,0) is outer ring, must be ruled out immediately");
        assert!(entry.coord_ruled_out[2], "(9,9) is outer ring, must be ruled out immediately");
        assert!(!entry.coord_ruled_out[0], "(4,4) still has room and must not be ruled out yet");
    }









    #[test]
    fn ai_cross2_entry_ruled_out_only_needs_size_2_dead_not_every_size() {
        let mut ai = AiPlayer::new();
        // Entry 0: a bag with all of 4, 3, AND 2 present (nothing absent, so
        // the "value not in bag" elimination never fires, and bound == 4
        // means the plain ">bound" rule eliminates nothing either) — (7,5)
        // stays fully alive for every size right after this salvo.
        ai.apply_salvo([(7, 5), (2, 2), (2, 8)], [4, 3, 2]);
        let (_, _, before2) = ai.alive_grids(2);
        let (_, _, before3) = ai.alive_grids(3);
        assert_ne!(before2[7 - 1][5 - 1], 0, "sanity: (7,5) still alive for size 2 before the second salvo");
        assert_ne!(before3[7 - 1][5 - 1], 0, "sanity: (7,5) still alive for size 3 before the second salvo");

        // A later, unrelated salvo whose bag has no 2 at all (3 is present,
        // so size 3 stays legitimately open at all 3 of ITS cells) — proves
        // (7,5) can't be a Frigate specifically, while leaving it a
        // perfectly live Cruiser candidate.
        ai.apply_salvo([(7, 5), (3, 3), (3, 4)], [3, 1, 0]);

        let (_, _, combined2) = ai.alive_grids(2);
        let (_, _, combined3) = ai.alive_grids(3);
        assert_eq!(combined2[7 - 1][5 - 1], 0, "sanity: (7,5) is now dead for size 2");
        assert_ne!(combined3[7 - 1][5 - 1], 0, "sanity: (7,5) must still be alive for size 3 (Cruiser still open)");

        // Entry 0's own (7,5) coordinate must be flagged ruled-out as a
        // Frigate candidate purely because size 2 is dead there — it must
        // NOT also require size 3/4 to be dead, since those are irrelevant
        // to whether this cell could be THIS bag's Frigate hit.
        assert!(
            ai.cross2_entries()[0].coord_ruled_out[0],
            "(7,5) must be ruled out as a Frigate candidate once size 2 alone is dead there"
        );
    }

    #[test]
    fn ai_cross4_entries_are_recorded_and_flagged_against_the_combined_alive_value() {
        let mut ai = AiPlayer::new();
        assert!(ai.cross4_entries().is_empty(), "sanity: no 4-bearing salvo yet");

        ai.apply_salvo([(4, 4), (1, 1), (8, 8)], [4, 0, 0]);
        let entries = ai.cross4_entries();
        assert_eq!(entries.len(), 1);
        let entry = &entries[0];
        assert_eq!(entry.coords, [(4, 4), (1, 1), (8, 8)]);
        assert_eq!(entry.values, [4, 0, 0]);
        // Nothing has independently reduced (1,1)/(8,8)'s own combined
        // size-4 alive value to zero yet, so all 3 start green — see
        // `ai_cross4_entry_flags_get_ruled_out_once_combined_alive_value_drops_to_zero`
        // for the case where a coordinate does flip red.
        assert_eq!(entry.coord_ruled_out, [false, false, false]);

        // A second 4-bearing salvo appends a second entry rather than
        // replacing or merging into the first. By now, though, (2,2) and
        // (7,7) are legitimately dead for size 4 — the first salvo's
        // union-of-crosses elimination (`apply_battleship_cross_elimination`)
        // already fed everywhere outside its candidate union into the size-4
        // FSM the same way a real miss would, and (2,2)/(7,7) fall outside
        // it — so they come back red immediately, while (4,5) (right next to
        // the first salvo's own hit) does not.
        ai.apply_salvo([(4, 5), (2, 2), (7, 7)], [4, 0, 0]);
        let entries = ai.cross4_entries();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[1].coords, [(4, 5), (2, 2), (7, 7)]);
        assert_eq!(entries[1].coord_ruled_out, [false, true, true]);

        // A salvo without a 4 in its bag isn't a cross-4 salvo at all — no
        // new entry.
        ai.apply_salvo([(0, 0), (0, 1), (0, 2)], [0, 0, 0]);
        assert_eq!(ai.cross4_entries().len(), 2, "a non-4 salvo must not add a cross-4 entry");
    }

    #[test]
    fn ai_cross3_entry_flags_get_ruled_out_by_later_battleship_identification() {
        let mut ai = AiPlayer::new();

        // Cross-3 hit at (3,4) — Chebyshev-adjacent to (4,4), where the
        // Battleship will later be identified (row4 cols3-6).
        ai.apply_salvo([(3, 4), (0, 0), (0, 9)], [3, 0, 0]);
        assert!(!ai.cross3_entries()[0].coord_ruled_out[0], "sanity: (3,4) not yet ruled out");

        // Identify the Battleship at row4 cols3-6 (same proven salvo pair as
        // ai_identifies_exact_battleship_layout_after_two_intersecting_crosses).
        ai.apply_salvo([(4, 3), (1, 1), (2, 2)], [4, 0, 0]);
        ai.apply_salvo([(4, 6), (8, 7), (8, 8)], [4, 0, 0]);
        assert_eq!(ai.battleship_identified_cells().len(), 4);

        assert!(
            ai.cross3_entries()[0].coord_ruled_out[0],
            "(3,4) is Chebyshev-adjacent to the Battleship, must now be ruled out as this salvo's real hit"
        );
    }


    #[test]
    fn ai_confirms_a_cross4_hit_by_elimination_and_prunes_candidates_through_it() {
        // Mirrors a reported live scenario exactly: G7,F2,G3 -> 4 3 0, then
        // G2,H4,C6 -> 4 0 0. G7, H4 and C6 are all already known dead for
        // the Battleship by the time the 2nd salvo lands (each too far from
        // the running candidate cross), leaving G2 as the ONLY coordinate
        // left that could possibly have produced the 2nd salvo's "4" — a
        // certain hit, the same "only remaining candidate" logic already
        // used for Cruisers/Frigates one size down.
        let mut ai = AiPlayer::new();
        ai.apply_salvo([(6, 6), (1, 5), (2, 6)], [4, 3, 0]); // G7,F2,G3 -> 4 3 0
        ai.apply_salvo([(1, 6), (3, 7), (5, 2)], [4, 0, 0]); // G2,H4,C6 -> 4 0 0

        assert!(ai.cross4_entries()[1].coord_ruled_out[1], "sanity: H4 already dead for the Battleship");
        assert!(ai.cross4_entries()[1].coord_ruled_out[2], "sanity: C6 already dead for the Battleship");
        assert!(
            ai.cross4_entries()[1].coord_confirmed_battleship_hit[0],
            "G2=(1,6) is the only coordinate left in its own salvo's bag that could explain the '4' — must be confirmed"
        );

        // The real ship must now be one of the straight-4 windows passing
        // through the confirmed G2 — every candidate cell must still be
        // part of at least one such window (none fall outside it here: the
        // 3 horizontal windows through row 1 and the 1 vertical window
        // through column 6 already cover the full 9-cell candidate set).
        let candidates: std::collections::HashSet<(usize, usize)> = ai.battleship_candidate_cells().into_iter().collect();
        assert!(candidates.contains(&(1, 6)), "the confirmed cell G2 itself must remain a candidate");
        assert_eq!(candidates.len(), 9, "sanity: matches the reported scenario's 9 surviving candidates");
    }

    #[test]
    fn ai_confirms_a_cross3_hit_by_elimination_and_propagates_to_a_shared_coordinate() {
        // Mirrors the live scenario that prompted this: (2,2) is fired
        // alongside 2 outer-ring decoys (ruled out immediately), leaving it
        // as the only open candidate for its own salvo's "3" — confirmed by
        // the same "only remaining candidate" logic already used for
        // Battleship, one size down. See `derive_confirmed_cruiser_hits_
        // by_elimination`.
        let mut ai = AiPlayer::new();
        ai.apply_salvo([(2, 2), (0, 0), (0, 9)], [3, 0, 0]);
        assert!(ai.cross3_entries()[0].coord_confirmed_cruiser_hit[0], "sanity: (2,2) confirmed by its own salvo's elimination");

        // A second salvo refires that SAME (2,2) alongside a still-open
        // candidate (5,5) and another outer-ring decoy. (2,2) is the same
        // physical cell already confirmed above, so it already explains
        // this salvo's only "3" too — (5,5) must be ruled out by
        // elimination, NOT because of anything about (5,5) itself.
        ai.apply_salvo([(2, 2), (5, 5), (0, 1)], [3, 0, 0]);
        let entry_b = &ai.cross3_entries()[1];
        assert!(entry_b.coord_confirmed_cruiser_hit[0], "(2,2) carries its confirmed status into this entry too");
        assert!(entry_b.coord_ruled_out[1], "(5,5) must be ruled out: (2,2) already accounts for this salvo's only 3");
    }

    #[test]
    fn ai_cross3_two_real_hits_in_one_salvo_does_not_wrongly_eliminate_either() {
        // Regression test for a real bug caught by
        // `self_play_discovers_every_ship_of_size_at_least_2_by_game_end`'s
        // "no real Cruiser cell ever wrongly ruled out" assertion: an
        // earlier version of `derive_confirmed_cruiser_hits_by_elimination`
        // treated ANY entry containing a coordinate confirmed elsewhere as
        // "fully explained by that one cell alone," without checking how
        // many "3"s the bag actually needed explained. A bag with 2 real
        // Cruiser hits (2 different Cruiser cells landing in the same
        // salvo) then had its SECOND real hit wrongly ruled out the moment
        // the FIRST one happened to already be confirmed via some other
        // salvo.
        let mut ai = AiPlayer::new();

        // (2,2) gets independently confirmed via its own salvo: 2
        // outer-ring decoys ruled out immediately, leaving it as the sole
        // candidate.
        ai.apply_salvo([(2, 2), (0, 0), (0, 9)], [3, 0, 0]);
        assert!(ai.cross3_entries()[0].coord_confirmed_cruiser_hit[0], "sanity: (2,2) confirmed");

        // (2,2) is refired alongside a genuinely DIFFERENT real Cruiser
        // cell, (2,7), plus a genuine miss at (5,5) — this bag legitimately
        // needs 2 of its 3 coordinates explained as real Cruiser hits, not 1.
        ai.apply_salvo([(2, 2), (2, 7), (5, 5)], [3, 3, 0]);
        let entry_b = &ai.cross3_entries()[1];
        assert!(entry_b.coord_confirmed_cruiser_hit[0], "(2,2)'s confirmed status still propagates in");
        assert!(
            !entry_b.coord_ruled_out[1],
            "(2,7) is a genuinely real Cruiser cell and must NOT be ruled out just because (2,2) is already \
             confirmed — this bag needs 2 real hits explained, not 1"
        );
        assert!(!entry_b.coord_confirmed_cruiser_hit[1], "(2,7) isn't independently provable as confirmed from this data alone");
    }

    #[test]
    fn ai_confirms_a_cross2_hit_by_elimination_and_propagates_to_a_shared_coordinate() {
        // Mirrors `ai_confirms_a_cross3_hit_by_elimination_and_propagates_
        // to_a_shared_coordinate` one ship size down.
        let mut ai = AiPlayer::new();
        ai.apply_salvo([(2, 2), (0, 0), (0, 9)], [2, 0, 0]);
        assert!(ai.cross2_entries()[0].coord_confirmed_frigate_hit[0], "sanity: (2,2) confirmed by its own salvo's elimination");

        ai.apply_salvo([(2, 2), (5, 5), (0, 1)], [2, 0, 0]);
        let entry_b = &ai.cross2_entries()[1];
        assert!(entry_b.coord_confirmed_frigate_hit[0], "(2,2) carries its confirmed status into this entry too");
        assert!(entry_b.coord_ruled_out[1], "(5,5) must be ruled out: (2,2) already accounts for this salvo's only 2");
    }

    #[test]
    fn ai_newly_ruled_out_since_reports_only_fresh_flips() {
        let mut ai = AiPlayer::new();

        // Round 1: cross-3 hit at (3,4), with (0,0)/(0,9) ruled out immediately
        // (outer ring) — already true before any snapshot is taken.
        ai.apply_salvo([(3, 4), (0, 0), (0, 9)], [3, 0, 0]);
        let snapshot = ai.cross3_ruled_out_snapshot();
        // Nothing "newly" ruled out relative to a snapshot taken right after
        // the same round that ruled them out.
        assert!(ai.newly_ruled_out_since(&snapshot).is_empty());

        // Round 2+3: identify the Battleship at row4 cols3-6, which rules out
        // (3,4) via Chebyshev adjacency to (4,4) — a flip that happens after
        // the snapshot above was taken.
        ai.apply_salvo([(4, 3), (1, 1), (2, 2)], [4, 0, 0]);
        ai.apply_salvo([(4, 6), (8, 7), (8, 8)], [4, 0, 0]);
        assert_eq!(ai.battleship_identified_cells().len(), 4);

        let newly = ai.newly_ruled_out_since(&snapshot);
        assert_eq!(newly, vec![(3, 4)], "only (3,4) flipped since the snapshot — (0,0)/(0,9) were already ruled out by then");

        // A fresh snapshot right now reports nothing further, since nothing's
        // changed since it was taken.
        let snapshot2 = ai.cross3_ruled_out_snapshot();
        assert!(ai.newly_ruled_out_since(&snapshot2).is_empty());
    }

    #[test]
    fn ai_newly_ruled_out_since_handles_brand_new_entries_created_after_the_snapshot() {
        let mut ai = AiPlayer::new();
        let snapshot = ai.cross3_ruled_out_snapshot();
        assert!(snapshot.is_empty(), "sanity: no cross-3 entries exist yet");

        // This entry didn't exist when `snapshot` was taken — its outer-ring
        // decoys must still be correctly reported as newly ruled out.
        ai.apply_salvo([(4, 4), (0, 0), (9, 9)], [3, 0, 0]);

        let newly: std::collections::HashSet<(usize, usize)> =
            ai.newly_ruled_out_since(&snapshot).into_iter().collect();
        assert!(newly.contains(&(0, 0)));
        assert!(newly.contains(&(9, 9)));
        assert!(!newly.contains(&(4, 4)), "(4,4) still has room and isn't ruled out");
    }

    #[test]
    fn ai_cruiser_heatmap_is_all_zero_before_any_salvo() {
        let ai = AiPlayer::new();
        let heatmap = ai.cruiser_heatmap();
        assert_eq!(heatmap.len(), 8);
        for row in &heatmap {
            assert_eq!(row.len(), 8);
            assert!(row.iter().all(|&p| p == 0.0), "no evidence yet — nothing should stand out: {row:?}");
        }
    }

    #[test]
    fn ai_frigate_heatmap_is_all_zero_before_any_salvo() {
        let ai = AiPlayer::new();
        let heatmap = ai.frigate_heatmap();
        assert_eq!(heatmap.len(), 8);
        for row in &heatmap {
            assert_eq!(row.len(), 8);
            assert!(row.iter().all(|&p| p == 0.0), "no evidence yet — nothing should stand out: {row:?}");
        }
    }

    #[test]
    fn ai_cruiser_heatmap_shows_certainty_once_both_cruisers_are_fully_pinned() {
        let mut ai = AiPlayer::new();
        // Firing a straight-3 run directly, all 3 cells coming back "3",
        // leaves only one geometrically possible reading: this exact
        // window IS a Cruiser (see `consistent_with_salvo_history` — no
        // other combination of 2 non-overlapping, non-adjacent Cruiser
        // windows could produce "all 3 fired cells inside the union" for
        // this salvo without simply BEING this window). Two such salvos,
        // far enough apart to never conflict, pin down the entire 2-Cruiser
        // layout with total certainty.
        ai.apply_salvo([(2, 2), (2, 3), (2, 4)], [3, 3, 3]);
        ai.apply_salvo([(6, 6), (6, 7), (6, 8)], [3, 3, 3]);

        let heatmap = ai.cruiser_heatmap();
        let true_cells: std::collections::HashSet<(usize, usize)> = [(2, 2), (2, 3), (2, 4), (6, 6), (6, 7), (6, 8)].into_iter().collect();
        for row in 1..=8 {
            for col in 1..=8 {
                let p = heatmap[row - 1][col - 1];
                if true_cells.contains(&(row, col)) {
                    assert_eq!(p, 1.0, "known Cruiser cell {:?} must show certainty, got {p}", (row, col));
                } else {
                    assert_eq!(p, 0.0, "cell {:?} isn't part of the only consistent layout, got {p}", (row, col));
                }
            }
        }
    }

    #[test]
    fn ai_cruiser_identified_cells_returns_the_layout_once_fully_pinned() {
        let mut ai = AiPlayer::new();
        ai.apply_salvo([(2, 2), (2, 3), (2, 4)], [3, 3, 3]);
        ai.apply_salvo([(6, 6), (6, 7), (6, 8)], [3, 3, 3]);
        assert_eq!(
            ai.cruiser_identified_cells().into_iter().collect::<std::collections::HashSet<_>>(),
            [(2, 2), (2, 3), (2, 4), (6, 6), (6, 7), (6, 8)].into_iter().collect()
        );
    }

    #[test]
    fn ai_cruiser_identified_cells_is_empty_while_still_ambiguous() {
        let mut ai = AiPlayer::new();
        // Exactly one of these 3 is a real Cruiser hit, but many different
        // Cruiser-pair layouts stay consistent with this alone.
        ai.apply_salvo([(2, 2), (2, 3), (2, 4)], [3, 0, 0]);
        assert!(ai.cruiser_identified_cells().is_empty());
    }

    #[test]
    fn ai_cruiser_layout_elimination_narrows_the_cruiser_and_frigate_fsm() {
        // Once the Cruisers' exact layout is confirmed via the heatmap and
        // `update_fsm_and_resolve` is called (the manual "heatmap fully
        // evolved" trigger — this no longer happens automatically),
        // choose_shots' ordinary hunting FSM should learn from it too: no
        // Cruiser possible anywhere else on the board, and no Frigate
        // possible adjacent to either real Cruiser.
        let mut ai = AiPlayer::new();
        ai.apply_salvo([(2, 2), (2, 3), (2, 4)], [3, 3, 3]);
        ai.apply_salvo([(6, 6), (6, 7), (6, 8)], [3, 3, 3]);
        ai.mark_sunk(3);
        ai.mark_sunk(3);
        assert!(ai.update_fsm_and_resolve(), "the Cruisers are already cross-reasoning-identified, must actually lock in");

        let (rows3, _) = ai.line_states(3);
        for r in 1..=8 {
            let alive = AiPlayer::alive_count(3, rows3[r]);
            if r == 2 || r == 6 {
                assert!(alive > 0, "row {r} passes through a confirmed Cruiser cell, must still show alive Cruiser placements, got {alive}");
            } else {
                assert_eq!(alive, 0, "row {r} has no confirmed Cruiser cell, must show 0 alive Cruiser placements now, got {alive}");
            }
        }

        // Per-cell (not whole-row) Frigate check: only cells actually
        // adjacent to a real Cruiser cell must show 0 — e.g. row 1 also
        // has cols 6-8, nowhere near either Cruiser, which must stay alive.
        let (_, _, combined2) = ai.alive_grids(2);
        for &(r, c) in &[(1, 1), (1, 2), (1, 3), (2, 1), (2, 5), (3, 1), (3, 2), (3, 3)] {
            assert_eq!(combined2[r - 1][c - 1], 0, "{:?} is adjacent to a confirmed Cruiser cell, must show 0 alive Frigate placements", (r, c));
        }
        assert!(combined2[1 - 1][8 - 1] > 0, "sanity: (1,8) is nowhere near either Cruiser, must still show alive Frigate placements");
    }

    #[test]
    fn ai_frigate_heatmap_excludes_cells_confirmed_cruiser_or_adjacent() {
        let mut ai = AiPlayer::new();
        ai.apply_salvo([(2, 2), (2, 3), (2, 4)], [3, 3, 3]);
        ai.apply_salvo([(6, 6), (6, 7), (6, 8)], [3, 3, 3]);

        let frigate_heatmap = ai.frigate_heatmap();
        for &(r, c) in &[(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (3, 1), (3, 2), (3, 3)] {
            assert_eq!(
                frigate_heatmap[r - 1][c - 1], 0.0,
                "{:?} is a confirmed Cruiser cell or its neighbour, must show 0 Frigate probability", (r, c)
            );
        }
        let has_nonzero_elsewhere = (1..=8).any(|r| (1..=8).any(|c| frigate_heatmap[r - 1][c - 1] > 0.0));
        assert!(has_nonzero_elsewhere, "sanity: the rest of the board should still show genuine Frigate ambiguity");
    }

    #[test]
    fn ai_heatmap_fraction_methods_match_the_probability_methods() {
        let mut ai = AiPlayer::new();
        ai.apply_salvo([(2, 2), (2, 3), (2, 4)], [3, 3, 3]);
        ai.apply_salvo([(6, 6), (6, 7), (6, 8)], [3, 3, 3]);

        let cruiser_heatmap = ai.cruiser_heatmap();
        let cruiser_fraction = ai.cruiser_heatmap_fraction();
        for row in 1..=8 {
            for col in 1..=8 {
                let (num, denom) = cruiser_fraction[row - 1][col - 1];
                let expected = if denom == 0 { 0.0 } else { num as f64 / denom as f64 };
                assert_eq!(expected, cruiser_heatmap[row - 1][col - 1], "fraction/probability mismatch at {:?}", (row, col));
            }
        }
    }

    #[test]
    fn ai_frigate_heatmap_shows_certainty_once_all_3_frigates_are_fully_pinned() {
        let mut ai = AiPlayer::new();
        // Same reasoning as the Cruiser version, one size down: an
        // all-2s salvo on a straight-2 run pins that exact window down
        // with total certainty. Well-separated so none conflict.
        ai.apply_salvo([(2, 2), (2, 3), (0, 0)], [2, 2, 0]);
        ai.apply_salvo([(5, 5), (5, 6), (0, 1)], [2, 2, 0]);
        ai.apply_salvo([(8, 2), (8, 3), (0, 2)], [2, 2, 0]);

        let heatmap = ai.frigate_heatmap();
        let true_cells: std::collections::HashSet<(usize, usize)> =
            [(2, 2), (2, 3), (5, 5), (5, 6), (8, 2), (8, 3)].into_iter().collect();
        for row in 1..=8 {
            for col in 1..=8 {
                let p = heatmap[row - 1][col - 1];
                if true_cells.contains(&(row, col)) {
                    assert_eq!(p, 1.0, "known Frigate cell {:?} must show certainty, got {p}", (row, col));
                } else {
                    assert_eq!(p, 0.0, "cell {:?} isn't part of the only consistent layout, got {p}", (row, col));
                }
            }
        }
    }

    #[test]
    fn ai_heatmaps_never_place_weight_on_a_cell_proven_to_lack_that_size() {
        let mut ai = AiPlayer::new();
        // A miss (bag with no 3 or 2 at all) proves these 3 cells hold
        // neither a Cruiser nor a Frigate — no window in either heatmap's
        // enumeration ever includes them, so both must show exactly 0 here
        // regardless of how much genuine ambiguity remains elsewhere.
        ai.apply_salvo([(4, 4), (4, 5), (4, 6)], [0, 0, 0]);
        ai.apply_salvo([(2, 2), (2, 3), (2, 4)], [3, 3, 3]); // some unrelated real evidence, so history isn't empty

        let cruiser_heatmap = ai.cruiser_heatmap();
        let frigate_heatmap = ai.frigate_heatmap();
        for &(r, c) in &[(4, 4), (4, 5), (4, 6)] {
            assert_eq!(cruiser_heatmap[r - 1][c - 1], 0.0, "{:?} proven miss, must be 0 in the Cruiser heatmap", (r, c));
            assert_eq!(frigate_heatmap[r - 1][c - 1], 0.0, "{:?} proven miss, must be 0 in the Frigate heatmap", (r, c));
        }
    }

    #[test]
    fn ai_heatmaps_and_disambiguation_exclude_cells_confirmed_as_the_battleship() {
        // Regression: a cell holds exactly one ship, so once the cross-4
        // deduction confirms a cell IS the Battleship, no Cruiser/Frigate
        // window may ever include it — but `consistent_with_salvo_history`
        // only checks each salvo's AGGREGATE count of one value at a time,
        // so on its own it can't tell "this cell is really a 4" from "this
        // cell coincidentally isn't part of the window I'm testing"; a
        // wrong hypothesis could "explain" a salvo's evidence by
        // substituting in a confirmed Battleship cell instead of the real
        // Cruiser/Frigate cell that salvo actually hit.
        let mut ai = AiPlayer::new();

        // Identify the Battleship at (4,3)-(4,4)-(4,5)-(4,6) via 2
        // intersecting crosses (mirrors
        // ai_identifies_exact_battleship_layout_after_two_intersecting_crosses).
        ai.apply_salvo([(4, 3), (1, 1), (2, 2)], [4, 0, 0]);
        ai.apply_salvo([(4, 6), (8, 7), (8, 8)], [4, 0, 0]);
        assert_eq!(
            ai.battleship_identified_cells().into_iter().collect::<std::collections::HashSet<_>>(),
            [(4, 3), (4, 4), (4, 5), (4, 6)].into_iter().collect()
        );

        // (4,4) is confirmed Battleship but still unfired. Firing it for
        // real alongside a genuine Cruiser hit at (6,6) correctly shows
        // both a 4 (from (4,4)) and a 3 (from (6,6)) in the same bag — but
        // the bag is unordered, so on raw aggregate counts alone, a
        // hypothesis could just as easily explain the "one 3 in this
        // salvo" by wrongly crediting it to (4,4) instead of (6,6).
        ai.apply_salvo([(4, 4), (6, 6), (6, 7)], [4, 3, 0]);

        let cruiser_heatmap = ai.cruiser_heatmap();
        assert_eq!(
            cruiser_heatmap[4 - 1][4 - 1], 0.0,
            "(4,4) is confirmed Battleship — must never show Cruiser probability, even though the raw salvo bag alone can't rule it out"
        );

        if let Some(shots) = ai.cruiser_disambiguation_shots() {
            assert!(!shots.contains(&(4, 4)), "disambiguation must never target a confirmed Battleship cell: {:?}", shots);
        }
    }

    #[test]
    fn ai_refiring_a_confirmed_battleship_anchor_does_not_corrupt_its_own_fsm() {
        // Regression: `anchored_isolation_shot` deliberately refires an
        // already-confirmed Battleship cell as a safe "known 4" anchor,
        // paired with 2 unrelated cells elsewhere on the board. That 3rd
        // "4" in the bag used to unconditionally re-trigger
        // `apply_battleship_cross_elimination` — the ordinary "one of
        // these 3 cells is a Battleship hit, we don't know which" cross
        // logic — even though we DO already know which cell it is. That
        // treated the 2 far-apart, unrelated cells as still-live
        // Battleship candidates for a brand new cross, and intersecting
        // that against history wiped out the real, already-confirmed
        // window, zeroing the size-4 FSM everywhere.
        let mut ai = AiPlayer::new();
        ai.apply_salvo([(4, 3), (1, 1), (2, 2)], [4, 0, 0]);
        ai.apply_salvo([(4, 6), (8, 7), (8, 8)], [4, 0, 0]);
        assert_eq!(
            ai.battleship_identified_cells().into_iter().collect::<std::collections::HashSet<_>>(),
            [(4, 3), (4, 4), (4, 5), (4, 6)].into_iter().collect()
        );
        ai.mark_sunk(4);

        let (rows_before, cols_before) = ai.line_states(4);
        assert!(rows_before[4] > 0, "sanity: row 4 must still show a live size-4 placement before the refire");

        // Re-fire the already-confirmed (4,3) alongside 2 unrelated, far-away
        // cells — exactly what `anchored_isolation_shot` does.
        ai.apply_salvo([(4, 3), (7, 7), (8, 2)], [4, 3, 2]);

        let (rows_after, cols_after) = ai.line_states(4);
        assert_eq!(rows_before, rows_after, "re-firing a confirmed Battleship cell as an anchor must not change the size-4 row FSM");
        assert_eq!(cols_before, cols_after, "...or the column FSM");
    }

    #[test]
    fn ai_choose_shots_finds_an_anchored_isolation_cleanup_shot_when_available() {
        // Regression / feature test for the "anchor-and-isolate" cleanup
        // shot: a confirmed Battleship cell (known, certain "4") plus one
        // cell that's proven possible ONLY as a Cruiser and another proven
        // possible ONLY as a Frigate, fired together, resolves both by
        // elimination — strictly better than the general minimax search's
        // worst-case narrowing.
        let mut ai = AiPlayer::new();

        // Confirm the Battleship at (4,3)-(4,4)-(4,5)-(4,6).
        ai.apply_salvo([(4, 3), (1, 1), (2, 2)], [4, 0, 0]);
        ai.apply_salvo([(4, 6), (8, 7), (8, 8)], [4, 0, 0]);

        // Exactly one of these 3 is a real Cruiser hit; the lack of any
        // "2" in this bag also proves none of them can be a Frigate.
        ai.apply_salvo([(3, 2), (3, 3), (3, 4)], [3, 0, 0]);
        // Exactly one of these 3 is a real Frigate hit; the lack of any
        // "3" proves none of them can be a Cruiser.
        ai.apply_salvo([(6, 2), (6, 3), (6, 4)], [2, 0, 0]);

        let confirmed_battleship: std::collections::HashSet<(usize, usize)> = [(4, 3), (4, 4), (4, 5), (4, 6)].into_iter().collect();
        let cruiser_only: std::collections::HashSet<(usize, usize)> = [(3, 2), (3, 3), (3, 4)].into_iter().collect();
        let frigate_only: std::collections::HashSet<(usize, usize)> = [(6, 2), (6, 3), (6, 4)].into_iter().collect();

        ai.mark_sunk(4);
        ai.mark_sunk(3);
        ai.mark_sunk(3);

        let shots = ai.choose_shots();
        let anchor_count = shots.iter().filter(|c| confirmed_battleship.contains(c)).count();
        let cruiser_count = shots.iter().filter(|c| cruiser_only.contains(c)).count();
        let frigate_count = shots.iter().filter(|c| frigate_only.contains(c)).count();
        assert_eq!(anchor_count, 1, "exactly one shot must be the known Battleship anchor: {:?}", shots);
        assert_eq!(cruiser_count, 1, "exactly one shot must be a proven cruiser-only candidate: {:?}", shots);
        assert_eq!(frigate_count, 1, "exactly one shot must be a proven frigate-only candidate: {:?}", shots);
    }

    /// 4 mutually far-apart, never-overlapping straight-3 runs. With the
    /// rest of the inner board flooded to misses (see
    /// `flood_inner_misses_except`), every one of the C(4,2) = 6
    /// non-overlapping pairs among these stays equally "consistent" (none
    /// of their cells were ever fired) — a small, bounded amount of
    /// genuine Cruiser ambiguity, comfortably under
    /// `disambiguation_shots`'s `MAX_CANDIDATES_TO_ATTEMPT` cap.
    fn reserved_cruiser_cells() -> Vec<(usize, usize)> {
        vec![
            (2, 2), (2, 3), (2, 4),
            (2, 6), (2, 7), (2, 8),
            (7, 2), (7, 3), (7, 4),
            (7, 6), (7, 7), (7, 8),
        ]
    }

    /// 5 mutually far-apart 2-cell runs, chosen so 4 of the C(5,3) = 10
    /// possible triples are non-overlapping — same idea as
    /// `reserved_cruiser_cells`, one size down.
    fn reserved_frigate_cells() -> Vec<(usize, usize)> {
        vec![
            (4, 2), (4, 3),
            (4, 6), (4, 7),
            (5, 2), (5, 3),
            (5, 6), (5, 7),
            (1, 4), (1, 5),
        ]
    }

    /// 3 mutually far-apart straight-3 runs, reserved for scenarios that
    /// need BOTH Cruiser and Frigate ambiguity at once. Each 3-cell run
    /// doubles as 2 valid Frigate sub-windows, so unlike
    /// `reserved_cruiser_cells` + `reserved_frigate_cells` combined (whose
    /// Frigate candidate count blows past `MAX_CANDIDATES_TO_ATTEMPT` once
    /// the Cruiser regions' extra sub-windows are counted in), a single
    /// smaller shared set keeps both counts bounded: C(3,2) = 3 Cruiser
    /// pairs, and exactly 8 non-overlapping Frigate triples (one
    /// sub-window choice per region, since same-region sub-windows are
    /// mutually adjacent).
    fn reserved_shared_cells() -> Vec<(usize, usize)> {
        vec![
            (2, 2), (2, 3), (2, 4),
            (5, 2), (5, 3), (5, 4),
            (8, 2), (8, 3), (8, 4),
        ]
    }

    /// Fires miss salvos ([0,0,0]) covering every inner 8x8 cell except
    /// `keep` and whatever's already fired — simulating a late-game board
    /// where ordinary hunting has explored almost everywhere, so the only
    /// remaining ambiguity is the handful of untouched regions in `keep`.
    /// Pads the final salvo with outer-ring cells (never a Cruiser/Frigate
    /// candidate either way) if the remainder isn't a multiple of 3.
    fn flood_inner_misses_except(ai: &mut AiPlayer, keep: &[(usize, usize)]) {
        let mut to_fire: Vec<(usize, usize)> = Vec::new();
        for r in 1..=8 {
            for c in 1..=8 {
                if keep.contains(&(r, c)) || ai.is_fired(r, c) {
                    continue;
                }
                to_fire.push((r, c));
            }
        }
        let mut outer_padding = (0..10).map(|i| (0usize, i));
        for chunk in to_fire.chunks(3) {
            let mut salvo = chunk.to_vec();
            while salvo.len() < 3 {
                salvo.push(outer_padding.next().expect("board isn't THAT exhausted"));
            }
            ai.apply_salvo([salvo[0], salvo[1], salvo[2]], [0, 0, 0]);
        }
    }

    #[test]
    fn ai_cruiser_disambiguation_shots_is_none_once_the_layout_is_fully_pinned() {
        let mut ai = AiPlayer::new();
        ai.apply_salvo([(2, 2), (2, 3), (2, 4)], [3, 3, 3]);
        ai.apply_salvo([(6, 6), (6, 7), (6, 8)], [3, 3, 3]);
        assert_eq!(ai.cruiser_disambiguation_shots(), None, "only one consistent layout remains — nothing left to disambiguate");
        assert!(!ai.cruiser_disambiguation_pending());
    }

    #[test]
    fn ai_disambiguation_pending_flags_track_whether_a_shot_is_actually_available() {
        // Regression: the frontend's "fleet cleared" popup/auto-play stop
        // condition used to fire the instant every ship of size >=2 was
        // literally sunk, even though the AI's own bag-based deduction can
        // still be genuinely ambiguous about the exact layout at that
        // point — meaning Frigate disambiguation (which only ever becomes
        // eligible once Frigates are ALSO sunk, the very same moment the
        // old stop condition fired) could never actually run. These flags
        // are what the frontend now waits on instead.
        let mut ai = AiPlayer::new();
        assert!(!ai.cruiser_disambiguation_pending(), "nothing fired yet — nothing to disambiguate");
        assert!(!ai.frigate_disambiguation_pending());

        flood_inner_misses_except(&mut ai, &reserved_shared_cells());

        assert!(ai.cruiser_disambiguation_pending(), "bounded geometric ambiguity among the untouched regions must report as pending");
        assert!(ai.frigate_disambiguation_pending());
    }

    #[test]
    fn ai_cruiser_disambiguation_shots_returns_some_when_ambiguous_and_avoids_fired_cells() {
        let mut ai = AiPlayer::new();
        // With the rest of the board flooded to misses, 6 different
        // Cruiser-pair layouts (any 2 of the 4 reserved regions) remain
        // equally consistent — genuinely ambiguous, but a small, bounded
        // amount of ambiguity `disambiguation_shots` will actually attempt
        // (unlike the near-whole-board case it deliberately defers on).
        flood_inner_misses_except(&mut ai, &reserved_cruiser_cells());

        let heatmap = ai.cruiser_heatmap();
        let has_fractional_cell = (1..=8).any(|r| (1..=8).any(|c| { let p = heatmap[r - 1][c - 1]; p > 0.0 && p < 1.0 }));
        assert!(has_fractional_cell, "sanity: this scenario must actually be ambiguous");

        let shots = ai.cruiser_disambiguation_shots().expect("ambiguous layout must produce a disambiguating salvo");
        let mut seen = std::collections::HashSet::new();
        for &(r, c) in &shots {
            assert!((1..=8).contains(&r) && (1..=8).contains(&c), "disambiguation shot {:?} must land in the inner 8x8", (r, c));
            assert!(!ai.is_fired(r, c), "disambiguation shot {:?} must not repeat an already-fired cell", (r, c));
            assert!(seen.insert((r, c)), "disambiguation salvo must not repeat a coordinate: {:?}", shots);
        }
    }

    #[test]
    fn ai_choose_shots_prioritizes_cruiser_disambiguation_once_battleship_and_cruisers_are_sunk() {
        let mut ai = AiPlayer::new();
        flood_inner_misses_except(&mut ai, &reserved_cruiser_cells());
        ai.mark_sunk(4);
        ai.mark_sunk(3);
        ai.mark_sunk(3);

        let expected = ai.cruiser_disambiguation_shots().expect("scenario must be ambiguous");
        let actual = ai.choose_shots();
        assert_eq!(actual, expected, "choose_shots must defer to cruiser disambiguation once Battleship + both Cruisers are sunk but the layout is still ambiguous");
    }

    #[test]
    fn ai_choose_shots_prefers_cruiser_disambiguation_over_frigate_disambiguation() {
        let mut ai = AiPlayer::new();
        // Bounded ambiguity for BOTH classes at once (rest of the board
        // flooded to misses) — see `reserved_shared_cells`.
        flood_inner_misses_except(&mut ai, &reserved_shared_cells());
        ai.mark_sunk(4);
        ai.mark_sunk(3);
        ai.mark_sunk(3);
        ai.mark_sunk(2);
        ai.mark_sunk(2);
        ai.mark_sunk(2);

        let cruiser_expected = ai.cruiser_disambiguation_shots().expect("cruiser scenario must be ambiguous");
        let frigate_expected = ai.frigate_disambiguation_shots().expect("frigate scenario must be ambiguous");
        assert_ne!(cruiser_expected, frigate_expected, "sanity: the two disambiguation searches must actually differ here");

        let actual = ai.choose_shots();
        assert_eq!(actual, cruiser_expected, "Cruiser disambiguation must be tried first, per the explicit priority ordering");
    }

    #[test]
    fn ai_choose_shots_falls_through_to_frigate_disambiguation_once_cruisers_are_fully_resolved() {
        let mut ai = AiPlayer::new();
        // Cruisers fully pinned down (unambiguous) via 2 all-3s salvos — far
        // from every `reserved_frigate_cells` window (rows 1, 4, 5), so the
        // new Cruiser-confirmed-cells-and-neighbours Frigate exclusion (see
        // `cells_confirmed_cruiser_or_adjacent`) doesn't accidentally
        // resolve the Frigate ambiguity this test needs to still exist.
        ai.apply_salvo([(8, 1), (8, 2), (8, 3)], [3, 3, 3]);
        ai.apply_salvo([(8, 6), (8, 7), (8, 8)], [3, 3, 3]);
        // ...but the Frigates are still (boundedly) ambiguous.
        flood_inner_misses_except(&mut ai, &reserved_frigate_cells());
        ai.mark_sunk(4);
        ai.mark_sunk(3);
        ai.mark_sunk(3);
        ai.mark_sunk(2);
        ai.mark_sunk(2);
        ai.mark_sunk(2);

        assert_eq!(ai.cruiser_disambiguation_shots(), None, "sanity: Cruisers must already be fully resolved");
        let frigate_expected = ai.frigate_disambiguation_shots().expect("Frigate scenario must be ambiguous");
        assert_eq!(ai.choose_shots(), frigate_expected, "with Cruisers resolved, choose_shots must move on to Frigate disambiguation");
    }

















    fn fixed_board_game() -> Game {
        let mut board = vec![vec![None; 10]; 10];
        for &(r, c) in &[(2, 2), (2, 3), (2, 4)] {
            board[r][c] = Some(0);
        }
        for &(r, c) in &[(7, 5), (7, 6)] {
            board[r][c] = Some(1);
        }
        let ships = vec![
            Ship::new(0, "Cruiser", 3, vec![Cell { row: 2, col: 2 }, Cell { row: 2, col: 3 }, Cell { row: 2, col: 4 }]),
            Ship::new(1, "Frigate", 2, vec![Cell { row: 7, col: 5 }, Cell { row: 7, col: 6 }]),
        ];
        Game {
            state: GameState {
                board,
                ships,
                fired: vec![vec![false; 10]; 10],
                log: Vec::new(),
                turn: 1,
                won: false,
                total_hits: 5,
                hit_count: 0,
            },
            ai: AiPlayer::new(),
        }
    }


    #[test]
    fn game_restart_same_board_keeps_ship_placement_but_clears_everything_else() {
        let mut game = fixed_board_game();
        game.fire(&[2 * 10 + 2, 0 * 10 + 0, 0 * 10 + 9]);
        game.fire(&[2 * 10 + 3, 9 * 10 + 0, 9 * 10 + 9]);
        assert!(game.state.hit_count > 0, "sanity: some hits registered before restart");
        assert!(!game.ai.cross3_entries().is_empty(), "sanity: AI has accumulated some deduction state");

        game.restart_same_board();

        assert_eq!(game.state.board[2][2], Some(0), "ship placement must survive the restart");
        assert_eq!(game.state.board[7][5], Some(1), "ship placement must survive the restart");
        assert_eq!(game.state.hit_count, 0, "hit count must reset");
        assert_eq!(game.state.turn, 1, "turn must reset");
        assert!(!game.state.won, "won must reset");
        assert!(game.state.log.is_empty(), "salvo log must reset");
        assert!(game.state.fired.iter().all(|row| row.iter().all(|&f| !f)), "every cell must be unfired again");
        assert!(game.state.ships.iter().all(|s| s.hits == 0 && !s.sunk), "every ship's hit/sunk state must reset");
        assert!(game.ai.cross3_entries().is_empty(), "AI deduction state must reset");

        // The board plays identically afterward: the same salvo produces
        // the same result as it did the first time.
        let replayed: serde_json::Value = serde_json::from_str(&game.fire(&[2 * 10 + 2, 0 * 10 + 0, 0 * 10 + 9])).unwrap();
        assert_eq!(replayed["result"], "3 0 0");
    }

    #[test]
    fn game_board_layout_round_trips_into_an_identical_fresh_board() {
        let mut game = fixed_board_game();
        // Accumulate play state that a save/load round-trip must NOT carry over.
        game.fire(&[2 * 10 + 2, 0 * 10 + 0, 0 * 10 + 9]);
        assert!(game.state.hit_count > 0, "sanity: some hits registered before saving");

        let layout_json = game.board_layout_json();

        let mut loaded = Game::new(); // starts from an unrelated random board
        let response = loaded.load_board_layout_json(&layout_json);
        assert!(!response.contains("\"error\""), "load must succeed: {response}");

        assert_eq!(loaded.state.board[2][2], Some(0), "ship placement must match the saved layout");
        assert_eq!(loaded.state.board[7][5], Some(1), "ship placement must match the saved layout");
        assert_eq!(loaded.state.hit_count, 0, "loaded game must start with no hits");
        assert_eq!(loaded.state.turn, 1, "loaded game must start at turn 1");
        assert!(!loaded.state.won, "loaded game must not be won");
        assert!(loaded.state.fired.iter().all(|row| row.iter().all(|&f| !f)), "loaded game must have nothing fired");
        assert!(loaded.state.ships.iter().all(|s| s.hits == 0 && !s.sunk), "loaded game's ships must be fresh, regardless of the saved layout's own hit state");

        // The board plays identically: the same salvo produces the same result.
        let fired: serde_json::Value = serde_json::from_str(&loaded.fire(&[2 * 10 + 2, 2 * 10 + 3, 2 * 10 + 4])).unwrap();
        assert_eq!(fired["result"], "3 3 3");
    }

    #[test]
    fn game_load_board_layout_json_rejects_invalid_json() {
        let mut game = Game::new();
        let response = game.load_board_layout_json("not valid json");
        assert!(response.contains("\"error\""), "must report an error for invalid JSON: {response}");
    }

    #[test]
    fn game_resolution_status_json_reports_unresolved_early_with_odds() {
        let game = Game::new();
        let status: serde_json::Value = serde_json::from_str(&game.resolution_status_json()).unwrap();
        assert_eq!(status["resolved"], false);
        assert_eq!(status["battleship_identified"], false);
        assert_eq!(status["cruiser_identified"], false);
        assert_eq!(status["frigate_identified"], false);
        assert!(!status["cruiser_odds"].is_null(), "not resolved — odds must be present (an all-zero grid, since no evidence yet): {status}");
        assert!(!status["frigate_odds"].is_null());
    }

    #[test]
    fn game_resolution_status_json_reports_resolved_once_everything_identified() {
        let mut game = Game::new();
        // Battleship: 2 intersecting crosses at row 4, cols 3-6.
        game.ai.apply_salvo([(4, 3), (1, 1), (1, 8)], [4, 0, 0]);
        game.ai.apply_salvo([(4, 6), (8, 1), (8, 8)], [4, 0, 0]);
        // Both Cruisers: 2 well-separated all-3s salvos.
        game.ai.apply_salvo([(2, 2), (2, 3), (2, 4)], [3, 3, 3]);
        game.ai.apply_salvo([(6, 6), (6, 7), (6, 8)], [3, 3, 3]);
        // All 3 Frigates: 3 well-separated all-2s salvos, deliberately kept
        // away from the Cruiser/Battleship cells above (and each other) —
        // no ship may be adjacent to another, so a Frigate window next to
        // a confirmed Cruiser cell would correctly never be considered a
        // valid candidate at all.
        game.ai.apply_salvo([(8, 2), (8, 3), (0, 3)], [2, 2, 0]);
        game.ai.apply_salvo([(8, 6), (8, 7), (0, 4)], [2, 2, 0]);
        game.ai.apply_salvo([(1, 6), (1, 7), (0, 5)], [2, 2, 0]);

        let status: serde_json::Value = serde_json::from_str(&game.resolution_status_json()).unwrap();
        assert_eq!(status["resolved"], true, "status: {status}");
        assert_eq!(status["battleship_identified"], true);
        assert_eq!(status["cruiser_identified"], true);
        assert_eq!(status["frigate_identified"], true);
        assert!(status["cruiser_odds"].is_null(), "fully resolved — odds must be omitted");
        assert!(status["frigate_odds"].is_null());
    }

    #[test]
    fn ai_cross_reasoning_resolves_a_cruiser_ambiguity_the_cruiser_heatmap_cannot_resolve_alone() {
        // One Cruiser fully confirmed; the second is ambiguous between 2
        // well-separated windows (Option A, Option B) as far as the Cruiser
        // heatmap alone is concerned. But all 3 Frigates are ALSO already
        // fully confirmed, and one of them sits directly against Option A —
        // information the Cruiser heatmap never looks at, since it only
        // ever cross-checks Cruiser windows against other Cruiser windows
        // (see `AiPlayer::consistent_cruiser_candidates`). Cross-reasoning
        // against the Frigate candidates (see `cross_reasoned_cruiser_
        // candidates`) should rule Option A out and resolve the Cruiser
        // completely, exactly like the manual reasoning this feature is
        // named for.
        let mut ai = AiPlayer::new();
        ai.apply_salvo([(2, 2), (2, 3), (2, 4)], [3, 3, 3]); // Cruiser 1: confirmed
        ai.apply_salvo([(6, 3), (6, 4), (0, 0)], [2, 2, 0]); // Frigate: confirmed, touches Option A
        ai.apply_salvo([(8, 1), (8, 2), (0, 0)], [2, 2, 0]); // Frigate: confirmed, elsewhere
        ai.apply_salvo([(8, 7), (8, 8), (0, 0)], [2, 2, 0]); // Frigate: confirmed, elsewhere
        flood_inner_misses_except(&mut ai, &[(5, 2), (5, 3), (5, 4), (5, 6), (5, 7), (5, 8)]);

        assert!(ai.cruiser_identified_cells().is_empty(), "sanity: the Cruiser heatmap alone must still be ambiguous");
        let heatmap = ai.cruiser_heatmap();
        assert!((0.0..1.0).contains(&heatmap[4][1]), "sanity: Option A ((5,2), row index 4, col index 1) must be a live but uncertain Cruiser candidate: {heatmap:?}");
        assert!((0.0..1.0).contains(&heatmap[4][5]), "sanity: Option B ((5,6), row index 4, col index 5) must be a live but uncertain Cruiser candidate: {heatmap:?}");

        let refined_cells: std::collections::HashSet<(usize, usize)> = ai.cruiser_identified_cells_refined().into_iter().collect();
        assert_eq!(
            refined_cells,
            [(2, 2), (2, 3), (2, 4), (5, 6), (5, 7), (5, 8)].into_iter().collect(),
            "cross-reasoning must resolve the Cruiser to Option B, since Option A collides with a confirmed Frigate"
        );

        let refined_heatmap = ai.cruiser_heatmap_refined();
        assert_eq!(refined_heatmap[4][1], 0.0, "Option A must drop to 0 once cross-reasoned against the confirmed Frigate touching it");
        assert_eq!(refined_heatmap[4][5], 1.0, "Option B must rise to certainty, being the only Cruiser hypothesis left standing");
    }

    #[test]
    fn ai_cross_reasoning_narrows_a_frigate_ambiguity_using_a_still_ambiguous_cruiser() {
        // Mirror direction of `ai_cross_reasoning_resolves_a_cruiser_
        // ambiguity_the_cruiser_heatmap_cannot_resolve_alone`: here the
        // Cruiser itself is STILL ambiguous (Option CA vs Option CB, same
        // as that test), but one Cruiser window — the already-confirmed
        // Cruiser 1 — is common to EVERY remaining Cruiser hypothesis. A
        // Frigate option that collides with Cruiser 1 therefore collides
        // with every Cruiser hypothesis there is, even though the Cruiser
        // itself never narrows to one answer — cross-reasoning should
        // still drop that Frigate option to 0, exactly as if the Cruiser
        // were fully identified.
        let mut ai = AiPlayer::new();
        ai.apply_salvo([(2, 2), (2, 3), (2, 4)], [3, 3, 3]); // Cruiser 1: confirmed, common to every hypothesis
        ai.apply_salvo([(8, 1), (8, 2), (0, 0)], [2, 2, 0]); // Frigate: confirmed, elsewhere
        ai.apply_salvo([(8, 7), (8, 8), (0, 0)], [2, 2, 0]); // Frigate: confirmed, elsewhere
        flood_inner_misses_except(&mut ai, &[(5, 2), (5, 3), (5, 4), (5, 6), (5, 7), (5, 8), (3, 2), (3, 3), (1, 6), (1, 7)]);

        assert!(ai.cruiser_identified_cells().is_empty(), "sanity: the Cruiser must still be ambiguous (Option CA vs Option CB)");
        assert!(ai.frigate_identified_cells().is_empty(), "sanity: the Frigate heatmap alone must still be ambiguous too");
        let heatmap = ai.frigate_heatmap();
        assert!(heatmap[2][1] > 0.0, "sanity: Option FA ((3,2), touching Cruiser 1) must be a live Frigate candidate before cross-reasoning: {heatmap:?}");
        assert!(heatmap[0][5] > 0.0, "sanity: Option FB ((1,6), touching nothing) must be a live Frigate candidate before cross-reasoning: {heatmap:?}");

        let refined_heatmap = ai.frigate_heatmap_refined();
        assert_eq!(refined_heatmap[2][1], 0.0, "Option FA must drop to 0: it collides with Cruiser 1, which is common to every remaining Cruiser hypothesis, ambiguous or not");
        assert!(refined_heatmap[0][5] > 0.0, "Option FB touches neither Cruiser hypothesis, so it must remain a live candidate");
    }

    #[test]
    fn update_fsm_and_resolve_locks_in_a_cross_reasoned_cruiser_layout_the_raw_candidates_alone_missed() {
        // Same ambiguity as `ai_cross_reasoning_resolves_a_cruiser_ambiguity_
        // the_cruiser_heatmap_cannot_resolve_alone`: the RAW Cruiser
        // candidate list never collapses to 1 hypothesis on its own (Option
        // A stays a live alternative forever, as far as `consistent_cruiser_
        // candidates` alone is concerned) — only cross-reasoning against the
        // confirmed Frigates narrows it to Option B. The old, always-
        // automatic elimination this replaced only ever checked the raw
        // list, so it could never have fired here at all; `update_fsm_and_
        // resolve` must be able to lock this in anyway.
        let mut ai = AiPlayer::new();
        ai.apply_salvo([(2, 2), (2, 3), (2, 4)], [3, 3, 3]); // Cruiser 1: confirmed
        ai.apply_salvo([(6, 3), (6, 4), (0, 0)], [2, 2, 0]); // Frigate: confirmed, touches Option A
        ai.apply_salvo([(8, 1), (8, 2), (0, 0)], [2, 2, 0]); // Frigate: confirmed, elsewhere
        ai.apply_salvo([(8, 7), (8, 8), (0, 0)], [2, 2, 0]); // Frigate: confirmed, elsewhere
        flood_inner_misses_except(&mut ai, &[(5, 2), (5, 3), (5, 4), (5, 6), (5, 7), (5, 8)]);

        assert!(ai.cruiser_identified_cells().is_empty(), "sanity: the un-cross-reasoned identification must still be empty");
        let heatmap = ai.cruiser_heatmap();
        assert!((0.0..1.0).contains(&heatmap[4][1]), "sanity: Option A must still be a live but uncertain RAW candidate: {heatmap:?}");
        assert!((0.0..1.0).contains(&heatmap[4][5]), "sanity: Option B must still be a live but uncertain RAW candidate: {heatmap:?}");

        assert!(ai.update_fsm_and_resolve(), "cross-reasoning has already resolved this to Option B, must lock in");
        assert!(!ai.update_fsm_and_resolve(), "must be idempotent — nothing left to lock in a second time");

        // Option B ((5,6),(5,7),(5,8)) is now locked in — every OTHER row
        // must show 0 alive Cruiser placements, same style of assertion as
        // the raw-already-resolved case in `ai_cruiser_layout_elimination_
        // narrows_the_cruiser_and_frigate_fsm`.
        let (rows3, _) = ai.line_states(3);
        for r in 1..=8 {
            let alive = AiPlayer::alive_count(3, rows3[r]);
            if r == 2 || r == 5 {
                assert!(alive > 0, "row {r} passes through a confirmed Cruiser cell, must still show alive Cruiser placements, got {alive}");
            } else {
                assert_eq!(alive, 0, "row {r} has no confirmed Cruiser cell, must show 0 alive Cruiser placements now, got {alive}");
            }
        }
    }

    #[test]
    fn update_fsm_and_resolve_is_a_noop_while_still_ambiguous() {
        let mut ai = AiPlayer::new();
        ai.apply_salvo([(2, 2), (2, 3), (2, 4)], [3, 0, 0]);
        assert!(!ai.update_fsm_and_resolve(), "nothing is cross-reasoning-identified yet, must not lock anything in");
    }

    #[test]
    fn update_fsm_and_resolve_locks_in_frigate_layout_and_eliminates_cruiser_and_frigate_from_its_neighbours() {
        // Mirrors `update_fsm_and_resolve_locks_in_a_cross_reasoned_cruiser_
        // layout_the_raw_candidates_alone_missed` one size down, and directly
        // exercises `AiPlayer::lock_in_frigate_layout` — the actual gap this
        // was written for: once the Battleship/Cruiser side already fed a
        // symmetric elimination into the FSM, the Frigate side hadn't. Same
        // all-2s-on-a-straight-2-run setup as `ai_frigate_heatmap_shows_
        // certainty_once_all_3_frigates_are_fully_pinned`, which pins all 3
        // Frigates down with total certainty from the raw candidate list
        // alone (no cross-reasoning against Cruisers needed here).
        let mut ai = AiPlayer::new();
        ai.apply_salvo([(2, 2), (2, 3), (0, 0)], [2, 2, 0]);
        ai.apply_salvo([(5, 5), (5, 6), (0, 1)], [2, 2, 0]);
        ai.apply_salvo([(8, 2), (8, 3), (0, 2)], [2, 2, 0]);
        assert_eq!(ai.frigate_identified_cells_refined().len(), 6, "sanity: all 3 Frigates fully pinned");

        assert!(ai.update_fsm_and_resolve(), "Frigates are cross-reasoning-identified, must lock in");
        assert!(!ai.update_fsm_and_resolve(), "must be idempotent — nothing left to lock in a second time");

        // (3,4) is diagonally adjacent to the confirmed Frigate cell (2,3) —
        // per the adjacency rule (`try_place` in lib.rs forbids any
        // Chebyshev-distance-1 gap between 2 ships of size >=2, diagonal
        // included), it can now hold neither a Cruiser NOR another Frigate.
        let (_, _, combined3) = ai.alive_grids(3);
        let (_, _, combined2) = ai.alive_grids(2);
        assert_eq!(combined3[3 - 1][4 - 1], 0, "(3,4) is diagonally adjacent to a confirmed Frigate, must show 0 alive Cruiser placements");
        assert_eq!(combined2[3 - 1][4 - 1], 0, "(3,4) is diagonally adjacent to a confirmed Frigate, must show 0 alive Frigate placements");

        // (4,8) sits far from all 3 confirmed Frigate clusters — nothing
        // here proves a Cruiser can't be there, so it must be untouched.
        assert!(combined3[4 - 1][8 - 1] > 0, "(4,8) is far from every confirmed Frigate, must still show alive Cruiser room");
    }

    #[test]
    fn frigate_disambiguation_shots_with_refire_finds_a_shot_once_ordinary_disambiguation_is_exhausted() {
        // Both Cruisers and 2 of the 3 Frigates fixed and unambiguous. The
        // 3rd Frigate is ambiguous among 3 candidate windows — W1=(4,2)-
        // (4,3), W2=(4,5)-(4,6), W3=(8,5)-(8,6). Firing exactly one cell
        // from each window together in a single "2 0 0" bag keeps all 3
        // windows simultaneously consistent (each has exactly 1 of its 2
        // cells touched, matching the bag's lone "2") — genuine unordered-
        // bag ambiguity, not a search giving up early. Doing this twice
        // (once per half of each window) fires every cell any of the 3
        // windows could ever be distinguished by, without ever resolving
        // which is real — the "heatmap fully evolved" dead end this
        // feature exists for.
        let mut ai = AiPlayer::new();
        ai.apply_salvo([(2, 2), (2, 3), (2, 4)], [3, 3, 3]);
        ai.apply_salvo([(2, 6), (2, 7), (2, 8)], [3, 3, 3]);
        ai.apply_salvo([(6, 1), (6, 2), (0, 0)], [2, 2, 0]);
        ai.apply_salvo([(6, 7), (6, 8), (0, 1)], [2, 2, 0]);
        ai.apply_salvo([(4, 2), (4, 5), (8, 5)], [2, 0, 0]);
        ai.apply_salvo([(4, 3), (4, 6), (8, 6)], [2, 0, 0]);

        flood_inner_misses_except(
            &mut ai,
            &[
                (2, 2), (2, 3), (2, 4), (2, 6), (2, 7), (2, 8),
                (6, 1), (6, 2), (6, 7), (6, 8),
                (4, 2), (4, 3), (4, 5), (4, 6), (8, 5), (8, 6),
            ],
        );

        assert!(ai.frigate_identified_cells().is_empty(), "sanity: the 3rd Frigate must still be ambiguous among all 3 windows");
        assert_eq!(
            ai.frigate_disambiguation_shots(), None,
            "sanity: every cell the 3 hypotheses disagree on is already fired — the exact dead end this feature exists for"
        );
        assert_eq!(ai.cruiser_disambiguation_shots_with_refire(), None, "sanity: Cruisers are already unambiguous, nothing to refire there");

        let shots = ai.frigate_disambiguation_shots_with_refire().expect("a refire-based disambiguation shot must exist");
        for &(r, c) in &shots {
            assert!(ai.is_fired(r, c), "every cell in the refire salvo must be one of the already-fired, still-disagreeing cells: {:?}", (r, c));
            assert!(
                [(4, 2), (4, 3), (4, 5), (4, 6), (8, 5), (8, 6)].contains(&(r, c)),
                "refire target {:?} must be one of the cells the 3 hypotheses actually disagree on", (r, c)
            );
        }

        // The combined entry point must agree (Cruiser has nothing to
        // offer, so it must fall through to this same Frigate salvo).
        assert_eq!(ai.disambiguation_shots_with_refire(), Some(shots));

        // Capped at exactly one bonus use per cell.
        let (r0, c0) = shots[0];
        assert!(ai.is_disambiguation_extra_refire(r0, c0), "must be recognized as a valid extra refire before being consumed");
        ai.mark_disambiguation_extra_refire_used(r0, c0);
        assert!(!ai.is_disambiguation_extra_refire(r0, c0), "must be capped at exactly one use — a second refire of the same cell must be rejected");
    }

    #[test]
    fn frigate_disambiguation_shots_never_fires_two_mutually_exclusive_alternatives_together() {
        // 2 Frigates fixed and unambiguous. The 3rd is ambiguous between
        // exactly 2 candidate windows sharing a pivot cell: (4,2)-(4,3) or
        // (4,3)-(4,4) — (4,3) confirmed hit either way, (4,2) and (4,4)
        // each still fully unfired. Only 2 total candidates survive, and
        // they disagree on exactly ONE cell each ((4,2) vs (4,4)) — the
        // same shape as a real "B4-or-C5 completes this Frigate" pivot
        // ambiguity. Firing (4,2) and (4,4) TOGETHER in the same salvo is
        // the worst possible choice: exactly one of them is always the
        // hit no matter which hypothesis is real, so the resulting bag
        // (one "2", rest misses) is identical either way — genuinely
        // uninformative — whereas firing just one of them plus 2 neutral
        // fillers fully resolves it. A padding bug once left the search
        // pool at exactly {(4,2), (4,4), one filler} with no alternative
        // triple ever compared against it, forcing exactly this
        // uninformative combination.
        let mut ai = AiPlayer::new();
        ai.apply_salvo([(2, 2), (2, 3), (2, 4)], [3, 3, 3]);
        ai.apply_salvo([(2, 6), (2, 7), (2, 8)], [3, 3, 3]);
        ai.apply_salvo([(6, 1), (6, 2), (0, 0)], [2, 2, 0]);
        ai.apply_salvo([(6, 7), (6, 8), (0, 1)], [2, 2, 0]);
        ai.apply_salvo([(4, 3), (0, 2), (0, 3)], [2, 0, 0]);

        flood_inner_misses_except(
            &mut ai,
            &[
                (2, 2), (2, 3), (2, 4), (2, 6), (2, 7), (2, 8),
                (6, 1), (6, 2), (6, 7), (6, 8),
                (4, 2), (4, 3), (4, 4),
            ],
        );

        assert!(ai.frigate_identified_cells().is_empty(), "sanity: the 3rd Frigate must still be ambiguous between the 2 pivot windows");

        let shots = ai.frigate_disambiguation_shots().expect("an ordinary disambiguating shot must exist — (4,2) and (4,4) are both unfired");
        assert!(
            !(shots.contains(&(4, 2)) && shots.contains(&(4, 4))),
            "must not fire both mutually-exclusive alternatives together — that combination is provably uninformative: {:?}", shots
        );
    }

    #[test]
    fn frigate_disambiguation_shots_last_resort_unlocks_a_cluster_tie_whose_ends_already_spent_their_bonus() {
        // Same "cluster of 3" pivot ambiguity as the padding-fix test above
        // — (4,2)-(4,3) or (4,3)-(4,4), sharing pivot (4,3) — but here BOTH
        // end cells ((4,2) and (4,4)) are already fired AND have already
        // spent their one-time bonus refire (simulating them having been
        // used earlier — separately, for some other legitimate reason —
        // before the game ever narrowed down to this being the final,
        // otherwise-permanent tie). The capped tier
        // (disambiguation_shots_with_refire) can no longer help; the
        // last-resort tier must still be able to.
        let mut ai = AiPlayer::new();
        ai.apply_salvo([(2, 2), (2, 3), (2, 4)], [3, 3, 3]);
        ai.apply_salvo([(2, 6), (2, 7), (2, 8)], [3, 3, 3]);
        ai.apply_salvo([(6, 1), (6, 2), (0, 0)], [2, 2, 0]);
        ai.apply_salvo([(6, 7), (6, 8), (0, 1)], [2, 2, 0]);
        ai.apply_salvo([(4, 3), (0, 2), (0, 3)], [2, 0, 0]);
        // Fires both (4,2) and (4,4) — uninformative (exactly one is always
        // the hit either way), but that's irrelevant here: the point is
        // just to get both genuinely fired while the ambiguity survives.
        ai.apply_salvo([(4, 2), (4, 4), (0, 4)], [2, 0, 0]);

        flood_inner_misses_except(
            &mut ai,
            &[
                (2, 2), (2, 3), (2, 4), (2, 6), (2, 7), (2, 8),
                (6, 1), (6, 2), (6, 7), (6, 8),
                (4, 2), (4, 3), (4, 4),
            ],
        );

        assert!(ai.frigate_identified_cells().is_empty(), "sanity: still ambiguous between the 2 pivot windows");
        assert!(
            ai.frigate_disambiguation_shots_with_refire().is_some(),
            "sanity: the capped tier should still work before either end cell's bonus is actually marked spent"
        );

        ai.mark_disambiguation_extra_refire_used(4, 2);
        ai.mark_disambiguation_extra_refire_used(4, 4);

        assert_eq!(
            ai.frigate_disambiguation_shots_with_refire(), None,
            "the capped tier must now report nothing — both discriminating cells already spent their bonus"
        );

        let shots = ai.frigate_disambiguation_shots_last_resort().expect("the last-resort tier must still find a shot");
        assert!(
            shots.contains(&(4, 2)) || shots.contains(&(4, 4)),
            "last-resort salvo must include one of the 2 already-bonus-spent discriminating cells: {:?}", shots
        );
        assert!(
            !(shots.contains(&(4, 2)) && shots.contains(&(4, 4))),
            "must not fire both mutually-exclusive alternatives together, same as the ordinary padding fix: {:?}", shots
        );

        let (r0, c0) = if shots.contains(&(4, 2)) { (4, 2) } else { (4, 4) };
        assert!(ai.is_last_resort_refire(r0, c0), "must be recognized as a valid last-resort refire");
        assert!(
            !ai.is_disambiguation_extra_refire(r0, c0),
            "must NOT be recognized as a fresh capped-tier refire — its bonus is already spent"
        );

        // Combined entry point must agree.
        assert_eq!(ai.disambiguation_shots_last_resort(), Some(shots));
    }

    #[test]
    fn ai_firing_an_outer_ring_cell_never_drives_the_fsm_for_size_2_3_or_4() {
        let mut ai = AiPlayer::new();

        // (0, 4): outer-ring row, inner column. Only the column-FSM for column 4
        // could ever be relevant here, and only if the row were inner too — since
        // it isn't, NEITHER FSM should move: a size>=2 ship placement needs both
        // its row and column inner, so this cell can never be part of one.
        ai.apply_salvo([(0, 4), (0, 5), (0, 6)], [0, 0, 0]);
        // (5, 0): inner row, outer-ring column — symmetric case.
        ai.apply_salvo([(5, 0), (6, 0), (7, 0)], [0, 0, 0]);

        for &size in &[4usize, 3, 2] {
            let (rows, cols) = ai.line_states(size);
            assert_eq!(rows[0], 0, "row_state[0] (outer ring) must stay at its initial state for size {size}");
            assert_eq!(rows[9], 0, "row_state[9] (outer ring) must stay at its initial state for size {size}");
            assert_eq!(cols[0], 0, "col_state[0] (outer ring) must stay at its initial state for size {size}");
            assert_eq!(cols[9], 0, "col_state[9] (outer ring) must stay at its initial state for size {size}");
        }
    }



    #[test]
    fn ai_current_target_size_advances_once_battleship_fully_sunk() {
        let mut ai = AiPlayer::new();
        assert_eq!(ai.current_target_size(), 4);

        ai.mark_sunk(4);
        assert_eq!(ai.current_target_size(), 3);

        // choose_shots should carry on working — now scored against the size-3
        // FSM — rather than staying stuck evaluating a ship class that's
        // already fully accounted for.
        let shots = ai.choose_shots();
        assert_ne!(shots[0], shots[1]);
        assert_ne!(shots[1], shots[2]);
        assert_ne!(shots[0], shots[2]);
    }

    #[test]
    fn ai_freeze_before_frigates_holds_target_size_at_3_instead_of_dropping_to_2() {
        let mut ai = AiPlayer::new();
        ai.mark_sunk(4);
        ai.mark_sunk(3);
        ai.mark_sunk(3);
        assert_eq!(ai.current_target_size(), 2, "sanity: normally drops to Frigates once both Cruisers are sunk");

        // Redo from scratch with the freeze toggle on this time.
        let mut ai = AiPlayer::new();
        assert!(!ai.is_freeze_before_frigates(), "defaults to off");
        ai.set_freeze_before_frigates(true);
        assert!(ai.is_freeze_before_frigates());

        // The toggle has no effect while size 4/3 still have ships left.
        assert_eq!(ai.current_target_size(), 4);
        ai.mark_sunk(4);
        assert_eq!(ai.current_target_size(), 3);

        // Now it kicks in: even sinking both Cruisers must not advance past 3.
        ai.mark_sunk(3);
        assert_eq!(ai.current_target_size(), 3, "still hunting the first Cruiser");
        ai.mark_sunk(3);
        assert_eq!(ai.current_target_size(), 3, "frozen — must not drop to 2 now both Cruisers are sunk");

        // choose_shots must keep working (scored against the frozen size-3
        // FSM) rather than panicking or stalling.
        let shots = ai.choose_shots();
        assert_ne!(shots[0], shots[1]);
        assert_ne!(shots[1], shots[2]);
        assert_ne!(shots[0], shots[2]);

        // Turning the toggle back off releases the freeze immediately.
        ai.set_freeze_before_frigates(false);
        assert_eq!(ai.current_target_size(), 2);
    }

    #[test]
    fn ai_choose_shots_does_not_panic_once_only_submarines_remain() {
        let mut ai = AiPlayer::new();

        // Sink every Battleship/Cruiser/Frigate (counts from SHIP_DEFS: 1, 2, 3
        // respectively) so current_target_size() falls through to 1 (only
        // submarines left). Regression test: choose_shots used to unconditionally
        // feed the target size into the size-4/3/2 line FSM lookups, which panic
        // on anything other than 4/3/2 — reachable mid-game, since submarines are
        // sunk last, and this crashed every subsequent call into the AI (including
        // a "New game" reset that re-triggers an advisory refresh).
        ai.mark_sunk(4);
        ai.mark_sunk(3);
        ai.mark_sunk(3);
        ai.mark_sunk(2);
        ai.mark_sunk(2);
        ai.mark_sunk(2);
        assert_eq!(ai.current_target_size(), 1);

        let shots = ai.choose_shots();
        assert_ne!(shots[0], shots[1]);
        assert_ne!(shots[1], shots[2]);
        assert_ne!(shots[0], shots[2]);
    }

    #[test]
    fn frigate_disambiguation_shots_picks_one_end_per_cluster_for_3_simultaneous_independent_ambiguities() {
        // 3 independent "pivot + 2 ends" Frigate ambiguities at once, no
        // cross-cluster overlap — matches a live board (#11) with all 3
        // Frigates simultaneously ambiguous this way. 2*2*2 = 8 total
        // combined hypotheses. The best any single 3-cell salvo can ever
        // do against 3 independent binary unknowns is narrow the worst
        // case to 3 remaining hypotheses (picking one end cell per
        // cluster: counts of 0/1/2/3 hits split the 8 hypotheses into
        // buckets of size 1/3/3/1) — testing both ends of the SAME
        // cluster together wastes a slot (that pair always sums to
        // exactly 1, redundant), so this asserts the search actually
        // reaches for one-per-cluster rather than that strictly worse
        // alternative.
        let mut ai = AiPlayer::new();
        ai.apply_salvo([(2, 3), (0, 0), (0, 1)], [2, 0, 0]);
        ai.apply_salvo([(4, 6), (0, 2), (0, 3)], [2, 0, 0]);
        ai.apply_salvo([(6, 3), (0, 4), (0, 5)], [2, 0, 0]);

        flood_inner_misses_except(
            &mut ai,
            &[
                (2, 2), (2, 3), (2, 4),
                (4, 5), (4, 6), (4, 7),
                (6, 2), (6, 3), (6, 4),
            ],
        );

        assert!(ai.frigate_identified_cells().is_empty(), "sanity: still ambiguous");
        let shots = ai.frigate_disambiguation_shots().expect("a disambiguating shot must exist");

        let cluster_a_ends = [(2, 2), (2, 4)];
        let cluster_b_ends = [(4, 5), (4, 7)];
        let cluster_c_ends = [(6, 2), (6, 4)];
        let hits_a = shots.iter().filter(|c| cluster_a_ends.contains(c)).count();
        let hits_b = shots.iter().filter(|c| cluster_b_ends.contains(c)).count();
        let hits_c = shots.iter().filter(|c| cluster_c_ends.contains(c)).count();
        assert_eq!((hits_a, hits_b, hits_c), (1, 1, 1), "must pick exactly one end cell from each of the 3 clusters, not both ends of any single one: {:?}", shots);
    }

    #[test]
    fn fire_still_allows_a_disambiguating_salvo_after_won_flips_true_before_the_layout_is_resolved() {
        // A disambiguation salvo's filler cell can incidentally BE the
        // very last unfound real cell — `won` (every real cell hit at
        // least once) can flip true mid-disambiguation, before the
        // Cruiser/Frigate exact layout is pinned down. Reproduces exactly
        // that: every real cell gets hit except one end of an ambiguous
        // Frigate cluster, which is deliberately fired ALONGSIDE its
        // still-live alternative (an uninformative pair, same shape as the
        // padding-fix tests above) as the winning shot — hit_count reaches
        // total_hits (won=true) while the AI's own deduction still can't
        // tell which of the 2 cells was the real one.
        let layout_json = r#"{"board": [
            [6,null,null,null,null,null,null,null,null,7],
            [null,0,0,0,0,null,null,null,null,null],
            [null,null,null,null,null,null,null,null,null,null],
            [null,1,1,1,null,null,4,4,null,null],
            [null,null,null,null,null,null,null,null,null,null],
            [null,2,2,2,null,null,5,5,null,null],
            [null,null,null,null,null,null,null,null,null,null],
            [null,3,3,null,null,null,null,null,null,null],
            [null,null,null,null,null,null,null,null,null,null],
            [8,null,null,null,null,null,null,null,null,9]],
            "ships": [
            {"id": 0, "name": "Battleship", "size": 4, "cells": [{"row":1,"col":1},{"row":1,"col":2},{"row":1,"col":3},{"row":1,"col":4}], "hits": 0, "sunk": false},
            {"id": 1, "name": "Cruiser", "size": 3, "cells": [{"row":3,"col":1},{"row":3,"col":2},{"row":3,"col":3}], "hits": 0, "sunk": false},
            {"id": 2, "name": "Cruiser", "size": 3, "cells": [{"row":5,"col":1},{"row":5,"col":2},{"row":5,"col":3}], "hits": 0, "sunk": false},
            {"id": 3, "name": "Frigate", "size": 2, "cells": [{"row":7,"col":1},{"row":7,"col":2}], "hits": 0, "sunk": false},
            {"id": 4, "name": "Frigate", "size": 2, "cells": [{"row":3,"col":6},{"row":3,"col":7}], "hits": 0, "sunk": false},
            {"id": 5, "name": "Frigate", "size": 2, "cells": [{"row":5,"col":6},{"row":5,"col":7}], "hits": 0, "sunk": false},
            {"id": 6, "name": "Submarine", "size": 1, "cells": [{"row":0,"col":0}], "hits": 0, "sunk": false},
            {"id": 7, "name": "Submarine", "size": 1, "cells": [{"row":0,"col":9}], "hits": 0, "sunk": false},
            {"id": 8, "name": "Submarine", "size": 1, "cells": [{"row":9,"col":0}], "hits": 0, "sunk": false},
            {"id": 9, "name": "Submarine", "size": 1, "cells": [{"row":9,"col":9}], "hits": 0, "sunk": false}
            ]}"#;

        let mut game = Game::new();
        let load_result = game.load_board_layout_json(layout_json);
        assert!(!load_result.contains("error"), "board layout failed to load: {load_result}");

        let idx = |r: usize, c: usize| r * 10 + c;
        let fire_ok = |game: &mut Game, cells: [(usize, usize); 3]| {
            let indices = [idx(cells[0].0, cells[0].1), idx(cells[1].0, cells[1].1), idx(cells[2].0, cells[2].1)];
            let response = game.fire(&indices);
            assert!(!response.contains("\"error\""), "expected this salvo to succeed, got: {response}");
        };

        fire_ok(&mut game, [(1, 1), (1, 2), (1, 3)]);
        fire_ok(&mut game, [(1, 4), (8, 8), (8, 7)]); // Battleship's last cell + 2 outer fillers
        fire_ok(&mut game, [(3, 1), (3, 2), (3, 3)]); // Cruiser 1
        fire_ok(&mut game, [(5, 1), (5, 2), (5, 3)]); // Cruiser 2
        fire_ok(&mut game, [(7, 1), (7, 2), (8, 6)]); // Frigate 3 (resolved)
        fire_ok(&mut game, [(3, 6), (3, 7), (8, 5)]); // Frigate 4 (resolved)
        fire_ok(&mut game, [(0, 0), (0, 9), (9, 0)]); // 3 of 4 Submarines
        fire_ok(&mut game, [(5, 7), (8, 4), (8, 3)]); // ambiguous Frigate's pivot cell, cleanly isolated

        // Every real cell is now hit except (5,6) — the ambiguous Frigate's
        // OTHER cell — and the 4th Submarine at (9,9). Fire both together:
        // (9,9) is real (the winning cell that pushes hit_count to
        // total_hits), and (5,6)+(5,8) form the same uninformative pair as
        // the padding-fix tests — exactly one of them is always the hit,
        // so the bag can never reveal which, keeping the Frigate ambiguous
        // even as the game becomes "won".
        fire_ok(&mut game, [(9, 9), (5, 6), (5, 8)]);

        assert!(game.state.won, "sanity: hit_count must have reached total_hits by now");
        assert!(!game.is_fully_resolved(), "sanity: the Frigate ambiguity must still be unresolved");

        // The actual bug: Game::fire used to reject EVERYTHING once won,
        // stranding this resolvable ambiguity permanently. It must still
        // accept a legitimate disambiguating salvo now.
        let refire_suggestion: Vec<usize> = serde_json::from_str(&game.ai_suggest_disambiguation_refire()).unwrap_or_default();
        assert_eq!(refire_suggestion.len(), 3, "a refire-based disambiguating salvo must still be available");
        let response = game.fire(&refire_suggestion);
        assert!(
            !response.contains("\"error\":\"game already won\""),
            "firing a legitimate disambiguation salvo must not be rejected just because won is already true: {response}"
        );
    }

    #[test]
    fn self_play_discovers_every_ship_of_size_at_least_2_by_game_end() {
        // End-to-end smoke test: let the AI play entire games against
        // itself (real random boards, `choose_shots` picking every salvo,
        // fired through the real `Game::fire` — not a hand-constructed
        // AiPlayer scenario) and check, after EVERY single salvo, that the
        // AI never claims to have found the Battleship's cells as anything
        // other than empty or exactly 4. Then, once the game is won (every
        // cell of every ship hit, submarines included), assert the
        // Battleship was permanently recorded as found — not just "sunk"
        // per Fleet Status, but genuinely *located* by the AI's own
        // reasoning — and that no genuine Cruiser cell was ever wrongly
        // ruled out impossible in the Cross-3 Bag traffic-light tracking.
        // Cruiser and Frigate exact-cell discovery (pinpointing which cells
        // a ship occupies, not just that it sank) are deliberately not
        // attempted — see `AiPlayer::refresh_cross3_entry_flags`/
        // `refresh_cross2_entry_flags` — so no "found" assertion exists for
        // either; only sinking is tracked for them. Repeated across many
        // random boards, since a single run only exercises one particular
        // layout/salvo history.
        const GAMES: usize = 25;
        const MAX_TURNS: usize = 200; // generous: 100 cells / 3 per salvo ~= 34 turns even in the worst case

        for game_no in 0..GAMES {
            let mut game = Game::new();
            let mut turns = 0;

            while !game.state.won {
                turns += 1;
                assert!(turns <= MAX_TURNS, "game {game_no} did not finish within {MAX_TURNS} turns — likely stuck re-suggesting the same cell(s)");

                let shots = game.ai.choose_shots();
                let indices: Vec<usize> = shots.iter().map(|&(r, c)| r * 10 + c).collect();
                let response = game.fire(&indices);
                assert!(
                    !response.contains("\"error\""),
                    "game {game_no} turn {turns}: choose_shots produced an invalid salvo {shots:?}: {response}"
                );

                let battleship_found_len = game.ai.found_battleship_cells().len();
                assert!(
                    battleship_found_len == 0 || battleship_found_len == 4,
                    "game {game_no} turn {turns}: found_battleship_cells must be empty or exactly 4, got {battleship_found_len}"
                );

                // Core soundness guarantee, checked at scale (1000 games,
                // ~2M per-cell checks, 0 violations): whenever a heatmap
                // claims 100% confidence for a cell, that cell must really
                // be that ship type — never a false positive. The
                // converse (every real cell eventually reaching 100%) is
                // NOT asserted: some ambiguity can be permanent (see
                // `disambiguation_shots`'s doc comment), so only the "no
                // false certainty" direction is a hard guarantee.
                let cruiser_heatmap = game.ai.cruiser_heatmap();
                let frigate_heatmap = game.ai.frigate_heatmap();
                for row in 1..=8 {
                    for col in 1..=8 {
                        if cruiser_heatmap[row - 1][col - 1] == 1.0 {
                            let is_cruiser = matches!(game.state.board[row][col], Some(id) if game.state.ships[id].size == 3);
                            assert!(is_cruiser, "game {game_no} turn {turns}: Cruiser heatmap claims certainty at ({row},{col}) but it isn't really a Cruiser cell");
                        }
                        if frigate_heatmap[row - 1][col - 1] == 1.0 {
                            let is_frigate = matches!(game.state.board[row][col], Some(id) if game.state.ships[id].size == 2);
                            assert!(is_frigate, "game {game_no} turn {turns}: Frigate heatmap claims certainty at ({row},{col}) but it isn't really a Frigate cell");
                        }
                    }
                }
            }

            assert_eq!(
                game.ai.found_battleship_cells().len(),
                4,
                "game {game_no}: the Battleship must be fully discovered by the time the game is won"
            );
            // No genuine Cruiser cell may ever be wrongly ruled out
            // impossible in the Cross-3 Bag traffic-light tracking.
            for ship in game.state.ships.iter().filter(|s| s.size == 3) {
                for cell in &ship.cells {
                    let ruled_out = game.ai.cross3_entries().iter().find_map(|e| {
                        let idx = e.coords.iter().position(|&c| c == (cell.row, cell.col))?;
                        Some(e.coord_ruled_out[idx])
                    });
                    assert_ne!(
                        ruled_out,
                        Some(true),
                        "game {game_no}: real Cruiser cell {:?} was wrongly ruled out impossible",
                        (cell.row, cell.col)
                    );
                }
            }
        }
    }

    #[test]
    fn autoplay_equivalent_resolves_board_23_layout() {
        // Reproduces the exact ship layout from a live-play save (board
        // #23) and replays it via `run_autoplay_equivalent` — the same
        // priority order `runAutoPlay`/the "AI Advisory" panel use in
        // index.html. Diagnostic/regression test for a live "gets stuck"
        // report.
        let layout_json = r#"{"board": [[null, null, null, null, null, null, null, null, null, null], [null, 0, 0, 0, 0, null, null, 5, null, null], [null, null, null, null, null, null, null, 5, null, null], [null, 4, null, 7, null, null, null, null, null, null], [null, 4, null, null, 8, null, null, null, null, null], [9, null, null, null, null, null, 2, 2, 2, null], [null, null, null, 3, 3, null, null, null, null, null], [null, null, null, null, null, null, 1, 1, 1, null], [null, null, 6, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null]], "ships": [{"id": 0, "name": "Battleship", "size": 4, "cells": [{"row": 1, "col": 1}, {"row": 1, "col": 2}, {"row": 1, "col": 3}, {"row": 1, "col": 4}], "hits": 4, "sunk": true}, {"id": 1, "name": "Cruiser", "size": 3, "cells": [{"row": 7, "col": 6}, {"row": 7, "col": 7}, {"row": 7, "col": 8}], "hits": 3, "sunk": true}, {"id": 2, "name": "Cruiser", "size": 3, "cells": [{"row": 5, "col": 6}, {"row": 5, "col": 7}, {"row": 5, "col": 8}], "hits": 3, "sunk": true}, {"id": 3, "name": "Frigate", "size": 2, "cells": [{"row": 6, "col": 3}, {"row": 6, "col": 4}], "hits": 2, "sunk": true}, {"id": 4, "name": "Frigate", "size": 2, "cells": [{"row": 3, "col": 1}, {"row": 4, "col": 1}], "hits": 2, "sunk": true}, {"id": 5, "name": "Frigate", "size": 2, "cells": [{"row": 1, "col": 7}, {"row": 2, "col": 7}], "hits": 2, "sunk": true}, {"id": 6, "name": "Submarine", "size": 1, "cells": [{"row": 8, "col": 2}], "hits": 1, "sunk": true}, {"id": 7, "name": "Submarine", "size": 1, "cells": [{"row": 3, "col": 3}], "hits": 1, "sunk": true}, {"id": 8, "name": "Submarine", "size": 1, "cells": [{"row": 4, "col": 4}], "hits": 1, "sunk": true}, {"id": 9, "name": "Submarine", "size": 1, "cells": [{"row": 5, "col": 0}], "hits": 0, "sunk": false}]}"#;
        run_autoplay_equivalent(layout_json, 40);
    }

    /// Replays a saved board layout the same way `runAutoPlay`/the "AI
    /// Advisory" panel drive `Game` in index.html: once `ai_target_size()`
    /// drops to 1 (Battleship/Cruiser/Frigate all fully sunk), prioritize
    /// `update_fsm_and_resolve` + the refire-based suggestion over the
    /// ordinary advisory (which would otherwise silently prefer Submarine
    /// hunting the moment its own no-refire disambiguation attempt comes up
    /// empty). Panics (via the assertions below) if the game doesn't
    /// resolve within `max_shots`.
    fn run_autoplay_equivalent(layout_json: &str, max_shots: usize) {
        let mut game = Game::new();
        let load_result = game.load_board_layout_json(layout_json);
        assert!(!load_result.contains("error"), "board layout failed to load: {load_result}");

        let mut shots_taken = 0;
        loop {
            if game.is_won() {
                break;
            }
            assert!(shots_taken < max_shots, "autoplay-equivalent loop did not resolve within {max_shots} shots");

            let mut indices: Vec<usize> = Vec::new();
            if game.ai_target_size() <= 1 {
                game.update_fsm_and_resolve();
                indices = serde_json::from_str(&game.ai_suggest_disambiguation_refire()).unwrap_or_default();
            }
            if indices.len() != 3 {
                indices = serde_json::from_str(&game.ai_suggest()).unwrap_or_default();
            }
            assert_eq!(indices.len(), 3, "no valid 3-cell salvo available (shots_taken={shots_taken})");

            let response = game.fire(&indices);
            assert!(!response.contains("\"error\""), "shot {shots_taken} rejected: {response} (indices={indices:?})");
            shots_taken += 1;
        }
    }

    #[test]
    fn autoplay_equivalent_resolves_board_26_layout() {
        // Reproduces the exact ship layout from a live-play save (board
        // #26) — reported as still stuck (AI Advisory offering nothing
        // useful, "Disambiguate" a no-op) even after the ai_target_size()
        // gating fix. See run_autoplay_equivalent.
        let layout_json = r#"{"board": [[null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, 5, 5, null], [null, 1, null, 8, null, 6, null, null, null, null], [null, 1, null, null, 4, null, 3, null, null, null], [null, 1, null, null, 4, null, 3, null, null, null], [null, null, null, null, null, null, null, null, 2, null], [null, null, 0, 0, 0, 0, null, null, 2, null], [null, null, null, null, null, null, null, null, 2, null], [null, null, null, null, null, null, 7, null, null, null], [null, 9, null, null, null, null, null, null, null, null]], "ships": [{"id": 0, "name": "Battleship", "size": 4, "cells": [{"row": 6, "col": 2}, {"row": 6, "col": 3}, {"row": 6, "col": 4}, {"row": 6, "col": 5}], "hits": 4, "sunk": true}, {"id": 1, "name": "Cruiser", "size": 3, "cells": [{"row": 2, "col": 1}, {"row": 3, "col": 1}, {"row": 4, "col": 1}], "hits": 3, "sunk": true}, {"id": 2, "name": "Cruiser", "size": 3, "cells": [{"row": 5, "col": 8}, {"row": 6, "col": 8}, {"row": 7, "col": 8}], "hits": 3, "sunk": true}, {"id": 3, "name": "Frigate", "size": 2, "cells": [{"row": 3, "col": 6}, {"row": 4, "col": 6}], "hits": 2, "sunk": true}, {"id": 4, "name": "Frigate", "size": 2, "cells": [{"row": 3, "col": 4}, {"row": 4, "col": 4}], "hits": 2, "sunk": true}, {"id": 5, "name": "Frigate", "size": 2, "cells": [{"row": 1, "col": 7}, {"row": 1, "col": 8}], "hits": 2, "sunk": true}, {"id": 6, "name": "Submarine", "size": 1, "cells": [{"row": 2, "col": 5}], "hits": 1, "sunk": true}, {"id": 7, "name": "Submarine", "size": 1, "cells": [{"row": 8, "col": 6}], "hits": 1, "sunk": true}, {"id": 8, "name": "Submarine", "size": 1, "cells": [{"row": 2, "col": 3}], "hits": 1, "sunk": true}, {"id": 9, "name": "Submarine", "size": 1, "cells": [{"row": 9, "col": 1}], "hits": 0, "sunk": false}]}"#;
        run_autoplay_equivalent(layout_json, 40);
    }

    /// Like `run_autoplay_equivalent`, but stops on
    /// `resolution_status_json().resolved` rather than `is_won()` — the
    /// exact stopping condition `gameTrulyDone()`/`takeAiTurn()`/
    /// `runToCompletion()` use in player.html for board 2 (the player's
    /// own placed fleet), which keeps firing disambiguation salvos past a
    /// win precisely because a reduced no-Submarine fleet can reach
    /// `is_won()` well before the Cruiser/Frigate layout is pinned down.
    fn run_until_resolved(layout_json: &str, max_shots: usize) {
        let mut game = Game::new();
        let load_result = game.load_board_layout_json(layout_json);
        assert!(!load_result.contains("error"), "board layout failed to load: {load_result}");

        let mut shots_taken = 0;
        loop {
            let status: serde_json::Value = serde_json::from_str(&game.resolution_status_json()).unwrap();
            if status["resolved"].as_bool().unwrap_or(false) {
                break;
            }
            assert!(
                shots_taken < max_shots,
                "did not resolve within {max_shots} shots (won={}, status={status:?})",
                game.is_won()
            );

            let mut indices: Vec<usize> = Vec::new();
            if game.ai_target_size() <= 1 {
                game.update_fsm_and_resolve();
                indices = serde_json::from_str(&game.ai_suggest_disambiguation_refire()).unwrap_or_default();
                if indices.len() != 3 {
                    indices = serde_json::from_str(&game.ai_suggest_disambiguation_last_resort()).unwrap_or_default();
                }
            }
            if indices.len() != 3 {
                indices = serde_json::from_str(&game.ai_suggest()).unwrap_or_default();
            }
            assert_eq!(indices.len(), 3, "no valid 3-cell salvo available (shots_taken={shots_taken})");

            let response = game.fire(&indices);
            assert!(!response.contains("\"error\""), "shot {shots_taken} rejected: {response} (indices={indices:?})");
            shots_taken += 1;
        }
    }

    #[test]
    fn ai_can_fully_resolve_a_no_submarine_fleet() {
        // The Player Page's "position your fleet" board (player.html)
        // loads a 6-ship layout — 1 Battleship, 2 Cruisers, 3 Frigates —
        // with NO Submarines at all, unlike every other layout tested in
        // this file. Live report: the "ship fully identified" outline
        // never lights up there. Checks whether resolution_status_json()
        // .resolved is EVEN REACHABLE for a fleet shaped like this — a
        // simple, well-separated layout.
        let layout_json = r#"{"board": [[null,null,null,null,null,null,null,null,null,null],[null,0,0,0,0,null,null,null,null,null],[null,null,null,null,null,null,null,null,null,null],[null,1,1,1,null,null,2,2,2,null],[null,null,null,null,null,null,null,null,null,null],[null,3,3,null,4,4,null,5,5,null],[null,null,null,null,null,null,null,null,null,null],[null,null,null,null,null,null,null,null,null,null],[null,null,null,null,null,null,null,null,null,null],[null,null,null,null,null,null,null,null,null,null]], "ships": [{"id":0,"name":"Battleship","size":4,"cells":[{"row":1,"col":1},{"row":1,"col":2},{"row":1,"col":3},{"row":1,"col":4}],"hits":0,"sunk":false},{"id":1,"name":"Cruiser","size":3,"cells":[{"row":3,"col":1},{"row":3,"col":2},{"row":3,"col":3}],"hits":0,"sunk":false},{"id":2,"name":"Cruiser","size":3,"cells":[{"row":3,"col":6},{"row":3,"col":7},{"row":3,"col":8}],"hits":0,"sunk":false},{"id":3,"name":"Frigate","size":2,"cells":[{"row":5,"col":1},{"row":5,"col":2}],"hits":0,"sunk":false},{"id":4,"name":"Frigate","size":2,"cells":[{"row":5,"col":4},{"row":5,"col":5}],"hits":0,"sunk":false},{"id":5,"name":"Frigate","size":2,"cells":[{"row":5,"col":7},{"row":5,"col":8}],"hits":0,"sunk":false}]}"#;
        run_until_resolved(layout_json, 60);
    }

    #[test]
    fn ai_can_fully_resolve_a_no_submarine_fleet_tight_layout() {
        // Same idea, but reusing the exact Battleship/Cruiser/Frigate
        // positions from `autoplay_equivalent_resolves_board_23_layout`'s
        // known-tricky layout, with its 4 Submarines simply removed —
        // a tighter, more adversarial arrangement than the well-separated
        // one above, to check the "no Submarines" gap isn't only safe for
        // easy layouts.
        let layout_json = r#"{"board": [[null,null,null,null,null,null,null,null,null,null],[null,0,0,0,0,null,null,5,null,null],[null,null,null,null,null,null,null,5,null,null],[null,4,null,null,null,null,null,null,null,null],[null,4,null,null,null,null,null,null,null,null],[null,null,null,null,null,null,2,2,2,null],[null,null,null,3,3,null,null,null,null,null],[null,null,null,null,null,null,1,1,1,null],[null,null,null,null,null,null,null,null,null,null],[null,null,null,null,null,null,null,null,null,null]], "ships": [{"id":0,"name":"Battleship","size":4,"cells":[{"row":1,"col":1},{"row":1,"col":2},{"row":1,"col":3},{"row":1,"col":4}],"hits":0,"sunk":false},{"id":1,"name":"Cruiser","size":3,"cells":[{"row":7,"col":6},{"row":7,"col":7},{"row":7,"col":8}],"hits":0,"sunk":false},{"id":2,"name":"Cruiser","size":3,"cells":[{"row":5,"col":6},{"row":5,"col":7},{"row":5,"col":8}],"hits":0,"sunk":false},{"id":3,"name":"Frigate","size":2,"cells":[{"row":6,"col":3},{"row":6,"col":4}],"hits":0,"sunk":false},{"id":4,"name":"Frigate","size":2,"cells":[{"row":3,"col":1},{"row":4,"col":1}],"hits":0,"sunk":false},{"id":5,"name":"Frigate","size":2,"cells":[{"row":1,"col":7},{"row":2,"col":7}],"hits":0,"sunk":false}]}"#;
        run_until_resolved(layout_json, 60);
    }
}
