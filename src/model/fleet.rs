//! "Generate fleet": board/ship data types plus placement/generation/
//! validation — the Model-layer code with zero AI/deduction dependency.
//! Moved verbatim out of the old `lib.rs` (see the refactor plan) as
//! Stage 1 of the module split; no behavior changes.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Random number generation (wraps getrandom which supports WASM via js feature)
// ---------------------------------------------------------------------------

pub(crate) fn random_usize(n: usize) -> usize {
    let mut buf = [0u8; 4];
    getrandom::getrandom(&mut buf).expect("getrandom failed");
    let v = u32::from_le_bytes(buf) as usize;
    v % n
}

pub(crate) fn random_bool() -> bool {
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
    pub(crate) fn new(id: usize, name: &str, size: usize, cells: Vec<Cell>) -> Self {
        Ship { id, name: name.to_string(), size, cells, hits: 0, sunk: false }
    }

    pub(crate) fn register_hit(&mut self) {
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
    ///
    /// `board`/`ships`/`fired` are `pub(crate)` (not bare-private, as they
    /// were when `GameState` and `Game` shared one module) purely so
    /// `controller::game::Game`'s impl block — a different module now —
    /// can keep reading/writing them exactly as before; a visibility
    /// widening, not a behavior change. See the refactor plan's Stage 1.
    pub(crate) board: Vec<Vec<Option<usize>>>,
    pub(crate) ships: Vec<Ship>,
    pub(crate) fired: Vec<Vec<bool>>,
    pub log: Vec<SalvoResult>,
    pub turn: usize,
    pub won: bool,
    pub total_hits: usize,
    pub hit_count: usize,
}

impl GameState {
    /// Rebuild a fresh (never-fired, turn-1) `GameState` from a saved
    /// `BoardLayout` — the placement-only save/replay format `board_
    /// layout_json` produces. Any hits/sunk flags already on the layout's
    /// ships are ignored; the result always starts completely fresh, same
    /// as `generate_board`'s output. Moved out of `controller::game::
    /// Game::load_board_layout_json` verbatim (Stage 7 of the refactor
    /// plan) — this was real Model-layer reconstruction logic, not JSON
    /// parsing/formatting, even though its only caller is that one
    /// JSON-facing method.
    pub(crate) fn from_board_layout(layout: BoardLayout) -> GameState {
        let total_hits: usize = layout.ships.iter().map(|s| s.size).sum();
        let mut ships = layout.ships;
        for ship in &mut ships {
            ship.hits = 0;
            ship.sunk = false;
        }
        GameState {
            board: layout.board,
            ships,
            fired: vec![vec![false; 10]; 10],
            log: Vec::new(),
            turn: 1,
            won: false,
            total_hits,
            hit_count: 0,
        }
    }
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
pub(crate) fn try_place(
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

pub(crate) fn generate_board() -> GameState {
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

pub(crate) fn col_letter(c: usize) -> char {
    (b'A' + c as u8) as char
}

pub(crate) fn cell_to_str(cell: &Cell) -> String {
    format!("{}{}", col_letter(cell.col), cell.row + 1)
}
