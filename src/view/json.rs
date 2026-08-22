//! View: presentation-formatting DTOs and the derivation logic that
//! builds them from Model data — pure data-transform-for-display, not
//! gameplay decision logic (none of it feeds back into any AI
//! decision), so it lives here rather than in `model`, but it's more
//! than a "call one Model fn and serialize" wrapper, so it doesn't
//! belong directly inline in `controller::game` either. Extracted out
//! of controller::game's `*_debug_json` methods verbatim (Stage 7 of
//! the refactor plan) — same computation, just returning the DTO
//! directly instead of an already-serialized JSON string, so
//! `controller::game` only has to call `serde_json::to_string` on the
//! result.

use serde::Serialize;

use crate::model::ai::{Cross2Entry, Cross3Entry, Cross4Entry};
use crate::model::fleet::GameState;

#[derive(Serialize)]
pub struct Cross3EntryDebug {
    coords: Vec<usize>,
    values: [usize; 3],
    true_cruiser_coords: Vec<usize>,
    ruled_out_coords: Vec<usize>,
    confirmed_coords: Vec<usize>,
}

#[derive(Serialize)]
pub struct Cross3Debug {
    entries: Vec<Cross3EntryDebug>,
}

/// See `controller::game::Game::cross3_debug_json`'s doc comment for
/// the field-by-field explanation — unchanged here, just relocated.
pub fn cross3_debug(state: &GameState, entries: &[Cross3Entry]) -> Cross3Debug {
    let is_cruiser_cell = |r: usize, c: usize| matches!(state.board[r][c], Some(id) if state.ships[id].size == 3);

    let entries: Vec<Cross3EntryDebug> = entries
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

    Cross3Debug { entries }
}

#[derive(Serialize)]
pub struct Cross2EntryDebug {
    coords: Vec<usize>,
    values: [usize; 3],
    true_frigate_coords: Vec<usize>,
    ruled_out_coords: Vec<usize>,
    confirmed_coords: Vec<usize>,
}

#[derive(Serialize)]
pub struct Cross2Debug {
    entries: Vec<Cross2EntryDebug>,
}

/// Same idea as `cross3_debug`, one size down.
pub fn cross2_debug(state: &GameState, entries: &[Cross2Entry]) -> Cross2Debug {
    let is_frigate_cell = |r: usize, c: usize| matches!(state.board[r][c], Some(id) if state.ships[id].size == 2);

    let entries: Vec<Cross2EntryDebug> = entries
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

    Cross2Debug { entries }
}

#[derive(Serialize)]
pub struct Cross4EntryDebug {
    coords: Vec<usize>,
    values: [usize; 3],
    ruled_out_coords: Vec<usize>,
    confirmed_coords: Vec<usize>,
}

#[derive(Serialize)]
pub struct Cross4Debug {
    entries: Vec<Cross4EntryDebug>,
}

/// Same idea as `cross3_debug`, one size up. Doesn't need `state` at
/// all (no ground-truth cross-reference the way Cross3/Cross2 have) —
/// kept parameter-free to match, rather than taking an unused `&GameState`.
pub fn cross4_debug(entries: &[Cross4Entry]) -> Cross4Debug {
    let entries: Vec<Cross4EntryDebug> = entries
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

    Cross4Debug { entries }
}
