mod fsm_tables;
mod model;
mod controller;

pub use controller::game::Game;
pub use model::fleet::{BoardLayout, Cell, GameState, ResolutionStatus, SalvoResult, Ship};

#[cfg(test)]
mod tests;
