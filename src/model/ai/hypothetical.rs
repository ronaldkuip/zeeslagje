//! A-flavored pure FSM-table lookups (duplicate `eliminate_size_at`/
//! `line_state_score`'s table-switch logic), but used only by
//! `heatmap_ops::choose_shots`/`best_cell_for_size_blended` on a LOCAL
//! SCRATCH COPY of FSM state - they never touch real `self.row_state`/
//! `col_state`. Kept separate from `fsm.rs` because it's C's private
//! algorithm detail (the "assume this shot is a miss, what would the
//! FSM look like" simulation greedy shot-picking runs), not part of A's
//! real deduction engine. Extracted from the old ai.rs verbatim (Stage 5
//! of the refactor plan).

use super::*;

impl AiPlayer {

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
    pub(crate) fn size_cell_score(row_line: &[usize; 10], col_line: &[usize; 10], row: usize, col: usize, size: usize) -> u32 {
        if !(INNER_LO..=INNER_HI).contains(&row) || !(INNER_LO..=INNER_HI).contains(&col) {
            return 0;
        }
        let table_col = col - INNER_LO;
        let table_row = row - INNER_LO;
        Self::line_state_score(row_line[row], size, table_col) + Self::line_state_score(col_line[col], size, table_row)
    }

    /// Fold a *hypothetical* miss at (row, col) into a working copy of `size`'s
    /// row/col FSM states — i.e. "if this shot comes back as a miss, what would
    /// that size's FSM look like afterwards". Mirrors `eliminate_size_at`, but
    /// operates on local scratch state rather than `self`.
    pub(crate) fn apply_hypothetical_miss(row_line: &mut [usize; 10], col_line: &mut [usize; 10], row: usize, col: usize, size: usize) {
        if !(INNER_LO..=INNER_HI).contains(&row) || !(INNER_LO..=INNER_HI).contains(&col) {
            return;
        }
        let table_col = col - INNER_LO;
        let table_row = row - INNER_LO;
        row_line[row] = Self::line_state_transition(row_line[row], size, table_col);
        col_line[col] = Self::line_state_transition(col_line[col], size, table_row);
    }
}
