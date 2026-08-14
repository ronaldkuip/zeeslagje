//! Batch-solves boards read from a large sequential file, one board per
//! line. Each line must be exactly 64 characters over {'0','2','3','4'} —
//! the inner 8x8 grid (row-major), water=0, Frigate=2, Cruiser=3,
//! Battleship=4. Submarines are deliberately not part of this format (see
//! the Player Page's own-fleet placement, which is the same 16-hit — not
//! 20-hit — fleet: 1 Battleship + 2 Cruisers + 3 Frigates).
//!
//! For each line:
//!   - If the board doesn't satisfy the placement rules (wrong cell counts,
//!     a non-straight or wrong-length run, or 2 ships closer than a full
//!     cell apart in any direction including diagonally — the same
//!     invariant `try_place` enforces during normal board generation), the
//!     original line is appended to `game_board_in_error.log` and skipped.
//!   - Otherwise the board is loaded into a fresh `Game` (via the same
//!     `load_board_layout_json` path the Player Page uses) and played out
//!     by the built-in AI exactly as "Autorun Batch" does in index.html:
//!     repeatedly pulling the disambiguation-aware advisory and firing it,
//!     capped at 30 total salvos, "solved" once the Battleship/Cruisers/
//!     Frigates are both sunk AND their exact layout is disambiguated
//!     (`resolution_status_json().resolved`), else bucketed as unresolved.
//!
//! Prints a histogram of salvo counts across every solved board, plus the
//! unresolved and invalid counts.
//!
//! Optionally (5th arg), boards resolved in one of `SPECIAL_SALVO_COUNTS`
//! salvos — the interesting tails of the distribution, unusually fast or
//! unusually slow — are additionally appended (one JSON object per line,
//! same `{"board": <BoardLayout>, "salvos": N}` shape as serve.py's
//! `fast_solves.jsonl`) to a dedicated file per count in that directory, so
//! a re-run over the same data accumulates into the same files rather than
//! clobbering them.
//!
//! Run with: `cargo run --release --bin solve_boards_file -- <input_file> [threads] [error_log_path] [special_cases_dir]`

use std::collections::HashMap;
use std::env;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::sync::mpsc::{channel, sync_channel};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;

use serde::Serialize;
use zeeslag::{BoardLayout, Cell, Game, Ship};

const CAP: usize = 30;

/// The "interesting" salvo counts worth saving a full board layout for:
/// unusually fast solves (9-11) and unusually slow ones (24-26), both
/// tails of the histogram far from the typical 16-22 range. See
/// `solve_boards_file`'s module doc for the output format/location.
const SPECIAL_SALVO_COUNTS: [usize; 6] = [9, 10, 11, 24, 25, 26];

/// Mirrors serve.py's `fast_solves.jsonl` record shape exactly, so tooling
/// that already reads that file can read these too.
#[derive(Serialize)]
struct SpecialCaseRecord<'a> {
    board: &'a BoardLayout,
    salvos: usize,
}

struct Component {
    value: u8,
    cells: Vec<(usize, usize)>,
}

/// Strips an optional enclosing `[...]` (and matching quotes just inside
/// it) some source files wrap each board string in — e.g.
/// `[4030302040...20]` — leaving the bare 64-char board string. A no-op on
/// input that's already bare.
fn extract_board_str(raw: &str) -> &str {
    let s = raw.trim();
    let s = s.strip_prefix('[').and_then(|s| s.strip_suffix(']')).unwrap_or(s);
    let s = s.trim();
    s.strip_prefix('"').and_then(|s| s.strip_suffix('"')).unwrap_or(s)
}

/// Parses a 64-char inner-board string and checks it against the exact
/// placement invariant `try_place` (src/lib.rs) enforces for ships sized
/// >=2: each ship's cells form a straight run of the right length, and no
/// 2 ships (of any size, including the same size) are within Chebyshev
/// distance 1 of each other. Returns the resulting `Ship` list (1-indexed
/// into the 10x10 board space, matching `BoardLayout`) or an error reason.
fn parse_and_validate(line: &str) -> Result<Vec<Ship>, String> {
    let line = extract_board_str(line);
    let bytes = line.as_bytes();
    if bytes.len() != 64 {
        return Err(format!("expected 64 characters, got {}", bytes.len()));
    }

    let mut grid = [[0u8; 8]; 8];
    for (i, &b) in bytes.iter().enumerate() {
        let v = match b {
            b'0' => 0u8,
            b'2' => 2u8,
            b'3' => 3u8,
            b'4' => 4u8,
            other => return Err(format!("invalid character '{}' at position {i}", other as char)),
        };
        grid[i / 8][i % 8] = v;
    }

    // Flood-fill 4-connected same-value components.
    let mut visited = [[false; 8]; 8];
    let mut comps: Vec<Component> = Vec::new();
    for r in 0..8 {
        for c in 0..8 {
            if grid[r][c] == 0 || visited[r][c] {
                continue;
            }
            let value = grid[r][c];
            let mut stack = vec![(r, c)];
            visited[r][c] = true;
            let mut cells = Vec::new();
            while let Some((cr, cc)) = stack.pop() {
                cells.push((cr, cc));
                for (dr, dc) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                    let nr = cr as i32 + dr;
                    let nc = cc as i32 + dc;
                    if (0..8).contains(&nr) && (0..8).contains(&nc) {
                        let (nr, nc) = (nr as usize, nc as usize);
                        if !visited[nr][nc] && grid[nr][nc] == value {
                            visited[nr][nc] = true;
                            stack.push((nr, nc));
                        }
                    }
                }
            }
            comps.push(Component { value, cells });
        }
    }

    let (mut battleships, mut cruisers, mut frigates) = (0, 0, 0);
    for comp in &comps {
        let expected_len = match comp.value {
            4 => 4,
            3 => 3,
            2 => 2,
            _ => unreachable!(),
        };
        if comp.cells.len() != expected_len {
            return Err(format!(
                "value-{} component has {} connected cells, expected {expected_len}",
                comp.value,
                comp.cells.len()
            ));
        }

        let rows: Vec<usize> = comp.cells.iter().map(|&(r, _)| r).collect();
        let cols: Vec<usize> = comp.cells.iter().map(|&(_, c)| c).collect();
        let same_row = rows.iter().all(|&r| r == rows[0]);
        let same_col = cols.iter().all(|&c| c == cols[0]);
        if same_row {
            let mut sorted = cols.clone();
            sorted.sort_unstable();
            if !sorted.windows(2).all(|w| w[1] == w[0] + 1) {
                return Err(format!("value-{} component is not a contiguous straight run", comp.value));
            }
        } else if same_col {
            let mut sorted = rows.clone();
            sorted.sort_unstable();
            if !sorted.windows(2).all(|w| w[1] == w[0] + 1) {
                return Err(format!("value-{} component is not a contiguous straight run", comp.value));
            }
        } else {
            return Err(format!("value-{} component is not a straight line", comp.value));
        }

        match comp.value {
            4 => battleships += 1,
            3 => cruisers += 1,
            2 => frigates += 1,
            _ => unreachable!(),
        }
    }

    if battleships != 1 {
        return Err(format!("expected 1 Battleship, found {battleships}"));
    }
    if cruisers != 2 {
        return Err(format!("expected 2 Cruisers, found {cruisers}"));
    }
    if frigates != 3 {
        return Err(format!("expected 3 Frigates, found {frigates}"));
    }

    // No 2 ships (any size, including same size) may touch, even diagonally.
    for i in 0..comps.len() {
        for j in (i + 1)..comps.len() {
            for &(ar, ac) in &comps[i].cells {
                for &(br, bc) in &comps[j].cells {
                    let dr = (ar as i32 - br as i32).abs();
                    let dc = (ac as i32 - bc as i32).abs();
                    if dr <= 1 && dc <= 1 {
                        return Err(format!(
                            "value-{} and value-{} ships are adjacent (cells ({ar},{ac}) / ({br},{bc}))",
                            comps[i].value, comps[j].value
                        ));
                    }
                }
            }
        }
    }

    let mut ships = Vec::with_capacity(comps.len());
    for (id, comp) in comps.into_iter().enumerate() {
        let name = match comp.value {
            4 => "Battleship",
            3 => "Cruiser",
            2 => "Frigate",
            _ => unreachable!(),
        };
        let cells = comp.cells.iter().map(|&(r, c)| Cell { row: r + 1, col: c + 1 }).collect();
        ships.push(Ship { id, name: name.to_string(), size: comp.value as usize, cells, hits: 0, sunk: false });
    }
    Ok(ships)
}

fn build_board(ships: &[Ship]) -> Vec<Vec<Option<usize>>> {
    let mut board = vec![vec![None; 10]; 10];
    for ship in ships {
        for cell in &ship.cells {
            board[cell.row][cell.col] = Some(ship.id);
        }
    }
    board
}

fn is_resolved(game: &Game) -> bool {
    let v: serde_json::Value = serde_json::from_str(&game.resolution_status_json()).unwrap_or(serde_json::Value::Null);
    v.get("resolved").and_then(|b| b.as_bool()).unwrap_or(false)
}

/// Mirrors `runAutoPlay`/`refreshAdvisory` in index.html: prioritize a
/// pending Cruiser/Frigate disambiguation (with its capped-refire, then
/// last-resort-refire fallbacks) once nothing bigger than a Submarine is
/// left to hunt, otherwise fall back to the ordinary advisory. Stops at the
/// 30-salvo cap or once resolved; returns the salvo count if resolved.
fn solve_one(game: &mut Game, cap: usize) -> Option<usize> {
    let mut salvos = 0usize;
    let mut cleared = game.is_won() && is_resolved(game);

    while !cleared {
        if salvos >= cap {
            break;
        }

        let mut parsed: Vec<usize> = Vec::new();
        if game.ai_target_size() <= 1 {
            let _locked = game.update_fsm_and_resolve();
            parsed = serde_json::from_str(&game.ai_suggest_disambiguation_refire()).unwrap_or_default();
            if parsed.len() != 3 {
                parsed = serde_json::from_str(&game.ai_suggest_disambiguation_last_resort()).unwrap_or_default();
            }
        }
        if parsed.len() != 3 {
            parsed = serde_json::from_str(&game.ai_suggest()).unwrap_or_default();
        }
        if parsed.len() != 3 {
            break; // no valid salvo available via any strategy
        }

        let raw = game.fire(&parsed);
        if raw.contains("\"error\"") {
            break;
        }
        salvos += 1;
        cleared = game.is_won() && is_resolved(game);
    }

    if cleared { Some(salvos) } else { None }
}

enum WorkResult {
    Invalid(String),
    /// Salvo count, plus a pre-serialized `SpecialCaseRecord` JSON line
    /// when that count is one of `SPECIAL_SALVO_COUNTS` and special-case
    /// tracking is enabled — `None` otherwise, so the common case doesn't
    /// pay for a board serialization it'll never use.
    Resolved(usize, Option<String>),
    Unresolved,
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: {} <input_file> [threads] [error_log_path] [special_cases_dir]", args[0]);
        std::process::exit(1);
    }
    let input_path = &args[1];
    let threads: usize = args
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| thread::available_parallelism().map(|n| n.get()).unwrap_or(4));
    let error_log_path = args.get(3).cloned().unwrap_or_else(|| "game_board_in_error.log".to_string());
    let special_cases_dir = args.get(4).cloned();
    let track_special = special_cases_dir.is_some();

    let file = File::open(input_path).unwrap_or_else(|e| {
        eprintln!("cannot open {input_path}: {e}");
        std::process::exit(1);
    });
    let reader = BufReader::with_capacity(1 << 20, file);

    let (work_tx, work_rx) = sync_channel::<String>(threads * 64);
    let work_rx = Arc::new(Mutex::new(work_rx));
    let (res_tx, res_rx) = channel::<WorkResult>();

    let mut workers = Vec::with_capacity(threads);
    for _ in 0..threads {
        let work_rx = Arc::clone(&work_rx);
        let res_tx = res_tx.clone();
        workers.push(thread::spawn(move || {
            let mut game = Game::new();
            loop {
                let line = {
                    let rx = work_rx.lock().unwrap();
                    rx.recv()
                };
                let line = match line {
                    Ok(l) => l,
                    Err(_) => break, // channel closed, no more work
                };

                match parse_and_validate(&line) {
                    Err(_reason) => {
                        let _ = res_tx.send(WorkResult::Invalid(line));
                    }
                    Ok(ships) => {
                        let layout = BoardLayout { board: build_board(&ships), ships };
                        let json = serde_json::to_string(&layout).expect("serialize layout");
                        let resp = game.load_board_layout_json(&json);
                        if resp.contains("\"error\"") {
                            let _ = res_tx.send(WorkResult::Invalid(line));
                            continue;
                        }
                        let result = match solve_one(&mut game, CAP) {
                            Some(n) => {
                                let special_json = if track_special && SPECIAL_SALVO_COUNTS.contains(&n) {
                                    Some(serde_json::to_string(&SpecialCaseRecord { board: &layout, salvos: n }).expect("serialize special case"))
                                } else {
                                    None
                                };
                                WorkResult::Resolved(n, special_json)
                            }
                            None => WorkResult::Unresolved,
                        };
                        let _ = res_tx.send(result);
                    }
                }
            }
        }));
    }
    drop(res_tx); // only the workers' clones should keep the channel alive

    let error_log = File::create(&error_log_path).unwrap_or_else(|e| {
        eprintln!("cannot create {error_log_path}: {e}");
        std::process::exit(1);
    });
    // Opened in append mode (not truncate-on-create) so re-running this
    // over the same data — or running it once per file in a batch script
    // like deploy/run_all_boards.sh — accumulates into the same 6 files
    // instead of each invocation clobbering the previous one's findings.
    let special_case_files: Option<Vec<File>> = special_cases_dir.as_ref().map(|dir| {
        std::fs::create_dir_all(dir).unwrap_or_else(|e| {
            eprintln!("cannot create special cases dir {dir}: {e}");
            std::process::exit(1);
        });
        SPECIAL_SALVO_COUNTS
            .iter()
            .map(|n| {
                let path = Path::new(dir).join(format!("salvos_{n:02}.jsonl"));
                OpenOptions::new().create(true).append(true).open(&path).unwrap_or_else(|e| {
                    eprintln!("cannot open {}: {e}", path.display());
                    std::process::exit(1);
                })
            })
            .collect()
    });

    let aggregator = thread::spawn(move || {
        let mut error_writer = BufWriter::new(error_log);
        let mut special_writers: Option<Vec<BufWriter<File>>> = special_case_files.map(|files| files.into_iter().map(BufWriter::new).collect());
        let mut histogram: HashMap<usize, usize> = HashMap::new();
        let mut unresolved = 0usize;
        let mut invalid = 0usize;
        let mut total = 0usize;
        for result in res_rx {
            total += 1;
            match result {
                WorkResult::Invalid(line) => {
                    invalid += 1;
                    writeln!(error_writer, "{line}").expect("write error log");
                }
                WorkResult::Resolved(n, special_json) => {
                    *histogram.entry(n).or_insert(0) += 1;
                    if let (Some(json_line), Some(writers)) = (special_json, special_writers.as_mut()) {
                        let idx = SPECIAL_SALVO_COUNTS.iter().position(|&s| s == n).expect("special_json only set for a SPECIAL_SALVO_COUNTS value");
                        writeln!(writers[idx], "{json_line}").expect("write special case file");
                    }
                }
                WorkResult::Unresolved => unresolved += 1,
            }
            if total % 100_000 == 0 {
                eprintln!("...{total} boards processed");
            }
        }
        error_writer.flush().expect("flush error log");
        if let Some(writers) = special_writers.as_mut() {
            for w in writers.iter_mut() {
                w.flush().expect("flush special case file");
            }
        }
        (total, invalid, unresolved, histogram)
    });

    let start = Instant::now();
    let mut lines_read = 0usize;
    for line in reader.lines() {
        let line = line.expect("read error");
        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }
        work_tx.send(line).expect("send to workers");
        lines_read += 1;
    }
    drop(work_tx);

    for w in workers {
        w.join().expect("worker thread panicked");
    }
    let (total, invalid, unresolved, histogram) = aggregator.join().expect("aggregator thread panicked");

    let elapsed = start.elapsed();
    println!("Processed {total} boards ({lines_read} non-empty lines read, {threads} threads) in {:.1}s", elapsed.as_secs_f64());
    println!("Invalid boards:  {invalid} (written to {error_log_path})");
    println!("Unresolved (hit the {CAP}-salvo cap): {unresolved}");
    if let Some(dir) = &special_cases_dir {
        let special_total: usize = SPECIAL_SALVO_COUNTS.iter().map(|n| *histogram.get(n).unwrap_or(&0)).sum();
        println!("Special cases (salvos in {SPECIAL_SALVO_COUNTS:?}): {special_total} (appended to {dir}/salvos_NN.jsonl)");
    }
    println!();
    println!("salvos,count");
    let mut keys: Vec<&usize> = histogram.keys().collect();
    keys.sort();
    for k in keys {
        println!("{k},{}", histogram[k]);
    }
}
