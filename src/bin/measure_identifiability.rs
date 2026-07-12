//! Measures whether the true Cruiser layout is uniquely recoverable from
//! the raw salvo history alone, once both Cruisers are confirmed sunk (and
//! at every turn afterward, to see how — or whether — the ambiguity
//! resolves further as the rest of the game plays out).
//!
//! This is deliberately independent of `AiPlayer`'s own combination-search
//! machinery (which this session's earlier work found to be unsound in
//! places): it re-derives everything from first principles — the raw
//! (3 fired cells, unordered result multiset) history — using ONLY two
//! facts that are always true by construction: the fleet has exactly 2
//! Cruisers, each a straight run of 3 cells, and no two ships (of size >=2)
//! may be orthogonally or diagonally adjacent to each other.
//!
//! A candidate pair of windows (windowA, windowB) is consistent with a
//! salvo iff the number of that salvo's 3 fired cells falling inside
//! windowA ∪ windowB exactly equals the number of "3"s in that salvo's
//! result multiset — necessary because a "3" can only ever come from a
//! Cruiser cell, and sufficient because it pins down, for every fired
//! cell, whether it does or doesn't belong to this hypothesis without
//! needing to know anything else about what's really at the other 2 cells
//! of that same salvo. Checked against every historical salvo at once, not
//! just the ones any single cell happens to appear in — so, unlike the old
//! combination search, this can never be defeated by a real ship's cells
//! landing in the same salvo, or by a neighbouring cell simply never having
//! been fired.
//!
//! Run with: `cargo run --release --bin measure_identifiability -- [games]`
//! (defaults to 200 games). Writes `cruiser_identifiability.csv` (one row
//! per turn, from the turn the first Cruiser sinks onward) and prints a
//! summary to stdout.

use std::collections::HashSet;
use std::env;
use std::fs::File;
use std::io::Write;

use zeeslag::Game;

const INNER_LO: usize = 1;
const INNER_HI: usize = 8;

type Coord = (usize, usize);
type Window = [Coord; 3];

struct SalvoRecord {
    coords: [Coord; 3],
    values: [usize; 3],
}

/// Every straight-3 window in the inner 8x8 grid: 48 horizontal + 48
/// vertical = 96 total. Small enough that brute-forcing every pair (4560
/// of them) against the full salvo history, every turn, is still far
/// faster than the per-turn self-play itself.
fn all_cruiser_windows() -> Vec<Window> {
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

/// Mirrors `try_place`'s own adjacency check in lib.rs: overlap or any
/// orthogonal/diagonal neighbour (dr<=1 && dc<=1) between any cell of `a`
/// and any cell of `b`.
fn overlaps_or_adjacent(a: &Window, b: &Window) -> bool {
    a.iter().any(|&(ar, ac)| {
        b.iter().any(|&(br, bc)| {
            let dr = (ar as isize - br as isize).unsigned_abs();
            let dc = (ac as isize - bc as isize).unsigned_abs();
            dr <= 1 && dc <= 1
        })
    })
}

/// Is the hypothesis "the only 2 real Cruisers are exactly windowA and
/// windowB" consistent with every salvo fired so far?
fn consistent_with_history(window_a: &Window, window_b: &Window, history: &[SalvoRecord]) -> bool {
    let cruiser_cells: HashSet<Coord> = window_a.iter().chain(window_b.iter()).copied().collect();
    history.iter().all(|salvo| {
        let hits_in_hypothesis = salvo.coords.iter().filter(|c| cruiser_cells.contains(c)).count();
        let threes_in_bag = salvo.values.iter().filter(|&&v| v == 3).count();
        hits_in_hypothesis == threes_in_bag
    })
}

/// All window-pairs consistent with the full history so far. When
/// `require_fully_fired` is set (only sound once BOTH Cruisers are
/// confirmed sunk — sunk means every real cell was hit), additionally
/// drops any pair with a never-fired cell: a genuinely unfired cell can't
/// be part of an already-fully-sunk ship.
fn enumerate_consistent_pairs(
    history: &[SalvoRecord],
    windows: &[Window],
    fired: &[[bool; 10]; 10],
    require_fully_fired: bool,
) -> Vec<(Window, Window)> {
    let mut results = Vec::new();
    for i in 0..windows.len() {
        for j in (i + 1)..windows.len() {
            if overlaps_or_adjacent(&windows[i], &windows[j]) {
                continue;
            }
            if require_fully_fired {
                let all_fired = windows[i].iter().chain(windows[j].iter()).all(|&(r, c)| fired[r][c]);
                if !all_fired {
                    continue;
                }
            }
            if consistent_with_history(&windows[i], &windows[j], history) {
                results.push((windows[i], windows[j]));
            }
        }
    }
    results
}

fn coord_label_to_rc(label: &str) -> Coord {
    let col = (label.as_bytes()[0] - b'A') as usize;
    let row: usize = label[1..].parse::<usize>().expect("numeric row suffix") - 1;
    (row, col)
}

fn sorted_window(mut coords: [Coord; 3]) -> Window {
    coords.sort();
    coords
}

/// Ground truth, straight from the board — for sanity-checking our OWN
/// enumeration (the true pair must always appear among the consistent
/// candidates; if it ever doesn't, that's a bug in this file, not genuine
/// ambiguity).
fn ground_truth_cruisers(debug_json: &str) -> [Window; 2] {
    let parsed: serde_json::Value = serde_json::from_str(debug_json).expect("valid debug_ships_json");
    let mut cruisers: Vec<Window> = parsed
        .as_array()
        .unwrap()
        .iter()
        .filter(|ship| ship["name"] == "Cruiser")
        .map(|ship| {
            let cells: Vec<Coord> = ship["cells"].as_array().unwrap().iter().map(|c| coord_label_to_rc(c.as_str().unwrap())).collect();
            sorted_window([cells[0], cells[1], cells[2]])
        })
        .collect();
    cruisers.sort();
    [cruisers[0], cruisers[1]]
}

fn pair_matches(candidate: (Window, Window), truth: [Window; 2]) -> bool {
    (candidate.0 == truth[0] && candidate.1 == truth[1]) || (candidate.0 == truth[1] && candidate.1 == truth[0])
}

fn main() {
    let games: usize = env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(200);
    let windows = all_cruiser_windows();

    let out_path = "cruiser_identifiability.csv";
    let mut out = File::create(out_path).expect("create output CSV");
    writeln!(out, "game,turn,cruisers_sunk,num_consistent_pairs,ground_truth_included").unwrap();

    let mut unique_count = 0usize;
    let mut ambiguous_count = 0usize;
    let mut checkpoints_seen = 0usize;
    let mut mismatches = 0usize;

    for game_id in 0..games {
        let mut game = Game::new();
        let mut history: Vec<SalvoRecord> = Vec::new();
        let mut fired = [[false; 10]; 10];
        let mut cruisers_sunk = 0usize;
        let mut turn = 0usize;
        // Only the FIRST turn cruisers_sunk reaches 2 counts toward the
        // summary ("at the moment both Cruisers sink") — cruisers_sunk
        // stays at 2 for every subsequent turn too (hunting whatever's
        // left), which the CSV still logs in full for the fuller trend,
        // but must not double-count in the headline stats.
        let mut both_sunk_checkpoint_recorded = false;

        while !game.is_won() {
            turn += 1;
            let indices: Vec<usize> = serde_json::from_str(&game.ai_suggest()).expect("valid ai_suggest JSON");
            let coords: [Coord; 3] = [(indices[0] / 10, indices[0] % 10), (indices[1] / 10, indices[1] % 10), (indices[2] / 10, indices[2] % 10)];

            let raw = game.fire(&indices);
            let salvo: serde_json::Value = serde_json::from_str(&raw).expect("valid fire() JSON");
            if salvo.get("error").is_some() {
                eprintln!("game {game_id} turn {turn}: unexpected fire() error: {salvo}");
                break;
            }

            let values: Vec<usize> = salvo["result"].as_str().unwrap().split_whitespace().map(|s| s.parse().unwrap()).collect();
            history.push(SalvoRecord { coords, values: [values[0], values[1], values[2]] });
            for &(r, c) in &coords {
                fired[r][c] = true;
            }

            cruisers_sunk += salvo["sunk_names"].as_array().unwrap().iter().filter(|n| n.as_str() == Some("Cruiser")).count();

            if cruisers_sunk >= 1 {
                let pairs = enumerate_consistent_pairs(&history, &windows, &fired, cruisers_sunk == 2);
                let truth = ground_truth_cruisers(&game.debug_ships_json());
                let included = pairs.iter().any(|&p| pair_matches(p, truth));

                writeln!(out, "{game_id},{turn},{cruisers_sunk},{},{}", pairs.len(), included).unwrap();

                if cruisers_sunk == 2 && !both_sunk_checkpoint_recorded {
                    both_sunk_checkpoint_recorded = true;
                    checkpoints_seen += 1;
                    if !included {
                        mismatches += 1;
                        eprintln!("game {game_id} turn {turn}: ground truth NOT among {} consistent pairs — bug in this file's own logic", pairs.len());
                    } else if pairs.len() == 1 {
                        unique_count += 1;
                    } else {
                        ambiguous_count += 1;
                    }
                }
            }
        }
    }

    println!("Wrote {out_path}");
    println!();
    println!("=== Summary (at the turn each game's 2nd Cruiser sinks) ===");
    println!("games measured:        {checkpoints_seen}");
    println!("uniquely identifiable: {unique_count} ({:.1}%)", 100.0 * unique_count as f64 / checkpoints_seen.max(1) as f64);
    println!("still ambiguous:       {ambiguous_count} ({:.1}%)", 100.0 * ambiguous_count as f64 / checkpoints_seen.max(1) as f64);
    if mismatches > 0 {
        println!("WARNING: {mismatches} checkpoint(s) where ground truth wasn't among the consistent candidates — see stderr, this indicates a bug in this measurement tool, not genuine ambiguity.");
    }
}
