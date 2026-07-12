//! Measures whether the true Cruiser/Frigate layout is uniquely recoverable
//! from the raw salvo history alone, once every ship of that size is
//! confirmed sunk (and at every turn afterward, to see how — or whether —
//! the ambiguity resolves further as the rest of the game plays out).
//!
//! This is deliberately independent of `AiPlayer`'s own combination-search
//! machinery (which this session's earlier work found to be unsound in
//! places): it re-derives everything from first principles — the raw
//! (3 fired cells, unordered result multiset) history — using ONLY facts
//! that are always true by construction: the fleet has exactly 2 Cruisers
//! (straight runs of 3 cells) and 3 Frigates (straight runs of 2 cells),
//! and no two ships (of size >=2) may be orthogonally or diagonally
//! adjacent to each other, including to each other's own kind.
//!
//! A candidate SET of windows (e.g. 2 Cruiser windows, or 3 Frigate ones)
//! is consistent with a salvo iff the number of that salvo's 3 fired cells
//! falling inside the union of the set exactly equals the number of
//! matching-size results in that salvo's result multiset — necessary
//! because a "3" (or "2") can only ever come from a Cruiser (or Frigate)
//! cell, and sufficient because it pins down, for every fired cell,
//! whether it does or doesn't belong to this hypothesis without needing to
//! know anything else about what's really at the other 2 cells of that
//! same salvo. Checked against every historical salvo at once, not just
//! the ones any single cell happens to appear in — so, unlike the old
//! combination search, this can never be defeated by 2 real hits of the
//! same ship type landing in the same salvo, or by a neighbouring cell
//! simply never having been fired.
//!
//! Run with: `cargo run --release --bin measure_identifiability -- [games]`
//! (defaults to 200 games). Writes `identifiability.csv` (one row per
//! ship-type per turn, from the turn the first ship of that type sinks
//! onward) and prints a summary to stdout for both ship types.

use std::collections::HashSet;
use std::env;
use std::fs::File;
use std::io::Write;
use std::time::Instant;

use zeeslag::Game;

const INNER_LO: usize = 1;
const INNER_HI: usize = 8;

type Coord = (usize, usize);
type CruiserWindow = [Coord; 3];
type FrigateWindow = [Coord; 2];

struct SalvoRecord {
    coords: [Coord; 3],
    values: [usize; 3],
}

/// Mirrors `try_place`'s own adjacency check in lib.rs: overlap or any
/// orthogonal/diagonal neighbour (dr<=1 && dc<=1) between any cell of `a`
/// and any cell of `b`. Works for any 2 windows regardless of their own
/// length (a Cruiser window and a Frigate window are just as mutually
/// exclusive as 2 windows of the same size).
fn overlaps_or_adjacent(a: &[Coord], b: &[Coord]) -> bool {
    a.iter().any(|&(ar, ac)| {
        b.iter().any(|&(br, bc)| {
            let dr = (ar as isize - br as isize).unsigned_abs();
            let dc = (ac as isize - bc as isize).unsigned_abs();
            dr <= 1 && dc <= 1
        })
    })
}

/// Is the hypothesis "the cells in `window_union` are exactly every real
/// cell of size `ship_value`, and nothing else is" consistent with every
/// salvo fired so far?
fn consistent_with_history(window_union: &HashSet<Coord>, history: &[SalvoRecord], ship_value: usize) -> bool {
    history.iter().all(|salvo| {
        let hits_in_hypothesis = salvo.coords.iter().filter(|c| window_union.contains(c)).count();
        let matches_in_bag = salvo.values.iter().filter(|&&v| v == ship_value).count();
        hits_in_hypothesis == matches_in_bag
    })
}

/// A cell is provably NOT a cell of size `ship_value` if it was ever fired
/// as part of a salvo whose result bag contains no `ship_value` at all —
/// the bag would have to contain at least one if this cell really were
/// that size. Cheap, always-sound necessary condition, used purely to
/// shrink the candidate window pool before the expensive combinatorial
/// search below (an imperfect/too-permissive filter would just mean more
/// work, never a wrong answer — the full consistency check is what
/// actually decides correctness).
fn cells_possibly_size(history: &[SalvoRecord], ship_value: usize) -> [[bool; 10]; 10] {
    let mut possible = [[true; 10]; 10];
    for salvo in history {
        if !salvo.values.contains(&ship_value) {
            for &(r, c) in &salvo.coords {
                possible[r][c] = false;
            }
        }
    }
    possible
}

// --- Cruisers: 2 ships, straight-3 windows, enumerate every pair -----------

/// Every straight-3 window in the inner 8x8 grid: 48 horizontal + 48
/// vertical = 96 total. Small enough that brute-forcing every pair (4560
/// of them) against the full salvo history, every turn, is still far
/// faster than the per-turn self-play itself — no pre-filtering needed.
fn all_cruiser_windows() -> Vec<CruiserWindow> {
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

/// All window-pairs consistent with the full history so far. When
/// `require_fully_fired` is set (only sound once BOTH Cruisers are
/// confirmed sunk — sunk means every real cell was hit), additionally
/// drops any pair with a never-fired cell: a genuinely unfired cell can't
/// be part of an already-fully-sunk ship.
fn enumerate_consistent_cruiser_pairs(
    history: &[SalvoRecord],
    windows: &[CruiserWindow],
    fired: &[[bool; 10]; 10],
    require_fully_fired: bool,
) -> Vec<[CruiserWindow; 2]> {
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
            let union: HashSet<Coord> = windows[i].iter().chain(windows[j].iter()).copied().collect();
            if consistent_with_history(&union, history, 3) {
                results.push([windows[i], windows[j]]);
            }
        }
    }
    results
}

// --- Frigates: 3 ships, straight-2 windows, enumerate every triple ---------

/// Every straight-2 window in the inner 8x8 grid: 56 horizontal + 56
/// vertical = 112 total. Enumerating every TRIPLE of these unfiltered
/// would be ~227,920 combinations, checked against the full history, every
/// turn, every game — too slow. `cells_possibly_size` (size 2) filters
/// this down first, usually drastically, since most cells get eliminated
/// as candidates well before any Frigate sinks.
fn all_frigate_windows() -> Vec<FrigateWindow> {
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

fn enumerate_consistent_frigate_triples(
    history: &[SalvoRecord],
    windows: &[FrigateWindow],
    fired: &[[bool; 10]; 10],
    require_fully_fired: bool,
) -> Vec<[FrigateWindow; 3]> {
    let possible = cells_possibly_size(history, 2);
    let candidates: Vec<FrigateWindow> = windows.iter().filter(|w| w.iter().all(|&(r, c)| possible[r][c])).copied().collect();

    let n = candidates.len();
    let mut results = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            if overlaps_or_adjacent(&candidates[i], &candidates[j]) {
                continue;
            }
            for k in (j + 1)..n {
                if overlaps_or_adjacent(&candidates[i], &candidates[k]) || overlaps_or_adjacent(&candidates[j], &candidates[k]) {
                    continue;
                }
                if require_fully_fired {
                    let all_fired = candidates[i].iter().chain(candidates[j].iter()).chain(candidates[k].iter()).all(|&(r, c)| fired[r][c]);
                    if !all_fired {
                        continue;
                    }
                }
                let union: HashSet<Coord> = candidates[i].iter().chain(candidates[j].iter()).chain(candidates[k].iter()).copied().collect();
                if consistent_with_history(&union, history, 2) {
                    results.push([candidates[i], candidates[j], candidates[k]]);
                }
            }
        }
    }
    results
}

// --- Ground truth (for self-validation only) -------------------------------

fn coord_label_to_rc(label: &str) -> Coord {
    let col = (label.as_bytes()[0] - b'A') as usize;
    let row: usize = label[1..].parse::<usize>().expect("numeric row suffix") - 1;
    (row, col)
}

fn ground_truth_windows<const N: usize>(debug_json: &str, name: &str) -> Vec<[Coord; N]> {
    let parsed: serde_json::Value = serde_json::from_str(debug_json).expect("valid debug_ships_json");
    let mut ships: Vec<[Coord; N]> = parsed
        .as_array()
        .unwrap()
        .iter()
        .filter(|ship| ship["name"] == name)
        .map(|ship| {
            let mut cells: Vec<Coord> = ship["cells"].as_array().unwrap().iter().map(|c| coord_label_to_rc(c.as_str().unwrap())).collect();
            cells.sort();
            let mut window = [(0usize, 0usize); N];
            window.copy_from_slice(&cells);
            window
        })
        .collect();
    ships.sort();
    ships
}

fn cruiser_pair_matches(candidate: &[CruiserWindow; 2], truth: &[CruiserWindow]) -> bool {
    let mut c = *candidate;
    c.sort();
    c.as_slice() == truth
}

fn frigate_triple_matches(candidate: &[FrigateWindow; 3], truth: &[FrigateWindow]) -> bool {
    let mut c = *candidate;
    c.sort();
    c.as_slice() == truth
}

// --- Self-play + measurement loop ------------------------------------------

#[derive(Default)]
struct ShipTypeStats {
    checkpoints_seen: usize,
    unique_count: usize,
    ambiguous_count: usize,
    mismatches: usize,
}

fn main() {
    let games: usize = env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(200);
    let cruiser_windows = all_cruiser_windows();
    let frigate_windows = all_frigate_windows();

    let out_path = "identifiability.csv";
    let mut out = File::create(out_path).expect("create output CSV");
    writeln!(out, "game,turn,ship_type,ships_sunk,num_consistent_placements,ground_truth_included").unwrap();

    let mut cruiser_stats = ShipTypeStats::default();
    let mut frigate_stats = ShipTypeStats::default();

    let start = Instant::now();

    for game_id in 0..games {
        let mut game = Game::new();
        let mut history: Vec<SalvoRecord> = Vec::new();
        let mut fired = [[false; 10]; 10];
        let mut cruisers_sunk = 0usize;
        let mut frigates_sunk = 0usize;
        let mut turn = 0usize;
        // Only the FIRST turn each count reaches its ship total counts
        // toward the summary — it stays at the max for every subsequent
        // turn too (hunting whatever's left), which the CSV still logs in
        // full for the fuller trend, but must not double-count headline stats.
        let mut cruiser_checkpoint_recorded = false;
        let mut frigate_checkpoint_recorded = false;

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

            let sunk_names: Vec<&str> = salvo["sunk_names"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
            cruisers_sunk += sunk_names.iter().filter(|&&n| n == "Cruiser").count();
            frigates_sunk += sunk_names.iter().filter(|&&n| n == "Frigate").count();

            if cruisers_sunk >= 1 {
                let pairs = enumerate_consistent_cruiser_pairs(&history, &cruiser_windows, &fired, cruisers_sunk == 2);
                let truth = ground_truth_windows::<3>(&game.debug_ships_json(), "Cruiser");
                let included = pairs.iter().any(|p| cruiser_pair_matches(p, &truth));
                writeln!(out, "{game_id},{turn},cruiser,{cruisers_sunk},{},{}", pairs.len(), included).unwrap();

                if cruisers_sunk == 2 && !cruiser_checkpoint_recorded {
                    cruiser_checkpoint_recorded = true;
                    record_checkpoint(&mut cruiser_stats, pairs.len(), included, game_id, turn, "Cruiser");
                }
            }

            if frigates_sunk >= 1 {
                let triples = enumerate_consistent_frigate_triples(&history, &frigate_windows, &fired, frigates_sunk == 3);
                let truth = ground_truth_windows::<2>(&game.debug_ships_json(), "Frigate");
                let included = triples.iter().any(|t| frigate_triple_matches(t, &truth));
                writeln!(out, "{game_id},{turn},frigate,{frigates_sunk},{},{}", triples.len(), included).unwrap();

                if frigates_sunk == 3 && !frigate_checkpoint_recorded {
                    frigate_checkpoint_recorded = true;
                    record_checkpoint(&mut frigate_stats, triples.len(), included, game_id, turn, "Frigate");
                }
            }
        }
    }

    let elapsed = start.elapsed();
    println!("Wrote {out_path} ({games} games in {:.1}s)", elapsed.as_secs_f64());
    print_summary("Cruiser", "both Cruisers sink", &cruiser_stats);
    print_summary("Frigate", "all 3 Frigates sink", &frigate_stats);
}

fn record_checkpoint(stats: &mut ShipTypeStats, num_consistent: usize, ground_truth_included: bool, game_id: usize, turn: usize, label: &str) {
    stats.checkpoints_seen += 1;
    if !ground_truth_included {
        stats.mismatches += 1;
        eprintln!("game {game_id} turn {turn}: {label} ground truth NOT among {num_consistent} consistent candidates — bug in this file's own logic");
    } else if num_consistent == 1 {
        stats.unique_count += 1;
    } else {
        stats.ambiguous_count += 1;
    }
}

fn print_summary(label: &str, moment: &str, stats: &ShipTypeStats) {
    println!();
    println!("=== {label} summary (at the turn {moment}) ===");
    println!("games measured:        {}", stats.checkpoints_seen);
    println!(
        "uniquely identifiable: {} ({:.1}%)",
        stats.unique_count,
        100.0 * stats.unique_count as f64 / stats.checkpoints_seen.max(1) as f64
    );
    println!(
        "still ambiguous:       {} ({:.1}%)",
        stats.ambiguous_count,
        100.0 * stats.ambiguous_count as f64 / stats.checkpoints_seen.max(1) as f64
    );
    if stats.mismatches > 0 {
        println!("WARNING: {} checkpoint(s) where ground truth wasn't among the consistent candidates — see stderr, this indicates a bug in this measurement tool, not genuine ambiguity.", stats.mismatches);
    }
}
