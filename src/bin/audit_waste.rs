//! Waste auditor: replays the built-in AI on boards from a batch file
//! (same 64-char format as `solve_boards_file`) and classifies every shot
//! against an independent, first-principles model of what was knowable
//! at that moment — so we can rank where salvos are still being lost
//! instead of waiting for someone to spot it on a screenshot.
//!
//! The model (same reasoning as `measure_identifiability`): per ship
//! class, the list of every placement (1 Battleship window, 2 Cruiser
//! windows, 3 Frigate windows, no two windows touching) consistent with
//! every salvo so far — a placement is consistent with a salvo iff the
//! number of its cells among the 3 fired equals the number of that
//! class's value in the result bag. When the three lists are small enough
//! their product is cross-checked too (no two ships of any class may
//! touch). Sunk messages are NOT used, so the model never knows more than
//! the AI does: every shot it calls wasted really was.
//!
//! Each shot is classified BEFORE it is fired:
//!   kill        unfired cell that is a ship cell in every surviving layout
//!   probe       unfired cell whose content is still uncertain
//!   dead        unfired inner cell that is empty in every surviving layout
//!   outer       outer-ring cell (Submarine-only; pure waste in this
//!               Submarine-free batch format, not in a real game)
//!   refire-info refire of a cell whose value is still uncertain
//!   refire-dead refire of a cell whose value is already certain
//!
//! Per game it also records `known_at` — the first moment the full layout
//! is unique — and the lower bound `known_at + ceil(unfired ship cells /
//! 3)`; anything above that is slack spent after the board was known.
//! And it flags bag-blind salvos: 2-3 probe cells of one class whose
//! combined hit count is the same in every surviving layout.
//!
//! Run with: `cargo run --release --bin audit_waste -- <boards_file> [games] [threads]`

use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::sync::Arc;
use std::thread;

use zeeslag::{BoardLayout, Cell, Game, Ship};

const CAP: usize = 30;
const JOINT_LIMIT: usize = 200_000;

type Mask = u128;

fn bit(r: usize, c: usize) -> Mask {
    1u128 << (r * 10 + c)
}

fn halo(m: Mask) -> Mask {
    let mut out = 0;
    for r in 0..10 {
        for c in 0..10 {
            if m & bit(r, c) != 0 {
                for dr in -1i32..=1 {
                    for dc in -1i32..=1 {
                        let (nr, nc) = (r as i32 + dr, c as i32 + dc);
                        if (0..10).contains(&nr) && (0..10).contains(&nc) {
                            out |= bit(nr as usize, nc as usize);
                        }
                    }
                }
            }
        }
    }
    out
}

fn windows(len: usize) -> Vec<Mask> {
    let mut out = Vec::new();
    for r in 1..=8 {
        for c in 1..=(9 - len) {
            out.push((0..len).fold(0, |m, i| m | bit(r, c + i)));
        }
    }
    for c in 1..=8 {
        for r in 1..=(9 - len) {
            out.push((0..len).fold(0, |m, i| m | bit(r + i, c)));
        }
    }
    out
}

/// Every placement of `k` mutually non-touching windows of length `len`, as union masks.
fn placements(len: usize, k: usize) -> Vec<Mask> {
    let w = windows(len);
    let h: Vec<Mask> = w.iter().map(|&m| halo(m)).collect();
    let mut out = Vec::new();
    fn rec(w: &[Mask], h: &[Mask], start: usize, k: usize, acc: Mask, acc_halo: Mask, out: &mut Vec<Mask>) {
        if k == 0 {
            out.push(acc);
            return;
        }
        for i in start..w.len() {
            if w[i] & acc_halo == 0 {
                rec(w, h, i + 1, k - 1, acc | w[i], acc_halo | h[i], out);
            }
        }
    }
    rec(&w, &h, 0, k, 0, 0, &mut out);
    out
}

fn parse_board(line: &str) -> Option<[[u8; 10]; 10]> {
    let s = line.trim().trim_start_matches('[').trim_end_matches(']').trim_matches('"');
    if s.len() != 64 {
        return None;
    }
    let mut g = [[0u8; 10]; 10];
    for (i, b) in s.bytes().enumerate() {
        g[i / 8 + 1][i % 8 + 1] = b.checked_sub(b'0')?;
    }
    Some(g)
}

fn ships_from_grid(g: &[[u8; 10]; 10]) -> Vec<Ship> {
    let mut seen = [[false; 10]; 10];
    let mut ships = Vec::new();
    for r in 0..10 {
        for c in 0..10 {
            let v = g[r][c];
            if v == 0 || seen[r][c] {
                continue;
            }
            let mut cells = vec![(r, c)];
            seen[r][c] = true;
            let horizontal = c + 1 < 10 && g[r][c + 1] == v;
            for i in 1..v as usize {
                let (nr, nc) = if horizontal { (r, c + i) } else { (r + i, c) };
                seen[nr][nc] = true;
                cells.push((nr, nc));
            }
            let name = match v {
                4 => "Battleship",
                3 => "Cruiser",
                _ => "Frigate",
            };
            ships.push(Ship {
                id: ships.len(),
                name: name.to_string(),
                size: v as usize,
                cells: cells.into_iter().map(|(row, col)| Cell { row, col }).collect(),
                hits: 0,
                sunk: false,
            });
        }
    }
    ships
}

fn is_resolved(game: &Game) -> bool {
    let v: serde_json::Value = serde_json::from_str(&game.resolution_status_json()).unwrap_or(serde_json::Value::Null);
    v.get("resolved").and_then(|b| b.as_bool()).unwrap_or(false)
}

/// Mirrors `solve_one` in solve_boards_file.rs: the next salvo the batch
/// runner would fire, or None once resolved / stuck.
fn next_salvo(game: &mut Game) -> Option<Vec<usize>> {
    let mut parsed: Vec<usize> = Vec::new();
    if game.ai_target_size() <= 1 {
        let _ = game.update_fsm_and_resolve();
        if game.is_won() && is_resolved(game) {
            return None;
        }
        parsed = serde_json::from_str(&game.ai_suggest_disambiguation_refire()).unwrap_or_default();
        if parsed.len() != 3 {
            parsed = serde_json::from_str(&game.ai_suggest_disambiguation_last_resort()).unwrap_or_default();
        }
    }
    if parsed.len() != 3 {
        parsed = serde_json::from_str(&game.ai_suggest()).unwrap_or_default();
    }
    if parsed.len() == 3 { Some(parsed) } else { None }
}

const KINDS: [&str; 7] = ["kill", "probe", "dead", "outer", "refire-info", "refire-dead", "dead-joint"];

#[derive(Default, Clone)]
struct Stats {
    games: usize,
    salvos: usize,
    unresolved: usize,
    /// (phase label, kind index) -> shots
    shots: HashMap<(String, usize), usize>,
    /// slack after known -> games
    slack: HashMap<i64, usize>,
    never_known: usize,
    /// salvos fired after the layout was unique
    salvos_after_known: usize,
    /// bag-blind salvos per class value
    bag_blind: HashMap<u8, usize>,
    /// salvos whose bag was fully predictable and that contained no kill
    predictable_no_kill: usize,
    /// worst examples: (slack, board line, salvo trace)
    examples: Vec<(i64, String, String)>,
}

impl Stats {
    fn merge(&mut self, o: Stats) {
        self.games += o.games;
        self.salvos += o.salvos;
        self.unresolved += o.unresolved;
        for (k, v) in o.shots {
            *self.shots.entry(k).or_default() += v;
        }
        for (k, v) in o.slack {
            *self.slack.entry(k).or_default() += v;
        }
        self.never_known += o.never_known;
        self.salvos_after_known += o.salvos_after_known;
        for (k, v) in o.bag_blind {
            *self.bag_blind.entry(k).or_default() += v;
        }
        self.predictable_no_kill += o.predictable_no_kill;
        self.examples.extend(o.examples);
    }
}

struct Knowledge {
    /// per class: OR / AND over surviving layouts
    or: [Mask; 3],
    and: [Mask; 3],
    unique: bool,
    /// surviving joint layouts as [bs, cr, fr], when enumerated
    joint: Option<Vec<[Mask; 3]>>,
}

fn knowledge(lists: &[Vec<Mask>; 3], halos: &[Vec<Mask>; 3]) -> Knowledge {
    let product = lists.iter().map(|l| l.len()).fold(1usize, |a, b| a.saturating_mul(b));
    if product <= JOINT_LIMIT {
        let mut joint = Vec::new();
        for (bi, &b) in lists[0].iter().enumerate() {
            for (ci, &c) in lists[1].iter().enumerate() {
                if c & halos[0][bi] != 0 {
                    continue;
                }
                for (fi, &f) in lists[2].iter().enumerate() {
                    if f & (halos[0][bi] | halos[1][ci]) != 0 {
                        continue;
                    }
                    let _ = fi;
                    joint.push([b, c, f]);
                }
            }
        }
        if !joint.is_empty() {
            let mut or = [0; 3];
            let mut and = [!0; 3];
            for j in &joint {
                for k in 0..3 {
                    or[k] |= j[k];
                    and[k] &= j[k];
                }
            }
            return Knowledge { or, and, unique: joint.len() == 1, joint: Some(joint) };
        }
    }
    let mut or = [0; 3];
    let mut and = [!0; 3];
    for k in 0..3 {
        for &m in &lists[k] {
            or[k] |= m;
            and[k] &= m;
        }
    }
    Knowledge { or, and, unique: lists.iter().all(|l| l.len() == 1), joint: None }
}

fn audit_game(line: &str, base: &[Vec<Mask>; 3], base_halos: &[Vec<Mask>; 3], st: &mut Stats) {
    let Some(g) = parse_board(line) else { return };
    let ships = ships_from_grid(&g);
    let mut board = vec![vec![None; 10]; 10];
    for s in &ships {
        for cell in &s.cells {
            board[cell.row][cell.col] = Some(s.id);
        }
    }
    let layout = BoardLayout { board, ships };
    let mut game = Game::new();
    if game.load_board_layout_json(&serde_json::to_string(&layout).unwrap()).contains("\"error\"") {
        return;
    }
    let truth_ship: Mask = (0..10).flat_map(|r| (0..10).map(move |c| (r, c))).filter(|&(r, c)| g[r][c] > 0).fold(0, |m, (r, c)| m | bit(r, c));

    let mut lists = base.clone();
    let mut halos = base_halos.clone();
    let mut fired: Mask = 0;
    let mut salvos = 0usize;
    let mut known_at: Option<(usize, usize)> = None;
    let mut trace = String::new();
    let mut wasted_in_game = 0i64;
    let class_of = |v: u8| -> usize { (4 - v) as usize }; // 4->0, 3->1, 2->2

    loop {
        if game.is_won() && is_resolved(&game) {
            break;
        }
        if salvos >= CAP {
            st.unresolved += 1;
            return;
        }
        let k = knowledge(&lists, &halos);
        if k.unique && known_at.is_none() {
            known_at = Some((salvos, (truth_ship & !fired).count_ones() as usize));
        }
        let phase = if k.unique { "after-known".to_string() } else { format!("hunt-size-{}", game.ai_target_size()) };

        let Some(idx) = next_salvo(&mut game) else {
            st.unresolved += 1;
            return;
        };
        let cells: Vec<(usize, usize)> = idx.iter().map(|&i| (i / 10, i % 10)).collect();
        let smask = cells.iter().fold(0, |m, &(r, c)| m | bit(r, c));
        let any_ship_or = k.or[0] | k.or[1] | k.or[2];
        // Same, from the per-class lists alone (no cross-class adjacency).
        let per_class_or = lists.iter().flat_map(|l| l.iter()).fold(0, |m, &p| m | p);
        let mut has_kill = false;
        let mut labels: Vec<&str> = Vec::new();
        for &(r, c) in &cells {
            let b = bit(r, c);
            let inner = (1..=8).contains(&r) && (1..=8).contains(&c);
            let certain = (0..3).any(|i| k.and[i] & b != 0) || any_ship_or & b == 0;
            let kind = if fired & b != 0 {
                if certain { 5 } else { 4 }
            } else if !inner {
                3
            } else if any_ship_or & b == 0 {
                if per_class_or & b == 0 { 2 } else { 6 }
            } else if (0..3).any(|i| k.and[i] & b != 0) {
                has_kill = true;
                0
            } else {
                1
            };
            *st.shots.entry((phase.clone(), kind)).or_default() += 1;
            if matches!(kind, 2 | 3 | 5 | 6) {
                wasted_in_game += 1;
            }
            labels.push(KINDS[kind]);
        }

        // Bag-blind groups per class among cells whose class membership varies.
        for cls in 0..3 {
            let varying: Vec<Mask> = cells.iter().map(|&(r, c)| bit(r, c)).filter(|&b| k.or[cls] & b != 0 && k.and[cls] & b == 0).collect();
            if varying.len() < 2 {
                continue;
            }
            let mut subsets: Vec<Mask> = Vec::new();
            for i in 0..varying.len() {
                for j in (i + 1)..varying.len() {
                    subsets.push(varying[i] | varying[j]);
                }
            }
            if varying.len() == 3 {
                subsets.push(varying[0] | varying[1] | varying[2]);
            }
            let blind = subsets.iter().any(|&sub| {
                let mut it = lists[cls].iter().map(|&h| (h & sub).count_ones());
                let first = it.next();
                it.all(|n| Some(n) == first)
            });
            if blind {
                *st.bag_blind.entry(4 - cls as u8).or_default() += 1;
            }
        }

        // Fully predictable bag (needs the joint list).
        if let Some(joint) = &k.joint {
            let bag = |j: &[Mask; 3]| [(j[0] & smask).count_ones(), (j[1] & smask).count_ones(), (j[2] & smask).count_ones()];
            let first = bag(&joint[0]);
            if !has_kill && joint.iter().all(|j| bag(j) == first) {
                st.predictable_no_kill += 1;
            }
        }

        let _ = game.fire(&idx);
        salvos += 1;
        fired |= smask;
        if k.unique {
            st.salvos_after_known += 1;
        }
        let vals: Vec<u8> = cells.iter().map(|&(r, c)| g[r][c]).collect();
        for v in [4u8, 3, 2] {
            let n = vals.iter().filter(|&&x| x == v).count() as u32;
            let cls = class_of(v);
            let keep: Vec<usize> = (0..lists[cls].len()).filter(|&i| (lists[cls][i] & smask).count_ones() == n).collect();
            lists[cls] = keep.iter().map(|&i| lists[cls][i]).collect();
            halos[cls] = keep.iter().map(|&i| halos[cls][i]).collect();
        }
        let name = |(r, c): (usize, usize)| format!("{}{}", (b'A' + c as u8) as char, r + 1);
        trace.push_str(&format!("T{salvos} {} -> {:?} [{}] {}\n", cells.iter().map(|&x| name(x)).collect::<Vec<_>>().join(" "), vals, phase, labels.join("/")));
    }

    st.games += 1;
    st.salvos += salvos;
    match known_at {
        Some((at, unfired)) => {
            let slack = salvos as i64 - (at + unfired.div_ceil(3)) as i64;
            *st.slack.entry(slack).or_default() += 1;
            trace = format!("known after T{at} with {unfired} unfired ship cells, slack {slack}\n{trace}");
        }
        None => st.never_known += 1,
    }
    st.examples.push((wasted_in_game, line.trim().to_string(), format!("{salvos} salvos, ") + &trace));
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: {} <boards_file> [games] [threads]", args[0]);
        std::process::exit(2);
    }
    let games: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(2000);
    let threads: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(8);
    let lines: Vec<String> = BufReader::new(File::open(&args[1]).expect("open boards file")).lines().map_while(Result::ok).filter(|l| !l.trim().is_empty()).take(games).collect();

    let base: [Vec<Mask>; 3] = [placements(4, 1), placements(3, 2), placements(2, 3)];
    let base_halos: [Vec<Mask>; 3] = [base[0].iter().map(|&m| halo(m)).collect(), base[1].iter().map(|&m| halo(m)).collect(), base[2].iter().map(|&m| halo(m)).collect()];
    eprintln!("placements: battleship {}, cruiser pairs {}, frigate triples {}", base[0].len(), base[1].len(), base[2].len());
    let base = Arc::new(base);
    let base_halos = Arc::new(base_halos);
    let lines = Arc::new(lines);

    let handles: Vec<_> = (0..threads)
        .map(|t| {
            let (base, base_halos, lines) = (base.clone(), base_halos.clone(), lines.clone());
            thread::spawn(move || {
                let mut st = Stats::default();
                for line in lines.iter().skip(t).step_by(threads) {
                    audit_game(line, &base, &base_halos, &mut st);
                }
                st
            })
        })
        .collect();
    let mut st = Stats::default();
    for h in handles {
        st.merge(h.join().unwrap());
    }

    let g = st.games as f64;
    println!("games {}  unresolved {}  mean salvos {:.3}", st.games, st.unresolved, st.salvos as f64 / g);
    println!();
    println!("shots per game by phase and kind (phase = AI target size, or after the layout was already unique):");
    let mut phases: Vec<String> = st.shots.keys().map(|(p, _)| p.clone()).collect();
    phases.sort();
    phases.dedup();
    print!("{:<14}", "phase");
    for k in KINDS {
        print!("{k:>12}");
    }
    println!("{:>12}", "total");
    for p in &phases {
        print!("{p:<14}");
        let mut tot = 0;
        for i in 0..KINDS.len() {
            let n = st.shots.get(&(p.clone(), i)).copied().unwrap_or(0);
            tot += n;
            print!("{:>12.3}", n as f64 / g);
        }
        println!("{:>12.3}", tot as f64 / g);
    }
    let wasted: usize = st.shots.iter().filter(|((_, k), _)| matches!(*k, 2 | 3 | 5 | 6)).map(|(_, v)| v).sum();
    println!();
    println!("provably wasted shots per game (dead + dead-joint + outer + refire-dead): {:.3}  (~{:.3} salvos)", wasted as f64 / g, wasted as f64 / g / 3.0);
    println!("salvos fired after the layout was unique: {:.3} per game", st.salvos_after_known as f64 / g);
    println!("salvos with a fully predictable bag and no kill shot: {:.3} per game", st.predictable_no_kill as f64 / g);
    let mut bb: Vec<_> = st.bag_blind.iter().collect();
    bb.sort();
    println!("bag-blind salvos per game: {}", bb.iter().map(|(v, n)| format!("size {v}: {:.3}", **n as f64 / g)).collect::<Vec<_>>().join(", "));
    println!();
    println!("slack after the layout became unique (actual - (known_at + ceil(unfired/3))):");
    let mut sl: Vec<_> = st.slack.iter().collect();
    sl.sort();
    for (s, n) in sl {
        println!("  {s:>3}: {n:>7} ({:.2}%)", *n as f64 / g * 100.0);
    }
    println!("  never unique before resolution: {}", st.never_known);

    st.examples.sort_by(|a, b| b.0.cmp(&a.0));
    if let Ok(mut f) = File::create("audit_examples.txt") {
        use std::io::Write;
        for (s, line, trace) in st.examples.iter().take(30) {
            let _ = writeln!(f, "wasted shots {s}  board {line}\n{trace}");
        }
        println!("\nworst examples written to audit_examples.txt");
    }
}
