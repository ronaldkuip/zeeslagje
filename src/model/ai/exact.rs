//! Exact, salvo-history-only "could a ship of size >= 2 be here at all?"
//! check, as u128 bitmasks (bit r*10+c).
//!
//! The line-wise FSM that drives hunting only reasons per row/column, so
//! it misses deductions that span several salvos: a cell can still look
//! alive to it long after every placement of every ship class through
//! that cell has been ruled out by the bags. Auditing replayed games
//! (see `src/bin/audit_waste.rs`) showed ~3 such provably-empty shots per
//! game, mostly while hunting Cruisers.
//!
//! Per class (1 Battleship window, 2 Cruiser windows, 3 Frigate windows,
//! no two windows of the class touching), a placement is consistent with
//! a salvo iff the number of its cells among the 3 fired equals the
//! number of that class's value in the bag. While the Battleship and the
//! Cruisers are down to a few placements each, those are branched on
//! explicitly so the no-touching rule BETWEEN classes prunes too (e.g.
//! nothing can lie next to an identified Battleship). The OR over every
//! surviving combination is the set of cells that could still hold a
//! ship; everything outside it is provably empty. Sound by construction:
//! the true layout always satisfies every salvo and the touching rule, so
//! it is never filtered out.

use std::collections::HashMap;
use std::sync::OnceLock;

pub(crate) type Mask = u128;

const CELLS: Mask = (1u128 << 100) - 1;

pub(crate) fn bit((r, c): (usize, usize)) -> Mask {
    1u128 << (r * 10 + c)
}

fn halo(m: Mask) -> Mask {
    let mut out = 0;
    for r in 0..10 {
        for c in 0..10 {
            if m & bit((r, c)) != 0 {
                for nr in r.saturating_sub(1)..=(r + 1).min(9) {
                    for nc in c.saturating_sub(1)..=(c + 1).min(9) {
                        out |= bit((nr, nc));
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
            out.push((0..len).fold(0, |m, i| m | bit((r, c + i))));
        }
    }
    for c in 1..=8 {
        for r in 1..=(9 - len) {
            out.push((0..len).fold(0, |m, i| m | bit((r + i, c))));
        }
    }
    out
}

/// Bits per salvo in a packed count signature: one hex digit per salvo,
/// so subtracting window counts never borrows across salvos (the top bit
/// of each digit is a guard), and 32 salvos fit a u128.
const FIELD_BITS: usize = 4;
const MAX_SALVOS: usize = 128 / FIELD_BITS;
const GUARD: u128 = 0x8888_8888_8888_8888_8888_8888_8888_8888;

/// `n - used`, field by field, or None if any field would go negative.
fn remainder(n: u128, used: u128) -> Option<u128> {
    let d = (n | GUARD).wrapping_sub(used);
    (d & GUARD == GUARD).then_some(d & !GUARD)
}

struct Class {
    value: usize,
    count: usize,
    windows: Vec<Mask>,
    halos: Vec<Mask>,
}

fn classes() -> &'static [Class; 3] {
    static C: OnceLock<[Class; 3]> = OnceLock::new();
    C.get_or_init(|| {
        let make = |value: usize, len: usize, count: usize| {
            let windows = windows(len);
            let halos = windows.iter().map(|&w| halo(w)).collect();
            Class { value, count, windows, halos }
        };
        [make(4, 4, 1), make(3, 3, 2), make(2, 2, 3)]
    })
}

/// One class, prepared against one salvo history: each window's packed
/// per-salvo hit counts, the class's target counts `n`, and an index from
/// signature to windows.
struct Prepared {
    class: &'static Class,
    sig: Vec<u128>,
    n: u128,
    by_sig: HashMap<u128, Vec<usize>>,
}

impl Prepared {
    fn new(class: &'static Class, salvos: &[Mask], history: &[([(usize, usize); 3], [usize; 3])]) -> Self {
        let n = history.iter().enumerate().fold(0u128, |acc, (i, (_, values))| {
            acc | ((values.iter().filter(|&&v| v == class.value).count() as u128) << (i * FIELD_BITS))
        });
        let sig: Vec<u128> = class
            .windows
            .iter()
            .map(|&w| salvos.iter().enumerate().fold(0u128, |s, (i, &salvo)| s | (((w & salvo).count_ones() as u128) << (i * FIELD_BITS))))
            .collect();
        let mut by_sig: HashMap<u128, Vec<usize>> = HashMap::new();
        for (i, &s) in sig.iter().enumerate() {
            by_sig.entry(s).or_default().push(i);
        }
        Prepared { class, sig, n, by_sig }
    }

    fn ok(&self, a: usize, forbidden: Mask) -> bool {
        self.class.windows[a] & forbidden == 0
    }

    fn apart(&self, a: usize, b: usize) -> bool {
        a != b && self.class.windows[a] & self.class.halos[b] == 0
    }

    fn with_sig(&self, s: u128) -> &[usize] {
        self.by_sig.get(&s).map(|l| l.as_slice()).unwrap_or(&[])
    }

    /// Union of every consistent placement avoiding `forbidden`. Rather
    /// than enumerating placements (over 100k Frigate triples), each window
    /// not yet covered searches for partners: the last partner's signature
    /// is fully determined, so it's a hash lookup.
    fn union(&self, forbidden: Mask) -> Mask {
        let ws = &self.class.windows;
        let mut covered: Mask = 0;
        for a in 0..ws.len() {
            if ws[a] & !covered == 0 || !self.ok(a, forbidden) {
                continue;
            }
            let Some(rest) = remainder(self.n, self.sig[a]) else { continue };
            match self.class.count {
                1 => {
                    if rest == 0 {
                        covered |= ws[a];
                    }
                }
                2 => {
                    if let Some(&b) = self.with_sig(rest).iter().find(|&&b| self.ok(b, forbidden) && self.apart(a, b)) {
                        covered |= ws[a] | ws[b];
                    }
                }
                _ => {
                    'b: for b in 0..ws.len() {
                        if !self.ok(b, forbidden) || !self.apart(a, b) {
                            continue;
                        }
                        let Some(last) = remainder(rest, self.sig[b]) else { continue };
                        for &c in self.with_sig(last) {
                            if self.ok(c, forbidden) && self.apart(a, c) && self.apart(b, c) {
                                covered |= ws[a] | ws[b] | ws[c];
                                break 'b;
                            }
                        }
                    }
                }
            }
        }
        covered
    }

    /// Every consistent placement avoiding `forbidden` (1 or 2 windows —
    /// Battleship or Cruisers), or None once there are more than `limit`.
    fn list(&self, forbidden: Mask, limit: usize) -> Option<Vec<Mask>> {
        let ws = &self.class.windows;
        let mut out = Vec::new();
        for a in 0..ws.len() {
            if !self.ok(a, forbidden) {
                continue;
            }
            let Some(rest) = remainder(self.n, self.sig[a]) else { continue };
            if self.class.count == 1 {
                if rest == 0 {
                    out.push(ws[a]);
                }
            } else {
                for &b in self.with_sig(rest) {
                    if b > a && self.ok(b, forbidden) && self.apart(a, b) {
                        out.push(ws[a] | ws[b]);
                    }
                }
            }
            if out.len() > limit {
                return None;
            }
        }
        Some(out)
    }
}

/// How many Battleship / Cruiser placements to branch on explicitly; above
/// this, that class and the ones after it fall back to their own unions
/// with no further cross-class pruning.
const BRANCH_LIMIT: usize = 6;

/// Cells possible for classes `idx..`, with `forbidden` (the halos of the
/// placements chosen for earlier classes) off-limits. None if some class
/// has no placement left at all — the branch leading here is impossible.
fn explore(prepared: &[Prepared], idx: usize, forbidden: Mask) -> Option<Mask> {
    let Some(p) = prepared.get(idx) else { return Some(0) };
    if p.class.count < 3 {
        if let Some(list) = p.list(forbidden, BRANCH_LIMIT) {
            let mut possible: Option<Mask> = None;
            for placement in list {
                if let Some(rest) = explore(prepared, idx + 1, forbidden | halo(placement)) {
                    possible = Some(possible.unwrap_or(0) | placement | rest);
                }
            }
            return possible;
        }
    }
    let mut possible = 0;
    for q in &prepared[idx..] {
        let u = q.union(forbidden);
        if u == 0 {
            return None;
        }
        possible |= u;
    }
    Some(possible)
}

/// Cells that could still hold a ship of size >= 2 given `history`. With
/// more than `MAX_SALVOS` salvos it gives up and reports every cell as
/// possible (never wrong, just no deduction) — only reachable in long
/// manual games, never in the 30-salvo-capped batch runs.
pub(crate) fn possible_ship_cells(history: &[([(usize, usize); 3], [usize; 3])]) -> Mask {
    if history.len() > MAX_SALVOS {
        return !0;
    }
    let salvos: Vec<Mask> = history.iter().map(|(coords, _)| coords.iter().fold(0, |m, &cell| m | bit(cell))).collect();
    let prepared: Vec<Prepared> = classes().iter().map(|c| Prepared::new(c, &salvos, history)).collect();
    match explore(&prepared, 0, 0) {
        Some(possible) if possible & CELLS != 0 => possible & CELLS,
        // The true layout is always consistent, so nothing possible can
        // only mean a bookkeeping error upstream: claim nothing.
        _ => !0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tiny deterministic PRNG (xorshift) — no test dependency needed.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self, n: usize) -> usize {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 % n as u64) as usize
        }
    }

    /// Random legal layout: per class, the window masks (value 4/3/2).
    fn random_layout(rng: &mut Rng) -> Vec<(usize, Mask)> {
        'retry: loop {
            let mut ships = Vec::new();
            let mut blocked: Mask = 0;
            for class in classes() {
                for _ in 0..class.count {
                    let mut placed = false;
                    for _ in 0..200 {
                        let i = rng.next(class.windows.len());
                        if class.windows[i] & blocked == 0 {
                            blocked |= class.halos[i];
                            ships.push((class.value, class.windows[i]));
                            placed = true;
                            break;
                        }
                    }
                    if !placed {
                        continue 'retry;
                    }
                }
            }
            return ships;
        }
    }

    /// Soundness: no true ship cell is ever reported impossible, whatever
    /// the salvos (random cells, refires included).
    #[test]
    fn never_rules_out_a_real_ship_cell() {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        for _ in 0..400 {
            let ships = random_layout(&mut rng);
            let truth: Mask = ships.iter().fold(0, |m, &(_, w)| m | w);
            let mut history = Vec::new();
            for _ in 0..30 {
                let mut cells = Vec::new();
                while cells.len() < 3 {
                    let cell = (rng.next(10), rng.next(10));
                    if !cells.contains(&cell) {
                        cells.push(cell);
                    }
                }
                let coords = [cells[0], cells[1], cells[2]];
                let value = |cell| ships.iter().find(|&&(_, w)| w & bit(cell) != 0).map_or(0, |&(v, _)| v);
                history.push((coords, [value(coords[0]), value(coords[1]), value(coords[2])]));
                let possible = possible_ship_cells(&history);
                assert_eq!(truth & !possible, 0, "a real ship cell was ruled out after {} salvos", history.len());
            }
        }
    }
}
