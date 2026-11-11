//! Exact, salvo-history-only reasoning over ship placements, as u128
//! bitmasks (cell bit r*10+c; window-set bit = window index).
//!
//! Two jobs share one index:
//!
//! 1. `possible_ship_cells`: which cells could still hold a ship of size
//!    >= 2 at all. The line-wise FSM that drives hunting only reasons per
//!    row/column, so it misses deductions spanning several salvos; the
//!    waste auditor (`src/bin/audit_waste.rs`) found ~3 shots per game on
//!    cells the bags had already proven empty.
//! 2. `Index::placements`: enumerate every placement of a class (2 Cruiser
//!    windows, 3 Frigate windows, no two touching) consistent with every
//!    bag — what `consistent_cruiser_candidates`/`consistent_frigate_
//!    candidates` need, without the O(windows^3) nested loops and a
//!    per-triple pass over the history.
//!
//! A placement is consistent with a salvo iff the number of its cells
//! among the 3 fired equals the number of that class's value in the bag.
//! Each window's per-salvo counts are packed into one u128 "signature"
//! (one hex digit per salvo), so "the partner that completes the bag" is
//! a signature subtraction plus a lookup; and because every class has at
//! most 112 windows, the windows sharing a signature, the windows apart
//! from a given one, and the windows allowed by a cell mask are all
//! window-set u128s — a candidate partner set is a few ANDs.
//!
//! For `possible_ship_cells`, while the Battleship and the Cruisers are
//! down to a few placements each, those are branched on explicitly so the
//! no-touching rule BETWEEN classes prunes too (nothing can lie next to an
//! identified Battleship). Sound by construction: the true layout always
//! satisfies every salvo and the touching rule, so it is never filtered
//! out — `never_rules_out_a_real_ship_cell` checks that at random.

use std::sync::OnceLock;

pub(crate) type Mask = u128;
/// A set of window indices of one class (at most 112 windows).
type WSet = u128;

const CELLS: Mask = (1u128 << 100) - 1;

pub(crate) fn bit((r, c): (usize, usize)) -> Mask {
    1u128 << (r * 10 + c)
}

pub(crate) fn cells(mask: Mask) -> impl Iterator<Item = (usize, usize)> {
    (0..100).filter(move |i| mask & (1u128 << i) != 0).map(|i| (i / 10, i % 10))
}

fn halo(m: Mask) -> Mask {
    let mut out = 0;
    for (r, c) in cells(m) {
        for nr in r.saturating_sub(1)..=(r + 1).min(9) {
            for nc in c.saturating_sub(1)..=(c + 1).min(9) {
                out |= bit((nr, nc));
            }
        }
    }
    out
}

/// Same order as `all_cruiser_windows`/`all_frigate_windows` in
/// heatmap_gen.rs (horizontal row-major, then vertical column-major), so
/// placements enumerate in the same order as the loops they replace.
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

/// Bits per salvo in a packed count signature: one hex digit per salvo
/// (counts are at most 3), with the top bit of each digit as a borrow
/// guard; 32 salvos fit a u128.
const FIELD_BITS: usize = 4;
pub(crate) const MAX_SALVOS: usize = 128 / FIELD_BITS;
const GUARD: u128 = 0x8888_8888_8888_8888_8888_8888_8888_8888;

/// `n - used`, digit by digit, or None if any digit would go negative.
fn remainder(n: u128, used: u128) -> Option<u128> {
    let d = (n | GUARD).wrapping_sub(used);
    (d & GUARD == GUARD).then_some(d & !GUARD)
}

/// One ship class: its windows, each window's halo (the window plus every
/// cell touching it), and for each window the set of windows apart from
/// it (not overlapping or touching — a window is never apart from itself).
pub(crate) struct Class {
    pub(crate) value: usize,
    pub(crate) count: usize,
    pub(crate) windows: Vec<Mask>,
    halos: Vec<Mask>,
    apart: Vec<WSet>,
}

impl Class {
    fn new(value: usize, len: usize, count: usize) -> Class {
        let windows = windows(len);
        let halos: Vec<Mask> = windows.iter().map(|&w| halo(w)).collect();
        let apart = (0..windows.len()).map(|a| (0..windows.len()).filter(|&b| windows[b] & halos[a] == 0).fold(0, |s, b| s | (1u128 << b))).collect();
        Class { value, count, windows, halos, apart }
    }

    /// Every window of the class, as a window-set.
    fn all(&self) -> WSet {
        (1u128 << self.windows.len()) - 1
    }

    /// Windows lying entirely inside `allowed` cells.
    pub(crate) fn windows_within(&self, allowed: Mask) -> WSet {
        self.windows.iter().enumerate().filter(|(_, &w)| w & !allowed == 0).fold(0, |s, (i, _)| s | (1u128 << i))
    }

    /// Halo of a placement: the union of its windows' halos.
    fn halo_of(&self, placement: Mask) -> Mask {
        self.windows.iter().zip(&self.halos).filter(|(&w, _)| w & placement == w).fold(0, |m, (_, &h)| m | h)
    }
}

pub(crate) fn cruisers() -> &'static Class {
    &classes()[1]
}
pub(crate) fn frigates() -> &'static Class {
    &classes()[2]
}

fn classes() -> &'static [Class; 3] {
    static C: OnceLock<[Class; 3]> = OnceLock::new();
    C.get_or_init(|| [Class::new(4, 4, 1), Class::new(3, 3, 2), Class::new(2, 2, 3)])
}

fn set_iter(mut s: WSet) -> impl Iterator<Item = usize> {
    std::iter::from_fn(move || {
        if s == 0 {
            None
        } else {
            let i = s.trailing_zeros() as usize;
            s &= s - 1;
            Some(i)
        }
    })
}

/// Windows with index > i.
fn above(i: usize) -> WSet {
    !0u128 << 1 << i
}

/// One class indexed against one salvo history: each window's signature,
/// the class's target signature `n` (the bag counts of its value), and a
/// sorted (signature, window-set) table for partner lookups.
pub(crate) struct Index {
    class: &'static Class,
    sig: Vec<u128>,
    n: u128,
    table: Vec<(u128, WSet)>,
}

impl Index {
    /// None once the history no longer fits a signature (see `MAX_SALVOS`).
    pub(crate) fn new(class: &'static Class, history: &[([(usize, usize); 3], [usize; 3])]) -> Option<Index> {
        if history.len() > MAX_SALVOS {
            return None;
        }
        let mut n = 0u128;
        let mut sig = vec![0u128; class.windows.len()];
        for (i, (coords, values)) in history.iter().enumerate() {
            let shift = i * FIELD_BITS;
            n |= (values.iter().filter(|&&v| v == class.value).count() as u128) << shift;
            let salvo = coords.iter().fold(0, |m, &cell| m | bit(cell));
            for (s, &w) in sig.iter_mut().zip(&class.windows) {
                *s |= ((w & salvo).count_ones() as u128) << shift;
            }
        }
        let mut table: Vec<(u128, WSet)> = Vec::new();
        for (i, &s) in sig.iter().enumerate() {
            match table.binary_search_by_key(&s, |&(k, _)| k) {
                Ok(pos) => table[pos].1 |= 1u128 << i,
                Err(pos) => table.insert(pos, (s, 1u128 << i)),
            }
        }
        Some(Index { class, sig, n, table })
    }

    fn with_sig(&self, s: u128) -> WSet {
        self.table.binary_search_by_key(&s, |&(k, _)| k).map_or(0, |pos| self.table[pos].1)
    }

    /// Union of the cells of every consistent placement using only
    /// `allowed` windows. A window is covered iff some full placement
    /// contains it, so each window only has to find one set of partners.
    fn union(&self, allowed: WSet) -> Mask {
        let (ws, apart) = (&self.class.windows, &self.class.apart);
        let allowed = allowed & self.class.all();
        let mut covered: Mask = 0;
        for a in set_iter(allowed) {
            if ws[a] & !covered == 0 {
                continue;
            }
            let Some(rest) = remainder(self.n, self.sig[a]) else { continue };
            let found = match self.class.count {
                1 => rest == 0,
                2 => self.with_sig(rest) & allowed & apart[a] != 0,
                _ => set_iter(allowed & apart[a]).any(|b| remainder(rest, self.sig[b]).is_some_and(|last| self.with_sig(last) & allowed & apart[a] & apart[b] != 0)),
            };
            if found {
                covered |= ws[a];
            }
        }
        covered
    }

    /// Every consistent placement using only `allowed` windows, as cell
    /// masks, in index order (a < b < c) — or None once more than `limit`.
    pub(crate) fn placements(&self, allowed: WSet, limit: usize) -> Option<Vec<Mask>> {
        let (ws, apart) = (&self.class.windows, &self.class.apart);
        let allowed = allowed & self.class.all();
        let mut out = Vec::new();
        for a in set_iter(allowed) {
            let Some(rest) = remainder(self.n, self.sig[a]) else { continue };
            match self.class.count {
                1 => {
                    if rest == 0 {
                        out.push(ws[a]);
                    }
                }
                2 => {
                    for b in set_iter(self.with_sig(rest) & allowed & apart[a] & above(a)) {
                        out.push(ws[a] | ws[b]);
                    }
                }
                _ => {
                    for b in set_iter(allowed & apart[a] & above(a)) {
                        let Some(last) = remainder(rest, self.sig[b]) else { continue };
                        for c in set_iter(self.with_sig(last) & allowed & apart[a] & apart[b] & above(b)) {
                            out.push(ws[a] | ws[b] | ws[c]);
                        }
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
fn explore(indexes: &[Index], idx: usize, forbidden: Mask) -> Option<Mask> {
    let Some(ix) = indexes.get(idx) else { return Some(0) };
    let allowed = ix.class.windows_within(!forbidden);
    if ix.class.count < 3 {
        if let Some(list) = ix.placements(allowed, BRANCH_LIMIT) {
            let mut possible: Option<Mask> = None;
            for placement in list {
                if let Some(rest) = explore(indexes, idx + 1, forbidden | ix.class.halo_of(placement)) {
                    possible = Some(possible.unwrap_or(0) | placement | rest);
                }
            }
            return possible;
        }
    }
    let mut possible = 0;
    for q in &indexes[idx..] {
        let u = q.union(q.class.windows_within(!forbidden));
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
    let Some(indexes) = classes().iter().map(|c| Index::new(c, history)).collect::<Option<Vec<_>>>() else {
        return !0;
    };
    match explore(&indexes, 0, 0) {
        Some(possible) if possible & CELLS != 0 => possible & CELLS,
        // The true layout is always consistent, so nothing possible can
        // only mean a bookkeeping error upstream: claim nothing.
        _ => !0,
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Tiny deterministic PRNG (xorshift) — no test dependency needed.
    pub(crate) struct Rng(pub u64);
    impl Rng {
        pub(crate) fn next(&mut self, n: usize) -> usize {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 % n as u64) as usize
        }
    }

    /// Random legal layout: per class, the window masks (value 4/3/2).
    pub(crate) fn random_layout(rng: &mut Rng) -> Vec<(usize, Mask)> {
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

    /// A random salvo (3 distinct cells anywhere) and its bag for `ships`.
    pub(crate) fn random_salvo(rng: &mut Rng, ships: &[(usize, Mask)]) -> ([(usize, usize); 3], [usize; 3]) {
        let mut cells = Vec::new();
        while cells.len() < 3 {
            let cell = (rng.next(10), rng.next(10));
            if !cells.contains(&cell) {
                cells.push(cell);
            }
        }
        let coords = [cells[0], cells[1], cells[2]];
        let value = |cell| ships.iter().find(|&&(_, w)| w & bit(cell) != 0).map_or(0, |&(v, _)| v);
        (coords, [value(coords[0]), value(coords[1]), value(coords[2])])
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
                history.push(random_salvo(&mut rng, &ships));
                let possible = possible_ship_cells(&history);
                assert_eq!(truth & !possible, 0, "a real ship cell was ruled out after {} salvos", history.len());
            }
        }
    }

    /// `Index::placements` against brute force over every window
    /// combination with a per-salvo recount.
    #[test]
    fn placements_match_brute_force() {
        let mut rng = Rng(0x5DEE_CE66_D1CE_5EED);
        for _ in 0..150 {
            let ships = random_layout(&mut rng);
            let mut history = Vec::new();
            for _ in 0..rng.next(12) + 1 {
                history.push(random_salvo(&mut rng, &ships));
            }
            for class in classes() {
                let ix = Index::new(class, &history).unwrap();
                let fast = ix.placements(!0, usize::MAX).unwrap();
                let ws = &class.windows;
                let consistent = |p: Mask| {
                    history.iter().all(|(coords, values)| {
                        let fired = coords.iter().fold(0, |m, &cell| m | bit(cell));
                        (p & fired).count_ones() as usize == values.iter().filter(|&&v| v == class.value).count()
                    })
                };
                let mut slow = Vec::new();
                for a in 0..ws.len() {
                    match class.count {
                        1 => slow.push(ws[a]),
                        2 => {
                            for b in (a + 1)..ws.len() {
                                if ws[b] & class.halos[a] == 0 {
                                    slow.push(ws[a] | ws[b]);
                                }
                            }
                        }
                        _ => {
                            for b in (a + 1)..ws.len() {
                                if ws[b] & class.halos[a] != 0 {
                                    continue;
                                }
                                for c in (b + 1)..ws.len() {
                                    if ws[c] & (class.halos[a] | class.halos[b]) == 0 {
                                        slow.push(ws[a] | ws[b] | ws[c]);
                                    }
                                }
                            }
                        }
                    }
                }
                slow.retain(|&p| consistent(p));
                assert_eq!(fast, slow, "class {} after {} salvos", class.value, history.len());
            }
        }
    }
}
