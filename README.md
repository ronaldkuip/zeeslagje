# Zeeslagje — Rust/WASM

A browser Battleship deduction game. All game logic runs in WebAssembly compiled from Rust.

## Rules

- Fire **3 coordinates per salvo**
- Result is **3 digits sorted descending** — ship size or 0 for miss (e.g. `4 1 0`)
- You are **not told which coordinate produced which result**
- A **sunk message** appears when a ship's last cell is hit
- Win by destroying all ships: 1 battleship (4), 2 cruisers (3), 3 frigates (2), 4 submarines (1)

### Placement rules

| Ship type      | Size | Count | Placement zone        | Adjacency             |
|---------------|------|-------|-----------------------|-----------------------|
| Battleship     | 4    | 1     | Inner 8×8 (rows/cols 1–8) | No orthogonal or diagonal neighbours |
| Cruiser        | 3    | 2     | Inner 8×8             | No orthogonal or diagonal neighbours |
| Frigate        | 2    | 3     | Inner 8×8             | No orthogonal or diagonal neighbours |
| Submarine      | 1    | 4     | Full 10×10            | No orthogonal neighbours; diagonal OK |

## The Solver's Underlying Methodology

**Problem formulation.** Unlike the canonical single-cell variant of Battleship, this implementation issues salvos of three shots per turn on a 10×10 grid, and returns results as an unordered multiset — e.g., "4, 1, 0" denotes one hit on a size-4 vessel, one hit on a size-1 vessel, and one miss — without disclosing which coordinate produced which outcome. This partial observability precludes naïve cell-by-cell inference and necessitates a constraint-satisfaction approach: the solver must maintain, and continually revise, the set of board configurations consistent with the complete observation history.

**Line-wise state compression.** For each row and column independently, the solver maintains a finite-state representation encoding the subset of admissible ship placements along that line, given a target vessel length. These automata are precomputed offline as lookup tables, such that each incoming shot induces an O(1) state transition rather than a re-derivation from first principles. This decomposition reduces what would otherwise be a two-dimensional combinatorial problem to a set of tractable one-dimensional subproblems, at the cost of not directly capturing cross-axis dependencies.

**Global consistency enumeration for larger vessels.** For the battleship and cruiser classes, the solver performs exhaustive enumeration over all candidate placements on the full board, filtering this hypothesis space against the entire salvo history — including the adjacency constraint that no two vessels may occupy orthogonally or diagonally adjacent cells. The surviving hypothesis set constitutes an implicit posterior over ship locations; marginal occupancy probability per cell is estimated by frequency of appearance across the surviving hypotheses, yielding a heatmap analogous to a uniform Bayesian posterior over a discrete, combinatorially pruned hypothesis space — rather than a sampled or Monte Carlo approximation.

**Search prioritization: hunt versus target.** Target selection follows a fixed size-priority ordering (battleship, then cruisers, then frigates, then submarines). Upon partial localization of a vessel, the objective function shifts from pure expected-hit maximization toward an information-theoretic criterion: the solver preferentially selects cells that bisect the remaining hypothesis space as evenly as possible, a strategy structurally analogous to optimal strategies in the Rényi–Ulam "twenty questions" problem, rather than continuing to maximize immediate hit probability alone.

**Joint salvo optimization.** Because feedback is only available at the granularity of the full three-shot salvo, shot selection is not performed greedily on a per-cell basis. Instead, the solver evaluates candidate triples of cells jointly, simulating the range of possible aggregate outcomes for each triple, and selects the combination maximizing expected information gain or hit yield over the salvo as a whole.

**Summary characterization.** The system is best described as an exact, incrementally-maintained constraint-propagation engine operating over a discretized hypothesis space — combining per-axis finite-state compression with full-board combinatorial filtering for larger vessels — rather than a stochastic or sampling-based estimator. Its behavior is deterministic given the observation history, and its probability estimates derive from exact counting over surviving logical hypotheses rather than from randomized simulation.

## Project structure

```
zeeslagje/
├── Cargo.toml        # Rust dependencies
├── build.sh          # One-step build script
├── index.html        # Frontend (loads WASM via ES module)
└── src/
    └── lib.rs        # All game logic (board gen, salvo, state)
```

## Building

Requires Rust 1.94+ and `wasm-pack`:

```bash
chmod +x build.sh
./build.sh
```

This produces `pkg/` containing the WASM binary and JS bindings.

## Running

```bash
python3 serve.py 8080
# open http://localhost:8080
```

> **Note:** Must be served over HTTP (not `file://`) due to WASM/ES module restrictions.

`serve.py` is a thin wrapper around `python -m http.server` that additionally
accepts `POST /log`, which the page uses to append one line per finished game
(date, time, turns played) to `simpleresult.txt`. Plain `python -m
http.server` still works for playing — the page just silently skips logging
if that endpoint isn't there.

## WASM API

The `Game` class exposed to JS:

| Method | Description |
|--------|-------------|
| `new Game()` | Generate a new random board |
| `game.reset()` | Start a fresh game |
| `game.fire(Uint32Array[3])` | Fire a salvo; returns JSON `SalvoResult` |
| `game.is_fired(idx)` | Check if flat index was already fired |
| `game.is_won()` | True when all ships are sunk |
| `game.turn()` | Current turn number |
| `game.log_json()` | Full salvo history as JSON |
| `game.ships_json()` | Fleet status (name, size, sunk) as JSON |

Flat index: `row * 10 + col`, both 0-based.

## Running tests

```bash
cargo test
```

## Cruiser identifiability measurement

`src/bin/measure_identifiability.rs` is a standalone tool (not part of the
game itself) that answers a specific question: once every ship of a given
size is sunk, is the exact layout always uniquely recoverable from the raw
salvo history alone? It re-derives this from first principles — brute-force
checking every possible placement of that ship type (pairs of windows for
the 2 Cruisers, triples for the 3 Frigates) against every salvo fired so
far — independent of the AI's own (fallible) deduction code, so it can be
trusted as ground truth for that question.

```bash
cargo run --release --bin measure_identifiability -- 2000   # number of self-play games
```

Writes `identifiability.csv` (one row per ship-type per turn, from the turn
the first ship of that type sinks onward: game id, turn, ship type, how
many of that type are sunk, how many candidate layouts are still
consistent with the evidence, and whether the true layout is among them —
always true unless there's a bug in the tool itself) and prints a summary
for both ship types. Empirically (2000 games): at the moment both Cruisers
are sunk, the layout is uniquely determined only about half the time; for
Frigates (3 ships instead of 2, so more combinatorial room) it's closer to
1 in 10. Neither ever narrows further by true game end under current play,
since the AI doesn't fire shots aimed at disambiguating either — see the
session notes for why this is a fundamental limit of the unordered-bag
observation model, not a fixable bug.
