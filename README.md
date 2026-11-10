# Zeeslagje — Rust/WASM

A browser Battleship-variant deduction game and solver. All game logic —
board generation, salvo resolution, and the AI's deduction engine — runs
in WebAssembly compiled from Rust; the browser only renders it.

## Rules

- Fire **3 coordinates per salvo**
- Result is **3 digits sorted descending** — ship size or 0 for a miss
  (e.g. `4 1 0`)
- You are **not told which coordinate produced which result** — that's
  the whole source of ambiguity the solver has to work through
- A **sunk message** appears when a ship's last cell is hit
- Win by destroying every ship: 1 battleship (4), 2 cruisers (3), 3
  frigates (2), 4 submarines (1)

### Placement rules

| Ship type      | Size | Count | Placement zone        | Adjacency             |
|---------------|------|-------|-----------------------|-----------------------|
| Battleship     | 4    | 1     | Inner 8×8 (rows/cols 1–8) | No orthogonal or diagonal neighbours |
| Cruiser        | 3    | 2     | Inner 8×8             | No orthogonal or diagonal neighbours |
| Frigate        | 2    | 3     | Inner 8×8             | No orthogonal or diagonal neighbours |
| Submarine      | 1    | 4     | Full 10×10            | No orthogonal neighbours; diagonal OK |

### Submarines are a minor, undirected part of the game

Submarines are placed and still count toward winning, but the solver
does not actively hunt them. `AiPlayer::choose_submarine_shots` only
ever kicks in once Battleship, both Cruisers, and all 3 Frigates are
fully sunk — at that point it just picks the first still-plausible,
unfired cells in scan order, with no deduction or scoring driving the
choice. Cells submarines couldn't occupy still get ruled out as a
*side effect* of eliminating them for the bigger ship types (adjacency
to a confirmed cell rules out a submarine there too), so it isn't
literally uniform-random — but nothing in the codebase reasons about
submarine placement on its own terms. In practice, finding them plays
out as close to random events layered on top of the real deduction work
happening for the other 4 ship types.

This is also why the batch/production solving path (see below)
excludes submarines from its board format entirely: since nothing
meaningful is being deduced about them, folding them into the
"salvos to solve a board" metric would just be adding noise to a number
this project actually optimizes.

## The Solver's Methodology

**Problem formulation.** Salvos are 3 shots per turn, and results come
back as an unordered multiset — e.g. "4, 1, 0" means one hit on a
size-4 ship, one hit on a size-1 ship, one miss — without saying which
coordinate produced which value. This partial observability rules out
naïve cell-by-cell inference; the solver has to maintain, and
continually revise, the set of board configurations still consistent
with the complete observation history.

**Line-wise state compression.** For each row and column independently,
the solver keeps a finite-state representation of which placements of a
given ship size remain possible along that line. These automata are
precomputed offline as lookup tables (`src/fsm_tables.rs`), so each shot
is an O(1) state transition rather than a re-derivation from scratch.
This turns what would otherwise be a 2D combinatorial problem into a set
of tractable 1D subproblems — at the cost of not directly capturing
cross-axis (row/column) dependencies on its own. See `model/ai/fsm.rs`.

**Confirmed-hit-by-elimination.** Because a salvo's result is an
unordered bag, a single "3" in one salvo doesn't say which of its 3
cells was the Cruiser hit — but bag arithmetic across *multiple* salvos
often does. If N salvos between them can only be explained by a
specific coordinate once every other candidate in each bag has been
ruled out elsewhere, that coordinate is provably a real hit, no
placement-shape reasoning required. This coordinate-identity-only
approach was deliberately chosen over enumerating candidate window
shapes for this specific inference, after an earlier, since-removed
version of the idea turned out capable of declaring "phantom" ships by
mixing cells from 2 different real ships of the same size. Once a cell
is confirmed this way, its neighbours get eliminated for every *other*
ship size immediately (never its own size — a single confirmed cell
can't yet know where the rest of its own ship lies). See
`derive_confirmed_cruiser_hits_by_elimination` /
`derive_confirmed_frigate_hits_by_elimination` /
`derive_confirmed_battleship_hits_by_elimination` and
`apply_adjacency_elimination_around` in `model/ai/fsm.rs`.

**Global consistency enumeration and cross-reasoning.** For Cruisers and
Frigates, the solver enumerates every straight-line window still
consistent with the FSM state and the full salvo history, pairs (or
triples, for the 3 Frigates) them into full-layout hypotheses, and
additionally cross-checks Cruiser hypotheses against Frigate hypotheses
for mutual adjacency — a hypothesis that's individually still
consistent can be provably wrong if *every* remaining hypothesis of the
other type happens to sit next to it, something neither ship type's own
salvo history says anything about on its own. The surviving hypothesis
set is an implicit posterior over layouts; per-cell occupancy
probability (the heatmap) is exact frequency-of-appearance across
survivors, not a sampled approximation. See `model/ai/heatmap_gen.rs`.

**Hunt-phase shot scoring, with cross-size blending.** Target priority
is fixed (Battleship → Cruiser → Frigate), and cell scoring for the
currently-hunted size is read straight from the FSM tables. On top of
that, scoring also blends in a small, capped contribution from the
*next smaller* size's current line state — heavily weighted so it can
only ever break an exact tie in the primary size's own ranking, never
override it — so that when two cells are genuinely equally good for the
ship being hunted, the one that's also useful for what's hunted next
gets picked. Measured to reduce average salvo count by roughly 3% on a
fixed board set. See `best_cell_for_size_blended` in
`model/ai/heatmap_ops.rs`.

**Sequential (not joint) salvo construction.** Each salvo's 3 cells are
chosen one at a time — best cell first, then a hypothetical miss is
folded into a scratch copy of the FSM state, then the next-best cell
against *that* updated state, and so on. This is a greedy algorithm,
not an exhaustive search over all candidate triples — an exhaustive
joint-triple search was implemented and measured against it directly,
and came back both slower (3–4×) and very slightly worse on average.
That result lines up with theory: the quantity being maximized (total
alive-placement elimination from a set of misses) is a submodular set
function, and greedy is a provably strong approximation for exactly
that class of problem, which the exhaustive search failed to beat in
practice. See `best_cell_by_score` and `apply_hypothetical_miss` in
`model/ai/heatmap_ops.rs` / `model/ai/hypothetical.rs`.

**Frigate endgame lookahead.** Greedy construction has one blind spot:
its hypothetical-miss folding can't see bag ambiguity. With "B4 plus
either B3 or C4" left, a miss assumed at C4 makes B3 a certain hit, so
greedy fires B3 and C4 together — and since exactly one of them is
always the hit, the bag comes back identical either way, forcing an
extra refire salvo later. So once the Battleship and both Cruisers are
sunk and at most 1000 joint (Cruiser, Frigate) hypotheses remain, every
candidate triple is scored by a one-step lookahead: group the surviving
hypotheses by the bag each would produce, estimate the salvos still
needed per group (remaining unfired Frigate cells, plus a penalty for
groups left ambiguous — larger when only a refire could still separate
them), and replace the greedy salvo only if some triple is strictly
better. Measured on a fixed 20,000-board set: average salvos to solve
dropped from 17.08 to 16.04, with every identified layout checked
against ground truth. (Letting the same lookahead also override the
minimax disambiguation salvos below was tried and came out slightly
worse — a one-step estimate is too coarse to beat that search at pure
information gathering.) See `frigate_endgame_salvo` in
`model/ai/heatmap_ops.rs`.

**Deferred Battleship cells as anchors.** Once the Battleship's exact
4 cells are identified, its still-unfired cells are no longer fired
straight away: their value (4) is known in advance, so firing them
teaches nothing — but they still have to be hit to sink it. Hunting
moves on to the Cruisers with all 3 shots, and the owed cells are
slotted into salvos only where they cost nothing: in place of a shot
that would teach nothing anyway (known-empty filler, an outer-ring
cell, a refire of an already-hit Battleship cell), as tie-breaking
fillers in the minimax disambiguation search, and as hits still owed
in the Frigate endgame lookahead — which is what keeps them from
piling up into an extra Battleship-only salvo at the very end (without
that, 455 of 1000 games ended that way and the average got worse).
Measured on a fixed 50,000-board set: 16.07 → 15.99 average salvos,
same compute, every identified layout (Battleship included) checked
against ground truth. See `pending_battleship_cells` and
`with_pending_battleship_cells` in `model/ai/heatmap_ops.rs`. Over all
34,537,411 production boards (run_006): 16.361 → 16.269, with boards
needing 21+ salvos nearly halved (0.235% → 0.129%) and 6% less compute.

**No bag-blind shot groups.** The greedy hunt's hypothetical-miss
folding can pick shots whose combined hit count is the same in every
surviving layout — e.g. both possible ends C4 and F4 of a Cruiser
already hit at D4+E4: exactly one of them hits, but the unordered bag
can't say which, so a refire is needed later anyway. While hunting
Cruisers or Frigates, any group of 2–3 nearby shots that each vary
across the surviving hypotheses but whose hit count never does gets
its last-picked cell excluded and the salvo re-picked (at most 6
times). Measured on the fixed 50,000/20,000-board sets: 15.986 →
15.927 and 15.976 → 15.928, about 10–15% more compute. See
`bag_blind_cell` in `model/ai/heatmap_ops.rs`.

**Kill shots once the layout is known.** The line-wise FSM that drives
hunting never sees cross-salvo deductions, so once the Cruiser or
Frigate layout was already down to a single hypothesis it kept probing
cells it still thought could hold a ship — e.g. Battleship B4–E4 with
only C4 unfired and the last Frigate pinned to I3–I4, yet the advisor
suggested H4 · E2 · I9. Now, while hunting size 3 or 2, if that size's
hypotheses collapse to one layout its unfired cells (plus any pending
Battleship cells) are fired first. Frigates use the cross-reasoned
list (cheap by then); Cruisers only the raw list, since cross-checking
them mid-hunt means enumerating still-open Frigate triples (~4× solve
time for a further ~0.05 salvo). Measured on the fixed 50,000/20,000-
board sets: 15.927 → 15.298 and 15.928 → 15.299, ~1.5× compute. See
`known_unfired_cells` in `model/ai/heatmap_ops.rs`.

**Never fire at provably empty cells.** The waste auditor (below)
showed the solver spending ~3 shots per game — mostly while hunting
Cruisers — on cells the salvo history already proves empty of every
ship of size >= 2: the line-wise FSM can't see deductions spanning
several salvos. `model/ai/exact.rs` computes the exact set of cells
that could still hold a ship, as u128 bitmasks: per class, a placement
must match every bag's count for that class; while the Battleship and
Cruisers are down to a few placements each, those are branched on so
the no-touching rule between classes prunes too (nothing can lie next
to an identified Battleship). Frigate triples (112,872 of them) are
never enumerated: each window looks for partners whose packed
per-salvo counts complete the bag, a hash lookup. The hunt (Cruiser
and Frigate phases) excludes every cell outside that set, and
`with_pending_battleship_cells` treats them as free slots. A randomized
test checks no real ship cell is ever ruled out. Measured on the fixed
50,000/20,000-board sets: 15.298 → 14.717 and 15.299 → 14.727, ~30%
more single-thread compute.

**Disambiguation, once hunting is done.** After every ship of size ≥2
is sunk, the exact layout can still be ambiguous (see
`measure_identifiability` below). Two dedicated strategies run at that
point, tried in order:
- **Anchored isolation** — if the heatmaps show one cell that's
  provably Cruiser-only and a different cell that's provably
  Frigate-only, firing both together alongside a confirmed Battleship
  cell (whose value is unambiguous) lets the resulting bag be decoded
  by elimination instead of probability, resolving both cells in one
  salvo.
- **Minimax disambiguation search** — otherwise, an exhaustive search
  over candidate triples (small pool, capped) picks the salvo that
  narrows the surviving hypothesis set as evenly as possible in the
  worst case — the "20 questions" strategy an exhaustive triple search
  is actually the right tool for, unlike the hunt phase above.

See `anchored_isolation_shot` and `disambiguation_shots` in
`model/ai/heatmap_ops.rs`.

**Summary.** The system is an exact, incrementally-maintained
constraint-propagation engine over a discretized hypothesis space —
per-axis FSM compression plus full-board combinatorial filtering and
cross-type reasoning for the larger ships — not a stochastic estimator.
Behavior is deterministic given the observation history; probabilities
come from exact counting over surviving logical hypotheses.

## Architecture (after the MVC/module refactor)

The original single-file `src/ai.rs` (deduction engine) and `src/lib.rs`
(wasm API, including all tests) have been split along both a functional
axis and a Model/Controller/View layering:

```
src/
├── lib.rs                    # crate root: module wiring + public re-exports
├── fsm_tables.rs              # precomputed FSM transition/value tables
├── tests.rs                    # the full test suite (moved out of lib.rs)
├── model/                      # pure data + pure logic, no JSON/wasm awareness
│   ├── fleet.rs                  # ship/board types, placement, board generation
│   └── ai/
│       ├── mod.rs                  # AiPlayer struct + per-round coordinators
│       ├── fsm.rs                    # line-state FSM, confirmed-hit-by-elimination
│       ├── heatmap_gen.rs             # candidate enumeration, cross-reasoning, heatmaps
│       ├── heatmap_ops.rs             # shot scoring, disambiguation, choose_shots
│       ├── battleship.rs              # the Battleship cross-4 candidate-mask subsystem
│       └── hypothetical.rs             # scratch-state FSM lookups used only by hunt scoring
├── controller/
│   └── game.rs                 # the #[wasm_bindgen] Game API — the only thing JS calls
└── view/
    └── json.rs                  # presentation DTOs + derivation logic for debug endpoints
```

`model/`, `controller/`, and `view/` map onto Model/Controller/View
fairly literally: the whole deduction engine and board/ship types are
pure Model code with no serialization concerns at all; `controller/
game.rs` is the thin orchestration layer JS actually talks to; `view/
json.rs` holds the DTOs and derivation logic (ground-truth
cross-referencing for the debug endpoints, mostly) that used to be
mixed into Controller methods. Splitting that mixing out — not just
moving files around — was the real point of applying MVC here.

Every `model/ai/*.rs` bucket file is a second `impl AiPlayer { ... }`
block in a sibling module of where `AiPlayer` itself is defined, rather
than a separate type each bucket owns a slice of — Rust lets a type's
impl block span multiple files as long as each lives in a descendant
module of the defining one, which gave the same file-per-concern
organization with zero call-site rewrites anywhere in the crate.

This reorganization was verified behavior-identical, not just via the
test suite: running the batch solver against a fixed 5000-board set
before and after produced a byte-for-byte identical salvo histogram
across all 5000 games, and the same held on the VPS's full ~34.5M-board
production set.

## Two ways to run it

**Interactive (play against/watch the AI):**
```bash
python3 serve.py 8080
# open http://localhost:8080       (index.html — full Chart Table + AI advisor)
# or   http://localhost:8080/player.html   (place your own fleet; submarines skipped, 16 hits not 20)
```
`serve.py` also handles saving/loading boards for later study and
per-session state — see its own docstring for the endpoint list.

**Batch (solve a large file of boards, no interactivity):**
```bash
cargo run --release --bin solve_boards_file -- <input_file> [threads] [error_log] [special_cases_dir]
```
Reads one 64-character board string per line (values 0/2/3/4 —
submarines deliberately excluded from this format), validates each
against the placement rules, solves the rest with the same AI the
browser uses, and reports a histogram of salvo counts (capped at 30)
plus an unresolved count. `deploy/run_all_boards.sh` sequentially drives
this across many `deel_NNNN.csv` files on a resource-constrained VPS;
`deploy/zeeslagje.service` + `deploy/Caddyfile` are the systemd +
reverse-proxy setup for hosting the interactive version there
alongside it.

## Building

Requires Rust and `wasm-pack`:
```bash
chmod +x build.sh
./build.sh
```
Produces `pkg/` (WASM binary + JS bindings). Must be served over HTTP,
not `file://`, due to ES module restrictions — use `serve.py`, not a
direct file open.

## Testing and verification methodology

```bash
cargo test          # 97 tests
```
Beyond the test suite itself, every solver change in this project's
history has been validated 3 ways before shipping: the full test suite,
repeated self-play stress runs (`self_play_discovers_every_ship_of_
size_at_least_2_by_game_end`, re-run dozens of times since it generates
a fresh random board each run), and A/B comparison — building the
old and new code as separate binaries via a `git worktree`, then
running both against an identical fixed board set and diffing the
resulting salvo histograms directly, not just their averages.

## WASM API

The `Game` class exposed to JS, grouped by purpose (see
`src/controller/game.rs` for full signatures/doc comments):

| Group | Methods |
|---|---|
| Game lifecycle | `new Game()`, `reset()`, `restart_same_board()`, `board_layout_json()`, `load_board_layout_json()` |
| Firing | `fire(indices)`, `is_fired(idx)`, `is_won()`, `turn()`, `log_json()`, `ships_json()` |
| AI advisory | `ai_suggest()`, `ai_target_size()`, `ai_suggest_disambiguation_refire()`, `ai_suggest_disambiguation_last_resort()`, `update_fsm_and_resolve()` |
| Resolution status | `resolution_status_json()`, `ai_cruiser_disambiguation_pending()`, `ai_frigate_disambiguation_pending()` |
| Heatmaps | `cruiser_heatmap_json()`, `frigate_heatmap_json()`, `frigate_heatmap_fraction_json()` |
| Identification | `battleship_candidates_json()`, `battleship_identified_json()`, `found_battleship_json()`, `cruiser_identified_json()`, `frigate_identified_json()` |
| Debug/inspector | `debug_ships_json()`, `fsm_status_json(size)`, `alive_grids_json(size)`, `fully_eliminated_cells_json()`, `cross3_debug_json()`, `cross2_debug_json()`, `cross4_debug_json()` |
| Toggles | `set_refire_allowed(size, bool)`, `is_refire_allowed(size)`, `set_freeze_before_frigates(bool)`, `is_freeze_before_frigates()` |

Flat index convention throughout: `row * 10 + col`, both 0-based.

## Waste auditor

`src/bin/audit_waste.rs` replays the AI on boards from a batch file and
classifies every shot against an independent, first-principles model
of what was knowable at that moment (every placement of every class
consistent with the bags, cross-checked for touching when small enough;
sunk messages unused, so it never knows more than the AI). Each shot is
`kill` (certain ship cell), `probe` (uncertain), `dead` / `dead-joint`
(provably empty, per class / only via the no-touching rule between
classes), `outer` (Submarine-only ring — waste in the Submarine-free
batch format, not in a real game), or a refire whose value was still
uncertain / already certain. It also reports bag-blind salvos and the
slack after the layout became unique, and writes the most wasteful
games, shot by shot, to `audit_examples.txt`.

```bash
cargo run --release --bin audit_waste -- boards.txt 2000 8   # file, games, threads
```

## Cruiser/Frigate identifiability measurement

`src/bin/measure_identifiability.rs` is a standalone tool (not part of
the game itself) that answers one specific question: once every ship of
a given size is sunk, is the exact layout always uniquely recoverable
from the raw salvo history alone? It re-derives this from first
principles — brute-force checking every possible placement of that ship
type against every salvo fired so far — independent of the AI's own
deduction code, so it's trustworthy as ground truth for that question.

```bash
cargo run --release --bin measure_identifiability -- 2000   # number of self-play games
```

Writes `identifiability.csv` (one row per ship-type per turn) and
prints a summary. The specific percentages from early in this project's
history (roughly half the time for Cruisers, 1 in 10 for Frigates, at
the moment both are sunk) predate the confirmed-hit-by-elimination,
cross-reasoning, and disambiguation-shot work described above, which
was built specifically to close gaps like this — re-run the tool for
current figures rather than trusting those old numbers.
