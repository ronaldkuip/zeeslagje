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
python3 -m http.server 8080
# open http://localhost:8080
```

> **Note:** Must be served over HTTP (not `file://`) due to WASM/ES module restrictions.

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
