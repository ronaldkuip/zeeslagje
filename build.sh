#!/usr/bin/env bash
# build.sh — compile Zeeslagje to WebAssembly
set -euo pipefail

# Check for wasm-pack
if ! command -v wasm-pack &>/dev/null; then
  echo "wasm-pack not found. Installing..."
  curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
fi

echo "Building WASM (release)..."
wasm-pack build --target web --out-dir pkg --release

echo ""
echo "Done! To run the game:"
echo "  python3 -m http.server 8080"
echo "  open http://localhost:8080"
