# build.ps1 — compile Zeeslagje to WebAssembly on Windows (PowerShell)

# Check for wasm-pack
if (-not (Get-Command wasm-pack -ErrorAction SilentlyContinue)) {
    Write-Host "wasm-pack not found." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Install it with one of these methods:"
    Write-Host "  1. Download the installer from: https://rustwasm.github.io/wasm-pack/installer/"
    Write-Host "  2. Or via cargo:  cargo install wasm-pack"
    Write-Host ""
    Write-Host "After installing, re-run this script."
    Read-Host "Press Enter to exit"
    exit 1
}

# Check for Rust/cargo
if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
    Write-Host "cargo not found. Install Rust from https://rustup.rs" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "Building WASM (release)..." -ForegroundColor Cyan
wasm-pack build --target web --out-dir pkg --release

if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed!" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "Done! To run the game:" -ForegroundColor Green
Write-Host "  python -m http.server 8080"
Write-Host "  Then open http://localhost:8080 in your browser"
Write-Host ""
Read-Host "Press Enter to exit"
