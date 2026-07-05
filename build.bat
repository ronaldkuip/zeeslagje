@echo off
REM build.bat — compile Zeeslagje to WebAssembly on Windows

REM Check for wasm-pack
where wasm-pack >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo wasm-pack not found. Installing...
    curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf -o wasm-pack-init.sh
    echo.
    echo ERROR: The wasm-pack installer requires a Unix shell (bash).
    echo Please install wasm-pack manually:
    echo   1. Go to https://rustwasm.github.io/wasm-pack/installer/
    echo   2. Download and run the Windows installer (.exe^)
    echo   3. Re-run this script after installation
    del wasm-pack-init.sh
    pause
    exit /b 1
)

echo Building WASM (release)...
wasm-pack build --target web --out-dir pkg --release

IF %ERRORLEVEL% NEQ 0 (
    echo.
    echo Build failed. Make sure Rust is installed: https://rustup.rs
    pause
    exit /b 1
)

echo.
echo Done! To run the game:
echo   python -m http.server 8080
echo   Then open http://localhost:8080 in your browser
echo.
pause
