#!/usr/bin/env python3
"""Local dev server for Zeeslagje: serves the static files (same as
`python -m http.server`) and additionally accepts POST requests to /log,
appending the request body as one line to simpleresult.txt. A plain
static server can't write to disk on the browser's behalf, so index.html
posts here to record when a game ends (date, time, turns played).

Usage: python3 serve.py [port]   (defaults to 8080)
"""
import http.server
import socketserver
import sys
from pathlib import Path

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
LOG_FILE = Path(__file__).resolve().parent / "simpleresult.txt"


class Handler(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/log":
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", 0))
        line = self.rfile.read(length).decode("utf-8", errors="replace").strip()
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"ok")


if __name__ == "__main__":
    with socketserver.ThreadingTCPServer(("", PORT), Handler) as httpd:
        print(f"Serving {Path(__file__).resolve().parent} at http://localhost:{PORT}")
        print(f"POST /log appends to {LOG_FILE}")
        httpd.serve_forever()
