#!/usr/bin/env python3
"""Local dev server for Zeeslagje: serves the static files (same as
`python -m http.server`) and additionally accepts POST requests to /log,
appending the request body as one line to simpleresult.txt. A plain
static server can't write to disk on the browser's behalf, so index.html
posts here to record when a game ends (date, time, turns played).

Also handles saving/loading boards for later study:
  POST /save_board   body: {"board": <BoardLayout>, "resolved": bool, "odds": {...}|null}
                      assigns the next free integer id, stores it in
                      saved_boards.json, responds {"id": N}.
  GET  /saved_boards  returns the full contents of saved_boards.json
                      (or {} if it doesn't exist yet).

Usage: python3 serve.py [port]   (defaults to 8080)
"""
import http.server
import json
import socketserver
import sys
from pathlib import Path

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
LOG_FILE = Path(__file__).resolve().parent / "simpleresult.txt"
SAVED_BOARDS_FILE = Path(__file__).resolve().parent / "saved_boards.json"


def load_saved_boards():
    if not SAVED_BOARDS_FILE.exists():
        return {}
    try:
        with SAVED_BOARDS_FILE.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/saved_boards":
            body = json.dumps(load_saved_boards()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def do_POST(self):
        if self.path == "/log":
            length = int(self.headers.get("Content-Length", 0))
            line = self.rfile.read(length).decode("utf-8", errors="replace").strip()
            with LOG_FILE.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"ok")
            return

        if self.path == "/save_board":
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length).decode("utf-8", errors="replace")
            try:
                entry = json.loads(raw)
            except json.JSONDecodeError:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b'{"error":"invalid JSON body"}')
                return

            boards = load_saved_boards()
            next_id = max((int(k) for k in boards), default=0) + 1
            boards[str(next_id)] = entry
            with SAVED_BOARDS_FILE.open("w", encoding="utf-8") as f:
                json.dump(boards, f)

            body = json.dumps({"id": next_id}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        self.send_response(404)
        self.end_headers()


if __name__ == "__main__":
    with socketserver.ThreadingTCPServer(("", PORT), Handler) as httpd:
        print(f"Serving {Path(__file__).resolve().parent} at http://localhost:{PORT}")
        print(f"POST /log appends to {LOG_FILE}")
        print(f"POST /save_board and GET /saved_boards use {SAVED_BOARDS_FILE}")
        httpd.serve_forever()
