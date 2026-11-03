#!/usr/bin/env python3
"""Local dev server for Zeeslagje: serves the static files (same as
`python -m http.server`) and additionally accepts POST requests to /log,
appending the request body as one line to simpleresult.txt. A plain
static server can't write to disk on the browser's behalf, so index.html
posts here to record when a game ends (date, time, turns played).

Also handles POST /save_fast_solve body: {"board": <BoardLayout>, "salvos": N}
which appends one line to fast_solves.jsonl (gitignored) — a plain,
lock-guarded append (see _fast_solve_lock) for the Autorun Batch feature
that harvests every game solved in exactly a fixed number of salvos.

Auth: this whole server sits behind one shared HTTP Basic Auth password
(not per-user — it exists only to keep the server from being wide open if
this machine's port ends up reachable from outside your own network).
Covers EVERY route, including plain static file serving — otherwise
anyone reaching the port could read this whole project (serve.py, .git
history, ...) over plain GET with no auth at all. The
password is generated once on first run and cached in .auth_secret
(gitignored) so it survives restarts; printed to the console every time
this starts. Browsers cache Basic Auth credentials per-origin after the
first prompt, so this needs no client-side code: index.html/player.html's
existing fetch() calls pick the cached credentials up automatically once
you've logged in once via any page load.

Repeated failed logins from the same IP are throttled — after
AUTH_MAX_FAILURES wrong attempts within AUTH_LOCKOUT_SECONDS, that IP gets
429s (with Retry-After) instead of another password check, and every
failure is logged to stderr with its source IP.

Binds to 127.0.0.1 by default, not every interface — the intended
deployment is behind a reverse proxy (Caddy/nginx) on the same machine
handling TLS, with only the proxy actually exposed to the internet (see
deploy/Caddyfile and deploy/zeeslagje.service). Override with a second
argument if you specifically need it reachable from elsewhere directly.

Usage: python3 serve.py [port] [host]   (defaults to 8080, 127.0.0.1)
"""
import base64
import http.server
import json
import secrets
import socketserver
import sys
import threading
import time
from pathlib import Path

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
HOST = sys.argv[2] if len(sys.argv) > 2 else "127.0.0.1"
LOG_FILE = Path(__file__).resolve().parent / "simpleresult.txt"
FAST_SOLVE_FILE = Path(__file__).resolve().parent / "fast_solves.jsonl"
AUTH_FILE = Path(__file__).resolve().parent / ".auth_secret"

AUTH_USERNAME = "zeeslagje"  # not meaningful on its own — only the password is actually checked

AUTH_MAX_FAILURES = 5
AUTH_LOCKOUT_SECONDS = 300  # also doubles as the sliding window failures are counted within


def load_or_create_auth_password():
    if AUTH_FILE.exists():
        existing = AUTH_FILE.read_text(encoding="utf-8").strip()
        if existing:
            return existing
    password = secrets.token_urlsafe(16)
    AUTH_FILE.write_text(password, encoding="utf-8")
    return password


AUTH_PASSWORD = load_or_create_auth_password()

# Guards _failed_logins — same reasoning as _fast_solve_lock: multiple
# connections hitting auth concurrently is exactly the scenario this is
# meant to survive.
_auth_lock = threading.Lock()
_failed_logins = {}  # ip -> [timestamps of recent failures, within AUTH_LOCKOUT_SECONDS]

# Guards append access to FAST_SOLVE_FILE — ThreadingTCPServer handles
# each connection on its own thread, so concurrent /save_fast_solve calls
# across sessions are a real possibility, not just a theoretical one.
_fast_solve_lock = threading.Lock()


class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Without this, browsers cache index.html/player.html/pkg/*
        # heuristically (for a fraction of the file's age), so after a
        # deploy, navigating between pages kept serving the old version.
        # no-cache still allows caching but revalidates every time — a
        # cheap 304 via Last-Modified when nothing changed.
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def _is_authenticated(self):
        header = self.headers.get("Authorization")
        if not header or not header.startswith("Basic "):
            return False
        try:
            decoded = base64.b64decode(header[6:].strip()).decode("utf-8")
        except (ValueError, UnicodeDecodeError):
            return False
        _, _, password = decoded.partition(":")
        # constant-time compare — this is the one check standing between
        # the whole project and anyone who can reach the port
        return secrets.compare_digest(password, AUTH_PASSWORD)

    def _require_auth(self):
        """Challenges for HTTP Basic Auth if missing/wrong. Called first,
        before any routing, so nothing (not even plain static file
        serving) is reachable without it. Throttles repeated
        failures per source IP — see AUTH_MAX_FAILURES/AUTH_LOCKOUT_SECONDS
        — and logs every failed attempt to stderr with that IP."""
        ip = self.client_address[0]
        now = time.time()

        with _auth_lock:
            recent = [t for t in _failed_logins.get(ip, []) if now - t < AUTH_LOCKOUT_SECONDS]
            _failed_logins[ip] = recent
            if len(recent) >= AUTH_MAX_FAILURES:
                retry_after = int(AUTH_LOCKOUT_SECONDS - (now - recent[0])) + 1

        if len(recent) >= AUTH_MAX_FAILURES:
            body = b"Too many failed login attempts. Try again later."
            self.send_response(429)
            self.send_header("Retry-After", str(retry_after))
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return False

        if self._is_authenticated():
            with _auth_lock:
                _failed_logins.pop(ip, None)
            return True

        with _auth_lock:
            _failed_logins.setdefault(ip, []).append(now)
            count = len(_failed_logins[ip])
        print(f"[auth] failed login from {ip} ({count} recent failure{'s' if count != 1 else ''})", file=sys.stderr)

        body = b"Authentication required."
        self.send_response(401)
        self.send_header("WWW-Authenticate", 'Basic realm="Zeeslagje"')
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        return False

    def _send_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_HEAD(self):
        if not self._require_auth():
            return
        super().do_HEAD()

    def do_GET(self):
        if not self._require_auth():
            return
        super().do_GET()

    def do_POST(self):
        if not self._require_auth():
            return
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

        if self.path == "/save_fast_solve":
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length).decode("utf-8", errors="replace")
            try:
                entry = json.loads(raw)
            except json.JSONDecodeError:
                self._send_json(400, {"error": "invalid JSON body"})
                return

            # Plain append — appending one line stays O(1) per save
            # regardless of how large the file has already grown. Still
            # lock-guarded: concurrent appends across sessions are a real
            # possibility (see _fast_solve_lock's own reasoning), and an
            # unguarded interleaved write could corrupt a line.
            with _fast_solve_lock:
                with FAST_SOLVE_FILE.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(entry) + "\n")

            self._send_json(200, {"ok": True})
            return

        self.send_response(404)
        self.end_headers()


if __name__ == "__main__":
    with socketserver.ThreadingTCPServer((HOST, PORT), Handler) as httpd:
        print(f"Serving {Path(__file__).resolve().parent} at http://{HOST}:{PORT}")
        if HOST not in ("127.0.0.1", "localhost"):
            print(f"WARNING: bound to {HOST}, not just localhost — reachable beyond this machine directly.")
        print(f"Auth required — username: {AUTH_USERNAME}  password: {AUTH_PASSWORD}  (saved in {AUTH_FILE})")
        print(f"Failed logins: max {AUTH_MAX_FAILURES} per IP per {AUTH_LOCKOUT_SECONDS}s before a 429 lockout")
        print(f"POST /log appends to {LOG_FILE}")
        print(f"POST /save_fast_solve appends to {FAST_SOLVE_FILE}")
        httpd.serve_forever()
