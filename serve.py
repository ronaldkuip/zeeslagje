#!/usr/bin/env python3
"""Local dev server for Zeeslagje: serves the static files (same as
`python -m http.server`) and additionally accepts POST requests to /log,
appending the request body as one line to simpleresult.txt. A plain
static server can't write to disk on the browser's behalf, so index.html
posts here to record when a game ends (date, time, turns played).

Also handles saving/loading boards for later study, scoped per browser
session (see "Sessions" below):
  POST /save_board    body: {"board": <BoardLayout>, "resolved": bool, "odds": {...}|null}
                      assigns the next free integer id (unique across
                      this session's own boards and the shared legacy
                      pool), stores it under this session in
                      saved_boards.json, responds {"id": N}.
  GET  /saved_boards  returns this session's boards merged with the
                      shared legacy pool (or {} if none yet).
  POST /delete_boards body: {"ids": [1, 2, 3]}
                      removes those ids from THIS session's own boards
                      (missing ids, or ids only in the shared pool, are
                      silently ignored), responds
                      {"deleted": [...ids actually removed...]}.

Sessions: identified by a "session_id" cookie, set automatically on a
browser's first request (any request — a plain page load included) and
echoed back by the browser on every request after that, no client-side
code required. Two browser sessions (tabs, machines, whatever) never see
each other's saved boards. saved_boards.json predates this and used to
store all boards flat at the top level with no session scoping at all —
on first load here, that old shape is detected and migrated in place
under a reserved "_shared" bucket, so every session keeps seeing those
boards (merged with its own) instead of them silently disappearing.

Auth: this whole server sits behind one shared HTTP Basic Auth password
(not per-user — it exists only to keep the server from being wide open if
this machine's port ends up reachable from outside your own network).
Covers EVERY route, including plain static file serving — otherwise
anyone reaching the port could read this whole project (serve.py, .git
history, saved_boards.json, ...) over plain GET with no auth at all. The
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
import http.cookies
import http.server
import json
import secrets
import socketserver
import sys
import threading
import time
import uuid
from pathlib import Path

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
HOST = sys.argv[2] if len(sys.argv) > 2 else "127.0.0.1"
LOG_FILE = Path(__file__).resolve().parent / "simpleresult.txt"
SAVED_BOARDS_FILE = Path(__file__).resolve().parent / "saved_boards.json"
AUTH_FILE = Path(__file__).resolve().parent / ".auth_secret"

SESSION_COOKIE_NAME = "session_id"
SESSION_COOKIE_MAX_AGE = 31536000  # 1 year — this is a personal dev tool, not a security boundary
SHARED_KEY = "_shared"  # reserved bucket name; a real session_id is a uuid4 hex string, never this

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

# Guards _failed_logins — same reasoning as _boards_lock: multiple
# connections hitting auth concurrently is exactly the scenario this is
# meant to survive.
_auth_lock = threading.Lock()
_failed_logins = {}  # ip -> [timestamps of recent failures, within AUTH_LOCKOUT_SECONDS]

# Guards read-modify-write access to SAVED_BOARDS_FILE — ThreadingTCPServer
# handles each connection on its own thread, and multiple-sessions-at-once
# is the entire point of this file, so concurrent /save_board or
# /delete_boards calls are a real possibility now, not just a theoretical one.
_boards_lock = threading.Lock()


def load_all_boards():
    """The full on-disk structure: {"_shared": {...}, "<session_id>": {...}, ...}."""
    if not SAVED_BOARDS_FILE.exists():
        return {}
    try:
        with SAVED_BOARDS_FILE.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}

    # One-time migration: the pre-session format stored boards flat at the
    # top level ({"1": {"board": ...}, ...}). Detect that shape (a value
    # that IS a board entry, i.e. has a "board" key) and nest it all under
    # SHARED_KEY instead of a real session id, so every session keeps
    # seeing those boards (see session_view) rather than losing them.
    if any(isinstance(v, dict) and "board" in v for v in raw.values()):
        return {SHARED_KEY: raw}
    return raw


def save_all_boards(all_boards):
    with SAVED_BOARDS_FILE.open("w", encoding="utf-8") as f:
        json.dump(all_boards, f)


def session_view(all_boards, session_id):
    """Boards visible to this session: the shared legacy pool merged with
    whatever this session has itself saved. New saves only ever go into
    the session's own bucket — sessions never see each other's boards."""
    merged = dict(all_boards.get(SHARED_KEY, {}))
    merged.update(all_boards.get(session_id, {}))
    return merged


class Handler(http.server.SimpleHTTPRequestHandler):
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
        before _ensure_session or any routing, so nothing (not even plain
        static file serving) is reachable without it. Throttles repeated
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

    def _ensure_session(self):
        """Reads the session_id cookie, or mints a new one. Runs once per
        request; end_headers() below sends the Set-Cookie if it's new —
        covers every response path, including plain static file serving,
        so even a first-ever page load gets a session established."""
        cookie_header = self.headers.get("Cookie")
        if cookie_header:
            jar = http.cookies.SimpleCookie()
            try:
                jar.load(cookie_header)
            except http.cookies.CookieError:
                jar = http.cookies.SimpleCookie()
            if SESSION_COOKIE_NAME in jar and jar[SESSION_COOKIE_NAME].value != SHARED_KEY:
                self.session_id = jar[SESSION_COOKIE_NAME].value
                self._new_session = False
                return
        self.session_id = uuid.uuid4().hex
        self._new_session = True

    def end_headers(self):
        if getattr(self, "_new_session", False):
            self.send_header(
                "Set-Cookie",
                f"{SESSION_COOKIE_NAME}={self.session_id}; Path=/; Max-Age={SESSION_COOKIE_MAX_AGE}; SameSite=Lax",
            )
            self._new_session = False  # this request's headers are only ever finalized once
        super().end_headers()

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
        self._ensure_session()
        super().do_HEAD()

    def do_GET(self):
        if not self._require_auth():
            return
        self._ensure_session()
        if self.path == "/saved_boards":
            self._send_json(200, session_view(load_all_boards(), self.session_id))
            return
        super().do_GET()

    def do_POST(self):
        if not self._require_auth():
            return
        self._ensure_session()
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
                self._send_json(400, {"error": "invalid JSON body"})
                return

            with _boards_lock:
                all_boards = load_all_boards()
                visible_ids = session_view(all_boards, self.session_id)
                next_id = max((int(k) for k in visible_ids), default=0) + 1
                all_boards.setdefault(self.session_id, {})[str(next_id)] = entry
                save_all_boards(all_boards)

            self._send_json(200, {"id": next_id})
            return

        if self.path == "/delete_boards":
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length).decode("utf-8", errors="replace")
            try:
                payload = json.loads(raw)
                ids = [str(int(i)) for i in payload["ids"]]
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                self._send_json(400, {"error": "invalid JSON body"})
                return

            with _boards_lock:
                all_boards = load_all_boards()
                own = all_boards.get(self.session_id, {})
                deleted = [int(i) for i in ids if i in own]
                for i in ids:
                    own.pop(i, None)
                save_all_boards(all_boards)

            self._send_json(200, {"deleted": deleted})
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
        print(f"POST /save_board and GET /saved_boards use {SAVED_BOARDS_FILE}, scoped per session_id cookie")
        httpd.serve_forever()
