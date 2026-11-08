"""Access control for the web UI.

Loopback is not automatically safe: any web page open in a browser on the
same machine can send requests to 127.0.0.1 (CSRF), and a hostile DNS name
can be re-pointed at 127.0.0.1 after the page loads (DNS rebinding). So:

  - Host must name localhost / 127.0.0.1 / [::1] (or web.allowed_hosts, or
    the explicit bind address) *and* the port actually being served -- a
    rebinding attack arrives with the attacker's host name;
  - Origin, when present, must be that same origin, on every state-changing
    request and on the WebSocket upgrade (browsers always send it there;
    non-browser clients such as curl/Companion send none and are covered by
    the token check instead);
  - every API call needs either the session cookie or a token. The session
    cookie is obtained by presenting this run's random startup token (logged,
    and stored 0600 in <var>/web-token.json for `narrator --open-ui`); it is
    HttpOnly + SameSite=Strict. A `Authorization: Bearer <startup token>`
    header grants the same full access. Per-integration tokens
    (web.api_tokens) only reach the control API.
"""
from __future__ import annotations

import hmac
import json
import os
import secrets
from pathlib import Path
from urllib.parse import urlsplit

from narrator.paths import var_dir

SESSION_COOKIE = "narrator_session"
TOKEN_FILENAME = "web-token.json"
_LOOPBACK_NAMES = frozenset({"localhost", "127.0.0.1", "::1"})


def default_token_path() -> Path:
    return var_dir() / TOKEN_FILENAME


def new_secret() -> str:
    return secrets.token_urlsafe(32)


def write_token_file(path: Path, token: str, url: str) -> None:
    """Write {"token", "url"} readable by the owner only. Created 0600 from
    the start (no window where another user could read it), then chmod'ed in
    case the file already existed with wider permissions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    if tmp.exists():
        tmp.unlink()
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump({"token": token, "url": url, "pid": os.getpid()}, f)
    os.chmod(tmp, 0o600)
    tmp.replace(path)


def read_token_file(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def remove_token_file(path: Path, token: str) -> None:
    """Remove the file if it still holds our token (a newer instance may
    have replaced it)."""
    try:
        if read_token_file(path).get("token") == token:
            path.unlink()
    except (OSError, ValueError):
        pass


def tokens_equal(given: str | None, expected: str | None) -> bool:
    if not given or not expected:
        return False
    return hmac.compare_digest(given.encode("utf-8"), expected.encode("utf-8"))


def split_host(host_header: str) -> tuple[str, int | None]:
    """'127.0.0.1:8765' -> ('127.0.0.1', 8765); '[::1]:80' -> ('::1', 80)."""
    host = host_header.strip().lower()
    if host.startswith("["):
        end = host.find("]")
        if end < 0:
            return host, None
        name, rest = host[1:end], host[end + 1 :]
    else:
        name, _, port_text = host.partition(":")
        rest = f":{port_text}" if port_text else ""
    if rest.startswith(":") and rest[1:].isdigit():
        return name, int(rest[1:])
    if rest:
        return name, -1  # garbage after the name
    return name, None


def allowed_host_names(bind: str, extra: list[str]) -> frozenset[str]:
    names = set(_LOOPBACK_NAMES)
    names.update(h.strip().lower() for h in extra if h.strip())
    if bind not in ("0.0.0.0", "::", ""):
        names.add(bind.strip().lower().strip("[]"))
    return frozenset(names)


def host_allowed(host_header: str | None, allowed: frozenset[str], port: int) -> bool:
    if not host_header:
        return False
    name, given_port = split_host(host_header)
    if name not in allowed:
        return False
    if given_port is None:
        return port == 80
    return given_port == port


def origin_allowed(origin: str | None, allowed: frozenset[str], port: int) -> bool:
    """None (no Origin header: not a browser) passes; "null" and anything
    cross-origin does not."""
    if origin is None:
        return True
    parts = urlsplit(origin)
    if parts.scheme != "http" or not parts.hostname:
        return False
    if parts.hostname.lower() not in allowed:
        return False
    return (parts.port or 80) == port
