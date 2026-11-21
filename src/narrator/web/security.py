"""Access control for the web UI.

Reachable from other computers by default (user decision 2026-11-16);
`narrator --local-only` listens on 127.0.0.1 only. So:

  - --local-only: Host must name localhost / 127.x / [::1] with the served
    port (DNS rebinding: a hostile name re-pointed at 127.0.0.1 arrives with
    the attacker's host name);
  - Origin, when present, must be the address the request was sent to (its
    Host), on every state-changing request and on the WebSocket upgrade
    (CSRF; browsers always send it there; non-browser clients such as
    curl/Companion send none and are covered by the login/token checks);
  - every API call needs the session cookie or an API token. The session
    cookie is obtained by logging in with the operator's password (scrypt
    hash in web-password.json next to config.yaml, mode 0600, set with
    `narrator --set-password` or from the web UI); it is HttpOnly +
    SameSite=Strict and lives in memory, so a restart logs everyone out.
    With no password set, only a browser on this machine (loopback client
    AND loopback Host, so a rebinding page cannot) gets in, without a login;
    other computers cannot log in until a password exists. Per-integration
    tokens (web.api_tokens) only reach the control API.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import ipaddress
import json
import os
import secrets
from pathlib import Path
from urllib.parse import urlsplit

from narrator.paths import var_dir

SESSION_COOKIE = "narrator_session"
UI_FILENAME = "web-ui.json"
# The retired per-run login token's file: removed at start if a stale one exists.
LEGACY_TOKEN_FILENAME = "web-token.json"
PASSWORD_FILENAME = "web-password.json"
MIN_PASSWORD_LENGTH = 8
# scrypt cost: ~16 MB and ~50 ms per check on the Pentium test box.
_SCRYPT = {"n": 2**14, "r": 8, "p": 1}
_LOOPBACK_NAMES = frozenset({"localhost", "127.0.0.1", "::1"})


def default_ui_file_path() -> Path:
    return var_dir() / UI_FILENAME


def legacy_token_path() -> Path:
    return var_dir() / LEGACY_TOKEN_FILENAME


def password_path_for(config_path: Path) -> Path:
    """The password hash lives beside config.yaml (never inside it: the web
    form posts the whole config, and the hash has no business in a browser)."""
    return Path(config_path).parent / PASSWORD_FILENAME


def new_secret() -> str:
    return secrets.token_urlsafe(32)


def _write_private_json(path: Path, data: dict) -> None:
    """Write JSON readable by the owner only. Created 0600 from the start (no
    window where another user could read it), then chmod'ed in case the file
    already existed with wider permissions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    if tmp.exists():
        tmp.unlink()
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(data, f)
    os.chmod(tmp, 0o600)
    tmp.replace(path)


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_ui_file(path: Path, url: str) -> None:
    """{"url", "pid"} of the running web UI, for `narrator --open-ui`."""
    _write_private_json(path, {"url": url, "pid": os.getpid()})


def read_ui_file(path: Path) -> dict:
    return _read_json(path)


def remove_ui_file(path: Path) -> None:
    """Remove the file if it is still ours (a newer instance may have replaced it)."""
    try:
        if _read_json(path).get("pid") == os.getpid():
            path.unlink()
    except (OSError, ValueError):
        pass


class PasswordError(ValueError):
    pass


def check_new_password(password: str) -> None:
    if len(password) < MIN_PASSWORD_LENGTH:
        raise PasswordError(f"the password needs at least {MIN_PASSWORD_LENGTH} characters")
    if password != password.strip():
        raise PasswordError("the password cannot start or end with a space")


def hash_password(password: str) -> dict:
    salt = os.urandom(16)
    key = hashlib.scrypt(password.encode("utf-8"), salt=salt, dklen=32, **_SCRYPT)
    return {
        "scheme": "scrypt", **_SCRYPT,
        "salt": base64.b64encode(salt).decode("ascii"),
        "hash": base64.b64encode(key).decode("ascii"),
    }


def verify_password(record: dict | None, password: str) -> bool:
    if not record or not password or record.get("scheme") != "scrypt":
        return False
    try:
        salt = base64.b64decode(record["salt"])
        expected = base64.b64decode(record["hash"])
        key = hashlib.scrypt(
            password.encode("utf-8"), salt=salt, dklen=len(expected),
            n=int(record["n"]), r=int(record["r"]), p=int(record["p"]),
        )
    except (KeyError, ValueError, TypeError):
        return False
    return hmac.compare_digest(key, expected)


def read_password_file(path: Path) -> dict | None:
    """The stored hash record, or None when no (readable) password is set."""
    try:
        record = _read_json(path)
    except (OSError, ValueError):
        return None
    return record if isinstance(record, dict) and record.get("hash") else None


def write_password_file(path: Path, password: str) -> None:
    check_new_password(password)
    _write_private_json(path, hash_password(password))


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


def is_loopback_host(host_header: str | None, port: int) -> bool:
    """The Host header names this machine's loopback (localhost / 127.x /
    [::1]) with the served port."""
    if not host_header:
        return False
    name, given_port = split_host(host_header)
    if name not in _LOOPBACK_NAMES and not name.startswith("127."):
        return False
    if given_port is None:
        return port == 80
    return given_port == port


def is_loopback_address(remote: str | None) -> bool:
    """The connecting client's IP is this machine."""
    if not remote:
        return False
    try:
        return ipaddress.ip_address(remote.split("%", 1)[0]).is_loopback
    except ValueError:
        return False


def origin_allowed(origin: str | None, host_header: str | None) -> bool:
    """CSRF guard: None (no Origin header: not a browser) passes; "null" and
    anything whose host:port differs from the address the request was sent
    to (its Host header) does not."""
    if origin is None:
        return True
    if not host_header:
        return False
    parts = urlsplit(origin)
    if parts.scheme != "http" or not parts.hostname:
        return False
    try:
        origin_port = parts.port or 80
    except ValueError:
        return False
    name, port = split_host(host_header)
    return parts.hostname.lower() == name and origin_port == (80 if port is None else port)
