"""Web admin UI + control API (narrator.web), against a fake controller:
no GPU, microphone, NDI or browser needed."""
from __future__ import annotations

import asyncio
import collections
import json
import os
import stat

import pytest
import yaml
from aiohttp.test_utils import TestClient, TestServer

from narrator.config.loader import load_config, save_config
from narrator.config.schema import RootConfig
from narrator.render.canvas import CaptionRenderer
from narrator.web import actions, osc
from narrator.web.security import (
    SESSION_COOKIE,
    is_loopback_address,
    is_loopback_host,
    origin_allowed,
    read_password_file,
    read_ui_file,
    split_host,
    verify_password,
    write_password_file,
    write_ui_file,
)
from narrator.web.server import WebServer

PASSWORD = "test-password-123"
API_TOKEN = "companion-token-abcdefghijklmnop"


class FakeController:
    """The controller surface the web layer uses."""

    def __init__(self, config_path):
        self.config_path = config_path
        self.config = load_config(config_path)
        self.renderer = CaptionRenderer(self.config.canvas, self.config.layout, self.config.style)
        self.caption_mode = "auto"
        self.captions_paused = False
        self.ui_events = collections.deque(maxlen=256)
        self.calls: list = []
        self.problems = [
            {
                "id": "asr.stopped", "severity": "error", "subsystem": "asr", "message": "ASR stopped",
                "fixes": [{"action": "asr.retry", "label": "Retry"}, {"action": "ui:asr", "label": "ASR settings"}],
            }
        ]

    def status_snapshot(self):
        return {
            "caption_mode": self.caption_mode,
            "captions_paused": self.captions_paused,
            "asr_active": True,
            "asr_backend": "fake",
            "asr_restarts": 0,
            "asr_crashes": 0,
            "asr_last_exit_code": None,
            "mic_level": 0.25,
            "on_air": list(self.renderer.visible_lines()),
            "uptime_s": 1.0,
            "problems": list(self.problems),
        }

    def apply_fix(self, action):
        # Same contract as AppController.apply_fix.
        if action not in ("asr.retry", "asr.use_cpu", "audio.retry", "ndi.retry", "health.recheck"):
            raise ValueError(f"unknown fix action: {action!r}")
        offered = {f["action"] for p in self.problems for f in p["fixes"]}
        if action != "health.recheck" and action not in offered:
            raise LookupError(f"fix {action!r} is not offered by any current problem")
        self.calls.append(("fix", action))
        if action == "asr.retry":
            self.problems = []

    def set_caption_mode(self, mode):
        self.calls.append(("mode", mode))
        self.caption_mode = mode

    def submit_manual_text(self, text):
        self.calls.append(("manual", text))
        self.renderer.append_final_text(text)

    def clear_captions(self):
        self.calls.append(("clear",))
        self.renderer.clear()

    def set_captions_paused(self, paused):
        self.calls.append(("pause", paused))
        self.captions_paused = paused

    def refresh_text_settings(self):
        self.calls.append(("refresh_text",))


@pytest.fixture
def config_path(tmp_path):
    config = RootConfig.model_validate(
        {"web": {"api_tokens": [{"name": "companion", "token": API_TOKEN}]}}
    )
    path = tmp_path / "config.yaml"
    save_config(config, path)
    return path


def run(coro):
    return asyncio.run(coro)


async def _client(controller, local_only: bool = False):
    password_path = controller.config_path.parent / "web-password.json"
    if not password_path.exists():
        write_password_file(password_path, PASSWORD)
    server = WebServer(controller, password_path=password_path, local_only=local_only,
                       ui_file_path=controller.config_path.parent / "web-ui.json")
    client = TestClient(TestServer(server.create_app(), host="127.0.0.1"))
    await client.start_server()
    return server, client


def _origin(client) -> str:
    return f"http://127.0.0.1:{client.port}"


async def _login(client, password: str = PASSWORD) -> None:
    resp = await client.post("/auth", data={"password": password}, headers={"Origin": _origin(client)},
                             allow_redirects=False)
    assert resp.status == 303
    assert resp.headers["Location"] == "/"
    cookie = resp.cookies[SESSION_COOKIE]
    assert cookie["httponly"]
    assert cookie["samesite"] == "Strict"


def with_client(test):
    """Run an async test body with a logged-out client on a fake controller."""

    def wrapper(config_path):
        async def body():
            controller = FakeController(config_path)
            server, client = await _client(controller)
            try:
                await test(controller, server, client)
            finally:
                await client.close()

        run(body())

    wrapper.__name__ = test.__name__
    return wrapper


# ---------------------------------------------------------------- auth


@with_client
async def test_api_requires_authentication(controller, server, client):
    for method, path in [("GET", "/api/state"), ("GET", "/api/config"), ("POST", "/api/clear"), ("GET", "/ws")]:
        resp = await client.request(method, path, headers={"Origin": _origin(client)})
        assert resp.status == 401, path
    assert controller.calls == []


@with_client
async def test_root_serves_login_page_until_logged_in(controller, server, client):
    resp = await client.get("/")
    assert 'name="password"' in await resp.text()
    await _login(client)
    resp = await client.get("/")
    assert "app.js" in await resp.text()
    assert "frame-ancestors 'none'" in resp.headers["Content-Security-Policy"]
    # no-referrer would make Chromium send "Origin: null" on the login form post
    assert resp.headers["Referrer-Policy"] == "same-origin"


@with_client
async def test_wrong_password_is_refused(controller, server, client):
    resp = await client.post("/auth", data={"password": "not-the-password"},
                             headers={"Origin": _origin(client)}, allow_redirects=False)
    assert resp.status == 303 and "failed" in resp.headers["Location"]
    assert SESSION_COOKIE not in resp.cookies
    assert (await client.get("/api/state")).status == 401


@with_client
async def test_login_form_post(controller, server, client):
    await _login(client)
    assert (await client.get("/api/config")).status == 200


@with_client
async def test_get_auth_and_token_query_no_longer_log_in(controller, server, client):
    resp = await client.get(f"/auth?token={PASSWORD}&password={PASSWORD}", allow_redirects=False)
    assert resp.status == 303 and SESSION_COOKIE not in resp.cookies
    assert (await client.get("/api/config", headers={"Authorization": f"Bearer {PASSWORD}"})).status == 401


@with_client
async def test_login_post_needs_same_origin(controller, server, client):
    resp = await client.post("/auth", data={"password": PASSWORD}, headers={"Origin": "http://evil.example"},
                             allow_redirects=False)
    assert resp.status == 403


@with_client
async def test_no_password_on_loopback_opens_without_login(controller, server, client):
    server.password_path.unlink()
    assert (await (await client.get("/api/login-info")).json()) == {"password_set": False, "login_required": False}
    resp = await client.get("/")
    assert "app.js" in await resp.text()
    assert (await client.get("/api/config")).status == 200
    # Origin/Host checks still apply.
    assert (await client.post("/api/clear", headers={"Origin": "http://evil.example"})).status == 403
    # An API token still gets only the control API.
    assert (await client.get("/api/config", headers={"Authorization": f"Bearer {API_TOKEN}"})).status == 403


@with_client
async def test_no_password_means_no_login_from_other_computers(controller, server, client):
    server.password_path.unlink()
    # A browser on another computer addresses this machine by its LAN IP;
    # a DNS-rebinding page in a local browser by its own name. Neither is let in.
    for host in (f"192.168.1.5:{client.port}", f"evil.example:{client.port}"):
        h = {"Host": host}
        assert (await (await client.get("/api/login-info", headers=h)).json()) == {
            "password_set": False, "login_required": True}
        assert (await client.get("/api/config", headers=h)).status == 401
    lan = f"192.168.1.5:{client.port}"
    resp = await client.post("/auth", data={"password": PASSWORD},
                             headers={"Host": lan, "Origin": f"http://{lan}"}, allow_redirects=False)
    assert resp.status == 303 and "nopassword" in resp.headers["Location"]
    assert SESSION_COOKIE not in resp.cookies


@with_client
async def test_first_password_set_from_the_open_page_keeps_this_browser_in(controller, server, client):
    server.password_path.unlink()
    resp = await client.post("/api/password", json={"new": "first-password"}, headers={"Origin": _origin(client)})
    assert resp.status == 200
    assert SESSION_COOKIE in resp.cookies
    assert (await client.get("/api/config")).status == 200  # this browser stays in
    other = TestClient(TestServer(server.create_app(), host="127.0.0.1"))
    await other.start_server()
    try:
        assert (await other.get("/api/config")).status == 401  # everyone else needs it now
        await _login(other, "first-password")
        assert (await other.get("/api/config")).status == 200
    finally:
        await other.close()


@with_client
async def test_change_password(controller, server, client):
    await _login(client)
    other = TestClient(TestServer(server.create_app(), host="127.0.0.1"))
    await other.start_server()
    try:
        await _login(other)
        origin = {"Origin": _origin(client)}
        resp = await client.post("/api/password", json={"current": "wrong-one!", "new": "brand-new-pass"}, headers=origin)
        assert resp.status == 403
        resp = await client.post("/api/password", json={"current": PASSWORD, "new": "short"}, headers=origin)
        assert resp.status == 400
        resp = await client.post("/api/password", json={"current": PASSWORD, "new": "brand-new-pass"}, headers=origin)
        assert resp.status == 200
        assert verify_password(read_password_file(server.password_path), "brand-new-pass")
        # This session stays; every other one ends.
        assert (await client.get("/api/config")).status == 200
        assert (await other.get("/api/config")).status == 401
    finally:
        await other.close()


@with_client
async def test_api_token_cannot_change_the_password(controller, server, client):
    resp = await client.post("/api/password", json={"current": PASSWORD, "new": "brand-new-pass"},
                             headers={"Authorization": f"Bearer {API_TOKEN}"})
    assert resp.status == 403


@with_client
async def test_api_token_reaches_only_the_control_api(controller, server, client):
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    resp = await client.post("/api/clear", headers=headers)
    assert resp.status == 200
    assert controller.calls == [("clear",)]
    assert (await client.get("/api/state", headers={"X-Narrator-Token": API_TOKEN})).status == 200
    assert (await client.get("/api/config", headers=headers)).status == 403
    assert (await client.post("/api/config", headers=headers, json={})).status == 403


@with_client
async def test_revoked_api_token_stops_working(controller, server, client):
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    assert (await client.post("/api/clear", headers=headers)).status == 200
    controller.config = controller.config.model_copy(
        update={"web": controller.config.web.model_copy(update={"api_tokens": []})}
    )
    assert (await client.post("/api/clear", headers=headers)).status == 401


@with_client
async def test_logout_ends_the_session(controller, server, client):
    await _login(client)
    assert (await client.post("/api/logout", headers={"Origin": _origin(client)})).status == 200
    assert (await client.get("/api/config")).status == 401


# --------------------------------------------------- Host / Origin checks


def test_local_only_refuses_foreign_host_header(config_path):
    """--local-only: DNS rebinding (evil.example resolved to 127.0.0.1)
    carries its own Host, and a LAN address is not this computer only."""

    async def body():
        controller = FakeController(config_path)
        server, client = await _client(controller, local_only=True)
        try:
            await _login(client)
            for host in ("evil.example", f"evil.example:{client.port}", "127.0.0.1:1",
                         "localhost.evil.example", f"192.168.1.5:{client.port}"):
                resp = await client.get("/api/state", headers={"Host": host})
                assert resp.status == 421, host
            ok = await client.get("/api/state", headers={"Host": f"localhost:{client.port}"})
            assert ok.status == 200
        finally:
            await client.close()

    run(body())


@with_client
async def test_other_computers_reach_it_by_default_with_a_login(controller, server, client):
    lan = f"192.168.1.5:{client.port}"
    h = {"Host": lan, "Origin": f"http://{lan}"}
    assert (await client.get("/api/config", headers={"Host": lan})).status == 401
    resp = await client.post("/auth", data={"password": PASSWORD}, headers=h, allow_redirects=False)
    assert resp.status == 303 and SESSION_COOKIE in resp.cookies
    assert (await client.get("/api/config", headers={"Host": lan})).status == 200
    assert (await client.post("/api/clear", headers=h)).status == 200
    # CSRF: an Origin other than the address used is still refused.
    assert (await client.post("/api/clear", headers={"Host": lan, "Origin": f"http://localhost:{client.port}"})).status == 403


@with_client
async def test_cross_origin_state_change_is_refused(controller, server, client):
    await _login(client)
    for origin in ("http://evil.example", "null", f"https://127.0.0.1:{client.port}", "http://127.0.0.1:1"):
        resp = await client.post("/api/clear", headers={"Origin": origin})
        assert resp.status == 403, origin
    assert controller.calls == []
    resp = await client.post("/api/clear", headers={"Origin": _origin(client)})
    assert resp.status == 200
    assert controller.calls == [("clear",)]


@with_client
async def test_problems_in_state_and_fix_endpoint(controller, server, client):
    await _login(client)
    h = {"Origin": _origin(client)}
    state = await (await client.get("/api/state")).json()
    assert state["problems"][0]["id"] == "asr.stopped"

    # not offered right now -> 409; unknown -> 400; client-side ui: ids are not server fixes
    assert (await client.post("/api/fix", json={"action": "ndi.retry"}, headers=h)).status == 409
    assert (await client.post("/api/fix", json={"action": "format.disk"}, headers=h)).status == 400
    assert (await client.post("/api/fix", json={"action": "ui:asr"}, headers=h)).status == 400
    assert (await client.post("/api/fix", json={}, headers=h)).status == 400

    resp = await client.post("/api/fix", json={"action": "asr.retry"}, headers=h)
    assert resp.status == 200
    body = await resp.json()
    assert body["result"] == "asr.retry"
    assert body["state"]["problems"] == []
    assert controller.calls == [("fix", "asr.retry")]


@with_client
async def test_fix_endpoint_needs_admin_and_same_origin(controller, server, client):
    # API tokens reach the control API only; a fix can write config.yaml.
    resp = await client.post(
        "/api/fix", json={"action": "asr.retry"}, headers={"Authorization": f"Bearer {API_TOKEN}"}
    )
    assert resp.status == 403
    await _login(client)
    resp = await client.post("/api/fix", json={"action": "asr.retry"}, headers={"Origin": "http://evil.example"})
    assert resp.status == 403
    assert controller.calls == []


@with_client
async def test_cross_origin_websocket_is_refused(controller, server, client):
    await _login(client)
    resp = await client.get("/ws", headers={"Origin": "http://evil.example"})
    assert resp.status == 403


def test_host_and_origin_helpers():
    assert split_host("[::1]:8765") == ("::1", 8765)
    assert is_loopback_host("[::1]:8765", 8765)
    assert is_loopback_host("127.0.0.1:8765", 8765)
    assert not is_loopback_host("192.168.1.5:8765", 8765)
    assert not is_loopback_host(None, 8765)
    assert not is_loopback_host("127.0.0.1:8765x", 8765)
    assert is_loopback_address("127.0.0.1") and is_loopback_address("::1")
    assert not is_loopback_address("192.168.1.5") and not is_loopback_address(None)
    assert origin_allowed(None, "localhost:8765")
    assert origin_allowed("http://localhost:8765", "localhost:8765")
    assert origin_allowed("http://192.168.1.5:8765", "192.168.1.5:8765")
    assert not origin_allowed("http://localhost:8766", "localhost:8765")
    assert not origin_allowed("http://localhost:8765", "192.168.1.5:8765")
    assert not origin_allowed("null", "localhost:8765")


def test_password_file_is_private_and_hashed(tmp_path):
    path = tmp_path / "web-password.json"
    write_password_file(path, "correct horse")
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    assert "correct horse" not in path.read_text()
    record = read_password_file(path)
    assert verify_password(record, "correct horse")
    assert not verify_password(record, "correct horsE")
    assert not verify_password(None, "correct horse")


def test_password_rules(tmp_path):
    from narrator.web.security import PasswordError

    for bad in ["short", " leading-space", "trailing-space "]:
        with pytest.raises(PasswordError):
            write_password_file(tmp_path / "pw.json", bad)
    assert read_password_file(tmp_path / "pw.json") is None


def test_ui_file_is_private(tmp_path):
    path = tmp_path / "var" / "web-ui.json"
    write_ui_file(path, "http://127.0.0.1:8765")
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    assert read_ui_file(path)["url"] == "http://127.0.0.1:8765"


# ------------------------------------------------------------ control API


@with_client
async def test_each_control_action(controller, server, client):
    await _login(client)
    h = {"Origin": _origin(client)}

    resp = await client.post("/api/mode", json={"mode": "manual"}, headers=h)
    assert (await resp.json())["state"]["caption_mode"] == "manual"
    resp = await client.post("/api/mode", json={"mode": "toggle"}, headers=h)
    assert (await resp.json())["result"] == "auto"
    assert (await client.post("/api/mode", json={"mode": "sideways"}, headers=h)).status == 400

    resp = await client.post("/api/manual-text", json={"text": "Dzień dobry"}, headers=h)
    assert " ".join((await resp.json())["state"]["on_air"]) == "Dzień dobry"

    resp = await client.post("/api/pause", headers=h)  # no body: toggle
    assert (await resp.json())["state"]["captions_paused"] is True
    resp = await client.post("/api/pause", json={"paused": False}, headers=h)
    assert (await resp.json())["state"]["captions_paused"] is False
    resp = await client.post("/api/pause?paused=on", headers=h)  # query form (Companion-friendly)
    assert (await resp.json())["state"]["captions_paused"] is True

    resp = await client.post("/api/clear", headers=h)
    assert (await resp.json())["state"]["on_air"] == []

    # Session recording was removed: its endpoints are gone.
    assert (await client.post("/api/recording", json={"recording": "toggle"}, headers=h)).status in (404, 405)
    assert (await client.get("/api/recordings")).status in (404, 405)

    state = await (await client.get("/api/state")).json()
    assert state["shortcuts"]["clear"] == "Escape"
    assert state["asr_backend"] == "fake"
    assert controller.calls == [
        ("mode", "manual"), ("mode", "auto"), ("manual", "Dzień dobry"),
        ("pause", True), ("pause", False), ("pause", True), ("clear",),
    ]


def test_parse_switch():
    assert actions.parse_switch(None, False) is True
    assert actions.parse_switch("toggle", True) is False
    assert actions.parse_switch("off", True) is False
    assert actions.parse_switch(1, False) is True
    with pytest.raises(actions.ActionError):
        actions.parse_switch("maybe", False)


# -------------------------------------------------------------- config


@with_client
async def test_config_save_validation_error_round_trip(controller, server, client):
    await _login(client)
    h = {"Origin": _origin(client)}
    data = (await (await client.get("/api/config")).json())["config"]
    before = controller.config_path.read_text()

    data["asr"]["step_ms"] = -5
    data["layout"]["margin_top_pct"] = 60
    data["layout"]["margin_bottom_pct"] = 50
    data["web"]["shortcuts"]["clear"] = "Ctrl+N"
    resp = await client.post("/api/config", json={"config": data}, headers=h)
    assert resp.status == 422
    errors = {e["loc"]: e["msg"] for e in (await resp.json())["errors"]}
    assert "asr.step_ms" in errors
    assert "web.shortcuts.clear" in errors and "reserved" in errors["web.shortcuts.clear"]
    assert controller.config_path.read_text() == before  # nothing written

    data = (await (await client.get("/api/config")).json())["config"]
    data["layout"]["margin_top_pct"] = 60
    data["layout"]["margin_bottom_pct"] = 50
    resp = await client.post("/api/config", json={"config": data}, headers=h)
    errors = {e["loc"]: e["msg"] for e in (await resp.json())["errors"]}
    assert "layout" in errors and "< 100" in errors["layout"]  # a model_validator error


@with_client
async def test_config_save_writes_yaml_via_loader(controller, server, client):
    await _login(client)
    h = {"Origin": _origin(client)}
    data = (await (await client.get("/api/config")).json())["config"]
    data["ndi"]["source_name"] = "Kaplica"
    data["asr"]["text_filters"] = [{"pattern": r"\bnapisy\b.*", "replacement": ""}]
    resp = await client.post("/api/config", json={"config": data}, headers=h)
    body = await resp.json()
    assert resp.status == 200 and body["changed"] is True
    saved = load_config(controller.config_path)
    assert saved.ndi.source_name == "Kaplica"
    assert saved.asr.text_filters[0].pattern == r"\bnapisy\b.*"
    resp = await client.post("/api/config", json={"config": data}, headers=h)
    assert (await resp.json())["changed"] is False


@with_client
async def test_invalid_file_falls_back_to_live_config(controller, server, client):
    await _login(client)
    controller.config_path.write_text("asr:\n  step_ms: -1\n")
    body = await (await client.get("/api/config")).json()
    assert body["file_error"]
    assert body["config"]["asr"]["step_ms"] == controller.config.asr.step_ms


@with_client
async def test_schema_and_choices(controller, server, client):
    await _login(client)
    schema = await (await client.get("/api/schema")).json()
    assert {"audio", "asr", "web", "captions"} <= set(schema["properties"])
    assert not {"transcript", "recording"} & set(schema["properties"])
    choices = await (await client.get("/api/choices")).json()
    assert "large-v3" in choices["models"]
    assert ["pl", "Polish (pl)"] in choices["languages"]
    assert any(f.endswith("NotoSans-Regular.ttf") for f in choices["fonts"])


@with_client
async def test_vocabulary_list_round_trip(controller, server, client):
    await _login(client)
    h = {"Origin": _origin(client)}
    assert (await (await client.get("/api/vocabulary/event")).json())["entries"] == []
    resp = await client.put("/api/vocabulary/event", json={"entries": ["Kowalski", " ", "Gdańsk"]}, headers=h)
    assert resp.status == 200
    assert (await (await client.get("/api/vocabulary/event")).json())["entries"] == ["Kowalski", "Gdańsk"]
    assert (await client.get("/api/vocabulary/..%2Fescape")).status in (400, 404)


# ------------------------------------------------------------- websocket


@with_client
async def test_websocket_streams_state_captions_and_preview(controller, server, client):
    await _login(client)
    controller.ui_events.append(("final", "Witamy", True))
    controller.renderer.append_final_text("Witamy")
    ws = await client.ws_connect("/ws", headers={"Origin": _origin(client)})
    got_state = got_caption = got_png = False
    for _ in range(60):
        msg = await asyncio.wait_for(ws.receive(), timeout=5)
        if msg.type.name == "BINARY":
            assert msg.data.startswith(b"\x89PNG")
            got_png = True
        else:
            data = json.loads(msg.data)
            if data["type"] == "state":
                got_state = True
            elif data["type"] == "captions":
                assert data["events"][0]["text"] == "Witamy"
                got_caption = True
        if got_state and got_caption and got_png:
            break
    # A draft with an invalid layout is answered with errors, not applied.
    await ws.send_str(json.dumps({"type": "draft", "display": {"layout": {"max_lines": 0}}}))
    for _ in range(60):
        data = (await asyncio.wait_for(ws.receive(), timeout=5)).data
        if isinstance(data, str) and json.loads(data)["type"] == "draft_error":
            assert json.loads(data)["errors"][0]["loc"] == "max_lines"
            break
    await ws.close()
    assert got_state and got_caption and got_png


# -------------------------------------------------------------- shortcuts + migration


@pytest.mark.parametrize("combo", ["Ctrl+N", "ctrl+t", "Control+W", "Ctrl+Shift+T", "Alt+F4", "F11", "Ctrl+1", "Meta+Q"])
def test_reserved_shortcuts_are_rejected(combo):
    with pytest.raises(ValueError, match="reserved"):
        RootConfig.model_validate({"web": {"shortcuts": {"clear": combo}}})


def test_shortcuts_are_canonicalized_and_unique():
    cfg = RootConfig.model_validate({"web": {"shortcuts": {"clear": "shift+ctrl+space", "help": "?"}}})
    assert cfg.web.shortcuts.clear == "Ctrl+Shift+Space"
    assert RootConfig.model_validate({"web": {"shortcuts": {"clear": "esc"}}}).web.shortcuts.clear == "Escape"
    assert RootConfig.model_validate({"web": {"shortcuts": {"pause_toggle": ""}}}).web.shortcuts.pause_toggle == ""
    with pytest.raises(ValueError, match="bound to both"):
        RootConfig.model_validate({"web": {"shortcuts": {"clear": "F2", "mode_toggle": "F2"}}})
    with pytest.raises(ValueError):
        RootConfig.model_validate({"web": {"shortcuts": {"clear": "Ctrl+Shift"}}})


def test_default_clear_shortcut_is_not_ctrl_n():
    assert RootConfig().web.shortcuts.clear not in ("Ctrl+N", "")


def test_gui_section_migrates_to_web(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"gui": {"enabled": False}}))
    config = load_config(path)
    assert config.web.enabled is False
    assert "gui" not in config.model_dump()
    explicit = RootConfig.model_validate({"gui": {"enabled": False}, "web": {"enabled": True}})
    assert explicit.web.enabled is True  # an explicit web key wins
    assert RootConfig.model_validate({"gui": {"enabled": True}}).web.enabled is True


def test_api_tokens_validated():
    with pytest.raises(ValueError):
        RootConfig.model_validate({"web": {"api_tokens": [{"name": "x", "token": "short"}]}})
    with pytest.raises(ValueError, match="unique"):
        RootConfig.model_validate({"web": {"api_tokens": [
            {"name": "x", "token": API_TOKEN}, {"name": "x", "token": API_TOKEN + "2"}]}})


# -------------------------------------------------------------------- OSC


class _OscController(FakeController):
    pass


def test_osc_parse_and_dispatch(config_path):
    controller = _OscController(config_path)
    for packet in (
        osc.encode_message("/narrator/clear"),
        osc.encode_message("/narrator/clear", 0.0),  # button release: ignored
        osc.encode_message("/narrator/mode", "manual"),
        osc.encode_message("/narrator/manual-text", "Zażółć gęślą jaźń"),
        osc.encode_message("/narrator/pause", 1),
        osc.encode_message("/narrator/pause", False),
        osc.encode_message("/other/thing", 1),
    ):
        for address, args in osc.parse_packet(packet):
            osc.dispatch(controller, address, args)
    assert controller.calls == [
        ("clear",), ("mode", "manual"), ("manual", "Zażółć gęślą jaźń"), ("pause", True), ("pause", False),
    ]


def test_osc_bundle():
    inner = osc.encode_message("/narrator/clear")
    bundle = b"#bundle\0" + b"\0" * 8 + len(inner).to_bytes(4, "big") + inner
    assert osc.parse_packet(bundle) == [("/narrator/clear", [])]
    with pytest.raises(osc.OscParseError):
        osc.parse_packet(b"no-slash\0\0\0\0")


def test_osc_over_udp(config_path):
    """The real datagram endpoint on loopback."""

    async def body():
        controller = FakeController(config_path)
        loop = asyncio.get_running_loop()
        transport, _ = await loop.create_datagram_endpoint(
            lambda: osc.OscProtocol(controller), local_addr=("127.0.0.1", 0)
        )
        port = transport.get_extra_info("sockname")[1]
        sender, _ = await loop.create_datagram_endpoint(asyncio.DatagramProtocol, remote_addr=("127.0.0.1", port))
        sender.sendto(osc.encode_message("/narrator/clear"))
        for _ in range(50):
            if controller.calls:
                break
            await asyncio.sleep(0.02)
        sender.close()
        transport.close()
        return controller.calls

    assert run(body()) == [("clear",)]


# ------------------------------------------------------------ server lifecycle


def test_server_thread_start_stop_writes_and_removes_ui_file(config_path):
    controller = FakeController(config_path)
    controller.config = controller.config.model_copy(
        update={"web": controller.config.web.model_copy(update={"port": _free_port()})}
    )
    ui_path = config_path.parent / "web-ui.json"
    write_password_file(config_path.parent / "web-password.json", PASSWORD)  # else the page is open
    server = WebServer(controller, ui_file_path=ui_path)
    server.start()
    try:
        assert read_ui_file(ui_path)["url"] == server.url
        import urllib.request

        with urllib.request.urlopen(server.url + "/api/state") as resp:  # noqa: S310 (loopback)
            pytest.fail(f"unauthenticated request succeeded: {resp.status}")
    except urllib.error.HTTPError as exc:
        assert exc.code == 401
    finally:
        server.stop()
    assert not ui_path.exists()


def _free_port() -> int:
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
