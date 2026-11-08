"""The web admin UI server: aiohttp on its own thread and asyncio loop.

Threading: `WebServer.start()` starts one daemon thread running an asyncio
loop for the life of the app. The HTTP listener and the optional OSC
endpoint live on that loop and are (re)started there when `web.*` changes.
Nothing here runs on, locks, or waits for the ASR/render/NDI threads:

  - state is read through AppController.status_snapshot() (plain attribute
    reads) and the caption feed through `controller.ui_events`, a bounded
    drop-oldest deque the event-drain thread appends to;
  - every browser connection has its own bounded drop-oldest queue, so a
    stalled tab only ever loses its own messages;
  - preview rendering and PNG encoding run in the default executor, never on
    the loop, and only when the frame actually changed.

Config is never mutated in memory: Save validates with RootConfig and writes
config.yaml through config.loader.save_config; the config watcher then
applies it exactly as it would a hand edit.
"""
from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path

from aiohttp import WSMsgType, web
from pydantic import ValidationError

from narrator.config.loader import load_config, save_config
from narrator.config.schema import CanvasConfig, LayoutConfig, RootConfig, StyleConfig, WebConfig
from narrator.render.preview_sync import PreviewSync
from narrator.web import actions
from narrator.web.logbuffer import log_ring
from narrator.web.security import (
    SESSION_COOKIE,
    allowed_host_names,
    default_token_path,
    host_allowed,
    new_secret,
    origin_allowed,
    remove_token_file,
    tokens_equal,
    write_token_file,
)

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).with_name("static")
_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
_TICK_SECONDS = 0.1  # state/caption/log fan-out cadence (mic meter: 10 Hz)
_CLIENT_QUEUE = 200
_CAPTION_HISTORY = 60
_PREVIEW_MAX_SIZE = (960, 540)
_FANOUT_TASK = web.AppKey("fanout", asyncio.Task)

# Paths reachable with a per-integration API token (web.api_tokens).
_CONTROL_PATHS = frozenset(
    {"/api/state", "/api/clear", "/api/mode", "/api/manual-text", "/api/pause", "/api/recording"}
)


def _error(message: str, status: int) -> web.Response:
    return web.json_response({"error": message}, status=status)


def _validation_errors(exc: ValidationError) -> list[dict]:
    errors = []
    for err in exc.errors():
        message = err.get("msg", "")
        message = message.removeprefix("Value error, ").removeprefix("Assertion failed, ")
        errors.append({"loc": ".".join(str(part) for part in err.get("loc", ())), "msg": message})
    return errors


class _Client:
    """One WebSocket connection."""

    def __init__(self, ws: web.WebSocketResponse) -> None:
        self.ws = ws
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=_CLIENT_QUEUE)
        self.preview_enabled = True
        self.draft: tuple[CanvasConfig, LayoutConfig, StyleConfig] | None = None
        self.draft_version = 0
        self.preview_sync: PreviewSync | None = None
        self.last_frame = None
        self.last_frame_key = None
        self.frame_consumed = asyncio.Event()
        self.frame_consumed.set()

    def send(self, message) -> None:
        """Queue a message (dict -> JSON text, bytes -> binary); on overflow
        the oldest queued message is dropped -- this never waits."""
        try:
            self.queue.put_nowait(message)
        except asyncio.QueueFull:
            try:
                dropped = self.queue.get_nowait()
            except asyncio.QueueEmpty:
                dropped = None
            if isinstance(dropped, bytes):
                self.frame_consumed.set()  # a dropped frame must not stall the preview
            self.queue.put_nowait(message)

    async def run_sender(self) -> None:
        while not self.ws.closed:
            message = await self.queue.get()
            try:
                if isinstance(message, bytes):
                    await self.ws.send_bytes(message)
                    self.frame_consumed.set()
                else:
                    await self.ws.send_str(json.dumps(message))
            except (ConnectionError, RuntimeError):
                return


class WebServer:
    def __init__(self, controller, *, token: str | None = None, token_path: Path | None = None) -> None:
        self._controller = controller
        self._token = token or new_secret()
        self._token_path = token_path or default_token_path()
        self._sessions: set[str] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._runner: web.AppRunner | None = None
        self._http_key: tuple | None = None
        self._osc_transport = None
        self._osc_key: tuple | None = None
        self._port: int | None = None
        self._url: str | None = None
        self._clients: set[_Client] = set()
        self._caption_history: list[dict] = []
        self._config_rev = 0
        self._devices: list[str] | None = None
        self._benchmark: dict = {"running": False, "exit_code": None, "started": None, "finished": None}
        self._lifecycle_lock = threading.Lock()

    # ---------------------------------------------------------------- lifecycle

    @property
    def token(self) -> str:
        return self._token

    @property
    def url(self) -> str | None:
        return self._url

    def login_url(self) -> str | None:
        return None if self._url is None else f"{self._url}/auth?token={self._token}"

    def start(self) -> None:
        loop = asyncio.new_event_loop()
        ready = threading.Event()

        def run() -> None:
            asyncio.set_event_loop(loop)
            ready.set()
            loop.run_forever()
            loop.close()

        self._loop = loop
        self._thread = threading.Thread(target=run, name="narrator-web", daemon=True)
        self._thread.start()
        ready.wait()
        self._call(self._apply(self._controller.config.web), timeout=15)

    def stop(self) -> None:
        with self._lifecycle_lock:
            loop, thread = self._loop, self._thread
            if loop is None:
                return
            self._loop = None
        try:
            asyncio.run_coroutine_threadsafe(self._shutdown(), loop).result(timeout=10)
        except Exception:
            logger.warning("web UI did not shut down cleanly", exc_info=True)
        loop.call_soon_threadsafe(loop.stop)
        if thread is not None:
            thread.join(timeout=5)

    def _call(self, coro, timeout: float):
        loop = self._loop
        if loop is None:
            coro.close()
            return None
        return asyncio.run_coroutine_threadsafe(coro, loop).result(timeout=timeout)

    def on_config_changed(self, old_config: RootConfig, new_config: RootConfig) -> None:
        """Called by AppController on the config watcher thread after a swap."""
        self._config_rev += 1
        if new_config.web != old_config.web:
            try:
                self._call(self._apply(new_config.web), timeout=15)
            except Exception:
                logger.exception("web UI: applying the new web settings failed")

    async def _apply(self, cfg: WebConfig) -> None:
        http_key = (cfg.bind, cfg.port) if cfg.enabled else None
        if http_key != self._http_key:
            await self._stop_http()
            if http_key is not None:
                await self._start_http(cfg)
        osc_key = (cfg.osc.bind, cfg.osc.port) if cfg.osc.enabled else None
        if osc_key != self._osc_key:
            self._stop_osc()
            if osc_key is not None:
                await self._start_osc(cfg)

    async def _start_http(self, cfg: WebConfig) -> None:
        runner = web.AppRunner(self.create_app(), access_log=None, handle_signals=False)
        await runner.setup()
        site = web.TCPSite(runner, cfg.bind, cfg.port)
        try:
            await site.start()
        except OSError as exc:
            logger.error("web UI: cannot listen on %s:%s (%s); captions keep running without it",
                         cfg.bind, cfg.port, exc)
            await runner.cleanup()
            return
        self._runner = runner
        self._http_key = (cfg.bind, cfg.port)
        sockets = getattr(site._server, "sockets", None) or []  # actual port when cfg.port is 0
        self._port = sockets[0].getsockname()[1] if sockets else cfg.port
        shown = "127.0.0.1" if cfg.bind in ("0.0.0.0", "::", "", "localhost") else cfg.bind
        if ":" in shown:
            shown = f"[{shown}]"
        self._url = f"http://{shown}:{self._port}"
        try:
            write_token_file(self._token_path, self._token, self._url)
        except OSError as exc:
            logger.warning("web UI: cannot write the token file %s: %s", self._token_path, exc)
        logger.info("Web UI at %s -- login token: %s (`narrator --open-ui` opens it logged in)",
                    self._url, self._token)
        if cfg.bind not in ("127.0.0.1", "localhost", "::1"):
            logger.warning("web UI is bound to %s, not loopback: keep it off untrusted networks", cfg.bind)

    async def _stop_http(self) -> None:
        runner, self._runner = self._runner, None
        self._http_key = None
        if runner is not None:
            await runner.cleanup()
            remove_token_file(self._token_path, self._token)
            self._url = None

    async def _start_osc(self, cfg: WebConfig) -> None:
        from narrator.web.osc import OscProtocol

        loop = asyncio.get_running_loop()
        try:
            transport, _ = await loop.create_datagram_endpoint(
                lambda: OscProtocol(self._controller), local_addr=(cfg.osc.bind, cfg.osc.port)
            )
        except OSError as exc:
            logger.error("OSC input: cannot listen on udp %s:%s (%s)", cfg.osc.bind, cfg.osc.port, exc)
            return
        self._osc_transport = transport
        self._osc_key = (cfg.osc.bind, cfg.osc.port)
        logger.info("OSC input listening on udp %s:%s", cfg.osc.bind, cfg.osc.port)

    def _stop_osc(self) -> None:
        transport, self._osc_transport = self._osc_transport, None
        self._osc_key = None
        if transport is not None:
            transport.close()

    async def _shutdown(self) -> None:
        self._stop_osc()
        await self._stop_http()

    # ------------------------------------------------------------------ the app

    def create_app(self) -> web.Application:
        app = web.Application(middlewares=[self._security_middleware], client_max_size=2 * 1024 * 1024)
        r = app.router
        r.add_get("/", self._index)
        r.add_get("/auth", self._auth)
        r.add_post("/auth", self._auth)
        r.add_post("/api/logout", self._logout)
        r.add_get("/api/state", self._state)
        r.add_post("/api/clear", self._clear)
        r.add_post("/api/mode", self._mode)
        r.add_post("/api/manual-text", self._manual_text)
        r.add_post("/api/pause", self._pause)
        r.add_post("/api/recording", self._recording)
        r.add_post("/api/fix", self._fix)
        r.add_get("/api/config", self._get_config)
        r.add_post("/api/config", self._save_config)
        r.add_get("/api/schema", self._schema)
        r.add_get("/api/choices", self._choices)
        r.add_get("/api/devices", self._list_devices)
        r.add_get("/api/log", self._log)
        r.add_get("/api/vocabulary/{name}", self._get_vocabulary)
        r.add_put("/api/vocabulary/{name}", self._put_vocabulary)
        r.add_post("/api/preset/benchmark", self._run_benchmark)
        r.add_get("/api/recordings", self._list_recordings)
        r.add_post("/api/recordings/delete-all", self._delete_all_recordings)
        r.add_get("/api/recordings/{id}/files/{name}", self._recording_file)
        r.add_put("/api/recordings/{id}/corrected", self._upload_corrected)
        r.add_post("/api/recordings/{id}/draft", self._generate_draft)
        r.add_post("/api/recordings/{id}/reference", self._use_as_reference)
        r.add_delete("/api/recordings/{id}", self._delete_recording)
        r.add_get("/ws", self._websocket)
        r.add_static("/static", STATIC_DIR, show_index=False)
        app.on_response_prepare.append(self._add_security_headers)
        app.on_startup.append(self._on_app_startup)
        app.on_shutdown.append(self._on_app_shutdown)
        return app

    async def _on_app_startup(self, app: web.Application) -> None:
        app[_FANOUT_TASK] = asyncio.create_task(self._fanout_loop())

    async def _on_app_shutdown(self, app: web.Application) -> None:
        task = app.get(_FANOUT_TASK)
        if task is not None:
            task.cancel()
        for client in list(self._clients):
            await client.ws.close(code=1001, message=b"server shutting down")

    # ---------------------------------------------------------------- security

    def _web_config(self) -> WebConfig:
        config = self._controller.config
        return config.web if config is not None else WebConfig()

    @web.middleware
    async def _security_middleware(self, request: web.Request, handler):
        cfg = self._web_config()
        sockname = request.transport.get_extra_info("sockname") if request.transport else None
        port = sockname[1] if sockname else (self._port or cfg.port)
        allowed = allowed_host_names(cfg.bind, cfg.allowed_hosts)
        if not host_allowed(request.headers.get("Host"), allowed, port):
            # DNS rebinding: a hostile name pointed at 127.0.0.1.
            return web.Response(status=421, text="Misdirected request: unexpected Host header\n")
        if request.method not in _SAFE_METHODS or request.path == "/ws":
            if not origin_allowed(request.headers.get("Origin"), allowed, port):
                return _error("cross-origin request refused", 403)
        return await handler(request)

    async def _add_security_headers(self, request: web.Request, response: web.StreamResponse) -> None:
        host = request.headers.get("Host", "")
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' blob: data:; style-src 'self'; script-src 'self'; "
            f"connect-src 'self' ws://{host}; frame-ancestors 'none'; base-uri 'none'; form-action 'self'",
        )
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        # same-origin, not no-referrer: with no-referrer Chromium sends
        # "Origin: null" on form posts, which the Origin check (rightly) refuses.
        response.headers.setdefault("Referrer-Policy", "same-origin")
        if not request.path.startswith("/static/"):
            response.headers.setdefault("Cache-Control", "no-store")

    @staticmethod
    def _presented_token(request: web.Request) -> str | None:
        auth = request.headers.get("Authorization", "")
        if auth.lower().startswith("bearer "):
            return auth[7:].strip()
        return request.headers.get("X-Narrator-Token")

    def _integration_name(self, request: web.Request) -> str | None:
        """The web.api_tokens entry the request presents, if any."""
        presented = self._presented_token(request)
        if not presented:
            return None
        for api_token in self._web_config().api_tokens:
            if tokens_equal(presented, api_token.token):
                return api_token.name
        return None

    def _access_level(self, request: web.Request) -> str | None:
        """"admin" (session cookie / startup token), "control" (an API token),
        or None."""
        session = request.cookies.get(SESSION_COOKIE)
        if session and any(tokens_equal(session, s) for s in list(self._sessions)):
            return "admin"
        if tokens_equal(self._presented_token(request), self._token):
            return "admin"
        return "control" if self._integration_name(request) else None

    def _require(self, request: web.Request, level: str = "admin") -> web.Response | None:
        granted = self._access_level(request)
        if granted == "admin" or (granted == "control" and level == "control"):
            return None
        if granted == "control":
            return _error("this API token only grants the control API", 403)
        return _error("authentication required", 401)

    # ----------------------------------------------------------------- pages

    async def _index(self, request: web.Request) -> web.StreamResponse:
        page = "index.html" if self._access_level(request) == "admin" else "login.html"
        return web.FileResponse(STATIC_DIR / page)

    async def _auth(self, request: web.Request) -> web.StreamResponse:
        if request.method == "POST":
            form = await request.post()
            given = str(form.get("token", "")).strip()
        else:
            given = request.query.get("token", "").strip()
        if not tokens_equal(given, self._token):
            await asyncio.sleep(0.5)  # slows guessing; the token is 256 bits anyway
            return web.Response(status=303, headers={"Location": "/?login=failed"})
        session = new_secret()
        self._sessions.add(session)
        # The redirect drops the token from the address bar.
        response = web.Response(status=303, headers={"Location": "/"})
        response.set_cookie(SESSION_COOKIE, session, httponly=True, samesite="Strict", path="/")
        return response

    async def _logout(self, request: web.Request) -> web.Response:
        session = request.cookies.get(SESSION_COOKIE)
        self._sessions.discard(session)
        response = web.json_response({"ok": True})
        response.del_cookie(SESSION_COOKIE, path="/")
        return response

    # ---------------------------------------------------------- control API

    def _state_payload(self) -> dict:
        state = self._controller.status_snapshot()
        cfg = self._web_config()
        state.update(
            config_rev=self._config_rev,
            shortcuts=cfg.shortcuts.model_dump(),
            benchmark=dict(self._benchmark),
        )
        return state

    async def _params(self, request: web.Request) -> dict:
        data: dict = dict(request.query)
        if not request.can_read_body:
            return data
        if request.content_type in ("application/x-www-form-urlencoded", "multipart/form-data"):
            data.update(dict(await request.post()))
            return data
        raw = await request.text()
        if raw.strip():
            try:
                body = json.loads(raw)
            except ValueError as exc:
                raise actions.ActionError(f"request body is not valid JSON: {exc}") from exc
            if not isinstance(body, dict):
                raise actions.ActionError("request body must be a JSON object")
            data.update(body)
        return data

    async def _control(
        self, request: web.Request, run, blocking: bool = False, level: str = "control"
    ) -> web.Response:
        denied = self._require(request, level)
        if denied is not None:
            return denied
        try:
            params = await self._params(request)
            if blocking:  # file I/O (recording start/stop flushes): off the loop
                result = await asyncio.get_running_loop().run_in_executor(None, run, params)
            else:
                result = run(params)
        except actions.ActionError as exc:
            return _error(str(exc), exc.status)
        integration = self._integration_name(request)
        if integration:
            logger.info("control API: %s via API token %r", request.path, integration)
        payload = {"ok": True, "state": self._state_payload()}
        if result is not None:
            payload["result"] = result
        return web.json_response(payload)

    async def _state(self, request: web.Request) -> web.Response:
        denied = self._require(request, "control")
        return denied or web.json_response(self._state_payload())

    async def _clear(self, request):
        return await self._control(request, lambda p: actions.clear(self._controller))

    async def _mode(self, request):
        return await self._control(request, lambda p: actions.set_mode(self._controller, p.get("mode")))

    async def _manual_text(self, request):
        return await self._control(request, lambda p: actions.manual_text(self._controller, p.get("text", "")))

    async def _pause(self, request):
        return await self._control(request, lambda p: actions.pause(self._controller, p.get("paused")))

    async def _recording(self, request):
        return await self._control(
            request, lambda p: actions.recording(self._controller, p.get("recording")), blocking=True
        )

    async def _fix(self, request):
        """Problems panel fix (narrator.health). Admin only, not the API
        tokens: "Use CPU instead" writes config.yaml. Blocking (restarts
        join threads/processes), so it runs in the executor."""
        return await self._control(
            request, lambda p: actions.fix(self._controller, p.get("action")), blocking=True, level="admin"
        )

    # ------------------------------------------------------------ recordings

    def _recordings_or_error(self):
        recordings = getattr(self._controller, "recordings", None)
        if recordings is None:
            raise actions.ActionError("recording is not available", status=501)
        return recordings

    async def _recordings_call(self, request: web.Request, fn) -> web.Response:
        """Admin-only recordings action, run in the executor (file I/O);
        maps the store's errors onto HTTP statuses."""
        denied = self._require(request)
        if denied is not None:
            return denied
        from narrator.recording.session import RecordingError
        from narrator.recording.srt import SrtError
        from narrator.recording.store import RecordingNotFound

        try:
            recordings = self._recordings_or_error()
            result = await asyncio.get_running_loop().run_in_executor(None, fn, recordings)
        except actions.ActionError as exc:
            return _error(str(exc), exc.status)
        except RecordingNotFound as exc:
            return _error(str(exc), 404)
        except SrtError as exc:
            return web.json_response({"ok": False, "error": f"invalid SRT: {exc}"}, status=422)
        except PermissionError as exc:
            return _error(str(exc), 409)
        except (FileNotFoundError, RecordingError, ValueError) as exc:
            return _error(str(exc), 409)
        if isinstance(result, web.StreamResponse):
            return result
        return web.json_response({"ok": True, **(result or {})})

    async def _list_recordings(self, request: web.Request) -> web.Response:
        return await self._recordings_call(
            request,
            lambda r: {"recordings": r.list(), "disk": r.disk(), "status": r.status()},
        )

    async def _recording_file(self, request: web.Request) -> web.StreamResponse:
        from narrator.recording.store import DOWNLOADABLE, session_dir

        recording_id, name = request.match_info["id"], request.match_info["name"]
        if name not in DOWNLOADABLE:
            return _error(f"unknown file {name!r}", 404)

        def locate(recordings):
            path = session_dir(recordings.root, recording_id) / name
            if not path.is_file():
                raise actions.ActionError(f"{name} does not exist for this recording", status=404)
            return web.FileResponse(
                path,
                headers={"Content-Disposition": f'attachment; filename="narrator-{recording_id}-{name}"'},
            )

        return await self._recordings_call(request, locate)

    async def _upload_corrected(self, request: web.Request) -> web.Response:
        denied = self._require(request)
        if denied is not None:
            return denied
        recording_id = request.match_info["id"]
        if request.content_type == "multipart/form-data":
            form = await request.post()
            field = form.get("file")
            data = field.file.read() if hasattr(field, "file") else str(field or "").encode("utf-8")
        else:
            data = await request.read()
        return await self._recordings_call(
            request, lambda r: {"cues": r.save_corrected(recording_id, data)}
        )

    async def _generate_draft(self, request: web.Request) -> web.Response:
        recording_id = request.match_info["id"]
        return await self._recordings_call(request, lambda r: r.request_draft(recording_id, force=True))

    async def _use_as_reference(self, request: web.Request) -> web.Response:
        recording_id = request.match_info["id"]
        return await self._recordings_call(request, lambda r: r.use_as_reference(recording_id))

    async def _delete_recording(self, request: web.Request) -> web.Response:
        recording_id = request.match_info["id"]
        return await self._recordings_call(request, lambda r: r.delete(recording_id))

    async def _delete_all_recordings(self, request: web.Request) -> web.Response:
        try:
            params = await self._params(request)
        except actions.ActionError as exc:
            return _error(str(exc), 400)
        if params.get("confirm") not in (True, "true", "yes", "1", 1):
            return _error('deleting every recording needs {"confirm": true}', 400)
        return await self._recordings_call(request, lambda r: {"deleted": r.delete_all()})

    # ------------------------------------------------------------- config API

    async def _get_config(self, request: web.Request) -> web.Response:
        denied = self._require(request)
        if denied is not None:
            return denied
        path = self._controller.config_path
        file_error = None
        try:
            config = await asyncio.get_running_loop().run_in_executor(None, load_config, path)
        except Exception as exc:  # an invalid hand edit: show the live config instead
            file_error = str(exc)
            config = self._controller.config
        return web.json_response(
            {
                "config": config.model_dump(mode="json"),
                "config_path": str(path),
                "file_error": file_error,
                "rev": self._config_rev,
            }
        )

    async def _save_config(self, request: web.Request) -> web.Response:
        """Validate with the schema and write config.yaml; the config watcher
        applies it (the same path as a hand edit). Validation errors come
        back as [{loc: "asr.step_ms", msg}] for the form to show."""
        denied = self._require(request)
        if denied is not None:
            return denied
        try:
            body = await request.json()
        except ValueError:
            return _error("request body is not valid JSON", 400)
        raw = body.get("config") if isinstance(body, dict) and "config" in body else body
        if not isinstance(raw, dict):
            return _error("expected a config object", 400)
        try:
            new_config = RootConfig.model_validate(raw)
        except ValidationError as exc:
            return web.json_response({"ok": False, "errors": _validation_errors(exc)}, status=422)
        path = self._controller.config_path
        loop = asyncio.get_running_loop()
        try:
            current = await loop.run_in_executor(None, load_config, path)
        except Exception:
            current = None
        changed = current != new_config
        if changed:
            await loop.run_in_executor(None, save_config, new_config, path)
            logger.info("config saved from the web UI (%s)", path)
        old_web = self._web_config()
        new_url = None
        if new_config.web.port != old_web.port and new_config.web.enabled:
            new_url = f"http://127.0.0.1:{new_config.web.port}/"
        return web.json_response(
            {"ok": True, "changed": changed, "config": new_config.model_dump(mode="json"), "new_url": new_url}
        )

    async def _schema(self, request: web.Request) -> web.Response:
        denied = self._require(request)
        return denied or web.json_response(RootConfig.model_json_schema())

    async def _choices(self, request: web.Request) -> web.Response:
        denied = self._require(request)
        if denied is not None:
            return denied
        from narrator.captions.transcript import default_transcript_directory
        from narrator.config import choices

        config = self._controller.config

        def gather() -> dict:
            from narrator.asr.catalog import catalog_rows
            from narrator.asr.vocabulary import list_vocabulary_lists

            return {
                "models": choices.model_dropdown_options(config.asr.model),
                # The model catalog: backend, unit kinds, installed (todo C6).
                "catalog": catalog_rows(config.asr.model),
                "fonts": choices.font_options(config.style.font_family),
                "languages": choices.language_options(),
                "presets": choices.preset_options(),
                "vocabulary_lists": list_vocabulary_lists(choices.vocabulary_dir(self._controller.config_path)),
                "transcript_default_dir": str(default_transcript_directory()),
            }

        return web.json_response(await asyncio.get_running_loop().run_in_executor(None, gather))

    async def _list_devices(self, request: web.Request) -> web.Response:
        """Microphones and NDI sources (NDI discovery takes ~1.5 s, so the
        list is cached until ?refresh=1)."""
        denied = self._require(request)
        if denied is not None:
            return denied
        if self._devices is None or request.query.get("refresh"):
            from narrator.audio.devices import list_device_choices

            own = self._controller.config.ndi.source_name
            try:
                found = await asyncio.get_running_loop().run_in_executor(None, list_device_choices, own)
                self._devices = list(dict.fromkeys(found))  # PortAudio may also call a device "default"
            except Exception as exc:
                logger.warning("device listing failed: %s", exc)
                self._devices = ["default"]
        return web.json_response({"devices": self._devices})

    async def _log(self, request: web.Request) -> web.Response:
        denied = self._require(request)
        if denied is not None:
            return denied
        ring = log_ring()
        try:
            after = int(request.query.get("after", "0"))
        except ValueError:
            after = 0
        return web.json_response({"lines": ring.since(after) if ring else []})

    def _vocabulary_args(self, request: web.Request):
        from narrator.asr.vocabulary import is_valid_list_name
        from narrator.config.choices import vocabulary_dir

        name = request.match_info["name"]
        if not is_valid_list_name(name):
            raise actions.ActionError(f"invalid vocabulary list name {name!r}")
        return name, vocabulary_dir(self._controller.config_path)

    async def _get_vocabulary(self, request: web.Request) -> web.Response:
        denied = self._require(request)
        if denied is not None:
            return denied
        from narrator.asr.vocabulary import load_vocabulary_list

        try:
            name, directory = self._vocabulary_args(request)
        except actions.ActionError as exc:
            return _error(str(exc), 400)
        try:
            entries = load_vocabulary_list(name, directory)
        except FileNotFoundError:
            entries = []
        except OSError as exc:
            return _error(f"cannot read list {name!r}: {exc}", 500)
        return web.json_response({"name": name, "entries": entries})

    async def _put_vocabulary(self, request: web.Request) -> web.Response:
        denied = self._require(request)
        if denied is not None:
            return denied
        from narrator.asr.vocabulary import save_vocabulary_list

        try:
            name, directory = self._vocabulary_args(request)
            body = await request.json()
        except actions.ActionError as exc:
            return _error(str(exc), 400)
        except ValueError:
            return _error("request body is not valid JSON", 400)
        entries = body.get("entries") if isinstance(body, dict) else None
        if not isinstance(entries, list) or not all(isinstance(e, str) for e in entries):
            return _error("entries must be a list of strings", 400)
        await asyncio.get_running_loop().run_in_executor(None, save_vocabulary_list, name, entries, directory)
        if self._controller.config.asr.vocabulary_list == name:
            self._controller.refresh_text_settings()  # the config did not change, the list did
        logger.info("vocabulary list %r saved from the web UI (%d entries)", name, len(entries))
        return web.json_response({"ok": True, "name": name})

    async def _run_benchmark(self, request: web.Request) -> web.Response:
        """`narrator --pick-preset` in a subprocess; its output goes to the
        log view, and the config it writes is applied by the watcher."""
        denied = self._require(request)
        if denied is not None:
            return denied
        if self._benchmark["running"]:
            return _error("a benchmark is already running", 409)
        if getattr(sys, "frozen", False):
            cmd = [sys.executable, "--pick-preset"]
        else:
            cmd = [sys.executable, "-m", "narrator.main", "--pick-preset"]
        cmd += ["--config", str(self._controller.config_path)]
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},  # progress lines as they happen
        )
        self._benchmark = {"running": True, "exit_code": None, "started": time.time(), "finished": None}
        logger.info("preset benchmark started (pid %s)", process.pid)
        asyncio.get_running_loop().create_task(self._watch_benchmark(process))
        return web.json_response({"ok": True, "pid": process.pid})

    async def _watch_benchmark(self, process) -> None:
        bench_log = logging.getLogger("narrator.preset_benchmark")
        assert process.stdout is not None
        async for raw in process.stdout:
            line = raw.decode("utf-8", errors="replace").rstrip()
            if line:
                bench_log.info("%s", line)
        code = await process.wait()
        self._benchmark = {**self._benchmark, "running": False, "exit_code": code, "finished": time.time()}
        logger.info("preset benchmark finished (exit code %s)", code)

    # --------------------------------------------------------------- websocket

    async def _websocket(self, request: web.Request) -> web.StreamResponse:
        denied = self._require(request)
        if denied is not None:
            return denied
        ws = web.WebSocketResponse(heartbeat=20, max_msg_size=1024 * 1024)
        await ws.prepare(request)
        client = _Client(ws)
        self._clients.add(client)
        client.send({"type": "state", **self._state_payload()})
        if self._caption_history:
            client.send({"type": "captions", "events": list(self._caption_history)})
        ring = log_ring()
        if ring is not None:
            client.send({"type": "log", "lines": ring.since(0, limit=300)})
        sender = asyncio.create_task(client.run_sender())
        preview = asyncio.create_task(self._preview_loop(client))
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    self._on_client_message(client, msg.data)
                elif msg.type == WSMsgType.ERROR:
                    break
        finally:
            self._clients.discard(client)
            sender.cancel()
            preview.cancel()
        return ws

    def _on_client_message(self, client: _Client, text: str) -> None:
        try:
            message = json.loads(text)
        except ValueError:
            return
        if not isinstance(message, dict):
            return
        kind = message.get("type")
        if kind == "preview":
            client.preview_enabled = bool(message.get("enabled", True))
            client.last_frame_key = None  # resend on resume
        elif kind == "draft":
            # Unsaved display settings to preview; null = show the live config.
            draft = message.get("display")
            if not draft:
                client.draft = None
            else:
                try:
                    client.draft = (
                        CanvasConfig.model_validate(draft.get("canvas", {})),
                        LayoutConfig.model_validate(draft.get("layout", {})),
                        StyleConfig.model_validate(draft.get("style", {})),
                    )
                except (ValidationError, AttributeError) as exc:
                    errors = _validation_errors(exc) if isinstance(exc, ValidationError) else []
                    client.send({"type": "draft_error", "errors": errors})
                    return
            client.draft_version += 1
            client.send({"type": "draft_error", "errors": []})

    def _render_preview(self, client: _Client) -> bytes | None:
        """Runs in the executor: the frame for this client, PNG-encoded, or
        None when it is unchanged since the last one sent."""
        renderer = self._controller.renderer
        if renderer is None:
            return None
        canvas, layout, style = client.draft or (
            renderer.canvas_config, renderer.layout_config, renderer.style_config
        )
        if client.preview_sync is None:
            client.preview_sync = PreviewSync(canvas, layout, style)
        frame = client.preview_sync.refresh(canvas, layout, style, renderer)
        key = (id(frame), client.draft_version)
        if frame is client.last_frame and key == client.last_frame_key:
            return None
        client.last_frame, client.last_frame_key = frame, key
        image = frame.copy()
        image.thumbnail(_PREVIEW_MAX_SIZE)
        out = io.BytesIO()
        image.save(out, format="PNG", compress_level=1)
        return out.getvalue()

    async def _preview_loop(self, client: _Client) -> None:
        loop = asyncio.get_running_loop()
        while not client.ws.closed:
            interval = 1.0 / self._web_config().preview_fps
            if client.preview_enabled and client.frame_consumed.is_set():
                try:
                    data = await loop.run_in_executor(None, self._render_preview, client)
                except Exception as exc:  # e.g. a draft whose geometry cannot be laid out
                    logger.debug("preview render failed: %s", exc)
                    client.send({"type": "preview_error", "message": str(exc)})
                    data = None
                    await asyncio.sleep(1.0)
                if data is not None:
                    client.frame_consumed.clear()  # one frame in flight per client
                    client.send(data)
            await asyncio.sleep(interval)

    async def _fanout_loop(self) -> None:
        """State (10 Hz), caption events and new log lines to every client."""
        last_log_seq = log_ring().last_seq if log_ring() else 0
        while True:
            await asyncio.sleep(_TICK_SECONDS)
            events = []
            feed = getattr(self._controller, "ui_events", None)
            while feed:
                try:
                    kind, text, applied = feed.popleft()
                except IndexError:
                    break
                events.append({"kind": kind, "text": text, "applied": applied, "time": time.time()})
            if events:
                self._caption_history = (self._caption_history + events)[-_CAPTION_HISTORY:]
            if not self._clients:
                continue
            try:
                state = {"type": "state", **self._state_payload()}
            except Exception:
                logger.debug("state snapshot failed", exc_info=True)
                continue
            ring = log_ring()
            lines = ring.since(last_log_seq) if ring is not None else []
            if lines:
                last_log_seq = lines[-1]["seq"]
            for client in list(self._clients):
                client.send(state)
                if events:
                    client.send({"type": "captions", "events": events})
                if lines:
                    client.send({"type": "log", "lines": lines})
