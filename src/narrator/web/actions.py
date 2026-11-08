"""Operator actions shared by the HTTP control API and the OSC adapter.

Each action takes loosely-typed input (JSON, form fields, query parameters,
OSC arguments), maps it onto one AppController call, and returns the fresh
state. Errors are ActionError with an HTTP status the API passes through.
"""
from __future__ import annotations

from narrator.app_controller import FeatureUnavailableError
from narrator.recording.session import RecordingError

_TRUE = {"1", "true", "on", "yes", "pause", "paused", "start"}
_FALSE = {"0", "false", "off", "no", "resume", "play", "stop"}


class ActionError(Exception):
    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


def parse_switch(value, current: bool) -> bool:
    """None / "toggle" -> the opposite of `current`; otherwise a bool from
    bool/number/string forms ("on", "off", 1, 0, ...)."""
    if value is None or (isinstance(value, str) and value.strip().lower() in ("", "toggle")):
        return not current
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _TRUE:
            return True
        if lowered in _FALSE:
            return False
    raise ActionError(f"expected true/false/on/off/toggle, got {value!r}")


def clear(controller) -> None:
    controller.clear_captions()


def set_mode(controller, mode) -> str:
    """mode: "auto", "manual", or "toggle"/None (switch to the other one)."""
    if mode is None or (isinstance(mode, str) and mode.strip().lower() in ("", "toggle")):
        target = "manual" if controller.caption_mode == "auto" else "auto"
    elif isinstance(mode, str) and mode.strip().lower() in ("auto", "manual"):
        target = mode.strip().lower()
    else:
        raise ActionError(f"mode must be auto, manual or toggle, got {mode!r}")
    controller.set_caption_mode(target)
    return target


def manual_text(controller, text) -> None:
    if not isinstance(text, str):
        raise ActionError("text must be a string")
    if len(text) > 2000:
        raise ActionError("text is too long (2000 characters max)")
    controller.submit_manual_text(text)


def pause(controller, value) -> bool:
    paused = parse_switch(value, controller.captions_paused)
    controller.set_captions_paused(paused)
    return paused


def recording(controller, value) -> bool:
    enabled = parse_switch(value, controller.recording)
    try:
        controller.set_recording(enabled)
    except FeatureUnavailableError as exc:
        raise ActionError(str(exc), status=501) from exc
    except RecordingError as exc:  # e.g. too little free disk space
        raise ActionError(str(exc), status=409) from exc
    return enabled


def fix(controller, action) -> str:
    """A fix action from the problems panel (narrator.health): "asr.retry",
    "asr.use_cpu", "audio.retry", "ndi.retry", "health.recheck". Blocking."""
    if not isinstance(action, str) or not action.strip():
        raise ActionError("action must be a fix action id, e.g. asr.retry")
    action = action.strip()
    try:
        controller.apply_fix(action)
    except ValueError as exc:
        raise ActionError(str(exc), status=400) from exc
    except LookupError as exc:  # no current problem offers it (already fixed?)
        raise ActionError(str(exc), status=409) from exc
    return action
