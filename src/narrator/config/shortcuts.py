"""Keyboard-shortcut strings for the web UI (`web.shortcuts`).

A shortcut is written like "Ctrl+Shift+Space", "Escape", "F2" or "?". It is
stored in a canonical form -- modifiers in the order Ctrl, Alt, Shift, Meta,
then the key -- so the browser can compare it with the combo it builds from a
KeyboardEvent (web/static/app.js `comboFromEvent` mirrors this module; keep
the two in step).

Rules shared with app.js:
  - letters are upper-cased ("ctrl+k" -> "Ctrl+K");
  - Shift is dropped for other single printable characters, because the
    character already encodes it ("Shift+/" types "?", stored as "?");
  - named keys use KeyboardEvent.key spelling (Escape, Enter, Space, Tab,
    ArrowUp, F1..F24, ...), with a few aliases (Esc, Return, Del, Cmd).

Combos the browser or OS keeps for itself are refused: Chrome never lets a
page intercept Ctrl+N / Ctrl+T / Ctrl+W (and friends), so binding the clear
action to Ctrl+N -- the old Tk GUI's shortcut -- would silently open a new
window on air instead.
"""
from __future__ import annotations

_MODIFIER_ORDER = ("Ctrl", "Alt", "Shift", "Meta")
_MODIFIER_ALIASES = {
    "ctrl": "Ctrl", "control": "Ctrl", "ctl": "Ctrl",
    "alt": "Alt", "option": "Alt", "opt": "Alt",
    "shift": "Shift",
    "meta": "Meta", "cmd": "Meta", "command": "Meta", "super": "Meta", "win": "Meta",
}
_NAMED_KEYS = {
    name.lower(): name
    for name in (
        "Escape", "Enter", "Space", "Tab", "Backspace", "Delete", "Insert", "Home", "End",
        "PageUp", "PageDown", "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight",
        "Pause", "ScrollLock", "PrintScreen", "ContextMenu",
        *(f"F{n}" for n in range(1, 25)),
    )
}
_KEY_ALIASES = {
    "esc": "Escape", "return": "Enter", "del": "Delete", "ins": "Insert",
    "up": "ArrowUp", "down": "ArrowDown", "left": "ArrowLeft", "right": "ArrowRight",
    "pgup": "PageUp", "pgdn": "PageDown", "plus": "+", "spacebar": "Space",
}

# Taken by the browser (Chrome/Firefox: new window/tab, close, quit, tab
# switching, devtools, history navigation) or the OS. A page cannot reliably
# preventDefault these, so they must never carry an operator action.
_RESERVED = frozenset(
    {
        "Ctrl+N", "Ctrl+Shift+N", "Ctrl+T", "Ctrl+Shift+T", "Ctrl+W", "Ctrl+Shift+W",
        "Ctrl+Q", "Ctrl+Shift+Q", "Ctrl+Tab", "Ctrl+Shift+Tab", "Ctrl+PageUp", "Ctrl+PageDown",
        "Ctrl+F4", "Alt+F4", "F11", "F12", "Ctrl+Shift+I", "Ctrl+Shift+J", "Ctrl+Shift+C",
        "Ctrl+Shift+Delete", "Alt+ArrowLeft", "Alt+ArrowRight", "Alt+Tab", "Tab", "Shift+Tab",
        "Meta+N", "Meta+Shift+N", "Meta+T", "Meta+Shift+T", "Meta+W", "Meta+Shift+W", "Meta+Q",
        "Meta+Tab",
        *(f"Ctrl+{n}" for n in range(1, 10)),
        *(f"Alt+{n}" for n in range(1, 10)),
        *(f"Meta+{n}" for n in range(1, 10)),
    }
)


def canonical_shortcut(combo: str) -> str:
    """Canonical spelling of `combo`; "" stays "" (action unbound).
    Raises ValueError for an unparseable combo."""
    text = combo.strip()
    if not text:
        return ""
    if text == "+":
        parts = ["+"]
    elif text.endswith("++"):
        parts = [p.strip() for p in text[:-2].split("+")] + ["+"]
    else:
        parts = [p.strip() for p in text.split("+")]
    if any(not p for p in parts):
        raise ValueError(f"shortcut {combo!r} is malformed")
    *mod_parts, key_part = parts
    modifiers: set[str] = set()
    for part in mod_parts:
        mod = _MODIFIER_ALIASES.get(part.lower())
        if mod is None:
            raise ValueError(f"shortcut {combo!r}: {part!r} is not a modifier (Ctrl, Alt, Shift, Meta)")
        modifiers.add(mod)
    if key_part.lower() in _MODIFIER_ALIASES:
        raise ValueError(f"shortcut {combo!r} has no key, only modifiers")
    lowered = key_part.lower()
    if lowered in _NAMED_KEYS:
        key = _NAMED_KEYS[lowered]
    elif lowered in _KEY_ALIASES:
        key = _KEY_ALIASES[lowered]
    elif len(key_part) == 1:
        key = key_part.upper() if key_part.isalpha() else key_part
        if not key_part.isalpha():
            modifiers.discard("Shift")
    else:
        raise ValueError(f"shortcut {combo!r}: unknown key {key_part!r}")
    return "+".join([m for m in _MODIFIER_ORDER if m in modifiers] + [key])


def reserved_reason(canonical: str) -> str | None:
    """Why a canonical combo cannot be bound, or None if it can."""
    if canonical in _RESERVED:
        return f"{canonical} is reserved by the browser or OS and cannot be intercepted by a web page"
    return None


def validate_shortcut(combo: str) -> str:
    """Canonicalize and refuse reserved combos (the schema's validator)."""
    canonical = canonical_shortcut(combo)
    reason = reserved_reason(canonical)
    if reason:
        raise ValueError(reason)
    return canonical
