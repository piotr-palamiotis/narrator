"""System tray icon: lets the operator hide the settings window (instead of
quitting) so the app keeps capturing/transcribing/sending NDI in the
background, with a way back to the window and a real Quit action.

Built lazily and defensively -- not every desktop environment this app might
run on has a tray host available, so callers should be prepared for
`start_tray_icon` to raise and fall back to normal close-on-X behavior (see
gui/app.py's `_on_close`).
"""
from __future__ import annotations

import threading
from typing import Callable

import pystray
from PIL import Image, ImageDraw

# Some backends (notably the X11/Xlib fallback used when no AppIndicator/GTK
# bindings are importable) can otherwise take up to 5s to report teardown
# when nothing ever docked the icon (e.g. a desktop with no systray host).
pystray.Icon.SETUP_THREAD_TIMEOUT = 2.0

_ICON_SIZE = 64
_ICON_BG = (30, 136, 229, 255)
_ICON_FG = (255, 255, 255, 255)


def _make_icon_image() -> Image.Image:
    """A simple filled circle with an "N" glyph -- generated at runtime so
    the tray icon doesn't need a bundled asset file.
    """
    image = Image.new("RGBA", (_ICON_SIZE, _ICON_SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.ellipse((2, 2, _ICON_SIZE - 2, _ICON_SIZE - 2), fill=_ICON_BG)

    from PIL import ImageFont

    try:
        font = ImageFont.load_default(size=int(_ICON_SIZE * 0.55))
    except TypeError:  # older Pillow: load_default() takes no size arg
        font = ImageFont.load_default()

    text = "N"
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(
        ((_ICON_SIZE - text_w) / 2 - bbox[0], (_ICON_SIZE - text_h) / 2 - bbox[1]),
        text,
        font=font,
        fill=_ICON_FG,
    )
    return image


def _build_icon(
    on_show: Callable[[], None],
    on_quit: Callable[[], None],
    on_activate: Callable[[], None],
) -> pystray.Icon:
    # The activation click (the only interaction some backends -- notably
    # pystray's Xlib/Xorg fallback used when no AppIndicator/GTK bindings are
    # importable -- ever deliver) is bound to a hidden default item that asks
    # the caller to show its own popup menu instead of acting directly. It's
    # `visible=False` so backends with a real native menu (GTK/AppIndicator)
    # don't render a redundant blank entry alongside the visible items below,
    # but `Menu.__call__` still finds and runs it by scanning *all* items
    # (visible or not) for the one marked `default`, so activation keeps
    # working the same way on every backend.
    menu = pystray.Menu(
        pystray.MenuItem("", lambda icon, item: on_activate(), default=True, visible=False),
        pystray.MenuItem("Show Narrator", lambda icon, item: on_show()),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Close Narrator", lambda icon, item: on_quit()),
    )
    return pystray.Icon("narrator", _make_icon_image(), "Narrator", menu)


def start_tray_icon(
    on_show: Callable[[], None],
    on_quit: Callable[[], None],
    on_activate: Callable[[], None],
) -> pystray.Icon:
    """Builds and starts the tray icon on its own thread(s), returning it
    (call `.stop()` to tear it down later). All three callbacks are invoked
    from pystray's thread, not the Tk thread -- callers must marshal back to
    the GUI thread themselves (e.g. via `root.after(0, ...)`) before touching
    any Tkinter widget.

    `on_activate` fires on plain icon activation (e.g. a left click) on
    every backend, including pystray's Xlib/Xorg fallback, which has no
    concept of a real menu and can't deliver right-clicks at all -- so it's
    the one hook guaranteed to work everywhere. Callers should use it to pop
    up their own menu (e.g. a Tk context menu) offering show/close, since
    `on_show`/`on_quit` only get wired to visible native menu items, which
    are reachable solely on backends that do support a real menu
    (GTK/AppIndicator).

    Threads spawned during startup are forced to be daemon threads (see
    `_daemon_threads_during`) -- confirmed empirically: pystray 0.19's Xorg
    backend's `run_detached()` calls its own internal `_start_setup()` twice
    (once directly, once again inside the detached thread's `run()`), and
    since exactly one `_mark_ready()` call ever unblocks it, the loser is a
    non-daemon thread that blocks forever and would otherwise wedge process
    exit -- e.g. on a desktop with no systray host to dock into. Without a
    real tray host to dock into, the icon simply won't be visible; the
    caller's own fallback (falling back to normal close-on-X if this raises)
    only covers construction failures, not a silently-invisible icon.
    """
    icon = _build_icon(on_show, on_quit, on_activate)
    with _daemon_threads_during():
        icon.run_detached()
    return icon


class _daemon_threads_during:
    """Context manager: while active, `threading.Thread(...)` calls default
    to `daemon=True` unless the caller passes `daemon` explicitly. Scoped
    narrowly around one library call so it can't affect unrelated code.
    """

    def __enter__(self) -> None:
        self._original_init = threading.Thread.__init__

        def _daemon_init(thread_self, *args, **kwargs):
            kwargs.setdefault("daemon", True)
            self._original_init(thread_self, *args, **kwargs)

        threading.Thread.__init__ = _daemon_init

    def __exit__(self, *exc_info: object) -> None:
        threading.Thread.__init__ = self._original_init
