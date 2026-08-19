"""NDI sending: wraps cyndilib to publish the composited RGBA caption canvas
as an NDI video source, and drives the fixed-cadence render+send loop.

Frames are sent every tick at a constant cadence -- including fully
transparent idle frames -- since NDI receivers generally expect steady frame
delivery; irregular cadence or silently stopping when idle risks a receiver
flagging the source as frozen/stale. The (relatively expensive) Pillow
compositing in CaptionRenderer.get_frame() is skipped when nothing visible
has changed; this loop still calls write_video() every tick regardless,
resending the cached bytes.
"""
from __future__ import annotations

import logging
import threading
import time
from fractions import Fraction

import numpy as np
from cyndilib import FourCC, Sender, VideoSendFrame

from narrator.config.schema import CanvasConfig, NdiConfig
from narrator.render.canvas import CaptionRenderer

logger = logging.getLogger(__name__)


class NdiCaptionSender:
    def __init__(self, config: NdiConfig, canvas: CanvasConfig) -> None:
        self._sender = Sender(ndi_name=config.source_name)
        self._video_frame = VideoSendFrame()
        self._video_frame.set_resolution(canvas.width_px, canvas.height_px)
        self._video_frame.set_fourcc(FourCC.RGBA)
        self._video_frame.set_frame_rate(Fraction(config.fps).limit_denominator(1000))
        self._sender.set_video_frame(self._video_frame)
        self._sender.open()

    def send_frame(self, rgba_bytes: np.ndarray) -> None:
        self._sender.write_video(rgba_bytes)

    def close(self) -> None:
        self._sender.close()


def run_render_send_loop(
    renderer: CaptionRenderer,
    sender: NdiCaptionSender,
    fps: float,
    stop_event: threading.Event,
) -> None:
    """Constant-cadence loop: composite (if changed) -> send, every tick,
    paced by a monotonic-clock target so compositing time doesn't accumulate
    drift over a long-running session.
    """
    period = 1.0 / fps
    next_tick = time.monotonic()
    while not stop_event.is_set():
        frame = renderer.get_frame()
        # cyndilib requires a writable buffer; np.asarray() on a PIL Image
        # yields a read-only view, so a copy (np.array(...)) is required here.
        rgba_bytes = np.array(frame, dtype=np.uint8).reshape(-1)
        try:
            sender.send_frame(rgba_bytes)
        except Exception:
            logger.exception("Failed to send NDI frame; continuing")

        next_tick += period
        sleep_for = next_tick - time.monotonic()
        if sleep_for > 0:
            stop_event.wait(sleep_for)
        else:
            next_tick = time.monotonic()  # fell behind; resync instead of spinning to catch up
