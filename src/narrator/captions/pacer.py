"""Typewriter-style caption output (`layout.pacing`), a port of
obs-localvocal's TokenBufferThread onto narrator's caption buffer.

Captions are revealed one unit at a time -- a character, a word or a
sentence -- instead of appearing the moment the ASR emits them. The reveal
speeds up with the backlog (char: 100/66/33 ms, word: 200/150/100 ms,
sentence: 1000/600/300 ms per unit), and after an idle stretch the first
unit appears at once, so pacing adds no delay to a caption that arrives
alone. A PARTIAL is typed out after all queued FINAL text; a newer PARTIAL
first pops the previous one's units that it does not share, and a FINAL
replaces the open partial (the units it already showed stay put, the rest
type out).

What is kept from narrator, deliberately: wrapping, `max_lines` and the
derived font size stay the CaptionBuffer's and the layout's (no
character-count line wrapping), and there is **no maxTime clear** -- the
caption retention rules (CLEAR prunes only off-screen history, WIPE is
opt-in) win over LocalVocal's blanking timer.

The pacer holds no thread: `advance(now)` is called by the renderer
whenever a frame is composed (the NDI loop runs at the output frame rate),
and returns the text to commit to the buffer and the in-progress tail to
show after it.

`smooth` (narrator's own, not LocalVocal's) types characters at a steady
rate instead of the stepped speeds: the rate follows the speaker's recent
speed (characters arrived over the last SMOOTH_RATE_WINDOW seconds) and is
nudged so the queue holds about SMOOTH_LAG seconds of text, catching up
over SMOOTH_CATCHUP seconds when it grows. Bursts of committed words then
flow out continuously instead of racing out and stopping. Added 2026-11-17
because the local-agreement decoder commits words in groups and `word`
pacing showed them as visible bursts.
"""
from __future__ import annotations

import re
from collections import deque

# Reveal interval per unit, by mode: (backlog above the second threshold,
# above the first, otherwise), and the thresholds in queued units.
_SPEEDS = {
    "char": ((0.033, 0.066, 0.100), (20, 40)),
    "word": ((0.100, 0.150, 0.200), (5, 10)),
    "sentence": ((0.300, 0.600, 1.000), (1, 3)),
}
# smooth: speaker-rate window (s), target queue (s of text), catch-up time
# (s) and the typing-rate bounds (characters per second).
SMOOTH_RATE_WINDOW = 6.0
SMOOTH_LAG = 0.8
SMOOTH_CATCHUP = 1.5
SMOOTH_RATE_BOUNDS = (8.0, 80.0)
_SENTENCE_RE = re.compile(r"\S.*?(?:[.!?…]+(?=\s|$)|$)", re.DOTALL)


def split_units(text: str, mode: str) -> list[str]:
    """`text` as reveal units; each unit carries its leading space, so
    "".join(units) == the normalized text."""
    text = " ".join(text.split())
    if not text:
        return []
    if mode in ("char", "smooth"):
        units: list[str] = []
        pending_space = False
        for ch in text:
            if ch == " ":
                pending_space = True
                continue
            units.append((" " if pending_space else "") + ch)
            pending_space = False
        return units
    if mode == "word":
        words = text.split(" ")
        return [words[0]] + [" " + w for w in words[1:]]
    sentences = [m.group(0) for m in _SENTENCE_RE.finditer(text)]
    return [sentences[0]] + [" " + s for s in sentences[1:]] if sentences else [text]


class CaptionPacer:
    def __init__(self, mode: str) -> None:
        if mode not in _SPEEDS and mode != "smooth":
            raise ValueError(f"unknown pacing mode {mode!r}")
        self.mode = mode
        self._speeds, self._thresholds = _SPEEDS.get(mode, _SPEEDS["char"])
        # smooth: characters queued since the last frame, and recent
        # arrivals as (time, characters) for the speaker-rate estimate
        self._arrived = 0
        self._arrivals: deque[tuple[float, int]] = deque()
        # Queued FINAL units; None marks the end of one FINAL (commit point).
        self._final_units: deque[str | None] = deque()
        self._typing = ""  # revealed text of the FINAL being typed, not yet committed
        self._partial: list[str] = []  # the latest partial's units
        self._partial_shown = 0  # how many of them are revealed
        self._commits: list[str] = []
        self._next_at = float("-inf")

    # -- input --------------------------------------------------------------

    def push_final(self, text: str) -> None:
        units = split_units(text, self.mode)
        if not self._final_units and self._partial_shown:
            # The open partial is on screen: what it shares with the FINAL
            # stays (revealed at once), the rest types out.
            same = _common_prefix(self._partial[: self._partial_shown], units)
            for unit in units[:same]:
                self._reveal_final(unit)
            units = units[same:]
        self._partial, self._partial_shown = [], 0
        self._final_units.extend(units)
        self._final_units.append(None)
        self._arrived += len(units)

    def push_partial(self, text: str) -> None:
        units = split_units(text, self.mode)
        self._arrived += max(len(units) - len(self._partial), 0)
        # Pop the shown units the new partial does not share.
        self._partial_shown = min(self._partial_shown, _common_prefix(self._partial, units))
        self._partial = units

    def reset(self) -> None:
        """Drop everything not yet committed (a screen clear)."""
        self._final_units.clear()
        self._typing = ""
        self._partial, self._partial_shown = [], 0
        self._commits = []
        self._next_at = float("-inf")

    # -- output -------------------------------------------------------------

    def backlog(self) -> int:
        return sum(1 for u in self._final_units if u is not None) + len(self._partial) - self._partial_shown

    def advance(self, now: float) -> tuple[list[str], str]:
        """Reveal what is due by `now`. Returns (texts to append to the
        caption buffer, in order; the tail to show after the buffer)."""
        if self.mode == "smooth":
            self._note_arrivals(now)
        interval = self._interval()
        if self._next_at < now - interval:
            self._next_at = now  # idle: the next unit is due at once
        while self.backlog() and now >= self._next_at:
            self._reveal_next()
            self._next_at += self._interval()
        # A FINAL's end marker right after its last unit commits it now.
        while self._final_units and self._final_units[0] is None:
            self._final_units.popleft()
            self._commit_typing(all_of_it=True)
        return self._take_commits(), self._tail()

    def flush(self) -> tuple[list[str], str]:
        """Reveal everything at once (pacing switched off)."""
        while self.backlog():
            self._reveal_next()
        while self._final_units:
            if self._final_units.popleft() is None:
                self._commit_typing(all_of_it=True)
        return self._take_commits(), self._tail()

    def _note_arrivals(self, now: float) -> None:
        """smooth: record what arrived since the last frame; forget
        arrivals older than the rate window."""
        if self._arrived:
            self._arrivals.append((now, self._arrived))
            self._arrived = 0
        while self._arrivals and self._arrivals[0][0] < now - SMOOTH_RATE_WINDOW:
            self._arrivals.popleft()

    def smooth_rate(self) -> float:
        """smooth: typing speed (characters/s) for the current backlog."""
        speaker = sum(n for _, n in self._arrivals) / SMOOTH_RATE_WINDOW
        rate = speaker + (self.backlog() - speaker * SMOOTH_LAG) / SMOOTH_CATCHUP
        low, high = SMOOTH_RATE_BOUNDS
        return min(max(rate, low), high)

    def _interval(self) -> float:
        backlog = self.backlog()
        if self.mode == "smooth":
            return 1.0 / self.smooth_rate()
        fast, medium, slow = self._speeds
        low, high = self._thresholds
        return fast if backlog > high else medium if backlog > low else slow

    def _reveal_next(self) -> None:
        while self._final_units and self._final_units[0] is None:
            self._final_units.popleft()
            self._commit_typing(all_of_it=True)
        if self._final_units:
            self._reveal_final(self._final_units.popleft())
        elif self._partial_shown < len(self._partial):
            self._partial_shown += 1

    def _reveal_final(self, unit: str) -> None:
        self._typing += unit
        self._commit_typing(all_of_it=False)

    def _commit_typing(self, *, all_of_it: bool) -> None:
        """Move complete words of the typed FINAL into the buffer (the
        buffer joins appended text with spaces, so only whole words go)."""
        if all_of_it:
            text, self._typing = self._typing.strip(), ""
        else:
            cut = self._typing.rfind(" ")
            if cut <= 0:
                return
            text, self._typing = self._typing[:cut].strip(), self._typing[cut + 1 :]
        if text:
            self._commits.append(text)

    def _take_commits(self) -> list[str]:
        commits, self._commits = self._commits, []
        return commits

    def _tail(self) -> str:
        partial = "".join(self._partial[: self._partial_shown]).strip()
        return " ".join(part for part in (self._typing.strip(), partial) if part)


def _common_prefix(a: list[str], b: list[str]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i].strip() == b[i].strip():
        i += 1
    return i
