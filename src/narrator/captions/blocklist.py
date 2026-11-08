"""Profanity blocklist (`captions.blocklist`): words that must never reach
the screen.

Matching is whole-word and case/diacritics-blind (see textnorm.fold). An
entry ending in `*` is a stem: it matches every word that STARTS with it, so
`kurw*` covers Polish inflection (kurwa, kurwy, kurwą, kurwami) -- but never
a word that merely contains it ("skurwysyn" needs its own `skurw*`), so an
innocent longer word cannot be hit from the inside. Words are runs of
letters/digits: hyphens and apostrophes separate them.

The filter is a standalone function of text, applied by the StreamingDecoder
to each hypothesis (next to text_filters, before agreement) and by
AppController to manual captions. The backend's raw output never passes
through it, so a recording path that needs the true transcript (draft.srt)
simply doesn't apply it.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from narrator.captions.textnorm import fold
from narrator.config.schema import BlocklistAction

logger = logging.getLogger(__name__)

BUILTIN_BLOCKLIST_PATH = Path(__file__).with_name("data") / "blocklist_starter.txt"

_WORD_RE = re.compile(r"[^\W_]+")
_SENTENCE_END_RE = re.compile(r"[.!?…]+$")
_SPACES_RE = re.compile(r"\s+")


def read_word_list(path: Path) -> list[str]:
    """One entry per line; blank lines and `#` comments ignored."""
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            entries.append(line)
    return entries


def mask_word(word: str) -> str:
    """First and last letter kept, the rest starred: "kurwa" -> "k***a"."""
    if len(word) <= 1:
        return "*"
    if len(word) == 2:
        return word[0] + "*"
    return word[0] + "*" * (len(word) - 2) + word[-1]


@dataclass(frozen=True)
class Blocklist:
    """A compiled blocklist. Picklable (sent to the ASR worker process)."""

    exact: frozenset[str] = frozenset()
    prefixes: tuple[str, ...] = ()
    action: BlocklistAction = BlocklistAction.MASK
    replacement: str = "***"
    _cache: dict = field(default_factory=dict, compare=False, repr=False)

    @classmethod
    def from_entries(
        cls,
        entries: list[str],
        action: BlocklistAction = BlocklistAction.MASK,
        replacement: str = "***",
    ) -> "Blocklist":
        exact, prefixes = set(), set()
        for entry in entries:
            entry = entry.strip()
            if entry.endswith("*"):
                prefixes.add(fold(entry[:-1]))
            elif entry:
                exact.add(fold(entry))
        return cls(frozenset(exact), tuple(sorted(prefixes)), BlocklistAction(action), replacement)

    def __bool__(self) -> bool:
        return bool(self.exact or self.prefixes)

    def is_blocked(self, word: str) -> bool:
        key = fold(word)
        hit = self._cache.get(key)
        if hit is None:
            hit = key in self.exact or any(key.startswith(p) for p in self.prefixes)
            if len(self._cache) < 10000:
                self._cache[key] = hit
        return hit

    def apply(self, text: str) -> str:
        """`text` with every blocked word masked, replaced or dropped."""
        if not self or not text:
            return text
        if self.action is BlocklistAction.DROP:
            return self._drop(text)
        if self.action is BlocklistAction.REPLACE:
            sub = lambda m: self.replacement if self.is_blocked(m.group()) else m.group()  # noqa: E731
        else:
            sub = lambda m: mask_word(m.group()) if self.is_blocked(m.group()) else m.group()  # noqa: E731
        return _WORD_RE.sub(sub, text)

    def _drop(self, text: str) -> str:
        """Remove blocked words. A whitespace token left with no letters is
        removed too ("kurwa," leaves no stray comma), but sentence-ending
        punctuation moves onto the previous word so the sentence still ends."""
        out: list[str] = []
        for token in text.split():
            kept = _WORD_RE.sub(lambda m: "" if self.is_blocked(m.group()) else m.group(), token)
            if kept == token:
                out.append(token)
            elif _WORD_RE.search(kept):
                out.append(kept)
            else:
                end = _SENTENCE_END_RE.search(kept)
                if end and out and not _SENTENCE_END_RE.search(out[-1]):
                    out[-1] += end.group()
        return _SPACES_RE.sub(" ", " ".join(out)).strip()


def build_blocklist(
    entries: list[str],
    *,
    builtin: bool = False,
    action: BlocklistAction | str = BlocklistAction.MASK,
    replacement: str = "***",
) -> Blocklist:
    """Compile the operator's entries, plus the bundled starter list when
    `builtin` is set."""
    all_entries = list(entries)
    if builtin:
        try:
            all_entries += read_word_list(BUILTIN_BLOCKLIST_PATH)
        except OSError:
            logger.exception("bundled blocklist %s could not be read", BUILTIN_BLOCKLIST_PATH)
    return Blocklist.from_entries(all_entries, BlocklistAction(action), replacement)


def blocklist_from_config(config) -> Blocklist:
    """Compile a `CaptionsConfig`."""
    return build_blocklist(
        config.blocklist,
        builtin=config.blocklist_builtin,
        action=config.blocklist_action,
        replacement=config.blocklist_replacement,
    )


def apply_blocklist(text: str, blocklist: Blocklist | None) -> str:
    """Standalone filter: `text` as it may air. None = unfiltered."""
    return blocklist.apply(text) if blocklist else text
