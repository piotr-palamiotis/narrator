"""Event vocabulary (`asr.vocabulary`): names, places and jargon for an
event, used two ways.

  * Prompt hint: the backend puts the list at the front of whisper's
    initial_prompt on every decode, ahead of the prompt carry-over
    (`build_initial_prompt`).
  * Optional fuzzy snapping (`asr.vocabulary_snap`, off by default): a
    hypothesis word within a small edit distance of a list word is replaced
    by it (`VocabularySnapper`). Runs before agreement, like text_filters.

Named per-event lists live as `<config dir>/vocabulary/<name>.txt` (one entry
per line, `#` comments), selected by `asr.vocabulary_list` and merged after
the inline `asr.vocabulary` entries.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

from narrator.captions.blocklist import read_word_list
from narrator.captions.textnorm import fold

logger = logging.getLogger(__name__)

VOCABULARY_DIRNAME = "vocabulary"
_LIST_NAME_RE = re.compile(r"^[\w][\w .-]*$")

# whisper.cpp keeps at most n_text_ctx/2 = 224 prompt tokens and, when the
# prompt is longer, drops tokens from the FRONT -- exactly where the
# vocabulary sits. So the prompt is sized here, carry-over truncated first.
_PROMPT_MAX_TOKENS = 223
# Token count is ESTIMATED from characters: pywhispercpp 1.5.1 binds
# whisper_tokenize with the output buffer typed as a plain int (no usable
# pointer), so the model's tokenizer can't be called cheaply or safely.
# 1.5 chars/token is conservative for whisper's BPE on Polish (diacritics
# and names split into short tokens; ordinary text runs ~3 chars/token).
_CHARS_PER_TOKEN = 1.5
PROMPT_MAX_CHARS = int(_PROMPT_MAX_TOKENS * _CHARS_PER_TOKEN)

# Snapping: only words of at least this many letters, on either side.
SNAP_MIN_CHARS = 5


def is_valid_list_name(name: str) -> bool:
    return bool(_LIST_NAME_RE.match(name)) and ".." not in name


def default_vocabulary_dir() -> Path:
    from platformdirs import user_config_dir

    return Path(user_config_dir("narrator")) / VOCABULARY_DIRNAME


def _list_path(name: str, directory: Path) -> Path:
    if not is_valid_list_name(name):
        raise ValueError(f"invalid vocabulary list name {name!r}")
    return directory / f"{name}.txt"


def list_vocabulary_lists(directory: Path) -> list[str]:
    """Names of the saved lists in `directory`, sorted."""
    if not directory.is_dir():
        return []
    return sorted(p.stem for p in directory.glob("*.txt") if is_valid_list_name(p.stem))


def load_vocabulary_list(name: str, directory: Path) -> list[str]:
    """Entries of list `name`. Raises OSError if it can't be read."""
    return read_word_list(_list_path(name, directory))


def save_vocabulary_list(name: str, entries: list[str], directory: Path) -> Path:
    """Write list `name` atomically (temp file, then rename)."""
    path = _list_path(name, directory)
    directory.mkdir(parents=True, exist_ok=True)
    lines = [e.strip() for e in entries if e.strip()]
    tmp = path.with_suffix(".txt.tmp")
    tmp.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    tmp.replace(path)
    return path


def merge_vocabulary(*lists: list[str]) -> list[str]:
    """Concatenate, dropping blanks and case/diacritics-blind duplicates
    (first spelling wins)."""
    seen, merged = set(), []
    for entries in lists:
        for entry in entries:
            entry = entry.strip()
            key = fold(entry)
            if entry and key not in seen:
                seen.add(key)
                merged.append(entry)
    return merged


def resolve_vocabulary(asr_config, directory: Path | None = None) -> list[str]:
    """The inline `asr.vocabulary` merged with the selected named list. A
    missing/unreadable list is logged and skipped -- it must never stop
    captioning."""
    named: list[str] = []
    if asr_config.vocabulary_list:
        directory = directory if directory is not None else default_vocabulary_dir()
        try:
            named = load_vocabulary_list(asr_config.vocabulary_list, directory)
        except (OSError, ValueError) as exc:
            logger.warning("vocabulary list %r not loaded: %s", asr_config.vocabulary_list, exc)
    return merge_vocabulary(asr_config.vocabulary, named)


def build_initial_prompt(vocabulary: list[str], carryover: str | None) -> str:
    """whisper's initial_prompt: the vocabulary hint, then the carry-over.

    With no vocabulary the carry-over passes through untouched (whisper.cpp's
    own front-truncation is right for it). With a vocabulary, the total is
    kept within PROMPT_MAX_CHARS: the carry-over is cut first (keeping its
    most recent words), then trailing vocabulary entries.
    """
    carryover = (carryover or "").strip()
    if not vocabulary:
        return carryover
    entries: list[str] = []
    length = 0
    for entry in vocabulary:
        extra = len(entry) + (2 if entries else 1)  # ", " separator / final "."
        if length + extra > PROMPT_MAX_CHARS:
            logger.debug("vocabulary prompt full; %d entries left out", len(vocabulary) - len(entries))
            break
        entries.append(entry)
        length += extra
    hint = ", ".join(entries) + "." if entries else ""
    room = PROMPT_MAX_CHARS - len(hint) - 1
    if carryover and room > 0:
        if len(carryover) > room:
            tail = carryover[-room:]
            cut = tail.find(" ")
            carryover = tail[cut + 1 :] if cut >= 0 else ""
        if carryover:
            return f"{hint} {carryover}" if hint else carryover
    return hint


def _edit_distance(a: str, b: str, limit: int) -> int:
    """Levenshtein distance, or limit+1 as soon as it must exceed `limit`."""
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        if min(current) > limit:
            return limit + 1
        previous = current
    return previous[-1]


def _snap_limit(length: int) -> int:
    return 1 if length < 9 else 2


_WORD_RE = re.compile(r"[^\W_]+")


class VocabularySnapper:
    """Replace near-miss spellings of vocabulary words with the list's
    spelling: "Kowalsky" -> "Kowalski", "Wroclaw" -> "Wrocław".

    Multi-word entries snap word by word. Only words of SNAP_MIN_CHARS+
    letters on both sides are considered; the allowed distance (on folded,
    diacritics-blind keys) is 1, or 2 from 9 letters. A word that one key
    is a prefix of the other ("Kowalskiego" vs "Kowalski") is left alone:
    that is Polish inflection, and snapping it would un-inflect the name.
    """

    def __init__(self, vocabulary: list[str]) -> None:
        targets: dict[str, str] = {}
        for entry in vocabulary:
            for word in _WORD_RE.findall(entry):
                if len(word) >= SNAP_MIN_CHARS:
                    targets.setdefault(fold(word), word)
        self._targets = list(targets.items())
        self._cache: dict[str, str] = {}

    def __bool__(self) -> bool:
        return bool(self._targets)

    def snap_word(self, word: str) -> str:
        if len(word) < SNAP_MIN_CHARS:
            return word
        cached = self._cache.get(word)
        if cached is not None:
            return cached
        key = fold(word)
        best, best_distance = word, None
        limit = _snap_limit(len(key))
        for target_key, target in self._targets:
            if target_key == key:
                best, best_distance = target, 0
                break
            if target_key.startswith(key) or key.startswith(target_key):
                continue
            distance = _edit_distance(key, target_key, limit)
            if distance <= limit and (best_distance is None or distance < best_distance):
                best, best_distance = target, distance
        if len(self._cache) < 10000:
            self._cache[word] = best
        return best

    def apply(self, text: str) -> str:
        if not self._targets or not text:
            return text
        return _WORD_RE.sub(lambda m: self.snap_word(m.group()), text)
