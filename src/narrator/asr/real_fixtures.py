"""Real-recording fixtures for autoconfigure: any audio file + matching SRT.

A fixture is a pair `name.<audio-ext>` + `name.srt` in one directory
(`scripts/test_audio/` by default; drop files in, no code changes). The SRT
is a live-caption transcript of the audio, which brings three artifacts the
clean Piper fixtures never have -- this module is the single place that
handles them, shared by the autoconfigure evaluation and the eval scripts:

  * ROLLUP repeats: live feeds re-show the previous line plus new words, so
    naive concatenation double-counts every transition word. Collapsed by
    dropping each cue's longest prefix that duplicates the running tail.
  * Non-speech annotations: "[Muzyka]"/"[Aplauz]"/"(Applause)" cues mark
    music/applause. Dropped from the reference AND used as windows where
    hypothesis words are expected to be hallucination, not missed speech.
  * Foreign-language speech: whisper forced to `pl` TRANSLATES foreign
    songs/speech instead of transcribing, which must not count as errors.
    Detected by DIACRITIC DENSITY (Polish prose runs ~8-12% of words with
    ą/ć/ę/ł/ń/ó/ś/ź/ż; English runs 0%) over groups of consecutive cues --
    a word list missed "cant remember color shining through" on the real
    recording this was built from; density does not.

Reference words carry an estimated TIME (linear interpolation over the span
of the cue that introduced them), so an excerpt's reference is the words
spoken inside it. Taking every cue that merely overlapped the window leaked
up to a cue's worth of words in at each edge -- YouTube auto-caption cues
run 5-9s.

Subtitle files downloaded with yt-dlp carry their language in the name:
"Title (Polish_ASR).srt" is YouTube's auto-captions, "Title (Polish).srt" an
uploaded (human) subtitle track. `RealRecording.reference_language` /
`reference_kind` expose that, so evaluation can skip e.g. an English
translation track that sits next to Polish audio.

Audio is decoded through a GStreamer PIPE (there is no ffmpeg in this
repo's environments): decodebin -> 16kHz mono S16LE -> stdout, and nothing
is written to disk. The previous temp-WAV route leaked a whole-recording
file into /tmp whenever the process died before cleanup (a timed-out probe
is SIGKILLed), and the eval scripts' copy never cleaned up at all -- 41
leaked 170 MB files filled the root disk.

Excerpts: recordings are long, so evaluation scores fixed-length windows
placed deterministically inside speech-dense regions -- no RNG, two runs
pick the same windows.
"""
from __future__ import annotations

import hashlib
import logging
import re
import subprocess
import threading
import unicodedata
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000

# Audio extensions accepted for fixtures. Anything GStreamer's decodebin can
# open works (aac/mp3/wav/m4a/ogg/opus/flac/webm); the SRT side is always
# `.srt`. A file with no matching SRT is skipped, not an error -- half-
# prepared fixture folders stay usable.
AUDIO_EXTENSIONS = {".aac", ".mp3", ".wav", ".m4a", ".ogg", ".opus", ".flac", ".webm"}

# Untagged subtitle files (no "(Language)" suffix) are taken as Polish: the
# hall recordings this module was first built for are Polish live captions.
DEFAULT_REFERENCE_LANGUAGE = "pl"

_NON_SPEECH_RE = re.compile(r"^[\[\(].*[\]\)]$")
_SRT_TIMING_RE = re.compile(r"(\d\d:\d\d:\d\d,\d\d\d)\s*-->\s*(\d\d:\d\d:\d\d,\d\d\d)")
_TRAILING_GROUP_RE = re.compile(r"\(([^()]*)\)\s*$")
# "(Polish_ASR)" / "(English)": letters only, optional _ASR suffix. A
# trailing group with digits or other marks ("(2009)", "(128kbit_AAC)") is
# part of the title, not a language tag.
_LANGUAGE_TAG_RE = re.compile(r"^\s*([A-Za-z][A-Za-z \-]*?)(_ASR)?\s*$")

# Foreign-language detection thresholds (rationale in the module docstring):
# a group needs enough words for density to mean anything, splits at caption
# gaps (songs sit between applause breaks) and at a max group length.
_POLISH_CHARS = set("ąćęłńóśźż")
_FOREIGN_GROUP_MAX_S = 90.0
_FOREIGN_GROUP_SPLIT_GAP_S = 3.0
_FOREIGN_MIN_GROUP_WORDS = 12
_FOREIGN_MAX_DIACRITIC_FRACTION = 0.02

# Excerpt edges move at most this far to land on a cue start -- the moment a
# new caption line (a new run of words) begins -- so an excerpt rarely cuts
# a word in half on either the audio or the reference side.
_SNAP_TOLERANCE_S = 3.0

_GST_PCM_CAPS = f"audio/x-raw,format=S16LE,channels=1,rate={SAMPLE_RATE}"
_PIPE_READ_BYTES = 1 << 20  # ~33s of 16kHz S16 mono per read
# A whole 3.5-hour film decodes in well under a minute; this only guards
# against a wedged decoder pipeline.
_DECODE_TIMEOUT_S = 900.0


def parse_reference_tag(srt_stem: str) -> tuple[str | None, str]:
    """(language_code, kind) from a subtitle file's trailing "(...)" tag.

    kind is "asr" (auto-captions, "_ASR" suffix), "human" (a plain language
    name: an uploaded subtitle track) or "unknown" (no language tag; taken
    as DEFAULT_REFERENCE_LANGUAGE). A tag shaped like a language name that
    whisper does not know ("Manx") returns language None -- such a file
    cannot be scored, and evaluation excludes it.
    """
    from narrator.asr.whisper_languages import language_code_for_name

    group = _TRAILING_GROUP_RE.search(srt_stem)
    if group is not None:
        tag = _LANGUAGE_TAG_RE.match(group.group(1))
        if tag is not None:
            kind = "asr" if tag.group(2) else "human"
            return language_code_for_name(tag.group(1)), kind
    return DEFAULT_REFERENCE_LANGUAGE, "unknown"


@dataclass(frozen=True)
class RealRecording:
    """One audio+SRT fixture pair."""

    audio_path: Path
    srt_path: Path

    @property
    def stem(self) -> str:
        return self.audio_path.stem

    @property
    def title(self) -> str:
        """Base title shared by the audio and SRT names (tags stripped)."""
        return _normalize_recording_stem(self.audio_path.stem)

    @property
    def reference_language(self) -> str | None:
        return parse_reference_tag(self.srt_path.stem)[0]

    @property
    def reference_kind(self) -> str:
        return parse_reference_tag(self.srt_path.stem)[1]


@dataclass(frozen=True)
class Cue:
    """One SRT cue (absolute seconds)."""

    start: float
    end: float
    text: str


@dataclass(frozen=True)
class RecordingReference:
    """The scoring-ready reference for one recording."""

    # union of Polish-speech cue spans (post foreign/non-speech exclusion)
    speech_spans: list[tuple[float, float]]
    # [Muzyka]/[Aplauz]-style windows: hypothesis words here are music
    # hallucination, not missed speech
    non_speech_windows: list[tuple[float, float]]
    # excluded foreign-speech windows (songs etc.) -- masked like non-speech
    foreign_windows: list[tuple[float, float]]
    duration_s: float
    # rollup-collapsed in-scope words, each with its estimated spoken time
    timed_words: list[tuple[float, str]] = field(default_factory=list)
    # start times of the in-scope speech cues (excerpt edges snap to these)
    cue_starts: list[float] = field(default_factory=list)

    @property
    def mask_windows(self) -> list[tuple[float, float]]:
        return self.non_speech_windows + self.foreign_windows


def _normalize_recording_stem(stem: str) -> str:
    """Strip trailing parenthetical suffix (codec / language tag) for matching.

    Audio files carry codec-tagged suffixes like ``(128kbit_AAC)`` or
    ``(152kbit_Opus)``; SRTs carry language tags like ``(Polish_ASR)``
    or ``(English)`` — so a same-title pair's literal stems never align.
    This strips the final ``(...)`` group from both sides so they match by
    base title. Files with no suffix (e.g. ``school_assembly.wav``) are
    returned unchanged.
    """
    return re.sub(r"\s*\([^)]*\)\s*$", "", stem).strip()


def discover_recordings(directory: Path) -> list[RealRecording]:
    """All audio files in `directory` that have a matching `.srt` alongside.

    Matching is by *base title*: trailing parenthetical suffixes are
    stripped from both sides before comparison (audio files carry codec
    tags like ``(128kbit_AAC)`` while SRTs carry language tags like
    ``(Polish_ASR)``; stripping both surfaces the shared base title). A
    file with no matching SRT is skipped, not an error -- half prepared
    fixture folders stay usable.
    """
    if not directory.is_dir():
        return []
    # Build SRT lookup by normalized stem (suffix-stripped base title).
    srt_by_stem: dict[str, Path] = {}
    for p in sorted(directory.iterdir()):
        if p.suffix.lower() == ".srt" and p.is_file():
            srt_by_stem[_normalize_recording_stem(p.stem)] = p

    recordings: list[RealRecording] = []
    for audio_path in sorted(directory.iterdir()):
        if audio_path.suffix.lower() not in AUDIO_EXTENSIONS or not audio_path.is_file():
            continue
        normalized = _normalize_recording_stem(audio_path.stem)
        srt_path = srt_by_stem.get(normalized)
        if srt_path is None:
            logger.warning(
                "real-recording fixture %s has no matching .srt "
                "(base title '%s') -- skipped",
                audio_path.name,
                normalized,
            )
            continue
        recordings.append(RealRecording(audio_path=audio_path, srt_path=srt_path))
    return recordings


def _parse_srt_timestamp(stamp: str) -> float:
    hh, mm, rest = stamp.split(":")
    ss, ms = rest.split(",")
    return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ms) / 1000.0


def parse_srt(path: Path) -> list[Cue]:
    """Minimal SRT parser: index line, timing line, text lines (possibly multi)."""
    blocks = re.split(r"\n\s*\n", path.read_text(encoding="utf-8-sig").strip())
    cues: list[Cue] = []
    for block in blocks:
        lines = [ln for ln in block.strip().splitlines() if ln.strip()]
        if len(lines) < 2:
            continue
        match = _SRT_TIMING_RE.search(lines[1])
        if match is None:
            continue
        cues.append(
            Cue(
                start=_parse_srt_timestamp(match.group(1)),
                end=_parse_srt_timestamp(match.group(2)),
                text=" ".join(lines[2:]).strip(),
            )
        )
    return cues


def strip_punctuation(word: str) -> str:
    """Lowercase + punctuation removal by Unicode category (P*): Polish
    diacritics pass through -- "gwiazdę" vs "gwiazde" stays a visible error."""
    return "".join(ch for ch in word.lower() if not unicodedata.category(ch).startswith("P"))


def words(text: str) -> list[str]:
    return [w for w in (strip_punctuation(tok) for tok in text.split()) if w]


def _split_cue_groups(cues: list[Cue]) -> list[list[Cue]]:
    """Consecutive speech cues grouped for language detection: split at caption
    gaps (songs sit between applause breaks) and at a max group length."""
    groups: list[list[Cue]] = []
    current: list[Cue] = []
    for cue in cues:
        if current and (
            cue.start - current[-1].end > _FOREIGN_GROUP_SPLIT_GAP_S
            or cue.end - current[0].start > _FOREIGN_GROUP_MAX_S
        ):
            groups.append(current)
            current = []
        current.append(cue)
    if current:
        groups.append(current)
    return groups


def _is_foreign_group(group: list[Cue]) -> bool:
    """Diacritic-density test; see the threshold constants' rationale."""
    group_words = [w for cue in group for w in words(cue.text)]
    if len(group_words) < _FOREIGN_MIN_GROUP_WORDS:
        return False
    marked = sum(1 for w in group_words if any(ch in _POLISH_CHARS for ch in w))
    return marked / len(group_words) <= _FOREIGN_MAX_DIACRITIC_FRACTION


def _split_reference(
    cues: list[Cue],
) -> tuple[list[Cue], list[tuple[float, float]], list[tuple[float, float]]]:
    """(in-scope speech cues, non_speech_windows, foreign_windows)."""
    speech: list[Cue] = []
    non_speech: list[tuple[float, float]] = []
    for cue in cues:
        if _NON_SPEECH_RE.match(cue.text.strip()):
            non_speech.append((cue.start, cue.end))
            continue
        speech.append(cue)

    kept: list[Cue] = []
    foreign: list[tuple[float, float]] = []
    for group in _split_cue_groups(speech):
        if _is_foreign_group(group):
            foreign.append((group[0].start, group[-1].end))
            continue
        kept.extend(group)
    return kept, non_speech, foreign


def _timed_words(kept: list[Cue]) -> list[tuple[float, str]]:
    """Rollup-collapsed words of the in-scope cues, each with an estimated
    spoken time.

    A cue's words are spread evenly over its EFFECTIVE span: from its start
    to the next cue's start (rollup cues stay on screen while the following
    line is already being spoken), or its own end when that comes first.
    A rollup echo -- the longest prefix duplicating the running tail -- is
    dropped, so it is neither counted twice nor timed twice.
    """
    out: list[tuple[float, str]] = []
    tail: list[str] = []
    for idx, cue in enumerate(kept):
        cue_words = words(cue.text)
        if not cue_words:
            continue
        overlap = 0
        if tail:
            max_overlap = min(len(cue_words), len(tail), 15)
            for k in range(max_overlap, 0, -1):
                if tail[-k:] == cue_words[:k]:
                    overlap = k
                    break
        next_start = kept[idx + 1].start if idx + 1 < len(kept) else cue.end
        end = min(cue.end, next_start) if next_start > cue.start else cue.end
        span = max(end - cue.start, 0.0)
        count = len(cue_words)
        for j in range(overlap, count):
            out.append((cue.start + span * (j + 0.5) / count, cue_words[j]))
            tail.append(cue_words[j])
    return out


def collapse_reference_cues(cues: list[Cue]) -> tuple[list[str], list[tuple[float, float]], list[tuple[float, float]]]:
    """Flatten Polish-speech cues into one word list with rollup echoes
    collapsed; returns (words, non_speech_windows, foreign_windows)."""
    kept, non_speech, foreign = _split_reference(cues)
    return [w for _, w in _timed_words(kept)], non_speech, foreign


def load_reference(srt_path: Path) -> RecordingReference:
    """Parse an SRT into the scoring-ready reference structure."""
    cues = parse_srt(srt_path)
    kept, non_speech, foreign = _split_reference(cues)
    timed = _timed_words(kept)
    duration_s = max((c.end for c in cues), default=0.0)
    # Polish-speech spans: cue spans minus the masked windows (union-merged
    # so excerpt sampling sees contiguous talk, not rollup fragments).
    masked = sorted(non_speech + foreign)
    spans: list[tuple[float, float]] = []
    for cue in speech_cue_spans(cues):
        s, e = cue
        for ms, me in masked:
            if me <= s or ms >= e:
                continue
            # clipped by a masked window; keep the visible remainder(s)
            if ms > s:
                spans.append((s, min(ms, e)))
            s = max(s, me)
            if s >= e:
                break
        if s < e:
            spans.append((s, e))
    logger.debug(
        "%s: %d reference words, %.1f min speech, %d non-speech + %d foreign windows",
        srt_path.stem,
        len(timed),
        sum(e - s for s, e in _merge_spans(spans)) / 60.0,
        len(non_speech),
        len(foreign),
    )
    return RecordingReference(
        speech_spans=_merge_spans(spans),
        non_speech_windows=non_speech,
        foreign_windows=foreign,
        duration_s=duration_s,
        timed_words=timed,
        cue_starts=sorted(c.start for c in kept),
    )


def speech_cue_spans(cues: list[Cue]) -> list[tuple[float, float]]:
    """Every non-annotation cue's span (pre foreign-detection; used to build
    the speech union before masked-window subtraction)."""
    return [(c.start, c.end) for c in cues if not _NON_SPEECH_RE.match(c.text.strip())]


def _merge_spans(spans: list[tuple[float, float]], gap: float = 0.5) -> list[tuple[float, float]]:
    out: list[list[float]] = []
    for s, e in sorted(spans):
        if out and s <= out[-1][1] + gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


# ---------------------------------------------------------------------------
# Audio: GStreamer pipe, no temp files
# ---------------------------------------------------------------------------


def _gst_pcm_command(path: Path) -> list[str]:
    return [
        "gst-launch-1.0", "-q", "filesrc", f"location={path}", "!", "decodebin",
        "!", "audioconvert", "!", "audioresample", "!", _GST_PCM_CAPS,
        "!", "fdsink", "fd=1",
    ]


def read_pcm_windows(
    path: Path,
    windows: Sequence[tuple[float, float]],
    timeout_s: float = _DECODE_TIMEOUT_S,
) -> list[np.ndarray]:
    """Decode `path` and return the int16 16kHz mono samples of each
    (start_s, end_s) window, in the order given.

    The decoder streams into a pipe; only the requested windows are kept,
    and decoding stops once the last window is complete. Memory is the
    windows themselves and disk use is zero. A window past the end of the
    audio comes back short (possibly empty) rather than raising.
    """
    bounds = [
        (max(0, int(round(s * SAMPLE_RATE))), max(0, int(round(e * SAMPLE_RATE))))
        for s, e in windows
    ]
    last = max((e for _, e in bounds), default=0)
    parts: list[list[np.ndarray]] = [[] for _ in bounds]
    if last == 0:
        return [np.zeros(0, dtype=np.int16) for _ in bounds]

    proc = subprocess.Popen(
        _gst_pcm_command(path), stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    stderr_tail: deque[str] = deque(maxlen=10)

    def _drain_stderr() -> None:
        assert proc.stderr is not None
        for line in proc.stderr:
            stderr_tail.append(line.decode("utf-8", "replace").rstrip())

    drain = threading.Thread(target=_drain_stderr, daemon=True)
    drain.start()
    watchdog = threading.Timer(timeout_s, proc.kill)
    watchdog.start()
    position = 0  # samples consumed so far
    carry = b""
    reached_eof = False
    try:
        assert proc.stdout is not None
        while position < last:
            chunk = proc.stdout.read(_PIPE_READ_BYTES)
            if not chunk:
                reached_eof = True
                break
            data = carry + chunk
            usable = len(data) - (len(data) % 2)
            carry = data[usable:]
            samples = np.frombuffer(data[:usable], dtype=np.int16)
            count = len(samples)
            for i, (s, e) in enumerate(bounds):
                lo, hi = max(s, position), min(e, position + count)
                if lo < hi:
                    parts[i].append(samples[lo - position : hi - position].copy())
            position += count
    finally:
        watchdog.cancel()
        if reached_eof:
            # The decoder finished (or died) on its own: collect its real
            # exit status instead of killing it, so failures surface.
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        else:
            # Everything requested has been read (or we are unwinding an
            # exception): stop decoding the rest of the file.
            if proc.poll() is None:
                proc.kill()
            proc.wait()
        drain.join(timeout=2)

    if reached_eof and proc.returncode != 0:
        detail = " | ".join(stderr_tail) or f"exit code {proc.returncode}"
        raise RuntimeError(f"decoding {path.name} failed: {detail}")
    return [
        np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int16)
        for chunks in parts
    ]


def load_audio_windows(
    path: Path, windows: Sequence[tuple[float, float]]
) -> list[np.ndarray]:
    """`read_pcm_windows` as float32 in [-1, 1) -- the decoder's input format."""
    return [pcm.astype(np.float32) / 32768.0 for pcm in read_pcm_windows(path, windows)]


def load_audio_16k_mono(path: Path) -> np.ndarray:
    """Decode a whole GStreamer-supported audio file to mono float32 at 16kHz.

    Same pipe as `read_pcm_windows` (decodebin picks the AAC/mp3/... decoder
    automatically); the whole recording is held in memory, nothing on disk.
    """
    result = subprocess.run(
        _gst_pcm_command(path), capture_output=True, timeout=_DECODE_TIMEOUT_S
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", "replace").strip() or f"exit code {result.returncode}"
        raise RuntimeError(f"decoding {path} failed: {detail}")
    data = result.stdout[: len(result.stdout) - (len(result.stdout) % 2)]
    return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0


# ---------------------------------------------------------------------------
# Excerpts
# ---------------------------------------------------------------------------


def hash_unit(text: str) -> float:
    """Deterministic pseudo-random number in [0, 1) derived from `text`."""
    digest = hashlib.sha1(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def _proportional_allocation(count: int, lengths: list[float]) -> list[int]:
    """Split `count` windows across hosts in proportion to their lengths
    (largest-remainder method): the longest hosts win any rounding, so a
    single window lands on the longest speech region, not the last one."""
    total = sum(lengths)
    quotas = [count * length / total for length in lengths]
    alloc = [int(q) for q in quotas]
    by_remainder = sorted(range(len(lengths)), key=lambda i: (quotas[i] - alloc[i], lengths[i]), reverse=True)
    for i in by_remainder[: count - sum(alloc)]:
        alloc[i] += 1
    return alloc


def _snap(t: float, cue_starts: Sequence[float], tolerance: float = _SNAP_TOLERANCE_S) -> float:
    """Nearest cue start within `tolerance` of `t`, else `t` unchanged."""
    best = t
    best_distance = tolerance
    for start in cue_starts:
        distance = abs(start - t)
        if distance <= best_distance:
            best, best_distance = start, distance
        elif start > t + tolerance:
            break
    return best


def select_excerpt_windows(
    reference: RecordingReference,
    count: int,
    duration_s: float,
    min_speech_fraction: float = 0.5,
    seed: str | None = None,
) -> list[tuple[float, float]]:
    """Deterministically pick `count` (start, end) excerpt windows.

    Speech spans are merged into regions (gaps <= 10s bridged), regions that
    can host a full window AND are speech-dense enough are the hosts, and
    windows are placed inside hosts. Even spacing over the raw timeline was
    measured to fail: on the motivating recording both windows landed on
    the concert intro and the applause tail -- all music/applause, all
    rejected by the speech-fraction filter. Fewer/no windows are returned
    when the recording cannot host `count` speech-dense windows.

    Without `seed`, windows are spread evenly inside hosts allocated in
    proportion to host length. With a `seed` (the evaluation passes the
    recording's title), positions are hash-derived and stratified over all
    usable start positions -- deterministic, but no longer biased toward
    the same part (intro/outro) of every recording. Edges then snap to the
    nearest cue start within 3s so excerpts rarely cut a word in half.
    """
    if not reference.speech_spans or count <= 0:
        return []

    def speech_seconds(s: float, e: float) -> float:
        return sum(max(0.0, min(e, se) - max(s, ss)) for ss, se in reference.speech_spans)

    regions = _merge_spans(reference.speech_spans, gap=10.0)
    hosts = [
        (s, e)
        for s, e in regions
        if e - s >= duration_s
        and speech_seconds(s, e) >= duration_s * min_speech_fraction
    ]
    # Degenerate recording (no region long enough): one window over the
    # longest region, shrunk to fit, rather than nothing.
    if not hosts:
        longest = max(regions, key=lambda r: r[1] - r[0])
        window_s = min(duration_s, (longest[1] - longest[0]) * 0.9)
        if window_s < 5.0:
            return []
        return [(longest[0], longest[0] + window_s)]

    starts: list[float] = []
    if seed is None:
        alloc = _proportional_allocation(count, [e - s for s, e in hosts])
        for (s, e), n in zip(hosts, alloc):
            for i in range(n):
                # Spread windows inside the host; center when n == 1.
                starts.append(s + (e - s - duration_s) * ((i + 0.5) / n))
    else:
        # Each host offers its free start range (at least 1s, so a host
        # exactly one window long can still be picked); stratum i of the
        # concatenated ranges receives window i.
        ranges = [max(1.0, (e - s) - duration_s) for s, e in hosts]
        total = sum(ranges)
        for i in range(count):
            x = (i + hash_unit(f"{seed}:{i}")) * total / count
            for (s, e), free in zip(hosts, ranges):
                if x < free:
                    starts.append(s + (x / free) * ((e - s) - duration_s))
                    break
                x -= free
            else:  # float edge case: x landed exactly on the total
                s, e = hosts[-1]
                starts.append(e - duration_s)

    windows: list[tuple[float, float]] = []
    for start in sorted(starts):
        snapped_start = _snap(start, reference.cue_starts)
        snapped_end = _snap(snapped_start + duration_s, reference.cue_starts)
        if snapped_end - snapped_start < duration_s * 0.5:
            snapped_end = snapped_start + duration_s
        windows.append((snapped_start, snapped_end))
    return windows[:count]


def excerpt_reference_words(
    reference: RecordingReference, start: float, end: float
) -> list[str]:
    """Reference words spoken inside [start, end), by estimated word time.
    Masked windows need no handling here: their cues were excluded, so they
    carry no reference words."""
    return [w for t, w in reference.timed_words if start <= t < end]


def excerpt_reference_timed_words(
    reference: RecordingReference, start: float, end: float
) -> list[tuple[float, str]]:
    """Like `excerpt_reference_words`, with each word's time relative to
    the excerpt start (what latency is measured against)."""
    return [(t - start, w) for t, w in reference.timed_words if start <= t < end]
