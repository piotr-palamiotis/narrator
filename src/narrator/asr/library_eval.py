"""Library evaluation: how closely would the LIVE captioner, under a given
model and decode configuration, match the SRT references in
scripts/test_audio/?

This is the measuring instrument behind `narrator --autoconfigure`. GPU work
happens in `quality_probe` subprocesses (crash/VRAM isolation, same as
always); everything here is plain Python and testable without a GPU.

  * Sampling -- recordings are filtered by reference language and kind,
    split deterministically into dev (tuning) and holdout (confirmation) by
    a hash of their title, and sampled as fixed-length excerpts round-robin
    across recordings (one per recording before any recording gets a
    second). Everything is hash-ordered, so adding files to the library
    leaves existing excerpts -- and their cached results -- in place.
  * Audio cache -- excerpts are decoded once (GStreamer pipe, see
    real_fixtures) into int16 .npy files, guarded by a free-disk floor and
    a size cap: this machine's disks have filled up before.
  * Result store -- one JSONL record per decoded (excerpt, model, config,
    cost model, decoder code) combination, appended the moment it exists,
    so an interrupted run resumes where it stopped and repeated runs reuse
    everything already decoded. Records hold the raw captions; scores are
    recomputed, so a change to scoring never needs a re-decode.
  * Scoring -- POOLED WER (all errors / all reference words; a mean of
    per-excerpt WERs over-weights near-empty excerpts), the S/D/I breakdown,
    caption latency (FINAL emit time minus the reference word's time), and
    diagnostics: doubled words, non-Latin output, top inserted phrases.

The references are YouTube auto-captions for the most part -- noisy. Two
configurations are therefore compared PAIRED, on the same excerpts
(`paired_delta`), which cancels most of the per-excerpt difficulty.
"""
from __future__ import annotations

import ast
import hashlib
import json
import logging
import math
import os
import shutil
import statistics
import subprocess
import sys
import threading
import time
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from narrator.asr import real_fixtures
from narrator.asr.metrics import align
from narrator.asr.real_fixtures import RealRecording, hash_unit
from narrator.asr.stream_sim import DecodeCostModel, SimResult
from narrator.asr.streaming_decoder import _collapse_indices, _repetition_key
from narrator.config.schema import AsrConfig

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
CACHE_DIR = REPO_ROOT / "scripts" / "autoconfigure_cache"
RESULTS_PATH = CACHE_DIR / "results.jsonl"

HOLDOUT_FRACTION = 0.25
# Candidate excerpt positions computed per recording. Fixed (not derived
# from the requested sample size) so a recording's excerpts -- and the
# results cached for them -- do not move when the sample size changes.
_WINDOWS_PER_RECORDING = 4
# An excerpt whose reference holds fewer words than this is mostly music or
# silence and says little about transcription quality.
_MIN_REFERENCE_WORDS = 20
_MB = 1024 * 1024

# AsrConfig fields that change the FINAL captions, i.e. what makes two runs
# comparable. Display fields and silence timers do not, so they do not
# split the cache. Fields a given schema version lacks are skipped.
DECODE_FIELDS = (
    "language_whitelist",
    "vad_threshold",
    "step_ms",
    "agreement_passes",
    "end_silence_ms",
    "max_utterance_ms",
    "max_context_seconds",
    "vad_preroll_ms",
    "prompt_carryover_chars",
)

# Source files whose behavior a stored result depends on. Hashing them into
# every result key invalidates cached decodes automatically when the
# decoder changes -- no manual version bump to forget.
_DECODER_SOURCES = (
    Path(__file__).with_name("streaming_decoder.py"),
    Path(__file__).with_name("backend.py"),
    Path(__file__).with_name("vad.py"),
    Path(__file__).with_name("stream_sim.py"),
    Path(__file__).resolve().parents[1] / "audio" / "ring_buffer.py",
)


# ---------------------------------------------------------------------------
# Excerpts and sampling
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Excerpt:
    """One scored stretch of audio with its reference."""

    id: str
    title: str
    audio_path: Path
    start_s: float
    end_s: float
    # (seconds from the excerpt start, word); times are 0.0 when untimed
    reference: tuple[tuple[float, str], ...]
    kind: str  # reference kind: asr / human / unknown / synthetic
    timed: bool = True  # reference word times are meaningful (latency)

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    @property
    def reference_words(self) -> list[str]:
        return [w for _, w in self.reference]


@dataclass
class Library:
    """The usable recordings of a fixture directory."""

    recordings: list[RealRecording]  # drive the objective
    trusted: list[RealRecording]  # human subtitles: reported separately
    excluded: list[tuple[str, str]]  # (srt file name, reason)


def scan_library(directory: Path, reference_languages: Sequence[str] = ("pl",)) -> Library:
    """Classify every audio+SRT pair by its reference.

    Auto-caption and untagged references in a wanted language drive the
    objective. Human subtitles in a wanted language are a separate, more
    trustworthy check (subtitles also condense speech, so they are not
    mixed into the objective). Anything else -- another language, or a tag
    whisper does not know -- cannot be scored and is listed as excluded.
    """
    wanted = set(reference_languages)
    library = Library(recordings=[], trusted=[], excluded=[])
    for recording in real_fixtures.discover_recordings(directory):
        language, kind = recording.reference_language, recording.reference_kind
        if language is None:
            library.excluded.append((recording.srt_path.name, "unrecognized language tag"))
        elif language not in wanted:
            library.excluded.append((recording.srt_path.name, f"reference language '{language}'"))
        elif kind == "human":
            library.trusted.append(recording)
        else:
            library.recordings.append(recording)
    return library


def split_of(title: str, holdout_fraction: float = HOLDOUT_FRACTION) -> str:
    """'dev' or 'holdout', fixed by the title's hash: adding recordings to
    the library never moves an existing one between the sets."""
    return "holdout" if hash_unit(f"split:{title}") < holdout_fraction else "dev"


def audio_fingerprint(path: Path) -> str:
    """Cheap content fingerprint (size + first/last 64 KiB). Survives the
    file being copied or moved, unlike mtime; changes if it is re-encoded."""
    size = path.stat().st_size
    digest = hashlib.sha1(str(size).encode())
    with path.open("rb") as f:
        digest.update(f.read(65536))
        if size > 65536:
            f.seek(max(size - 65536, 65536))
            digest.update(f.read(65536))
    return digest.hexdigest()[:10]


def _make_excerpt(recording: RealRecording, reference, start: float, end: float, fingerprint: str) -> Excerpt:
    return Excerpt(
        id=f"{recording.title}@{start:.2f}-{end:.2f}#{fingerprint}",
        title=recording.title,
        audio_path=recording.audio_path,
        start_s=start,
        end_s=end,
        reference=tuple(real_fixtures.excerpt_reference_timed_words(reference, start, end)),
        kind=recording.reference_kind,
    )


def sample_excerpts(
    recordings: Sequence[RealRecording],
    count: int,
    excerpt_seconds: float,
    split: str | None = None,
) -> list[Excerpt]:
    """Up to `count` excerpts, round-robin across `recordings` (restricted
    to one dev/holdout `split` when given).

    Recordings are visited in hash order and each contributes its excerpt
    positions in hash order, so the sample of size N is a prefix of the
    sample of size N+1 -- the screen set is literally the start of the dev
    set, and their results are shared.
    """
    pool = [r for r in recordings if split is None or split_of(r.title) == split]
    pool.sort(key=lambda r: hash_unit(f"order:{r.title}"))
    per_recording: list[list[Excerpt]] = []
    for recording in pool:
        try:
            reference = real_fixtures.load_reference(recording.srt_path)
            fingerprint = audio_fingerprint(recording.audio_path)
        except (OSError, ValueError) as exc:
            logger.warning("skipping %s: %s", recording.srt_path.name, exc)
            continue
        windows = real_fixtures.select_excerpt_windows(
            reference, _WINDOWS_PER_RECORDING, excerpt_seconds, seed=recording.title
        )
        windows.sort(key=lambda w: hash_unit(f"pick:{recording.title}:{w[0]:.2f}"))
        excerpts = [_make_excerpt(recording, reference, s, e, fingerprint) for s, e in windows]
        per_recording.append([x for x in excerpts if len(x.reference) >= _MIN_REFERENCE_WORDS])

    out: list[Excerpt] = []
    for round_index in range(_WINDOWS_PER_RECORDING):
        for excerpts in per_recording:
            if round_index < len(excerpts):
                out.append(excerpts[round_index])
                if len(out) >= count:
                    return out
    return out


def synthetic_excerpts(clips_dir: Path) -> list[Excerpt]:
    """The Piper fixtures (scripts/test_clips) as excerpts: whole clips with
    known transcripts, no word timing (so no latency). The fallback when no
    real recordings are present."""
    manifest = json.loads((clips_dir / "manifest.json").read_text(encoding="utf-8"))
    out: list[Excerpt] = []
    for clip in manifest["clips"]:
        wav = clips_dir / f"{clip['id']}.wav"
        reference = clip.get("reference", clip["transcript"])
        out.append(
            Excerpt(
                id=f"clip:{clip['id']}#{audio_fingerprint(wav)}",
                title=str(clip["id"]),
                audio_path=wav,
                start_s=0.0,
                end_s=0.0,  # whole file
                reference=tuple((0.0, w) for w in real_fixtures.words(reference)),
                kind="synthetic",
                timed=False,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Audio cache
# ---------------------------------------------------------------------------


class AudioCache:
    """Decoded excerpt audio as int16 .npy files (about 2.9 MB per 90 s).

    Writing is refused when it would leave less than `min_free_mb` free on
    the cache's disk; such excerpts are then decoded in the evaluation
    subprocess straight from the pipe instead (slower, but never fills the
    disk). The cache is trimmed to `max_mb`, least recently used first.
    """

    def __init__(self, directory: Path = CACHE_DIR / "excerpts", max_mb: float = 1024, min_free_mb: float = 1024) -> None:
        self.directory = directory
        self.max_bytes = int(max_mb * _MB)
        self.min_free_bytes = int(min_free_mb * _MB)
        self._warned_full = False

    def path_for(self, excerpt: Excerpt) -> Path:
        return self.directory / (hashlib.sha1(excerpt.id.encode("utf-8")).hexdigest()[:20] + ".npy")

    def free_bytes(self) -> int:
        probe = self.directory if self.directory.exists() else self.directory.parent
        while not probe.exists():
            probe = probe.parent
        return shutil.disk_usage(probe).free

    def _can_write(self, nbytes: int) -> bool:
        return self.free_bytes() - nbytes >= self.min_free_bytes

    def prepare(self, excerpts: Iterable[Excerpt], progress: Callable[[str], None] | None = None) -> dict[str, Path | None]:
        """Cached .npy path per excerpt id (None = not cached, decode at use).
        Missing excerpts are decoded with one pipe pass per recording."""
        out: dict[str, Path | None] = {}
        missing: dict[Path, list[Excerpt]] = defaultdict(list)
        for excerpt in excerpts:
            if excerpt.kind == "synthetic":
                out[excerpt.id] = None  # a small WAV already on disk
                continue
            path = self.path_for(excerpt)
            if path.exists():
                os.utime(path)  # LRU touch
                out[excerpt.id] = path
            else:
                missing[excerpt.audio_path].append(excerpt)
        if missing:
            self.directory.mkdir(parents=True, exist_ok=True)
        for index, (audio_path, group) in enumerate(sorted(missing.items()), start=1):
            if progress is not None:
                progress(f"decoding {audio_path.name} ({index}/{len(missing)})")
            try:
                pcm = real_fixtures.read_pcm_windows(audio_path, [(x.start_s, x.end_s) for x in group])
            except (OSError, RuntimeError) as exc:
                logger.warning("could not decode %s: %s", audio_path.name, exc)
                for excerpt in group:
                    out[excerpt.id] = None
                continue
            for excerpt, samples in zip(group, pcm):
                out[excerpt.id] = self._write(excerpt, samples)
        self.evict(keep={p for p in out.values() if p is not None})
        return out

    def _write(self, excerpt: Excerpt, samples: np.ndarray) -> Path | None:
        if not self._can_write(samples.nbytes):
            if not self._warned_full:
                logger.warning(
                    "less than %d MB would stay free on the cache disk; decoding excerpts "
                    "on the fly instead of caching them",
                    self.min_free_bytes // _MB,
                )
                self._warned_full = True
            return None
        path = self.path_for(excerpt)
        tmp = path.with_suffix(".tmp")
        with tmp.open("wb") as f:
            np.save(f, samples.astype(np.int16, copy=False))
        os.replace(tmp, path)
        return path

    def evict(self, keep: set[Path] = frozenset()) -> None:
        if not self.directory.is_dir():
            return
        files = sorted(self.directory.glob("*.npy"), key=lambda p: p.stat().st_mtime)
        total = sum(p.stat().st_size for p in files)
        for path in files:
            if total <= self.max_bytes:
                break
            if path in keep:
                continue
            total -= path.stat().st_size
            path.unlink(missing_ok=True)


def load_excerpt_audio(excerpt: Excerpt, cache_path: Path | None) -> np.ndarray:
    """Float32 16 kHz audio of an excerpt: from the cache when present,
    else straight from the decoder pipe (or the clip WAV)."""
    if cache_path is not None:
        return np.load(cache_path).astype(np.float32) / 32768.0
    if excerpt.kind == "synthetic":
        from narrator.asr.metrics import load_wav_16k_mono

        return load_wav_16k_mono(excerpt.audio_path)
    (audio,) = real_fixtures.load_audio_windows(excerpt.audio_path, [(excerpt.start_s, excerpt.end_s)])
    return audio


# ---------------------------------------------------------------------------
# Result store and keys
# ---------------------------------------------------------------------------


def decode_settings(config: AsrConfig) -> dict:
    """The decode-relevant fields of `config` (see DECODE_FIELDS)."""
    return {name: getattr(config, name) for name in DECODE_FIELDS if hasattr(config, name)}


def config_signature(config: AsrConfig) -> str:
    return hashlib.sha1(
        json.dumps(decode_settings(config), sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]


def cost_signature(cost: DecodeCostModel | None) -> str:
    if cost is None:
        return "none"
    return hashlib.sha1(json.dumps(cost.to_dict(), sort_keys=True).encode("utf-8")).hexdigest()[:12]


def _code_fingerprint(source: str) -> str:
    """The module's code with comments and docstrings ignored (hash of the
    AST), so editing documentation does not invalidate cached decodes."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (
            isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            node.body = body[1:] or [ast.Pass()]
    return ast.dump(tree, include_attributes=False)


def code_signature() -> str:
    digest = hashlib.sha1()
    for path in _DECODER_SOURCES:
        digest.update(_code_fingerprint(path.read_text(encoding="utf-8")).encode("utf-8"))
    return digest.hexdigest()[:12]


def record_key(*, excerpt_id: str, model: str, mode: str, config_sig: str, cost_sig: str, code_sig: str) -> str:
    raw = "|".join((excerpt_id, model, mode, config_sig, cost_sig, code_sig))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


class ResultStore:
    """Append-only JSONL of raw decode results, read incrementally."""

    def __init__(self, path: Path = RESULTS_PATH) -> None:
        self.path = path
        self._records: dict[str, dict] = {}
        self._offset = 0

    def refresh(self) -> dict[str, dict]:
        """Read records appended since the last call; returns all records."""
        if not self.path.exists():
            return self._records
        with self.path.open("rb") as f:
            f.seek(self._offset)
            for line in f:
                if not line.endswith(b"\n"):
                    break  # a torn last line (writer killed mid-write)
                self._offset += len(line)
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self._records[record["key"]] = record
        return self._records

    def get(self, key: str) -> dict | None:
        return self._records.get(key)

    def append(self, record: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with self.path.open("ab") as f:
            if f.tell() > 0:
                with self.path.open("rb") as tail:
                    tail.seek(-1, os.SEEK_END)
                    if tail.read(1) != b"\n":
                        # a writer was killed mid-line: start a fresh line so
                        # this record is not glued onto the torn one
                        line = "\n" + line
            f.write(line.encode("utf-8"))


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _collapse_timed_repetitions(pairs: list[tuple[float, str]]) -> list[tuple[float, str]]:
    """`streaming_decoder._collapse_repetitions` for (time, word) pairs: the
    decoder collapses immediate repeats in its own output, so the reference
    gets the same treatment before the two are compared."""
    keys = [_repetition_key(w) for _, w in pairs]
    return [pairs[i] for i in _collapse_indices(keys, 32)]


def _is_non_latin(word: str) -> bool:
    """True when a word contains letters outside the Latin script (e.g. the
    Cyrillic captions large-v3 has produced for Polish speech)."""
    return any(
        ch.isalpha() and not unicodedata.name(ch, "LATIN").startswith("LATIN") for ch in word
    )


@dataclass
class ExcerptScore:
    excerpt_id: str
    reference_words: int
    substitutions: int
    deletions: int
    insertions: int
    hypothesis_words: int
    latencies: list[float]
    doubled: int
    non_latin: int
    inserted_phrases: list[str]

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions


def score_excerpt(excerpt: Excerpt, result: SimResult) -> ExcerptScore:
    reference = _collapse_timed_repetitions(list(excerpt.reference))
    ref_words = [w for _, w in reference]
    hyp: list[str] = []
    hyp_times: list[float] = []
    for emitted_at, text in result.finals:
        for word in real_fixtures.words(text):
            hyp.append(word)
            hyp_times.append(emitted_at)
    alignment = align(ref_words, hyp)
    latencies = (
        [hyp_times[j] - reference[i][0] for i, j in alignment.pairs if ref_words[i] == hyp[j]]
        if excerpt.timed
        else []
    )
    phrases: list[str] = []
    run: list[int] = []
    for index in alignment.inserted:
        if run and index != run[-1] + 1:
            phrases.append(" ".join(hyp[k] for k in run[:6]))
            run = []
        run.append(index)
    if run:
        phrases.append(" ".join(hyp[k] for k in run[:6]))
    return ExcerptScore(
        excerpt_id=excerpt.id,
        reference_words=len(ref_words),
        substitutions=alignment.substitutions,
        deletions=alignment.deletions,
        insertions=alignment.insertions,
        hypothesis_words=len(hyp),
        latencies=latencies,
        doubled=sum(1 for k in range(1, len(hyp)) if hyp[k] == hyp[k - 1]),
        non_latin=sum(1 for w in hyp if _is_non_latin(w)),
        inserted_phrases=phrases,
    )


@dataclass
class ConfigScore:
    """One (model, config) evaluated over a set of excerpts."""

    excerpts: int = 0
    missing: int = 0  # excerpts with no result (decode failed / timed out)
    reference_words: int = 0
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    hypothesis_words: int = 0
    latency_p50: float | None = None
    latency_p90: float | None = None
    audio_seconds: float = 0.0
    busy_seconds: float = 0.0
    doubled: int = 0
    non_latin: int = 0
    rollovers: int = 0
    per_excerpt: dict[str, tuple[int, int]] = field(default_factory=dict)  # id -> (errors, ref words)
    top_insertions: list[tuple[str, int]] = field(default_factory=list)

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def wer(self) -> float:
        return self.errors / self.reference_words if self.reference_words else 0.0

    @property
    def busy_fraction(self) -> float:
        return self.busy_seconds / self.audio_seconds if self.audio_seconds else 0.0

    @property
    def doubled_rate(self) -> float:
        return self.doubled / self.hypothesis_words if self.hypothesis_words else 0.0

    @property
    def non_latin_rate(self) -> float:
        return self.non_latin / self.hypothesis_words if self.hypothesis_words else 0.0

    def to_dict(self) -> dict:
        return {
            "excerpts": self.excerpts,
            "missing": self.missing,
            "wer": round(self.wer, 4),
            "reference_words": self.reference_words,
            "substitutions": self.substitutions,
            "deletions": self.deletions,
            "insertions": self.insertions,
            "hypothesis_words": self.hypothesis_words,
            "latency_p50": None if self.latency_p50 is None else round(self.latency_p50, 3),
            "latency_p90": None if self.latency_p90 is None else round(self.latency_p90, 3),
            "busy_fraction": round(self.busy_fraction, 3),
            "doubled_rate": round(self.doubled_rate, 5),
            "non_latin_rate": round(self.non_latin_rate, 5),
            "rollovers": self.rollovers,
            "top_insertions": self.top_insertions,
        }


def aggregate(scored: Sequence[tuple[ExcerptScore, SimResult]], missing: int = 0) -> ConfigScore:
    total = ConfigScore(missing=missing)
    latencies: list[float] = []
    phrases: Counter[str] = Counter()
    for score, result in scored:
        total.excerpts += 1
        total.reference_words += score.reference_words
        total.substitutions += score.substitutions
        total.deletions += score.deletions
        total.insertions += score.insertions
        total.hypothesis_words += score.hypothesis_words
        total.audio_seconds += result.audio_seconds
        total.busy_seconds += result.busy_seconds
        total.doubled += score.doubled
        total.non_latin += score.non_latin
        total.rollovers += result.rollovers
        total.per_excerpt[score.excerpt_id] = (score.errors, score.reference_words)
        latencies.extend(score.latencies)
        phrases.update(score.inserted_phrases)
    if latencies:
        total.latency_p50 = float(np.percentile(latencies, 50))
        total.latency_p90 = float(np.percentile(latencies, 90))
    total.top_insertions = [(p, n) for p, n in phrases.most_common(15) if n > 1]
    return total


def paired_delta(baseline: ConfigScore, candidate: ConfigScore) -> tuple[int, float, int]:
    """(errors saved by `candidate` relative to `baseline`, its standard
    error, number of shared excerpts), over the excerpts both were scored
    on. Positive = candidate better. Pairing cancels per-excerpt difficulty,
    which dominates the variance with noisy references."""
    common = sorted(baseline.per_excerpt.keys() & candidate.per_excerpt.keys())
    diffs = [baseline.per_excerpt[k][0] - candidate.per_excerpt[k][0] for k in common]
    if len(diffs) < 2:
        return sum(diffs), math.inf, len(diffs)
    return sum(diffs), math.sqrt(len(diffs)) * statistics.stdev(diffs), len(diffs)


# ---------------------------------------------------------------------------
# Running evaluations
# ---------------------------------------------------------------------------


def run_probe_subprocess(module: str, payload: dict, timeout_s: float) -> dict:
    """Run a bench/probe worker module in a fresh subprocess and parse its
    `BENCH_JSON:` line. Any failure becomes {"error": ...} rather than an
    exception -- a broken candidate must disqualify itself, not crash the
    run.

    Fresh process per job: a native crash or VRAM exhaustion from one model
    cannot leak into the next. stderr and non-JSON stdout lines are
    forwarded to the parent's stderr in real time, so long runs show
    progress; the last stderr lines are attached to errors so a failure
    says where it happened.
    """
    command = [sys.executable, "-m", module, json.dumps(payload)]
    try:
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except FileNotFoundError as exc:
        return {"error": f"{module} failed to start: {exc}"}

    json_result: dict | None = None
    stderr_buffer: list[str] = []

    def _read_stdout() -> None:
        nonlocal json_result
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip("\n")
            if line.startswith("BENCH_JSON:"):
                try:
                    json_result = json.loads(line[len("BENCH_JSON:") :])
                except json.JSONDecodeError:
                    json_result = {"error": f"{module} produced unparseable JSON"}
            elif line:
                sys.stderr.write(line + "\n")
                sys.stderr.flush()

    def _read_stderr() -> None:
        assert proc.stderr is not None
        for line in proc.stderr:
            stderr_buffer.append(line)
            if len(stderr_buffer) > 200:
                del stderr_buffer[:100]
            sys.stderr.write(line)
            sys.stderr.flush()

    readers = [threading.Thread(target=_read_stdout, daemon=True), threading.Thread(target=_read_stderr, daemon=True)]
    for reader in readers:
        reader.start()

    def _stderr_tail(n: int = 3) -> str:
        lines = [ln.strip() for ln in stderr_buffer if ln.strip()]
        return " | ".join(lines[-n:])

    try:
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        for reader in readers:
            reader.join(timeout=2)
        tail = _stderr_tail()
        return {"error": f"{module} timed out after {timeout_s:.0f}s" + (f" (last output: {tail})" if tail else "")}
    for reader in readers:
        reader.join(timeout=5)
    if json_result is not None:
        return json_result
    tail = _stderr_tail()
    return {"error": f"{module} exited {proc.returncode} with no BENCH_JSON line" + (f" (last output: {tail})" if tail else "")}


@dataclass
class ModelProfile:
    """A candidate model as measured on this machine."""

    model: str
    device: str
    resolved_device: str
    cost: DecodeCostModel
    # evaluation wall time per second of audio observed so far (budgeting)
    observed_rates: list[float] = field(default_factory=list)

    def estimate_wall_per_audio_s(self, config: AsrConfig) -> float:
        """Rough evaluation cost per audio second for budgeting."""
        if self.observed_rates:
            return sum(self.observed_rates) / len(self.observed_rates)
        # A step decodes a window averaging ~6 s (~15 words); the worker
        # decodes about once per max(step, decode) -- plus VAD/Python.
        decode = self.cost.decode_seconds(6.0, 15)
        return min(1.5, decode / max(config.step_ms / 1000.0, decode) + 0.02)


Runner = Callable[[str, dict, float], dict]


class Evaluator:
    """Scores (model, config) pairs over excerpts, decoding only what the
    result store does not already hold."""

    MODULE = "narrator.asr.quality_probe"

    def __init__(
        self,
        cache: AudioCache,
        store: ResultStore,
        runner: Runner = run_probe_subprocess,
        job_dir: Path = CACHE_DIR,
    ) -> None:
        self.cache = cache
        self.store = store
        self.runner = runner
        self.job_dir = job_dir
        self.code_sig = code_signature()
        self.decode_seconds_spent = 0.0  # wall time spent in decode jobs this session
        # result keys whose decode crashed the probe repeatedly (skipped)
        self.failed: set[str] = set()
        self._jobs_started = 0

    def _key(self, excerpt: Excerpt, profile: ModelProfile, config: AsrConfig, mode: str) -> str:
        # stream results depend on the cost model; measured ones are keyed
        # by it too, so re-benching (a GPU or driver change) re-measures
        cost = profile.cost if mode in ("stream", "measured") else None
        return record_key(
            excerpt_id=excerpt.id,
            model=profile.model,
            mode=mode,
            config_sig=config_signature(config) if mode != "offline" else "offline",
            cost_sig=cost_signature(cost),
            code_sig=self.code_sig,
        )

    def record(self, excerpt: Excerpt, profile: ModelProfile, config: AsrConfig, mode: str = "stream") -> dict | None:
        """The stored result record for one evaluation, if any."""
        self.store.refresh()
        return self.store.get(self._key(excerpt, profile, config, mode))

    def pending_audio_seconds(self, profile: ModelProfile, configs: Sequence[AsrConfig], excerpts: Sequence[Excerpt], mode: str = "stream") -> float:
        self.store.refresh()
        return sum(
            _excerpt_seconds(x)
            for config in configs
            for x in excerpts
            if self.store.get(self._key(x, profile, config, mode)) is None
        )

    def estimate_seconds(self, profile: ModelProfile, configs: Sequence[AsrConfig], excerpts: Sequence[Excerpt], mode: str = "stream") -> float:
        """Wall-clock estimate for evaluating what is not cached yet."""
        self.store.refresh()
        seconds = 0.0
        for config in configs:
            pending = sum(
                _excerpt_seconds(x)
                for x in excerpts
                if self.store.get(self._key(x, profile, config, mode)) is None
            )
            per_s = 0.06 if mode == "offline" else profile.estimate_wall_per_audio_s(config)
            seconds += pending * per_s
        return seconds + (15.0 if seconds else 0.0)  # model load

    def evaluate(
        self,
        profile: ModelProfile,
        configs: Sequence[AsrConfig],
        excerpts: Sequence[Excerpt],
        mode: str = "stream",
        language: str | None = None,
    ) -> list[ConfigScore]:
        """One ConfigScore per config. `mode` is "stream" (the virtual-clock
        live replay), "offline" (whole-excerpt transcription: the ceiling),
        "measured" or "realtime" (fidelity checks)."""
        attempts = 0
        while attempts < 3 + len(self.failed):
            tasks = self._pending_tasks(profile, configs, excerpts, mode)
            if not tasks:
                break
            attempts += 1
            if self._run_job(profile, tasks, mode, language):
                continue
            remaining = self._pending_tasks(profile, configs, excerpts, mode)
            if len(remaining) == len(tasks):
                # No progress at all: the decode the job died on is the
                # culprit (tasks run in order). Skip it, keep the rest.
                self.failed.add(tasks[0]["key"])
                logger.warning("%s: skipping %s after repeated failures", profile.model, tasks[0]["excerpt"].id)
            elif remaining:
                logger.warning("%s: evaluation incomplete; retrying the remaining excerpts", profile.model)
        scores: list[ConfigScore] = []
        for config in configs:
            scored: list[tuple[ExcerptScore, SimResult]] = []
            missing = 0
            for excerpt in excerpts:
                record = self.store.get(self._key(excerpt, profile, config, mode))
                if record is None:
                    missing += 1
                    continue
                result = SimResult.from_dict(record["result"])
                scored.append((score_excerpt(excerpt, result), result))
            scores.append(aggregate(scored, missing=missing))
        return scores

    def _pending_tasks(self, profile, configs, excerpts, mode) -> list[dict]:
        self.store.refresh()
        tasks: list[dict] = []
        for config in configs:
            for excerpt in excerpts:
                key = self._key(excerpt, profile, config, mode)
                if self.store.get(key) is None and key not in self.failed:
                    tasks.append({"key": key, "excerpt": excerpt, "config": config})
        return tasks

    def _run_job(self, profile: ModelProfile, tasks: list[dict], mode: str, language: str | None) -> bool:
        excerpts = {t["excerpt"].id: t["excerpt"] for t in tasks}
        cache_paths = self.cache.prepare(excerpts.values())
        job = {
            "mode": mode,
            "model": profile.model,
            "device": profile.device,
            "language": language,
            "cost_model": profile.cost.to_dict() if mode == "stream" else None,
            "store_path": str(self.store.path),
            "meta": {
                "model": profile.model,
                "cost_signature": cost_signature(profile.cost if mode in ("stream", "measured") else None),
                "code_signature": self.code_sig,
            },
            "tasks": [
                {
                    "key": t["key"],
                    "excerpt_id": t["excerpt"].id,
                    "config_signature": config_signature(t["config"]),
                    "asr": t["config"].model_dump(mode="json"),
                    "audio": {
                        "cache": None if cache_paths.get(t["excerpt"].id) is None else str(cache_paths[t["excerpt"].id]),
                        "path": str(t["excerpt"].audio_path),
                        "start_s": t["excerpt"].start_s,
                        "end_s": t["excerpt"].end_s,
                        "synthetic": t["excerpt"].kind == "synthetic",
                    },
                }
                for t in tasks
            ],
        }
        self.job_dir.mkdir(parents=True, exist_ok=True)
        # Unique per job: two runs sharing the cache must not overwrite each
        # other's job file before their probe has read it.
        self._jobs_started += 1
        job_path = self.job_dir / f"job-{os.getpid()}-{self._jobs_started}.json"
        job_path.write_text(json.dumps(job, ensure_ascii=False), encoding="utf-8")
        audio_s = sum(_excerpt_seconds(t["excerpt"]) for t in tasks)
        per_s = 0.06 if mode == "offline" else max(profile.estimate_wall_per_audio_s(t["config"]) for t in tasks)
        if mode == "realtime":
            per_s = 1.1
        timeout_s = 300.0 + 3.0 * audio_s * per_s
        started = time.monotonic()
        try:
            payload = self.runner(self.MODULE, {"job_file": str(job_path)}, timeout_s)
        finally:
            job_path.unlink(missing_ok=True)
        elapsed = time.monotonic() - started
        self.decode_seconds_spent += elapsed
        if "error" in payload:
            logger.warning("%s evaluation job failed: %s", profile.model, payload["error"])
            return False
        decoded_audio = float(payload.get("audio_seconds", 0.0))
        if mode == "stream" and decoded_audio > 60.0:
            profile.observed_rates.append(float(payload.get("wall_seconds", elapsed)) / decoded_audio)
        return True


def _excerpt_seconds(excerpt: Excerpt) -> float:
    # whole-file synthetic clips report 0 duration; they are a few seconds
    return excerpt.duration_s if excerpt.duration_s > 0 else 5.0


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
