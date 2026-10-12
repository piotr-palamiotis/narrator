"""Library evaluation (narrator.asr.library_eval) and word alignment: no
GPU, no subprocess -- the evaluator runs against a fake job runner."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from narrator.asr import library_eval as le
from narrator.asr.metrics import align
from narrator.asr.stream_sim import DecodeCostModel, SimResult
from narrator.config.schema import AsrConfig


# ---------------------------------------------------------------------------
# alignment
# ---------------------------------------------------------------------------


def test_align_counts_each_edit_kind():
    a = align("a b c d".split(), "a x c d e".split())
    assert (a.substitutions, a.deletions, a.insertions) == (1, 0, 1)
    a = align("a b c d".split(), "a c d".split())
    assert (a.substitutions, a.deletions, a.insertions) == (0, 1, 0)
    assert a.errors == 1 and a.reference_words == 4


def test_align_pairs_and_insertions_are_positional():
    a = align("ala ma kota".split(), "ala ma ma kota".split())
    assert a.insertions == 1 and a.substitutions == 0
    assert (0, 0) in a.pairs and (2, 3) in a.pairs
    assert len(a.inserted) == 1


def test_align_empty_sides():
    assert align([], "a b".split()).insertions == 2
    assert align("a b".split(), []).deletions == 2


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------


def excerpt(words_with_times, excerpt_id="rec@0.00-10.00#x", timed=True) -> le.Excerpt:
    return le.Excerpt(
        id=excerpt_id,
        title="rec",
        audio_path=Path("rec.m4a"),
        start_s=0.0,
        end_s=10.0,
        reference=tuple(words_with_times),
        kind="asr",
        timed=timed,
    )


def sim(finals, audio_seconds=10.0, busy=2.0, rollovers=0) -> SimResult:
    return SimResult(
        finals=finals, audio_seconds=audio_seconds, decode_calls=5, detect_calls=0,
        busy_seconds=busy, rollovers=rollovers,
    )


def test_score_latency_is_emit_time_minus_reference_word_time():
    ex = excerpt([(1.0, "dzień"), (1.5, "dobry"), (4.0, "państwu")])
    result = sim([(3.0, "Dzień dobry"), (6.5, "państwu.")])
    score = le.score_excerpt(ex, result)
    assert score.errors == 0
    assert sorted(score.latencies) == pytest.approx([1.5, 2.0, 2.5])


def test_score_counts_doubles_non_latin_and_inserted_phrases():
    ex = excerpt([(1.0, "to"), (2.0, "jest"), (3.0, "test")])
    result = sim([(4.0, "to jest jest test"), (6.0, "dziękuję za uwagę привет")])
    score = le.score_excerpt(ex, result)
    assert score.doubled == 1
    assert score.non_latin == 1
    assert "dziękuję za uwagę привет" in score.inserted_phrases


def test_reference_repeats_collapse_like_the_decoder():
    # the decoder collapses "tak tak" in its own output; the reference gets
    # the same treatment so an emphatic repeat is not a deletion
    ex = excerpt([(1.0, "tak"), (1.2, "tak"), (2.0, "oczywiście")])
    score = le.score_excerpt(ex, sim([(3.0, "tak oczywiście")]))
    assert score.errors == 0


def test_untimed_reference_has_no_latency():
    ex = excerpt([(0.0, "a"), (0.0, "b")], timed=False)
    assert le.score_excerpt(ex, sim([(1.0, "a b")])).latencies == []


def test_pooled_wer_weights_by_reference_words():
    long_ex = excerpt([(float(i), f"w{i}") for i in range(90)], excerpt_id="long")
    short_ex = excerpt([(0.0, "a"), (1.0, "b"), (2.0, "c"), (3.0, "d"), (4.0, "e"), (5.0, "f"), (6.0, "g"), (7.0, "h"), (8.0, "i"), (9.0, "j")], excerpt_id="short")
    long_res = sim([(100.0, " ".join(f"w{i}" for i in range(90)))])  # perfect
    short_res = sim([(20.0, "x")])  # 9 deletions + 1 substitution = 10 errors
    scored = [(le.score_excerpt(long_ex, long_res), long_res), (le.score_excerpt(short_ex, short_res), short_res)]
    total = le.aggregate(scored)
    assert total.wer == pytest.approx(10 / 100)  # not the 50% a per-excerpt mean gives
    assert total.busy_fraction == pytest.approx(4.0 / 20.0)


def make_score(errors_by_excerpt: dict[str, int]) -> le.ConfigScore:
    score = le.ConfigScore()
    for key, errors in errors_by_excerpt.items():
        score.per_excerpt[key] = (errors, 100)
    return score


def test_paired_delta_sign_and_standard_error():
    base = make_score({"a": 10, "b": 12, "c": 9, "d": 11})
    better = make_score({"a": 8, "b": 10, "c": 7, "d": 9})
    saved, se, n = le.paired_delta(base, better)
    assert saved == 8 and n == 4
    assert se == pytest.approx(0.0)  # identical improvement on every excerpt
    worse_saved, _, _ = le.paired_delta(better, base)
    assert worse_saved == -8


def test_paired_delta_uses_only_shared_excerpts():
    base = make_score({"a": 10, "b": 10})
    cand = make_score({"a": 5, "z": 0})
    saved, se, n = le.paired_delta(base, cand)
    assert (saved, n) == (5, 1) and math.isinf(se)


# ---------------------------------------------------------------------------
# library scan, split, sampling
# ---------------------------------------------------------------------------


def write_srt(path: Path, seconds: int, words_per_cue: int = 8) -> None:
    lines = []
    for i, start in enumerate(range(0, seconds, 4), start=1):
        text = " ".join(f"słowo{start}_{k}" for k in range(words_per_cue))
        lines.append(f"{i}\n00:{start // 60:02d}:{start % 60:02d},000 --> 00:{(start + 4) // 60:02d}:{(start + 4) % 60:02d},000\n{text}\n")
    path.write_text("\n".join(lines), encoding="utf-8")


def make_library(tmp_path: Path, titles: list[str], tag: str = "Polish_ASR", seconds: int = 600) -> None:
    for title in titles:
        (tmp_path / f"{title} (128kbit_AAC).m4a").write_bytes(title.encode() * 100)
        write_srt(tmp_path / f"{title} ({tag}).srt", seconds)


def test_scan_library_classifies_by_reference_tag(tmp_path):
    make_library(tmp_path, ["A"], "Polish_ASR")
    make_library(tmp_path, ["B"], "Polish")
    make_library(tmp_path, ["C"], "English_ASR")
    make_library(tmp_path, ["D"], "Manx")
    (tmp_path / "E.wav").write_bytes(b"RIFF")
    write_srt(tmp_path / "E.srt", 60)
    library = le.scan_library(tmp_path, ["pl"])
    assert sorted(r.title for r in library.recordings) == ["A", "E"]  # untagged counts as pl
    assert [r.title for r in library.trusted] == ["B"]
    reasons = dict(library.excluded)
    assert "reference language 'en'" in reasons.values()
    assert "unrecognized language tag" in reasons.values()


def test_split_is_stable_and_roughly_a_quarter_holdout():
    titles = [f"film {i}" for i in range(400)]
    first = [le.split_of(t) for t in titles]
    assert first == [le.split_of(t) for t in titles]
    holdout = first.count("holdout") / len(titles)
    assert 0.18 < holdout < 0.32


def test_sample_is_a_prefix_of_any_larger_sample(tmp_path):
    make_library(tmp_path, [f"film {i}" for i in range(8)])
    library = le.scan_library(tmp_path)
    small = le.sample_excerpts(library.recordings, 5, 60.0)
    large = le.sample_excerpts(library.recordings, 12, 60.0)
    assert [x.id for x in small] == [x.id for x in large[:5]]
    # round-robin: one excerpt per recording before any second one
    assert len({x.title for x in large[:8]}) == 8


def test_sample_respects_split(tmp_path):
    make_library(tmp_path, [f"film {i}" for i in range(12)])
    library = le.scan_library(tmp_path)
    for split in ("dev", "holdout"):
        for x in le.sample_excerpts(library.recordings, 50, 60.0, split=split):
            assert le.split_of(x.title) == split


def test_excerpts_carry_reference_words_inside_the_window_only(tmp_path):
    make_library(tmp_path, ["film"], seconds=600)
    library = le.scan_library(tmp_path)
    (x,) = le.sample_excerpts(library.recordings, 1, 60.0)
    assert x.reference
    assert all(0.0 <= t < x.duration_s for t, _ in x.reference)


# ---------------------------------------------------------------------------
# keys
# ---------------------------------------------------------------------------


def test_config_signature_ignores_display_fields():
    base = AsrConfig(language_whitelist=["pl"])
    assert le.config_signature(base) == le.config_signature(base.model_copy(update={"show_partial_captions": True}))
    assert le.config_signature(base) != le.config_signature(base.model_copy(update={"step_ms": 300}))


# ---------------------------------------------------------------------------
# audio cache
# ---------------------------------------------------------------------------


def test_cache_refuses_to_fill_the_disk(tmp_path, monkeypatch):
    cache = le.AudioCache(tmp_path / "cache", max_mb=100, min_free_mb=10)
    ex = le.Excerpt("id", "t", tmp_path / "a.m4a", 0.0, 1.0, ((0.0, "a"),), "asr")
    monkeypatch.setattr(le.real_fixtures, "read_pcm_windows", lambda path, windows: [np.ones(16000, np.int16) for _ in windows])
    monkeypatch.setattr(cache, "free_bytes", lambda: 5 * 1024 * 1024)
    assert cache.prepare([ex]) == {"id": None}
    assert not list((tmp_path / "cache").glob("*.npy"))
    monkeypatch.setattr(cache, "free_bytes", lambda: 500 * 1024 * 1024)
    path = cache.prepare([ex])["id"]
    assert path is not None and np.load(path).dtype == np.int16


def test_cache_evicts_least_recently_used(tmp_path):
    directory = tmp_path / "cache"
    directory.mkdir()
    for i in range(4):
        np.save(directory / f"{i}.npy", np.zeros(300_000, np.int16))  # ~0.6 MB each
    import os
    for i in range(4):
        os.utime(directory / f"{i}.npy", (1000 + i, 1000 + i))
    cache = le.AudioCache(directory, max_mb=1.3, min_free_mb=0)
    cache.evict()
    assert sorted(p.name for p in directory.glob("*.npy")) == ["2.npy", "3.npy"]


# ---------------------------------------------------------------------------
# result store + evaluator
# ---------------------------------------------------------------------------


def test_store_reads_incrementally_and_skips_torn_lines(tmp_path):
    store = le.ResultStore(tmp_path / "r.jsonl")
    store.append({"key": "a", "v": 1})
    assert set(store.refresh()) == {"a"}
    with store.path.open("a", encoding="utf-8") as f:
        f.write('{"key": "b", "v"')  # writer killed mid-line
    assert set(store.refresh()) == {"a"}
    store.path.write_text(store.path.read_text(encoding="utf-8") + ': 2}\n', encoding="utf-8")
    assert set(store.refresh()) == {"a", "b"}


class FakeRunner:
    """Stands in for the quality_probe subprocess: 'decodes' every task as
    a perfect transcript of the excerpt's reference."""

    def __init__(self, excerpts: dict[str, le.Excerpt], text_for=None) -> None:
        self.excerpts = excerpts
        self.jobs = 0
        self.text_for = text_for or (lambda x, asr: " ".join(x.reference_words))

    def __call__(self, module, payload, timeout_s):
        self.jobs += 1
        job = json.loads(Path(payload["job_file"]).read_text(encoding="utf-8"))
        store = le.ResultStore(Path(job["store_path"]))
        for task in job["tasks"]:
            x = self.excerpts[task["excerpt_id"]]
            result = SimResult([(x.duration_s, self.text_for(x, task["asr"]))], x.duration_s, 3, 0, 1.0)
            store.append({"key": task["key"], "result": result.to_dict()})
        return {"done": len(job["tasks"]), "audio_seconds": 0.0, "wall_seconds": 0.0}


def test_evaluator_decodes_once_then_reuses_the_store(tmp_path):
    xs = [
        le.Excerpt(f"x{i}", "t", tmp_path / "a.m4a", 0.0, 10.0, tuple((float(k), f"w{k}") for k in range(10)), "asr")
        for i in range(3)
    ]
    runner = FakeRunner({x.id: x for x in xs})
    cache = le.AudioCache(tmp_path / "cache")
    cache.prepare = lambda excerpts, progress=None: {x.id: None for x in excerpts}
    evaluator = le.Evaluator(cache, le.ResultStore(tmp_path / "r.jsonl"), runner=runner, job_dir=tmp_path)
    profile = le.ModelProfile("m", "auto", "fake", DecodeCostModel(0.1, 0.01))
    configs = [AsrConfig(language_whitelist=["pl"]), AsrConfig(language_whitelist=["pl"], step_ms=300)]
    scores = evaluator.evaluate(profile, configs, xs)
    assert runner.jobs == 1
    assert [s.wer for s in scores] == [0.0, 0.0]
    assert all(s.excerpts == 3 and s.missing == 0 for s in scores)
    evaluator.evaluate(profile, configs, xs)
    assert runner.jobs == 1  # everything was already in the store
    assert evaluator.estimate_seconds(profile, configs, xs) == 0.0


def test_evaluator_reports_missing_results(tmp_path):
    xs = [le.Excerpt("x0", "t", tmp_path / "a.m4a", 0.0, 10.0, ((0.0, "a"),), "asr")]
    cache = le.AudioCache(tmp_path / "cache")
    cache.prepare = lambda excerpts, progress=None: {x.id: None for x in excerpts}
    evaluator = le.Evaluator(
        cache, le.ResultStore(tmp_path / "r.jsonl"),
        runner=lambda module, payload, timeout: {"error": "boom"}, job_dir=tmp_path,
    )
    profile = le.ModelProfile("m", "auto", "fake", DecodeCostModel(0.1, 0.01))
    (score,) = evaluator.evaluate(profile, [AsrConfig(language_whitelist=["pl"])], xs)
    assert score.missing == 1 and score.excerpts == 0


def test_store_starts_a_fresh_line_after_a_torn_one(tmp_path):
    store = le.ResultStore(tmp_path / "r.jsonl")
    store.append({"key": "a"})
    with store.path.open("a", encoding="utf-8") as f:
        f.write('{"key": "torn"')  # writer killed mid-line
    store.append({"key": "b"})
    assert set(store.refresh()) == {"a", "b"}


def test_evaluator_skips_a_decode_that_keeps_crashing_the_probe(tmp_path):
    xs = [le.Excerpt(f"x{i}", "t", tmp_path / "a.m4a", 0.0, 10.0, ((0.0, "a"),), "asr") for i in range(4)]

    def crashing_runner(module, payload, timeout_s):
        """Decodes in order and dies on x1 every time (a native crash)."""
        job = json.loads(Path(payload["job_file"]).read_text(encoding="utf-8"))
        store = le.ResultStore(Path(job["store_path"]))
        for task in job["tasks"]:
            if task["excerpt_id"] == "x1":
                return {"error": "exited -11"}
            result = SimResult([(10.0, "a")], 10.0, 1, 0, 1.0)
            store.append({"key": task["key"], "result": result.to_dict()})
        return {"done": len(job["tasks"])}

    cache = le.AudioCache(tmp_path / "cache")
    cache.prepare = lambda excerpts, progress=None: {x.id: None for x in excerpts}
    evaluator = le.Evaluator(cache, le.ResultStore(tmp_path / "r.jsonl"), runner=crashing_runner, job_dir=tmp_path)
    profile = le.ModelProfile("m", "auto", "fake", DecodeCostModel(0.1, 0.01))
    (score,) = evaluator.evaluate(profile, [AsrConfig(language_whitelist=["pl"])], xs)
    assert score.excerpts == 3 and score.missing == 1  # everything after x1 still decoded
    assert len(evaluator.failed) == 1


def test_code_fingerprint_ignores_comments_and_docstrings_only():
    base = 'def f(x):\n    """Doc."""\n    return x + 1  # note\n'
    reworded = 'def f(x):\n    """Other words."""\n    # a comment\n    return x + 1\n'
    changed = 'def f(x):\n    """Doc."""\n    return x + 2\n'
    assert le._code_fingerprint(base) == le._code_fingerprint(reworded)
    assert le._code_fingerprint(base) != le._code_fingerprint(changed)
