"""Partial-caption display semantics (todo B Phase 4): a PARTIAL replaces
the whole in-progress text, a FINAL replaces the open partial atomically,
and the optional typewriter pacing (narrator.captions.pacer, a port of
obs-localvocal's TokenBufferThread) on top of the caption buffer."""
from __future__ import annotations

import pytest

from narrator.captions.pacer import CaptionPacer, split_units
from narrator.config.schema import RootConfig
from narrator.render.canvas import CaptionRenderer


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def renderer(pacing="off", max_lines=3):
    config = RootConfig.model_validate({"layout": {"pacing": pacing, "max_lines": max_lines}})
    r = CaptionRenderer(config.canvas, config.layout, config.style)
    clock = Clock()
    r._clock = clock
    return r, clock


def shown(r) -> str:
    return " ".join(r.visible_lines())


# -- unpaced: partial replace / final replace ---------------------------------


def test_partial_replaces_the_previous_partial_not_appends():
    r, _ = renderer()
    r.append_final_text("Dzień dobry.")
    r.set_partial_preview("Witam")
    r.set_partial_preview("Witam państwa")
    assert shown(r) == "Dzień dobry. Witam państwa"


def test_final_replaces_the_open_partial_atomically():
    r, _ = renderer()
    r.set_partial_preview("Witam państwa na")
    r.append_final_text("Witam państwa na sali.")
    assert shown(r) == "Witam państwa na sali."
    assert r.snapshot_text() == "Witam państwa na sali."  # the partial was never stored


def test_empty_partial_clears_the_preview():
    r, _ = renderer()
    r.append_final_text("Raz.")
    r.set_partial_preview("dwa")
    r.set_partial_preview("")
    assert shown(r) == "Raz."


def test_prune_keeps_the_visible_text_and_clear_drops_the_partial():
    r, _ = renderer(max_lines=1)
    for i in range(30):
        r.append_final_text(f"zdanie numer {i} jest dość długie żeby się zawijało na ekranie")
    before = r.visible_lines()
    assert r.prune_history() > 0
    assert r.visible_lines() == before  # CLEAR never blanks what is on screen
    r.set_partial_preview("coś")
    r.clear()
    assert r.visible_lines() == ()


# -- the pacer ------------------------------------------------------------------


def test_split_units_keep_leading_spaces():
    assert split_units("Ala  ma kota", "word") == ["Ala", " ma", " kota"]
    assert "".join(split_units("Ala ma", "char")) == "Ala ma"
    assert split_units("Raz. Dwa? Trzy", "sentence") == ["Raz.", " Dwa?", " Trzy"]
    assert split_units("   ", "word") == []


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        CaptionPacer("off")


def test_word_pacing_reveals_one_word_at_a_time_first_at_once():
    p = CaptionPacer("word")
    p.push_final("Ala ma kota")
    commits, tail = p.advance(0.0)
    assert (commits, tail) == ([], "Ala")  # idle: the first unit appears at once
    commits, tail = p.advance(0.1)
    assert (commits, tail) == ([], "Ala")
    commits, tail = p.advance(0.2)
    assert (commits, tail) == (["Ala"], "ma")
    commits, tail = p.advance(0.4)
    assert (commits, tail) == (["ma", "kota"], "")
    assert p.backlog() == 0


def test_speed_adapts_to_the_backlog():
    p = CaptionPacer("word")
    p.push_final(" ".join(f"w{i}" for i in range(30)))
    p.advance(0.0)
    assert p._interval() == pytest.approx(0.100)  # > 10 queued words: fastest
    p2 = CaptionPacer("word")
    p2.push_final("a b c")
    assert p2._interval() == pytest.approx(0.200)
    c = CaptionPacer("char")
    c.push_final("x" * 50)
    assert c._interval() == pytest.approx(0.033)


def test_new_partial_pops_what_it_does_not_share():
    p = CaptionPacer("word")
    p.push_partial("Ala ma psa")
    t = 0.0
    for _ in range(3):
        p.advance(t)
        t += 0.2
    assert p._tail() == "Ala ma psa"
    p.push_partial("Ala ma kota i psa")
    assert p._tail() == "Ala ma"  # "psa" popped, the shared prefix stays
    while p.backlog():
        p.advance(t)
        t += 0.2
    assert p._tail() == "Ala ma kota i psa"


def test_final_keeps_the_shown_partial_prefix_and_types_the_rest():
    p = CaptionPacer("word")
    p.push_partial("Ala ma")
    p.advance(0.0)
    p.advance(0.2)
    p.push_final("Ala ma kota")
    commits, tail = p.advance(0.21)
    assert " ".join(commits + [tail]).split() == ["Ala", "ma"]  # nothing shown is taken back
    commits2, tail2 = p.advance(0.5)
    assert " ".join(commits + commits2 + [tail2]).split() == ["Ala", "ma", "kota"]


def test_partials_wait_behind_queued_finals():
    p = CaptionPacer("word")
    p.push_final("raz dwa trzy")
    p.push_partial("cztery")
    commits, tail = p.advance(0.0)
    assert tail == "raz"
    all_commits = list(commits)
    t = 0.0
    for _ in range(10):
        t += 0.2
        c, tail = p.advance(t)
        all_commits += c
    assert all_commits == ["raz", "dwa", "trzy"] and tail == "cztery"


def test_flush_and_reset():
    p = CaptionPacer("char")
    p.push_final("Ala ma kota")
    p.push_partial("i psa")
    commits, tail = p.flush()
    assert " ".join(commits) == "Ala ma kota" and tail == "i psa"
    p.push_final("coś")
    p.reset()
    assert p.advance(10.0) == ([], "")


# -- the pacer inside the renderer ------------------------------------------------


def test_renderer_paces_finals_and_never_shows_a_partial_twice():
    r, clock = renderer(pacing="word")
    r.set_partial_preview("Witam")
    assert shown(r) == "Witam"
    clock.now += 0.2
    r.set_partial_preview("Witam państwa")
    assert shown(r) == "Witam państwa"
    r.append_final_text("Witam państwa serdecznie")
    seen = []
    for _ in range(10):
        clock.now += 0.2
        seen.append(shown(r))
    assert seen[-1] == "Witam państwa serdecznie"
    for text in seen:  # every frame is a prefix of the FINAL: no doubled words
        assert "Witam państwa serdecznie".startswith(text)
    assert r.snapshot_text() == "Witam państwa serdecznie"


def test_switching_pacing_off_flushes_held_text():
    r, clock = renderer(pacing="sentence")
    r.append_final_text("Raz. Dwa. Trzy.")
    assert shown(r) == "Raz."
    config = RootConfig.model_validate({"layout": {"pacing": "off", "max_lines": 3}})
    r.update_config(config.canvas, config.layout, config.style)
    assert shown(r) == "Raz. Dwa. Trzy."


def test_renderer_clear_drops_paced_backlog():
    r, clock = renderer(pacing="word")
    r.append_final_text("jeden dwa trzy cztery")
    shown(r)
    r.clear()
    clock.now += 5
    assert r.visible_lines() == ()


def _reveal_times(p, until, fps=30.0, start=0.0):
    """Frame times at which each queued unit became visible (no pushes
    happen meanwhile, so the backlog only shrinks)."""
    times, before, t = [], p.backlog(), start
    while t <= until:
        p.advance(t)
        after = p.backlog()
        times += [t] * (before - after)
        before = after
        t = round(t + 1.0 / fps, 6)
    return times


def test_smooth_types_bursts_at_the_speakers_steady_rate():
    import narrator.captions.pacer as pacer_module

    p = CaptionPacer("smooth")
    # a speaker producing 15 characters/s, committed in bursts of 30 every 2 s
    times, t, before = [], 0.0, 0
    for k in range(8):
        p.push_final("abcdefghij" * 3)
        before += 30
        while t < 2.0 * (k + 1):
            p.advance(t)
            revealed = before - p.backlog()
            times += [t] * (revealed - len(times))
            t = round(t + 1 / 30, 6)
    # after the first seconds the typing never stalls: every 0.5 s window
    # shows characters, at about the speaker's rate (not 30 at once)
    steady = [x for x in times if x >= 4.0]
    counts = [sum(1 for x in steady if w <= x < w + 0.5) for w in [4.0 + 0.5 * i for i in range(23)]]
    assert min(counts) >= 4 and max(counts) <= 13
    assert p.backlog() <= 15 * pacer_module.SMOOTH_LAG * 2  # the queue stays near the target lag


def test_smooth_rate_follows_the_backlog_within_bounds():
    import narrator.captions.pacer as pacer_module

    low, high = pacer_module.SMOOTH_RATE_BOUNDS
    p = CaptionPacer("smooth")
    p.push_final("x" * 400)
    p.advance(0.0)
    assert p.smooth_rate() == pytest.approx(high)  # a big backlog: fastest, capped
    q = CaptionPacer("smooth")
    q.push_final("ab")
    q.advance(0.0)
    assert q.smooth_rate() == pytest.approx(low)  # a lone short caption: slowest bound
    r = CaptionPacer("smooth")
    r.push_final("y" * 60)  # 60 chars in the rate window
    r.advance(10.0)
    speaker = 60 / pacer_module.SMOOTH_RATE_WINDOW
    expected = speaker + (r.backlog() - speaker * pacer_module.SMOOTH_LAG) / pacer_module.SMOOTH_CATCHUP
    assert r.smooth_rate() == pytest.approx(expected)


def test_smooth_keeps_text_and_commits_whole_words():
    renderer("smooth")  # the schema and renderer accept the mode
    p = CaptionPacer("smooth")
    p.push_final("Ala ma kota")
    out, t = [], 0.0
    while t < 3.0:
        commits, tail = p.advance(t)
        out += commits
        t += 1 / 30
    assert out == ["Ala", "ma", "kota"] and p.backlog() == 0
