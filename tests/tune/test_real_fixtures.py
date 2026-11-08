"""Unit tests for real-recording fixtures (no GPU, no subprocess)."""
from __future__ import annotations

from pathlib import Path

import pytest

import shutil
import wave

import numpy as np

from narrator_tune.real_fixtures import (
    Cue,
    collapse_reference_cues,
    discover_recordings,
    excerpt_reference_words,
    load_audio_16k_mono,
    load_reference,
    parse_reference_tag,
    parse_srt,
    read_pcm_windows,
    select_excerpt_windows,
)
from narrator_tune.real_fixtures import _normalize_recording_stem, _proportional_allocation


def make_srt(tmp_path: Path, cues: list[tuple[str, str, str]], name: str = "rec") -> Path:
    """Write an SRT from (start, end, text) 'HH:MM:SS,mmm' tuples."""
    lines = []
    for i, (start, end, text) in enumerate(cues, 1):
        lines.append(f"{i}\n{start} --> {end}\n{text}\n")
    srt = tmp_path / f"{name}.srt"
    srt.write_text("\n".join(lines), encoding="utf-8")
    return srt


# ---------------------------------------------------------------------------
# SRT parsing + rollup collapse
# ---------------------------------------------------------------------------


def test_parse_srt_handles_multiline_and_bom(tmp_path):
    srt = tmp_path / "x.srt"
    srt.write_text(
        "\ufeff1\n00:00:01,000 --> 00:00:03,000\nhello\nworld\n\n2\n00:00:04,000 --> 00:00:05,000\nnext\n",
        encoding="utf-8",
    )
    cues = parse_srt(srt)
    assert len(cues) == 2
    assert cues[0].start == 1.0 and cues[0].end == 3.0
    assert cues[0].text == "hello world"


def test_rollup_echo_collapsed():
    cues = [
        Cue(0.0, 2.0, "everything when you close"),
        Cue(2.0, 4.0, "your ey when you close your ey"),
        Cue(4.0, 6.0, "mam nadzieję, że przez wakacje"),
    ]
    words, non_speech, foreign = collapse_reference_cues(cues)
    assert non_speech == [] and foreign == []
    # the echo "when you close your ey" is not double-counted
    text = " ".join(words)
    assert text.count("close") == 2  # once per genuine occurrence, not 3x
    assert "mam nadzieję" in text


# ---------------------------------------------------------------------------
# non-speech annotations
# ---------------------------------------------------------------------------


def test_non_speech_cues_dropped_and_windowed():
    cues = [
        Cue(0.0, 5.0, "[Muzyka]"),
        Cue(5.0, 7.0, "mowa polska"),
        Cue(7.0, 12.0, "[Aplauz]"),
    ]
    ref_words, non_speech, foreign = collapse_reference_cues(cues)
    assert ref_words == ["mowa", "polska"]
    assert non_speech == [(0.0, 5.0), (7.0, 12.0)]
    assert foreign == []


# ---------------------------------------------------------------------------
# foreign-language detection (diacritic density over cue groups)
# ---------------------------------------------------------------------------


def test_english_song_group_excluded():
    """An English song between applause gaps is excluded as foreign; the
    Polish cue after the gap stays in."""
    cues = [
        Cue(0.0, 5.0, "cant remember color shining through"),
        Cue(5.0, 10.0, "thats why i love show their true colors"),
        Cue(10.0, 15.0, "true colors"),
        # 10s gap (applause break in reality) splits the groups here
        Cue(25.0, 30.0, "to jest już mowa polska z diakrytykami"),
    ]
    ref_words, non_speech, foreign = collapse_reference_cues(cues)
    assert foreign == [(0.0, 15.0)]  # the English group
    assert " ".join(ref_words) == "to jest już mowa polska z diakrytykami"


def test_short_ambiguous_group_stays_in():
    """A <12-word group can't reach density significance -- it stays in and is
    scored (conservatively counting as errors rather than hiding content)."""
    cues = [Cue(0.0, 3.0, "ah ah ah")]
    ref_words, _, foreign = collapse_reference_cues(cues)
    assert foreign == []
    assert ref_words == ["ah", "ah", "ah"]


def test_polish_group_with_few_diacritic_words_kept():
    """A couple of marked words in a short group keeps it Polish (density
    well over the 2% foreign threshold)."""
    cues = [
        Cue(0.0, 6.0, "ala ma Goldena więc daje mu coś over the rainbow today"),
        Cue(6.0, 12.0, "no i co ty na to całą tą sytuacją jest bardzo dziwną"),
    ]
    _, _, foreign = collapse_reference_cues(cues)
    assert foreign == []


# ---------------------------------------------------------------------------
# load_reference: speech spans subtract masked windows
# ---------------------------------------------------------------------------


def test_load_reference_subtracts_masked_windows(tmp_path):
    srt = make_srt(
        tmp_path,
        [
            ("00:00:00,000", "00:00:10,000", "[Muzyka]"),
            ("00:00:10,000", "00:00:30,000", "pierwsze zdanie po muzyce jest tutaj"),
            ("00:00:30,000", "00:00:40,000", "[Aplauz]"),
            ("00:00:40,000", "00:01:00,000", "druga część przemowy po aplauzie"),
        ],
    )
    ref = load_reference(srt)
    assert (10.0, 30.0) in ref.speech_spans
    assert all(not (30.0 <= s < 40.0) for s, e in ref.speech_spans for _ in [0])
    assert ref.non_speech_windows == [(0.0, 10.0), (30.0, 40.0)]


# ---------------------------------------------------------------------------
# excerpt selection
# ---------------------------------------------------------------------------


def _ts(seconds: float) -> str:
    whole = int(seconds)
    return f"{whole // 3600:02d}:{whole % 3600 // 60:02d}:{whole % 60:02d},000"


def _dense_reference(tmp_path, blocks: list[tuple[float, float]], name="rec") -> object:
    """SRT whose Polish-speech cues fill the given time blocks (plus [Aplauz]
    padding between blocks so they form separate regions)."""
    cues = []
    for s, e in blocks:
        cues.append((_ts(s), _ts(e), "mowa polska o szkole"))
        cues.append((_ts(e), _ts(e + 30), "[Aplauz]"))
    return load_reference(make_srt(tmp_path, cues, name=name))


def test_excerpt_windows_skip_music_and_land_in_speech(tmp_path):
    """The regression that motivated the region-based placement: even spacing
    over the raw timeline put windows on the concert intro and applause tail
    -- all music, all rejected. Windows must land inside speech regions."""
    # 0-100s music cue, then 100-250s speech, applause, 300-450s speech
    cues = [
        ("00:00:00,000", "00:01:40,000", "[Muzyka]"),
        ("00:01:40,000", "00:04:10,000", "mowa polska o szkole i o nauce"),
        ("00:04:10,000", "00:04:40,000", "[Aplauz]"),
        ("00:04:40,000", "00:07:10,000", "druga część przemowy pełna polskich słów"),
    ]
    ref = load_reference(make_srt(tmp_path, cues))
    windows = select_excerpt_windows(ref, 2, 60.0)
    assert len(windows) == 2
    for start, end in windows:
        speech = sum(
            max(0.0, min(end, se) - max(start, ss)) for ss, se in ref.speech_spans
        )
        assert speech >= 0.5 * 60.0  # speech-dense by the same rule as the code


def test_excerpt_windows_deterministic(tmp_path):
    ref = _dense_reference(tmp_path, [(0.0, 200.0), (300.0, 500.0)])
    a = select_excerpt_windows(ref, 2, 60.0)
    b = select_excerpt_windows(ref, 2, 60.0)
    assert len(a) == 2
    assert a == b


def test_single_window_lands_on_the_longest_region(tmp_path):
    """count=1 used to land on the LAST host (usually the outro)."""
    ref = _dense_reference(tmp_path, [(0.0, 90.0), (120.0, 300.0), (330.0, 420.0)])
    ((start, end),) = select_excerpt_windows(ref, 1, 60.0)
    assert 120.0 <= start and end <= 300.0


def test_proportional_allocation_favors_longer_hosts():
    assert _proportional_allocation(1, [90.0, 180.0, 90.0]) == [0, 1, 0]
    assert sum(_proportional_allocation(5, [10.0, 20.0, 30.0])) == 5


def test_seeded_windows_are_deterministic_and_stay_in_speech(tmp_path):
    ref = _dense_reference(tmp_path, [(0.0, 200.0), (300.0, 500.0)])
    a = select_excerpt_windows(ref, 3, 60.0, seed="film")
    assert a == select_excerpt_windows(ref, 3, 60.0, seed="film")
    assert a != select_excerpt_windows(ref, 3, 60.0, seed="another film")
    for start, end in a:
        assert any(s - 3.0 <= start and end <= e + 3.0 for s, e in [(0.0, 200.0), (300.0, 500.0)])


def test_window_edges_snap_to_cue_starts(tmp_path):
    cues = [(_ts(s), _ts(s + 7), "mowa polska tutaj") for s in range(0, 300, 7)]
    ref = load_reference(make_srt(tmp_path, cues))
    for start, end in select_excerpt_windows(ref, 2, 60.0, seed="x"):
        assert start in ref.cue_starts
        assert end in ref.cue_starts


def test_excerpt_windows_degenerate_short_recording(tmp_path):
    """A recording whose longest region is shorter than one window still
    yields one shrunk window (or none if truly tiny)."""
    ref = _dense_reference(tmp_path, [(0.0, 40.0)])
    windows = select_excerpt_windows(ref, 4, 75.0)
    assert len(windows) == 1
    start, end = windows[0]
    assert end - start <= 40.0 * 0.9 + 0.1


# ---------------------------------------------------------------------------
# excerpt reference words
# ---------------------------------------------------------------------------


def test_excerpt_reference_words_scoped_to_window(tmp_path):
    srt = make_srt(
        tmp_path,
        [
            ("00:00:00,000", "00:00:10,000", "pierwsze zdanie w oknie"),
            ("00:00:10,000", "00:00:20,000", "drugie zdanie w oknie"),
            ("00:01:00,000", "00:01:10,000", "trzecie daleko poza oknem"),
        ],
    )
    ref = load_reference(srt)
    words_ = excerpt_reference_words(ref, 0.0, 20.0)
    assert " ".join(words_) == "pierwsze zdanie w oknie drugie zdanie w oknie"
    assert "daleko" not in " ".join(words_)


def test_excerpt_reference_words_select_by_word_time_not_cue_overlap(tmp_path):
    """A cue that merely overlaps the window edge contributes only the words
    estimated to fall inside it -- whole overlapping cues used to leak in."""
    srt = make_srt(
        tmp_path,
        [
            ("00:00:00,000", "00:00:08,000", "jeden dwa trzy cztery"),  # ~1,3,5,7 s
            ("00:00:08,000", "00:00:16,000", "pięć sześć siedem osiem"),  # ~9,11,13,15 s
        ],
    )
    ref = load_reference(srt)
    assert excerpt_reference_words(ref, 4.0, 12.0) == ["trzy", "cztery", "pięć", "sześć"]


def test_timed_words_use_effective_span_and_skip_rollup_echo(tmp_path):
    """Rollup cues stay on screen while the next line is spoken: a cue's
    words are spread over [start, next cue start], and an echoed prefix is
    neither counted nor timed twice."""
    srt = make_srt(
        tmp_path,
        [
            ("00:00:00,000", "00:00:08,000", "ala ma kota"),
            ("00:00:03,000", "00:00:09,000", "ala ma kota a kot ma psa"),
        ],
    )
    ref = load_reference(srt)
    assert [w for _, w in ref.timed_words] == "ala ma kota a kot ma psa".split()
    times = [t for t, _ in ref.timed_words]
    assert times[:3] == pytest.approx([0.5, 1.5, 2.5])  # first cue: 0-3 s, not 0-8 s
    assert all(3.0 <= t <= 9.0 for t in times[3:])
    assert ref.cue_starts == [0.0, 3.0]


# ---------------------------------------------------------------------------
# reference language tags
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "stem, expected",
    [
        ("Film (Polish_ASR)", ("pl", "asr")),
        ("Film (Polish_corrected)", ("pl", "corrected")),
        ("narrator 2026-10-16_18-00-00 (Polish_corrected)", ("pl", "corrected")),
        ("Film (Polish)", ("pl", "human")),
        ("Film (English_ASR)", ("en", "asr")),
        ("Film (napisy PL) (English)", ("en", "human")),
        ("Film (Manx)", (None, "human")),
        ("school_assembly", ("pl", "unknown")),
        ("Bocznica (2009)", ("pl", "unknown")),  # a year is part of the title
    ],
)
def test_parse_reference_tag(stem, expected):
    assert parse_reference_tag(stem) == expected


def test_recording_exposes_reference_language(tmp_path):
    (tmp_path / "Film (128kbit_AAC).m4a").write_bytes(b"x")
    (tmp_path / "Film (English_ASR).srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nx\n", encoding="utf-8")
    (rec,) = discover_recordings(tmp_path)
    assert (rec.reference_language, rec.reference_kind, rec.title) == ("en", "asr", "Film")


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------


def test_discover_requires_matching_srt(tmp_path):
    (tmp_path / "good.wav").write_bytes(b"RIFF")
    (tmp_path / "good.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nx\n", encoding="utf-8")
    (tmp_path / "orphan.mp3").write_bytes(b"ID3")
    (tmp_path / "notes.txt").write_text("not audio")

    found = discover_recordings(tmp_path)
    assert [r.stem for r in found] == ["good"]
    assert found[0].audio_path.suffix == ".wav"


def test_discover_missing_dir_returns_empty(tmp_path):
    assert discover_recordings(tmp_path / "nope") == []


# ---------------------------------------------------------------------------
# suffix normalization (codec tags on audio, language tags on SRT)
# ---------------------------------------------------------------------------


def test_normalize_stem_strips_codec_and_language_suffixes():
    assert _normalize_recording_stem("My Event (128kbit_AAC)") == "My Event"
    assert _normalize_recording_stem("My Event (152kbit_Opus)") == "My Event"
    assert _normalize_recording_stem("My Event (Polish_ASR)") == "My Event"
    assert _normalize_recording_stem("My Event (English_ASR)") == "My Event"
    assert _normalize_recording_stem("My Event (English)") == "My Event"
    assert _normalize_recording_stem("My Event (128kbit_AAC-Polish)") == "My Event"


def test_normalize_stem_no_suffix_unchanged():
    assert _normalize_recording_stem("school_assembly") == "school_assembly"


def test_discover_matches_audio_with_codec_suffix_to_srt_with_language_suffix(tmp_path):
    """The real-world naming scheme: audio carries a codec tag, the SRT carries
    a language tag. The base titles must match after both tags are stripped."""
    (tmp_path / "Historia telewizji część 1 (128kbit_AAC).m4a").write_bytes(b"fake")
    (tmp_path / "Historia telewizji część 1 (Polish_ASR).srt").write_text(
        "1\n00:00:00,000 --> 00:00:01,000\nx\n", encoding="utf-8"
    )

    found = discover_recordings(tmp_path)
    assert len(found) == 1
    assert found[0].audio_path.name == "Historia telewizji część 1 (128kbit_AAC).m4a"
    assert found[0].srt_path.name == "Historia telewizji część 1 (Polish_ASR).srt"


def test_discover_multiple_suffix_variants(tmp_path):
    """AAC audio + Polish SRT, Opus audio + English SRT, no-suffix pair."""
    (tmp_path / "Event A (128kbit_AAC).m4a").write_bytes(b"fake")
    (tmp_path / "Event A (Polish_ASR).srt").write_text(
        "1\n00:00:00,000 --> 00:00:01,000\nx\n", encoding="utf-8"
    )
    (tmp_path / "Event B (152kbit_Opus).opus").write_bytes(b"fake")
    (tmp_path / "Event B (English_ASR).srt").write_text(
        "1\n00:00:00,000 --> 00:00:01,000\nx\n", encoding="utf-8"
    )
    (tmp_path / "Event C.wav").write_bytes(b"RIFF")
    (tmp_path / "Event C.srt").write_text(
        "1\n00:00:00,000 --> 00:00:01,000\nx\n", encoding="utf-8"
    )

    found = discover_recordings(tmp_path)
    assert len(found) == 3
    stems = sorted(r.stem for r in found)  # stem comes from audio_path.stem
    assert "Event A (128kbit_AAC)" in stems
    assert "Event B (152kbit_Opus)" in stems
    assert "Event C" in stems


def test_discover_orphan_audio_with_suffix_is_skipped(tmp_path):
    """An audio file whose base title has no SRT at all is skipped."""
    (tmp_path / "Lonely Clip (128kbit_AAC).m4a").write_bytes(b"fake")
    (tmp_path / "Other Clip (Polish_ASR).srt").write_text(
        "1\n00:00:00,000 --> 00:00:01,000\nx\n", encoding="utf-8"
    )

    found = discover_recordings(tmp_path)
    assert found == []


# ---------------------------------------------------------------------------
# audio decoding through the GStreamer pipe (no temp files)
# ---------------------------------------------------------------------------


needs_gst = pytest.mark.skipif(shutil.which("gst-launch-1.0") is None, reason="GStreamer not installed")


def _ramp_wav(path, seconds: float) -> np.ndarray:
    samples = (np.arange(int(seconds * 16000)) % 30000).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(samples.tobytes())
    return samples


@needs_gst
def test_read_pcm_windows_returns_exact_samples(tmp_path):
    wav = tmp_path / "ramp.wav"
    samples = _ramp_wav(wav, 4.0)
    first, second, past_end = read_pcm_windows(wav, [(0.5, 1.0), (2.0, 3.5), (3.9, 5.0)])
    assert np.array_equal(first, samples[8000:16000])
    assert np.array_equal(second, samples[32000:56000])
    assert len(past_end) == 1600  # clipped at the end of the audio, no error


@needs_gst
def test_load_audio_16k_mono_whole_file(tmp_path):
    wav = tmp_path / "ramp.wav"
    samples = _ramp_wav(wav, 1.0)
    audio = load_audio_16k_mono(wav)
    assert audio.dtype == np.float32
    assert np.allclose(audio * 32768.0, samples)


@needs_gst
def test_read_pcm_windows_surfaces_decoder_errors(tmp_path):
    bogus = tmp_path / "not_audio.m4a"
    bogus.write_text("plain text, not audio")
    with pytest.raises(RuntimeError):
        read_pcm_windows(bogus, [(0.0, 1.0)])


def test_corrected_srt_wins_and_is_an_objective_reference(tmp_path):
    """A narrator recording corrected by hand ("Use as reference") is the best
    reference there is: it beats the auto-captions of the same title, and it
    drives the objective (not the separate human-subtitle check)."""
    from narrator_tune.library_eval import scan_library

    cue = "1\n00:00:00,000 --> 00:00:01,000\nx\n"
    (tmp_path / "Talk (FLAC).flac").write_bytes(b"x")
    for tag in ("Polish_ASR", "Polish_corrected", "Polish"):
        (tmp_path / f"Talk ({tag}).srt").write_text(cue, encoding="utf-8")
    (rec,) = discover_recordings(tmp_path)
    assert rec.srt_path.name == "Talk (Polish_corrected).srt"
    assert rec.reference_kind == "corrected"
    library = scan_library(tmp_path)
    assert [r.srt_path.name for r in library.recordings] == ["Talk (Polish_corrected).srt"]
    assert library.trusted == []
