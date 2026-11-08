"""Transcript file output (captions/transcript.py)."""
from datetime import datetime

from narrator.asr.events import CaptionEvent, CaptionEventType
from narrator.captions.transcript import TranscriptWriter, format_srt_time
from narrator.config.schema import TranscriptConfig

NOW = lambda: datetime(2026, 10, 13, 14, 5, 0)  # noqa: E731


def final(text, start, end):
    return CaptionEvent(CaptionEventType.FINAL, text, start=start, end=end)


def make(tmp_path, fmt="srt"):
    config = TranscriptConfig(enabled=True, directory=str(tmp_path), format=fmt)
    return TranscriptWriter(config, clock=lambda: 100.0, now=NOW)


def test_srt_time_format():
    assert format_srt_time(0) == "00:00:00,000"
    assert format_srt_time(3725.5) == "01:02:05,500"


def test_fragments_are_grouped_into_sentence_cues(tmp_path):
    writer = make(tmp_path)
    assert writer.path == tmp_path / "narrator-2026-10-13_14-05-00.srt"
    writer.add_final(final("Dzień dobry", 101.0, 101.8))
    writer.add_final(final("wszystkim na sali.", 101.8, 103.0))
    writer.add_final(final("Zaczynamy", 103.2, 103.9))
    writer.close()
    assert writer.path.read_text(encoding="utf-8") == (
        "1\n00:00:01,000 --> 00:00:03,000\nDzień dobry wszystkim na sali.\n\n"
        "2\n00:00:03,200 --> 00:00:03,900\nZaczynamy\n\n"
    )


def test_pause_splits_a_cue(tmp_path):
    writer = make(tmp_path, fmt="txt")
    writer.add_final(final("pierwsza myśl", 101.0, 102.0))
    writer.add_final(final("po przerwie", 105.0, 106.0))
    writer.close()
    assert writer.path.read_text(encoding="utf-8") == "pierwsza myśl\npo przerwie\n"


def test_long_cue_is_cut_at_subtitle_length(tmp_path):
    writer = make(tmp_path, fmt="txt")
    for i in range(12):
        writer.add_final(final("słowo słowo", 101.0 + i * 0.5, 101.5 + i * 0.5))
    writer.close()
    lines = writer.path.read_text(encoding="utf-8").splitlines()
    assert len(lines) >= 2 and all(len(line) <= 84 + len("słowo słowo") for line in lines)


def test_cues_never_overlap_or_vanish(tmp_path):
    writer = make(tmp_path)
    writer.add_final(final("Jedno zdanie tutaj, dość długie.", 101.0, 103.0))
    writer.add_final(final("Drugie zdanie zaczęte wcześniej.", 102.5, 102.6))
    writer.close()
    cues = writer.path.read_text(encoding="utf-8").strip().split("\n\n")
    assert "00:00:03,000 --> 00:00:03,500" in cues[1]


def test_falls_back_to_emit_time_without_spoken_times(tmp_path):
    writer = make(tmp_path, fmt="srt")
    writer.add_final(CaptionEvent(CaptionEventType.FINAL, "Bez czasu.", timestamp=110.0))
    writer.close()
    assert "00:00:10,000 --> 00:00:10,500" in writer.path.read_text(encoding="utf-8")


def test_write_failure_disables_the_transcript(tmp_path):
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("x")
    writer = TranscriptWriter(TranscriptConfig(enabled=True, directory=str(blocker)), now=NOW)
    writer.add_final(final("To się nie zapisze.", 1.0, 2.0))  # must not raise
    writer.add_final(final("Ani to.", 3.0, 4.0))
    writer.close()
    assert not writer.path.exists()
