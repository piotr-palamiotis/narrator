"""Pure helpers of narrator.asr.backend that read whisper token data
(no model needed)."""
from narrator.asr.backend import WordTime, dtw_preset_name, words_from_tokens


def test_words_from_tokens_groups_on_leading_space_and_prefers_dtw():
    tokens = [
        (b" Wit", 0.1, 0.3, -1.0),
        (b"ajcie", 0.3, 0.6, -1.0),
        (b" na", 0.7, 0.8, 0.65),
        (b" spotkaniu", 0.9, 1.4, -1.0),
        (b".", 1.4, 1.5, -1.0),
    ]
    assert words_from_tokens(tokens, window_seconds=10, strip_brackets=True) == [
        WordTime("Witajcie", 0.1, 0.6),
        WordTime("na", 0.65, 0.8),
        WordTime("spotkaniu.", 0.9, 1.5),
    ]


def test_words_from_tokens_joins_split_utf8_and_clamps_to_window():
    # "ż" split across two tokens; the last timestamp runs past the audio.
    zet = "ż".encode()
    tokens = [(b" " + zet[:1], 0.0, 0.2, -1.0), (zet[1:] + b"ywo", 0.2, 9.0, -1.0)]
    assert words_from_tokens(tokens, window_seconds=2.0, strip_brackets=False) == [WordTime("żywo", 0.0, 2.0)]


def test_words_from_tokens_drops_bracketed_annotations_when_stripping():
    tokens = [(b" [w", 0, 1, -1), (b"ypowied\xc5\xba", 1, 2, -1), (b" w", 2, 3, -1), (b" j\xc4\x99zyku]", 3, 4, -1),
              (b" tak", 4, 5, -1)]
    assert [w.text for w in words_from_tokens(tokens, window_seconds=9, strip_brackets=True)] == ["tak"]
    assert len(words_from_tokens(tokens, window_seconds=9, strip_brackets=False)) == 4


def test_dtw_preset_name():
    assert dtw_preset_name("large-v3") == "WHISPER_AHEADS_LARGE_V3"
    assert dtw_preset_name("large-v3-turbo-q5_0") == "WHISPER_AHEADS_LARGE_V3_TURBO"
    assert dtw_preset_name("/models/ggml-small.bin") == "WHISPER_AHEADS_SMALL"
    assert dtw_preset_name("medium.en") == "WHISPER_AHEADS_MEDIUM_EN"
    assert dtw_preset_name("my-finetune") is None
