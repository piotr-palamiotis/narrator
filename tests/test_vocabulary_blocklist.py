"""Event vocabulary (prompt hint + snapping) and the caption blocklist."""
import queue
import threading

import numpy as np
import pytest
from pydantic import ValidationError

from narrator.app_controller import AppController
from narrator.asr.backend import WhisperBackend
from narrator.asr.events import CaptionEventType
from narrator.asr.streaming_decoder import StreamingDecoder
from narrator.asr.text_settings import TextSettings, text_settings_from_config
from narrator.asr.vocabulary import (
    PROMPT_MAX_CHARS,
    VocabularySnapper,
    build_initial_prompt,
    list_vocabulary_lists,
    load_vocabulary_list,
    resolve_vocabulary,
    save_vocabulary_list,
)
from narrator.captions.blocklist import (
    Blocklist,
    BlocklistAction,
    apply_blocklist,
    build_blocklist,
    mask_word,
)
from narrator.config.schema import AsrConfig, CaptionsConfig, RootConfig

# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------


def test_prompt_without_vocabulary_is_the_carryover_untouched():
    long_carryover = "słowo " * 200
    assert build_initial_prompt([], long_carryover) == long_carryover.strip()
    assert build_initial_prompt([], None) == ""


def test_prompt_puts_vocabulary_before_carryover():
    prompt = build_initial_prompt(["Jan Kowalski", "Wrocław"], "poprzednie zdanie")
    assert prompt == "Jan Kowalski, Wrocław. poprzednie zdanie"
    assert build_initial_prompt(["Orlen"], None) == "Orlen."


def test_prompt_truncates_carryover_first_keeping_its_tail():
    vocabulary = ["Jan Kowalski", "Wrocław"]
    carryover = " ".join(f"w{i}" for i in range(400))
    prompt = build_initial_prompt(vocabulary, carryover)
    assert len(prompt) <= PROMPT_MAX_CHARS
    assert prompt.startswith("Jan Kowalski, Wrocław. ")
    assert prompt.endswith(" w399")  # the most recent words survive
    assert " w0 " not in prompt


def test_prompt_vocabulary_alone_over_budget_drops_trailing_entries():
    vocabulary = [f"Nazwisko{i:03d}" for i in range(100)]
    prompt = build_initial_prompt(vocabulary, "carry over")
    assert len(prompt) <= PROMPT_MAX_CHARS
    assert prompt.startswith("Nazwisko000, Nazwisko001")
    assert "Nazwisko099" not in prompt
    assert "carry" not in prompt  # no room left for it


class _FakeModel:
    def __init__(self):
        self.calls = []

    def transcribe(self, audio, **kwargs):
        self.calls.append(kwargs)
        return []


def _fake_backend(config):
    backend = WhisperBackend.__new__(WhisperBackend)
    backend._config = config
    backend._model = _FakeModel()
    backend._vocabulary = list(config.vocabulary)
    backend.rejected_decodes = {}
    return backend


def test_backend_prepends_vocabulary_on_every_call_and_updates_live():
    backend = _fake_backend(AsrConfig(language_whitelist=["pl"], vocabulary=["Kowalski"]))
    audio = np.zeros(16000, dtype=np.float32)
    backend.transcribe(audio, prompt="wcześniej")
    backend.transcribe(audio)
    backend.set_vocabulary([])
    backend.transcribe(audio)
    prompts = [call["initial_prompt"] for call in backend._model.calls]
    assert prompts == ["Kowalski. wcześniej", "Kowalski.", ""]


# ---------------------------------------------------------------------------
# Named vocabulary lists
# ---------------------------------------------------------------------------


def test_vocabulary_list_save_load_and_merge(tmp_path):
    save_vocabulary_list("gala-2026", ["Jan Kowalski", " ", "Orlen"], tmp_path)
    (tmp_path / "notes.txt").write_text("# comment\nWrocław  # venue\n\n", encoding="utf-8")
    assert list_vocabulary_lists(tmp_path) == ["gala-2026", "notes"]
    assert load_vocabulary_list("notes", tmp_path) == ["Wrocław"]
    config = AsrConfig(vocabulary=["Orlen", "Spodek"], vocabulary_list="gala-2026")
    # Inline first, then the list; case/diacritics-blind duplicates dropped.
    assert resolve_vocabulary(config, tmp_path) == ["Orlen", "Spodek", "Jan Kowalski"]


def test_missing_vocabulary_list_is_skipped_not_fatal(tmp_path):
    config = AsrConfig(vocabulary=["Orlen"], vocabulary_list="nope")
    assert resolve_vocabulary(config, tmp_path) == ["Orlen"]


def test_vocabulary_list_name_is_validated():
    with pytest.raises(ValidationError):
        AsrConfig(vocabulary_list="../etc/passwd")
    with pytest.raises(ValueError):
        save_vocabulary_list("a/b", [], __import__("pathlib").Path("/tmp"))


# ---------------------------------------------------------------------------
# Snapping
# ---------------------------------------------------------------------------


def test_snap_fixes_near_misses_and_diacritics():
    snapper = VocabularySnapper(["Jan Kowalski", "Wrocław", "Szczebrzeszyn"])
    assert snapper.apply("pan Kowalsky z Wroclaw.") == "pan Kowalski z Wrocław."
    assert snapper.apply("w Szczebżeszyn") == "w Szczebrzeszyn"  # ż folds to z; one letter missing


def test_snap_leaves_short_words_and_inflections_alone():
    snapper = VocabularySnapper(["Jan Kowalski", "Ewa", "Marek"])
    # "Jan"/"Ewa" are under 5 letters: never targets; "Jen"/"ewy" untouched.
    assert snapper.apply("Jen ewy") == "Jen ewy"
    # Inflected forms (the entry is a prefix) keep their ending.
    assert snapper.apply("Kowalskiego Kowalskim") == "Kowalskiego Kowalskim"
    # Two edits away on a 5-letter word: not a near miss.
    assert snapper.apply("marka") == "marka"
    # Ordinary unrelated words are untouched.
    assert snapper.apply("dzisiaj mówimy o pogodzie") == "dzisiaj mówimy o pogodzie"


# ---------------------------------------------------------------------------
# Blocklist
# ---------------------------------------------------------------------------


def test_blocklist_matches_inflections_case_and_diacritics():
    bl = Blocklist.from_entries(["kurw*", "chuj"])
    assert bl.apply("Kurwa, co za kurwą KURWAMI") == "K***a, co za k***ą K*****I"
    assert bl.apply("CHUJ i chuja") == "C**J i chuja"  # exact entry: no inflection


def test_blocklist_has_no_false_hits_inside_longer_words():
    bl = Blocklist.from_entries(["kurw*", "shit"])
    assert bl.apply("skurwysyn shitake bullshit") == "skurwysyn shitake bullshit"
    # Hyphen/apostrophe separate words: the part is still whole.
    assert bl.apply("o-kurwa shit's") == "o-k***a s**t's"


def test_blocklist_mask_replace_drop():
    text = "to jest kurwa, dobre. Co to kurwa?"
    assert build_blocklist(["kurw*"]).apply(text) == "to jest k***a, dobre. Co to k***a?"
    replace = build_blocklist(["kurw*"], action="replace", replacement="[—]")
    assert replace.apply(text) == "to jest [—], dobre. Co to [—]?"
    drop = build_blocklist(["kurw*"], action=BlocklistAction.DROP)
    assert drop.apply(text) == "to jest dobre. Co to?"
    assert mask_word("ab") == "a*" and mask_word("a") == "*"


def test_builtin_starter_list_is_opt_in():
    assert not build_blocklist([])
    builtin = build_blocklist([], builtin=True)
    assert builtin.apply("o kurwa, what the fuck") == "o k***a, what the f**k"
    assert builtin.apply("pedał rowerowy, suka i szczeniaki") == "pedał rowerowy, suka i szczeniaki"


def test_blocklist_schema_validation():
    assert CaptionsConfig(blocklist=[" kurw* ", "", "kurw*"]).blocklist == ["kurw*"]
    for bad in (["*urwa"], ["ku*rwa"], ["ku*"], ["two words"]):
        with pytest.raises(ValidationError):
            CaptionsConfig(blocklist=bad)
    with pytest.raises(ValidationError):
        CaptionsConfig(blocklist_action="replace", blocklist_replacement=" ")
    assert RootConfig().captions.blocklist == []


def test_apply_blocklist_none_is_the_raw_bypass():
    assert apply_blocklist("kurwa", None) == "kurwa"


# ---------------------------------------------------------------------------
# Decoder integration (before agreement)
# ---------------------------------------------------------------------------


class _Vad:
    level = 1.0

    def process(self, frame):
        return [self.level]


class _Backend:
    resolved_device_description = "fake"

    def __init__(self, responses):
        self._responses = iter(responses)

    def detect_language(self, audio):
        return ("pl", 0.99, 0.9)

    def transcribe(self, audio, language=None, prompt=None):
        return [(0.0, 1.0, next(self._responses))]


class _Clock:
    now = 0.0

    def __call__(self):
        return self.now


def _run_two_steps(responses, text_settings=None, **config_kwargs):
    config = AsrConfig(
        step_ms=100, agreement_passes=2, end_silence_ms=300, language_whitelist=["pl"], **config_kwargs
    )
    events, clock = [], _Clock()
    decoder = StreamingDecoder(
        _Backend(responses), _Vad(), config, on_event=events.append, clock=clock,
        min_decode_seconds=0.01, text_settings=text_settings,
    )
    frame = np.zeros(480, dtype=np.float32)
    for _ in range(2):
        clock.now += 0.101
        decoder.feed(frame)
    return decoder, [e.text for e in events if e.type == CaptionEventType.FINAL]


def test_decoder_blocklist_applies_before_agreement():
    # The hypotheses differ in the profanity's inflection; dropped before
    # agreement, that difference can't hold back the commit.
    settings = TextSettings(blocklist=build_blocklist(["kurw*"], action="drop"))
    _, finals = _run_two_steps(["no kurwa jedziemy dalej", "no kurwy jedziemy dalej teraz"], settings)
    assert finals == ["no jedziemy dalej"]


def test_decoder_without_text_settings_stays_raw():
    # No settings = no blocklist: the raw path (draft.srt, offline eval).
    _, finals = _run_two_steps(["no kurwa jedziemy", "no kurwa jedziemy"])
    assert finals == ["no kurwa jedziemy"]


def test_decoder_snap_uses_inline_vocabulary_and_updates_live():
    decoder, finals = _run_two_steps(
        ["witamy pana Kowalsky", "witamy pana Kowalskj dziś"], vocabulary=["Kowalski"], vocabulary_snap=True
    )
    # Two different misspellings, both snapped to the entry: they now agree.
    assert finals == ["witamy pana Kowalski"]
    decoder.set_text_settings(TextSettings(blocklist=build_blocklist(["witamy"])))
    assert decoder._hypothesis_words([(0, 1, "witamy Kowalsky")]) == ["w****y", "Kowalsky"]


def test_text_settings_from_config_resolves_everything(tmp_path):
    save_vocabulary_list("event", ["Spodek"], tmp_path)
    config = RootConfig.model_validate(
        {
            "asr": {"vocabulary": ["Orlen"], "vocabulary_list": "event", "vocabulary_snap": True},
            "captions": {"blocklist": ["kurw*"]},
        }
    )
    settings = text_settings_from_config(config, tmp_path)
    assert settings.vocabulary == ("Orlen", "Spodek")
    assert settings.vocabulary_snap
    assert settings.blocklist.apply("kurwa") == "k***a"
    assert text_settings_from_config(RootConfig(), tmp_path).blocklist is None


# ---------------------------------------------------------------------------
# AppController: manual captions + hot-reload without an ASR restart
# ---------------------------------------------------------------------------


class _Renderer:
    def __init__(self):
        self.final_texts = []

    def set_partial_preview(self, text):
        pass

    def append_final_text(self, text):
        self.final_texts.append(text)

    def update_config(self, *args):
        pass


def _controller(config):
    controller = AppController.__new__(AppController)
    controller._renderer = _Renderer()
    controller._caption_mode = "manual"
    controller._config = config
    controller._config_lock = threading.Lock()
    return controller


def test_manual_captions_pass_the_blocklist():
    controller = _controller(RootConfig.model_validate({"captions": {"blocklist": ["kurw*"]}}))
    controller.submit_manual_text("  no kurwa  ")
    controller.submit_manual_text("kurwa")
    assert controller._renderer.final_texts == ["no k***a", "k***a"]


def test_manual_caption_fully_dropped_is_not_shown():
    controller = _controller(
        RootConfig.model_validate({"captions": {"blocklist": ["kurw*"], "blocklist_action": "drop"}})
    )
    controller.submit_manual_text("kurwa!")
    assert controller._renderer.final_texts == []


def test_vocabulary_and_blocklist_changes_update_the_worker_without_restart(monkeypatch):
    controller = _controller(RootConfig())
    controller._asr_control_queue = queue.Queue()
    restarts = []
    controller._restart_asr_process = lambda: restarts.append(1)

    new = RootConfig.model_validate({"asr": {"vocabulary": ["Orlen"], "vocabulary_snap": True}})
    controller._on_config_changed(new)
    newer = RootConfig.model_validate(
        {"asr": {"vocabulary": ["Orlen"], "vocabulary_snap": True}, "captions": {"blocklist": ["kurw*"]}}
    )
    controller._on_config_changed(newer)
    assert restarts == []
    first, second = controller._asr_control_queue.get_nowait(), controller._asr_control_queue.get_nowait()
    assert first.vocabulary == ("Orlen",) and first.blocklist is None
    assert second.blocklist.apply("kurwa") == "k***a"

    # Any other asr change still restarts the worker.
    controller._on_config_changed(RootConfig.model_validate({"asr": {"vocabulary": ["Orlen"], "step_ms": 700}}))
    assert restarts == [1]


def test_tuning_cache_keys_on_vocabulary_but_never_writes_it(tmp_path):
    from narrator.config.loader import save_config
    from narrator_tune.autoconfigure import apply_settings
    from narrator_tune.library_eval import config_signature

    base = AsrConfig()
    assert config_signature(base) != config_signature(AsrConfig(vocabulary=["Orlen"]))
    assert config_signature(base) != config_signature(AsrConfig(vocabulary_snap=True))

    path = tmp_path / "config.yaml"
    save_config(RootConfig.model_validate({"asr": {"vocabulary": ["Spodek"]}}), path)
    saved = apply_settings(path, AsrConfig(vocabulary=["Orlen"], vocabulary_snap=True, step_ms=700))
    assert saved.asr.step_ms == 700
    assert saved.asr.vocabulary == ["Spodek"] and not saved.asr.vocabulary_snap
