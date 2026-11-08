"""The hot-reloadable text settings of the ASR worker: event vocabulary
(prompt hint + optional snapping) and the caption blocklist.

They are resolved in the app process (the named vocabulary list is read
from the config dir) and handed to the worker at spawn and, on a change
that touches nothing else, through its control queue -- so editing them
never reloads the whisper model (see AppController._on_config_changed).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from narrator.asr.vocabulary import resolve_vocabulary
from narrator.captions.blocklist import Blocklist, blocklist_from_config

# AsrConfig fields carried by TextSettings: a change confined to these (and
# the `captions` section) is applied live instead of restarting the worker.
ASR_TEXT_FIELDS = ("vocabulary", "vocabulary_list", "vocabulary_snap")


@dataclass(frozen=True)
class TextSettings:
    vocabulary: tuple[str, ...] = ()
    vocabulary_snap: bool = False
    # None = captions unfiltered (also how a raw/draft path opts out).
    blocklist: Blocklist | None = None


def text_settings_from_config(root_config, vocabulary_dir: Path | None = None) -> TextSettings:
    """Resolve a RootConfig's live text settings."""
    blocklist = blocklist_from_config(root_config.captions)
    return TextSettings(
        vocabulary=tuple(resolve_vocabulary(root_config.asr, vocabulary_dir)),
        vocabulary_snap=root_config.asr.vocabulary_snap,
        blocklist=blocklist or None,
    )


def asr_config_without_text(asr_config):
    """`asr_config` with the TextSettings fields reset, for comparing whether
    anything else changed."""
    defaults = type(asr_config)()
    return asr_config.model_copy(update={name: getattr(defaults, name) for name in ASR_TEXT_FIELDS})
