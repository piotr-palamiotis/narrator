"""Option lists for config pickers (the web UI's dropdowns).

Pure lookups, no UI toolkit: moved here from the retired Tk GUI so the web
layer (and anything else) can reuse them.
"""
from __future__ import annotations

import sys
from pathlib import Path

_FONT_EXTENSIONS = {".ttf", ".otf"}


def model_dropdown_options(current: str = "") -> list[str]:
    """Options for the Whisper model dropdown: every model name pywhispercpp
    can download on demand (its canonical list), plus any model already
    cached locally (narrator.paths search dirs), so a custom name or a direct
    path to a ggml file -- which config also accepts -- stays selectable
    instead of being lost on the next save. The field stays free-text
    editable for the same reason."""
    try:
        from pywhispercpp.constants import AVAILABLE_MODELS

        from narrator.paths import cached_model_names

        options = sorted({*AVAILABLE_MODELS, *cached_model_names()})
    except Exception:  # pragma: no cover - only on broken installs
        options = ["large-v3", "medium", "small", "base", "tiny"]
    if current and current not in options:
        options.insert(0, current)
    return options


def fonts_dir() -> Path:
    """assets/fonts in a checkout, or inside a PyInstaller bundle."""
    bundle = getattr(sys, "_MEIPASS", None)
    if bundle:
        return Path(bundle) / "assets" / "fonts"
    return Path(__file__).resolve().parents[3] / "assets" / "fonts"


def font_options(current: str = "") -> list[str]:
    """Bundled font files as config values ("assets/fonts/<file>"), plus the
    configured one if it is elsewhere."""
    directory = fonts_dir()
    options = (
        sorted(f"assets/fonts/{p.name}" for p in directory.iterdir() if p.suffix.lower() in _FONT_EXTENSIONS)
        if directory.is_dir()
        else []
    )
    if current and current not in options:
        options.insert(0, current)
    return options


def language_options() -> list[tuple[str, str]]:
    """(code, "Polish (pl)") for every language this whisper build knows."""
    from narrator.asr.whisper_languages import language_display_name, whisper_language_codes

    return [(code, language_display_name(code)) for code in whisper_language_codes()]


def vocabulary_dir(config_path: Path) -> Path:
    """Saved vocabulary lists live next to the config file (the same rule as
    AppController._text_settings)."""
    from narrator.asr.vocabulary import VOCABULARY_DIRNAME

    return config_path.parent / VOCABULARY_DIRNAME


def preset_options() -> list[dict]:
    """[{name, model, step_ms, agreement_passes, measured, note}] for the
    asr.preset dropdown ("custom" = no preset is not listed); [] when the
    presets module is unavailable."""
    try:
        from narrator.config.presets import load_presets
    except ImportError:  # pragma: no cover - presets not built
        return []
    rows = []
    for name, preset in load_presets().items():
        rows.append({"name": name, **preset.values(), "measured": preset.measured, "note": preset.note})
    return rows
