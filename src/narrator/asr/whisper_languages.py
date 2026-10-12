"""Whisper's supported languages, resolved from the native binding.

pywhispercpp exposes `whisper_lang_str(id)` (and `whisper_lang_id(code)`) on
the loaded `_pywhispercpp` extension module -- callable without a model
context, unlike the detection APIs. Resolving at import time keeps the GUI's
dropdown and the backend's whitelist validation on the same authoritative
list (whatever this exact whisper.cpp build supports, all 99/100 of them)
instead of a hand-maintained copy that drifts from the binding.
"""
from __future__ import annotations

from functools import lru_cache

# (code, English display name) for the languages whisper ships with, in
# whisper's own id order. Kept in sync with whisper.cpp's lang auto-detect
# table; codes are the two-letter ISO set whisper uses for its --language
# prompt token. Names are for display only.
_LANGUAGE_NAMES: dict[str, str] = {
    "en": "English", "zh": "Chinese", "de": "German", "es": "Spanish",
    "ru": "Russian", "ko": "Korean", "fr": "French", "ja": "Japanese",
    "pt": "Portuguese", "tr": "Turkish", "pl": "Polish", "ca": "Catalan",
    "nl": "Dutch", "ar": "Arabic", "sv": "Swedish", "it": "Italian",
    "id": "Indonesian", "hi": "Hindi", "fi": "Finnish", "vi": "Vietnamese",
    "he": "Hebrew", "uk": "Ukrainian", "el": "Greek", "ms": "Malay",
    "cs": "Czech", "ro": "Romanian", "da": "Danish", "hu": "Hungarian",
    "ta": "Tamil", "no": "Norwegian", "th": "Thai", "ur": "Urdu",
    "hr": "Croatian", "bg": "Bulgarian", "lt": "Lithuanian", "la": "Latin",
    "mi": "Maori", "ml": "Malayalam", "cy": "Welsh", "sk": "Slovak",
    "te": "Telugu", "fa": "Persian", "lv": "Latvian", "bn": "Bengali",
    "sr": "Serbian", "az": "Azerbaijani", "sl": "Slovenian", "kn": "Kannada",
    "et": "Estonian", "mk": "Macedonian", "br": "Breton", "eu": "Basque",
    "is": "Icelandic", "hy": "Armenian", "ne": "Nepali", "mn": "Mongolian",
    "bs": "Bosnian", "kk": "Kazakh", "sq": "Albanian", "sw": "Swahili",
    "gl": "Galician", "mr": "Marathi", "pa": "Punjabi", "si": "Sinhala",
    "km": "Khmer", "sn": "Shona", "yo": "Yoruba", "so": "Somali",
    "af": "Afrikaans", "oc": "Occitan", "ka": "Georgian", "be": "Belarusian",
    "tg": "Tajik", "sd": "Sindhi", "gu": "Gujarati", "am": "Amharic",
    "yi": "Yiddish", "lo": "Lao", "uz": "Uzbek", "fo": "Faroese",
    "ht": "Haitian Creole", "ps": "Pashto", "tk": "Turkmen", "nn": "Nynorsk",
    "mt": "Maltese", "sa": "Sanskrit", "lb": "Luxembourgish", "my": "Burmese",
    "bo": "Tibetan", "tl": "Tagalog", "mg": "Malagasy", "as": "Assamese",
    "tt": "Tatar", "haw": "Hawaiian", "ln": "Lingala", "ha": "Hausa",
    "ba": "Bashkir", "jw": "Javanese", "su": "Sundanese", "yue": "Cantonese",
}


@lru_cache(maxsize=1)
def whisper_language_codes() -> tuple[str, ...]:
    """All language codes this whisper.cpp build supports, in id order.

    Resolved from the native binding (`whisper_lang_str`); falls back to the
    curated table above if the extension module is unavailable (e.g. exotic
    packaging where importing it has side effects).
    """
    try:
        from pywhispercpp.model import pw  # the pybind11 extension module

        codes: list[str] = []
        # Bounded by the curated table's size: probing past the last valid id
        # makes whisper.cpp print "unknown language id" to stderr on every
        # import. If a future build ships more languages, extend the table.
        for lang_id in range(len(_LANGUAGE_NAMES)):
            code = pw.whisper_lang_str(lang_id)
            if code is None:
                break
            codes.append(str(code))
        if codes:
            return tuple(codes)
    except Exception:  # pragma: no cover - only on broken installs
        pass
    return tuple(_LANGUAGE_NAMES)


def language_display_name(code: str) -> str:
    """'pl' -> 'Polish (pl)' style label for GUI dropdowns; falls back to the
    code itself for anything unknown."""
    name = _LANGUAGE_NAMES.get(code)
    return f"{name} ({code})" if name else code


def language_code_for_name(name: str) -> str | None:
    """'Polish' -> 'pl' (case-insensitive), or None for a name whisper does
    not know. Used to read the language tag off downloaded subtitle files
    ("... (Polish_ASR).srt")."""
    wanted = name.strip().lower()
    for code, display in _LANGUAGE_NAMES.items():
        if display.lower() == wanted:
            return code
    return None
