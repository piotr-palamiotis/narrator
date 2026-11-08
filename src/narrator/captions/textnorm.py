"""Text normalization shared by caption word matching (the blocklist and the
vocabulary snapper): case- and diacritics-blind comparison keys."""
from __future__ import annotations

import unicodedata

# Letters NFKD does not decompose into base + combining mark (Polish ł is the
# one that matters here; the rest keep the fold symmetric for loanwords).
_UNDECOMPOSED = str.maketrans({"ł": "l", "Ł": "l", "đ": "d", "Đ": "d", "ø": "o", "Ø": "o", "ß": "ss"})


def fold(text: str) -> str:
    """Lowercase, diacritics stripped: "Kurwą" -> "kurwa", "Wrocław" ->
    "wroclaw". Used only as a matching key, never shown."""
    decomposed = unicodedata.normalize("NFKD", text.translate(_UNDECOMPOSED))
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).lower()
