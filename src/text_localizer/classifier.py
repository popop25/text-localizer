from __future__ import annotations

import re
import unicodedata

from text_localizer.domain import RegionAction

_HANGUL_RE = re.compile(r"[\uac00-\ud7a3]")
_URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)\S+")
_VERSION_RE = re.compile(r"(?i)^v?\d+(?:\.\d+){1,4}(?:[-+][\w.-]+)?$")
_UNIT_RE = re.compile(
    r"(?i)^[-+]?\d+(?:\.\d+)?\s*(?:%|ms|s|min|h|px|kb|mb|gb|tb|hz|khz|mhz|ghz|"
    r"b|kbps|mbps|fps|°c|°f|m|cm|mm|km|g|kg|w|kw|v|a)$"
)
_ENGLISH_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.:/#@+\-]*$")
_ONLY_PUNCTUATION_RE = re.compile(r"^[\W_]+$", re.UNICODE)


def normalize_source_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    return " ".join(normalized.split()).strip()


def classify_text(text: str) -> RegionAction:
    return classify_text_with_reason(text)[0]


def classify_text_with_reason(text: str) -> tuple[RegionAction, str]:
    normalized = normalize_source_text(text)
    if not normalized or _ONLY_PUNCTUATION_RE.fullmatch(normalized):
        return RegionAction.IGNORE, "empty_or_punctuation"

    if _URL_RE.search(normalized):
        return RegionAction.PRESERVE, "url"
    if _VERSION_RE.fullmatch(normalized) or _UNIT_RE.fullmatch(normalized):
        return RegionAction.PRESERVE, "version_or_unit"
    if _ENGLISH_IDENTIFIER_RE.fullmatch(normalized):
        return RegionAction.PRESERVE, "english_identifier"

    code_markers = ("{", "}", "=>", "</", "/>", "::", "()", "[]", "import ", "def ", "class ")
    if any(marker in normalized for marker in code_markers):
        return RegionAction.PRESERVE, "source_code"

    if _HANGUL_RE.search(normalized):
        return RegionAction.TRANSLATE, "contains_hangul"
    return RegionAction.PRESERVE, "non_korean_text"
