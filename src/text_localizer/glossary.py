from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from text_localizer.classifier import normalize_source_text

_HANGUL_LATIN_BOUNDARY_RE = re.compile(
    r"(?<=[\uac00-\ud7a3])(?=[A-Za-z])|(?<=[A-Za-z])(?=[\uac00-\ud7a3])"
)
_MISSING_SPACE_AFTER_PUNCTUATION_RE = re.compile(r"(?<=[,;:])(?=\S)")
_MISSING_SPACE_AFTER_BRACKET_RE = re.compile(r"(?<=[\]\)])(?=[A-Za-z\uac00-\ud7a3])")
_LEADING_NUMBERED_TEXT_RE = re.compile(r"^(\d+)(?=[A-Za-z\uac00-\ud7a3])")
_PLACEHOLDER_RE = re.compile(r"<TERM(\d+)>")


@dataclass(frozen=True, slots=True)
class GlossaryEntry:
    source: str
    target: str
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ProtectedTerm:
    placeholder: str
    source: str
    target: str


@dataclass(frozen=True, slots=True)
class PreparedText:
    raw_text: str
    normalized_text: str
    protected_text: str
    normalization_edits: tuple[str, ...]
    protected_terms: tuple[ProtectedTerm, ...]

    @property
    def glossary_only(self) -> bool:
        without_terms = _PLACEHOLDER_RE.sub("", self.protected_text)
        return not without_terms.strip()


class Glossary:
    def __init__(self, entries: tuple[GlossaryEntry, ...] = ()) -> None:
        self.entries = entries
        fingerprint_payload = [
            {"source": entry.source, "target": entry.target, "aliases": entry.aliases}
            for entry in entries
        ]
        encoded = json.dumps(
            fingerprint_payload,
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
        self.fingerprint = hashlib.sha256(encoded).hexdigest()[:12] if entries else "none"

    @classmethod
    def empty(cls) -> Glossary:
        return cls()

    @classmethod
    def from_yaml(cls, path: Path) -> Glossary:
        if not path.is_file():
            raise FileNotFoundError(f"Glossary file does not exist: {path}")
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        raw_entries = payload.get("entries", [])
        if not isinstance(raw_entries, list):
            raise ValueError("Glossary 'entries' must be a list.")

        entries: list[GlossaryEntry] = []
        for index, item in enumerate(raw_entries):
            if not isinstance(item, dict):
                raise ValueError(f"Glossary entry {index} must be a mapping.")
            source = _required_text(item, "source", index)
            target = _required_text(item, "target", index)
            raw_aliases: Any = item.get("aliases", [])
            if not isinstance(raw_aliases, list) or not all(
                isinstance(alias, str) and alias.strip() for alias in raw_aliases
            ):
                raise ValueError(f"Glossary entry {index} aliases must be non-empty strings.")
            entries.append(
                GlossaryEntry(
                    source=normalize_source_text(source),
                    target=normalize_source_text(target),
                    aliases=tuple(normalize_source_text(alias) for alias in raw_aliases),
                )
            )
        return cls(tuple(entries))

    def cache_namespace(self, provider_name: str) -> str:
        if not self.entries:
            return provider_name
        return f"{provider_name}:glossary:{self.fingerprint}"

    def prepare(self, raw_text: str) -> PreparedText:
        normalized = normalize_source_text(raw_text)
        edits: list[str] = []
        alias_replacements: list[tuple[str, str]] = []

        for entry in sorted(self.entries, key=_longest_alias, reverse=True):
            for alias in sorted(entry.aliases, key=len, reverse=True):
                placeholder = f"\ue000{len(alias_replacements)}\ue001"
                replaced, count = _replace_term(normalized, alias, placeholder)
                if count:
                    normalized = replaced
                    alias_replacements.append((placeholder, entry.source))
                    edits.append(f"{alias} -> {entry.source}")
                    break

        for placeholder, source in alias_replacements:
            normalized = normalized.replace(placeholder, source)

        spaced = _HANGUL_LATIN_BOUNDARY_RE.sub(" ", normalized)
        spaced = _MISSING_SPACE_AFTER_PUNCTUATION_RE.sub(" ", spaced)
        spaced = _MISSING_SPACE_AFTER_BRACKET_RE.sub(" ", spaced)
        spaced = _LEADING_NUMBERED_TEXT_RE.sub(r"\1 ", spaced)
        spaced = normalize_source_text(spaced)
        if spaced != normalized:
            edits.append("restore-script-boundary-spacing")
        normalized = spaced

        protected = normalized
        terms: list[ProtectedTerm] = []
        for entry in sorted(self.entries, key=lambda item: len(item.source), reverse=True):
            if not entry.source:
                continue
            while True:
                placeholder = f"<TERM{len(terms)}>"
                replaced, count = _replace_term(protected, entry.source, placeholder, count=1)
                if not count:
                    break
                protected = replaced
                terms.append(
                    ProtectedTerm(
                        placeholder=placeholder,
                        source=entry.source,
                        target=entry.target,
                    )
                )

        return PreparedText(
            raw_text=raw_text,
            normalized_text=normalized,
            protected_text=protected,
            normalization_edits=tuple(edits),
            protected_terms=tuple(terms),
        )

    @staticmethod
    def restore(translated_text: str, prepared: PreparedText) -> tuple[str, tuple[str, ...]]:
        restored = translated_text
        flags: list[str] = []
        for term in prepared.protected_terms:
            placeholder_number = term.placeholder.removeprefix("<TERM").removesuffix(">")
            placeholder_pattern = re.compile(
                rf"(?:<|＜|〈|《|「|『|\[\[|\[|__)?\s*TERM{placeholder_number}\s*"
                rf"(?:>|＞|〉|》|」|』|\]\]|\]|__)?",
                re.IGNORECASE,
            )
            restored, count = placeholder_pattern.subn(
                _literal_replacement(term.target),
                restored,
                count=1,
            )
            if not count and term.target not in restored:
                flags.append(f"missing_placeholder:{term.placeholder}")
        return normalize_source_text(restored), tuple(flags)


def _required_text(item: dict[str, Any], key: str, index: int) -> str:
    value = item.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Glossary entry {index} requires a non-empty '{key}'.")
    return value


def _longest_alias(entry: GlossaryEntry) -> int:
    return max((len(alias) for alias in entry.aliases), default=0)


def _replace_term(
    text: str,
    source: str,
    replacement: str,
    *,
    count: int = 0,
) -> tuple[str, int]:
    if not source:
        return text, 0
    flags = re.IGNORECASE if source.isascii() else 0
    pattern = re.compile(re.escape(source), flags)
    return pattern.subn(lambda _: replacement, text, count=count)


def _literal_replacement(value: str) -> Callable[[re.Match[str]], str]:
    def replace(_: re.Match[str]) -> str:
        return value

    return replace
