from __future__ import annotations

import sqlite3
from pathlib import Path

from text_localizer.classifier import normalize_source_text


class TranslationCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS translations (
                    provider TEXT NOT NULL,
                    source_language TEXT NOT NULL,
                    target_language TEXT NOT NULL,
                    source_normalized TEXT NOT NULL,
                    translated_text TEXT NOT NULL,
                    PRIMARY KEY (
                        provider,
                        source_language,
                        target_language,
                        source_normalized
                    )
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def get(
        self,
        provider: str,
        source_text: str,
        source_language: str = "ko",
        target_language: str = "ja",
    ) -> str | None:
        normalized = normalize_source_text(source_text)
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT translated_text
                FROM translations
                WHERE provider = ?
                  AND source_language = ?
                  AND target_language = ?
                  AND source_normalized = ?
                """,
                (provider, source_language, target_language, normalized),
            ).fetchone()
        return str(row[0]) if row else None

    def put(
        self,
        provider: str,
        source_text: str,
        translated_text: str,
        source_language: str = "ko",
        target_language: str = "ja",
    ) -> None:
        normalized = normalize_source_text(source_text)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO translations (
                    provider,
                    source_language,
                    target_language,
                    source_normalized,
                    translated_text
                )
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT (
                    provider,
                    source_language,
                    target_language,
                    source_normalized
                )
                DO UPDATE SET translated_text = excluded.translated_text
                """,
                (
                    provider,
                    source_language,
                    target_language,
                    normalized,
                    translated_text,
                ),
            )
