from pathlib import Path

from text_localizer.cache import TranslationCache


def test_translation_cache_normalizes_keys(tmp_path: Path) -> None:
    cache = TranslationCache(tmp_path / "translations.sqlite3")
    cache.put("fake", "모델   학습", "モデル学習")

    assert cache.get("fake", " 모델 학습 ") == "モデル学習"
