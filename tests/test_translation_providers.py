from pathlib import Path

import pytest

from text_localizer.providers.gemma3_translator import (
    Gemma3TranslationProvider,
    _clean_translation,
)
from text_localizer.providers.translation import (
    GEMMA3_270M_PROVIDER,
    M2M100_PROVIDER,
    create_translation_provider,
)


def test_translation_provider_factory_selects_local_model_directories(tmp_path: Path) -> None:
    m2m = tmp_path / "m2m100_418M"
    gemma = tmp_path / "gemma3-multilingual-translator-270m"
    m2m.mkdir()
    gemma.mkdir()
    (m2m / "config.json").write_text("{}", encoding="utf-8")
    (gemma / "config.json").write_text("{}", encoding="utf-8")

    assert create_translation_provider(M2M100_PROVIDER, tmp_path).name == "facebook/m2m100_418M"
    assert (
        create_translation_provider(GEMMA3_270M_PROVIDER, tmp_path).name
        == Gemma3TranslationProvider.name
    )


def test_translation_provider_factory_rejects_unknown_provider(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported translation provider"):
        create_translation_provider("unknown", tmp_path)


def test_gemma_translation_cleanup_removes_delimiters_and_labels() -> None:
    assert _clean_translation("日本語： 講義紹介\n###\nnoise") == "講義紹介"
