import importlib
from pathlib import Path

import pytest

from text_localizer.providers import m2m100, paddleocr


def _unexpected_import(module_name: str) -> None:
    raise AssertionError(f"unexpected eager import: {module_name}")


def test_m2m100_defers_runtime_import_until_nonempty_translation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_dir = tmp_path / "m2m100"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(importlib, "import_module", _unexpected_import)

    provider = m2m100.M2M100TranslationProvider(model_dir)

    assert provider.translate_many(()) == ()


def test_paddleocr_defers_runtime_import_until_recognition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = tmp_path / "paddleocr" / ".ready.json"
    marker.parent.mkdir()
    marker.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(importlib, "import_module", _unexpected_import)

    provider = paddleocr.PaddleOcrProvider(tmp_path)

    assert provider.name == "paddleocr-ppocrv5-mobile-korean"
