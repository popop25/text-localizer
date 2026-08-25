import importlib
import os
from pathlib import Path
from typing import Any

import pytest

from text_localizer.setup_models import setup_ocr_models, setup_translation_model


class FakeTruststore:
    injected = False

    @classmethod
    def inject_into_ssl(cls) -> None:
        cls.injected = True


class FakePaddleOcr:
    arguments: dict[str, Any] = {}

    def __init__(self, **kwargs: Any) -> None:
        type(self).arguments = kwargs


class FakeHub:
    arguments: dict[str, Any] = {}

    @classmethod
    def snapshot_download(cls, **kwargs: Any) -> None:
        cls.arguments = kwargs


def test_ocr_setup_uses_windows_store_and_dedicated_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeTruststore.injected = False

    def fake_import(name: str) -> Any:
        if name == "paddleocr":
            return type("PaddleModule", (), {"PaddleOCR": FakePaddleOcr})
        if name == "truststore":
            return FakeTruststore
        return importlib.import_module(name)

    monkeypatch.setattr("text_localizer.setup_models.importlib.import_module", fake_import)
    marker = setup_ocr_models(tmp_path)

    assert marker.is_file()
    assert os.environ["PADDLE_PDX_CACHE_HOME"] == str(tmp_path / "paddleocr")
    assert FakeTruststore.injected is True
    assert FakePaddleOcr.arguments["enable_mkldnn"] is False


def test_translation_setup_uses_truststore_and_required_files_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeTruststore.injected = False

    def fake_import(name: str) -> Any:
        if name == "huggingface_hub":
            return FakeHub
        if name == "truststore":
            return FakeTruststore
        return importlib.import_module(name)

    monkeypatch.setattr("text_localizer.setup_models.importlib.import_module", fake_import)
    target = setup_translation_model(tmp_path)

    assert target == tmp_path / "m2m100_418M"
