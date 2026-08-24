from __future__ import annotations

import importlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def setup_ocr_models(models_root: Path) -> Path:
    target = models_root / "paddleocr"
    target.mkdir(parents=True, exist_ok=True)
    os.environ["PADDLE_OCR_BASE_DIR"] = str(target)
    try:
        module = importlib.import_module("paddleocr")
        paddleocr_class: Any = module.PaddleOCR
    except (ImportError, AttributeError) as exc:
        raise RuntimeError("Install OCR dependencies first with: pip install -e '.[ocr]'") from exc

    paddleocr_class(
        text_detection_model_name="PP-OCRv5_mobile_det",
        text_recognition_model_name="korean_PP-OCRv5_mobile_rec",
        lang="korean",
        ocr_version="PP-OCRv5",
        device="cpu",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )
    marker = target / ".ready.json"
    marker.write_text(
        json.dumps(
            {
                "prepared_at": datetime.now(UTC).isoformat(),
                "detection_model": "PP-OCRv5_mobile_det",
                "recognition_model": "korean_PP-OCRv5_mobile_rec",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return marker


def setup_translation_model(models_root: Path) -> Path:
    target = models_root / "m2m100_418M"
    target.mkdir(parents=True, exist_ok=True)
    try:
        hub: Any = importlib.import_module("huggingface_hub")
    except ImportError as exc:
        raise RuntimeError(
            "Install translation dependencies first with: pip install -e '.[translation]'"
        ) from exc

    hub.snapshot_download(
        repo_id="facebook/m2m100_418M",
        local_dir=str(target),
    )
    return target
