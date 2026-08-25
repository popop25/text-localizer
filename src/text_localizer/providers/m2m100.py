from __future__ import annotations

import importlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any


class M2M100TranslationProvider:
    name = "facebook/m2m100_418M"

    def __init__(self, model_dir: Path) -> None:
        if not (model_dir / "config.json").is_file():
            raise RuntimeError(
                "M2M100 model files are not prepared. Run "
                "'text-localizer setup-models --translation' explicitly first."
            )
        self._model_dir = model_dir
        self._tokenizer: Any = None
        self._model: Any = None
        self._torch: Any = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            transformers: Any = importlib.import_module("transformers")
            torch: Any = importlib.import_module("torch")
        except ImportError as exc:
            raise RuntimeError(
                "Translation dependencies are unavailable. Install the 'translation' extra."
            ) from exc

        tokenizer_class = transformers.M2M100Tokenizer
        model_class = transformers.M2M100ForConditionalGeneration
        self._tokenizer = tokenizer_class.from_pretrained(
            str(self._model_dir),
            local_files_only=True,
        )
        self._model = model_class.from_pretrained(
            str(self._model_dir),
            local_files_only=True,
        )
        self._tokenizer.src_lang = "ko"
        self._model.to("cpu")
        self._model.eval()
        self._torch = torch

    def translate(self, source_text: str) -> str:
        return self.translate_many((source_text,))[0]

    def translate_many(self, source_texts: Sequence[str]) -> tuple[str, ...]:
        if not source_texts:
            return ()
        self._ensure_loaded()
        encoded = self._tokenizer(list(source_texts), return_tensors="pt", padding=True)
        forced_bos_token_id = self._tokenizer.get_lang_id("ja")
        with self._torch.inference_mode():
            generated = self._model.generate(
                **encoded,
                forced_bos_token_id=forced_bos_token_id,
                max_new_tokens=256,
            )
        decoded = self._tokenizer.batch_decode(generated, skip_special_tokens=True)
        return tuple(str(item).strip() for item in decoded)
