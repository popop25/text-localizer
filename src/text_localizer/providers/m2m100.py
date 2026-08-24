from __future__ import annotations

import importlib
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
        try:
            transformers: Any = importlib.import_module("transformers")
            torch: Any = importlib.import_module("torch")
        except ImportError as exc:
            raise RuntimeError(
                "Translation dependencies are unavailable. Install the 'translation' extra."
            ) from exc

        tokenizer_class = transformers.M2M100Tokenizer
        model_class = transformers.M2M100ForConditionalGeneration
        self._tokenizer: Any = tokenizer_class.from_pretrained(
            str(model_dir),
            local_files_only=True,
        )
        self._model: Any = model_class.from_pretrained(
            str(model_dir),
            local_files_only=True,
        )
        self._tokenizer.src_lang = "ko"
        self._model.to("cpu")
        self._model.eval()
        self._torch = torch

    def translate(self, source_text: str) -> str:
        encoded = self._tokenizer(source_text, return_tensors="pt")
        forced_bos_token_id = self._tokenizer.get_lang_id("ja")
        with self._torch.inference_mode():
            generated = self._model.generate(
                **encoded,
                forced_bos_token_id=forced_bos_token_id,
                max_new_tokens=256,
            )
        return str(self._tokenizer.batch_decode(generated, skip_special_tokens=True)[0]).strip()
