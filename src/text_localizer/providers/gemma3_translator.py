from __future__ import annotations

import importlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any


class Gemma3TranslationProvider:
    name = "sappho192/gemma3-multilingual-translator-270m"

    def __init__(self, model_dir: Path) -> None:
        if not (model_dir / "config.json").is_file():
            raise RuntimeError(
                "Gemma3 translation model files are not prepared. Run "
                "'text-localizer setup-models --translation "
                "--translation-provider gemma3-270m' explicitly first."
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

        self._tokenizer = transformers.AutoTokenizer.from_pretrained(
            str(self._model_dir),
            local_files_only=True,
        )
        self._tokenizer.padding_side = "left"
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
        self._model = transformers.AutoModelForCausalLM.from_pretrained(
            str(self._model_dir),
            local_files_only=True,
            torch_dtype=torch.float32,
        )
        self._model.to("cpu")
        self._model.eval()
        self._torch = torch

    def translate(self, source_text: str) -> str:
        return self.translate_many((source_text,))[0]

    def translate_many(self, source_texts: Sequence[str]) -> tuple[str, ...]:
        if not source_texts:
            return ()
        self._ensure_loaded()
        prompts = [f"<src:ko><tgt:ja>\n{source}\n###\n" for source in source_texts]
        encoded = self._tokenizer(prompts, return_tensors="pt", padding=True)
        input_width = int(encoded["input_ids"].shape[1])
        with self._torch.inference_mode():
            generated = self._model.generate(
                **encoded,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=self._tokenizer.pad_token_id,
            )
        continuations = generated[:, input_width:]
        decoded = self._tokenizer.batch_decode(continuations, skip_special_tokens=True)
        return tuple(_clean_translation(str(item)) for item in decoded)


def _clean_translation(value: str) -> str:
    cleaned = value.split("###", maxsplit=1)[0].strip()
    for prefix in ("Japanese:", "日本語:", "日本語："):
        if cleaned.startswith(prefix):
            cleaned = cleaned.removeprefix(prefix).strip()
    return cleaned
