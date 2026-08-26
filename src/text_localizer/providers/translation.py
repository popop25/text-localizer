from __future__ import annotations

from pathlib import Path

from text_localizer.providers.base import TranslationProvider
from text_localizer.providers.gemma3_translator import Gemma3TranslationProvider
from text_localizer.providers.m2m100 import M2M100TranslationProvider

M2M100_PROVIDER = "m2m100"
GEMMA3_270M_PROVIDER = "gemma3-270m"
TRANSLATION_PROVIDER_CHOICES = (M2M100_PROVIDER, GEMMA3_270M_PROVIDER)


def create_translation_provider(provider: str, models_root: Path) -> TranslationProvider:
    if provider == M2M100_PROVIDER:
        return M2M100TranslationProvider(models_root / "m2m100_418M")
    if provider == GEMMA3_270M_PROVIDER:
        return Gemma3TranslationProvider(models_root / "gemma3-multilingual-translator-270m")
    raise ValueError(f"Unsupported translation provider: {provider}")
