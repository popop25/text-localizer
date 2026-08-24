# Repository instructions

- Run and validate this project with Windows-native tools only. Do not use WSL or Docker.
- Use Python 3.12 and CPU-only inference.
- Never commit videos, extracted frames, generated outputs, model weights, caches, or secrets.
- Do not download model weights during normal commands. Downloads require an explicit setup command.
- Keep OCR and translation behind interfaces and keep all coordinates normalized to `[0, 1]`.
- Preserve URLs, source code, version numbers, values with units, product names, and English-only identifiers.
- Keep the original input MP4 unchanged.
- Run Ruff, mypy, and pytest before committing.
