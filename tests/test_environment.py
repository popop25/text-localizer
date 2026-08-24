from pathlib import Path

import pytest

from text_localizer.environment import resolve_executable


def test_resolve_executable_prefers_environment_variable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = tmp_path / "tool.exe"
    executable.write_bytes(b"test")
    monkeypatch.setenv("TEST_TOOL_BIN", str(executable))

    assert (
        resolve_executable("missing-tool", environment_variable="TEST_TOOL_BIN")
        == executable.resolve()
    )
