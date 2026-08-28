from __future__ import annotations

import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import cast

import pytest

from text_localizer.viewer_server import create_viewer_server


def test_viewer_server_supports_full_and_range_media_requests(tmp_path: Path) -> None:
    (tmp_path / "viewer.html").write_text("<video></video>", encoding="utf-8")
    media = bytes(range(64))
    (tmp_path / "source.mp4").write_bytes(media)
    server = create_viewer_server(tmp_path, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host = cast(str, server.server_address[0])
    port = server.server_address[1]
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/viewer.html") as response:
            assert response.status == 200
            assert response.headers["Accept-Ranges"] == "bytes"
            assert response.read() == b"<video></video>"

        request = urllib.request.Request(
            f"http://{host}:{port}/source.mp4",
            headers={"Range": "bytes=10-19"},
        )
        with urllib.request.urlopen(request) as response:
            assert response.status == 206
            assert response.headers["Content-Range"] == "bytes 10-19/64"
            assert response.headers["Content-Length"] == "10"
            assert response.read() == media[10:20]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_viewer_server_rejects_non_loopback_binding(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="loopback"):
        create_viewer_server(tmp_path, host="0.0.0.0")


def test_viewer_server_returns_416_for_invalid_range(tmp_path: Path) -> None:
    (tmp_path / "source.mp4").write_bytes(b"1234")
    server = create_viewer_server(tmp_path, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host = cast(str, server.server_address[0])
    port = server.server_address[1]
    try:
        request = urllib.request.Request(
            f"http://{host}:{port}/source.mp4",
            headers={"Range": "bytes=10-20"},
        )
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request)
        assert error.value.code == 416
        assert error.value.headers["Content-Range"] == "bytes */4"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
