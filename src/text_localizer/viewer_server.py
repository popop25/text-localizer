from __future__ import annotations

import argparse
import ipaddress
import shutil
from collections.abc import Sequence
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, BinaryIO, cast


class ViewerRequestHandler(SimpleHTTPRequestHandler):
    server_version = "text-localizer-viewer/1"
    _byte_range: tuple[int, int] | None = None

    def send_head(self) -> BinaryIO | None:
        path = Path(self.translate_path(self.path))
        if path.is_dir():
            return super().send_head()
        try:
            source = path.open("rb")
        except OSError:
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return None
        try:
            stat = path.stat()
            file_size = stat.st_size
            try:
                selected_range = _parse_range_header(self.headers.get("Range"), file_size)
            except ValueError:
                source.close()
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{file_size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return None
            self._byte_range = selected_range
            status = HTTPStatus.PARTIAL_CONTENT if selected_range is not None else HTTPStatus.OK
            self.send_response(status)
            self.send_header("Content-Type", self.guess_type(str(path)))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Cache-Control", "no-store")
            if selected_range is None:
                content_length = file_size
            else:
                start, end = selected_range
                content_length = end - start + 1
                self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
            self.send_header("Content-Length", str(content_length))
            self.send_header("Last-Modified", self.date_time_string(stat.st_mtime))
            self.end_headers()
            return source
        except Exception:
            source.close()
            raise

    def copyfile(self, source: Any, outputfile: Any) -> None:
        if self._byte_range is None:
            shutil.copyfileobj(source, outputfile)
            return
        start, end = self._byte_range
        source.seek(start)
        remaining = end - start + 1
        while remaining:
            chunk = source.read(min(64 * 1024, remaining))
            if not chunk:
                break
            outputfile.write(chunk)
            remaining -= len(chunk)


def create_viewer_server(
    directory: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> ThreadingHTTPServer:
    root = directory.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Viewer directory does not exist: {directory}")
    try:
        address = ipaddress.ip_address(host)
    except ValueError as exc:
        raise ValueError("Viewer host must be a loopback IP address.") from exc
    if not address.is_loopback:
        raise ValueError("Viewer server may bind only to a loopback address.")
    if isinstance(port, bool) or not 0 <= port <= 65535:
        raise ValueError("Viewer port must be between 0 and 65535.")
    handler = partial(ViewerRequestHandler, directory=str(root))
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    return server


def serve_viewer(directory: Path, *, host: str = "127.0.0.1", port: int = 8765) -> None:
    server = create_viewer_server(directory, host=host, port=port)
    actual_host = cast(str, server.server_address[0])
    actual_port = server.server_address[1]
    print(f"Serving text-localizer viewer at http://{actual_host}:{actual_port}/viewer.html")
    try:
        server.serve_forever()
    finally:
        server.server_close()


def _parse_range_header(raw: str | None, file_size: int) -> tuple[int, int] | None:
    if raw is None:
        return None
    if not raw.startswith("bytes=") or "," in raw:
        raise ValueError("Unsupported Range header.")
    value = raw.removeprefix("bytes=").strip()
    if "-" not in value or file_size <= 0:
        raise ValueError("Invalid Range header.")
    raw_start, raw_end = value.split("-", 1)
    try:
        if not raw_start:
            suffix_length = int(raw_end)
            if suffix_length <= 0:
                raise ValueError("Invalid suffix range.")
            start = max(0, file_size - suffix_length)
            return start, file_size - 1
        start = int(raw_start)
        end = file_size - 1 if not raw_end else int(raw_end)
    except ValueError:
        raise ValueError("Invalid Range header.") from None
    if start < 0 or start >= file_size or end < start:
        raise ValueError("Range is outside the file.")
    return start, min(end, file_size - 1)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="text-localizer-viewer")
    parser.add_argument("directory", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    try:
        serve_viewer(args.directory, host=args.host, port=args.port)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
