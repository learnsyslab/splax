"""Test the asset download cache against a local server."""

from __future__ import annotations

import http.server
import threading
from typing import TYPE_CHECKING

import pytest

from splax.io import fetch

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


@pytest.fixture
def server() -> Iterator[tuple[str, dict]]:
    """Serve one in-memory body over HTTP on an ephemeral local port.

    Yields:
        The file url and the mutable state driving the responses. ``body`` is the payload, or
        ``None`` once the asset should be gone, ``etag`` is sent only when set, and ``gets``
        counts the downloads the test triggered.
    """
    state = {"body": b"", "etag": None, "gets": 0}

    class Handler(http.server.BaseHTTPRequestHandler):
        def _respond(self) -> bool:
            if state["body"] is None:
                self.send_error(404)
                return False
            self.send_response(200)
            self.send_header("Content-Length", str(len(state["body"])))
            if state["etag"] is not None:
                self.send_header("ETag", state["etag"])
            self.end_headers()
            return True

        def do_HEAD(self):  # noqa: N802 (BaseHTTPRequestHandler API)
            self._respond()

        def do_GET(self):  # noqa: N802 (BaseHTTPRequestHandler API)
            state["gets"] += 1
            if self._respond():
                self.wfile.write(state["body"])

        def log_message(self, format: str, *args: object):
            pass  # Keep pytest output clean.

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/scene.ply", state
    httpd.shutdown()
    httpd.server_close()


def test_fetch_downloads(server: tuple[str, dict], tmp_path: Path):
    """Download a remote file into the cache directory on the first fetch."""
    url, state = server
    state["body"] = b"splat bytes"
    cache = tmp_path / "cache"

    path = fetch(url, cache=cache)

    assert path.parent == cache
    assert path.name.endswith("-scene.ply")
    assert path.read_bytes() == b"splat bytes"


def test_fetch_cache_hit(server: tuple[str, dict], tmp_path: Path):
    """Return the cached path on a second fetch without downloading again."""
    url, state = server
    state["body"], state["etag"] = b"splat bytes", '"fixed"'
    cache = tmp_path / "cache"

    first = fetch(url, cache=cache)
    assert state["gets"] == 1
    second = fetch(url, cache=cache)

    assert second == first
    assert state["gets"] == 1  # The unchanged tag is enough, no new download.


def test_fetch_unchecked_serves_cache(server: tuple[str, dict], tmp_path: Path):
    """Serve the cache without contacting the remote, even once the remote is gone."""
    url, state = server
    state["body"], state["etag"] = b"splat bytes", '"fixed"'
    cache = tmp_path / "cache"

    fetch(url, cache=cache)
    state["body"] = None  # Any request from here on fails, so a served cache proves none was made.

    assert fetch(url, cache=cache, allow_unchecked=True).read_bytes() == b"splat bytes"


def test_fetch_force_redownloads(server: tuple[str, dict], tmp_path: Path):
    """Re-download and overwrite the cached copy when the caller forces it."""
    url, state = server
    state["body"], state["etag"] = b"old bytes", '"fixed"'
    cache = tmp_path / "cache"

    path = fetch(url, cache=cache)
    state["body"] = b"new bytes"  # The tag stays put, so an ordinary fetch keeps serving the cache.
    assert fetch(url, cache=cache).read_bytes() == b"old bytes"

    forced = fetch(url, cache=cache, force=True)

    assert forced == path
    assert forced.read_bytes() == b"new bytes"


def test_fetch_etag_invalidates_cache(server: tuple[str, dict], tmp_path: Path):
    """Reuse the cache while the remote tag is unchanged and refetch once it changes."""
    url, state = server
    state["body"], state["etag"] = b"v1 bytes", '"aaa"'
    cache = tmp_path / "cache"

    assert fetch(url, cache=cache).read_bytes() == b"v1 bytes"
    assert state["gets"] == 1
    fetch(url, cache=cache)  # Same tag: cache hit, no new download.
    assert state["gets"] == 1

    state["body"], state["etag"] = b"v2 bytes longer", '"bbb"'
    assert fetch(url, cache=cache).read_bytes() == b"v2 bytes longer"  # Changed tag: refetch.
    assert state["gets"] == 2


def test_fetch_without_etag(server: tuple[str, dict], tmp_path: Path):
    """Download from a remote that sends no tag, refetching on every call."""
    url, state = server
    state["body"] = b"no etag bytes"
    cache = tmp_path / "cache"

    assert fetch(url, cache=cache).read_bytes() == b"no etag bytes"
    assert state["gets"] == 1
    fetch(url, cache=cache)  # No tag to compare: download again rather than serve stale.
    assert state["gets"] == 2


def test_fetch_env_cache(server: tuple[str, dict], tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Fall back to the cache directory the environment names when none is given."""
    url, state = server
    state["body"] = b"splat bytes"
    env_cache = tmp_path / "env_cache"
    monkeypatch.setenv("SPLAX_CACHE", str(env_cache))

    path = fetch(url)

    assert path.parent == env_cache
    assert path.read_bytes() == b"splat bytes"
