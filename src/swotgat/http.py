"""Polite HTTP access for every script that downloads from a public service.

All download code in this repository goes through this module, which

* sends a generic client name (set the environment variable
  ``SWOTGAT_CLIENT_NAME`` if a data provider asks you to identify yourself);
* allows one request at a time per host, and waits at least 1.1 seconds
  between two requests to the same host (5 seconds for HydroShare, its
  published crawl delay), however many threads the caller uses.

The harvest scripts additionally cache every response on disk, so a re-run
does not fetch anything twice.
"""
from __future__ import annotations

import os
import threading
import time
from urllib.parse import urlparse

import requests

CLIENT_NAME = os.environ.get("SWOTGAT_CLIENT_NAME", "swot-gat-research/1.0")
MIN_INTERVAL_S = 1.1
HOST_INTERVAL_S = {"www.hydroshare.org": 5.0}

_guard = threading.Lock()
_host_locks: dict[str, threading.Lock] = {}
_last_request: dict[str, float] = {}


def wait_turn(url: str) -> threading.Lock:
    """Block until a request to ``url``'s host is allowed; return that host's lock, held."""
    host = urlparse(url).netloc
    with _guard:
        lock = _host_locks.setdefault(host, threading.Lock())
    lock.acquire()
    pause = HOST_INTERVAL_S.get(host, MIN_INTERVAL_S) - (time.time() - _last_request.get(host, 0.0))
    if pause > 0:
        time.sleep(pause)
    return lock


class PoliteSession(requests.Session):
    """A ``requests.Session`` with the generic client name and the per-host rate limit."""

    def __init__(self) -> None:
        """Create the session and set the generic client name."""
        super().__init__()
        self.headers["User-Agent"] = CLIENT_NAME

    def request(self, method, url, *args, **kwargs):
        """Send one request after waiting for this host's turn (same contract as ``requests``)."""
        lock = wait_turn(url)
        try:
            return super().request(method, url, *args, **kwargs)
        finally:
            _last_request[urlparse(url).netloc] = time.time()
            lock.release()


_session: PoliteSession | None = None


def polite_get(url: str, **kwargs) -> requests.Response:
    """Rate-limited replacement for ``requests.get``."""
    global _session
    if _session is None:
        _session = PoliteSession()
    return _session.get(url, **kwargs)
