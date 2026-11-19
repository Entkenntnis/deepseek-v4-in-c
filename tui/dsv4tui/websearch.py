# SPDX-License-Identifier: Apache-2.0
"""The web_search tool, backed by DeepSeek's Search API.

The API is a plain JSON endpoint, so the TUI only needs the standard library.
The key is read from DEEPSEEK_API_KEY (real environment first) or from the
repository's gitignored `.env` file. Search results are normalised into the
same small result dict shape the Python tool uses, which keeps the agent loop
and the transcript renderer tool-agnostic.
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request

from .defaults import DEFAULT_ENV, SEARCH_MAX_RESULTS, SEARCH_TIMEOUT_S

SEARCH_URL = "https://api.deepseek.com/v1/tools/search"


def _env_file(path) -> dict[str, str]:
    """Parse a tiny KEY=VALUE file; missing or unreadable means empty."""
    values: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return values
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key.strip()] = value
    return values


def _usable(key: str) -> bool:
    """Reject the obvious placeholders shipped in .env.example."""
    key = key.strip()
    if not key:
        return False
    low = key.lower()
    return not (low.startswith(("sk-your", "sk-replace", "your-", "replace-"))
                or low in ("changeme", "placeholder"))


def api_key() -> str:
    """DEEPSEEK_API_KEY from the environment, then from the repo .env."""
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if _usable(key):
        return key
    key = _env_file(DEFAULT_ENV).get("DEEPSEEK_API_KEY", "").strip()
    return key if _usable(key) else ""


def _clip(text: str, limit: int) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…"


class WebSearch:
    """One Search API call at a time, with the same async shape as PySession."""

    def __init__(self, timeout: float = SEARCH_TIMEOUT_S,
                 max_results: int = SEARCH_MAX_RESULTS) -> None:
        self.timeout = timeout
        self.max_results = max_results
        self._io = threading.Lock()
        self._busy = threading.Lock()
        self._cancelled = False

    def key_configured(self) -> bool:
        return bool(api_key())

    def cancel(self) -> None:
        self._cancelled = True

    def reset(self) -> None:
        self._cancelled = True

    def stop(self) -> None:
        self._cancelled = True

    def run(self, query: str, count: int | None = None) -> dict:
        """Search the web. Never raises."""
        with self._io:
            self._cancelled = False
            started = time.monotonic()
            query = str(query or "").strip()
            if not query:
                return self._failure("search query is empty", query, started)
            if len(query) > 128:
                return self._failure(
                    "search query is longer than 128 characters", query,
                    started)
            key = api_key()
            if not key:
                return self._failure(
                    "no DeepSeek API key; put DEEPSEEK_API_KEY in "
                    f"{DEFAULT_ENV} or the environment", query, started)

            try:
                n = int(count if count is not None else self.max_results)
            except (TypeError, ValueError):
                n = self.max_results
            n = max(1, min(50, n))

            body = json.dumps({"query": query, "count": n,
                               "need_content": False}).encode("utf-8")
            request = urllib.request.Request(
                os.environ.get("DSV4_SEARCH_URL", SEARCH_URL),
                data=body, method="POST",
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                    "User-Agent": "dsv4-tui/websearch",
                })

            try:
                with urllib.request.urlopen(request,
                                            timeout=self.timeout) as resp:
                    raw = resp.read()
            except urllib.error.HTTPError as exc:
                return self._failure(self._http_error(exc), query, started)
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                reason = getattr(exc, "reason", exc)
                return self._failure(f"search request failed: {reason}",
                                     query, started)

            if self._cancelled:
                return self._failure("cancelled by user", query, started)
            try:
                data = json.loads(raw.decode("utf-8", "replace"))
            except (ValueError, TypeError):
                return self._failure("unreadable reply from search API",
                                     query, started)
            if not isinstance(data, dict):
                return self._failure("unexpected reply from search API",
                                     query, started)

            results = []
            for item in (data.get("results") or [])[:n]:
                if not isinstance(item, dict):
                    continue
                results.append({
                    "title": str(item.get("title") or item.get("url") or ""),
                    "url": str(item.get("url") or ""),
                    "summary": _clip(item.get("summary")
                                     or item.get("snippet") or "", 900),
                    "site_name": str(item.get("site_name") or ""),
                    "date": str(item.get("date") or ""),
                })
            return {
                "ok": True,
                "query": query,
                "results": results,
                "seconds": round(time.monotonic() - started, 3),
            }

    @staticmethod
    def _http_error(exc: urllib.error.HTTPError) -> str:
        detail = ""
        try:
            payload = json.loads(exc.read().decode("utf-8", "replace"))
            error = payload.get("error") if isinstance(payload, dict) else None
            if isinstance(error, dict):
                detail = str(error.get("message") or "")
            elif error:
                detail = str(error)
        except (OSError, ValueError, TypeError):
            pass
        if not detail:
            detail = str(exc.reason or "request rejected")
        return _clip(f"search API HTTP {exc.code}: {detail}", 500)

    def _failure(self, message: str, query: str, started: float) -> dict:
        return {
            "ok": False,
            "query": query,
            "results": [],
            "stderr": _clip(message, 500),
            "seconds": round(time.monotonic() - started, 3),
        }

    def run_async(self, query: str, count: int | None, on_done) -> bool:
        """Run on a worker thread; False when a search is already in flight."""
        if not self._busy.acquire(blocking=False):
            return False

        def work() -> None:
            try:
                result = self.run(query, count)
            except Exception as exc:  # defensive: a bug must not wedge the TUI
                result = self._failure(f"search error: {exc!r}", query,
                                       time.monotonic())
            finally:
                self._busy.release()
            try:
                on_done(result)
            except Exception:
                pass

        threading.Thread(target=work, daemon=True).start()
        return True
