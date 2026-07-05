"""Fetch crypto/macro headlines from RSS feeds (free, no API key)."""

from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

import feedparser
import requests
import yaml

FEEDS_FILE = Path(__file__).parent / "feeds.yaml"
_UA = "Mozilla/5.0 (compatible; DoomIndexBot/1.0; +daily_signals)"


def _apply_ssl_fix() -> None:
    """On Windows, use the OS trust store (mirrors core/ssl_fix.py). No-op on Linux."""
    import sys
    if sys.platform != "win32":
        return
    try:
        import truststore
        truststore.inject_into_ssl()
    except ImportError:
        pass


_apply_ssl_fix()


@dataclass
class Headline:
    source: str
    tier: str
    weight: float
    title: str
    summary: str
    link: str
    published: str  # ISO 8601 UTC

    def as_dict(self) -> dict:
        return asdict(self)


def _load_feeds() -> list[dict]:
    with open(FEEDS_FILE, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)["feeds"]


def _entry_time(entry) -> datetime | None:
    for key in ("published_parsed", "updated_parsed"):
        t = entry.get(key)
        if t:
            return datetime.fromtimestamp(time.mktime(t), tz=timezone.utc)
    return None


def fetch_headlines(hours: int = 24) -> list[Headline]:
    """Return de-duplicated headlines published within the last `hours`."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    seen: set[str] = set()
    out: list[Headline] = []

    for feed in _load_feeds():
        try:
            resp = requests.get(feed["url"], headers={"User-Agent": _UA}, timeout=20)
            resp.raise_for_status()
            parsed = feedparser.parse(resp.content)
        except Exception as exc:  # network / parse errors must not kill the run
            print(f"[news_feed] WARN failed to fetch {feed['name']}: {exc}")
            continue

        for entry in parsed.entries:
            title = (entry.get("title") or "").strip()
            if not title:
                continue
            key = title.lower()
            if key in seen:
                continue

            ts = _entry_time(entry)
            if ts is not None and ts < cutoff:
                continue

            seen.add(key)
            out.append(
                Headline(
                    source=feed["name"],
                    tier=feed.get("tier", "crypto"),
                    weight=float(feed.get("weight", 1.0)),
                    title=title,
                    summary=(entry.get("summary") or "")[:400].strip(),
                    link=entry.get("link", ""),
                    published=(ts or datetime.now(timezone.utc)).isoformat(),
                )
            )

    print(f"[news_feed] collected {len(out)} headlines from {len(_load_feeds())} feeds")
    return out


if __name__ == "__main__":
    for h in fetch_headlines():
        print(f"[{h.source}] {h.title}")
