"""Optional image candidates from Wikipedia (free, no API key, openly licensed content).

Failure here must never fail the research job, so every error is swallowed.
"""
import logging

import httpx

from .schemas import Image

log = logging.getLogger("images")
WIKI_API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "DeckPilot/1.0 (university course project)"}


def find_images(topic: str, limit: int = 3) -> list[Image]:
    try:
        resp = httpx.get(
            WIKI_API,
            params={
                "action": "query",
                "generator": "search",
                "gsrsearch": topic,
                "gsrlimit": 6,
                "prop": "pageimages",
                "piprop": "thumbnail",
                "pithumbsize": 900,
                "format": "json",
            },
            headers=HEADERS,
            timeout=8,
        )
        resp.raise_for_status()
        pages = (resp.json().get("query") or {}).get("pages", {})
        ordered = sorted(pages.values(), key=lambda p: p.get("index", 99))
        images = []
        for page in ordered:
            thumb = (page.get("thumbnail") or {}).get("source")
            if thumb:
                images.append(Image(url=thumb, caption=page.get("title", ""), source="Wikipedia"))
            if len(images) >= limit:
                break
        return images
    except Exception as exc:  # noqa: BLE001 - best effort by design
        log.warning("image lookup skipped: %s", exc)
        return []
