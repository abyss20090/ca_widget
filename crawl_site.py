#!/usr/bin/env python3
"""
Crawl a website (prefer sitemap) and write clean .txt sources for the HF Space RAG.

Designed for GitHub Actions:
- Uses polite crawling (rate limit + timeouts).
- Stays on allowed domains.
- Writes a manifest so the Space can hot-reload sources.

Env vars (optional):
  BASE_URL        (default: https://www.cheshireacademy.org/)
  OUT_DIR         (default: hf_space/sources)
  MAX_PAGES       (default: 600)
  RATE_LIMIT_SEC  (default: 0.3)
  USER_AGENT      (default: "CA-Chatbot-Crawler/1.0 (+GitHub Actions)")
  INCLUDE_REGEX   (default: "")
  EXCLUDE_REGEX   (default: r"(\\.(jpg|jpeg|png|gif|webp|svg|mp4|mov|zip|rar|7z)$)")
"""

from __future__ import annotations

import os
import re
import time
import json
import hashlib
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple, Dict

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import xml.etree.ElementTree as ET


@dataclass(frozen=True)
class Config:
    base_url: str
    out_dir: Path
    max_pages: int
    rate_limit_sec: float
    user_agent: str
    include_re: Optional[re.Pattern]
    exclude_re: Optional[re.Pattern]


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if not v:
        return default
    try:
        return int(v)
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if not v:
        return default
    try:
        return float(v)
    except Exception:
        return default


def _compile_pat(pat: str) -> Optional[re.Pattern]:
    pat = (pat or "").strip()
    if not pat:
        return None
    return re.compile(pat, re.IGNORECASE)


def _get_config() -> Config:
    base_url = (os.getenv("BASE_URL") or "https://www.cheshireacademy.org/").strip()
    if not base_url.endswith("/"):
        base_url += "/"

    out_dir = Path(os.getenv("OUT_DIR", "hf_space/sources")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    max_pages = _env_int("MAX_PAGES", 600)
    rate_limit_sec = _env_float("RATE_LIMIT_SEC", 0.3)
    user_agent = os.getenv("USER_AGENT", "CA-Chatbot-Crawler/1.0 (+GitHub Actions)")

    include_re = _compile_pat(os.getenv("INCLUDE_REGEX", ""))
    exclude_re = _compile_pat(os.getenv("EXCLUDE_REGEX", r"(\.(jpg|jpeg|png|gif|webp|svg|mp4|mov|zip|rar|7z)$)"))

    return Config(
        base_url=base_url,
        out_dir=out_dir,
        max_pages=max_pages,
        rate_limit_sec=rate_limit_sec,
        user_agent=user_agent,
        include_re=include_re,
        exclude_re=exclude_re,
    )


def _make_session(user_agent: str) -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=0.6,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "HEAD"),
        raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.mount("http://", HTTPAdapter(max_retries=retries))
    s.headers.update({"User-Agent": user_agent, "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"})
    return s


def _same_host(url_a: str, url_b: str) -> bool:
    try:
        a = urllib.parse.urlparse(url_a)
        b = urllib.parse.urlparse(url_b)
        return a.netloc.lower() == b.netloc.lower()
    except Exception:
        return False


def _normalize_url(url: str) -> str:
    # Remove fragments, normalize trailing slash in a consistent way (keep as-is otherwise)
    p = urllib.parse.urlparse(url)
    p = p._replace(fragment="")
    return p.geturl()


def _safe_filename(url: str) -> str:
    """
    Make a stable filename for a URL.
    Keep it readable but guarantee uniqueness using a short hash.
    """
    p = urllib.parse.urlparse(url)
    path = p.path.strip("/")
    if not path:
        path = "home"
    path = re.sub(r"[^a-zA-Z0-9/_-]+", "-", path)
    path = path.strip("-").replace("/", "__")
    h = hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]
    return f"{path}__{h}.txt"


def _extract_text(html: str, url: str) -> Tuple[str, str]:
    soup = BeautifulSoup(html, "lxml")

    # Title
    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()

    # Remove noisy sections
    for tag in soup(["script", "style", "noscript", "svg", "canvas", "form", "iframe"]):
        tag.decompose()
    for tag_name in ("header", "footer", "nav", "aside"):
        for t in soup.find_all(tag_name):
            t.decompose()

    text = soup.get_text(" ", strip=True)
    text = re.sub(r"\s+", " ", text).strip()

    # Keep the file useful for retrieval:
    # - add URL + title header
    header = f"URL: {url}\n"
    if title:
        header += f"TITLE: {title}\n"
    header += "\n"
    return header, text


def _get_robots_sitemap(session: requests.Session, base_url: str, timeout: int = 20) -> List[str]:
    """
    Try to discover sitemap URLs from robots.txt.
    """
    robots_url = urllib.parse.urljoin(base_url, "/robots.txt")
    try:
        r = session.get(robots_url, timeout=timeout)
        if r.status_code != 200:
            return []
        sitemaps = []
        for line in r.text.splitlines():
            if line.lower().startswith("sitemap:"):
                sm = line.split(":", 1)[1].strip()
                if sm:
                    sitemaps.append(sm)
        return list(dict.fromkeys(sitemaps))
    except Exception:
        return []


def _parse_sitemap_xml(xml_text: str) -> Tuple[List[str], List[str]]:
    """
    Returns (page_urls, nested_sitemaps)
    Handles common sitemap namespaces.
    """
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return ([], [])

    # Try both namespaced and non-namespaced queries
    def findall_any(tag: str) -> List[ET.Element]:
        elems = root.findall(tag)
        if elems:
            return elems
        # namespaced
        ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
        elems = root.findall(tag.replace("/", "/sm:"), ns)
        return elems

    page_urls = []
    for el in findall_any(".//url/loc"):
        if el.text:
            page_urls.append(el.text.strip())

    nested = []
    for el in findall_any(".//sitemap/loc"):
        if el.text:
            nested.append(el.text.strip())

    return (page_urls, nested)


def _fetch_sitemap_urls(session: requests.Session, base_url: str) -> List[str]:
    """
    Fetch sitemap URLs, including sitemap indexes.
    """
    candidates = _get_robots_sitemap(session, base_url)
    if not candidates:
        candidates = [
            urllib.parse.urljoin(base_url, "/sitemap.xml"),
            urllib.parse.urljoin(base_url, "/sitemap_index.xml"),
        ]

    seen_sm: Set[str] = set()
    out_urls: List[str] = []

    def walk(sm_url: str) -> None:
        sm_url = _normalize_url(sm_url)
        if sm_url in seen_sm:
            return
        seen_sm.add(sm_url)

        try:
            r = session.get(sm_url, timeout=30)
            if r.status_code != 200 or not r.text:
                return
            page_urls, nested = _parse_sitemap_xml(r.text)
            for u in page_urls:
                out_urls.append(_normalize_url(u))
            for n in nested:
                walk(n)
        except Exception:
            return

    for sm in candidates:
        walk(sm)

    # de-dup while preserving order
    dedup = []
    seen = set()
    for u in out_urls:
        if u not in seen:
            seen.add(u)
            dedup.append(u)
    return dedup


def _bfs_urls(session: requests.Session, base_url: str, max_pages: int, rate_limit_sec: float) -> List[str]:
    """
    Fallback crawl if sitemap isn't available.
    """
    queue: List[str] = [_normalize_url(base_url)]
    seen: Set[str] = set()
    out: List[str] = []

    while queue and len(out) < max_pages:
        url = queue.pop(0)
        if url in seen:
            continue
        seen.add(url)

        try:
            r = session.get(url, timeout=25)
            time.sleep(rate_limit_sec)
            if r.status_code != 200 or "text/html" not in r.headers.get("Content-Type", ""):
                continue
        except Exception:
            continue

        out.append(url)
        soup = BeautifulSoup(r.text, "lxml")
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if not href:
                continue
            nxt = urllib.parse.urljoin(url, href)
            nxt = _normalize_url(nxt)
            if _same_host(nxt, base_url):
                queue.append(nxt)

    # de-dup
    dedup = []
    seen2 = set()
    for u in out:
        if u not in seen2:
            seen2.add(u)
            dedup.append(u)
    return dedup


def _should_keep(url: str, cfg: Config) -> bool:
    u = url.lower()
    if cfg.exclude_re and cfg.exclude_re.search(u):
        return False
    if cfg.include_re and not cfg.include_re.search(u):
        return False
    # Same host only
    return _same_host(url, cfg.base_url)


def main() -> int:
    cfg = _get_config()
    session = _make_session(cfg.user_agent)

    urls = _fetch_sitemap_urls(session, cfg.base_url)
    if not urls:
        urls = _bfs_urls(session, cfg.base_url, cfg.max_pages, cfg.rate_limit_sec)

    urls = [u for u in urls if _should_keep(u, cfg)]
    urls = urls[: cfg.max_pages]

    # Clean output directory (but keep hidden cache if any)
    for p in cfg.out_dir.glob("*.txt"):
        try:
            p.unlink()
        except Exception:
            pass

    manifest: Dict[str, object] = {
        "base_url": cfg.base_url,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "count": 0,
        "items": [],
    }

    ok = 0
    for i, url in enumerate(urls, start=1):
        try:
            r = session.get(url, timeout=30)
            time.sleep(cfg.rate_limit_sec)
            ctype = r.headers.get("Content-Type", "")
            if r.status_code != 200 or "text/html" not in ctype:
                continue

            header, text = _extract_text(r.text, url)
            if not text or len(text) < 200:
                continue

            fn = _safe_filename(url)
            fp = cfg.out_dir / fn
            fp.write_text(header + text, encoding="utf-8")

            manifest["items"].append({"url": url, "file": fn})
            ok += 1
        except Exception:
            continue

    manifest["count"] = ok
    (cfg.out_dir / "_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[crawl_site] Saved {ok} pages into {cfg.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
