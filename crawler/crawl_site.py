#!/usr/bin/env python3
"""
Crawl Cheshire Academy public pages into plain-text files for RAG.

Key goals:
- Crawl as broadly as possible across official Cheshire Academy web properties,
  without pulling in random third-party domains.
- Prefer XML sitemaps (most complete + fewer 404s).
- Skip "soft 404" pages (custom 404 pages that still return HTTP 200).
- Optional PDF text extraction (for calendars/forms that are PDF).

Environment variables:
  BASE_URL            (default: https://cheshireacademy.org/)
  OUT_DIR             (default: hf_space/sources)
  MAX_PAGES           (default: 2500)
  RATE_LIMIT_SEC      (default: 0.35)

  INCLUDE_RE          optional regex filter to keep only matching URLs
  EXCLUDE_RE          optional regex filter to skip matching URLs

  # Allowed hosts
  # - If set, overrides automatic host derivation.
  #   Example: "cheshireacademy.org,www.cheshireacademy.org,give.cheshireacademy.org"
  ALLOWED_HOSTS

  # Extra sitemap URLs to try (comma-separated). Useful if the site changes.
  EXTRA_SITEMAPS

  # Content controls
  MIN_TEXT_LEN        (default: 120)  # skip extremely short pages
  INCLUDE_PDF         (default: 1)    # 1/true to extract PDFs, 0/false to skip
  MAX_FILE_MB         (default: 12)   # skip huge files (HTML/PDF)
  VERIFY_HOSTS        (default: 1)    # 1/true to drop invalid/soft-404 hosts
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple

import requests
from bs4 import BeautifulSoup


# --------------------------
# Config
# --------------------------

def _env_bool(name: str, default: bool) -> bool:
    v = (os.getenv(name, "").strip().lower() or ("1" if default else "0"))
    return v in ("1", "true", "yes", "y", "on")

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)).strip())
    except Exception:
        return default

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)).strip())
    except Exception:
        return default

def _env_csv(name: str) -> List[str]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]

@dataclass
class Config:
    base_url: str
    out_dir: Path
    max_pages: int
    rate_limit_sec: float
    include_re: Optional[re.Pattern]
    exclude_re: Optional[re.Pattern]
    allowed_hosts: Set[str]
    min_text_len: int
    include_pdf: bool
    max_file_mb: int
    verify_hosts: bool


def _canonicalize_url(u: str) -> str:
    u = (u or "").strip()
    if not u:
        return u
    # Remove fragments; keep query (some sites use it)
    parsed = urllib.parse.urlsplit(u)
    parsed = parsed._replace(fragment="")
    # Normalize scheme/host casing
    netloc = parsed.netloc.lower()
    scheme = parsed.scheme.lower() or "https"
    return urllib.parse.urlunsplit((scheme, netloc, parsed.path or "/", parsed.query, ""))


def _derive_allowed_hosts(base_url: str) -> Set[str]:
    """
    Allow:
      - base host
      - its www/non-www counterpart
      - give.cheshireacademy.org (official giving subdomain) if base is cheshireacademy.org*
    """
    host = urllib.parse.urlsplit(base_url).netloc.lower()
    hosts = {host}

    # add www/non-www variant
    if host.startswith("www."):
        hosts.add(host[len("www."):])
    else:
        hosts.add("www." + host)

    # default include the giving subdomain (safe + official)
    if host.endswith("cheshireacademy.org"):
        hosts.add("give.cheshireacademy.org")

    # user overrides
    override = _env_csv("ALLOWED_HOSTS")
    if override:
        hosts = {h.lower() for h in override}

    return {h for h in hosts if h}


def _get_config() -> Config:
    base_url = _canonicalize_url(os.getenv("BASE_URL", "https://cheshireacademy.org/"))
    out_dir = Path(os.getenv("OUT_DIR", "hf_space/sources")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    include_pat = os.getenv("INCLUDE_RE", "").strip()
    exclude_pat = os.getenv("EXCLUDE_RE", "").strip()
    include_re = re.compile(include_pat, re.I) if include_pat else None
    exclude_re = re.compile(exclude_pat, re.I) if exclude_pat else None

    allowed_hosts = _derive_allowed_hosts(base_url)

    return Config(
        base_url=base_url,
        out_dir=out_dir,
        max_pages=_env_int("MAX_PAGES", 2500),
        rate_limit_sec=_env_float("RATE_LIMIT_SEC", 0.35),
        include_re=include_re,
        exclude_re=exclude_re,
        allowed_hosts=allowed_hosts,
        min_text_len=_env_int("MIN_TEXT_LEN", 120),
        include_pdf=_env_bool("INCLUDE_PDF", True),
        max_file_mb=_env_int("MAX_FILE_MB", 12),
        verify_hosts=_env_bool("VERIFY_HOSTS", True),
    )


# --------------------------
# URL discovery (sitemap first)
# --------------------------

_SITEMAP_NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}


def _robots_sitemaps(session: requests.Session, base_url: str) -> List[str]:
    """Parse robots.txt for Sitemap: lines."""
    robots_url = urllib.parse.urljoin(base_url, "/robots.txt")
    try:
        r = session.get(robots_url, timeout=20, allow_redirects=True)
        if r.status_code != 200:
            return []
        sitemaps = []
        for line in r.text.splitlines():
            if line.lower().startswith("sitemap:"):
                sm = line.split(":", 1)[1].strip()
                if sm:
                    sitemaps.append(_canonicalize_url(sm))
        return sitemaps
    except Exception:
        return []


def _default_sitemap_candidates(base_url: str) -> List[str]:
    """
    Try a few common sitemap locations.
    We try both www and non-www variants because some sites redirect.
    """
    parsed = urllib.parse.urlsplit(base_url)
    host = parsed.netloc.lower()

    hosts_to_try = {host}
    if host.startswith("www."):
        hosts_to_try.add(host[len("www."):])
    else:
        hosts_to_try.add("www." + host)

    paths = [
        "/sitemap.xml",
        "/sitemap_index.xml",
        "/sitemap-index.xml",
        "/sitemap.php",
    ]

    out = []
    for h in hosts_to_try:
        for p in paths:
            out.append(_canonicalize_url(urllib.parse.urlunsplit((parsed.scheme, h, p, "", ""))))
    return out


def _parse_sitemap_xml(xml_text: str) -> Tuple[List[str], List[str]]:
    """
    Return (urls, sitemap_children).
    Handles both <urlset> and <sitemapindex>.
    """
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return ([], [])

    tag = root.tag.lower()
    urls: List[str] = []
    children: List[str] = []

    if tag.endswith("urlset"):
        for loc in root.findall(".//sm:url/sm:loc", _SITEMAP_NS):
            if loc.text:
                urls.append(_canonicalize_url(loc.text.strip()))
    elif tag.endswith("sitemapindex"):
        for loc in root.findall(".//sm:sitemap/sm:loc", _SITEMAP_NS):
            if loc.text:
                children.append(_canonicalize_url(loc.text.strip()))
    return (urls, children)


def _fetch_sitemap_urls(session: requests.Session, sitemap_url: str, max_children: int = 200) -> List[str]:
    """
    Fetch a sitemap URL. If it's a sitemap index, fetch its children.
    """
    try:
        r = session.get(sitemap_url, timeout=30, allow_redirects=True)
        if r.status_code != 200:
            return []
        # Some servers return text/html for XML; allow as long as it parses.
        xml = r.text
    except Exception:
        return []

    urls, children = _parse_sitemap_xml(xml)
    if urls:
        return urls

    out: List[str] = []
    for child in children[:max_children]:
        out.extend(_fetch_sitemap_urls(session, child, max_children=max_children))
    return out


def _is_allowed_host(url: str, cfg: Config) -> bool:
    try:
        host = urllib.parse.urlsplit(url).netloc.lower()
    except Exception:
        return False
    return host in cfg.allowed_hosts


def _discover_urls(session: requests.Session, cfg: Config) -> List[str]:
    """
    1) robots.txt sitemaps
    2) EXTRA_SITEMAPS
    3) common sitemap locations
    4) fallback: light link crawl from BASE_URL (same allowed hosts)
    """
    sitemap_candidates: List[str] = []

    sitemap_candidates.extend(_robots_sitemaps(session, cfg.base_url))
    sitemap_candidates.extend([_canonicalize_url(x) for x in _env_csv("EXTRA_SITEMAPS")])
    sitemap_candidates.extend(_default_sitemap_candidates(cfg.base_url))

    seen = set()
    sitemap_candidates = [u for u in sitemap_candidates if u and not (u in seen or seen.add(u))]

    urls: List[str] = []
    for sm in sitemap_candidates:
        got = _fetch_sitemap_urls(session, sm)
        if got:
            urls = got
            break

    # filter to allowed hosts, then include/exclude regex
    def keep(u: str) -> bool:
        if not u:
            return False
        if not _is_allowed_host(u, cfg):
            return False
        if cfg.include_re and not cfg.include_re.search(u):
            return False
        if cfg.exclude_re and cfg.exclude_re.search(u):
            return False
        return True

    urls = [u for u in urls if keep(u)]
    # de-dup
    out: List[str] = []
    seen2: Set[str] = set()
    for u in urls:
        if u not in seen2:
            seen2.add(u)
            out.append(u)

    if out:
        return out

    # ---------------- fallback crawl ----------------
    print("[crawl_site] No sitemap found; falling back to link crawl from BASE_URL")
    queue = [cfg.base_url]
    seen3: Set[str] = set()
    out2: List[str] = []

    while queue and len(out2) < cfg.max_pages:
        u = queue.pop(0)
        if u in seen3:
            continue
        seen3.add(u)

        if not keep(u):
            continue

        out2.append(u)

        try:
            r = session.get(u, timeout=25, allow_redirects=True)
            time.sleep(cfg.rate_limit_sec)
            if r.status_code != 200:
                continue
            ctype = (r.headers.get("Content-Type", "") or "").lower()
            if "text/html" not in ctype:
                continue

            soup = BeautifulSoup(r.text, "html.parser")
            for a in soup.select("a[href]"):
                href = (a.get("href") or "").strip()
                if not href:
                    continue
                nxt = _canonicalize_url(urllib.parse.urljoin(u, href))
                if keep(nxt) and nxt not in seen3:
                    queue.append(nxt)
        except Exception:
            continue

    return out2


# --------------------------
# Content extraction
# --------------------------

def _safe_filename(url: str) -> str:
    h = hashlib.sha256(url.encode("utf-8")).hexdigest()[:20]
    # keep a tiny hint from path for readability
    path = urllib.parse.urlsplit(url).path.strip("/").replace("/", "_")
    path = re.sub(r"[^a-zA-Z0-9_\-]+", "_", path)[:40] or "root"
    return f"{path}__{h}.txt"


def _looks_like_soft_404(title: str, text: str, url: str) -> bool:
    s = (title + " " + text).lower()
    bad = [
        "page not found",
        "404",
        "the page you requested could not be found",
        "we can't find the page",
        "error 404",
    ]
    if any(b in s for b in bad):
        return True
    if "/404" in url.lower():
        return True
    return False


def _extract_text_html(html: str, url: str) -> Tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()

    title = (soup.title.get_text(" ", strip=True) if soup.title else "").strip()
    text = soup.get_text("\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    header = f"URL: {url}\nTITLE: {title}\n\n"
    return header, (text or "")


def _extract_text_pdf(pdf_bytes: bytes, url: str) -> Tuple[str, str]:
    """
    Best-effort PDF extraction. If it fails, return empty.
    """
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception:
        return ("", "")

    import io
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        chunks = []
        for page in reader.pages[:30]:  # cap pages to keep files small
            t = page.extract_text() or ""
            if t.strip():
                chunks.append(t.strip())
        text = "\n\n".join(chunks).strip()
        header = f"URL: {url}\nTITLE: (PDF)\n\n"
        return header, text
    except Exception:
        return ("", "")


def _too_large(r: requests.Response, cfg: Config) -> bool:
    try:
        clen = r.headers.get("Content-Length")
        if clen:
            mb = int(clen) / (1024 * 1024)
            if mb > cfg.max_file_mb:
                return True
    except Exception:
        pass
    return False


def _verify_allowed_hosts(session: requests.Session, cfg: Config) -> Set[str]:
    """
    Ensure each allowed host is a real, public site (not a dead host or soft-404 landing page).
    This prevents "added domains" from being useless.
    """
    if not cfg.verify_hosts:
        return cfg.allowed_hosts

    parsed = urllib.parse.urlsplit(cfg.base_url)
    scheme = parsed.scheme or "https"
    good: Set[str] = set()

    for host in sorted(cfg.allowed_hosts):
        try:
            test_url = urllib.parse.urlunsplit((scheme, host, "/", "", ""))
            r = session.get(test_url, timeout=20, allow_redirects=True)
            if r.status_code != 200:
                continue
            ctype = (r.headers.get("Content-Type", "") or "").lower()
            if "text/html" in ctype:
                header, text = _extract_text_html(r.text, test_url)
                title_line = header.splitlines()[1].replace("TITLE:", "").strip() if "\nTITLE:" in header else ""
                if _looks_like_soft_404(title_line, text[:800], test_url):
                    continue
            good.add(host)
        except Exception:
            continue

    return good


# --------------------------
# Main
# --------------------------

def main() -> int:
    cfg = _get_config()
    print("[crawl_site] BASE_URL:", cfg.base_url)
    print("[crawl_site] Allowed hosts:", ", ".join(sorted(cfg.allowed_hosts)))

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (compatible; CA-RAG-Crawler/1.2; +https://cheshireacademy.org/)"
        }
    )
    cfg.allowed_hosts = _verify_allowed_hosts(session, cfg)
    print("[crawl_site] Verified hosts:", ", ".join(sorted(cfg.allowed_hosts)))

    urls = _discover_urls(session, cfg)
    if not urls:
        print("[crawl_site] No URLs discovered.")
        return 1

    urls = urls[: cfg.max_pages]

    manifest = {
        "base_url": cfg.base_url,
        "allowed_hosts": sorted(cfg.allowed_hosts),
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "items": [],
        "count": 0,
    }

    saved = 0
    for url in urls:
        try:
            r = session.get(url, timeout=35, allow_redirects=True)
            time.sleep(cfg.rate_limit_sec)

            if r.status_code != 200:
                continue
            if _too_large(r, cfg):
                continue

            ctype = (r.headers.get("Content-Type", "") or "").lower()
            header = ""
            text = ""

            if "text/html" in ctype:
                header, text = _extract_text_html(r.text, url)
                if not text or len(text) < cfg.min_text_len:
                    continue
                title_line = header.splitlines()[1].replace("TITLE:", "").strip() if "\nTITLE:" in header else ""
                if _looks_like_soft_404(title_line, text[:800], url):
                    continue

            elif cfg.include_pdf and ("application/pdf" in ctype or url.lower().endswith(".pdf")):
                header, text = _extract_text_pdf(r.content, url)
                if not text or len(text) < cfg.min_text_len:
                    continue
            else:
                continue

            fn = _safe_filename(url)
            (cfg.out_dir / fn).write_text(header + text, encoding="utf-8")

            manifest["items"].append({"url": url, "file": fn})
            saved += 1

        except Exception:
            continue

    manifest["count"] = saved
    (cfg.out_dir / "_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"[crawl_site] Saved {saved} pages into {cfg.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
