import os
import re
import time
import json
import hashlib
from urllib.parse import urlparse, urljoin, urldefrag

import requests
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET


BASE_URL = os.getenv("CA_BASE_URL", "https://www.cheshireacademy.org").rstrip("/")
OUT_DIR = os.getenv("CA_SOURCES_DIR", "sources")
MAX_PAGES = int(os.getenv("CA_MAX_PAGES", "2500"))          # 防止把全站抓爆；你可以调大
TIMEOUT = int(os.getenv("CA_HTTP_TIMEOUT", "15"))
SLEEP = float(os.getenv("CA_SLEEP", "0.10"))
USER_AGENT = os.getenv(
    "CA_UA",
    "Mozilla/5.0 (compatible; CA-Crawler/1.0; +https://www.cheshireacademy.org)"
)

# 允许的域名（可用逗号扩展，比如加 www.cheshireacademy.org）
ALLOWED_DOMAINS = [
    d.strip().lower()
    for d in os.getenv("CA_ALLOWED_DOMAINS", "cheshireacademy.org,www.cheshireacademy.org").split(",")
    if d.strip()
]

# 跳过的后缀（不抓二进制）
SKIP_EXT = {
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".ico",
    ".pdf", ".zip", ".rar", ".7z",
    ".mp4", ".mov", ".avi", ".mp3", ".wav",
    ".css", ".js", ".json",
    ".woff", ".woff2", ".ttf", ".eot",
}

PAGE_NOT_FOUND_PATTERNS = [
    r"\b404\b",
    r"page not found",
    r"we can['’]t find the page",
    r"the page you are looking for",
]

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": USER_AGENT})


def _norm_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return ""
    url, _ = urldefrag(url)
    return url.strip()


def _allowed(url: str) -> bool:
    try:
        p = urlparse(url)
        if p.scheme not in ("http", "https"):
            return False
        host = (p.netloc or "").lower()
        if not host:
            return False
        return any(host == d or host.endswith("." + d) for d in ALLOWED_DOMAINS)
    except Exception:
        return False


def _has_skip_ext(url: str) -> bool:
    path = (urlparse(url).path or "").lower()
    for ext in SKIP_EXT:
        if path.endswith(ext):
            return True
    return False


def _looks_like_404(html_text: str) -> bool:
    t = (html_text or "").lower()
    for pat in PAGE_NOT_FOUND_PATTERNS:
        if re.search(pat, t):
            return True
    return False


def _safe_filename(url: str) -> str:
    """
    把 URL 变成稳定文件名，避免重复/非法字符。
    """
    p = urlparse(url)
    path = p.path.strip("/")
    if not path:
        path = "home"
    path = re.sub(r"[^a-zA-Z0-9/_-]+", "-", path)
    path = path[:160].strip("-") or "page"
    h = hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]
    return f"{path.replace('/', '__')}__{h}.txt"


def _extract_text(html: str) -> str:
    soup = BeautifulSoup(html or "", "html.parser")
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()
    # 你也可以按站点结构删 header/nav/footer，提高“正文”纯度
    for tag in soup.select("header, nav, footer"):
        tag.decompose()

    text = soup.get_text(" ", strip=True)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _fetch(url: str):
    """
    返回 (ok, final_url, status, content_type, html)
    """
    try:
        r = SESSION.get(url, timeout=TIMEOUT, allow_redirects=True)
        ct = (r.headers.get("content-type") or "").lower()
        return True, r.url, r.status_code, ct, r.text
    except Exception:
        return False, url, 0, "", ""


def _discover_sitemaps_from_robots() -> list[str]:
    robots_url = urljoin(BASE_URL + "/", "robots.txt")
    ok, final_url, status, ct, body = _fetch(robots_url)
    if not ok or status >= 400:
        return []
    sitemaps = []
    for line in (body or "").splitlines():
        if line.lower().startswith("sitemap:"):
            sm = line.split(":", 1)[1].strip()
            sm = _norm_url(sm)
            if sm:
                sitemaps.append(sm)
    return list(dict.fromkeys(sitemaps))


def _default_sitemap_candidates() -> list[str]:
    return [
        urljoin(BASE_URL + "/", "sitemap.xml"),
        urljoin(BASE_URL + "/", "sitemap_index.xml"),
        urljoin(BASE_URL + "/", "sitemap-index.xml"),
        urljoin(BASE_URL + "/", "sitemap/sitemap.xml"),
    ]


def _parse_sitemap(xml_text: str) -> tuple[list[str], list[str]]:
    """
    返回 (urls, nested_sitemaps)
    """
    urls, nested = [], []
    if not xml_text:
        return urls, nested

    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return urls, nested

    tag = root.tag.lower()
    is_index = "sitemapindex" in tag
    is_urlset = "urlset" in tag

    ns = ""
    if "}" in root.tag:
        ns = root.tag.split("}", 1)[0] + "}"

    if is_index:
        for sm in root.findall(f".//{ns}sitemap/{ns}loc"):
            loc = _norm_url(sm.text or "")
            if loc:
                nested.append(loc)
    elif is_urlset:
        for u in root.findall(f".//{ns}url/{ns}loc"):
            loc = _norm_url(u.text or "")
            if loc:
                urls.append(loc)

    return urls, nested


def _collect_urls_from_sitemaps() -> list[str]:
    sitemaps = _discover_sitemaps_from_robots()
    if not sitemaps:
        sitemaps = _default_sitemap_candidates()

    seen_sm = set()
    all_urls = []
    queue = list(dict.fromkeys(sitemaps))

    while queue and len(seen_sm) < 200:
        sm = queue.pop(0)
        if sm in seen_sm:
            continue
        seen_sm.add(sm)

        ok, final_url, status, ct, body = _fetch(sm)
        if not ok or status >= 400:
            continue
        if "xml" not in ct and not (body or "").lstrip().startswith("<?xml"):
            continue

        urls, nested = _parse_sitemap(body)
        for u in urls:
            if _allowed(u) and not _has_skip_ext(u):
                all_urls.append(u)
        for n in nested:
            if n not in seen_sm:
                queue.append(n)

        time.sleep(SLEEP)

    # 去重
    all_urls = list(dict.fromkeys(all_urls))
    return all_urls


def _crawl_fallback_bfs(seed_urls: list[str], max_depth: int = 2) -> list[str]:
    """
    sitemap 不可用时备用：从种子页面 BFS 抓内部链接
    """
    out = []
    seen = set()
    q = [(u, 0) for u in seed_urls if _allowed(u)]

    while q and len(out) < MAX_PAGES:
        url, depth = q.pop(0)
        url = _norm_url(url)
        if not url or url in seen:
            continue
        seen.add(url)

        if _has_skip_ext(url):
            continue

        ok, final_url, status, ct, html = _fetch(url)
        if not ok or status >= 400:
            continue
        if "text/html" not in ct:
            continue
        if _looks_like_404(html):
            continue

        out.append(final_url)

        if depth < max_depth:
            soup = BeautifulSoup(html, "html.parser")
            for a in soup.find_all("a", href=True):
                href = _norm_url(a["href"])
                if not href:
                    continue
                if href.startswith("/"):
                    href = urljoin(BASE_URL + "/", href)
                if not _allowed(href):
                    continue
                if _has_skip_ext(href):
                    continue
                q.append((href, depth + 1))

        time.sleep(SLEEP)

    return list(dict.fromkeys(out))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    urls = _collect_urls_from_sitemaps()
    method = "sitemap"

    if not urls:
        # 备用：用更“多”的种子，但依然会过滤 404
        seeds = [
            BASE_URL + "/",
            BASE_URL + "/academics/",
            BASE_URL + "/admissions/",
            BASE_URL + "/campus-life/",
            BASE_URL + "/athletics/",
            BASE_URL + "/news/",
            BASE_URL + "/events/",
            BASE_URL + "/calendar/",
            BASE_URL + "/student-life/",
            BASE_URL + "/arts/",
            BASE_URL + "/about/",
            BASE_URL + "/parents/",
            BASE_URL + "/alumni/",
        ]
        urls = _crawl_fallback_bfs(seeds, max_depth=3)
        method = "bfs"

    # 最终再做一次“有效性”过滤（保证不把 not found 写进去）
    kept = []
    meta = []
    for i, url in enumerate(urls[:MAX_PAGES], start=1):
        ok, final_url, status, ct, html = _fetch(url)
        if not ok or status >= 400:
            continue
        if "text/html" not in ct:
            continue
        if _looks_like_404(html):
            continue

        text = _extract_text(html)
        if len(text) < 200:
            # 太短的页面（空壳/跳转/无内容）不写
            continue

        filename = _safe_filename(final_url)
        path = os.path.join(OUT_DIR, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"URL: {final_url}\n\n")
            f.write(text[:20000])  # 单页上限，避免 sources 太大

        kept.append(final_url)
        meta.append({"url": final_url, "file": filename, "status": status})
        print(f"[{i}] saved: {final_url} -> {filename}")
        time.sleep(SLEEP)

    with open(os.path.join(OUT_DIR, "_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"base_url": BASE_URL, "method": method, "count": len(meta), "items": meta},
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\nDONE. method={method}, saved_pages={len(meta)}")


if __name__ == "__main__":
    main()
