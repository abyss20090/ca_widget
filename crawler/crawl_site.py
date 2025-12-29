import os
import re
import time
import json
import hashlib
from collections import deque
from dataclasses import dataclass
from typing import Dict, Set, Tuple, List, Optional
from urllib.parse import urlparse, urljoin, urldefrag, parse_qsl, urlencode

import requests
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET


# =========================
# Config (env overridable)
# =========================
BASE_URL = os.getenv("CA_BASE_URL", "https://www.cheshireacademy.org").rstrip("/")
OUT_DIR = os.getenv("CA_SOURCES_DIR", "sources")

MAX_PAGES = int(os.getenv("CA_MAX_PAGES", "4000"))            # 最多保存多少页面（写入 sources）
MAX_DISCOVERED = int(os.getenv("CA_MAX_DISCOVERED", "50000")) # 最多发现多少 URL（队列上限，防爆）
TIMEOUT = float(os.getenv("CA_HTTP_TIMEOUT", "15"))
CONNECT_TIMEOUT = float(os.getenv("CA_CONNECT_TIMEOUT", "8"))
SLEEP = float(os.getenv("CA_SLEEP", "0.10"))
MAX_DEPTH = int(os.getenv("CA_MAX_DEPTH", "6"))               # BFS 最大深度（fallback 用）
MAX_HTML_CHARS = int(os.getenv("CA_MAX_HTML_CHARS", "250000"))# 防止抓到超大页面
MAX_TEXT_CHARS = int(os.getenv("CA_MAX_TEXT_CHARS", "25000")) # 单页写入 sources 上限

# 如果你体育内容在别的域名（比如 maxpreps / finalsite 子域），在 Actions 里设置：
# CA_ALLOWED_DOMAINS="cheshireacademy.org,www.cheshireacademy.org,athletics.cheshireacademy.org"
ALLOWED_DOMAINS = [
    d.strip().lower()
    for d in os.getenv("CA_ALLOWED_DOMAINS", "cheshireacademy.org,www.cheshireacademy.org").split(",")
    if d.strip()
]

USER_AGENT = os.getenv(
    "CA_UA",
    "Mozilla/5.0 (compatible; CA-Crawler/2.0; +https://www.cheshireacademy.org)"
)

# 只抓 HTML（避免 pdf/图片等二进制）
SKIP_EXT = {
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".ico",
    ".pdf", ".zip", ".rar", ".7z",
    ".mp4", ".mov", ".avi", ".mp3", ".wav",
    ".css", ".js", ".json",
    ".woff", ".woff2", ".ttf", ".eot",
}

# 常见“无限/噪声”页面（路径命中就跳过）
SKIP_PATH_CONTAINS = [
    "/wp-json/", "/xmlrpc.php", "/cgi-bin/", "/cdn-cgi/",
    "/ajax/", "/admin", "/login", "/signin", "/signup", "/auth",
    "/print/", "/printerfriendly/", "/emailtofriend",
]

# query 里出现这些 key，基本就是日历翻页/导出/过滤，容易无限扩张
BAD_QUERY_KEYS = {
    "ical", "ics", "format", "output", "view", "month", "year", "day",
    "date", "start", "end", "range", "from", "to",
    "calendar", "occurrence", "instance", "eventinstance",
    "add-to-calendar", "download", "share", "replytocom",
}

# UTM/追踪参数统一去掉（减少重复）
DROP_QUERY_KEYS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "gclid", "fbclid", "mc_cid", "mc_eid", "mkt_tok", "hsCtaTracking",
}

PAGE_NOT_FOUND_PATTERNS = [
    r"\b404\b",
    r"page not found",
    r"we can['’]t find the page",
    r"the page you are looking for",
    r"does not exist",
]

# 内容太短（空壳/跳转/无正文）就不写入 sources
MIN_TEXT_LEN = int(os.getenv("CA_MIN_TEXT_LEN", "220"))

# 网络错误重试
RETRIES = int(os.getenv("CA_RETRIES", "2"))
BACKOFF = float(os.getenv("CA_BACKOFF", "0.8"))

SESSION = requests.Session()
SESSION.headers.update(
    {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
    }
)


# =========================
# Helpers
# =========================
def _norm_url(url: str) -> str:
    """Normalize URL: strip fragment, canonicalize query (drop tracking, sort), remove trailing slash for non-root."""
    url = (url or "").strip()
    if not url:
        return ""
    url, _ = urldefrag(url)
    url = url.strip()

    try:
        p = urlparse(url)
        if not p.scheme:
            return url

        # normalize hostname casing
        netloc = (p.netloc or "").lower()

        # normalize path
        path = p.path or "/"
        # collapse multiple slashes
        path = re.sub(r"/{2,}", "/", path)

        # canonical query: drop tracking keys, keep rest, sort
        q = []
        for k, v in parse_qsl(p.query, keep_blank_values=True):
            lk = (k or "").lower()
            if lk in DROP_QUERY_KEYS:
                continue
            q.append((k, v))
        q.sort(key=lambda kv: (kv[0].lower(), kv[1]))

        query = urlencode(q, doseq=True)

        # strip trailing slash except root
        if path != "/" and path.endswith("/"):
            path = path[:-1]

        return p._replace(netloc=netloc, path=path, query=query).geturl()
    except Exception:
        return url


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
    return any(path.endswith(ext) for ext in SKIP_EXT)


def _looks_like_404(html_text: str) -> bool:
    t = (html_text or "").lower()
    for pat in PAGE_NOT_FOUND_PATTERNS:
        if re.search(pat, t):
            return True
    return False


def _is_noisy_or_infinite(url: str) -> bool:
    """Heuristic: avoid infinite calendar/filter pages and obvious noise."""
    try:
        p = urlparse(url)
        path = (p.path or "").lower()
        qs = dict((k.lower(), v) for k, v in parse_qsl(p.query, keep_blank_values=True))

        # path contains noisy segments
        for frag in SKIP_PATH_CONTAINS:
            if frag in path:
                return True

        # too many query params => often filters / infinite
        if len(qs) >= 4:
            return True

        # bad query keys
        for k in qs.keys():
            if k in BAD_QUERY_KEYS:
                return True

        # specific calendar-like patterns in query values
        # e.g. view=month, format=ical, date=2025-...
        for k, v in qs.items():
            vv = (v or "").lower()
            if k in ("view", "format") and vv in ("month", "week", "day", "ical", "ics", "rss"):
                return True
            if re.search(r"\b\d{4}-\d{2}-\d{2}\b", vv) and (k in BAD_QUERY_KEYS or "calendar" in path):
                return True

        return False
    except Exception:
        return False


def _safe_filename(url: str) -> str:
    p = urlparse(url)
    path = (p.path or "").strip("/") or "home"
    path = re.sub(r"[^a-zA-Z0-9/_-]+", "-", path)
    path = path[:160].strip("-") or "page"
    h = hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]
    return f"{path.replace('/', '__')}__{h}.txt"


def _extract_text(html: str) -> str:
    soup = BeautifulSoup(html or "", "html.parser")

    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()

    # 去掉站点通用噪声区（你也可以按需要增删）
    for tag in soup.select("header, nav, footer, form"):
        tag.decompose()

    text = soup.get_text(" ", strip=True)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _request_with_retries(method: str, url: str, **kwargs) -> Optional[requests.Response]:
    last_err = None
    for attempt in range(RETRIES + 1):
        try:
            r = SESSION.request(method, url, timeout=(CONNECT_TIMEOUT, TIMEOUT), allow_redirects=True, **kwargs)
            return r
        except Exception as e:
            last_err = e
            time.sleep(BACKOFF * (2 ** attempt))
    return None


@dataclass
class ProbeResult:
    ok: bool
    final_url: str
    status: int
    content_type: str


_probe_cache: Dict[str, ProbeResult] = {}


def _probe_html_200(url: str) -> ProbeResult:
    """
    入队前先验证：尽量用 HEAD；如果 HEAD 不给力，再用轻量 GET（stream 读取少量 bytes）。
    仅当 200 且 content-type 像 HTML 才算 ok。
    """
    url = _norm_url(url)
    if not url:
        return ProbeResult(False, url, 0, "")

    if url in _probe_cache:
        return _probe_cache[url]

    # quick rejects
    if not _allowed(url) or _has_skip_ext(url) or _is_noisy_or_infinite(url):
        res = ProbeResult(False, url, 0, "")
        _probe_cache[url] = res
        return res

    # 1) HEAD
    r = _request_with_retries("HEAD", url)
    if r is not None:
        ct = (r.headers.get("content-type") or "").lower()
        if r.status_code == 200 and ("text/html" in ct or "application/xhtml" in ct or ct == ""):
            res = ProbeResult(True, _norm_url(r.url), r.status_code, ct)
            _probe_cache[url] = res
            return res

        # 405/403/501: some servers block HEAD, fallback to GET
        if r.status_code not in (405, 403, 501):
            res = ProbeResult(False, _norm_url(r.url), r.status_code, ct)
            _probe_cache[url] = res
            return res

    # 2) lightweight GET (stream)
    r2 = _request_with_retries("GET", url, stream=True)
    if r2 is None:
        res = ProbeResult(False, url, 0, "")
        _probe_cache[url] = res
        return res

    ct2 = (r2.headers.get("content-type") or "").lower()
    final2 = _norm_url(r2.url)

    # only accept html-like
    if r2.status_code != 200 or ("text/html" not in ct2 and "application/xhtml" not in ct2):
        # close stream
        try:
            r2.close()
        except Exception:
            pass
        res = ProbeResult(False, final2, r2.status_code, ct2)
        _probe_cache[url] = res
        return res

    # read a small chunk to detect soft-404
    sample = ""
    try:
        chunk = next(r2.iter_content(chunk_size=4096), b"")
        sample = chunk.decode("utf-8", errors="ignore")
    except Exception:
        sample = ""
    finally:
        try:
            r2.close()
        except Exception:
            pass

    if _looks_like_404(sample):
        res = ProbeResult(False, final2, 200, ct2)
        _probe_cache[url] = res
        return res

    res = ProbeResult(True, final2, 200, ct2)
    _probe_cache[url] = res
    return res


def _fetch_full_html(url: str) -> Tuple[bool, str, int, str, str]:
    """
    真正抓取页面内容（用于写 sources）。
    返回 (ok, final_url, status, content_type, html)
    """
    r = _request_with_retries("GET", url)
    if r is None:
        return False, url, 0, "", ""

    ct = (r.headers.get("content-type") or "").lower()
    final = _norm_url(r.url)

    if r.status_code != 200:
        return False, final, r.status_code, ct, ""

    if "text/html" not in ct and "application/xhtml" not in ct:
        return False, final, r.status_code, ct, ""

    html = r.text or ""
    if len(html) > MAX_HTML_CHARS:
        html = html[:MAX_HTML_CHARS]

    if _looks_like_404(html):
        return False, final, 200, ct, ""

    return True, final, 200, ct, html


# =========================
# Discovery: sitemap + robots + feeds + bfs
# =========================
def _discover_sitemaps_from_robots() -> List[str]:
    robots_url = urljoin(BASE_URL + "/", "robots.txt")
    ok, final, status, ct, html = _fetch_full_html(robots_url)
    # robots.txt 不是 html，所以这里不用 _fetch_full_html；用轻量 GET
    r = _request_with_retries("GET", robots_url)
    if r is None or r.status_code >= 400:
        return []
    body = r.text or ""
    sitemaps = []
    for line in body.splitlines():
        if line.lower().startswith("sitemap:"):
            sm = line.split(":", 1)[1].strip()
            sm = _norm_url(sm)
            if sm:
                sitemaps.append(sm)
    return list(dict.fromkeys(sitemaps))


def _default_sitemap_candidates() -> List[str]:
    return [
        urljoin(BASE_URL + "/", "sitemap.xml"),
        urljoin(BASE_URL + "/", "sitemap_index.xml"),
        urljoin(BASE_URL + "/", "sitemap-index.xml"),
        urljoin(BASE_URL + "/", "sitemap/sitemap.xml"),
    ]


def _parse_sitemap(xml_text: str) -> Tuple[List[str], List[str]]:
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


def _fetch_sitemap_text(url: str) -> str:
    r = _request_with_retries("GET", url)
    if r is None or r.status_code >= 400:
        return ""
    ct = (r.headers.get("content-type") or "").lower()
    text = r.text or ""
    if "xml" in ct or text.lstrip().startswith("<?xml") or "<urlset" in text or "<sitemapindex" in text:
        return text
    return ""


def _collect_urls_from_sitemaps() -> List[str]:
    sitemaps = _discover_sitemaps_from_robots()
    if not sitemaps:
        sitemaps = _default_sitemap_candidates()

    seen_sm: Set[str] = set()
    all_urls: List[str] = []
    queue = deque(list(dict.fromkeys([_norm_url(s) for s in sitemaps if s])))

    while queue and len(seen_sm) < 300:
        sm = queue.popleft()
        if not sm or sm in seen_sm:
            continue
        seen_sm.add(sm)

        body = _fetch_sitemap_text(sm)
        if not body:
            continue

        urls, nested = _parse_sitemap(body)
        for u in urls:
            u = _norm_url(u)
            if not u:
                continue
            if _allowed(u) and (not _has_skip_ext(u)) and (not _is_noisy_or_infinite(u)):
                all_urls.append(u)

        for n in nested:
            n = _norm_url(n)
            if n and n not in seen_sm:
                queue.append(n)

        time.sleep(SLEEP)

    return list(dict.fromkeys(all_urls))


def _feed_candidates() -> List[str]:
    # 常见 feed 入口（有些站是 /rss.xml 或 /feed/）
    return [
        urljoin(BASE_URL + "/", "feed/"),
        urljoin(BASE_URL + "/", "rss"),
        urljoin(BASE_URL + "/", "rss.xml"),
        urljoin(BASE_URL + "/", "atom.xml"),
        urljoin(BASE_URL + "/", "news/rss"),
        urljoin(BASE_URL + "/", "athletics/rss"),
    ]


def _collect_urls_from_feeds() -> List[str]:
    out: List[str] = []
    for f in _feed_candidates():
        r = _request_with_retries("GET", f)
        if r is None or r.status_code >= 400:
            continue
        text = (r.text or "").strip()
        if not text:
            continue

        # crude parse: look for <link>...</link> or href="..."
        # (feed 格式多样，这里用稳一点的正则兜底)
        links = set()

        for m in re.finditer(r"<link>([^<]+)</link>", text, re.I):
            links.add(_norm_url(m.group(1).strip()))
        for m in re.finditer(r'href="([^"]+)"', text, re.I):
            links.add(_norm_url(m.group(1).strip()))

        for u in links:
            if u.startswith("/"):
                u = urljoin(BASE_URL + "/", u)
            u = _norm_url(u)
            if _allowed(u) and (not _has_skip_ext(u)) and (not _is_noisy_or_infinite(u)):
                out.append(u)

        time.sleep(SLEEP)

    return list(dict.fromkeys(out))


def _extract_links_from_html(base_url: str, html: str) -> List[str]:
    soup = BeautifulSoup(html or "", "html.parser")
    found = []

    # a href
    for a in soup.find_all("a", href=True):
        href = (a.get("href") or "").strip()
        if not href:
            continue
        if href.startswith("mailto:") or href.startswith("tel:") or href.startswith("javascript:"):
            continue
        if href.startswith("#"):
            continue

        if href.startswith("/"):
            href = urljoin(BASE_URL + "/", href)
        else:
            href = urljoin(base_url, href)

        href = _norm_url(href)
        if not href:
            continue
        found.append(href)

    # also canonical link
    can = soup.find("link", rel=lambda x: x and "canonical" in x.lower())
    if can and can.get("href"):
        c = can["href"].strip()
        if c.startswith("/"):
            c = urljoin(BASE_URL + "/", c)
        c = _norm_url(c)
        if c:
            found.append(c)

    return found


def _seed_urls() -> List[str]:
    # 你关心体育新闻/战报：这里把 athletics/news 等入口优先加入
    return [
        BASE_URL + "/",
        BASE_URL + "/athletics/",
        BASE_URL + "/athletics/news/",
        BASE_URL + "/news/",
        BASE_URL + "/events/",
        BASE_URL + "/calendar/",
        BASE_URL + "/campus-life/",
        BASE_URL + "/student-life/",
        BASE_URL + "/about/",
        BASE_URL + "/admissions/",
        BASE_URL + "/alumni/",
        BASE_URL + "/parents/",
    ]


def _discover_site_bfs(initial: List[str], max_depth: int = MAX_DEPTH) -> List[str]:
    """
    BFS 自动发现站内链接。
    关键点：新 URL 入队前先 probe，确保 200+HTML，避免 404/二进制/无限页污染队列。
    """
    discovered: List[str] = []
    seen: Set[str] = set()

    q = deque()
    for u in initial:
        u = _norm_url(u)
        if not u:
            continue
        q.append((u, 0))

    while q and len(discovered) < MAX_DISCOVERED:
        url, depth = q.popleft()
        url = _norm_url(url)
        if not url or url in seen:
            continue
        seen.add(url)

        if not _allowed(url) or _has_skip_ext(url) or _is_noisy_or_infinite(url):
            continue

        # 入队前验证（这里已经在队列里了，所以做一次 probe 决定是否继续展开）
        pr = _probe_html_200(url)
        if not pr.ok:
            continue

        final_url = pr.final_url
        if final_url and final_url not in seen:
            # 规范化后可能是新 URL（重定向）
            seen.add(final_url)

        discovered.append(final_url)

        if depth >= max_depth:
            continue

        # 拉取少量内容用于发现链接（更省流量：只拿完整 html 的前 MAX_HTML_CHARS）
        ok, final2, status2, ct2, html = _fetch_full_html(final_url)
        if not ok:
            continue

        links = _extract_links_from_html(final2, html)

        for link in links:
            link = _norm_url(link)
            if not link or link in seen:
                continue
            if not _allowed(link) or _has_skip_ext(link) or _is_noisy_or_infinite(link):
                continue

            # 新 link：入队前先验证 200+HTML
            pr2 = _probe_html_200(link)
            if pr2.ok:
                q.append((pr2.final_url, depth + 1))

        time.sleep(SLEEP)

        if len(discovered) >= MAX_DISCOVERED:
            break

    return list(dict.fromkeys(discovered))


# =========================
# Main: collect candidates -> fetch full -> write sources
# =========================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) sitemap urls
    urls_sitemap = _collect_urls_from_sitemaps()

    # 2) feeds urls
    urls_feeds = _collect_urls_from_feeds()

    # 3) bfs discovery (from seeds + sitemap/feed 的前一部分做“引爆”)
    seeds = _seed_urls()
    ignite = []
    ignite.extend(seeds)
    ignite.extend(urls_sitemap[:200])
    ignite.extend(urls_feeds[:200])
    ignite = list(dict.fromkeys([_norm_url(u) for u in ignite if u]))

    urls_bfs = _discover_site_bfs(ignite, max_depth=MAX_DEPTH)

    # merge candidates (keep order preference)
    candidates = []
    for bucket in (urls_sitemap, urls_feeds, urls_bfs):
        for u in bucket:
            u = _norm_url(u)
            if u and u not in candidates:
                candidates.append(u)

    # 最终：逐个抓取全文写入 sources（再次做硬校验）
    kept: List[str] = []
    meta: List[dict] = []

    saved = 0
    for i, url in enumerate(candidates, start=1):
        if saved >= MAX_PAGES:
            break

        url = _norm_url(url)
        if not url:
            continue
        if not _allowed(url) or _has_skip_ext(url) or _is_noisy_or_infinite(url):
            continue

        ok, final_url, status, ct, html = _fetch_full_html(url)
        if not ok:
            continue

        text = _extract_text(html)
        if len(text) < MIN_TEXT_LEN:
            continue

        filename = _safe_filename(final_url)
        path = os.path.join(OUT_DIR, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"URL: {final_url}\n\n")
            f.write(text[:MAX_TEXT_CHARS])

        saved += 1
        kept.append(final_url)
        meta.append(
            {
                "url": final_url,
                "file": filename,
                "status": status,
                "content_type": ct,
            }
        )

        print(f"[{saved}/{MAX_PAGES}] saved: {final_url} -> {filename}")
        time.sleep(SLEEP)

    manifest = {
        "base_url": BASE_URL,
        "saved_pages": len(meta),
        "candidates_total": len(candidates),
        "buckets": {
            "sitemap": len(urls_sitemap),
            "feeds": len(urls_feeds),
            "bfs": len(urls_bfs),
        },
        "items": meta,
    }

    with open(os.path.join(OUT_DIR, "_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("\nDONE.")
    print(json.dumps({k: manifest[k] for k in ["base_url", "saved_pages", "candidates_total", "buckets"]}, indent=2))


if __name__ == "__main__":
    main()
