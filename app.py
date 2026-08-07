import os
import re
import uuid
import ipaddress
import socket
import threading
import time
import mimetypes
import base64
import shutil
import random
import json
import html
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from urllib.parse import urlparse, urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError
from PIL import Image
import requests

load_dotenv()

app = Flask(__name__)

# The upload is now just a list of URLs (pasted text or a small .txt/.csv) -
# no more multi-gigabyte HTML dumps, so a generous-but-sane cap is enough.
MAX_CONTENT_LENGTH_MB = 16
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH_MB * 1024 * 1024
JOB_MAX_AGE_SECONDS = 7200       # Job retention time in memory (2h) - long batches must never expire mid-run
MAX_WORKERS = 5                  # Number of parallel connections (page fetch + OpenAI)
PAGE_IMAGE_WORKERS = 3            # Concurrent image downloads+ALT generations *within* a single page

JOBS = {}
JOBS_LOCK = threading.Lock()
# Statuses a job never leaves on its own - background_worker is done touching
# it once it's in one of these, so it's the only point clean_old_jobs() is
# allowed to reap it from.
FINISHED_JOB_STATUSES = {"completed", "error", "stopped_error"}

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
TMP_UPLOADS_DIR = os.path.join(PROJECT_DIR, "tmp_uploads")
os.makedirs(TMP_UPLOADS_DIR, exist_ok=True)

CONSECUTIVE_ERROR_LIMIT = 5          # Isolated transient errors shouldn't abort the whole batch
MAX_URLS_PER_BATCH = 1000            # Safety cap on how many page URLs one batch can queue (after sitemap expansion)
MAX_SITEMAP_URLS_PER_FILE = 2000     # Cap per individual sitemap/sitemap-index fetch
MAX_SITEMAP_DEPTH = 3                # How many levels of nested sitemap indexes to follow
MAX_IMAGES_PER_PAGE = 30             # Safety cap on how many images one page can queue (incl. the main image)
PAGE_FETCH_TIMEOUT_SECONDS = 20
MAX_PAGE_FETCH_BYTES = 15 * 1024 * 1024   # a product page's HTML/sitemap XML shouldn't exceed this
MAX_URL_REDIRECTS = 5


# ---------------------------------------------------------------------------
# SSRF guard + generic URL fetching
# ---------------------------------------------------------------------------

def _check_hostname_is_public(hostname: str):
    """Resolves hostname and rejects private/loopback/link-local/reserved IPs.
    Basic SSRF guard - every URL processed here comes from user input and
    could point at internal/network-local addresses."""
    try:
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        raise ValueError("Nie udało się rozwiązać hosta w adresie URL.")
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast or ip.is_unspecified:
            raise ValueError("Adres URL wskazuje na prywatny/wewnętrzny adres i został zablokowany.")


def _is_http_url(text: str) -> bool:
    parsed = urlparse(text)
    return parsed.scheme in ("http", "https") and bool(parsed.hostname)


def _fetch_url_bytes(url: str, max_bytes: int) -> tuple:
    """SSRF-guarded GET with manual redirect re-validation (each hop is
    re-checked, same as the image downloader) and a byte-size cap. Returns
    (content_bytes, final_url, content_type) - used for both page HTML and
    sitemap XML fetches, which are always parsed in memory."""
    session = requests.Session()
    current_url = url

    for _ in range(MAX_URL_REDIRECTS + 1):
        parsed = urlparse(current_url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ValueError("Nieprawidłowy URL (dozwolone są tylko linki http/https).")
        _check_hostname_is_public(parsed.hostname)

        response = session.get(
            current_url, stream=True, timeout=PAGE_FETCH_TIMEOUT_SECONDS,
            headers={"User-Agent": "AltTextGenerator/1.0"}, allow_redirects=False,
        )
        try:
            if response.is_redirect or response.is_permanent_redirect:
                location = response.headers.get("Location")
                if not location:
                    raise ValueError("Serwer zwrócił przekierowanie bez adresu docelowego.")
                current_url = urljoin(current_url, location)
                continue

            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "").split(";")[0].strip().lower()

            chunks = []
            total = 0
            for chunk in response.iter_content(chunk_size=65536):
                total += len(chunk)
                if total > max_bytes:
                    raise ValueError(f"Zawartość przekracza limit {max_bytes // (1024 * 1024)} MB.")
                chunks.append(chunk)
            return b"".join(chunks), current_url, content_type
        finally:
            response.close()

    raise ValueError("Zbyt wiele przekierowań.")


def fetch_page_html(url: str) -> tuple:
    """Fetches a product page's live HTML. Returns (html_text, final_url)."""
    content, final_url, content_type = _fetch_url_bytes(url, MAX_PAGE_FETCH_BYTES)
    if content_type and not (content_type.startswith("text/html") or content_type.startswith("application/xhtml")):
        raise ValueError(f"Adres nie zwraca strony HTML (Content-Type: {content_type}).")
    try:
        html_text = content.decode("utf-8")
    except UnicodeDecodeError:
        html_text = content.decode("utf-8", errors="replace")
    return html_text, final_url


# ---------------------------------------------------------------------------
# Page content extraction (og:title/description + every real <img> on the
# page) - deliberately still not a full DOM parse, just the handful of tags
# we actually need.
# ---------------------------------------------------------------------------

_META_IMAGE_KEYS = ("og:image", "og:image:url", "og:image:secure_url", "twitter:image", "twitter:image:src")
IMG_SRC_ATTR_PRIORITY = ("data-lazy-src", "data-src", "data-original", "src")

# Priority 3 signals for the main-image cascade: an <img> is treated as "the"
# product photo if it's explicitly hinted as high-priority, or sits inside a
# gallery/media container - even without any OpenGraph/JSON-LD data.
_GALLERY_CONTAINER_MARKERS = ("gallery", "product-media")
_VOID_TAGS = {
    "area", "base", "br", "col", "embed", "hr", "img", "input",
    "link", "meta", "param", "source", "track", "wbr",
}

# Real product-description containers, ranked best-match first - checked
# against every open tag's itemprop/class/id so the body's actual
# description text can be captured instead of the marketing-blurb meta
# description ("Dobra cena", "Szybka wysyłka"...). Deliberately precise (no
# generic <article>/<main> fallback) - a broad fallback is exactly how
# cross-sell/recommendation widget text used to leak into the context.
DESCRIPTION_MAX_CHARS = 300
_DESCRIPTION_ID_RANKS = (
    ("description", 1),
    ("product-description", 2),
    ("tab-description", 5),
)
_DESCRIPTION_CLASS_RANKS = (
    ("product-description", 3),
    ("tab-content", 4),
    ("description-content", 6),
)

# Cross-sell/recommendation/navigation containers to ignore entirely while
# scanning for the description - these are the actual source of the
# "wrong product's description ends up in the context" bug: a recommended-
# products widget or "recently viewed" carousel can itself contain an
# element that matches one of the description selectors above.
_EXCLUDED_CONTAINER_CLASS_MARKERS = (
    "cross-sell", "crosssell", "related", "related-products", "recommended",
    "bestsellers", "recently-viewed", "product-slider", "carousel", "widget", "sidebar",
)
_EXCLUDED_CONTAINER_TAGS = {"header", "footer", "nav"}
_EXCLUDED_CONTAINER_IDS = {"header", "footer"}

# Marketing/price chrome that leaks in from nearby product-card/widget
# fragments even inside an otherwise legitimate description container -
# filtered out fragment-by-fragment rather than discarding the whole
# description just because one sentence happens to mention a price.
_DESCRIPTION_NOISE_PHRASES = (
    "najniższa cena", "zł", "zobacz", "kup teraz", "darmowa dostawa",
)


def _is_excluded_description_container(tag: str, attrs_dict: dict) -> bool:
    if tag in _EXCLUDED_CONTAINER_TAGS:
        return True

    elid = (attrs_dict.get("id") or "").strip().lower()
    if elid in _EXCLUDED_CONTAINER_IDS:
        return True

    classes = (attrs_dict.get("class") or "").lower().split()
    return any(marker in cls for cls in classes for marker in _EXCLUDED_CONTAINER_CLASS_MARKERS)


def _description_container_rank(tag: str, attrs_dict: dict):
    """Returns the priority rank (lower = better) if this tag looks like a
    dedicated product-description container, else None. Rank order:
    itemprop, then id/class keywords roughly by specificity."""
    itemprop = (attrs_dict.get("itemprop") or "").strip().lower()
    if itemprop == "description":
        return 0

    elid = (attrs_dict.get("id") or "").strip().lower()
    for marker, rank in _DESCRIPTION_ID_RANKS:
        if elid == marker:
            return rank

    classes = (attrs_dict.get("class") or "").lower().split()
    for marker, rank in _DESCRIPTION_CLASS_RANKS:
        if marker in classes:
            return rank

    return None


def _finalize_description_text(buffer: list) -> str:
    """Joins a captured description container's raw text fragments,
    dropping any individual fragment that looks like marketing/price noise
    leaked in from nearby widget chrome (a price tag, a 'Kup teraz' button)
    rather than discarding the whole description over one bad fragment."""
    kept_fragments = []
    for fragment in buffer:
        collapsed = ' '.join(fragment.split())
        if not collapsed:
            continue
        lowered = collapsed.lower()
        if any(phrase in lowered for phrase in _DESCRIPTION_NOISE_PHRASES):
            continue
        kept_fragments.append(collapsed)
    return ' '.join(kept_fragments)


def _is_svg_url(url: str) -> bool:
    return urlparse(url).path.lower().endswith(".svg")


def _decode_unicode_js_escapes(text: str) -> str:
    """Undoes JS-style \\uXXXX escapes (and the \\/ escape) that leak into
    image URLs when a page embeds them as a raw JSON string (Magento/Hyva
    inline state, JSON-LD, etc.) instead of a real HTML attribute."""
    def repl(match):
        try:
            return chr(int(match.group(1), 16))
        except ValueError:
            return match.group(0)
    return re.sub(r'\\u([0-9a-fA-F]{4})', repl, text).replace('\\/', '/')


def sanitize_image_url(raw_url, page_url: str):
    """Mandatory cleanup for every image URL candidate before it's used or
    stored: decodes JS unicode escapes and HTML entities, resolves relative
    paths against the page URL, and validates the result is a genuine,
    well-formed http(s) URL. Returns None if the candidate can't be turned
    into a usable URL."""
    if not raw_url or not isinstance(raw_url, str):
        return None

    text = raw_url.strip()
    if not text:
        return None

    text = _decode_unicode_js_escapes(text)
    text = html.unescape(text)
    text = re.sub(r'\s+', '', text).strip()

    if not text or text.startswith(("data:", "javascript:", "#")):
        return None

    if not text.lower().startswith(("http://", "https://")):
        text = urljoin(page_url, text)

    parsed = urlparse(text)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return None

    return text


def _find_jsonld_product_nodes(ld_json_blocks: list) -> list:
    """Parses every <script type="application/ld+json"> block and returns
    every schema.org Product node found (optionally nested in an @graph
    array), in document order - shared by the main-image and description
    cascades below."""
    nodes = []
    for block in ld_json_blocks:
        block = block.strip()
        if not block:
            continue
        try:
            data = json.loads(block)
        except (ValueError, TypeError):
            continue

        items = data if isinstance(data, list) else [data]
        expanded = []
        for item in items:
            if isinstance(item, dict) and isinstance(item.get("@graph"), list):
                expanded.extend(item["@graph"])
            else:
                expanded.append(item)

        for item in expanded:
            if not isinstance(item, dict):
                continue
            item_type = item.get("@type")
            type_list = item_type if isinstance(item_type, list) else [item_type]
            if any(isinstance(t, str) and t.lower() == "product" for t in type_list):
                nodes.append(item)

    return nodes


def _extract_jsonld_product_image(ld_json_blocks: list):
    """Priority 2 of the main-image cascade: the first Product node's
    "image" field."""
    for item in _find_jsonld_product_nodes(ld_json_blocks):
        image = item.get("image")
        if isinstance(image, list) and image:
            image = image[0]
        if isinstance(image, dict):
            image = image.get("url") or image.get("@id")
        if isinstance(image, str) and image.strip():
            return image.strip()

    return None


def _extract_jsonld_product_description(ld_json_blocks: list):
    """Priority 1 of the description cascade: the first Product node's
    "description" field - schema.org structured data is authored for that
    exact product, so unlike scanning the body it can never leak text from a
    cross-sell/recommendation widget elsewhere on the page."""
    for item in _find_jsonld_product_nodes(ld_json_blocks):
        description = item.get("description")
        if isinstance(description, str) and description.strip():
            return description.strip()

    return None


class _PageParser(HTMLParser):
    """Single pass over the page: captures <title>/<h1>/og:title (title
    context), the actual body description text (from a product-description
    container - never the SEO meta description), the main-image cascade
    signals (og:image/twitter:image, JSON-LD Product.image, fetchpriority/
    eager/gallery <img> hints), AND every <img> tag actually rendered on the
    page (so nothing gets missed)."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.title = ""
        self.og_title = ""
        self.h1 = ""
        self.meta_image_candidates = []
        self.img_urls = []
        self.priority_img_urls = []
        self.ld_json_blocks = []
        self.description_candidates = {}  # rank -> first captured text for that rank
        self._in_title = False
        self._in_h1 = False
        self._h1_done = False
        self._h1_buffer = []
        self._in_ldjson = False
        self._ldjson_buffer = ""
        self._seen_img_urls = set()
        self._seen_priority_urls = set()
        self._container_stack = []  # bool per open non-void ancestor: "looks like a gallery"
        self._description_stack = []  # {"rank": int, "buffer": list} per open non-void ancestor
        self._excluded_stack = []  # bool per open non-void ancestor: "cross-sell/nav/widget etc."

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        classes = (attrs_dict.get("class") or "").lower().split()
        elid = (attrs_dict.get("id") or "").lower()
        is_gallery_frame = (
            any(marker in cls for cls in classes for marker in _GALLERY_CONTAINER_MARKERS)
            or any(marker in elid for marker in _GALLERY_CONTAINER_MARKERS)
        )
        is_excluded_frame = _is_excluded_description_container(tag, attrs_dict)
        currently_excluded = is_excluded_frame or any(self._excluded_stack)
        description_rank = None if currently_excluded else _description_container_rank(tag, attrs_dict)

        if tag == "title":
            self._in_title = True
        elif tag == "h1" and not self._h1_done:
            self._in_h1 = True
        elif tag == "script":
            script_type = (attrs_dict.get("type") or "").strip().lower()
            if script_type == "application/ld+json":
                self._in_ldjson = True
                self._ldjson_buffer = ""
        elif tag == "meta":
            key = (attrs_dict.get("property") or attrs_dict.get("name") or "").strip().lower()
            content = (attrs_dict.get("content") or "").strip()
            if content and key == "og:title" and not self.og_title:
                self.og_title = content
            elif content and key in _META_IMAGE_KEYS and not content.startswith("data:"):
                if content not in self.meta_image_candidates:
                    self.meta_image_candidates.append(content)
        elif tag == "img":
            chosen = None
            for attr in IMG_SRC_ATTR_PRIORITY:
                v = attrs_dict.get(attr)
                if v and v.strip() and not v.strip().startswith("data:"):
                    chosen = v.strip()
                    break
            if not chosen:
                srcset = attrs_dict.get("srcset") or attrs_dict.get("data-srcset")
                if srcset:
                    first = srcset.split(",")[0].strip().split(" ")[0]
                    if first and not first.startswith("data:"):
                        chosen = first

            if chosen and chosen not in self._seen_img_urls:
                self._seen_img_urls.add(chosen)
                self.img_urls.append(chosen)

            if chosen and chosen not in self._seen_priority_urls:
                fetchpriority = (attrs_dict.get("fetchpriority") or "").strip().lower()
                loading = (attrs_dict.get("loading") or "").strip().lower()
                in_gallery = is_gallery_frame or any(self._container_stack)
                if fetchpriority == "high" or loading == "eager" or in_gallery:
                    self._seen_priority_urls.add(chosen)
                    self.priority_img_urls.append(chosen)

        if tag not in _VOID_TAGS:
            self._container_stack.append(is_gallery_frame)
            self._description_stack.append(
                {"rank": description_rank, "buffer": []} if description_rank is not None else None
            )
            self._excluded_stack.append(is_excluded_frame)

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
        elif tag == "h1":
            if self._in_h1:
                self.h1 = ' '.join(''.join(self._h1_buffer).split())
                self._in_h1 = False
                self._h1_done = True
        elif tag == "script":
            if self._in_ldjson:
                self.ld_json_blocks.append(self._ldjson_buffer)
            self._in_ldjson = False

        if tag not in _VOID_TAGS:
            if self._container_stack:
                self._container_stack.pop()
            if self._excluded_stack:
                self._excluded_stack.pop()
            if self._description_stack:
                frame = self._description_stack.pop()
                if frame is not None and frame["rank"] not in self.description_candidates:
                    text = _finalize_description_text(frame["buffer"])
                    if text:
                        self.description_candidates[frame["rank"]] = text

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        if self._in_h1:
            self._h1_buffer.append(data)
        if self._in_ldjson:
            self._ldjson_buffer += data
        for frame in self._description_stack:
            if frame is not None:
                frame["buffer"].append(data)


def extract_page_content(html_text: str, page_url: str) -> dict:
    """Pulls the page context (H1/og:title + a description that belongs to
    THIS product only) plus every image on the page. The title prefers the
    visible <h1>, falling back to og:title then <title>.

    The description follows a cascade, strongest/safest signal first:
    (1) schema.org Product.description from JSON-LD - authored for exactly
    this product, so it can never leak cross-sell/recommendation text;
    (2) failing that, a dedicated description container in the body
    (itemprop="description", #description, #product-description,
    .product-description, .tab-content, #tab-description,
    .description-content) - with cross-sell/related/recommended/carousel/
    widget/sidebar/header/footer/nav containers explicitly excluded from the
    scan, and marketing/price fragments ("zł", "Kup teraz", ...) dropped
    from whatever text is captured. NOT the SEO meta description, which is
    usually marketing filler rather than real product knowledge.

    The main image is picked via a cascade, strongest signal first: (1)
    og:image/twitter:image meta tags, (2) JSON-LD Product.image, (3) an <img>
    hinted as high-priority/eager or sitting inside a gallery container -
    falling back to the first <img> on the page if none of that is present.
    Every candidate URL is sanitized/decoded/resolved to an absolute http(s)
    URL before use. SVG icons are excluded - everything else (including
    small graphics) is kept, since size alone isn't a reliable signal of
    relevance here."""
    parser = _PageParser()
    try:
        parser.feed(html_text)
    except Exception:
        pass  # tolerate malformed markup - keep whatever was parsed so far

    title = re.sub(r'\s+', ' ', (parser.h1 or parser.og_title or parser.title or "").strip())

    # Priority 1: schema.org Product.description from JSON-LD.
    jsonld_description = _extract_jsonld_product_description(parser.ld_json_blocks)
    if jsonld_description:
        description = re.sub(r'\s+', ' ', jsonld_description)[:DESCRIPTION_MAX_CHARS].strip()
    else:
        # Priority 2: a dedicated description container in the body.
        description = ""
        if parser.description_candidates:
            best_rank = min(parser.description_candidates)
            description = parser.description_candidates[best_rank][:DESCRIPTION_MAX_CHARS].strip()

    context_parts = []
    if title:
        context_parts.append(f"Produkt: {title}")
    if description:
        context_parts.append(f"Opis: {description}")
    context = " | ".join(context_parts) or "Brak dodatkowego kontekstu tekstowego."

    seen = set()

    def add_candidates(raw_urls):
        clean_urls = []
        for raw_url in raw_urls:
            clean = sanitize_image_url(raw_url, page_url)
            if clean and not _is_svg_url(clean) and clean not in seen:
                seen.add(clean)
                clean_urls.append(clean)
        return clean_urls

    # Priority 1: OpenGraph / Twitter meta image.
    meta_candidates = add_candidates(parser.meta_image_candidates)

    # Priority 2: JSON-LD structured data (schema.org Product.image).
    jsonld_image = _extract_jsonld_product_image(parser.ld_json_blocks)
    jsonld_candidates = add_candidates([jsonld_image] if jsonld_image else [])

    # Priority 3: a dedicated/eager-loaded <img> or one inside a gallery container.
    priority_candidates = add_candidates(parser.priority_img_urls)

    # Everything else on the page, queued as additional images.
    body_images = []
    for raw_url in parser.img_urls:
        if len(seen) >= MAX_IMAGES_PER_PAGE:
            break
        body_images.extend(add_candidates([raw_url]))

    ranked_main_candidates = meta_candidates + jsonld_candidates + priority_candidates
    main_url = ranked_main_candidates[0] if ranked_main_candidates else (body_images[0] if body_images else None)

    remaining = [u for u in ranked_main_candidates if u != main_url] + [u for u in body_images if u != main_url]
    main_fallbacks = remaining
    other_urls = remaining[:MAX_IMAGES_PER_PAGE]

    return {
        "context": context,
        "title": title,
        "main_url": main_url,
        "main_fallbacks": main_fallbacks,
        "other_urls": other_urls,
    }


# ---------------------------------------------------------------------------
# Sitemap expansion
# ---------------------------------------------------------------------------

def _looks_like_sitemap_url(url: str) -> bool:
    path = urlparse(url).path.lower()
    return path.endswith(".xml") or "sitemap" in path


def fetch_sitemap_urls(sitemap_url: str, depth: int = 0) -> list:
    """Fetches a sitemap.xml (or sitemap index) and returns the page URLs it
    lists, following nested sitemap indexes up to MAX_SITEMAP_DEPTH."""
    if depth > MAX_SITEMAP_DEPTH:
        return []

    content, _final_url, _content_type = _fetch_url_bytes(sitemap_url, MAX_PAGE_FETCH_BYTES)
    try:
        root = ET.fromstring(content)
    except ET.ParseError:
        return []

    tag = root.tag.lower()
    urls = []

    if tag.endswith("sitemapindex"):
        sub_sitemaps = []
        for sitemap_el in root:
            for loc in sitemap_el.iter():
                if loc.tag.lower().endswith("loc") and loc.text and loc.text.strip():
                    sub_sitemaps.append(loc.text.strip())
                    break
            if len(sub_sitemaps) >= MAX_SITEMAP_URLS_PER_FILE:
                break
        for sub_url in sub_sitemaps:
            if not _is_http_url(sub_url):
                continue
            try:
                urls.extend(fetch_sitemap_urls(sub_url, depth=depth + 1))
            except Exception:
                continue
            if len(urls) >= MAX_SITEMAP_URLS_PER_FILE:
                break
    elif tag.endswith("urlset"):
        for url_el in root:
            for loc in url_el.iter():
                if loc.tag.lower().endswith("loc") and loc.text and loc.text.strip():
                    loc_url = loc.text.strip()
                    if _is_http_url(loc_url):
                        urls.append(loc_url)
                    break
            if len(urls) >= MAX_SITEMAP_URLS_PER_FILE:
                break

    return urls[:MAX_SITEMAP_URLS_PER_FILE]


def expand_seed_urls(seed_urls: list) -> list:
    """Any seed that looks like (or turns out to be) a sitemap gets expanded
    into the page URLs it lists; everything else is used as-is."""
    final_urls = []
    seen = set()
    for seed in seed_urls:
        candidates = [seed]
        if _looks_like_sitemap_url(seed):
            try:
                sitemap_urls = fetch_sitemap_urls(seed)
                if sitemap_urls:
                    candidates = sitemap_urls
            except Exception:
                candidates = [seed]  # fall back to treating it as a literal page URL

        for url in candidates:
            if url not in seen:
                seen.add(url)
                final_urls.append(url)
        if len(final_urls) >= MAX_URLS_PER_BATCH:
            break

    return final_urls[:MAX_URLS_PER_BATCH]


def parse_url_list_text(raw_text: str) -> list:
    """Extracts one URL per non-empty line (also tolerates CSV-ish rows -
    the first http(s) token on a line is picked up, the rest is ignored)."""
    urls = []
    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        for token in re.split(r'[;,\s]+', line):
            token = token.strip().strip('"')
            if _is_http_url(token):
                urls.append(token)
                break

    seen = set()
    deduped = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


# ---------------------------------------------------------------------------
# Image download
# ---------------------------------------------------------------------------

MAX_URL_IMAGE_BYTES = 25 * 1024 * 1024


def download_image_from_url(url: str, dest_dir: str) -> str:
    """Downloads an image from a URL into dest_dir. Manually follows redirects
    (re-validating the host on each hop) and enforces scheme/type/size limits."""
    session = requests.Session()
    current_url = url

    for _ in range(MAX_URL_REDIRECTS + 1):
        parsed = urlparse(current_url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ValueError("Nieprawidłowy URL (dozwolone są tylko linki http/https).")
        _check_hostname_is_public(parsed.hostname)

        response = session.get(
            current_url, stream=True, timeout=PAGE_FETCH_TIMEOUT_SECONDS,
            headers={"User-Agent": "AltTextGenerator/1.0"}, allow_redirects=False,
        )
        try:
            if response.is_redirect or response.is_permanent_redirect:
                location = response.headers.get("Location")
                if not location:
                    raise ValueError("Serwer zwrócił przekierowanie bez adresu docelowego.")
                current_url = urljoin(current_url, location)
                continue

            response.raise_for_status()

            content_type = response.headers.get("Content-Type", "").split(";")[0].strip().lower()
            if not content_type.startswith("image/") or content_type == "image/svg+xml":
                raise ValueError(f"URL nie jest obrazem (Content-Type: {content_type or 'nieznany'}).")

            ext = mimetypes.guess_extension(content_type) or os.path.splitext(parsed.path)[1] or ".jpg"
            if ext == ".jpe":
                ext = ".jpg"

            dest_path = os.path.join(dest_dir, f"{uuid.uuid4().hex}{ext}")
            total = 0
            with open(dest_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=65536):
                    total += len(chunk)
                    if total > MAX_URL_IMAGE_BYTES:
                        raise ValueError(f"Obraz przekracza limit {MAX_URL_IMAGE_BYTES // (1024 * 1024)} MB.")
                    f.write(chunk)
            return dest_path
        finally:
            response.close()

    raise ValueError("Zbyt wiele przekierowań podczas pobierania obrazu.")


# ---------------------------------------------------------------------------
# Image compression
# ---------------------------------------------------------------------------

MAX_IMAGE_DIMENSION = 768
JPEG_QUALITY = 90
JPEG_SAFETY_MAX_BYTES = 400 * 1024


def compress_image(image_path: str) -> str:
    """Compresses an image on disk (downscaling dimensions + high-quality
    save) before it's sent to the AI model. Returns the path to the
    compressed file (may differ from the input path if the format changed)."""
    try:
        with Image.open(image_path) as img:
            if getattr(img, "is_animated", False):
                return image_path

            has_alpha = img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info)

            if max(img.size) > MAX_IMAGE_DIMENSION:
                ratio = MAX_IMAGE_DIMENSION / max(img.size)
                new_size = (max(1, int(img.size[0] * ratio)), max(1, int(img.size[1] * ratio)))
                img = img.resize(new_size, Image.LANCZOS)

            base_path = os.path.splitext(image_path)[0]

            if has_alpha:
                out_path = base_path + "_c.png"
                img.convert("RGBA").save(out_path, format="PNG", optimize=True)
            else:
                out_path = base_path + "_c.jpg"
                rgb_img = img.convert("RGB")
                rgb_img.save(out_path, format="JPEG", quality=JPEG_QUALITY, optimize=True)
                if os.path.getsize(out_path) > JPEG_SAFETY_MAX_BYTES:
                    rgb_img.save(out_path, format="JPEG", quality=JPEG_QUALITY - 15, optimize=True)

        if out_path != image_path and os.path.exists(image_path):
            os.remove(image_path)
        return out_path
    except Exception:
        return image_path


# ---------------------------------------------------------------------------
# OpenAI Vision integration
# ---------------------------------------------------------------------------

OPENAI_MODEL = "gpt-4o-mini"
OPENAI_TIMEOUT_SECONDS = 60
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=OPENAI_TIMEOUT_SECONDS)

ALT_TEXT_SYSTEM_PROMPT = (
    "Jesteś ekspertem SEO i WCAG dla sklepów e-commerce. Otrzymujesz zdjęcie oraz opis "
    "produktu wyciągnięty z treści strony (body).\n\n"
    "ZASADY ANALIZY I GENEROWANIA ALT:\n"
    "1. OCENA POWIĄZANIA GRAFIKI Z PRODUKTEM:\n"
    "   - Przed wygenerowaniem opisu oceń, czy grafika przedstawia PRODUKT (lub jego "
    "część/użycie), czy jest to GRAFIKA NIEZWIĄZANA / ELEMENT UNIWERSALNY (np. logo "
    "sklepu, ikona dostawy, baner płatności, certyfikat, grafika ozdobna).\n"
    "2. JEŚLI GRAFIKA JEST ZWIĄZANA Z PRODUKTEM:\n"
    "   - Wykorzystaj opis z body jako wskazówkę, czym dokładnie jest przedmiot (np. "
    "forma preparatu, przeznaczenie, składnik, sposób użycia).\n"
    "   - Stwórz precyzyjny ALT opisujący to, co widać (np. 'Opakowanie preparatu "
    "probiotycznego Otibiom w kapsułkach do pielęgnacji uszu psa').\n"
    "   - NIE ZACZYNAJ każdego opisu od tej samej, sztywnej sekwencji słów z opisu - "
    "wpleć markę/model naturalnie w zdanie (nie zawsze na początku) i różnicuj strukturę "
    "zdań między kolejnymi zdjęciami.\n"
    "3. JEŚLI GRAFIKA NIE JEST ZWIĄZANA Z PRODUKTEM:\n"
    "   - ABSOLUTNY ZAKAZ dodawania informacji z opisu produktu lub nazwy produktu!\n"
    "   - Opisz WYŁĄCZNIE to, co jest na obrazku (np. 'Logo sklepu Zooclick', 'Ikona "
    "darmowej dostawy powyżej 100 zł', 'Płatność kartą Visa i Mastercard').\n"
    "4. WYMOGI FORMALNE:\n"
    "   - Zwięźle: 4-12 słów, maksymalnie 120 znaków.\n"
    "   - Pisz po polsku, poprawnie gramatycznie i naturalnie dla człowieka, z poprawnymi "
    "znakami diakrytycznymi.\n"
    "   - Brak zwrotów typu 'Zdjęcie przedstawia', 'Obrazek z', 'Grafika'.\n"
    "   - Odpowiadasz WYŁĄCZNIE gotowym tekstem alt, bez cudzysłowów, bez prefiksów typu "
    "'Alt:', bez pytań i komentarzy."
)

ALT_TEXT_PROMPT_TEMPLATE = (
    "OPIS PRODUKTU Z TREŚCI STRONY (informacja pomocnicza, wyciągnięta automatycznie z "
    "sekcji opisu produktu w body - NIE z meta description - wykorzystaj ją tylko jeśli "
    "grafika faktycznie przedstawia produkt): {context}\n\n"
    "Przeanalizuj załączone zdjęcie: najpierw oceń, czy jest ono związane z produktem, "
    "czy jest elementem uniwersalnym/niezwiązanym (logo, ikona, baner, certyfikat), a "
    "następnie stwórz do niego tekst ALT zgodnie z zasadami z instrukcji systemowej."
)

# Shared "cooldown" between worker threads so a burst of 429s doesn't cause every
# thread to retry at the same moment.
_rate_limit_lock = threading.Lock()
_rate_limit_until = 0.0


def _wait_for_shared_rate_limit():
    with _rate_limit_lock:
        wait_until = _rate_limit_until
    remaining = wait_until - time.time()
    if remaining > 0:
        time.sleep(remaining)


def _register_rate_limit_cooldown(wait_seconds: float):
    global _rate_limit_until
    with _rate_limit_lock:
        candidate = time.time() + wait_seconds
        if candidate > _rate_limit_until:
            _rate_limit_until = candidate


def _extract_retry_after_seconds(rate_limit_error):
    try:
        headers = getattr(rate_limit_error.response, "headers", None)
        if headers and headers.get("retry-after"):
            return float(headers["retry-after"])
    except Exception:
        pass

    match = re.search(r"try again in ([\d.]+)(ms|s)", str(rate_limit_error))
    if match:
        value = float(match.group(1))
        return value / 1000 if match.group(2) == "ms" else value

    return None


def generate_alt_via_openai(image_path: str, context: str) -> str:
    media_type = mimetypes.guess_type(image_path)[0] or "image/jpeg"
    with open(image_path, "rb") as f:
        encoded_image = base64.b64encode(f.read()).decode("utf-8")
    data_uri = f"data:{media_type};base64,{encoded_image}"

    prompt_text = ALT_TEXT_PROMPT_TEMPLATE.format(context=context or "Brak dodatkowego kontekstu.")

    max_retries = 8
    last_error = None
    alt = ""

    for attempt in range(max_retries):
        _wait_for_shared_rate_limit()

        try:
            response = openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": ALT_TEXT_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            # detail="low" -> fixed ~85 tokens per image instead of hundreds
                            # from tiling with "auto"/"high" - important for parallel batches.
                            {"type": "image_url", "image_url": {"url": data_uri, "detail": "low"}},
                        ],
                    },
                ],
                max_completion_tokens=60,
            )
            alt = (response.choices[0].message.content or "").strip()
            last_error = None
            break
        except RateLimitError as e:
            last_error = e
            if attempt < max_retries - 1:
                backoff = min(30.0, 2.0 * (2 ** attempt)) + random.uniform(0, 1)
                wait = max(_extract_retry_after_seconds(e) or 0.0, backoff)
                _register_rate_limit_cooldown(wait)
                time.sleep(wait)
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(4 * (attempt + 1))

    if last_error is not None:
        raise RuntimeError(f"Błąd API OpenAI: {last_error}")

    if "\n" in alt:
        lines = [line.strip() for line in alt.split("\n") if line.strip()]
        alt = lines[-1] if lines else alt

    return alt.strip('"').strip("'").strip()


def _get_cached_image_result(task_id: str, image_url: str):
    """Returns a copy of a previously-generated result for this exact image
    URL within the same job, if any. A shared asset (logo, delivery icon,
    payment banner) that repeats across many product pages in the same
    batch then skips the download + OpenAI call entirely on every repeat
    after the first - 0 ms, 0 API requests. Safe to call from any worker
    thread: only the dict lookup happens under JOBS_LOCK, never the slow
    network/API work."""
    if not task_id:
        return None
    with JOBS_LOCK:
        job = JOBS.get(task_id)
        if not job:
            return None
        cached = job.get("image_cache", {}).get(image_url)
        return dict(cached) if cached is not None else None


def _store_cached_image_result(task_id: str, image_url: str, result: dict):
    if not task_id:
        return
    with JOBS_LOCK:
        job = JOBS.get(task_id)
        if job is not None:
            job.setdefault("image_cache", {})[image_url] = dict(result)


def process_single_image(image_url: str, context: str, job_dir: str, fallback_urls: list = None,
                          task_id: str = None) -> dict:
    cached = _get_cached_image_result(task_id, image_url)
    if cached is not None:
        return cached

    # og:image/twitter:image variants can point at the same photo under
    # different (hash-based) cache URLs - one of them 404-ing shouldn't sink
    # the whole image, so try each candidate in turn.
    candidate_urls = [image_url] + [u for u in (fallback_urls or []) if u != image_url]
    local_path = None
    used_url = image_url
    last_error = None
    for candidate_url in candidate_urls:
        try:
            local_path = download_image_from_url(candidate_url, job_dir)
            used_url = candidate_url
            break
        except Exception as e:
            last_error = e
    if local_path is None:
        raise last_error if last_error is not None else RuntimeError("Nie udało się pobrać obrazu.")

    compressed_path = compress_image(local_path)
    alt_text = generate_alt_via_openai(compressed_path, context)

    media_type = mimetypes.guess_type(compressed_path)[0] or "image/jpeg"
    with open(compressed_path, "rb") as f:
        encoded_image = base64.b64encode(f.read()).decode("utf-8")

    result = {
        "image_url": used_url,
        "context": context,
        "alt": alt_text,
        "skipped": False,
        "skip_reason": None,
        "image_data": f"data:{media_type};base64,{encoded_image}",
    }
    # Only successful results are cached - a transient download/API failure
    # shouldn't poison every later page that happens to share the URL.
    _store_cached_image_result(task_id, image_url, result)
    return result


def _process_image_safe(image_url: str, context: str, job_dir: str, fallback_urls: list = None,
                         task_id: str = None) -> dict:
    """process_single_image, but never raises - a failure on one image (main
    or one of the "others") shouldn't take down the rest of the page."""
    try:
        return process_single_image(image_url, context, job_dir, fallback_urls=fallback_urls, task_id=task_id)
    except Exception as e:
        return {
            "image_url": image_url,
            "context": context,
            "alt": f"Błąd przetwarzania: {str(e)}",
            "skipped": False,
            "skip_reason": None,
            "image_data": "",
        }


def process_page_url(page_url: str, job_dir: str, task_id: str = None) -> dict:
    """One full unit of work: fetch a product page, pull every image on it
    (main + the rest) from its markup, and generate an ALT for each one.
    The main image is processed first, then the remaining images are
    processed concurrently (up to PAGE_IMAGE_WORKERS at a time) - each
    candidate is checked against the job's shared image_cache first, so an
    asset that repeats across pages is only ever downloaded/analyzed once."""
    html_text, final_url = fetch_page_html(page_url)
    content = extract_page_content(html_text, final_url)

    result = {
        "page_url": page_url,
        "context": content["context"],
        "main_image": None,
        "other_images": [],
    }

    if not content["main_url"]:
        result["main_image"] = {
            "image_url": "",
            "context": content["context"],
            "alt": None,
            "skipped": True,
            "skip_reason": "Nie znaleziono żadnej grafiki na stronie.",
            "image_data": "",
        }
        return result

    result["main_image"] = _process_image_safe(
        content["main_url"], content["context"], job_dir,
        fallback_urls=content["main_fallbacks"], task_id=task_id,
    )

    other_urls = content["other_urls"]
    if other_urls:
        with ThreadPoolExecutor(max_workers=min(PAGE_IMAGE_WORKERS, len(other_urls))) as page_executor:
            # executor.map preserves input order in its output, so results
            # still line up with other_urls despite running concurrently.
            result["other_images"] = list(page_executor.map(
                lambda u: _process_image_safe(u, content["context"], job_dir, task_id=task_id),
                other_urls,
            ))

    return result


# ---------------------------------------------------------------------------
# Background job processing
# ---------------------------------------------------------------------------

def clean_old_jobs():
    """Reaps only jobs that have actually finished (status in
    FINISHED_JOB_STATUSES) AND are older than JOB_MAX_AGE_SECONDS. A job
    still 'parsing'/'processing' is never removed here, no matter its age -
    deleting it out from under a still-running background_worker thread is
    exactly what caused 404s ("Zadanie nie istnieje lub wygasło") on jobs
    that were still actively working. A finished job also stays in JOBS
    until this age cutoff, so the frontend always has time to poll the
    final 'completed' status before it's cleaned up."""
    now = time.time()
    with JOBS_LOCK:
        expired_ids = [
            t_id for t_id, job in JOBS.items()
            if job.get("status") in FINISHED_JOB_STATUSES
            and now - job.get("created_at", now) > JOB_MAX_AGE_SECONDS
        ]
        for t_id in expired_ids:
            del JOBS[t_id]


def background_worker(task_id: str, seed_urls: list, job_dir: str):
    consecutive_errors = 0
    try:
        with JOBS_LOCK:
            if task_id in JOBS:
                JOBS[task_id]["status"] = "parsing"

        try:
            final_urls = expand_seed_urls(seed_urls)
        except Exception as e:
            with JOBS_LOCK:
                if task_id in JOBS:
                    JOBS[task_id]["status"] = "error"
                    JOBS[task_id]["error_message"] = f"Błąd przygotowania listy adresów: {str(e)}"
            return

        if not final_urls:
            with JOBS_LOCK:
                if task_id in JOBS:
                    JOBS[task_id]["status"] = "error"
                    JOBS[task_id]["error_message"] = "Nie znaleziono żadnych prawidłowych adresów URL do przetworzenia."
            return

        with JOBS_LOCK:
            if task_id in JOBS:
                JOBS[task_id]["total"] = len(final_urls)
                JOBS[task_id]["status"] = "processing"

        stop_event = threading.Event()

        def worker_task(page_url):
            nonlocal consecutive_errors
            if stop_event.is_set():
                return

            is_error = False
            error_detail = None
            try:
                res = process_page_url(page_url, job_dir, task_id=task_id)
            except Exception as e:
                is_error = True
                error_detail = str(e)
                res = {
                    "page_url": page_url,
                    "context": "",
                    "main_image": {
                        "image_url": "", "context": "", "alt": f"Błąd przetwarzania: {error_detail}",
                        "skipped": False, "skip_reason": None, "image_data": "",
                    },
                    "other_images": [],
                }

            with JOBS_LOCK:
                if task_id not in JOBS or stop_event.is_set():
                    return

                JOBS[task_id]["results"].append(res)
                JOBS[task_id]["processed"] += 1

                # consecutive_errors is read/incremented/reset exclusively
                # inside this JOBS_LOCK block - multiple worker_task threads
                # run concurrently, so mutating it outside the lock would be
                # a race condition.
                if is_error:
                    JOBS[task_id]["error_count"] += 1
                    consecutive_errors += 1

                    if consecutive_errors >= CONSECUTIVE_ERROR_LIMIT:
                        stop_event.set()
                        JOBS[task_id]["status"] = "stopped_error"
                        succ = JOBS[task_id]["success_count"]
                        tot = JOBS[task_id]["total"]
                        JOBS[task_id]["error_message"] = (
                            f"Zatrzymano z powodu serii błędów: {error_detail} "
                            f"Poprawnie wygenerowano ALT dla {succ}/{tot} podstron."
                        )
                else:
                    JOBS[task_id]["success_count"] += 1
                    consecutive_errors = 0

        max_workers = min(MAX_WORKERS, len(final_urls))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(worker_task, u) for u in final_urls]
            for future in as_completed(futures):
                if stop_event.is_set():
                    break
                future.result()

        with JOBS_LOCK:
            if task_id in JOBS and JOBS[task_id]["status"] == "processing":
                JOBS[task_id]["status"] = "completed"

    except Exception as e:
        with JOBS_LOCK:
            if task_id in JOBS:
                JOBS[task_id]["status"] = "error"
                JOBS[task_id]["error_message"] = str(e)
    finally:
        if job_dir and os.path.exists(job_dir):
            shutil.rmtree(job_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.errorhandler(413)
def request_entity_too_large(error):
    max_mb = app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)
    return jsonify({
        "error": f"Przekroczono maksymalny rozmiar żądania ({max_mb} MB). Zmniejsz listę adresów i spróbuj ponownie."
    }), 413


@app.route('/')
def home():
    return render_template('index.html', max_urls_per_batch=MAX_URLS_PER_BATCH)


@app.route('/generate-alt', methods=['POST'])
def generate_alt():
    clean_old_jobs()

    urls_text = (request.form.get('urls_text') or "").strip()
    urls_file = request.files.get('urls_file')

    seed_urls = []
    if urls_text:
        seed_urls = parse_url_list_text(urls_text)
    elif urls_file and urls_file.filename:
        ext = os.path.splitext(urls_file.filename)[1].lower()
        if ext not in (".txt", ".csv"):
            return jsonify({"error": "Akceptowane są wyłącznie pliki .txt/.csv."}), 400
        try:
            raw_text = urls_file.read().decode("utf-8-sig", errors="replace")
        except Exception as e:
            return jsonify({"error": f"Nie udało się odczytać pliku: {str(e)}"}), 400
        seed_urls = parse_url_list_text(raw_text)

    if not seed_urls:
        return jsonify({
            "error": "Wklej listę adresów URL (po jednym w linijce) albo wgraj plik .txt/.csv z linkami. "
                     "Link do sitemap.xml zostanie automatycznie rozwinięty na wszystkie podstrony."
        }), 400

    if len(seed_urls) > MAX_URLS_PER_BATCH:
        return jsonify({
            "error": f"Zbyt wiele adresów ({len(seed_urls)}). Maksymalnie {MAX_URLS_PER_BATCH} w jednej partii."
        }), 400

    task_id = str(uuid.uuid4())
    job_dir = os.path.join(TMP_UPLOADS_DIR, task_id)
    os.makedirs(job_dir, exist_ok=True)

    with JOBS_LOCK:
        JOBS[task_id] = {
            "status": "parsing",
            "total": 0,
            "processed": 0,
            "success_count": 0,
            "error_count": 0,
            "results": [],
            "error_message": None,
            "created_at": time.time(),
            "image_cache": {},  # image_url -> already-generated result, deduped across pages in this batch
        }

    thread = threading.Thread(target=background_worker, args=(task_id, seed_urls, job_dir))
    thread.daemon = True
    thread.start()

    return jsonify({"task_id": task_id, "status": "parsing"})


@app.route('/status/<task_id>', methods=['GET'])
def get_status(task_id):
    clean_old_jobs()

    with JOBS_LOCK:
        job = JOBS.get(task_id)
        if not job:
            return jsonify({"error": "Zadanie nie istnieje lub wygasło."}), 404
        # image_cache is an internal dedup structure (can hold duplicate
        # base64 image blobs) - never worth shipping to the frontend on
        # every poll, it would just bloat the response and slow polling down.
        public_job = {k: v for k, v in job.items() if k != "image_cache"}

    return jsonify(public_job)


if __name__ == '__main__':
    app.run(debug=True, port=5000, threaded=True)
