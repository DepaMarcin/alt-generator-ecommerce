import os
import re
import sys
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
from html.parser import HTMLParser
from urllib.parse import urlparse, urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError
from PIL import Image
import requests
from curl_cffi import requests as curl_requests

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
MAX_URLS_PER_BATCH = 1000            # Safety cap on how many page URLs one batch can queue
MAX_IMAGES_PER_PAGE = 10             # Safety cap on how many images one page can queue (incl. the main image)
MAX_RAW_IMAGE_CANDIDATES = 150       # How many raw <img> tags to scan per page before junk-filtering/capping
PAGE_FETCH_TIMEOUT_SECONDS = 20
MAX_PAGE_FETCH_BYTES = 15 * 1024 * 1024   # a product page's HTML shouldn't exceed this
MAX_URL_REDIRECTS = 5

# Plain `requests` gets a plaintext 403 from a lot of e-commerce anti-bot
# protection (Cloudflare/Akamai-style TLS fingerprinting) - curl_cffi
# impersonates a real Chrome's TLS/HTTP fingerprint instead, paired with the
# matching set of headers a real Chrome navigation would send.
PAGE_FETCH_IMPERSONATE = "chrome120"
PAGE_FETCH_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "pl-PL,pl;q=0.9,en-US;q=0.8,en;q=0.7",
    "Sec-Ch-Ua": '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"Windows"',
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
}


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
    re-checked, same as the image downloader) and a byte-size cap. Uses
    curl_cffi with a Chrome TLS/HTTP fingerprint (impersonate) plus a
    matching realistic header set - plain `requests` gets flat-out 403'd by
    a lot of e-commerce anti-bot protection (Cloudflare/Akamai-style TLS
    fingerprinting) that this is specifically meant to get past. Returns
    (content_bytes, final_url, content_type)."""
    session = curl_requests.Session(impersonate=PAGE_FETCH_IMPERSONATE)
    current_url = url

    for _ in range(MAX_URL_REDIRECTS + 1):
        parsed = urlparse(current_url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ValueError("Nieprawidłowy URL (dozwolone są tylko linki http/https).")
        _check_hostname_is_public(parsed.hostname)

        try:
            response = session.get(
                current_url, stream=True, timeout=PAGE_FETCH_TIMEOUT_SECONDS,
                headers=PAGE_FETCH_HEADERS, allow_redirects=False,
            )
        except curl_requests.RequestsError as e:
            # Network/TLS-level failure (DNS, connection reset, timeout, ...) -
            # caught here so it surfaces as a plain, catchable ValueError like
            # every other fetch failure, instead of a raw curl exception.
            raise ValueError(f"Błąd sieciowy podczas pobierania strony: {e}") from e

        try:
            if response.is_redirect:
                location = response.headers.get("Location")
                if not location:
                    raise ValueError("Serwer zwrócił przekierowanie bez adresu docelowego.")
                current_url = urljoin(current_url, location)
                continue

            try:
                response.raise_for_status()
            except curl_requests.RequestsError as e:
                if response.status_code == 403:
                    raise ValueError(
                        "Serwer zablokował dostęp do strony (403 Forbidden) - sklep "
                        "prawdopodobnie stosuje zabezpieczenia anty-botowe."
                    ) from e
                raise ValueError(f"Błąd HTTP {response.status_code} podczas pobierania strony.") from e

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

# Their raw contents are never real page text - <style>/<script> hold
# CSS/JS source (a stray <style> block leaking CSS rules into the extracted
# description was the actual root cause of the Answear ".Icon_icon-v_XzHkY
# { content: ... }" bug), <noscript> holds fallback markup a real browser
# never renders, and <svg> holds vector markup/metadata.
_CONTENT_SKIP_TAGS = {"style", "script", "noscript", "svg"}

# Real product-description containers, ranked best-match first - checked
# against every open tag's itemprop/class/id so the body's actual
# description text can be captured instead of the marketing-blurb meta
# description ("Dobra cena", "Szybka wysyłka"...). Deliberately precise (no
# generic <article>/<main> fallback) - a broad fallback is exactly how
# cross-sell/recommendation widget text used to leak into the context.
DESCRIPTION_MAX_CHARS = 1000
# A description candidate (JSON-LD or HTML) shorter than this isn't treated
# as a full-fledged description - some Auchan pages return a stub like
# "Marka Pantene" from JSON-LD, or split the real description across
# several short HTML sections instead of one substantial block.
MIN_DESCRIPTION_LENGTH = 100
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

# Supplementary sections (skład/właściwości - "ingredients"/"properties")
# that often live in their own container completely separate from the main
# description - e.g. Auchan splits the general blurb, the attributes table,
# and the ingredients list into three different elements. Their text is
# combined with whichever main description wins the cascade above, rather
# than competing with it for a single "best rank" slot.
_SUPPLEMENTARY_ID_MARKERS = (
    "product-info", "product-attributes", "ingredients", "details", "specification",
)
_SUPPLEMENTARY_CLASS_MARKERS = (
    "product-info", "product-attributes", "ingredients", "details", "specification",
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


def _is_supplementary_description_container(tag: str, attrs_dict: dict) -> bool:
    """True for a "skład"/"właściwości" (ingredients/attributes) container -
    see _SUPPLEMENTARY_ID_MARKERS/_SUPPLEMENTARY_CLASS_MARKERS above."""
    elid = (attrs_dict.get("id") or "").strip().lower()
    if elid in _SUPPLEMENTARY_ID_MARKERS:
        return True

    classes = (attrs_dict.get("class") or "").lower().split()
    return any(marker in classes for marker in _SUPPLEMENTARY_CLASS_MARKERS)


_TRIVIAL_BRAND_ONLY_RE = re.compile(r'^\s*marka\s*:?\s+(?P<rest>[^.!?]+)\.?\s*$', re.IGNORECASE)
_TRIVIAL_BRAND_ONLY_MAX_WORDS = 4  # "Marka Pantene Pro-V Repair" - a real brand label, not a sentence


def _is_trivial_jsonld_description(text: str) -> bool:
    """True when a JSON-LD "description" is too thin to be useful on its
    own - some Auchan product pages return just a brand-name stub like
    "Marka Pantene" instead of the real product description. The cascade
    should fall through to the full HTML body description instead of
    accepting a stub like this as the final result.

    Checks word count (not just character count) for the "Marka X" pattern
    too, since a real sentence that happens to start with the word "Marka"
    (e.g. "Marka Informacje o składzie i zastosowaniu...") would otherwise
    be misjudged as trivial - a genuine brand label is only ever a few
    words long, regardless of MIN_DESCRIPTION_LENGTH."""
    if not text:
        return True
    stripped = text.strip()
    if len(stripped) < MIN_DESCRIPTION_LENGTH:
        return True
    match = _TRIVIAL_BRAND_ONLY_RE.match(stripped)
    return bool(match and len(match.group("rest").split()) <= _TRIVIAL_BRAND_ONLY_MAX_WORDS)


_HTML_TAG_RE = re.compile(r'<[^>]+>')


def clean_html_text(text: str) -> str:
    """Strips literal HTML markup out of a description that itself embeds
    ready-made HTML (a JSON-LD "description" field - or a mis-encoded body
    text node - containing raw <div style="...">/<br>/<p> instead of plain
    text, as seen on zakupy.auchan.pl) instead of plain text, decodes HTML
    entities (&nbsp;, &amp;, &quot;, ...), and collapses all whitespace/
    newlines down to single spaces."""
    if not text:
        return ""
    text = _HTML_TAG_RE.sub(' ', text)
    text = html.unescape(text)
    return re.sub(r'\s+', ' ', text).strip()


_CSS_RULE_BLOCK_RE = re.compile(r'(?<!\w)[.#][\w-]+(?:::?[\w-]+)?(?:\s*,\s*[.#][\w-]+(?:::?[\w-]+)?)*\s*\{[^{}]*\}')
_CSS_SELECTOR_TOKEN_RE = re.compile(r'(?<!\w)[.#][A-Za-z_][\w-]*(?:::?[A-Za-z-]+)?')


def _strip_css_artifacts(text: str) -> str:
    """Defense-in-depth cleanup for CSS/selector-looking text that leaks
    into an extracted description (a stray <style> block that slipped past
    _PageParser's <style>/<script>/<noscript>/<svg> skip, or malformed
    markup) - e.g. Answear's ".Icon_icon-v_XzHkY:before { content: ... }".
    Strips whole "selector { declarations }" rule blocks first, then any
    leftover bare .class/#id-looking tokens."""
    text = _CSS_RULE_BLOCK_RE.sub(' ', text)
    text = _CSS_SELECTOR_TOKEN_RE.sub(' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def _finalize_description_text(buffer: list) -> str:
    """Joins a captured description container's raw text fragments,
    dropping any individual fragment that looks like marketing/price noise
    leaked in from nearby widget chrome (a price tag, a 'Kup teraz' button)
    rather than discarding the whole description over one bad fragment,
    stripping any raw HTML markup/entities (see clean_html_text) and
    leftover CSS-artifact text (see _strip_css_artifacts)."""
    kept_fragments = []
    for fragment in buffer:
        collapsed = ' '.join(fragment.split())
        if not collapsed:
            continue
        lowered = collapsed.lower()
        if any(phrase in lowered for phrase in _DESCRIPTION_NOISE_PHRASES):
            continue
        kept_fragments.append(collapsed)
    return _strip_css_artifacts(clean_html_text(' '.join(kept_fragments)))


_SENTENCE_END_CHARS = (".", "!", "?")
_TRUNCATE_MIN_SENTENCE_LENGTH = 100  # don't cut off after a suspiciously short "sentence"


def _truncate_to_sentence(text: str, max_chars: int = DESCRIPTION_MAX_CHARS) -> str:
    """Trims text to at most max_chars without cutting a word - or ideally a
    sentence - in half. Prefers to end at the last full sentence within the
    limit (so the description reads as a complete thought); if no
    sentence-ending punctuation shows up early enough to be useful, falls
    back to the last whole word instead of chopping one in the middle."""
    if len(text) <= max_chars:
        return text

    truncated = text[:max_chars]

    last_sentence_end = max(truncated.rfind(ch) for ch in _SENTENCE_END_CHARS)
    if last_sentence_end >= _TRUNCATE_MIN_SENTENCE_LENGTH:
        return truncated[:last_sentence_end + 1]

    last_space = truncated.rfind(" ")
    if last_space > 0:
        return truncated[:last_space]

    return truncated


def _is_svg_url(url: str) -> bool:
    return urlparse(url).path.lower().endswith(".svg")


# Keywords that show up in the URL of UI chrome/branding rather than an
# actual product photo: logos, payment/delivery/social/courier icons, trust
# badges, star ratings, avatars, buttons/arrows, placeholders.
_JUNK_URL_KEYWORDS = (
    "logo", "icon", "ikona", "banner", "badge", "payment", "platn", "delivery",
    "dostaw", "inpost", "dpd", "courier", "visa", "mastercard", "blik", "social",
    "facebook", "instagram", "footer", "header", "certyfikat", "trust", "star",
    "rating", "avatar", "button", "arrow", "placeholder",
    "share", "logo_share", "qr", "code", "og-", "og_", "ans.png",
    "jakkupowac", "piggy", "nowosc", "marki-", "b2b", "okazje", "zgarnij", "promocj",
    "kategoria-", "kategoria", "sezonowe", "partnerzy",
)
JUNK_IMAGE_MIN_DIMENSION_PX = 80


def _parse_pixel_size(value):
    """Parses a leading integer out of an HTML width/height attribute value
    (e.g. "80", "80px") - returns None for anything else (percentages,
    "auto", missing), since those aren't a reliable pixel size signal."""
    if not value:
        return None
    match = re.match(r'\s*(\d+)', str(value))
    return int(match.group(1)) if match else None


def _is_junk_image(url: str, attrs_dict: dict = None) -> bool:
    """True for images that are almost certainly UI chrome/branding rather
    than a real product photo - identified by a keyword in the URL (logos,
    share/QR-code icons, social/payment/delivery badges, watermark files
    like Answear's "ans.png", ...), or by an explicit width/height
    attribute under JUNK_IMAGE_MIN_DIMENSION_PX (an icon-sized <img>).
    Applied to EVERY image candidate, including the main-image cascade - a
    share icon or QR code must never be crowned "main image" just because
    it happened to be the first og:image/gallery <img> on the page."""
    lowered = (url or "").lower()
    if any(keyword in lowered for keyword in _JUNK_URL_KEYWORDS):
        return True

    attrs_dict = attrs_dict or {}
    for attr in ("width", "height"):
        size = _parse_pixel_size(attrs_dict.get(attr))
        if size is not None and size < JUNK_IMAGE_MIN_DIMENSION_PX:
            return True

    return False


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


def _extract_jsonld_product_images(ld_json_blocks: list) -> list:
    """Priority 2 of the main-image cascade: every image URL in the first
    Product node's "image" field - a single string, a list of strings, or a
    list/single ImageObject dict. JSON-LD often carries the site's *entire*
    product gallery here, not just one photo, so the first entry feeds the
    main-image cascade and the rest join the "other images" pool."""
    for item in _find_jsonld_product_nodes(ld_json_blocks):
        image = item.get("image")
        if image is None:
            continue

        candidates = image if isinstance(image, list) else [image]
        urls = []
        for candidate in candidates:
            if isinstance(candidate, dict):
                candidate = candidate.get("url") or candidate.get("@id")
            if isinstance(candidate, str) and candidate.strip():
                urls.append(candidate.strip())

        if urls:
            return urls

    return []


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


# ---------------------------------------------------------------------------
# Last-resort description fallbacks - for pages like zakupy.auchan.pl that
# don't use any of the standard class/id/itemprop selectors at all, so the
# normal cascade above comes up with nothing (or something too short) and
# the context would otherwise be left with just "Produkt: <name>". Only
# ever invoked when everything above still hasn't reached
# MIN_DESCRIPTION_LENGTH - building the little DOM tree below is extra work
# that the common case (a page that DOES use recognizable selectors)
# never needs to pay for.
# ---------------------------------------------------------------------------

class _DomNode:
    """Minimal DOM element: just enough (tag/attrs/parent/children) to walk
    up to a parent, across to following siblings, or down through
    descendants - none of which a single-pass streaming parser can express
    once it's already moved on. Not used for the main extraction cascade,
    only for the fallback strategies below."""
    __slots__ = ("tag", "attrs", "parent", "children")

    def __init__(self, tag, attrs, parent=None):
        self.tag = tag
        self.attrs = attrs
        self.parent = parent
        self.children = []  # list of _DomNode | str

    def own_text(self) -> str:
        """Only this node's direct text, not its descendants' - so a huge
        wrapper <div> doesn't look like it "contains" a giant paragraph."""
        return ''.join(c for c in self.children if isinstance(c, str))

    def full_text(self) -> str:
        """This node's entire text content, descendants included (skipping
        <style>/<script>/<noscript>/<svg> subtrees, same as the main
        parser)."""
        parts = []
        for child in self.children:
            if isinstance(child, str):
                parts.append(child)
            elif child.tag not in _CONTENT_SKIP_TAGS:
                parts.append(child.full_text())
        return ''.join(parts)


class _SimpleDomBuilder(HTMLParser):
    """Builds a _DomNode tree of the whole document. Deliberately tolerant
    of malformed markup: handle_endtag pops back to the nearest matching
    open tag (if any) instead of assuming well-formed nesting."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = _DomNode("[root]", {})
        self._stack = [self.root]

    def handle_starttag(self, tag, attrs):
        node = _DomNode(tag, dict(attrs), parent=self._stack[-1])
        self._stack[-1].children.append(node)
        if tag not in _VOID_TAGS:
            self._stack.append(node)

    def handle_endtag(self, tag):
        for i in range(len(self._stack) - 1, 0, -1):
            if self._stack[i].tag == tag:
                del self._stack[i:]
                break

    def handle_data(self, data):
        self._stack[-1].children.append(data)


def _build_simple_dom(html_text: str) -> _DomNode:
    builder = _SimpleDomBuilder()
    try:
        builder.feed(html_text)
    except Exception:
        pass  # tolerate malformed markup - keep whatever was parsed so far
    return builder.root


def _iter_dom_nodes(node):
    """Depth-first walk over every element node in the tree, in document
    order."""
    for child in node.children:
        if not isinstance(child, str):
            yield child
            yield from _iter_dom_nodes(child)


def _has_excluded_ancestor(node: _DomNode) -> bool:
    current = node.parent
    while current is not None:
        if _is_excluded_description_container(current.tag, current.attrs):
            return True
        current = current.parent
    return False


def _following_siblings(node: _DomNode) -> list:
    if node.parent is None:
        return []
    siblings = node.parent.children
    try:
        idx = siblings.index(node)
    except ValueError:
        return []
    return siblings[idx + 1:]


# 1. Heading-based extraction: a page that skips every standard selector
# often still visually labels its description with a heading (or a <div>
# styled to look like one).
_HEADING_TAGS = {"h1", "h2", "h3", "h4", "strong", "b", "div"}
_SECTION_BOUNDARY_TAGS = {"h1", "h2", "h3", "h4"}
_HEADING_PHRASES = (
    "opis produktu", "informacje o produkcie", "składniki", "o produkcie",
    "szczegóły", "składniki, alergeny",
)


def _parent_has_other_heading_siblings(node: _DomNode) -> bool:
    """True if node's parent holds more than one heading-level section -
    in which case the parent is a broad multi-section container (e.g.
    <body> itself), not a tight per-section wrapper, and its full text
    would pull in unrelated sections too."""
    if node.parent is None:
        return False
    return any(
        isinstance(child, _DomNode) and child is not node and child.tag in _SECTION_BOUNDARY_TAGS
        for child in node.parent.children
    )


def _find_heading_based_description(root: _DomNode) -> str:
    for node in _iter_dom_nodes(root):
        if node.tag not in _HEADING_TAGS or _has_excluded_ancestor(node):
            continue

        # A bare <div> is only trusted as a "heading" via its OWN text - a
        # huge wrapper <div> would otherwise "match" just because the
        # phrase appears somewhere, anywhere, in its entire subtree.
        heading_text = clean_html_text(
            node.own_text() if node.tag == "div" else node.full_text()
        ).lower()
        if not heading_text or not any(phrase in heading_text for phrase in _HEADING_PHRASES):
            continue

        # Try the parent container's full text first (covers the common
        # <div><h3>Opis produktu</h3><p>...</p></div> shape) - but only if
        # the parent looks like a tight, single-section wrapper. A parent
        # holding several heading-level sections (e.g. <body> itself) would
        # pull unrelated sections into the text too, so that case falls
        # through to the sibling-walk below instead.
        if node.parent is not None and not _parent_has_other_heading_siblings(node):
            parent_text = clean_html_text(node.parent.full_text())
            if len(parent_text) >= MIN_DESCRIPTION_LENGTH:
                return parent_text

        # Otherwise, walk the heading's own following siblings and combine
        # their text until long enough, stopping at the next section.
        combined = ""
        for sibling in _following_siblings(node):
            if isinstance(sibling, str):
                chunk = clean_html_text(sibling)
            else:
                if sibling.tag in _SECTION_BOUNDARY_TAGS:
                    break
                chunk = clean_html_text(sibling.full_text())
            if not chunk:
                continue
            combined = f"{combined} {chunk}".strip() if combined else chunk
            if len(combined) >= MIN_DESCRIPTION_LENGTH:
                break
        if combined:
            return combined

    return ""


# 2. App-state JSON scripts (Next.js/Nuxt/etc. embed the whole page's data
# as JSON instead of - or in addition to - schema.org markup).
_JSON_STATE_SCRIPT_ID_MARKERS = ("__next_data__", "__initial_state__")
_JSON_STATE_DESCRIPTION_KEYS = (
    "description", "longdescription", "ingredients", "attributes", "productdetails",
)
_JSON_SCAN_MAX_DEPTH = 12
_JSON_SCAN_MAX_NODES = 20000


def _search_json_for_description(data) -> str:
    """Recursively scans a parsed JSON value for the longest string found
    under a description-like key, bounded so a huge state blob can't make
    this pathologically slow."""
    best = ""
    visited = [0]

    def walk(value, depth):
        nonlocal best
        if visited[0] >= _JSON_SCAN_MAX_NODES or depth > _JSON_SCAN_MAX_DEPTH:
            return
        visited[0] += 1

        if isinstance(value, dict):
            for key, val in value.items():
                if str(key).lower() in _JSON_STATE_DESCRIPTION_KEYS and isinstance(val, str):
                    if len(val) > len(best):
                        best = val
                walk(val, depth + 1)
        elif isinstance(value, list):
            for item in value:
                walk(item, depth + 1)

    walk(data, 0)
    return best


def _find_json_state_description(root: _DomNode) -> str:
    for node in _iter_dom_nodes(root):
        if node.tag != "script":
            continue
        script_type = (node.attrs.get("type") or "").strip().lower()
        script_id = (node.attrs.get("id") or "").strip().lower()
        if script_type != "application/json" and script_id not in _JSON_STATE_SCRIPT_ID_MARKERS:
            continue

        raw = node.own_text().strip()
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except (ValueError, TypeError):
            continue

        found = _search_json_for_description(data)
        if found:
            cleaned = clean_html_text(found)
            if len(cleaned) >= MIN_DESCRIPTION_LENGTH:
                return cleaned

    return ""


# 3. Absolute last resort: the single longest <p>/<div> text block on the
# page, outside nav/header/footer/cross-sell-style containers.
def _find_longest_paragraph_description(root: _DomNode) -> str:
    best = ""
    for node in _iter_dom_nodes(root):
        if node.tag not in ("p", "div") or _has_excluded_ancestor(node):
            continue
        text = clean_html_text(node.own_text())
        if len(text) > len(best):
            best = text
    return best


def find_fallback_html_description(html_text: str) -> str:
    """Last-resort description extraction, tried only when the standard
    class/id/itemprop cascade still comes up short of MIN_DESCRIPTION_LENGTH.
    Three escalating strategies, tried in order until one clears the
    threshold: (1) a heading whose text matches a known "this is the
    description" phrase, (2) any __NEXT_DATA__/__INITIAL_STATE__/
    application/json <script> scanned for description-like keys, (3) the
    single longest paragraph-like text block on the page. Returns the best
    candidate found across all three even if it's still short - never
    raises."""
    root = _build_simple_dom(html_text)

    best = ""
    for finder in (
        _find_heading_based_description,
        _find_json_state_description,
        _find_longest_paragraph_description,
    ):
        try:
            candidate = finder(root)
        except Exception:
            candidate = ""
        if candidate and len(candidate) > len(best):
            best = candidate
        if len(best) >= MIN_DESCRIPTION_LENGTH:
            break

    return best


class _PageParser(HTMLParser):
    """Single pass over the page: captures <title>/<h1>/og:title (title
    context), the actual body description text (from every matching
    product-description container, not just the first one found - never
    the SEO meta description, and never CSS/JS source text leaking in from
    a nested <style>/<script>/<noscript>/<svg>), separately-scraped
    supplementary text (skład/właściwości - "ingredients"/"attributes"
    sections that often live outside the main description container), the
    main-image cascade signals (og:image/twitter:image, JSON-LD
    Product.image, fetchpriority/eager/gallery <img> hints), AND every
    <img> tag actually rendered on the page (so nothing gets missed)."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.title = ""
        self.og_title = ""
        self.h1 = ""
        self.meta_image_candidates = []
        self.img_urls = []
        self.img_attrs_by_url = {}  # raw <img> url -> its attrs dict (for the junk-image size check)
        self.priority_img_urls = []
        self.ld_json_blocks = []
        self.description_candidates = {}  # rank -> list of captured texts for that rank
        self.supplementary_texts = []  # text from every product-info/product-attributes/ingredients container
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
        self._supplementary_stack = []  # {"buffer": list} per open non-void ancestor (ingredients/attributes)
        self._excluded_stack = []  # bool per open non-void ancestor: "cross-sell/nav/widget etc."
        self._skip_stack = []  # bool per open non-void ancestor: "style/script/noscript/svg"

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
        is_supplementary_frame = (
            not currently_excluded and _is_supplementary_description_container(tag, attrs_dict)
        )
        is_skip_frame = tag in _CONTENT_SKIP_TAGS

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

            if chosen and chosen not in self.img_attrs_by_url:
                self.img_attrs_by_url[chosen] = attrs_dict

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
            self._supplementary_stack.append({"buffer": []} if is_supplementary_frame else None)
            self._excluded_stack.append(is_excluded_frame)
            self._skip_stack.append(is_skip_frame)

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
            if self._skip_stack:
                self._skip_stack.pop()
            if self._description_stack:
                frame = self._description_stack.pop()
                if frame is not None:
                    text = _finalize_description_text(frame["buffer"])
                    if text:
                        self.description_candidates.setdefault(frame["rank"], []).append(text)
            if self._supplementary_stack:
                supp_frame = self._supplementary_stack.pop()
                if supp_frame is not None:
                    text = _finalize_description_text(supp_frame["buffer"])
                    if text:
                        self.supplementary_texts.append(text)

    def handle_data(self, data):
        # <script type="application/ld+json"> content still needs capturing
        # even though <script> is itself a skip-tag for every other buffer -
        # it's structured data, not page text.
        if self._in_ldjson:
            self._ldjson_buffer += data

        if any(self._skip_stack):
            return  # inside <style>/<script>/<noscript>/<svg> - never real page text

        if self._in_title:
            self.title += data
        if self._in_h1:
            self._h1_buffer.append(data)
        for frame in self._description_stack:
            if frame is not None:
                frame["buffer"].append(data)
        for supp_frame in self._supplementary_stack:
            if supp_frame is not None:
                supp_frame["buffer"].append(data)


def extract_page_content(html_text: str, page_url: str) -> dict:
    """Pulls the page context (H1/og:title + a description that belongs to
    THIS product only) plus every image on the page. The title prefers the
    visible <h1>, falling back to og:title then <title>.

    The description follows a cascade, strongest/safest signal first:
    (1) schema.org Product.description from JSON-LD - authored for exactly
    this product, so it can never leak cross-sell/recommendation text -
    UNLESS it's shorter than MIN_DESCRIPTION_LENGTH, or just a bare
    "Marka X" brand label (some Auchan pages return only that), in which
    case it's treated as absent so the cascade falls through to (2);
    (2) failing that, the single best-ranked description container in the
    body (itemprop="description", #description, #product-description,
    .product-description, .tab-content, #tab-description,
    .description-content - all siblings at that one rank combined, but
    NOT lower-ranked alternatives too, since itemprop/#description/
    .product-description etc. are competing signals for the same thing,
    not complementary sections) - with cross-sell/related/recommended/
    carousel/widget/sidebar/header/footer/nav containers explicitly excluded
    from the scan, and marketing/price fragments ("zł", "Kup teraz", ...)
    dropped from whatever text is captured. NOT the SEO meta description,
    which is usually marketing filler rather than real product knowledge.

    Whichever description wins (1) or (2) - if it's still under
    MIN_DESCRIPTION_LENGTH, it's extended with supplementary sections
    (#/.product-info, #/.product-attributes, #/.ingredients, #/.details,
    #/.specification - these complement rather than compete with the main
    description, so combining them in is always safe), in the order found,
    until the combined text clears the threshold or they run out.

    If it's STILL short (a page that doesn't use any recognizable selector
    at all, e.g. zakupy.auchan.pl), find_fallback_html_description() is
    tried as a last resort: a heading-labeled section, an app-state JSON
    <script> (__NEXT_DATA__ and friends), then the single longest
    paragraph-like block on the page - see that function for details.

    Every image candidate - main and "other" alike - is ranked into one
    pool, strongest signal first: (1) og:image/twitter:image meta tags,
    (2) JSON-LD Product.image (often the site's full product gallery),
    (3) an <img> hinted as high-priority/eager or sitting inside a gallery
    container, (4) every other <img> on the page. Every candidate URL is
    sanitized/decoded/resolved to an absolute http(s) URL before use, and
    SVGs are always excluded.

    The WHOLE pool is then run through _is_junk_image() - logos, share/QR
    icons, payment/delivery/social/courier badges, ratings, avatars,
    buttons/arrows, placeholders, and any <img> with an explicit
    width/height under JUNK_IMAGE_MIN_DIMENSION_PX are dropped, since those
    are UI chrome, not product photos. main_url is the first candidate left
    standing - so a share icon or QR code that happened to be the page's
    og:image/first gallery <img> gets skipped in favor of the next real
    product photo instead of being crowned "main image". Everything else
    remaining becomes both the main-photo download-retry fallbacks and the
    "other images" pool, capped so main_url + other_urls together never
    exceed MAX_IMAGES_PER_PAGE."""
    parser = _PageParser()
    try:
        parser.feed(html_text)
    except Exception:
        pass  # tolerate malformed markup - keep whatever was parsed so far

    title = re.sub(r'\s+', ' ', (parser.h1 or parser.og_title or parser.title or "").strip())

    # Priority 1: schema.org Product.description from JSON-LD - unless it's
    # just a thin brand-name stub, in which case treat it as absent.
    jsonld_description = _extract_jsonld_product_description(parser.ld_json_blocks)
    description = ""
    if jsonld_description:
        clean_jsonld_description = _strip_css_artifacts(clean_html_text(jsonld_description))
        if not _is_trivial_jsonld_description(clean_jsonld_description):
            description = clean_jsonld_description

    if not description:
        # Priority 2: JSON-LD was absent or too thin - use the best
        # matching body description container instead (all its siblings at
        # that rank combined). Deliberately does NOT fall back to a lower-
        # ranked description container too (itemprop vs #description vs
        # .product-description are competing alternatives for the same
        # thing, not complementary sections) - only supplementary sections
        # get combined in below.
        if parser.description_candidates:
            best_rank = min(parser.description_candidates)
            description = ' '.join(parser.description_candidates[best_rank]).strip()

    # Whichever description won (JSON-LD or body) - if it's still short of
    # MIN_DESCRIPTION_LENGTH, extend it with supplementary sections
    # (ingredients/attributes/details/specification), in the order they
    # were found, until it clears the threshold or they run out. These
    # complement rather than compete with the main description, so they're
    # always fair game to combine in, regardless of which priority won.
    if len(description) < MIN_DESCRIPTION_LENGTH:
        for chunk in parser.supplementary_texts:
            if not chunk:
                continue
            description = f"{description} {chunk}".strip() if description else chunk
            if len(description) >= MIN_DESCRIPTION_LENGTH:
                break

    # Still short (or empty) - the page likely doesn't use any of the
    # standard selectors at all (e.g. zakupy.auchan.pl). Try the heading/
    # JSON-state/longest-paragraph fallbacks before giving up.
    if len(description) < MIN_DESCRIPTION_LENGTH:
        fallback_description = find_fallback_html_description(html_text)
        if len(fallback_description) > len(description):
            description = fallback_description

    description = _truncate_to_sentence(description) if description else ""

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

    # Priority 2: JSON-LD structured data - schema.org Product.image is often
    # the site's *entire* product gallery, not just one photo. The first
    # entry feeds the main-image cascade; the rest join the "other" pool.
    jsonld_gallery_raw = _extract_jsonld_product_images(parser.ld_json_blocks)
    jsonld_candidates = add_candidates(jsonld_gallery_raw[:1])
    jsonld_gallery_extra = add_candidates(jsonld_gallery_raw[1:])

    # Priority 3: a dedicated/eager-loaded <img> or one inside a gallery container.
    priority_candidates = add_candidates(parser.priority_img_urls)

    # Everything else on the page (cap how many raw <img> tags we even
    # bother sanitizing - well above MAX_IMAGES_PER_PAGE since most will be
    # junk-filtered out below).
    body_images = add_candidates(parser.img_urls[:MAX_RAW_IMAGE_CANDIDATES])

    # clean (sanitized/absolute) image URL -> its original <img> attrs, so
    # the junk filter can check width/height even after the URL itself has
    # been rewritten (relative -> absolute, unicode-decoded, ...).
    clean_url_attrs = {}
    for raw_url in parser.img_urls[:MAX_RAW_IMAGE_CANDIDATES]:
        clean = sanitize_image_url(raw_url, page_url)
        if clean and clean not in clean_url_attrs:
            clean_url_attrs[clean] = parser.img_attrs_by_url.get(raw_url) or {}

    # Full ranked candidate pool for the main photo, strongest signal first:
    # og:image/twitter:image, then the rest of the JSON-LD gallery, then
    # priority/gallery <img>, then everything else on the page. add_candidates
    # dedupes globally, so this pool never repeats a URL across tiers.
    ranked_main_candidates = meta_candidates + jsonld_candidates + priority_candidates
    full_candidate_pool = ranked_main_candidates + jsonld_gallery_extra + body_images

    # The main image must ALSO pass the junk filter - a share icon, QR code,
    # or logo must never be crowned "main image" just because it happened to
    # be the first og:image/gallery <img> on the page. If the top-ranked
    # candidate is junk, this naturally promotes the next clean one.
    clean_candidates = [
        u for u in full_candidate_pool if not _is_junk_image(u, clean_url_attrs.get(u))
    ]
    main_url = clean_candidates[0] if clean_candidates else None

    # Every other clean candidate doubles as a download-retry fallback for
    # the main photo and as the source for "other images", capped so
    # main_url + other_urls never exceeds MAX_IMAGES_PER_PAGE.
    remaining_clean = [u for u in clean_candidates if u != main_url]
    main_fallbacks = remaining_clean
    other_budget = MAX_IMAGES_PER_PAGE - (1 if main_url else 0)
    other_urls = remaining_clean[:other_budget]

    return {
        "context": context,
        "title": title,
        "main_url": main_url,
        "main_fallbacks": main_fallbacks,
        "other_urls": other_urls,
    }


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


# RateLimitError codes that mean "no budget left on the account/project" -
# unlike a transient burst-of-requests 429, waiting and retrying can never
# fix these, so it's pointless (and slow) to burn through max_retries
# attempts and their backoff waits.
_NON_RETRYABLE_RATE_LIMIT_MARKERS = ("insufficient_quota", "project_spend_limit_exceeded")


def _is_quota_exhausted_error(rate_limit_error: RateLimitError) -> bool:
    parts = [str(rate_limit_error)]
    for attr in ("code", "body"):
        value = getattr(rate_limit_error, attr, None)
        if value:
            parts.append(str(value))
    haystack = " ".join(parts).lower()
    return any(marker in haystack for marker in _NON_RETRYABLE_RATE_LIMIT_MARKERS)


class QuotaExhaustedError(Exception):
    """Raised instead of a generic RuntimeError when OpenAI reports the
    account/project is out of budget (insufficient_quota /
    project_spend_limit_exceeded). Deliberately NOT caught per-image -
    _process_image_safe lets it propagate so the whole batch stops instead
    of recording it as just another failed image."""
    pass


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
            if _is_quota_exhausted_error(e):
                # No budget left on the account/project - retrying/waiting
                # can't fix that, so abort immediately with a dedicated,
                # user-facing exception instead of sitting through up to 8
                # useless attempts and surfacing a generic API error.
                raise QuotaExhaustedError(
                    "Wyczerpano limit środków lub zapytań API. Doładuj saldo u dostawcy usługi, aby kontynuować."
                ) from e
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
        # A cache hit does zero real work - report that honestly instead of
        # replaying whatever timings were measured when this URL was first
        # processed (possibly on a completely different page).
        cached["_perf"] = {"download": 0.0, "compress": 0.0, "openai": 0.0}
        return cached

    # og:image/twitter:image variants can point at the same photo under
    # different (hash-based) cache URLs - one of them 404-ing shouldn't sink
    # the whole image, so try each candidate in turn.
    candidate_urls = [image_url] + [u for u in (fallback_urls or []) if u != image_url]
    local_path = None
    used_url = image_url
    last_error = None
    download_start = time.perf_counter()
    for candidate_url in candidate_urls:
        try:
            local_path = download_image_from_url(candidate_url, job_dir)
            used_url = candidate_url
            break
        except Exception as e:
            last_error = e
    download_elapsed = time.perf_counter() - download_start
    if local_path is None:
        error = last_error if last_error is not None else RuntimeError("Nie udało się pobrać obrazu.")
        # Remember the (failed) download time before raising, so
        # _process_image_safe can still report an honest [PERF] duration
        # instead of defaulting to 0.0s for a request that clearly wasn't free.
        error._perf = {"download": download_elapsed, "compress": 0.0, "openai": 0.0}
        raise error

    compress_start = time.perf_counter()
    compressed_path = compress_image(local_path)
    compress_elapsed = time.perf_counter() - compress_start

    openai_start = time.perf_counter()
    try:
        alt_text = generate_alt_via_openai(compressed_path, context)
    except Exception as e:
        e._perf = {
            "download": download_elapsed,
            "compress": compress_elapsed,
            "openai": time.perf_counter() - openai_start,
        }
        raise
    openai_elapsed = time.perf_counter() - openai_start

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
        # Internal profiling data - process_page_url reads and strips this
        # before the result ever reaches JOBS["results"] / the frontend.
        "_perf": {"download": download_elapsed, "compress": compress_elapsed, "openai": openai_elapsed},
    }
    # Only successful results are cached - a transient download/API failure
    # shouldn't poison every later page that happens to share the URL.
    _store_cached_image_result(task_id, image_url, result)
    return result


def _process_image_safe(image_url: str, context: str, job_dir: str, fallback_urls: list = None,
                         task_id: str = None) -> dict:
    """process_single_image, but never raises - a failure on one image (main
    or one of the "others") shouldn't take down the rest of the page.

    QuotaExhaustedError is the one exception deliberately let through
    uncaught: it doesn't mean "this image failed", it means "the whole batch
    needs to stop right now" - worker_task handles it at the page level, so
    it must never be recorded here as a per-image error entry."""
    try:
        return process_single_image(image_url, context, job_dir, fallback_urls=fallback_urls, task_id=task_id)
    except QuotaExhaustedError:
        raise
    except Exception as e:
        return {
            "image_url": image_url,
            "context": context,
            "alt": f"Błąd przetwarzania: {str(e)}",
            "skipped": False,
            "skip_reason": None,
            "image_data": "",
            # process_single_image attaches _perf to the exception before
            # raising, so a failed image still contributes real timing to
            # the page's [PERF] summary instead of silently reading 0.0s.
            "_perf": getattr(e, "_perf", None) or {"download": 0.0, "compress": 0.0, "openai": 0.0},
        }


def _log_page_perf(page_url: str, page_fetch_s: float, download_s: float,
                    compress_s: float, openai_s: float) -> None:
    total_s = page_fetch_s + download_s + compress_s + openai_s
    # This runs inside background worker threads, so it never reaches the
    # request/response cycle Flask's dev server auto-flushes for you - print
    # straight to sys.stdout (same stream as Flask's HTTP request logs) with
    # flush=True, otherwise Windows line-buffers stdout when it's not an
    # interactive console and the [PERF] lines only show up in bursts (or
    # not at all before the process exits).
    print(
        f"[PERF] {page_url} | Str: {page_fetch_s:.1f}s | Pobranie obr: {download_s:.1f}s | "
        f"Kompresja: {compress_s:.1f}s | OpenAI: {openai_s:.1f}s | SUMA: {total_s:.1f}s",
        file=sys.stdout, flush=True,
    )


def process_page_url(page_url: str, job_dir: str, task_id: str = None) -> dict:
    """One full unit of work: fetch a product page, pull every image on it
    (main + the rest) from its markup, and generate an ALT for each one.
    The main image is processed first, then the remaining images are
    processed concurrently (up to PAGE_IMAGE_WORKERS at a time) - each
    candidate is checked against the job's shared image_cache first, so an
    asset that repeats across pages is only ever downloaded/analyzed once.

    Prints a [PERF] summary line per page (page fetch + the images' summed
    download/compression/OpenAI time) to make slow pages/steps visible."""
    page_fetch_start = time.perf_counter()
    html_text, final_url = fetch_page_html(page_url)
    page_fetch_elapsed = time.perf_counter() - page_fetch_start

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
        _log_page_perf(page_url, page_fetch_elapsed, 0.0, 0.0, 0.0)
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

    # Sum each image's measured download/compression/OpenAI time for the
    # page-level summary, then strip the internal _perf field - it's never
    # meant to reach JOBS["results"] / the frontend.
    download_total = compress_total = openai_total = 0.0
    for image_result in [result["main_image"]] + result["other_images"]:
        perf = image_result.pop("_perf", None)
        if perf:
            download_total += perf.get("download", 0.0)
            compress_total += perf.get("compress", 0.0)
            openai_total += perf.get("openai", 0.0)

    _log_page_perf(page_url, page_fetch_elapsed, download_total, compress_total, openai_total)

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

        final_urls = seed_urls

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

            try:
                res = process_page_url(page_url, job_dir, task_id=task_id)
            except QuotaExhaustedError:
                # Out of API budget - stop the whole batch right now rather
                # than recording this as just another failed page.
                stop_event.set()
                with JOBS_LOCK:
                    if task_id in JOBS:
                        JOBS[task_id]["status"] = "stopped_error"
                        JOBS[task_id]["error_message"] = (
                            "Przetwarzanie przerwane: Wyczerpano limit środków lub zapytań API. "
                            "Doładuj saldo u dostawcy usługi, aby kontynuować."
                        )
                return
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
            else:
                is_error = False
                error_detail = None

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
            "error": "Wklej listę adresów URL podstron produktowych (po jednym w linijce) "
                     "albo wgraj plik .txt/.csv z linkami."
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
