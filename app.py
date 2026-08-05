import os
import re
import io
import csv
import random
import uuid
import ipaddress
import socket
import threading
import time
import mimetypes
import base64
import shutil
import requests
from urllib.parse import urlparse, urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError
from PIL import Image
from lxml import etree

load_dotenv()

app = Flask(__name__)

# Upload limits - HTML exports can be large (up to ~3 GB), but Werkzeug always
# spools multipart file parts to a disk-backed temp file, so RAM usage stays low
# regardless of this cap; it only bounds total request size.
MAX_CONTENT_LENGTH_MB = 3072
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH_MB * 1024 * 1024
JOB_MAX_AGE_SECONDS = 3600      # Job retention time in memory (1h)
MAX_WORKERS = 5                  # Number of parallel connections to OpenAI

# In-memory job store + thread lock
JOBS = {}
JOBS_LOCK = threading.Lock()

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
TMP_UPLOADS_DIR = os.path.join(PROJECT_DIR, "tmp_uploads")
os.makedirs(TMP_UPLOADS_DIR, exist_ok=True)

CONSECUTIVE_ERROR_LIMIT = 5              # Isolated transient errors shouldn't abort the whole batch
MAX_IMAGE_TASKS_PER_BATCH = 500          # Safety cap on how many images one HTML file can queue for OpenAI
MIN_IMAGE_DIMENSION = 150                # px - images smaller than this on either side are treated as icons
CONTEXT_MAX_CHARS = 500                  # Cap on the context string sent to OpenAI (controls token usage)


# ---------------------------------------------------------------------------
# SEM helper regexes
# ---------------------------------------------------------------------------

PLACEHOLDER_IMAGE_RE = re.compile(
    r'(brak[-_]?zdj[ea]ci?a|placeholder|coming[-_]?soon|wkr[oó]tce|no[-_]?image|noimage|'
    r'no[-_]?photo|blank|niedostepn[ae]|default[-_]?(image|product)|zastepcze)',
    re.IGNORECASE,
)

PROMO_SPAM_RE = re.compile(
    r'(darmowa\s+wysy[łl]ka|free\s+shipping|dostawa\s+gratis|rabat\s*-?\s*\d{1,3}\s*%|'
    r'-\s?\d{1,3}\s?%|promocja!*|super\s*cena|najni[żz]sza\s*cena(?:\s*w\s*histori[i]?)?|'
    r'\bgratis\b|wyprzeda[żz]|okazja!*|\bhit!*\b|nowo[śs][ćc]!*|bestseller!*)',
    re.IGNORECASE,
)

CERT_RE = re.compile(
    r'\b(EN\s?\d{3,5}(?:[-:]\d+)?|ECE\s?R\d{2,3}(?:\.\d+)?|OEKO-?TEX(?:\s?Standard\s?100)?|ISO\s?\d{4,5})\b',
    re.IGNORECASE,
)

DIMENSION_WEIGHT_RE = re.compile(
    r'\b(\d+(?:[.,]\d+)?\s?(?:kg|g|cm|mm|m|l|ml))\b'
    r'|\b(\d{1,2}\s?-\s?\d{1,2}\s?lat|od\s?\d+\s?(?:kg|lat|miesi[ęe]cy)|\d{1,2}\s?m(?:ies)?\.?\s?-\s?\d{1,2}\s?lat)\b',
    re.IGNORECASE,
)

PRODUCT_CONTAINER_RE = re.compile(
    r'(product[-_]?(item|card|box|tile|listing|miniature|preview|grid-item)'
    r'|produkt[-_]?(box|kafelek|karta|item|miniaturka)'
    r'|offer[-_]?item|listing[-_]?item)',
    re.IGNORECASE,
)
BREADCRUMB_RE = re.compile(r'breadcrumb|okruszk', re.IGNORECASE)
BRAND_RE = re.compile(r'\b(brand|producer|manufacturer|marka|producent)\b', re.IGNORECASE)
TITLE_RE = re.compile(
    r'(product[-_]?(name|title)|nazwa[-_]?produktu|product-info-name)',
    re.IGNORECASE,
)

# "Related/upsell/cross-sell" widgets (e.g. "Może Ci się spodobać") on real
# product pages reuse product-card-like markup for their recommendation tiles -
# those must never be mistaken for the products actually being alt-tagged.
EXCLUDED_CONTAINER_RE = re.compile(
    r'(related|cross-?sell|up-?sell|recommend|carousel|swiper|slider|'
    r'you-?may-?also-?like|similar[-_]?products?|polecane|podobne)',
    re.IGNORECASE,
)

IMG_SRC_ATTR_PRIORITY = ("data-lazy-src", "data-src", "data-original", "src")


def is_placeholder_image(url: str) -> bool:
    """Detects "no photo yet" placeholder images so they're skipped instead of
    being sent to the AI (saves API calls and avoids nonsense ALT text)."""
    return bool(PLACEHOLDER_IMAGE_RE.search(url))


def clean_product_title_for_context(text: str) -> str:
    """Strips promotional spam (shipping/discount banners) out of raw scraped
    text so it doesn't pollute the context sent to the AI model."""
    if not text:
        return ""
    cleaned = PROMO_SPAM_RE.sub(" ", text)
    return re.sub(r'\s+', ' ', cleaned).strip()


def extract_product_facts(text: str) -> dict:
    """Extracts hard technical facts (certifications, dimensions/weight/age
    ranges) from a block of specification text, deduped and order-preserving."""
    if not text:
        return {"certificates": [], "dimensions": []}

    certs = []
    for m in CERT_RE.finditer(text):
        val = m.group(0).strip()
        if val not in certs:
            certs.append(val)

    dims = []
    for m in DIMENSION_WEIGHT_RE.finditer(text):
        val = (m.group(1) or m.group(2) or "").strip()
        if val and val not in dims:
            dims.append(val)

    return {"certificates": certs[:8], "dimensions": dims[:8]}


def build_product_context(title: str, brand: str, breadcrumbs: str, specs_text: str) -> str:
    """Combines the extracted signals into one compact context string used
    both for the AI prompt and displayed in the results table."""
    clean_title = clean_product_title_for_context(title)
    clean_specs = clean_product_title_for_context(specs_text)
    facts = extract_product_facts(clean_specs)

    parts = []
    if brand:
        parts.append(f"Marka: {brand.strip()}")
    if clean_title:
        parts.append(f"Produkt: {clean_title.strip()}")
    if breadcrumbs:
        parts.append(f"Kategoria: {breadcrumbs.strip()}")
    if facts["certificates"]:
        parts.append("Certyfikaty: " + ", ".join(facts["certificates"]))
    if facts["dimensions"]:
        parts.append("Parametry: " + ", ".join(facts["dimensions"]))

    context = " | ".join(parts)
    return context[:CONTEXT_MAX_CHARS] if context else "Brak dodatkowego kontekstu tekstowego."


# ---------------------------------------------------------------------------
# Streaming HTML parsing (lxml.etree.iterparse) - keeps memory bounded on very
# large exported HTML files by clearing each product block once it's been read.
# ---------------------------------------------------------------------------

def _text_of(elem) -> str:
    return " ".join(t.strip() for t in elem.itertext() if t and t.strip())


def _elem_classes_and_id(elem) -> str:
    return f"{elem.get('class', '')} {elem.get('id', '')}"


def _first_match_text(elem, class_re) -> str:
    for e in elem.iter():
        tag = e.tag
        if not isinstance(tag, str):
            continue
        if class_re.search(_elem_classes_and_id(e)):
            txt = _text_of(e)
            if txt:
                return txt
    return ""


def _extract_images(elem, base_url: str, skip_excluded: bool = False) -> list:
    urls = []
    seen = set()
    for img in elem.iter("img"):
        if skip_excluded and _has_excluded_ancestor(img):
            continue
        chosen = None
        for attr in IMG_SRC_ATTR_PRIORITY:
            v = img.get(attr)
            if v and v.strip() and not v.strip().startswith("data:"):
                chosen = v.strip()
                break
        if not chosen:
            srcset = img.get("srcset") or img.get("data-srcset")
            if srcset:
                first = srcset.split(",")[0].strip().split(" ")[0]
                if first and not first.startswith("data:"):
                    chosen = first
        if not chosen:
            continue

        full_url = urljoin(base_url, chosen) if base_url else chosen
        if full_url not in seen:
            seen.add(full_url)
            urls.append(full_url)
    return urls


def _looks_like_product_container(elem) -> bool:
    tag = elem.tag
    if not isinstance(tag, str) or tag not in ("div", "li", "article", "section"):
        return False
    if not PRODUCT_CONTAINER_RE.search(_elem_classes_and_id(elem)):
        return False
    return elem.find(".//img") is not None


def _has_excluded_ancestor(elem) -> bool:
    """True if elem sits inside a related/upsell/cross-sell/carousel widget -
    those reuse product-card-like markup for recommendation tiles, which are
    never the actual product(s) being alt-tagged on the page."""
    parent = elem.getparent()
    while parent is not None:
        tag = parent.tag
        if isinstance(tag, str) and EXCLUDED_CONTAINER_RE.search(_elem_classes_and_id(parent)):
            return True
        parent = parent.getparent()
    return False


def _clear_element(elem):
    """Frees a processed subtree and every sibling that came before it, which
    is what keeps memory bounded when iterparse-ing a huge listing page."""
    elem.clear()
    parent = elem.getparent()
    if parent is None:
        return
    while elem.getprevious() is not None:
        del parent[0]


_META_CHARSET_RE = re.compile(rb'charset=["\']?\s*([a-zA-Z0-9_\-]+)', re.IGNORECASE)


def _detect_html_encoding(file_path: str) -> str:
    """Peeks at the first few KB for a declared <meta charset>. Defaults to
    UTF-8 (the overwhelming majority of modern e-commerce exports) instead of
    trusting libxml2's heuristic guess, which can silently mis-decode accented
    characters (e.g. Polish diacritics) when no charset is declared."""
    try:
        with open(file_path, "rb") as f:
            head = f.read(4096)
        match = _META_CHARSET_RE.search(head)
        if match:
            candidate = match.group(1).decode("ascii", errors="ignore").strip()
            try:
                "test".encode(candidate)
                return candidate
            except LookupError:
                pass
    except Exception:
        pass
    return "utf-8"


def _extract_single_product_fallback(file_path: str, base_url: str):
    """Used when no repeated product containers were found - treats the whole
    document as a single product page (typical single-product export)."""
    try:
        parser = etree.HTMLParser(recover=True, huge_tree=True, encoding=_detect_html_encoding(file_path))
        tree = etree.parse(file_path, parser)
    except Exception:
        return
    root = tree.getroot()
    if root is None:
        return

    images = [u for u in _extract_images(root, base_url, skip_excluded=True) if not is_placeholder_image(u)]
    if not images:
        return

    h1_elem = root.find(".//h1")
    h1_text = _text_of(h1_elem) if h1_elem is not None else ""
    title = _first_match_text(root, TITLE_RE) or h1_text
    brand = _first_match_text(root, BRAND_RE)
    breadcrumbs = _first_match_text(root, BREADCRUMB_RE)
    specs_text = _text_of(root)

    context = build_product_context(title, brand, breadcrumbs, specs_text)
    yield {"images": images, "context": context, "label": (title or brand or "Produkt")[:80], "id": uuid.uuid4().hex}


def extract_products_from_html(file_path: str, base_url: str = ""):
    """Streams the HTML file with lxml.etree.iterparse and yields one dict per
    detected product block: {"images": [...], "context": "...", "label": "..."}.
    Falls back to treating the whole document as a single product when no
    repeated product containers are found (e.g. a single product page)."""
    products_found = False
    page_h1 = ""
    page_breadcrumbs = ""

    parse_context = etree.iterparse(
        file_path, events=("end",), html=True, recover=True, huge_tree=True,
        encoding=_detect_html_encoding(file_path),
        tag=("div", "li", "article", "section", "h1", "nav", "ul", "ol"),
    )

    for _event, elem in parse_context:
        tag = elem.tag
        if not isinstance(tag, str):
            continue

        if tag == "h1" and not page_h1:
            t = _text_of(elem)
            if t:
                page_h1 = t
        elif tag in ("nav", "ul", "ol", "div") and not page_breadcrumbs:
            if BREADCRUMB_RE.search(_elem_classes_and_id(elem)):
                t = _text_of(elem)
                if t:
                    page_breadcrumbs = t

        if _looks_like_product_container(elem):
            if not _has_excluded_ancestor(elem):
                images = [u for u in _extract_images(elem, base_url) if not is_placeholder_image(u)]
                if images:
                    products_found = True
                    title = _first_match_text(elem, TITLE_RE) or page_h1
                    brand = _first_match_text(elem, BRAND_RE)
                    specs_text = _text_of(elem)
                    context = build_product_context(title, brand, page_breadcrumbs, specs_text)
                    yield {"images": images, "context": context, "label": (title or brand or "Produkt")[:80], "id": uuid.uuid4().hex}
            _clear_element(elem)

    del parse_context

    if not products_found:
        yield from _extract_single_product_fallback(file_path, base_url)


# ---------------------------------------------------------------------------
# SSRF guard + image download / dimension probing
# ---------------------------------------------------------------------------

def _check_hostname_is_public(hostname: str):
    """Resolves hostname and rejects private/loopback/link-local/reserved IPs.
    Basic SSRF guard - image URLs come from a user-supplied HTML file that
    could point at internal/network-local addresses."""
    try:
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        raise ValueError("Nie udało się rozwiązać hosta w adresie URL.")
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast or ip.is_unspecified:
            raise ValueError("Adres URL wskazuje na prywatny/wewnętrzny adres i został zablokowany.")


def fetch_image_dimensions_fast(url: str):
    """Reads image dimensions from just the first 4 KB (HTTP Range request),
    so tiny icons can be rejected without downloading the whole file. Returns
    (width, height) or None if the size couldn't be determined from the
    partial data (in which case the caller falls back to a full download)."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return None
    try:
        _check_hostname_is_public(parsed.hostname)
    except ValueError:
        return None

    try:
        resp = requests.get(
            url, stream=True, timeout=10, allow_redirects=False,
            headers={"User-Agent": "AltTextGenerator/1.0", "Range": "bytes=0-4095"},
        )
        try:
            if resp.status_code >= 300:
                return None  # redirect or error - let the real download handle it safely
            chunk = resp.raw.read(4096, decode_content=True) if resp.raw else b""
            if not chunk:
                chunk = resp.content[:4096]
            with Image.open(io.BytesIO(chunk)) as img:
                return img.size
        finally:
            resp.close()
    except Exception:
        return None


URL_DOWNLOAD_TIMEOUT_SECONDS = 20
MAX_URL_IMAGE_BYTES = 25 * 1024 * 1024
MAX_URL_REDIRECTS = 5


def _friendly_name_from_url(url: str) -> str:
    """Derives a short display filename from a URL's path (falls back to the
    full URL when the path has no usable basename, e.g. a query-only image
    endpoint) - used for the "main image name" / image list in the CSV export."""
    basename = os.path.basename(urlparse(url).path)
    return basename if basename else url


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
            current_url, stream=True, timeout=URL_DOWNLOAD_TIMEOUT_SECONDS,
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


PAGE_FETCH_TIMEOUT_SECONDS = 20
MAX_PAGE_FETCH_BYTES = 50 * 1024 * 1024  # a product/listing page's HTML shouldn't exceed this


def download_html_page(url: str, dest_dir: str) -> str:
    """Fetches a product/listing page's live HTML server-side (SSRF-guarded,
    manual redirect re-validation - same pattern as download_image_from_url).
    This is the alternative to uploading a browser-saved .html file: a
    "Save Complete" export rewrites every <img src> to a local file in its
    "..._files" folder, discarding the real, downloadable image URLs - fetching
    the live page keeps those URLs intact."""
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
            if content_type and not (content_type.startswith("text/html") or content_type.startswith("application/xhtml")):
                raise ValueError(f"Adres nie zwraca strony HTML (Content-Type: {content_type}).")

            dest_path = os.path.join(dest_dir, "source.html")
            total = 0
            with open(dest_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=65536):
                    total += len(chunk)
                    if total > MAX_PAGE_FETCH_BYTES:
                        raise ValueError(f"Strona przekracza limit {MAX_PAGE_FETCH_BYTES // (1024 * 1024)} MB.")
                    f.write(chunk)
            return dest_path
        finally:
            response.close()

    raise ValueError("Zbyt wiele przekierowań podczas pobierania strony.")


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
    "Jesteś asystentem SEO e-commerce generującym atrybuty alt do zdjęć produktowych. "
    "Odpowiadasz WYŁĄCZNIE gotowym tekstem alt w języku polskim, z poprawnymi znakami "
    "diakrytycznymi, bez cudzysłowów, bez prefiksów typu 'Alt:', bez pytań i komentarzy."
)

ALT_TEXT_PROMPT_TEMPLATE = (
    "Jesteś ekspertem SEO e-commerce. Przeanalizuj to zdjęcie produktu oraz poniższy "
    "kontekst ze strony produktowej: {context}. Wygeneruj zwięzły, naturalny tekst ALT "
    "(4-8 słów) po polsku. Uwzględnij dokładną markę i model wyciągnięty z kontekstu oraz "
    "to, co faktycznie widać na zdjęciu. Nie używaj słów 'zdjęcie przedstawia' ani 'obrazek'."
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


def process_single_image(image_url: str, context: str, job_dir: str) -> dict:
    dims = fetch_image_dimensions_fast(image_url)
    if dims is not None and (dims[0] < MIN_IMAGE_DIMENSION or dims[1] < MIN_IMAGE_DIMENSION):
        return {
            "image_url": image_url,
            "context": context,
            "alt": None,
            "skipped": True,
            "skip_reason": f"Pominięto - zbyt mała grafika ({dims[0]}x{dims[1]} px, prawdopodobnie ikona).",
            "image_data": "",
        }

    local_path = download_image_from_url(image_url, job_dir)
    compressed_path = compress_image(local_path)
    alt_text = generate_alt_via_openai(compressed_path, context)

    media_type = mimetypes.guess_type(compressed_path)[0] or "image/jpeg"
    with open(compressed_path, "rb") as f:
        encoded_image = base64.b64encode(f.read()).decode("utf-8")

    return {
        "image_url": image_url,
        "context": context,
        "alt": alt_text,
        "skipped": False,
        "skip_reason": None,
        "image_data": f"data:{media_type};base64,{encoded_image}",
    }


# ---------------------------------------------------------------------------
# Background job processing
# ---------------------------------------------------------------------------

def clean_old_jobs():
    now = time.time()
    with JOBS_LOCK:
        expired_ids = [
            t_id for t_id, job in JOBS.items()
            if now - job.get("created_at", now) > JOB_MAX_AGE_SECONDS
        ]
        for t_id in expired_ids:
            del JOBS[t_id]


def background_worker(task_id: str, html_path: str, base_url: str, job_dir: str):
    consecutive_errors = 0
    try:
        with JOBS_LOCK:
            if task_id in JOBS:
                JOBS[task_id]["status"] = "parsing"

        tasks = []
        truncated = False
        try:
            for product in extract_products_from_html(html_path, base_url):
                for image_url in product["images"]:
                    tasks.append((image_url, product["context"], product["label"], product["id"]))
                    if len(tasks) >= MAX_IMAGE_TASKS_PER_BATCH:
                        truncated = True
                        break
                if truncated:
                    break
        except Exception as e:
            with JOBS_LOCK:
                if task_id in JOBS:
                    JOBS[task_id]["status"] = "error"
                    JOBS[task_id]["error_message"] = f"Błąd parsowania pliku HTML: {str(e)}"
            return

        if not tasks:
            with JOBS_LOCK:
                if task_id in JOBS:
                    JOBS[task_id]["status"] = "error"
                    JOBS[task_id]["error_message"] = "Nie znaleziono żadnych produktów ani grafik w przesłanym pliku HTML."
            return

        with JOBS_LOCK:
            if task_id in JOBS:
                JOBS[task_id]["total"] = len(tasks)
                JOBS[task_id]["status"] = "processing"
                if truncated:
                    JOBS[task_id]["error_message"] = (
                        f"Uwaga: znaleziono więcej niż {MAX_IMAGE_TASKS_PER_BATCH} grafik - "
                        f"przetworzono pierwsze {MAX_IMAGE_TASKS_PER_BATCH}."
                    )

        stop_event = threading.Event()

        def worker_task(item):
            nonlocal consecutive_errors

            if stop_event.is_set():
                return

            image_url, context, label, page_id = item
            is_error = False
            error_detail = None

            try:
                res = process_single_image(image_url, context, job_dir)
            except Exception as e:
                is_error = True
                error_detail = str(e)
                res = {
                    "image_url": image_url,
                    "context": context,
                    "alt": f"Błąd przetwarzania: {error_detail}",
                    "skipped": False,
                    "skip_reason": None,
                    "image_data": "",
                }

            # Page/product identity + filename - used to regroup flat results
            # back into "one row per page" for the CSV export.
            res["page"] = label
            res["page_id"] = page_id
            res["image_name"] = _friendly_name_from_url(image_url)

            with JOBS_LOCK:
                if task_id not in JOBS or stop_event.is_set():
                    return

                JOBS[task_id]["results"].append(res)
                JOBS[task_id]["processed"] += 1

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
                            f"Poprawnie wygenerowano ALT dla {succ}/{tot} obrazów."
                        )
                else:
                    JOBS[task_id]["success_count"] += 1
                    consecutive_errors = 0

        max_workers = min(MAX_WORKERS, len(tasks))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(worker_task, t) for t in tasks]
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
        "error": f"Przekroczono maksymalny rozmiar pliku ({max_mb} MB). Zmniejsz plik i spróbuj ponownie."
    }), 413


@app.route('/')
def home():
    return render_template('index.html', max_content_mb=MAX_CONTENT_LENGTH_MB)


@app.route('/generate-alt', methods=['POST'])
def generate_alt():
    clean_old_jobs()

    html_file = request.files.get('html_file')
    page_url = (request.form.get('page_url') or "").strip()
    base_url = (request.form.get('base_url') or "").strip()

    has_file = bool(html_file and html_file.filename)

    if not has_file and not page_url:
        return jsonify({"error": "Prześlij plik .html/.htm albo podaj adres URL strony (http/https)."}), 400

    task_id = str(uuid.uuid4())
    job_dir = os.path.join(TMP_UPLOADS_DIR, task_id)
    os.makedirs(job_dir, exist_ok=True)

    if has_file:
        ext = os.path.splitext(html_file.filename)[1].lower()
        if ext not in (".html", ".htm"):
            shutil.rmtree(job_dir, ignore_errors=True)
            return jsonify({"error": "Akceptowane są wyłącznie pliki .html/.htm."}), 400

        if base_url:
            parsed_base = urlparse(base_url)
            if parsed_base.scheme not in ("http", "https") or not parsed_base.hostname:
                shutil.rmtree(job_dir, ignore_errors=True)
                return jsonify({"error": "Nieprawidłowy adres bazowy (base_url)."}), 400

        html_path = os.path.join(job_dir, "source.html")
        try:
            html_file.save(html_path)
        except Exception as e:
            shutil.rmtree(job_dir, ignore_errors=True)
            return jsonify({"error": f"Błąd zapisu pliku: {str(e)}"}), 400

        effective_base_url = base_url
    else:
        parsed_page = urlparse(page_url)
        if parsed_page.scheme not in ("http", "https") or not parsed_page.hostname:
            shutil.rmtree(job_dir, ignore_errors=True)
            return jsonify({"error": "Nieprawidłowy adres URL strony (dozwolone są tylko linki http/https)."}), 400

        try:
            html_path = download_html_page(page_url, job_dir)
        except ValueError as e:
            shutil.rmtree(job_dir, ignore_errors=True)
            return jsonify({"error": f"Nie udało się pobrać strony: {str(e)}"}), 400
        except Exception as e:
            shutil.rmtree(job_dir, ignore_errors=True)
            return jsonify({"error": f"Błąd pobierania strony: {str(e)}"}), 400

        effective_base_url = base_url or page_url

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
        }

    thread = threading.Thread(target=background_worker, args=(task_id, html_path, effective_base_url, job_dir))
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

    return jsonify(job)


if __name__ == '__main__':
    app.run(debug=True, port=5000, threaded=True)
