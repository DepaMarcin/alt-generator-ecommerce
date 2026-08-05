import os
import re
import io
import uuid
import ipaddress
import socket
import threading
import time
import mimetypes
import base64
import shutil
import random
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
JOB_MAX_AGE_SECONDS = 3600      # Job retention time in memory (1h)
MAX_WORKERS = 5                  # Number of parallel connections (page fetch + OpenAI)

JOBS = {}
JOBS_LOCK = threading.Lock()

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
TMP_UPLOADS_DIR = os.path.join(PROJECT_DIR, "tmp_uploads")
os.makedirs(TMP_UPLOADS_DIR, exist_ok=True)

CONSECUTIVE_ERROR_LIMIT = 5          # Isolated transient errors shouldn't abort the whole batch
MAX_URLS_PER_BATCH = 1000            # Safety cap on how many page URLs one batch can queue (after sitemap expansion)
MAX_SITEMAP_URLS_PER_FILE = 2000     # Cap per individual sitemap/sitemap-index fetch
MAX_SITEMAP_DEPTH = 3                # How many levels of nested sitemap indexes to follow
MIN_IMAGE_DIMENSION = 150            # px - images smaller than this on either side are treated as icons
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
# Meta-tag extraction (og:title/og:image) - deliberately minimal: we only
# need a handful of <meta> tags and <title>, not a full DOM.
# ---------------------------------------------------------------------------

_META_IMAGE_KEYS = ("og:image", "og:image:url", "og:image:secure_url", "twitter:image", "twitter:image:src")


class _MetaTitleImageParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.title = ""
        self.og_title = ""
        self.og_description = ""
        self.image_candidates = []
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        if tag == "title":
            self._in_title = True
        elif tag == "meta":
            key = (attrs_dict.get("property") or attrs_dict.get("name") or "").strip().lower()
            content = (attrs_dict.get("content") or "").strip()
            if not content:
                return
            if key == "og:title" and not self.og_title:
                self.og_title = content
            elif key == "og:description" and not self.og_description:
                self.og_description = content
            elif key in _META_IMAGE_KEYS and not content.startswith("data:"):
                if content not in self.image_candidates:
                    self.image_candidates.append(content)

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title:
            self.title += data


def extract_page_meta(html_text: str, page_url: str) -> dict:
    """Pulls og:title/og:description (context) and og:image + fallbacks
    (main image, with alternates for resilience against a single stale/404
    image-cache URL) out of a page's HTML."""
    parser = _MetaTitleImageParser()
    try:
        parser.feed(html_text)
    except Exception:
        pass  # tolerate malformed markup - keep whatever was parsed so far

    title = re.sub(r'\s+', ' ', (parser.og_title or parser.title or "").strip())

    context_parts = []
    if title:
        context_parts.append(f"Produkt: {title}")
    if parser.og_description:
        desc = re.sub(r'\s+', ' ', parser.og_description.strip())[:200]
        if desc:
            context_parts.append(f"Opis: {desc}")
    context = " | ".join(context_parts) or "Brak dodatkowego kontekstu tekstowego."

    seen = set()
    image_urls = []
    for candidate in parser.image_candidates:
        full_url = urljoin(page_url, candidate)
        if _is_http_url(full_url) and full_url not in seen:
            seen.add(full_url)
            image_urls.append(full_url)

    return {"context": context, "image_urls": image_urls, "title": title}


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
    "Jesteś asystentem SEO e-commerce generującym atrybuty alt do zdjęć produktowych. "
    "Odpowiadasz WYŁĄCZNIE gotowym tekstem alt w języku polskim, z poprawnymi znakami "
    "diakrytycznymi, bez cudzysłowów, bez prefiksów typu 'Alt:', bez pytań i komentarzy."
)

ALT_TEXT_PROMPT_TEMPLATE = (
    "Jesteś ekspertem SEO e-commerce. Przeanalizuj to zdjęcie produktu oraz poniższy "
    "kontekst wyciągnięty ze strony produktowej: {context}. Wygeneruj zwięzły, naturalny "
    "tekst ALT (4-8 słów) po polsku. Uwzględnij dokładną nazwę produktu/markę wyciągniętą "
    "z kontekstu oraz to, co faktycznie widać na zdjęciu. Nie używaj słów 'zdjęcie "
    "przedstawia' ani 'obrazek'."
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


def process_single_image(image_url: str, context: str, job_dir: str, fallback_urls: list = None) -> dict:
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

    return {
        "image_url": used_url,
        "context": context,
        "alt": alt_text,
        "skipped": False,
        "skip_reason": None,
        "image_data": f"data:{media_type};base64,{encoded_image}",
    }


def process_page_url(page_url: str, job_dir: str) -> dict:
    """One full unit of work: fetch a product page, pull its title/main image
    from meta tags, download the image and generate its ALT text."""
    html_text, final_url = fetch_page_html(page_url)
    meta = extract_page_meta(html_text, final_url)

    if not meta["image_urls"]:
        return {
            "page_url": page_url,
            "image_url": "",
            "context": meta["context"],
            "alt": None,
            "skipped": True,
            "skip_reason": "Nie znaleziono obrazu głównego (brak og:image) na stronie.",
            "image_data": "",
        }

    primary, *fallback_urls = meta["image_urls"]
    result = process_single_image(primary, meta["context"], job_dir, fallback_urls=fallback_urls)
    result["page_url"] = page_url
    return result


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
                res = process_page_url(page_url, job_dir)
            except Exception as e:
                is_error = True
                error_detail = str(e)
                res = {
                    "page_url": page_url,
                    "image_url": "",
                    "context": "",
                    "alt": f"Błąd przetwarzania: {error_detail}",
                    "skipped": False,
                    "skip_reason": None,
                    "image_data": "",
                }

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

    return jsonify(job)


if __name__ == '__main__':
    app.run(debug=True, port=5000, threaded=True)
