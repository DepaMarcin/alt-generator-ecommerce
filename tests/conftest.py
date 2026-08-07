import os
import sys
from types import SimpleNamespace

import pytest
import requests

# Belt-and-suspenders alongside pytest.ini's `pythonpath = .` - makes sure
# `import app` works even if tests are invoked in a way that skips ini files.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# app.py builds its OpenAI client at import time from OPENAI_API_KEY - a
# dummy key keeps the whole suite hermetic (no real .env / network needed),
# since every actual API call is mocked via the openai_stub fixture below.
os.environ.setdefault("OPENAI_API_KEY", "sk-test-dummy-key-for-unit-tests")

import app  # noqa: E402


# ---------------------------------------------------------------------------
# Sample HTML fixtures - one per main-image cascade priority, plus one for
# the body-description extraction.
# ---------------------------------------------------------------------------

@pytest.fixture
def html_with_og_image():
    """Priority 1: og:image should win even though JSON-LD Product.image and
    a gallery <img loading="eager"> are also present on the page."""
    return """
    <html><head>
    <title>Sklep XYZ</title>
    <meta property="og:title" content="Cybex Priam wozek 2w1">
    <meta property="og:image" content="https://cdn.sklep.pl/img/og-main.jpg">
    <script type="application/ld+json">
    {"@type": "Product", "name": "Cybex Priam", "image": ["https://cdn.sklep.pl/img/jsonld-main.jpg"]}
    </script>
    </head><body>
    <div class="gallery"><img src="https://cdn.sklep.pl/img/gallery-1.jpg" loading="eager"></div>
    </body></html>
    """


@pytest.fixture
def html_with_jsonld_only():
    """Priority 2: no og:image - the schema.org Product.image from JSON-LD
    should be picked over the plain (non-priority) <img> tags below it."""
    return """
    <html><head><title>Sklep XYZ</title>
    <script type="application/ld+json">
    {"@context": "https://schema.org", "@type": "Product", "name": "Cybex Priam",
     "image": {"@type": "ImageObject", "url": "https://cdn.sklep.pl/img/jsonld-main.jpg"}}
    </script>
    </head><body>
    <img src="https://cdn.sklep.pl/img/random-1.jpg">
    <img src="https://cdn.sklep.pl/img/random-2.jpg">
    </body></html>
    """


@pytest.fixture
def html_with_gallery_img_only():
    """Priority 3: no og:image and no JSON-LD - an <img> inside a gallery
    container should win over plain <img> tags elsewhere on the page. The
    two non-gallery images use neutral filenames on purpose - this fixture
    isn't meant to exercise the junk-image filter (see test_junk_filter
    fixtures/tests for that)."""
    return """
    <html><head><title>Sklep XYZ</title></head><body>
    <img src="https://cdn.sklep.pl/img/top-of-page-photo.jpg">
    <div id="gallery">
      <img src="https://cdn.sklep.pl/img/gallery-main.jpg">
    </div>
    <img src="https://cdn.sklep.pl/img/bottom-of-page-photo.jpg">
    </body></html>
    """


@pytest.fixture
def html_with_body_description():
    """A noisy SEO meta description alongside a real .product-description
    container in the body - extract_page_content must prefer the latter."""
    return """
    <html><head>
    <title>Fallback tytul</title>
    <meta name="description" content="Dobra Cena! Szybka wysylka! Kup teraz w super promocji!">
    <meta property="og:title" content="Otibiom krople do uszu 15ml">
    </head><body>
    <h1>Otibiom krople do uszu 15ml</h1>
    <div class="product-description">
      Preparat probiotyczny w formie kropli do uszu psa i kota, wspomaga naturalna
      flore bakteryjna przewodu sluchowego oraz lagodzi podraznienia.
    </div>
    </body></html>
    """


# ---------------------------------------------------------------------------
# Mocked external services - OpenAI Vision + outbound HTTP (requests) - so
# the suite never makes a real network call or spends real API credits.
# ---------------------------------------------------------------------------

@pytest.fixture
def openai_stub(mocker):
    """Patches the OpenAI chat-completions call. Call the returned function
    with either a single string (every call returns that text) or a list of
    strings (each successive call returns the next one - handy for
    simulating a batch of images from the same product page)."""
    mock_create = mocker.patch.object(app.openai_client.chat.completions, "create")

    def _wrap(text):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])

    def _configure(content):
        if isinstance(content, (list, tuple)):
            mock_create.side_effect = [_wrap(c) for c in content]
        else:
            mock_create.return_value = _wrap(content)
        return mock_create

    return _configure


@pytest.fixture
def dummy_image_path(tmp_path):
    """A tiny real JPEG on disk - generate_alt_via_openai only reads and
    base64-encodes it, it's never actually sent anywhere in tests."""
    from PIL import Image
    path = tmp_path / "sample.jpg"
    Image.new("RGB", (10, 10), color="red").save(path, format="JPEG")
    return str(path)


@pytest.fixture
def mock_html_fetch(mocker):
    """Patches the SSRF hostname guard (no real DNS lookups) and the
    underlying requests.Session.get call, so fetch_page_html() can be
    exercised against canned HTML without ever touching the network."""
    mocker.patch.object(app, "_check_hostname_is_public", return_value=None)

    def _configure(html_text: str, content_type: str = "text/html"):
        fake_response = mocker.MagicMock()
        fake_response.is_redirect = False
        fake_response.is_permanent_redirect = False
        fake_response.raise_for_status.return_value = None
        fake_response.headers = {"Content-Type": content_type}
        fake_response.iter_content.return_value = [html_text.encode("utf-8")]
        mocker.patch.object(requests.Session, "get", return_value=fake_response)
        return fake_response

    return _configure
