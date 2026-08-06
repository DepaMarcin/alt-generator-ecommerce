"""AI Evaluation Suite - validates generated ALT text against business rules
(length/word count, forbidden phrasing, hallucinated product info leaking
into unrelated graphics, keyword stuffing across a batch). The OpenAI call
itself is always mocked via the `openai_stub` fixture (see conftest.py), so
these tests run offline and cost nothing - what's under test is (a) the real
generate_alt_via_openai() post-processing (quote/newline cleanup) and (b) the
quality-gate helper functions below, exercised against both compliant and
deliberately bad canned model outputs.
"""
import pytest

import app

ALT_MAX_CHARS = 120
ALT_MIN_WORDS = 4
ALT_MAX_WORDS = 12
FORBIDDEN_PHRASES = ("zdjęcie przedstawia", "obrazek", "grafika", "alt:")

PRODUCT_CONTEXT = (
    "Produkt: Wózek spacerowy Cybex Priam 5.0 Comfort | "
    "Opis z body: Wózek wielofunkcyjny 2w1 z gondolą i spacerówką, rama aluminiowa."
)


def _word_count(text: str) -> int:
    return len(text.split())


def has_forbidden_phrase(text: str) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in FORBIDDEN_PHRASES)


def has_stray_quotes(text: str) -> bool:
    return text.startswith(('"', "'")) or text.endswith(('"', "'"))


def leaks_terms(text: str, forbidden_terms) -> bool:
    lowered = text.lower()
    return any(term.lower() in lowered for term in forbidden_terms)


def has_keyword_stuffing(alts, prefix_words: int = 3, threshold: float = 0.5) -> bool:
    """Flags a batch of ALT texts as 'stuffed' if more than `threshold` of
    them share an identical first-`prefix_words`-word prefix (case
    insensitive) - the pattern the original bug report complained about."""
    prefixes = [' '.join(a.strip().lower().split()[:prefix_words]) for a in alts if a.strip()]
    if not prefixes:
        return False
    most_common = max(prefixes.count(p) for p in set(prefixes))
    return (most_common / len(prefixes)) > threshold


# ---------------------------------------------------------------------------
# Formal requirements: length, word count
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mocked_response", [
    "Czarny wózek dziecięcy Cybex Priam z beżowym siedziskiem.",
    "Uchwyt na kubek zamontowany na ramie wózka Cybex.",
    "Adaptery do montażu fotelika na stelażu wózka.",
])
def test_alt_length_and_word_count(openai_stub, dummy_image_path, mocked_response):
    openai_stub(mocked_response)
    alt = app.generate_alt_via_openai(dummy_image_path, PRODUCT_CONTEXT)

    assert len(alt) <= ALT_MAX_CHARS
    assert ALT_MIN_WORDS <= _word_count(alt) <= ALT_MAX_WORDS


# ---------------------------------------------------------------------------
# Forbidden phrases / stray quotes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mocked_response, expect_forbidden", [
    ("Czarny wózek dziecięcy Cybex Priam z beżowym siedziskiem.", False),
    ("Zdjęcie przedstawia czarny wózek dziecięcy Cybex.", True),
    ("Obrazek wózka spacerowego Cybex Priam.", True),
    ("Grafika przedstawiająca ramę wózka Cybex.", True),
])
def test_no_forbidden_phrases(openai_stub, dummy_image_path, mocked_response, expect_forbidden):
    openai_stub(mocked_response)
    alt = app.generate_alt_via_openai(dummy_image_path, PRODUCT_CONTEXT)

    assert has_forbidden_phrase(alt) is expect_forbidden


def test_generated_alt_strips_stray_quotes(openai_stub, dummy_image_path):
    openai_stub('"Czarny wózek dziecięcy Cybex Priam."')
    alt = app.generate_alt_via_openai(dummy_image_path, PRODUCT_CONTEXT)

    assert not has_stray_quotes(alt)


# ---------------------------------------------------------------------------
# Hallucination guard: product info must never leak into an ALT for a
# graphic that isn't the product itself (logo, delivery icon, ...).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mocked_response, should_leak", [
    ("Logo sklepu internetowego na białym tle.", False),
    ("Ikona darmowej dostawy powyżej 100 zł.", False),
    ("Płatność kartą Visa i Mastercard.", False),
    ("Wózek Cybex Priam widoczny obok logo sklepu.", True),
])
def test_non_product_image_hallucination_guard(openai_stub, dummy_image_path, mocked_response, should_leak):
    # The page context still talks about the main product even though this
    # particular image is a logo/delivery icon - the model must not leak the
    # product name/brand into an ALT for an unrelated graphic.
    openai_stub(mocked_response)
    alt = app.generate_alt_via_openai(dummy_image_path, PRODUCT_CONTEXT)

    leaked = leaks_terms(alt, ["wózek", "cybex"])
    assert leaked is should_leak


# ---------------------------------------------------------------------------
# Keyword-stuffing detection across a batch of images for one product.
# ---------------------------------------------------------------------------

def test_keyword_stuffing_detection_flags_stuffed_batch(openai_stub, dummy_image_path):
    stuffed_responses = [
        f"Cybex Priam Comfort wózek widok numer {i}." for i in range(10)
    ]
    openai_stub(stuffed_responses)

    alts = [app.generate_alt_via_openai(dummy_image_path, PRODUCT_CONTEXT) for _ in range(10)]

    assert has_keyword_stuffing(alts) is True


def test_keyword_stuffing_detection_passes_varied_batch(openai_stub, dummy_image_path):
    varied_responses = [
        "Czarny wózek dziecięcy Cybex Priam z beżową budką.",
        "Uchwyt na kubek zamontowany na ramie wózka.",
        "Adaptery do montażu fotelika na stelażu wózka.",
        "Moskitiera dedykowana do gondoli wózka Cybex.",
        "Spacerówka Cybex w kolorze czarnym, widok z boku.",
        "Koła tylne z systemem amortyzacji wózka.",
        "Torba na akcesoria montowana pod siedziskiem.",
        "Osłona przeciwdeszczowa na budkę wózka.",
        "Pas bezpieczeństwa pięciopunktowy w foteliku.",
        "Rączka regulowana wózka spacerowego Cybex.",
    ]
    openai_stub(varied_responses)

    alts = [app.generate_alt_via_openai(dummy_image_path, PRODUCT_CONTEXT) for _ in range(10)]

    assert has_keyword_stuffing(alts) is False
