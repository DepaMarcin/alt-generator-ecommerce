import app


class TestMainImageCascade:
    def test_priority_1_og_image_wins(self, html_with_og_image):
        content = app.extract_page_content(html_with_og_image, "https://sklep.pl/produkt")
        assert content["main_url"] == "https://cdn.sklep.pl/img/og-main.jpg"

    def test_priority_2_jsonld_product_image(self, html_with_jsonld_only):
        content = app.extract_page_content(html_with_jsonld_only, "https://sklep.pl/produkt")
        assert content["main_url"] == "https://cdn.sklep.pl/img/jsonld-main.jpg"

    def test_priority_3_eager_or_gallery_img(self, html_with_gallery_img_only):
        content = app.extract_page_content(html_with_gallery_img_only, "https://sklep.pl/produkt")
        assert content["main_url"] == "https://cdn.sklep.pl/img/gallery-main.jpg"
        assert "https://cdn.sklep.pl/img/header-logo.jpg" in content["other_urls"]
        assert "https://cdn.sklep.pl/img/footer-banner.jpg" in content["other_urls"]

    def test_priority_3_fetchpriority_high_attribute(self):
        html = """
        <html><head><title>Sklep XYZ</title></head><body>
        <img src="https://cdn.sklep.pl/img/random.jpg">
        <img src="https://cdn.sklep.pl/img/priority.jpg" fetchpriority="high">
        </body></html>
        """
        content = app.extract_page_content(html, "https://sklep.pl/produkt")
        assert content["main_url"] == "https://cdn.sklep.pl/img/priority.jpg"

    def test_falls_back_to_first_img_when_no_signals_present(self):
        html = """
        <html><head><title>Fallback</title></head><body>
        <img src="https://cdn.sklep.pl/img/first.jpg">
        <img src="https://cdn.sklep.pl/img/second.jpg">
        </body></html>
        """
        content = app.extract_page_content(html, "https://sklep.pl/produkt")
        assert content["main_url"] == "https://cdn.sklep.pl/img/first.jpg"
        assert content["other_urls"] == ["https://cdn.sklep.pl/img/second.jpg"]

    def test_svg_icons_are_never_chosen_as_main_image(self):
        html = """
        <html><head>
        <meta property="og:image" content="https://cdn.sklep.pl/icons/logo.svg">
        </head><body>
        <img src="https://cdn.sklep.pl/img/real-photo.jpg">
        </body></html>
        """
        content = app.extract_page_content(html, "https://sklep.pl/produkt")
        assert content["main_url"] == "https://cdn.sklep.pl/img/real-photo.jpg"


class TestBodyDescriptionExtraction:
    def test_uses_product_description_container_not_meta_description(self, html_with_body_description):
        content = app.extract_page_content(html_with_body_description, "https://sklep.pl/otibiom")
        assert "Opis:" in content["context"]
        assert "probiotyczny" in content["context"]
        assert "Dobra Cena" not in content["context"]
        assert "Szybka wysylka" not in content["context"]

    def test_title_prefers_h1_over_og_title_and_title_tag(self, html_with_body_description):
        content = app.extract_page_content(html_with_body_description, "https://sklep.pl/otibiom")
        assert content["title"] == "Otibiom krople do uszu 15ml"

    def test_itemprop_description_has_priority_over_id_description(self):
        html = """
        <html><head><title>Fallback</title></head><body>
        <h1>Produkt testowy</h1>
        <div id="description">Ogolny opis w kontenerze #description.</div>
        <div itemprop="description">Dokladny opis w itemprop, powinien wygrac.</div>
        </body></html>
        """
        content = app.extract_page_content(html, "https://sklep.pl/produkt")
        assert "Dokladny opis w itemprop" in content["context"]
        assert "Ogolny opis w kontenerze" not in content["context"]

    def test_no_description_container_omits_description_segment(self):
        html = """
        <html><head><title>Fallback</title></head><body>
        <h1>Produkt bez opisu</h1>
        <p>Losowy tekst poza jakimkolwiek kontenerem opisu.</p>
        </body></html>
        """
        content = app.extract_page_content(html, "https://sklep.pl/produkt")
        assert "Opis:" not in content["context"]
        assert content["context"] == "Produkt: Produkt bez opisu"

    def test_description_is_truncated_to_max_chars(self):
        long_text = "Slowo " * 200  # ~1200 chars, well over the cap
        html = f"""
        <html><head><title>Fallback</title></head><body>
        <h1>Dlugi opis</h1>
        <div class="product-description">{long_text}</div>
        </body></html>
        """
        content = app.extract_page_content(html, "https://sklep.pl/produkt")
        desc_part = content["context"].split("Opis: ")[1]
        assert len(desc_part) <= app.DESCRIPTION_MAX_CHARS


class TestDescriptionContextLeakPrevention:
    """Regression coverage for the "wrong product's description leaks into
    the context" bug: a recommended-products/cross-sell/carousel widget
    elsewhere on the page must never contribute to the extracted context."""

    def test_jsonld_product_description_is_used_when_available(self):
        html = """
        <html><head><title>Fallback</title>
        <script type="application/ld+json">
        {"@context": "https://schema.org", "@type": "Product", "name": "Krzeselko do karmienia",
         "description": "Krzeselko do karmienia regulowane na 6 poziomow wysokosci, tacka zdejmowana."}
        </script>
        </head><body>
        <h1>Krzeselko do karmienia</h1>
        <div class="recommended">
          <h3>Polecane produkty</h3>
          <p class="product-description">Wozek spacerowy Cybex Priam - najlepsza cena, kup teraz!</p>
        </div>
        </body></html>
        """
        content = app.extract_page_content(html, "https://sklep.pl/krzeselko")
        assert "Opis: Krzeselko do karmienia regulowane" in content["context"]
        assert "Wozek spacerowy Cybex" not in content["context"]

    def test_recommended_and_cross_sell_blocks_are_ignored_in_favor_of_product_description(self):
        html = """
        <html><head><title>Fallback</title></head><body>
        <h1>Krzeselko do karmienia</h1>

        <div class="recently-viewed">
          <p class="description-content">Ostatnio ogladane: Wozek spacerowy Cybex Priam, zobacz teraz.</p>
        </div>

        <div class="cross-sell">
          <div id="product-description">Fotelik samochodowy Maxi-Cosi - darmowa dostawa!</div>
        </div>

        <aside class="sidebar">
          <div class="product-description">Bestsellery: Gondola dzieciecia - najnizsza cena 199 zl.</div>
        </aside>

        <div id="product-description">
          Krzeselko do karmienia z regulacja wysokosci, tacka zdejmowana, rama aluminiowa.
        </div>
        </body></html>
        """
        content = app.extract_page_content(html, "https://sklep.pl/krzeselko")
        assert "Krzeselko do karmienia z regulacja wysokosci" in content["context"]
        assert "Wozek spacerowy Cybex" not in content["context"]
        assert "Fotelik samochodowy" not in content["context"]
        assert "Gondola dzieciecia" not in content["context"]

    def test_marketing_and_price_noise_fragments_are_stripped(self):
        html = """
        <html><head><title>Fallback</title></head><body>
        <h1>Krzeselko do karmienia</h1>
        <div id="product-description">
          <p>Krzeselko do karmienia z regulacja wysokosci i zdejmowana tacka.</p>
          <span>Najnizsza cena: 249 zł</span>
          <button>Kup teraz</button>
          <p>Darmowa dostawa od 200 zł</p>
        </div>
        </body></html>
        """
        content = app.extract_page_content(html, "https://sklep.pl/krzeselko")
        assert "Krzeselko do karmienia z regulacja wysokosci" in content["context"]
        assert "249" not in content["context"]
        assert "Kup teraz" not in content["context"]
        assert "Darmowa dostawa" not in content["context"]


class TestFetchPageHtmlWithMockedRequests:
    """Demonstrates the 'safe mock of external services' requirement for the
    Requests side (OpenAI is covered in test_ai_eval.py): no real network
    call is made, fetch_page_html() just returns the canned HTML."""

    def test_fetch_page_html_returns_mocked_content(self, mock_html_fetch, html_with_og_image):
        mock_html_fetch(html_with_og_image)
        html_text, final_url = app.fetch_page_html("https://sklep.pl/produkt")
        assert "og-main.jpg" in html_text
        assert final_url == "https://sklep.pl/produkt"

    def test_fetch_page_html_rejects_non_html_content_type(self, mock_html_fetch):
        mock_html_fetch("not html", content_type="image/jpeg")
        try:
            app.fetch_page_html("https://sklep.pl/produkt")
            assert False, "expected ValueError for a non-HTML content type"
        except ValueError:
            pass
