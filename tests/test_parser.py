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
        assert "https://cdn.sklep.pl/img/top-of-page-photo.jpg" in content["other_urls"]
        assert "https://cdn.sklep.pl/img/bottom-of-page-photo.jpg" in content["other_urls"]

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


class TestJunkImageFilter:
    def test_is_junk_image_flags_known_keywords(self):
        assert app._is_junk_image("https://cdn.sklep.pl/img/store-logo.png") is True
        assert app._is_junk_image("https://cdn.sklep.pl/icons/payment-visa.svg") is True
        assert app._is_junk_image("https://cdn.sklep.pl/img/inpost-delivery.png") is True
        assert app._is_junk_image("https://cdn.sklep.pl/img/facebook-share.png") is True

    def test_is_junk_image_passes_a_normal_product_photo_url(self):
        assert app._is_junk_image("https://cdn.sklep.pl/img/cybex-priam-front.jpg") is False

    def test_is_junk_image_flags_small_width_or_height_attribute(self):
        assert app._is_junk_image("https://cdn.sklep.pl/img/photo.jpg", {"width": "40"}) is True
        assert app._is_junk_image("https://cdn.sklep.pl/img/photo.jpg", {"height": "60px"}) is True

    def test_is_junk_image_ignores_non_numeric_size_attributes(self):
        assert app._is_junk_image("https://cdn.sklep.pl/img/photo.jpg", {"width": "100%"}) is False
        assert app._is_junk_image("https://cdn.sklep.pl/img/photo.jpg", {"width": "auto"}) is False

    def test_is_junk_image_passes_large_dimensioned_image(self):
        assert app._is_junk_image(
            "https://cdn.sklep.pl/img/photo.jpg", {"width": "800", "height": "600"}
        ) is False

    def test_other_urls_excludes_logos_icons_and_tiny_ui_images(self):
        html = """
        <html><head>
        <meta property="og:image" content="https://cdn.sklep.pl/img/main-photo.jpg">
        </head><body>
        <img src="https://cdn.sklep.pl/img/gallery-2.jpg">
        <img src="https://cdn.sklep.pl/img/store-logo.png">
        <img src="https://cdn.sklep.pl/icons/inpost-courier.png">
        <img src="https://cdn.sklep.pl/img/star-rating.png">
        <img src="https://cdn.sklep.pl/img/tiny-swatch.png" width="32" height="32">
        </body></html>
        """
        content = app.extract_page_content(html, "https://sklep.pl/produkt")
        assert content["other_urls"] == ["https://cdn.sklep.pl/img/gallery-2.jpg"]

    def test_main_image_is_never_junk_filtered(self):
        # og:image is trusted as-is even if its URL happens to contain a
        # junk keyword - only the "other images" pool is filtered.
        html = """
        <html><head>
        <meta property="og:image" content="https://cdn.sklep.pl/img/banner-hero-product.jpg">
        </head><body>
        <img src="https://cdn.sklep.pl/img/second-photo.jpg">
        </body></html>
        """
        content = app.extract_page_content(html, "https://sklep.pl/produkt")
        assert content["main_url"] == "https://cdn.sklep.pl/img/banner-hero-product.jpg"


class TestMaxImagesPerPageCap:
    def test_total_images_never_exceed_max_images_per_page(self):
        many_imgs = "".join(
            f'<img src="https://cdn.sklep.pl/img/photo-{i}.jpg">\n' for i in range(25)
        )
        html = f"""
        <html><head>
        <meta property="og:image" content="https://cdn.sklep.pl/img/main.jpg">
        </head><body>
        {many_imgs}
        </body></html>
        """
        content = app.extract_page_content(html, "https://sklep.pl/produkt")
        total = (1 if content["main_url"] else 0) + len(content["other_urls"])
        assert total <= app.MAX_IMAGES_PER_PAGE
        assert len(content["other_urls"]) == app.MAX_IMAGES_PER_PAGE - 1

    def test_max_images_per_page_is_ten(self):
        assert app.MAX_IMAGES_PER_PAGE == 10


class TestJsonldFullGalleryExtraction:
    def test_all_jsonld_gallery_images_join_the_other_urls_pool(self):
        html = """
        <html><head>
        <script type="application/ld+json">
        {"@type": "Product", "name": "Cybex Priam",
         "image": ["https://cdn.sklep.pl/img/jsonld-1.jpg",
                    "https://cdn.sklep.pl/img/jsonld-2.jpg",
                    "https://cdn.sklep.pl/img/jsonld-3.jpg"]}
        </script>
        </head><body></body></html>
        """
        content = app.extract_page_content(html, "https://sklep.pl/produkt")
        assert content["main_url"] == "https://cdn.sklep.pl/img/jsonld-1.jpg"
        assert "https://cdn.sklep.pl/img/jsonld-2.jpg" in content["other_urls"]
        assert "https://cdn.sklep.pl/img/jsonld-3.jpg" in content["other_urls"]

    def test_jsonld_gallery_images_are_still_junk_filtered(self):
        html = """
        <html><head>
        <script type="application/ld+json">
        {"@type": "Product", "name": "Cybex Priam",
         "image": ["https://cdn.sklep.pl/img/jsonld-main.jpg",
                    "https://cdn.sklep.pl/img/store-logo.jpg"]}
        </script>
        </head><body></body></html>
        """
        content = app.extract_page_content(html, "https://sklep.pl/produkt")
        assert content["main_url"] == "https://cdn.sklep.pl/img/jsonld-main.jpg"
        assert content["other_urls"] == []


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


class TestTruncateToSentence:
    def test_short_text_is_returned_unchanged(self):
        text = "Krotki opis produktu."
        assert app._truncate_to_sentence(text, max_chars=100) == text

    def test_text_at_exactly_max_chars_is_unchanged(self):
        text = "x" * 50
        assert app._truncate_to_sentence(text, max_chars=50) == text

    def test_cuts_at_last_full_sentence_within_limit(self):
        sentence_1 = "Poczatek opisu produktu ktory jest dosyc dlugi i szczegolowy, opisujacy wszystkie cechy."
        sentence_2 = "Drugie zdanie z dodatkowymi informacjami o produkcie i jego zastosowaniu."
        sentence_3 = "Trzecie zdanie zostanie ucie"
        text = f"{sentence_1} {sentence_2} {sentence_3}"

        result = app._truncate_to_sentence(text, max_chars=170)

        assert result == f"{sentence_1} {sentence_2}"
        assert result.endswith(".")

    def test_falls_back_to_last_word_when_no_sentence_end_found(self):
        text = "Slowo " * 30  # no punctuation anywhere
        result = app._truncate_to_sentence(text, max_chars=50)
        assert len(result) <= 50
        assert not result.endswith(" ")
        # must not cut a word in half - the fragment right at the cut
        # should be a whole "Slowo", not e.g. "Slo"
        assert text.startswith(result)
        assert result == "" or text[len(result):len(result) + 1] in (" ", "")

    def test_ignores_a_sentence_end_that_is_too_early(self):
        # A period at position 3 is too close to the start to be a useful
        # cut point - falls back to the last whole word instead.
        text = "Sp. z o.o. oferuje szeroki wybor produktow dla domu i ogrodu w atrakcyjnych cenach"
        result = app._truncate_to_sentence(text, max_chars=40)
        assert len(result) <= 40
        assert not result.endswith(" ")

    def test_default_max_chars_matches_description_max_chars(self):
        assert app.DESCRIPTION_MAX_CHARS == 1000
        long_text = "Zdanie numer jeden. " * 100  # ~2000 chars
        result = app._truncate_to_sentence(long_text)
        assert len(result) <= 1000
        assert result.endswith(".")

    def test_extract_page_content_description_ends_on_a_full_sentence(self):
        sentences = " ".join(f"To jest zdanie numer {i} opisu produktu." for i in range(60))
        html = f"""
        <html><head><title>Fallback</title></head><body>
        <h1>Produkt z dlugim opisem</h1>
        <div id="product-description">{sentences}</div>
        </body></html>
        """
        content = app.extract_page_content(html, "https://sklep.pl/produkt")
        desc_part = content["context"].split("Opis: ")[1]
        assert len(desc_part) <= app.DESCRIPTION_MAX_CHARS
        assert desc_part.endswith(".")
        assert not desc_part.endswith(" ")

    def test_jsonld_description_is_also_sentence_truncated(self):
        sentences = " ".join(f"Zdanie numer {i} opisujace produkt szczegolowo." for i in range(60))
        html = f"""
        <html><head>
        <script type="application/ld+json">
        {{"@type": "Product", "name": "Produkt", "description": "{sentences}"}}
        </script>
        </head><body></body></html>
        """
        content = app.extract_page_content(html, "https://sklep.pl/produkt")
        desc_part = content["context"].split("Opis: ")[1]
        assert len(desc_part) <= app.DESCRIPTION_MAX_CHARS
        assert desc_part.endswith(".")


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
