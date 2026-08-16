import json

import pytest
from curl_cffi import requests as curl_requests

import app


class TestMainImageCascade:
    def test_priority_1_og_image_wins(self, html_with_og_image):
        content = app.extract_page_content(html_with_og_image, "https://sklep.pl/produkt")
        assert content["main_url"] == "https://cdn.sklep.pl/img/product-main.jpg"

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

    def test_legitimate_og_image_is_still_used_as_main(self):
        # Sanity check for the happy path: a real, non-junk og:image is
        # still trusted as the main photo.
        html = """
        <html><head>
        <meta property="og:image" content="https://cdn.sklep.pl/img/product-front-view.jpg">
        </head><body>
        <img src="https://cdn.sklep.pl/img/second-photo.jpg">
        </body></html>
        """
        content = app.extract_page_content(html, "https://sklep.pl/produkt")
        assert content["main_url"] == "https://cdn.sklep.pl/img/product-front-view.jpg"

    def test_junk_og_image_is_rejected_and_next_candidate_promoted_to_main(self):
        # Regression coverage for the Answear bug: og:image pointing at a
        # share icon must NOT become main_url - the next real candidate
        # (here, a gallery photo) gets promoted instead.
        html = """
        <html><head>
        <meta property="og:image" content="https://cdn.sklep.pl/img/logo_share.ans.png">
        </head><body>
        <div class="gallery"><img src="https://cdn.sklep.pl/img/product_F1.jpg"></div>
        <img src="https://cdn.sklep.pl/img/qr-code.png">
        </body></html>
        """
        content = app.extract_page_content(html, "https://sklep.pl/produkt")
        assert content["main_url"] == "https://cdn.sklep.pl/img/product_F1.jpg"
        assert "https://cdn.sklep.pl/img/logo_share.ans.png" not in content["other_urls"]
        assert "https://cdn.sklep.pl/img/qr-code.png" not in content["other_urls"]

    def test_junk_priority_gallery_image_falls_through_to_plain_body_image(self):
        # No og:image/JSON-LD at all here - the only "priority" signal is a
        # share icon sitting inside the gallery container, which must be
        # skipped in favor of the first genuinely non-junk <img> on the page.
        html = """
        <html><head><title>Sklep XYZ</title></head><body>
        <div class="gallery"><img src="https://cdn.sklep.pl/img/share-button.png"></div>
        <img src="https://cdn.sklep.pl/img/product_1.jpg">
        </body></html>
        """
        content = app.extract_page_content(html, "https://sklep.pl/produkt")
        assert content["main_url"] == "https://cdn.sklep.pl/img/product_1.jpg"

    def test_all_candidates_junk_yields_no_main_image(self):
        html = """
        <html><head>
        <meta property="og:image" content="https://cdn.sklep.pl/img/logo_share.ans.png">
        </head><body>
        <img src="https://cdn.sklep.pl/img/qr-code.png">
        </body></html>
        """
        content = app.extract_page_content(html, "https://sklep.pl/produkt")
        assert content["main_url"] is None
        assert content["other_urls"] == []


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

    def test_no_extractable_text_anywhere_omits_description_segment(self):
        # A genuinely empty body - not even the longest-paragraph fallback
        # (see TestFallbackHtmlDescription) has anything to find.
        html = """
        <html><head><title>Fallback</title></head><body>
        <h1>Produkt bez opisu</h1>
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


class TestStyleScriptContentExcludedFromDescription:
    """Regression coverage for the Answear bug: a <style> block nested
    inside the description container was leaking raw CSS rules (e.g.
    ".Icon_icon-v_XzHkY:before { content: "\\D"; }") into the extracted
    product description/context."""

    def test_style_tag_inside_description_container_is_ignored(self):
        html = """
        <html><head><title>Fallback</title></head><body>
        <h1>Produkt testowy</h1>
        <div id="description">
          <style>.Icon_icon-v_XzHkY:before { content: "\\D"; } .btn { color: red; }</style>
          Prawdziwy opis produktu, ktory powinien zostac zachowany w calosci.
        </div>
        </body></html>
        """
        content = app.extract_page_content(html, "https://sklep.pl/produkt")
        assert "Prawdziwy opis produktu" in content["context"]
        assert "Icon_icon-v_XzHkY" not in content["context"]
        assert "content:" not in content["context"]
        assert "{" not in content["context"] and "}" not in content["context"]

    def test_script_and_noscript_inside_description_container_are_ignored(self):
        html = """
        <html><head><title>Fallback</title></head><body>
        <h1>Produkt testowy</h1>
        <div class="product-description">
          <script>var trackingId = "abc123"; console.log(trackingId);</script>
          <noscript>Wlacz JavaScript, aby zobaczyc pelna tresc.</noscript>
          Opis wlasciwy produktu bez zadnego szumu technicznego.
        </div>
        </body></html>
        """
        content = app.extract_page_content(html, "https://sklep.pl/produkt")
        assert "Opis wlasciwy produktu" in content["context"]
        assert "trackingId" not in content["context"]
        assert "Wlacz JavaScript" not in content["context"]

    def test_svg_markup_inside_description_container_is_ignored(self):
        html = """
        <html><head><title>Fallback</title></head><body>
        <h1>Produkt testowy</h1>
        <div id="description">
          <svg viewBox="0 0 24 24"><path d="M12 2L2 7"></path><title>strzalka</title></svg>
          Opis produktu bez tresci z wektorowej ikony svg.
        </div>
        </body></html>
        """
        content = app.extract_page_content(html, "https://sklep.pl/produkt")
        assert "Opis produktu bez tresci" in content["context"]
        assert "strzalka" not in content["context"]

    def test_jsonld_still_captured_even_though_script_is_a_skip_tag(self):
        # <script type="application/ld+json"> must still be parsed for its
        # structured data even though plain <script> content is skipped
        # everywhere else.
        html = """
        <html><head>
        <script type="application/ld+json">
        {"@type": "Product", "name": "Produkt", "description": "Pelny opis produktu pochodzacy wprost z danych strukturalnych strony, zawierajacy pelen zestaw informacji o zastosowaniu."}
        </script>
        </head><body></body></html>
        """
        content = app.extract_page_content(html, "https://sklep.pl/produkt")
        assert "Pelny opis produktu pochodzacy wprost z danych strukturalnych" in content["context"]

    def test_css_artifact_regex_strips_rule_blocks_and_bare_selectors(self):
        text = 'Prawdziwy tekst. .Icon_icon-v_XzHkY:before { content: "\\D"; } #footer { display:none; } Reszta opisu.'
        cleaned = app._strip_css_artifacts(text)
        assert "Prawdziwy tekst." in cleaned
        assert "Reszta opisu." in cleaned
        assert "{" not in cleaned and "}" not in cleaned
        assert "Icon_icon-v_XzHkY" not in cleaned
        assert "#footer" not in cleaned


class TestCleanHtmlText:
    """Regression coverage for the zakupy.auchan.pl bug: the JSON-LD (or
    body) "description" field itself contained ready-made HTML markup
    (<div style="...">Marka</div> <br> <p style="...">Informacje...</p>)
    instead of plain text."""

    def test_strips_html_tags(self):
        text = '<div style="color:red">Marka</div> <br> <p style="font-size:12px">Informacje o produkcie</p>'
        cleaned = app.clean_html_text(text)
        assert "<" not in cleaned and ">" not in cleaned
        assert "Marka" in cleaned
        assert "Informacje o produkcie" in cleaned

    def test_decodes_html_entities(self):
        text = "Krem 50&nbsp;ml &amp; balsam &quot;Premium&quot;"
        cleaned = app.clean_html_text(text)
        assert "&nbsp;" not in cleaned
        assert "&amp;" not in cleaned
        assert "&quot;" not in cleaned
        assert "50" in cleaned and "ml" in cleaned
        assert "&" in cleaned  # the decoded literal ampersand itself is fine
        assert '"Premium"' in cleaned

    def test_collapses_whitespace_and_newlines(self):
        text = "Linia pierwsza\n\n   Linia   druga\t\tLinia trzecia"
        cleaned = app.clean_html_text(text)
        assert cleaned == "Linia pierwsza Linia druga Linia trzecia"

    def test_empty_input_returns_empty_string(self):
        assert app.clean_html_text("") == ""
        assert app.clean_html_text(None) == ""

    def test_jsonld_description_with_raw_html_markup_is_cleaned(self):
        # The actual Auchan bug: JSON-LD "description" contains a literal,
        # ready-made HTML fragment instead of plain text.
        html = """
        <html><head>
        <script type="application/ld+json">
        {"@type": "Product", "name": "Produkt",
         "description": "<div style=\\"font-weight:bold\\">Marka</div> <br> <p style=\\"margin:0\\">Informacje o skladzie, zastosowaniu i sposobie uzycia produktu na co dzien, w pelnej formie tekstowej.</p>"}
        </script>
        </head><body></body></html>
        """
        content = app.extract_page_content(html, "https://zakupy.auchan.pl/produkt")
        assert "<div" not in content["context"]
        assert "<br>" not in content["context"]
        assert "<p" not in content["context"]
        assert "style=" not in content["context"]
        assert "Marka" in content["context"]
        assert "Informacje o skladzie" in content["context"]


class TestIsTrivialJsonldDescription:
    def test_short_text_is_trivial(self):
        assert app._is_trivial_jsonld_description("Marka Pantene") is True

    def test_empty_or_none_is_trivial(self):
        assert app._is_trivial_jsonld_description("") is True
        assert app._is_trivial_jsonld_description(None) is True

    def test_bare_brand_label_is_trivial_even_past_the_length_threshold(self):
        # Over MIN_DESCRIPTION_LENGTH chars (not caught by the length check
        # alone) but still just a 4-word brand label - the regex path must
        # catch this too.
        text = "Marka Bardzo-Dlugiej-Miedzynarodowej-Nazwy-Marketingowej-Kosmetycznej-Firmy Coco Mademoiselle Intensive"
        assert len(text) >= app.MIN_DESCRIPTION_LENGTH
        assert app._is_trivial_jsonld_description(text) is True

    def test_real_sentence_starting_with_marka_is_not_trivial(self):
        text = (
            "Marka Informacje o skladzie i zastosowaniu produktu, ktory nawilza, "
            "odzywia i regeneruje wlosy suche oraz zniszczone."
        )
        assert len(text) >= app.MIN_DESCRIPTION_LENGTH
        assert app._is_trivial_jsonld_description(text) is False

    def test_long_real_description_is_not_trivial(self):
        text = (
            "Szampon do wlosow suchych i zniszczonych, wzbogacony o kompleks "
            "odzywczy Pro-V, ktory glebokp nawilza i regeneruje strukture wlosa."
        )
        assert app._is_trivial_jsonld_description(text) is False


class TestJsonldStubFallsThroughToBodyDescription:
    """Regression coverage for the zakupy.auchan.pl Pantene shampoo bug:
    JSON-LD returning only a thin "Marka Pantene" stub must not be accepted
    as the final description - the app must fall through to the full HTML
    body description (skład, właściwości, opakowanie, ...)."""

    def test_trivial_jsonld_falls_through_to_full_body_description(self):
        html = """
        <html><head>
        <script type="application/ld+json">
        {"@type": "Product", "name": "Szampon Pantene", "description": "Marka Pantene"}
        </script>
        </head><body>
        <h1>Szampon Pantene Pro-V</h1>
        <div id="description">
        Szampon do wlosow suchych i zniszczonych z kompleksem Pro-V, ktory glebogo
        nawilza, wygladza i chroni wlosy przed uszkodzeniami przy codziennym myciu.
        </div>
        </body></html>
        """
        content = app.extract_page_content(html, "https://zakupy.auchan.pl/szampon-pantene")
        assert "Marka Pantene" not in content["context"]
        assert "kompleksem Pro-V" in content["context"]
        assert "nawilza" in content["context"]

    def test_non_trivial_jsonld_description_is_still_used_directly(self):
        html = """
        <html><head>
        <script type="application/ld+json">
        {"@type": "Product", "name": "Szampon",
         "description": "Szampon Pantene Pro-V do wlosow suchych i zniszczonych, wzbogacony o kompleks odzywczy oraz witaminy regenerujace."}
        </script>
        </head><body>
        <div id="description">Zupelnie inny, dluzszy opis z body, ktory nie powinien zostac uzyty.</div>
        </body></html>
        """
        content = app.extract_page_content(html, "https://zakupy.auchan.pl/produkt")
        assert "kompleks odzywczy" in content["context"]
        assert "Zupelnie inny" not in content["context"]


class TestMultipleDescriptionSectionsCombined:
    """Regression coverage: the parser used to stop at the FIRST matching
    description container per rank - if a real description was split
    across several sibling containers (e.g. multiple .product-info blocks
    for "Opis"/"Działanie"), everything after the first was silently
    dropped."""

    def test_multiple_sibling_containers_at_same_rank_are_combined(self):
        html = """
        <html><head><title>Fallback</title></head><body>
        <h1>Produkt testowy</h1>
        <div class="product-description">Pierwsza sekcja opisu produktu.</div>
        <div class="product-description">Druga sekcja opisu z dodatkowymi informacjami.</div>
        </body></html>
        """
        content = app.extract_page_content(html, "https://sklep.pl/produkt")
        assert "Pierwsza sekcja opisu produktu" in content["context"]
        assert "Druga sekcja opisu z dodatkowymi informacjami" in content["context"]

    def test_ingredients_and_attributes_sections_are_appended_to_main_description(self):
        html = """
        <html><head><title>Fallback</title></head><body>
        <h1>Szampon Pantene</h1>
        <div id="description">Szampon do wlosow suchych i zniszczonych.</div>
        <div class="product-attributes">Pojemnosc: 400ml. Marka: Pantene.</div>
        <div class="ingredients">Sklad: Aqua, Sodium Laureth Sulfate, Dimethicone.</div>
        </body></html>
        """
        content = app.extract_page_content(html, "https://zakupy.auchan.pl/produkt")
        assert "Szampon do wlosow suchych" in content["context"]
        assert "Pojemnosc: 400ml" in content["context"]
        assert "Sklad: Aqua" in content["context"]

    def test_supplementary_section_excluded_from_cross_sell_widget(self):
        html = """
        <html><head><title>Fallback</title></head><body>
        <h1>Produkt</h1>
        <div id="description">Opis wlasciwego produktu.</div>
        <div class="recommended">
          <div class="ingredients">Sklad innego, polecanego produktu.</div>
        </div>
        </body></html>
        """
        content = app.extract_page_content(html, "https://sklep.pl/produkt")
        assert "Opis wlasciwego produktu" in content["context"]
        assert "innego, polecanego produktu" not in content["context"]


class TestMinDescriptionLengthThreshold:
    """Regression coverage for the "opis nadal zbyt krotki" follow-up bug:
    MIN_DESCRIPTION_LENGTH = 100 is now enforced on every description
    candidate (JSON-LD or HTML), with short HTML sections combined until
    the threshold is cleared instead of being accepted as-is."""

    def test_min_description_length_constant_is_100(self):
        assert app.MIN_DESCRIPTION_LENGTH == 100

    def test_short_jsonld_description_rejected_in_favor_of_longer_html(self):
        short_jsonld = "Marka Pantene, znana marka kosmetykow do wlosow."
        long_html = (
            "Szampon do wlosow suchych i zniszczonych z formula odbudowujaca "
            "strukture wlosa, nadajaca mu blask i miekkosc na co dzien."
        )
        assert len(short_jsonld) < app.MIN_DESCRIPTION_LENGTH
        assert len(long_html) >= app.MIN_DESCRIPTION_LENGTH

        html = f"""
        <html><head>
        <script type="application/ld+json">
        {{"@type": "Product", "name": "Szampon", "description": "{short_jsonld}"}}
        </script>
        </head><body>
        <div id="description">{long_html}</div>
        </body></html>
        """
        content = app.extract_page_content(html, "https://zakupy.auchan.pl/produkt")
        assert short_jsonld not in content["context"]
        assert "formula odbudowujaca strukture wlosa" in content["context"]

    def test_no_single_html_section_reaches_threshold_alone_so_they_are_combined(self):
        desc_chunk = "Krotki ogolny opis produktu."
        details_chunk = "Szczegoly techniczne produktu."
        spec_chunk = "Specyfikacja obejmuje wage, wymiary oraz material wykonania tego konkretnego produktu."
        # None of the three chunks alone clears the threshold, and neither
        # do the first two together - only all three combined do.
        assert len(desc_chunk) < app.MIN_DESCRIPTION_LENGTH
        assert len(details_chunk) < app.MIN_DESCRIPTION_LENGTH
        assert len(spec_chunk) < app.MIN_DESCRIPTION_LENGTH
        assert len(desc_chunk) + 1 + len(details_chunk) < app.MIN_DESCRIPTION_LENGTH

        html = f"""
        <html><head><title>Fallback</title></head><body>
        <h1>Produkt testowy</h1>
        <div id="description">{desc_chunk}</div>
        <div class="details">{details_chunk}</div>
        <div class="specification">{spec_chunk}</div>
        </body></html>
        """
        content = app.extract_page_content(html, "https://zakupy.auchan.pl/produkt")
        assert desc_chunk in content["context"]
        assert details_chunk in content["context"]
        assert spec_chunk in content["context"]
        desc_part = content["context"].split("Opis: ")[1]
        assert len(desc_part) >= app.MIN_DESCRIPTION_LENGTH

    def test_combined_sections_still_short_are_used_as_is_without_error(self):
        # Even after combining everything available, the result stays under
        # the threshold - the app must still use it (better than nothing)
        # rather than erroring out or leaving the description empty.
        html = """
        <html><head><title>Fallback</title></head><body>
        <h1>Produkt</h1>
        <div id="description">Bardzo krotki opis.</div>
        </body></html>
        """
        content = app.extract_page_content(html, "https://zakupy.auchan.pl/produkt")
        assert "Bardzo krotki opis" in content["context"]


class TestFallbackHtmlDescription:
    """Regression coverage for zakupy.auchan.pl pages (e.g. Szampon
    Pantene) that use NONE of the standard class/id/itemprop selectors at
    all, so the normal cascade finds nothing and the context is left with
    just "Produkt: <name>". find_fallback_html_description() is the last
    resort: heading-labeled sections, then app-state JSON <script>s, then
    the single longest paragraph-like block on the page."""

    def test_heading_based_extraction_uses_parent_container_text(self):
        heading_parent_text = (
            "Szampon do wlosow suchych i zniszczonych z kompleksem odzywczym, "
            "ktory regeneruje strukture wlosa na co dzien."
        )
        assert len(heading_parent_text) >= app.MIN_DESCRIPTION_LENGTH
        html = f"""
        <html><body>
        <div class="section-xyz">
          <h3>Opis produktu</h3>
          <p>{heading_parent_text}</p>
        </div>
        </body></html>
        """
        result = app.find_fallback_html_description(html)
        assert "kompleksem odzywczym" in result
        assert len(result) >= app.MIN_DESCRIPTION_LENGTH

    def test_heading_based_extraction_uses_following_siblings_when_no_shared_parent(self):
        sib1 = "Pierwszy akapit opisu produktu z istotnymi informacjami."
        sib2 = "Drugi akapit z dodatkowymi szczegolami technicznymi produktu."
        html = f"""
        <html><body>
        <h3>Informacje o produkcie</h3>
        <p>{sib1}</p>
        <p>{sib2}</p>
        <h3>Nastepna sekcja, ktora nie powinna zostac dolaczona</h3>
        <p>Tresc zupelnie innej sekcji, ktora musi zostac pominieta.</p>
        </body></html>
        """
        result = app.find_fallback_html_description(html)
        assert sib1 in result
        assert sib2 in result
        assert "innej sekcji" not in result

    def test_heading_phrase_matching_is_case_insensitive_and_covers_all_listed_phrases(self):
        for phrase in ("SKŁADNIKI", "O Produkcie", "szczegóły", "Składniki, alergeny"):
            html = f"""
            <html><body>
            <h3>{phrase}</h3>
            <p>Tresc sekcji zawierajaca wystarczajaco duzo znakow, aby zostac
            uznana za poprawny kandydujacy opis produktu w tescie.</p>
            </body></html>
            """
            result = app.find_fallback_html_description(html)
            assert "Tresc sekcji" in result, f"failed for heading phrase {phrase!r}"

    def test_next_data_script_is_scanned_for_description_key(self):
        nextdata_desc = (
            "Pelny opis produktu pobrany z danych aplikacji Next.js, "
            "zawierajacy sklad oraz zastosowanie kosmetyku."
        )
        assert len(nextdata_desc) >= app.MIN_DESCRIPTION_LENGTH
        state = {"props": {"pageProps": {"product": {"name": "Szampon", "description": nextdata_desc}}}}
        html = f"""
        <html><body>
        <script id="__NEXT_DATA__" type="application/json">{json.dumps(state)}</script>
        </body></html>
        """
        result = app.find_fallback_html_description(html)
        assert result == nextdata_desc

    def test_generic_application_json_script_is_also_scanned(self):
        longer_desc = "Opis produktu zapisany w generycznym skrypcie JSON stanu aplikacji sklepu internetowego, w pelnej formie."
        assert len(longer_desc) >= app.MIN_DESCRIPTION_LENGTH
        state = {"productDetails": longer_desc}
        html = f"""
        <html><body>
        <script type="application/json">{json.dumps(state)}</script>
        </body></html>
        """
        result = app.find_fallback_html_description(html)
        assert result == longer_desc

    def test_longest_paragraph_is_used_as_absolute_last_resort(self):
        longest_p = (
            "To jest najdluzszy akapit na stronie, zawierajacy realny opis "
            "produktu z istotnymi szczegolami technicznymi i uzytkowymi."
        )
        nav_p = (
            "To jest bardzo dlugi tekst nawigacyjny umieszczony w stopce strony, "
            "ktory zdecydowanie nie powinien zostac uzyty jako opis produktu wcale, nigdy."
        )
        assert len(nav_p) > len(longest_p)  # the excluded text is the longer one
        html = f"""
        <html><body>
        <footer><p>{nav_p}</p></footer>
        <div class="something-unrecognizable">
          <p>Krotki akapit.</p>
          <p>{longest_p}</p>
        </div>
        </body></html>
        """
        result = app.find_fallback_html_description(html)
        assert result == longest_p

    def test_returns_empty_string_when_nothing_is_found(self):
        html = "<html><body><h1>Produkt</h1></body></html>"
        assert app.find_fallback_html_description(html) == ""

    def test_end_to_end_pantene_style_page_with_no_standard_selectors(self):
        # The actual reported bug: JSON-LD gives only a brand stub, and the
        # body uses no recognizable class/id/itemprop at all - only a
        # heading-labeled section identifies the real description.
        heading_parent_text = (
            "Szampon Pantene Pro-V do wlosow suchych i zniszczonych, z kompleksem "
            "odzywczym oraz witaminami regenerujacymi strukture wlosa."
        )
        assert len(heading_parent_text) >= app.MIN_DESCRIPTION_LENGTH
        html = f"""
        <html><head>
        <script type="application/ld+json">
        {{"@type": "Product", "name": "Szampon Pantene", "description": "Marka Pantene"}}
        </script>
        </head><body>
        <h1>Szampon Pantene Pro-V</h1>
        <div class="section-abc123">
          <h4>Opis produktu</h4>
          <p>{heading_parent_text}</p>
        </div>
        </body></html>
        """
        content = app.extract_page_content(html, "https://zakupy.auchan.pl/szampon-pantene")
        assert "Marka Pantene" not in content["context"]
        assert "kompleksem odzywczym" in content["context"]
        assert content["context"].startswith("Produkt: Szampon Pantene Pro-V")


class TestAuchanJunkKeywords:
    """Regression coverage for the zakupy.auchan.pl banner/icon leak: promo
    banners and store icons (nowosc.png, piggy_2.png, jakkupowac.png,
    b2b.png, kategoria-okazje.png) were showing up as "other" images."""

    @pytest.mark.parametrize("url", [
        "https://zakupy.auchan.pl/img/jakkupowac.png",
        "https://zakupy.auchan.pl/img/piggy_2.png",
        "https://zakupy.auchan.pl/img/nowosc.png",
        "https://zakupy.auchan.pl/img/marki-premium.png",
        "https://zakupy.auchan.pl/img/b2b.png",
        "https://zakupy.auchan.pl/img/kategoria-okazje.png",
        "https://zakupy.auchan.pl/img/zgarnij-rabat.png",
        "https://zakupy.auchan.pl/img/gazetka-promocji.png",
        "https://zakupy.auchan.pl/img/kategoria-sezonowe.png",
        "https://zakupy.auchan.pl/img/kategoria-partnerzy.png",
        "https://zakupy.auchan.pl/icons/sezonowe.png",
        "https://zakupy.auchan.pl/icons/partnerzy.png",
    ])
    def test_is_junk_image_flags_auchan_specific_keywords(self, url):
        assert app._is_junk_image(url) is True

    def test_other_urls_excludes_auchan_banners_and_icons(self):
        html = """
        <html><head>
        <meta property="og:image" content="https://zakupy.auchan.pl/img/produkt-glowny.jpg">
        </head><body>
        <img src="https://zakupy.auchan.pl/img/produkt-detal.jpg">
        <img src="https://zakupy.auchan.pl/img/nowosc.png">
        <img src="https://zakupy.auchan.pl/img/piggy_2.png">
        <img src="https://zakupy.auchan.pl/img/jakkupowac.png">
        <img src="https://zakupy.auchan.pl/img/b2b.png">
        <img src="https://zakupy.auchan.pl/img/kategoria-okazje.png">
        <img src="https://zakupy.auchan.pl/img/kategoria-sezonowe.png">
        <img src="https://zakupy.auchan.pl/img/kategoria-partnerzy.png">
        </body></html>
        """
        content = app.extract_page_content(html, "https://zakupy.auchan.pl/produkt")
        assert content["main_url"] == "https://zakupy.auchan.pl/img/produkt-glowny.jpg"
        assert content["other_urls"] == ["https://zakupy.auchan.pl/img/produkt-detal.jpg"]


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
         "description": "Krzeselko do karmienia regulowane na 6 poziomow wysokosci, z tacka zdejmowana oraz pasami zabezpieczajacymi dziecko."}
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
        assert "product-main.jpg" in html_text
        assert final_url == "https://sklep.pl/produkt"

    def test_fetch_page_html_rejects_non_html_content_type(self, mock_html_fetch):
        mock_html_fetch("not html", content_type="image/jpeg")
        try:
            app.fetch_page_html("https://sklep.pl/produkt")
            assert False, "expected ValueError for a non-HTML content type"
        except ValueError:
            pass


class TestFetchPageHtmlAntiBotImpersonation:
    """Regression coverage for the curl_cffi Chrome-impersonation swap: a
    lot of e-commerce anti-bot protection (Answear and friends) flat-out
    403s plain `requests`, so fetch_page_html must go through curl_cffi
    with impersonate="chrome120" plus a realistic Chrome header set, and a
    403 must come back as a clean, catchable Polish error - not a crash."""

    def _stub_session(self, mocker, status_code=200, html_text="<html></html>"):
        fake_response = mocker.MagicMock()
        fake_response.is_redirect = False
        fake_response.status_code = status_code
        fake_response.headers = {"Content-Type": "text/html"}
        fake_response.iter_content.return_value = [html_text.encode("utf-8")]
        if status_code >= 400:
            fake_response.raise_for_status.side_effect = curl_requests.RequestsError(
                f"HTTP Error {status_code}:", response=fake_response
            )
        else:
            fake_response.raise_for_status.return_value = None

        mocker.patch.object(app, "_check_hostname_is_public", return_value=None)
        mock_session_cls = mocker.patch.object(app.curl_requests, "Session")
        mock_session_cls.return_value.get.return_value = fake_response
        return mock_session_cls

    def test_uses_chrome120_impersonation(self, mocker):
        mock_session_cls = self._stub_session(mocker)

        app.fetch_page_html("https://sklep.pl/produkt")

        assert app.PAGE_FETCH_IMPERSONATE == "chrome120"
        mock_session_cls.assert_called_once_with(impersonate="chrome120")

    def test_sends_realistic_chrome_headers(self, mocker):
        mock_session_cls = self._stub_session(mocker)

        app.fetch_page_html("https://sklep.pl/produkt")

        _, call_kwargs = mock_session_cls.return_value.get.call_args
        sent_headers = call_kwargs["headers"]
        assert "Chrome" in sent_headers["User-Agent"]
        assert sent_headers["Sec-Ch-Ua-Platform"] == '"Windows"'
        assert sent_headers["Sec-Fetch-Mode"] == "navigate"
        assert sent_headers == app.PAGE_FETCH_HEADERS

    def test_403_response_becomes_a_readable_polish_error_not_a_crash(self, mocker):
        self._stub_session(mocker, status_code=403)

        with pytest.raises(ValueError) as exc_info:
            app.fetch_page_html("https://sklep.pl/produkt")

        message = str(exc_info.value)
        assert "403" in message
        assert "zablokował" in message.lower()

    def test_other_http_error_also_becomes_a_readable_error(self, mocker):
        self._stub_session(mocker, status_code=500)

        with pytest.raises(ValueError) as exc_info:
            app.fetch_page_html("https://sklep.pl/produkt")

        assert "500" in str(exc_info.value)
