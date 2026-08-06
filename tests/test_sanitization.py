import pytest

import app

# Built via chr(92) rather than typed as a literal backslash-escape, so the
# JS-style "\uXXXX" sequences these tests exercise survive intact instead of
# being silently decoded by whatever's transporting this source text.
_BS = chr(92)


def _js_unicode_escape(hex4: str) -> str:
    return _BS + "u" + hex4


class TestDecodeUnicodeJsEscapes:
    def test_uXXXX_escape_becomes_slash(self):
        raw = "a" + _js_unicode_escape("002f") + "b"
        assert app._decode_unicode_js_escapes(raw) == "a/b"

    def test_backslash_slash_escape_becomes_slash(self):
        raw = "a" + _BS + "/b"
        assert app._decode_unicode_js_escapes(raw) == "a/b"

    def test_combined_escapes_in_full_url(self):
        esc = _js_unicode_escape("002f")
        raw = f"https:{esc}{esc}cdn.sklep.pl{esc}img{esc}cache{esc}photo_1.jpg"
        assert app._decode_unicode_js_escapes(raw) == "https://cdn.sklep.pl/img/cache/photo_1.jpg"

    def test_uXXXX_escape_becomes_space(self):
        raw = "a" + _js_unicode_escape("0020") + "b"
        assert app._decode_unicode_js_escapes(raw) == "a b"

    def test_text_without_escapes_is_untouched(self):
        raw = "https://cdn.sklep.pl/img/photo.jpg"
        assert app._decode_unicode_js_escapes(raw) == raw


class TestSanitizeImageUrl:
    def test_decodes_html_entities(self):
        result = app.sanitize_image_url(
            "https://cdn.sklep.pl/img/photo.jpg?a=1&amp;b=2", "https://sklep.pl/produkt"
        )
        assert result == "https://cdn.sklep.pl/img/photo.jpg?a=1&b=2"

    def test_resolves_relative_path_to_absolute(self):
        result = app.sanitize_image_url("/media/img.jpg", "https://sklep.pl")
        assert result == "https://sklep.pl/media/img.jpg"

    def test_resolves_relative_path_against_deep_page_url(self):
        result = app.sanitize_image_url("/media/img.jpg", "https://sklep.pl/kategoria/produkt-1")
        assert result == "https://sklep.pl/media/img.jpg"

    def test_decodes_js_unicode_escapes_and_resolves_absolute(self):
        raw = r"https:\/\/www.bobowozki.com.pl\/img\/cache\/8610bc8\/photo_1.jpg"
        result = app.sanitize_image_url(raw, "https://www.bobowozki.com.pl/produkt")
        assert result == "https://www.bobowozki.com.pl/img/cache/8610bc8/photo_1.jpg"
        assert "\\" not in result

    @pytest.mark.parametrize("bad_url", [
        "data:image/png;base64,AAAA",
        "javascript:void(0)",
        "ftp://cdn.sklep.pl/img.jpg",
        "",
        "   ",
        None,
    ])
    def test_rejects_invalid_or_disallowed_urls(self, bad_url):
        assert app.sanitize_image_url(bad_url, "https://sklep.pl/x") is None


class TestParseUrlListText:
    def test_extracts_one_url_per_line(self):
        text = "https://sklep.pl/produkt-1\nhttps://sklep.pl/produkt-2"
        assert app.parse_url_list_text(text) == [
            "https://sklep.pl/produkt-1", "https://sklep.pl/produkt-2"
        ]

    def test_csv_row_picks_first_http_token_and_ignores_rest(self):
        text = "https://sklep.pl/produkt-1;Nazwa produktu;12.99 PLN"
        assert app.parse_url_list_text(text) == ["https://sklep.pl/produkt-1"]

    def test_space_separated_row_picks_first_token(self):
        text = "https://sklep.pl/produkt-1 dodatkowy tekst bez adresu"
        assert app.parse_url_list_text(text) == ["https://sklep.pl/produkt-1"]

    def test_strips_surrounding_quotes(self):
        text = '"https://sklep.pl/produkt-1"'
        assert app.parse_url_list_text(text) == ["https://sklep.pl/produkt-1"]

    def test_deduplicates_preserving_first_seen_order(self):
        text = "https://sklep.pl/a\nhttps://sklep.pl/b\nhttps://sklep.pl/a"
        assert app.parse_url_list_text(text) == ["https://sklep.pl/a", "https://sklep.pl/b"]

    def test_ignores_lines_without_a_url(self):
        text = "Naglowek;Opis;Cena\nhttps://sklep.pl/produkt-1;Opis;12.99"
        assert app.parse_url_list_text(text) == ["https://sklep.pl/produkt-1"]

    def test_empty_or_blank_input_returns_empty_list(self):
        assert app.parse_url_list_text("") == []
        assert app.parse_url_list_text("   \n  \n") == []


class TestCheckHostnameIsPublicSsrfGuard:
    @pytest.mark.parametrize("blocked_host", [
        "127.0.0.1",       # loopback
        "localhost",       # loopback hostname
        "192.168.1.10",    # private range
        "10.0.0.5",        # private range
        "169.254.1.1",     # link-local
    ])
    def test_blocks_private_and_internal_hosts(self, blocked_host):
        with pytest.raises(ValueError):
            app._check_hostname_is_public(blocked_host)

    def test_allows_public_ip_literal(self):
        # Should not raise - a real, public IP address literal.
        app._check_hostname_is_public("8.8.8.8")
