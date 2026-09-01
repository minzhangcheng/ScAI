import pytest

from backend.utils.url_validation import validated_http_url, validated_proxy_url


def test_http_url_is_preserved_verbatim():
    assert validated_http_url("http://localhost:8000/v1", "llm_base_url") == (
        "http://localhost:8000/v1"
    )


@pytest.mark.parametrize("value", ["http:/broken", "https:/broken", "localhost:8000/v1"])
def test_malformed_http_url_is_rejected(value):
    with pytest.raises(ValueError, match="valid http or https URL"):
        validated_http_url(value, "llm_base_url")


def test_socks5_proxy_url_remains_supported():
    assert validated_proxy_url("socks5://127.0.0.1:1080", "proxy_url") == (
        "socks5://127.0.0.1:1080"
    )
