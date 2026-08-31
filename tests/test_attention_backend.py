from unittest.mock import patch

from eval.attention_backend import resolve_attention_backend


def test_default_uses_fa2_when_available():
    with patch("eval.attention_backend.is_flash_attn_2_available", return_value=True):
        assert resolve_attention_backend(None) == "flash_attention_2"


def test_fa2_falls_back_to_sdpa_when_unavailable():
    with patch("eval.attention_backend.is_flash_attn_2_available", return_value=False):
        assert resolve_attention_backend("flash_attention_2") == "sdpa"


def test_explicit_non_fa2_backend_is_preserved():
    with patch("eval.attention_backend.is_flash_attn_2_available") as availability:
        assert resolve_attention_backend("eager") == "eager"
        availability.assert_not_called()
