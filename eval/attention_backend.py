from transformers.utils import is_flash_attn_2_available


def resolve_attention_backend(requested: str | None) -> str:
    """Use FA2 when requested and available; otherwise fall back to SDPA."""
    implementation = requested or "flash_attention_2"
    if implementation != "flash_attention_2":
        return implementation
    if is_flash_attn_2_available():
        return implementation
    print(
        "[attention] flash_attention_2 is unavailable; falling back to sdpa.",
        flush=True,
    )
    return "sdpa"
