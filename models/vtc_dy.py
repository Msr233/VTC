from __future__ import annotations

from typing import Any

from .visual_token_coding import VisualTokenCoding


class VTCDy(VisualTokenCoding):
    """VTC_Dy with DyRSO, DyTA, and SC-TopK."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("semantic_version", "v5.0.0")
        kwargs.setdefault("semantic_frame_type_source", "residual")
        super().__init__(**kwargs)


__all__ = ["VTCDy"]
