from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class IFrameTokenSelection:
    keep_indices: torch.Tensor
    num_anchor_tokens: int
    num_salient_tokens: int


def parse_anchor_grid(value: str) -> tuple[int, int]:
    parts = str(value).lower().split("x")
    if len(parts) != 2:
        raise ValueError(f"anchor_grid must look like 'HxW', got {value!r}")
    try:
        h = int(parts[0])
        w = int(parts[1])
    except ValueError as exc:
        raise ValueError(f"anchor_grid must contain integers, got {value!r}") from exc
    if h <= 0 or w <= 0:
        raise ValueError(f"anchor_grid dimensions must be > 0, got {value!r}")
    return h, w


class IFrameTokenSelector:
    VALID_MODES = {"spatial_anchor_salient", "salient_only", "random"}

    def __init__(self, *, random_seed_offset: int = 104729) -> None:
        self.random_seed_offset = int(random_seed_offset)

    def select(
        self,
        frame_features: torch.Tensor,
        *,
        keep_tokens: int,
        token_grid_h: int,
        token_grid_w: int,
        residual_energy: torch.Tensor | None = None,
        is_first_iframe: bool = False,
        mode: str = "spatial_anchor_salient",
        anchor_ratio: float = 0.2,
        anchor_grid: str = "4x6",
        salient_residual_weight: float = 0.5,
        global_frame_idx: int = 0,
    ) -> IFrameTokenSelection:
        mode = str(mode)
        if mode not in self.VALID_MODES:
            raise ValueError(f"Unknown I-frame selection mode: {mode!r}")
        if not math.isfinite(float(anchor_ratio)) or anchor_ratio < 0.0 or anchor_ratio > 1.0:
            raise ValueError(f"anchor_ratio must be in [0, 1], got {anchor_ratio!r}")
        if (
            not math.isfinite(float(salient_residual_weight))
            or salient_residual_weight < 0.0
            or salient_residual_weight > 1.0
        ):
            raise ValueError(
                "salient_residual_weight must be in [0, 1], "
                f"got {salient_residual_weight!r}"
            )

        N = int(frame_features.shape[0])
        K = min(N, max(0, int(keep_tokens)))
        H = int(token_grid_h)
        W = int(token_grid_w)
        if H * W != N:
            raise AssertionError(
                f"I-frame token grid mismatch: token_grid_h*token_grid_w={H}*{W}={H*W}, N={N}"
            )

        features_f32 = frame_features.float()
        frame_mean = features_f32.mean(dim=0, keepdim=True)
        contrast = torch.linalg.vector_norm(features_f32 - frame_mean, ord=2, dim=-1)
        contrast = torch.nan_to_num(contrast, nan=0.0, posinf=0.0, neginf=0.0)

        residual_score = None
        if residual_energy is not None:
            residual_score = torch.nan_to_num(
                residual_energy.detach().float().reshape(-1),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            if residual_score.numel() != N:
                raise AssertionError(
                    f"residual_energy length mismatch: got {residual_score.numel()}, expected {N}"
                )

        if is_first_iframe or residual_score is None:
            score = contrast
        else:
            score = (
                float(salient_residual_weight) * residual_score.to(contrast.device)
                + (1.0 - float(salient_residual_weight)) * contrast
            )
            score = torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

        if K == 0:
            keep = torch.empty(0, dtype=torch.long, device=frame_features.device)
            return self._result(keep, 0, 0)
        if K == N:
            keep = torch.arange(N, dtype=torch.long, device=frame_features.device)
            return self._result(keep, 0, N)

        if mode == "random":
            generator_device = frame_features.device.type if frame_features.device.type == "cuda" else "cpu"
            gen = torch.Generator(device=generator_device)
            gen.manual_seed(int(global_frame_idx) + self.random_seed_offset)
            perm = torch.randperm(N, generator=gen, device=frame_features.device)
            keep = torch.sort(perm[:K]).values
            return self._result(keep, 0, K)

        if mode == "salient_only":
            keep = self._topk_sorted(score, K)
            return self._result(keep, 0, K)

        grid_h, grid_w = parse_anchor_grid(anchor_grid)
        anchor_indices = self._select_region_anchors(score, H, W, grid_h, grid_w)
        desired_anchor_count = min(
            K,
            max(int(math.floor(float(anchor_ratio) * K)), min(grid_h * grid_w, K)),
        )
        if anchor_indices.numel() > desired_anchor_count:
            anchor_scores = score[anchor_indices]
            top_local = torch.topk(anchor_scores, desired_anchor_count, largest=True).indices
            anchor_indices = anchor_indices[top_local]
        anchor_indices = torch.unique(anchor_indices, sorted=True)

        selected = torch.zeros(N, dtype=torch.bool, device=frame_features.device)
        selected[anchor_indices] = True
        need = K - int(selected.sum().item())
        if need > 0:
            fill_scores = score.masked_fill(selected, -torch.inf)
            fill = torch.topk(fill_scores, need, largest=True).indices
            selected[fill] = True
        keep = torch.nonzero(selected, as_tuple=False).squeeze(1)
        if keep.numel() != K:
            raise AssertionError(f"I-frame keep count mismatch: got {keep.numel()}, expected {K}")
        return self._result(
            keep,
            int(anchor_indices.numel()),
            K - int(anchor_indices.numel()),
        )

    def _select_region_anchors(
        self,
        score: torch.Tensor,
        H: int,
        W: int,
        grid_h: int,
        grid_w: int,
    ) -> torch.Tensor:
        anchors = []
        for gh in range(grid_h):
            h0 = int(math.floor(gh * H / grid_h))
            h1 = int(math.floor((gh + 1) * H / grid_h))
            for gw in range(grid_w):
                w0 = int(math.floor(gw * W / grid_w))
                w1 = int(math.floor((gw + 1) * W / grid_w))
                if h1 <= h0 or w1 <= w0:
                    continue
                rows = torch.arange(h0, h1, device=score.device, dtype=torch.long)
                cols = torch.arange(w0, w1, device=score.device, dtype=torch.long)
                region = (rows[:, None] * W + cols[None, :]).reshape(-1)
                local = torch.argmax(score[region])
                anchors.append(region[local])
        if not anchors:
            return torch.empty(0, dtype=torch.long, device=score.device)
        return torch.stack(anchors).long()

    def _topk_sorted(self, score: torch.Tensor, K: int) -> torch.Tensor:
        indices = torch.topk(score, int(K), largest=True).indices
        return torch.sort(indices).values.long()

    def _result(
        self,
        keep: torch.Tensor,
        num_anchor: int,
        num_salient: int,
    ) -> IFrameTokenSelection:
        keep = torch.sort(torch.unique(keep.long())).values
        return IFrameTokenSelection(
            keep_indices=keep,
            num_anchor_tokens=int(num_anchor),
            num_salient_tokens=int(num_salient),
        )
