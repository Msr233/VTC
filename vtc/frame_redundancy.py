"""Region selection with fixed-length indices and cached GPU layouts."""

import math
import torch


def select_candidates(scores, candidates, count):
    if count == 0:
        return candidates[:0]
    if count == candidates.numel():
        return candidates.sort().values
    values = torch.nan_to_num(
        scores[candidates].detach().float(),
        nan=-torch.inf,
        posinf=torch.inf,
        neginf=-torch.inf,
    )
    use_scores = torch.isfinite(values).any() & (values.max() >= 1e-8)
    local = torch.topk(values, count, largest=True, sorted=False).indices
    return torch.where(use_scores, candidates[local], candidates[:count]).sort().values


class FrameRedundancySuppression:
    def __init__(self, grid_h=4, grid_w=4, coverage_ratio=0.2):
        self.grid_h, self.grid_w = grid_h, grid_w
        self.coverage_ratio = coverage_ratio
        self._layouts = {}
        self._batches = {}

    def _regions(self, height, width, device):
        key = (height, width, str(device))
        if key not in self._layouts:
            rows = torch.tensor_split(torch.arange(height, device=device), self.grid_h)
            cols = torch.tensor_split(torch.arange(width, device=device), self.grid_w)
            self._layouts[key] = [
                (r[:, None] * width + c[None, :]).flatten()
                for r in rows
                for c in cols
                if r.numel() and c.numel()
            ]
        return self._layouts[key]

    @staticmethod
    def _quotas(sizes, count):
        q, r = divmod(count, len(sizes))
        quotas = [min(size, q + (i < r)) for i, size in enumerate(sizes)]
        left = count - sum(quotas)
        while left:
            for i, size in enumerate(sizes):
                if quotas[i] < size:
                    quotas[i] += 1
                    left -= 1
                    if left == 0:
                        break
        return quotas

    def select(self, energy, count, height, width):
        size = energy.numel()
        if count == 0:
            return torch.empty(0, device=energy.device, dtype=torch.long)
        if count == size:
            return torch.arange(size, device=energy.device)
        regions = self._regions(height, width, energy.device)
        coverage = min(
            count, max(len(regions), math.floor(self.coverage_ratio * count))
        )
        quotas = self._quotas([r.numel() for r in regions], coverage)
        key = (height, width, str(energy.device), tuple(quotas))
        if key not in self._batches:
            groups = {}
            for region, quota in zip(regions, quotas):
                if quota:
                    groups.setdefault((region.numel(), quota), []).append(region)
            self._batches[key] = [
                (torch.stack(items), quota) for (_, quota), items in groups.items()
            ]
        mask = torch.zeros(size, device=energy.device, dtype=torch.bool)
        clean = torch.nan_to_num(
            energy.detach().float(), nan=-torch.inf, posinf=torch.inf, neginf=-torch.inf
        )
        for candidates, quota in self._batches[key]:
            if quota == candidates.shape[1]:
                chosen = candidates
            else:
                scores = clean[candidates]
                active = torch.isfinite(scores).any(-1) & (
                    scores.max(-1).values >= 1e-8
                )
                local = torch.topk(scores, quota, dim=-1, sorted=False).indices
                chosen = torch.where(
                    active[:, None], candidates.gather(1, local), candidates[:, :quota]
                )
            mask.scatter_(0, chosen.flatten(), True)
        if count > coverage:
            # Unselected candidates retain the original ascending order.
            order = (
                torch.arange(size, device=energy.device)
                .masked_fill(mask, size)
                .argsort()
            )
            remaining = order[: size - coverage]
            extra = select_candidates(energy, remaining, count - coverage)
            mask[extra] = True
        # Exactly count bits are set, so no dynamic-size nonzero is necessary.
        return mask.to(torch.int32).topk(count).indices.sort().values


class ReferenceTokenSelector:
    def __init__(self, grid_h=4, grid_w=6, anchor_ratio=0.2, residual_weight=0.5):
        self.grid_h, self.grid_w = grid_h, grid_w
        self.anchor_ratio, self.residual_weight = anchor_ratio, residual_weight
        self._layouts = {}

    def select(self, features, count, height, width, residual=None):
        size = features.shape[0]
        if count == 0:
            return torch.empty(0, device=features.device, dtype=torch.long)
        if count == size:
            return torch.arange(size, device=features.device)
        values = features.float()
        contrast = torch.linalg.vector_norm(
            values - values.mean(0, keepdim=True), dim=-1
        )
        contrast = torch.nan_to_num(contrast, nan=0.0, posinf=0.0, neginf=0.0)
        if residual is None:
            score = contrast
        else:
            residual = torch.nan_to_num(
                residual.detach().float(), nan=0.0, posinf=0.0, neginf=0.0
            )
            score = (
                self.residual_weight * residual + (1 - self.residual_weight) * contrast
            )
            score = torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
        key = (height, width, str(score.device))
        if key not in self._layouts:
            regions = []
            for gh in range(self.grid_h):
                h0, h1 = gh * height // self.grid_h, (gh + 1) * height // self.grid_h
                for gw in range(self.grid_w):
                    w0, w1 = gw * width // self.grid_w, (gw + 1) * width // self.grid_w
                    if h1 > h0 and w1 > w0:
                        regions.append(
                            [
                                h * width + w
                                for h in range(h0, h1)
                                for w in range(w0, w1)
                            ]
                        )
            length = max(map(len, regions))
            padded = torch.tensor(
                [r + [0] * (length - len(r)) for r in regions], device=score.device
            )
            lengths = torch.tensor([len(r) for r in regions], device=score.device)
            valid = (
                torch.arange(length, device=score.device)[None, :] < lengths[:, None]
            )
            self._layouts[key] = padded, valid
        padded, valid = self._layouts[key]
        local = score[padded].masked_fill(~valid, -torch.inf).argmax(-1, keepdim=True)
        anchors = padded.gather(1, local).flatten()
        requested = min(
            count,
            max(
                math.floor(self.anchor_ratio * count),
                min(self.grid_h * self.grid_w, count),
            ),
        )
        if anchors.numel() > requested:
            anchors = anchors[
                torch.topk(score[anchors], requested, largest=True).indices
            ]
        anchors = anchors.sort().values  # Disjoint regions already have unique anchors.
        selected = torch.zeros(size, dtype=torch.bool, device=features.device)
        selected[anchors] = True
        left = count - anchors.numel()
        if left:
            fill = torch.topk(
                score.masked_fill(selected, -torch.inf), left, largest=True
            ).indices
            selected[fill] = True
        return selected.to(torch.int32).topk(count).indices.sort().values
