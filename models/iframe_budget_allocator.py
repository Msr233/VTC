from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


@dataclass
class IFrameBudgetAllocation:
    keep_counts: list[int]
    release_counts: list[int]
    p_budget_old: int
    p_budget_new: int
    released_tokens: int
    skipped_reason: str | None = None
    warning: str | None = None


class IFrameBudgetAllocator:
    """Allocate budgets for lightweight semantic I-frame compression."""

    VALID_MODES = {"none", "lightweight"}

    def allocate(
        self,
        *,
        mode: str,
        tokens_per_frame: int,
        is_first_iframe: Sequence[bool],
        num_p_frames: int,
        p_budget_old: int,
        overall_keep_ratio: float,
        compress_when_ratio_one: bool = False,
        keep_ratio: float = 0.9,
        first_keep_ratio: float | None = None,
        dynamic_keep_ratio: float | None = None,
        min_keep_ratio: float = 0.8,
        adaptive_p_target_ratio: float = 0.25,
        adaptive_max_release_ratio: float = 0.15,
    ) -> IFrameBudgetAllocation:
        mode = str(mode)
        if mode not in self.VALID_MODES:
            raise ValueError(f"Unknown I-frame compression mode: {mode!r}")
        N = int(tokens_per_frame)
        if N < 0:
            raise ValueError(f"tokens_per_frame must be >= 0, got {N}")
        flags = [bool(x) for x in is_first_iframe]
        num_i = len(flags)
        M = int(num_p_frames)
        p_budget_old = int(p_budget_old)
        if p_budget_old < 0:
            raise ValueError(f"p_budget_old must be >= 0, got {p_budget_old}")

        self._validate_ratio("keep_ratio", keep_ratio)
        self._validate_optional_ratio("first_keep_ratio", first_keep_ratio)
        self._validate_optional_ratio("dynamic_keep_ratio", dynamic_keep_ratio)
        self._validate_ratio("min_keep_ratio", min_keep_ratio)
        self._validate_ratio("adaptive_p_target_ratio", adaptive_p_target_ratio)
        self._validate_ratio("adaptive_max_release_ratio", adaptive_max_release_ratio)

        full = [N for _ in flags]
        if mode == "none" or num_i == 0:
            return IFrameBudgetAllocation(full, [0 for _ in flags], p_budget_old, p_budget_old, 0)
        if float(overall_keep_ratio) >= 1.0 and not bool(compress_when_ratio_one):
            return IFrameBudgetAllocation(
                full,
                [0 for _ in flags],
                p_budget_old,
                p_budget_old,
                0,
                skipped_reason="ratio_one_requires_explicit_iframe_compress",
            )
        if M == 0:
            return IFrameBudgetAllocation(
                full,
                [0 for _ in flags],
                p_budget_old,
                p_budget_old,
                0,
                skipped_reason="no_p_frames_in_chunk",
            )

        min_keep = min(N, max(0, int(math.ceil(float(min_keep_ratio) * N))))
        max_p_budget = M * N
        p_capacity = max(0, max_p_budget - p_budget_old)

        if mode == "lightweight":
            desired_keep = []
            for is_first in flags:
                rho = (
                    first_keep_ratio if is_first and first_keep_ratio is not None
                    else dynamic_keep_ratio if (not is_first) and dynamic_keep_ratio is not None
                    else keep_ratio
                )
                k = int(math.ceil(float(rho) * N))
                desired_keep.append(min(N, max(min_keep, k)))
            desired_release = [N - k for k in desired_keep]
            allow_unreallocated_release = (
                float(overall_keep_ratio) >= 1.0
                and bool(compress_when_ratio_one)
                and p_capacity <= 0
            )
            if allow_unreallocated_release:
                releases = desired_release
                warning = "ratio_one_iframe_compression_drops_below_full_budget"
                p_budget_new = p_budget_old
            else:
                releases, warning = self._cap_releases(desired_release, p_capacity)
                p_budget_new = p_budget_old + sum(releases)
            keep_counts = [N - r for r in releases]
            return IFrameBudgetAllocation(
                keep_counts,
                releases,
                p_budget_old,
                p_budget_new,
                sum(releases),
                warning=warning,
            )

        return self._allocate_adaptive(
            N=N,
            flags=flags,
            M=M,
            p_budget_old=p_budget_old,
            p_capacity=p_capacity,
            min_keep=min_keep,
            adaptive_p_target_ratio=float(adaptive_p_target_ratio),
            adaptive_max_release_ratio=float(adaptive_max_release_ratio),
        )

    def _allocate_adaptive(
        self,
        *,
        N: int,
        flags: Sequence[bool],
        M: int,
        p_budget_old: int,
        p_capacity: int,
        min_keep: int,
        adaptive_p_target_ratio: float,
        adaptive_max_release_ratio: float,
    ) -> IFrameBudgetAllocation:
        p_avg_old = p_budget_old / max(1, M)
        p_target = adaptive_p_target_ratio * N
        if p_avg_old >= p_target:
            full = [N for _ in flags]
            return IFrameBudgetAllocation(
                full,
                [0 for _ in flags],
                p_budget_old,
                p_budget_old,
                0,
                skipped_reason="p_budget_already_sufficient",
            )

        need = int(math.ceil(M * p_target - p_budget_old))
        per_iframe_cap = min(
            max(0, N - min_keep),
            max(0, int(math.floor(adaptive_max_release_ratio * N))),
        )
        caps = [per_iframe_cap for _ in flags]
        total_release = min(max(0, need), sum(caps), max(0, p_capacity))
        releases = self._weighted_largest_remainder(
            total_release,
            weights=[0.5 if is_first else 1.0 for is_first in flags],
            caps=caps,
        )
        warning = None
        if total_release < need:
            warning = (
                "adaptive_release_capped:"
                f"need={need},release={total_release},capacity={p_capacity},iframe_cap={sum(caps)}"
            )
        return IFrameBudgetAllocation(
            [N - r for r in releases],
            releases,
            p_budget_old,
            p_budget_old + sum(releases),
            sum(releases),
            warning=warning,
        )

    def _cap_releases(self, desired: Sequence[int], capacity: int) -> tuple[list[int], str | None]:
        desired = [max(0, int(x)) for x in desired]
        total = sum(desired)
        if total <= capacity:
            return desired, None
        releases = self._weighted_largest_remainder(
            max(0, int(capacity)),
            weights=[float(x) for x in desired],
            caps=desired,
        )
        return releases, f"iframe_release_capped_by_p_capacity:desired={total},capacity={capacity}"

    def _weighted_largest_remainder(
        self,
        total: int,
        *,
        weights: Sequence[float],
        caps: Sequence[int],
    ) -> list[int]:
        total = int(total)
        caps = [max(0, int(c)) for c in caps]
        if total <= 0 or not caps:
            return [0 for _ in caps]
        if total > sum(caps):
            raise ValueError(f"total release {total} exceeds caps {sum(caps)}")

        clean_weights = [
            max(0.0, float(w)) if cap > 0 else 0.0
            for w, cap in zip(weights, caps)
        ]
        if sum(clean_weights) <= 0.0:
            clean_weights = [1.0 if cap > 0 else 0.0 for cap in caps]

        releases = [0 for _ in caps]
        remaining = total
        active = {i for i, cap in enumerate(caps) if cap > 0}
        fractions = [0.0 for _ in caps]

        while remaining > 0 and active:
            weight_sum = sum(clean_weights[i] for i in active)
            if weight_sum <= 0.0:
                weight_sum = float(len(active))
                clean_weights = [
                    1.0 if i in active else clean_weights[i]
                    for i in range(len(caps))
                ]
            saturated = []
            proposals = {}
            for i in active:
                value = remaining * clean_weights[i] / weight_sum
                proposals[i] = value
                if value >= caps[i] - releases[i]:
                    saturated.append(i)
            if not saturated:
                for i, value in proposals.items():
                    add = int(math.floor(value))
                    releases[i] += min(caps[i] - releases[i], add)
                    fractions[i] = value - add
                break
            for i in sorted(saturated):
                add = caps[i] - releases[i]
                releases[i] += add
                remaining -= add
                active.remove(i)

        leftover = total - sum(releases)
        order = sorted(range(len(caps)), key=lambda i: (-fractions[i], i))
        for i in order:
            if leftover == 0:
                break
            if releases[i] < caps[i]:
                releases[i] += 1
                leftover -= 1
        if leftover != 0:
            for i in range(len(caps)):
                if leftover == 0:
                    break
                room = caps[i] - releases[i]
                take = min(room, leftover)
                releases[i] += take
                leftover -= take
        if sum(releases) != total:
            raise AssertionError(f"I-frame release mismatch: sum={sum(releases)} total={total}")
        return releases

    def _validate_optional_ratio(self, name: str, value: float | None) -> None:
        if value is not None:
            self._validate_ratio(name, value)

    def _validate_ratio(self, name: str, value: float) -> None:
        value = float(value)
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {value!r}")
