from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


@dataclass
class CodecBudgetAllocation:
    keep_counts: list[int]
    min_keep_relaxed: bool
    effective_max_keep: int
    total_budget: int


class CodecGuidedBudgetAllocator:
    def __init__(self, *, eps: float = 1e-6) -> None:
        self.eps = float(eps)

    def allocate(
        self,
        complexities: Sequence[float],
        total_budget: int,
        min_keep: int,
        max_keep: int,
        budget_power: float = 0.5,
        max_multiplier: float = 2.0,
    ) -> CodecBudgetAllocation:
        M = len(complexities)
        total_budget = int(total_budget)
        min_keep = max(0, int(min_keep))
        max_keep = max(0, int(max_keep))
        if M == 0:
            return CodecBudgetAllocation([], False, max_keep, total_budget)
        if total_budget < 0:
            raise ValueError(f"total_budget must be >= 0, got {total_budget}")
        if total_budget > M * max_keep:
            raise ValueError(
                f"total_budget={total_budget} exceeds maximum P-frame capacity {M * max_keep}"
            )
        if M == 1:
            return CodecBudgetAllocation(
                [min(max_keep, total_budget)], False, max_keep, total_budget
            )

        clean = []
        for value in complexities:
            if value is None or not math.isfinite(float(value)):
                clean.append(self.eps)
            else:
                clean.append(max(self.eps, float(value)))

        if total_budget < M * min_keep:
            dynamic_min = total_budget // M
            remainder = total_budget - dynamic_min * M
            counts = [dynamic_min for _ in range(M)]
            order = sorted(range(M), key=lambda i: (-clean[i], i))
            for idx in order[:remainder]:
                counts[idx] += 1
            return CodecBudgetAllocation(counts, True, max_keep, total_budget)

        avg = total_budget / max(1, M)
        effective_max = min(max_keep, int(math.ceil(float(max_multiplier) * avg)))
        effective_max = max(min_keep, effective_max)
        if total_budget > M * effective_max:
            effective_max = max_keep
        if total_budget > M * effective_max:
            raise ValueError(
                f"Unable to fit total_budget={total_budget} with effective_max={effective_max}"
            )

        weights = [max(self.eps, c) ** float(budget_power) for c in clean]
        counts = self._bounded_largest_remainder(
            total_budget=total_budget,
            min_keep=min_keep,
            max_keep=effective_max,
            weights=weights,
        )
        if sum(counts) != total_budget:
            raise AssertionError(
                f"Codec budget mismatch: sum={sum(counts)} total={total_budget}"
            )
        if any(c < 0 or c > max_keep for c in counts):
            raise AssertionError(
                f"Codec budget out of range: counts={counts}, max_keep={max_keep}"
            )
        return CodecBudgetAllocation(counts, False, effective_max, total_budget)

    def allocate_counts(
        self,
        complexities: Sequence[float],
        total_budget: int,
        min_keep: int,
        max_keep: int,
        budget_power: float = 0.5,
        max_multiplier: float = 2.0,
    ) -> list[int]:
        return self.allocate(
            complexities,
            total_budget=total_budget,
            min_keep=min_keep,
            max_keep=max_keep,
            budget_power=budget_power,
            max_multiplier=max_multiplier,
        ).keep_counts

    def _bounded_largest_remainder(
        self,
        *,
        total_budget: int,
        min_keep: int,
        max_keep: int,
        weights: Sequence[float],
    ) -> list[int]:
        M = len(weights)
        counts = [int(min_keep) for _ in range(M)]
        remaining = int(total_budget) - int(min_keep) * M
        if remaining == 0:
            return counts

        capacities = [int(max_keep) - int(min_keep) for _ in range(M)]
        float_extra = [0.0 for _ in range(M)]
        active = {i for i, cap in enumerate(capacities) if cap > 0}
        remaining_float = float(remaining)
        while active and remaining_float > self.eps:
            weight_sum = sum(max(self.eps, float(weights[i])) for i in active)
            proposed = {
                i: remaining_float * max(self.eps, float(weights[i])) / weight_sum
                for i in active
            }
            saturated = [
                i for i, value in proposed.items() if value >= capacities[i] - self.eps
            ]
            if not saturated:
                for i, value in proposed.items():
                    float_extra[i] = value
                break
            for i in sorted(saturated):
                float_extra[i] = float(capacities[i])
                remaining_float -= float(capacities[i])
                active.remove(i)

        extras = [min(capacities[i], int(math.floor(float_extra[i]))) for i in range(M)]
        leftover = remaining - sum(extras)
        order = sorted(
            range(M),
            key=lambda i: (-(float_extra[i] - math.floor(float_extra[i])), i),
        )
        for idx in order:
            if leftover == 0:
                break
            if extras[idx] < capacities[idx]:
                extras[idx] += 1
                leftover -= 1
        if leftover != 0:
            raise AssertionError(
                f"Unable to distribute codec budget leftover={leftover}"
            )
        return [counts[i] + extras[i] for i in range(M)]
