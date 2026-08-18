"""Validation-only candidate selection utilities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def _number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def select_by_validation_mae(
    rows: Sequence[Mapping[str, Any]],
    *,
    validation_split: str = "validation",
) -> dict[int, str]:
    """Select the lowest-validation-MAE candidate independently per horizon."""
    grouped: dict[int, list[Mapping[str, Any]]] = {}
    for row in rows:
        if str(row.get("split")) != validation_split:
            continue
        mae = _number(row.get("mae"))
        if mae is None:
            continue
        horizon = int(float(row["forecast_horizon_days"]))
        grouped.setdefault(horizon, []).append(row)
    if not grouped:
        raise ValueError("no validation metrics available for model selection")
    return {
        horizon: str(
            min(
                candidates,
                key=lambda row: (_number(row["mae"]), str(row["candidate"])),
            )["candidate"]
        )
        for horizon, candidates in sorted(grouped.items())
    }
