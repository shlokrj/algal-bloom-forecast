"""Metrics for continuous forecast evaluation."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any


def regression_metrics(pairs: Iterable[tuple[Any, Any]]) -> dict[str, float | int | None]:
    """Compute MAE and RMSE over non-null actual/prediction pairs."""
    errors = [
        float(prediction) - float(actual)
        for actual, prediction in pairs
        if actual is not None and prediction is not None
    ]
    if not errors:
        return {"n": 0, "mae": None, "rmse": None}
    return {
        "n": len(errors),
        "mae": sum(abs(error) for error in errors) / len(errors),
        "rmse": math.sqrt(sum(error * error for error in errors) / len(errors)),
    }
