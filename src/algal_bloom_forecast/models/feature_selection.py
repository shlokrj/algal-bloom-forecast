"""Training-only feature selection rules."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def select_by_train_missing_rate(
    rows: Sequence[Mapping[str, Any]],
    feature_names: Sequence[str],
    *,
    max_missing_rate: float,
) -> tuple[tuple[str, ...], dict[str, float]]:
    """Keep fields whose missingness is within a threshold on train rows only."""
    if not 0.0 <= max_missing_rate <= 1.0:
        raise ValueError("max_missing_rate must be between 0 and 1")
    train_rows = [row for row in rows if str(row.get("split")) == "train"]
    if not train_rows:
        raise ValueError("no train rows available for feature selection")
    rates = {
        field: sum(row.get(field) in (None, "") for row in train_rows) / len(train_rows)
        for field in feature_names
    }
    selected = tuple(sorted(field for field, rate in rates.items() if rate <= max_missing_rate))
    if not selected:
        raise ValueError("missing-rate threshold selected no features")
    return selected, rates
