"""Deterministic held-out-year splits for aligned feature records."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any


def _parse_date(value: Any) -> date:
    if not isinstance(value, str) or not value:
        raise ValueError("observation_date must be an ISO date string")
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"observation_date must be an ISO date string: {value!r}") from error


def _parse_horizon(value: Any) -> int:
    try:
        horizon = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError("forecast_horizon_days must be an integer") from error
    if horizon <= 0:
        raise ValueError("forecast_horizon_days must be positive")
    return horizon


def _is_missing(value: Any) -> bool:
    return value in (None, "")


def _is_available(value: Any) -> bool:
    return value in (True, 1, "1", "True", "true")


def build_temporal_splits(
    records: Sequence[Mapping[str, Any]],
    *,
    validation_years: Sequence[int],
    test_years: Sequence[int],
    target_field: str = "ci_sum",
    predictor_available_field: str = "predictor_available",
) -> tuple[list[dict[str, Any]], dict[str, object]]:
    """Split eligible rows by observation year without randomization.

    Rows with a missing target or no eligible predictor are excluded and counted
    in the returned report. All other years become the training split.
    """
    validation = set(validation_years)
    test = set(test_years)
    if validation & test:
        raise ValueError("validation_years and test_years must be disjoint")
    if any(year < 1900 for year in validation | test):
        raise ValueError("split years must be four-digit calendar years")

    seen: set[tuple[int, date]] = set()
    included: list[dict[str, Any]] = []
    excluded: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    year_counts: dict[str, Counter[str]] = {}
    horizon_counts: dict[str, Counter[str]] = {}
    for record in records:
        observation_date = _parse_date(record.get("observation_date"))
        horizon = _parse_horizon(record.get("forecast_horizon_days"))
        key = (horizon, observation_date)
        if key in seen:
            raise ValueError(
                f"duplicate aligned row for horizon {horizon} and date {observation_date.isoformat()}"
            )
        seen.add(key)
        year = observation_date.year
        year_key = str(year)
        if _is_missing(record.get(target_field)):
            excluded["missing_target"] += 1
            year_counts.setdefault(year_key, Counter())["missing_target"] += 1
            continue
        if not _is_available(record.get(predictor_available_field)):
            excluded["missing_predictor"] += 1
            year_counts.setdefault(year_key, Counter())["missing_predictor"] += 1
            continue

        split = "test" if year in test else "validation" if year in validation else "train"
        if "split" in record:
            raise ValueError("input records already contain a split field")
        output = dict(record)
        output["split"] = split
        included.append(output)
        split_counts[split] += 1
        year_counts.setdefault(year_key, Counter())[split] += 1
        horizon_counts.setdefault(str(horizon), Counter())[split] += 1

    report = {
        "rows_total": len(records),
        "rows_included": len(included),
        "rows_excluded": sum(excluded.values()),
        "excluded_by_reason": dict(sorted(excluded.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "year_counts": {
            year: dict(sorted(counts.items())) for year, counts in sorted(year_counts.items())
        },
        "horizon_counts": {
            horizon: dict(sorted(counts.items()))
            for horizon, counts in sorted(horizon_counts.items(), key=lambda item: int(item[0]))
        },
        "validation_years": sorted(validation),
        "test_years": sorted(test),
        "training_rule": "all eligible years not assigned to validation or test",
        "eligibility_rule": f"{target_field} is non-null and {predictor_available_field} is true",
    }
    return included, report
