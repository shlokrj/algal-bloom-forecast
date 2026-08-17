"""Validation and missingness profiling for aligned research tables."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import date, timedelta
from typing import Any


def _as_date(value: Any, *, field_name: str) -> date:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be an ISO date string")
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{field_name} must be an ISO date string: {value!r}") from error


def _as_int(value: Any, *, field_name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must be an integer") from error


def validate_aligned_records(records: Sequence[Mapping[str, Any]]) -> None:
    """Reject duplicate horizon/date rows and any predictor cutoff violation."""
    seen: set[tuple[int, date]] = set()
    for record in records:
        horizon = _as_int(record.get("forecast_horizon_days"), field_name="forecast_horizon_days")
        if horizon <= 0:
            raise ValueError("forecast_horizon_days must be positive")
        target_date = _as_date(record.get("observation_date"), field_name="observation_date")
        key = (horizon, target_date)
        if key in seen:
            raise ValueError(
                f"duplicate aligned row for horizon {horizon} and date {target_date.isoformat()}"
            )
        seen.add(key)

        predictor_value = record.get("predictor_date")
        lag_value = record.get("feature_lag_days")
        if predictor_value in (None, ""):
            if lag_value not in (None, ""):
                raise ValueError("feature_lag_days must be empty when predictor_date is empty")
            continue
        predictor_date = _as_date(predictor_value, field_name="predictor_date")
        cutoff = target_date - timedelta(days=horizon)
        if predictor_date > cutoff:
            raise ValueError(
                f"predictor date {predictor_date.isoformat()} is after cutoff {cutoff.isoformat()}"
            )
        lag = _as_int(lag_value, field_name="feature_lag_days")
        expected_lag = (target_date - predictor_date).days
        if lag != expected_lag:
            raise ValueError(f"feature_lag_days {lag} disagrees with date lag {expected_lag}")


def profile_aligned_records(records: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    """Return deterministic row, horizon, coverage, and field-missingness counts."""
    validate_aligned_records(records)
    horizon_counts: Counter[str] = Counter()
    predictor_matches: Counter[str] = Counter()
    lag_values: dict[str, list[int]] = {}
    field_counts: dict[str, dict[str, int]] = {}
    for record in records:
        horizon = str(record["forecast_horizon_days"])
        horizon_counts[horizon] += 1
        if record.get("predictor_date") not in (None, ""):
            predictor_matches[horizon] += 1
            lag_values.setdefault(horizon, []).append(int(record["feature_lag_days"]))
        for field, value in record.items():
            stats = field_counts.setdefault(field, {"records": 0, "non_null": 0, "missing": 0})
            stats["records"] += 1
            if value in (None, ""):
                stats["missing"] += 1
            else:
                stats["non_null"] += 1

    horizon_profiles: dict[str, dict[str, int | None]] = {}
    for horizon in sorted(horizon_counts, key=int):
        lags = lag_values.get(horizon, [])
        horizon_profiles[horizon] = {
            "records": horizon_counts[horizon],
            "predictor_matches": predictor_matches[horizon],
            "predictor_missing": horizon_counts[horizon] - predictor_matches[horizon],
            "feature_lag_min_days": min(lags) if lags else None,
            "feature_lag_max_days": max(lags) if lags else None,
        }
    return {
        "records": len(records),
        "horizons": horizon_profiles,
        "fields": dict(sorted(field_counts.items())),
    }
