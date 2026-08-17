"""Leakage-safe date alignment for target and daily predictor records."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, timedelta


def _parse_date(value: object, *, field_name: str) -> date:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be an ISO date string")
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{field_name} must be an ISO date string: {value!r}") from error


def _index_unique_records(
    records: Sequence[Mapping[str, object]],
    *,
    date_key: str,
) -> dict[date, Mapping[str, object]]:
    indexed: dict[date, Mapping[str, object]] = {}
    for record in records:
        record_date = _parse_date(record.get(date_key), field_name=date_key)
        if record_date in indexed:
            raise ValueError(f"Duplicate {date_key}: {record_date.isoformat()}")
        indexed[record_date] = record
    return indexed


def align_daily_predictors_to_targets(
    target_records: Sequence[Mapping[str, object]],
    predictor_records: Sequence[Mapping[str, object]],
    *,
    horizon_days: int,
    target_date_key: str = "observation_date",
    predictor_date_key: str = "observation_date",
) -> list[dict[str, object]]:
    """Attach the latest safely available predictor snapshot to each target.

    The predictor cutoff is ``target date - horizon``. Target values remain exactly
    as supplied, including nulls; no target or predictor interpolation occurs.
    """
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive")
    targets = _index_unique_records(target_records, date_key=target_date_key)
    predictors = _index_unique_records(predictor_records, date_key=predictor_date_key)
    predictor_dates = sorted(predictors)
    aligned: list[dict[str, object]] = []
    for target_date in sorted(targets):
        target = targets[target_date]
        cutoff = target_date - timedelta(days=horizon_days)
        eligible_dates = [candidate for candidate in predictor_dates if candidate <= cutoff]
        predictor_date = max(eligible_dates) if eligible_dates else None
        result = dict(target)
        result["predictor_date"] = predictor_date.isoformat() if predictor_date else None
        result["feature_lag_days"] = (target_date - predictor_date).days if predictor_date else None
        if predictor_date:
            for key, value in predictors[predictor_date].items():
                if key == predictor_date_key:
                    continue
                if key in result:
                    raise ValueError(f"Target and predictor field collision: {key}")
                result[key] = value
        aligned.append(result)
    return aligned
