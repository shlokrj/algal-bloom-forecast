"""Leakage-safe normalization and calendar-window feature engineering."""

from __future__ import annotations

import math
from bisect import bisect_right
from collections.abc import Mapping, Sequence
from datetime import date, timedelta
from typing import Any

USGS_CFS_TO_M3S = 0.028316846592
DEFAULT_LAG_DAYS = (1, 3, 7)
DEFAULT_ROLLING_WINDOWS_DAYS = (3, 7, 14)


def _parse_date(value: Any, *, field_name: str) -> date:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be an ISO date string")
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{field_name} must be an ISO date string: {value!r}") from error


def _as_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _latest_date_at_or_before(dates: list[date], cutoff: date) -> date | None:
    index = bisect_right(dates, cutoff)
    return dates[index - 1] if index else None


def normalize_predictor_records(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Add explicit SI discharge and numeric quality-flag features.

    Original source-unit fields remain unchanged. USGS discharge is additionally
    represented in cubic metres per second; NDBC and GLERL values retain their
    source-documented units and their valid-count fields.
    """
    normalized: list[dict[str, Any]] = []
    seen_dates: set[date] = set()
    for record in records:
        observation_date = _parse_date(
            record.get("observation_date"), field_name="observation_date"
        )
        if observation_date in seen_dates:
            raise ValueError(f"duplicate predictor date: {observation_date.isoformat()}")
        seen_dates.add(observation_date)
        output = dict(record)
        discharge = _as_number(record.get("usgs_maumee_discharge_cfs"))
        output["usgs_maumee_discharge_m3s"] = (
            discharge * USGS_CFS_TO_M3S if discharge is not None else None
        )
        estimated = record.get("usgs_maumee_discharge_estimated")
        if estimated in (True, 1, "True", "true", "ESTIMATED"):
            output["usgs_maumee_discharge_estimated_flag"] = 1
        elif estimated in (False, 0, "False", "false", None, ""):
            output["usgs_maumee_discharge_estimated_flag"] = 0
        else:
            raise ValueError(f"unrecognized USGS quality flag: {estimated!r}")
        normalized.append(output)
    return sorted(normalized, key=lambda record: record["observation_date"])


def _numeric_value_fields(records: Sequence[Mapping[str, Any]]) -> list[str]:
    excluded_suffixes = (
        "_valid_count",
        "_sample_count",
        "_record_count",
        "_station_count",
        "_estimated_flag",
    )

    def is_model_value_field(field: str) -> bool:
        if field.startswith("usgs_"):
            return field in {
                "usgs_maumee_discharge_cfs",
                "usgs_maumee_discharge_m3s",
            }
        if field.startswith("ndbc_"):
            return field.endswith(("_mean", "_circular_mean"))
        if field.startswith("glerl_"):
            return field.endswith(
                (
                    "_air_temp_mean",
                    "_barometric_pressure_mean",
                    "_water_temperature_mean",
                    "_turbidity_mean",
                    "_chlorophylla_mean",
                    "_phycocyanin_mean",
                    "_wind_speed_mean",
                )
            )
        return True

    candidate_fields = {
        field
        for record in records
        for field in record
        if field != "observation_date"
        and not field.endswith(excluded_suffixes)
        and is_model_value_field(field)
    }
    return sorted(
        field
        for field in candidate_fields
        if any(_as_number(record.get(field)) is not None for record in records)
    )


def build_feature_records(
    target_records: Sequence[Mapping[str, Any]],
    predictor_records: Sequence[Mapping[str, Any]],
    *,
    horizons: Sequence[int],
    lag_days: Sequence[int] = DEFAULT_LAG_DAYS,
    rolling_windows_days: Sequence[int] = DEFAULT_ROLLING_WINDOWS_DAYS,
) -> list[dict[str, Any]]:
    """Build target rows with safe snapshots, lags, rolling means, and seasonality."""
    if not horizons or any(horizon <= 0 for horizon in horizons):
        raise ValueError("horizons must contain only positive integers")
    if len(set(horizons)) != len(horizons):
        raise ValueError("horizons must not contain duplicates")
    if any(lag <= 0 for lag in lag_days) or len(set(lag_days)) != len(lag_days):
        raise ValueError("lag_days must contain unique positive integers")
    if any(window <= 0 for window in rolling_windows_days) or len(set(rolling_windows_days)) != len(
        rolling_windows_days
    ):
        raise ValueError("rolling_windows_days must contain unique positive integers")

    normalized_predictors = normalize_predictor_records(predictor_records)
    predictor_by_date = {
        _parse_date(record["observation_date"], field_name="observation_date"): record
        for record in normalized_predictors
    }
    predictor_dates = sorted(predictor_by_date)
    numeric_fields = _numeric_value_fields(normalized_predictors)
    predictor_fields = sorted(
        {
            field
            for record in normalized_predictors
            for field in record
            if field != "observation_date"
        }
    )
    feature_records: list[dict[str, Any]] = []
    for target in target_records:
        target_date = _parse_date(target.get("observation_date"), field_name="observation_date")
        for horizon in horizons:
            cutoff = target_date - timedelta(days=horizon)
            current_date = _latest_date_at_or_before(predictor_dates, cutoff)
            current = predictor_by_date.get(current_date) if current_date else None
            result: dict[str, Any] = {
                "forecast_horizon_days": horizon,
                **target,
                "predictor_date": current_date.isoformat() if current_date else None,
                "feature_lag_days": (target_date - current_date).days if current_date else None,
                "predictor_available": int(current_date is not None),
                "seasonal_day_of_year_sin": math.sin(
                    2 * math.pi * target_date.timetuple().tm_yday / 365.25
                ),
                "seasonal_day_of_year_cos": math.cos(
                    2 * math.pi * target_date.timetuple().tm_yday / 365.25
                ),
            }
            for field in predictor_fields:
                if field in result:
                    raise ValueError(f"target and predictor field collision: {field}")
                result[field] = current.get(field) if current else None

            for lag in lag_days:
                lag_date = _latest_date_at_or_before(predictor_dates, cutoff - timedelta(days=lag))
                lag_record = predictor_by_date.get(lag_date) if lag_date else None
                result[f"lag_{lag}d_predictor_available"] = int(lag_date is not None)
                for field in numeric_fields:
                    result[f"lag_{lag}d_{field}"] = lag_record.get(field) if lag_record else None

            cutoff_index = bisect_right(predictor_dates, cutoff)
            for window in rolling_windows_days:
                window_start = cutoff - timedelta(days=window - 1)
                start_index = bisect_right(predictor_dates, window_start - timedelta(days=1))
                window_dates = predictor_dates[start_index:cutoff_index]
                result[f"rolling_{window}d_predictor_days"] = len(window_dates)
                for field in numeric_fields:
                    values = [
                        value
                        for window_date in window_dates
                        if (value := _as_number(predictor_by_date[window_date].get(field)))
                        is not None
                    ]
                    result[f"rolling_{window}d_{field}_mean"] = (
                        sum(values) / len(values) if values else None
                    )
                    result[f"rolling_{window}d_{field}_valid_count"] = len(values)
            feature_records.append(result)
    return feature_records
