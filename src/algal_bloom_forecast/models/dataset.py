"""Training-frame contracts built without fitting a model."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

from algal_bloom_forecast.features.quality import validate_aligned_records

ID_FIELDS = (
    "split",
    "forecast_horizon_days",
    "observation_date",
    "predictor_date",
)
NON_FEATURE_FIELDS = frozenset(
    {
        *ID_FIELDS,
        "feature_lag_days",
        "predictor_available",
        "bloom_area_sqkm",
    }
)


@dataclass(frozen=True)
class TrainingFrame:
    """Rows and schema metadata for a future model-fitting stage."""

    rows: list[dict[str, Any]]
    feature_names: tuple[str, ...]
    target_name: str
    id_fields: tuple[str, ...]
    excluded_fields: tuple[str, ...]
    missing_rates: dict[str, float]

    @property
    def feature_count(self) -> int:
        return len(self.feature_names)


def _number(value: Any) -> float | None:
    if value in (None, "") or isinstance(value, bool):
        return None if value in (None, "") else float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_date(value: Any, field_name: str) -> date:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be an ISO date string")
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{field_name} must be an ISO date string") from error


def _validate_keys(records: Sequence[Mapping[str, Any]], target_field: str) -> None:
    required = set(ID_FIELDS) | {target_field}
    missing = sorted(required - set().union(*(set(record) for record in records)))
    if missing:
        raise ValueError(f"training frame is missing required fields: {', '.join(missing)}")
    seen: set[tuple[str, int, str]] = set()
    for record in records:
        split = str(record["split"])
        horizon = int(record["forecast_horizon_days"])
        observation_date = _as_date(record["observation_date"], "observation_date")
        key = (split, horizon, observation_date.isoformat())
        if key in seen:
            raise ValueError(f"duplicate training row for {key}")
        seen.add(key)
        if _number(record.get(target_field)) is None:
            raise ValueError(f"training target {target_field} must be non-null for every row")


def build_training_frame(
    records: Sequence[Mapping[str, Any]],
    *,
    target_field: str = "ci_sum",
) -> TrainingFrame:
    """Select numeric predictor columns and preserve audit identifiers.

    This function intentionally performs no imputation, scaling, feature
    selection, or model fitting. Those decisions belong to the train-only
    pipeline after this immutable boundary.
    """
    if not records:
        raise ValueError("cannot build a training frame from zero rows")
    if target_field in NON_FEATURE_FIELDS:
        excluded = set(NON_FEATURE_FIELDS) | {target_field}
    else:
        excluded = set(NON_FEATURE_FIELDS) | {target_field}
    _validate_keys(records, target_field)
    validate_aligned_records(records)

    all_fields = sorted(set().union(*(set(record) for record in records)))
    feature_names: list[str] = []
    non_numeric_fields: set[str] = set()
    for field in all_fields:
        if field in excluded or field.startswith("target_"):
            continue
        values = [record.get(field) for record in records]
        if any(_number(value) is None for value in values if value not in (None, "")):
            non_numeric_fields.add(field)
            continue
        feature_names.append(field)
    if not feature_names:
        raise ValueError("training frame contains no numeric predictor fields")

    output_rows: list[dict[str, Any]] = []
    missing_rates: dict[str, float] = {}
    row_count = len(records)
    for field in feature_names:
        missing_rates[field] = sum(
            value in (None, "") for value in (record.get(field) for record in records)
        ) / row_count
    for record in records:
        output = {field: record.get(field) for field in ID_FIELDS}
        output["target_" + target_field] = record[target_field]
        output.update({field: record.get(field) for field in feature_names})
        output_rows.append(output)
    output_rows.sort(
        key=lambda row: (str(row["split"]), int(row["forecast_horizon_days"]), str(row["observation_date"]))
    )
    return TrainingFrame(
        rows=output_rows,
        feature_names=tuple(feature_names),
        target_name="target_" + target_field,
        id_fields=ID_FIELDS,
        excluded_fields=tuple(sorted(excluded | non_numeric_fields)),
        missing_rates=missing_rates,
    )


def build_training_schema(frame: TrainingFrame) -> dict[str, Any]:
    """Return a JSON-safe schema and coverage summary for an unfitted frame."""
    split_counts = Counter(str(row["split"]) for row in frame.rows)
    horizon_counts = Counter(int(row["forecast_horizon_days"]) for row in frame.rows)
    target_values = [float(row[frame.target_name]) for row in frame.rows]
    return {
        "status": "prepared_not_fitted",
        "target": frame.target_name,
        "id_fields": list(frame.id_fields),
        "feature_count": frame.feature_count,
        "feature_names": list(frame.feature_names),
        "excluded_fields": list(frame.excluded_fields),
        "missing_rate_by_feature": frame.missing_rates,
        "rows": len(frame.rows),
        "split_counts": dict(sorted(split_counts.items())),
        "horizon_counts": {
            str(horizon): count for horizon, count in sorted(horizon_counts.items())
        },
        "target_summary": {
            "count": len(target_values),
            "min": min(target_values),
            "max": max(target_values),
            "mean": sum(target_values) / len(target_values),
        },
        "fit_policy": "fit preprocessing and models on train rows only; validation and test remain held out",
        "missing_value_policy": "preserve nulls for train-only imputation or model-specific missing-value handling",
    }
