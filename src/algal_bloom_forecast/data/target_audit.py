"""Audit the historical regional target before model comparison."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import date
from itertools import pairwise
from typing import Any


def _parse_date(value: Any) -> date:
    if not isinstance(value, str) or not value:
        raise ValueError("observation_date must be an ISO date string")
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"observation_date must be an ISO date string: {value!r}") from error


def _numeric(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"target value must be numeric or null: {value!r}") from error


def _field_profile(records: Sequence[Mapping[str, Any]], field: str) -> dict[str, Any]:
    values = [_numeric(record.get(field)) for record in records]
    present = [value for value in values if value is not None]
    negative_count = sum(value < 0 for value in present)
    return {
        "records": len(values),
        "present": len(present),
        "missing": len(values) - len(present),
        "missing_rate": (len(values) - len(present)) / len(values) if values else None,
        "minimum": min(present) if present else None,
        "maximum": max(present) if present else None,
        "negative_values": negative_count,
        "nonnegative_values": negative_count == 0,
    }


def audit_target_records(
    records: Sequence[Mapping[str, Any]],
    *,
    target_field: str = "ci_sum",
    auxiliary_fields: Sequence[str] = ("bloom_area_sqkm",),
    expected_interval_days: int = 10,
) -> dict[str, Any]:
    """Return deterministic temporal, missingness, and value-quality diagnostics.

    The historical target is a date-only composite series. This function does not
    interpolate records, infer a timezone, or silently repair source values.
    """
    if not records:
        raise ValueError("cannot audit an empty target table")
    if expected_interval_days <= 0:
        raise ValueError("expected_interval_days must be positive")

    parsed_dates = [_parse_date(record.get("observation_date")) for record in records]
    if len(parsed_dates) != len(set(parsed_dates)):
        raise ValueError("target observation dates must be unique")
    if parsed_dates != sorted(parsed_dates):
        raise ValueError("target observation dates must be sorted")

    intervals = Counter((right - left).days for left, right in pairwise(parsed_dates))
    unexpected_intervals = sum(
        count for interval, count in intervals.items() if interval != expected_interval_days
    )
    fields = [target_field, *auxiliary_fields]
    profiles = {field: _field_profile(records, field) for field in fields}
    years = Counter(str(observation_date.year) for observation_date in parsed_dates)
    return {
        "records": len(records),
        "observed_start": parsed_dates[0].isoformat(),
        "observed_end": parsed_dates[-1].isoformat(),
        "year_counts": dict(sorted(years.items())),
        "date_interval_days": dict(sorted(intervals.items())),
        "expected_interval_days": expected_interval_days,
        "unexpected_interval_records": unexpected_intervals,
        "fields": profiles,
    }
