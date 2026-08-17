"""Combine date-keyed predictor snapshots without filling missing values."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def merge_daily_predictor_records(
    sources: Mapping[str, Iterable[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Outer-join source snapshots by date and reject duplicate/colliding fields."""
    by_date: dict[str, dict[str, Any]] = {}
    for source_name, records in sources.items():
        seen_dates: set[str] = set()
        for record in records:
            observation_date = record.get("observation_date")
            if not isinstance(observation_date, str) or not observation_date:
                raise ValueError(f"{source_name} record has no observation_date")
            if observation_date in seen_dates:
                raise ValueError(f"{source_name} contains duplicate date: {observation_date}")
            seen_dates.add(observation_date)
            target = by_date.setdefault(observation_date, {"observation_date": observation_date})
            for field, value in record.items():
                if field == "observation_date":
                    continue
                if field in target:
                    raise ValueError(f"predictor field collision for {field!r} from {source_name}")
                target[field] = value

    fields = sorted(
        {field for record in by_date.values() for field in record if field != "observation_date"}
    )
    return [
        {
            "observation_date": observation_date,
            **{field: by_date[observation_date].get(field) for field in fields},
        }
        for observation_date in sorted(by_date)
    ]
