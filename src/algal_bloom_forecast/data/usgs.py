"""USGS daily-values access for the Maumee River source."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

DAILY_VALUES_URL = "https://api.waterdata.usgs.gov/ogcapi/v0/collections/daily/items"


@dataclass(frozen=True)
class DailyValuesQuery:
    """Filters for one USGS daily-values request."""

    monitoring_location_id: str
    parameter_code: str
    statistic_id: str
    start_date: str
    end_date: str
    limit: int = 10_000


def build_daily_values_url(query: DailyValuesQuery) -> str:
    """Build a deterministic JSON request URL."""
    params = {
        "f": "json",
        "monitoring_location_id": query.monitoring_location_id,
        "parameter_code": query.parameter_code,
        "statistic_id": query.statistic_id,
        "datetime": f"{query.start_date}/{query.end_date}",
        "limit": str(query.limit),
    }
    return f"{DAILY_VALUES_URL}?{urlencode(params)}"


def fetch_daily_values(
    query: DailyValuesQuery,
    *,
    output_path: Path | None = None,
    timeout_seconds: int = 60,
) -> tuple[dict, str]:
    """Fetch one complete daily-values page and optionally preserve its JSON bytes.

    The caller should use a date range that fits within the API limit. A pagination
    link causes an error instead of silently truncating the source record.
    """
    url = build_daily_values_url(query)
    request = Request(url, headers={"User-Agent": "algal-bloom-forecast/0.1"})
    with urlopen(request, timeout=timeout_seconds) as response:
        raw = response.read()

    payload = json.loads(raw)
    if any(link.get("rel") == "next" for link in payload.get("links", [])):
        raise ValueError("USGS response is paginated; reduce the requested date range")

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(raw)

    return payload, url


def _parse_discharge(value: Any) -> float | None:
    if value is None or str(value).strip().lower() in {"", "na", "nan", "n/a", "null"}:
        return None
    return float(value)


def parse_daily_values_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize a USGS daily-values feature collection into date-keyed records."""
    records: list[dict[str, Any]] = []
    seen_dates: set[str] = set()
    for feature in payload.get("features", []):
        properties = feature.get("properties", {})
        observation_date = str(properties.get("time", ""))
        try:
            date.fromisoformat(observation_date)
        except ValueError as error:
            raise ValueError(f"USGS record has an invalid date: {observation_date!r}") from error
        if observation_date in seen_dates:
            raise ValueError(f"USGS response contains duplicate date: {observation_date}")
        seen_dates.add(observation_date)
        qualifiers = properties.get("qualifier") or []
        if isinstance(qualifiers, str):
            qualifiers = [qualifiers]
        records.append(
            {
                "observation_date": observation_date,
                "usgs_maumee_discharge_cfs": _parse_discharge(properties.get("value")),
                "usgs_maumee_discharge_estimated": "ESTIMATED" in qualifiers,
            }
        )
    return sorted(records, key=lambda record: record["observation_date"])
