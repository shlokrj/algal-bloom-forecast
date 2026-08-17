"""USGS daily-values access for the Maumee River source."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
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
