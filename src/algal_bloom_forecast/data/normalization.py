"""Stable source-normalization contract for the regional forecast table."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

_NORMALIZATION_CONTRACT: dict[str, dict[str, Any]] = {
    "target": {
        "timestamp_basis": "date-only fused composite center date; timezone not applicable",
        "unit_status": "source unit is not explicitly provided; retain ci_sum metric name",
        "location_scope": "western Lake Erie regional fused series; no coordinate fields",
        "missing_policy": "preserve nulls; no interpolation",
        "quality_policy": "per-record source quality flags are not present in the fused table",
    },
    "usgs_maumee": {
        "source_location": "USGS-04193500",
        "timestamp_basis": "USGS daily-value observation date",
        "source_unit": "ft^3/s",
        "model_unit": "m^3/s",
        "conversion": "multiply by 0.028316846592; retain the source-unit field",
        "quality_policy": "convert the estimated qualifier to usgs_maumee_discharge_estimated_flag",
    },
    "ndbc_45005": {
        "source_location": "NOAA NDBC station 45005",
        "timestamp_basis": "UTC ten-minute records aggregated by UTC calendar day",
        "unit_policy": "retain source-documented standard meteorology units",
        "missing_policy": "recognized NDBC sentinel values become null; retain valid counts",
        "quality_policy": "source sentinel handling is explicit; no interpolation",
    },
    "glerl_continuous": {
        "source_location": "GLERL stations encoded by WE02, WE04, WE08, and WE13 feature prefixes",
        "timestamp_basis": "UTC as documented by annual-summary timestamp rows",
        "unit_policy": "retain the source units row; no cross-sensor unit conversion",
        "missing_policy": "non-numeric and missing values become null; retain valid counts",
        "quality_policy": "raw *_flags columns are excluded from numeric features; flag-code mapping remains pending",
    },
    "location_policy": {
        "regional_scope": "western Lake Erie",
        "station_features": "retain station-specific predictors; do not spatially interpolate",
        "spatial_extension": "deferred until a gridded target and coordinate contract are defined",
    },
}


def build_normalization_contract() -> dict[str, dict[str, Any]]:
    """Return a fresh JSON-safe copy of the normalization contract."""
    return deepcopy(_NORMALIZATION_CONTRACT)
