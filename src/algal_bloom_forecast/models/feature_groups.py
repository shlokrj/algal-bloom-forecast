"""Explicit feature groups for one-factor linear ablation studies."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from algal_bloom_forecast.models.baselines import LINEAR_FEATURES

NDBC_WEATHER_FEATURES = (
    "ndbc_wspd_mean",
    "ndbc_gst_mean",
    "ndbc_wvht_mean",
    "ndbc_atmp_mean",
    "ndbc_wtmp_mean",
    "ndbc_pres_mean",
)
GLERL_WATER_QUALITY_SUFFIXES = (
    "_water_temperature_mean",
    "_turbidity_mean",
    "_chlorophylla_mean",
    "_phycocyanin_mean",
)


def _available_fields(records: Sequence[Mapping[str, Any]]) -> set[str]:
    return {field for record in records for field in record}


def build_feature_groups(records: Sequence[Mapping[str, Any]]) -> dict[str, tuple[str, ...]]:
    """Return reproducible feature groups limited to fields present in the table."""
    available = _available_fields(records)
    seasonal = tuple(field for field in LINEAR_FEATURES if field.startswith("seasonal_"))
    discharge = tuple(field for field in LINEAR_FEATURES if field.startswith("usgs_"))
    buoy = tuple(field for field in NDBC_WEATHER_FEATURES if field in available)
    water_quality = tuple(
        field
        for field in sorted(available)
        if field.startswith("glerl_") and field.endswith(GLERL_WATER_QUALITY_SUFFIXES)
    )
    groups = {
        "seasonal_only": seasonal,
        "discharge_seasonal": discharge + seasonal,
        "buoy_weather": discharge + seasonal + buoy,
        "water_quality": discharge + seasonal + water_quality,
        "combined_core": discharge + seasonal + buoy + water_quality,
    }
    if any(not fields for fields in groups.values()):
        raise ValueError("feature table does not contain fields required for ablation groups")
    return groups
