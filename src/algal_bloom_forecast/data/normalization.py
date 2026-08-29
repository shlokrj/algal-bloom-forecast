"""Stable source-normalization contract for the regional forecast table."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

_NORMALIZATION_CONTRACT: dict[str, dict[str, Any]] = {
    "target": {
        "field": "ci_sum",
        "value_semantics": "cross-sensor fused western Lake Erie intensity aggregate",
        "label_observation": "10-day composite center date",
        "horizon_semantics": (
            "predictor cutoff is target observation date minus the requested horizon; "
            "the label is not an exact daily t-plus-horizon observation"
        ),
        "daily_horizon_status": "pending target-product calibration and temporal alignment",
        "timestamp_basis": "date-only fused composite center date; timezone not applicable",
        "unit_status": (
            "NOAA guidance identifies legacy historical Lake Erie CI pixel values as sr^-1 "
            "while current Lake Erie CI/CIcyano products are treated as dimensionless; "
            "the aggregate unit of ci_sum remains unresolved"
        ),
        "unit_reference": (
            "NOAA HAB-F ocean-color processing guidance section 3.1.2 and NOAA's curated "
            "annual Western Lake Erie CI reference; aggregate calibration remains pending"
        ),
        "unit_evidence": (
            "historical MERIS CI pixel values carried legacy sr^-1 labeling, current Lake Erie "
            "CI/CIcyano products are dimensionless, and the source does not define the summed "
            "ci_sum aggregate unit; the curated annual reference describes CI as bloom biomass "
            "and records updated calibrations but does not state a physical CI unit"
        ),
        "curated_annual_reference": (
            "https://nccospublicstor.blob.core.windows.net/hab-data/bulletins/lake-erie/2025/"
            "NOAA_NCCOS_2000to2025_Curated_WLE_Annual_CI.xlsx"
        ),
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
        "quality_policy": (
            "raw *_flags columns are excluded from numeric features; official NCEI metadata "
            "documents 1=pass, 2=not evaluated, 3=suspect, and 4=failed for the audited "
            "moored-buoy accessions; nonnumeric tokens remain unresolved"
        ),
        "quality_flag_references": {
            "0190201": "https://www.ncei.noaa.gov/archive/accession/0190201",
            "0190729": "https://www.ncei.noaa.gov/archive/accession/0190729",
            "0194301": "https://www.ncei.noaa.gov/archive/accession/0194301",
            "0194302": "https://www.ncei.noaa.gov/archive/accession/0194302",
        },
        "quality_flag_mapping": {
            "1": "pass",
            "2": "not evaluated",
            "3": "suspect",
            "4": "failed",
        },
        "quality_flag_mapping_scope": (
            "official NCEI metadata for GLERL annual-summary *_flags fields in the audited "
            "moored-buoy accessions 0190201, 0190729, 0194301, and 0194302"
        ),
        "quality_flag_unmapped_policy": (
            "retain raw flag sequences and do not infer meanings for nonnumeric tokens outside "
            "the documented mapping"
        ),
    },
    "location_policy": {
        "regional_scope": "western Lake Erie",
        "station_features": "retain station-specific predictors; do not spatially interpolate",
        "station_coordinates": "normalize fixed GLERL station metadata in a separate coordinate table",
        "spatial_extension": "deferred until a gridded target and coordinate contract are defined",
    },
}


def build_normalization_contract() -> dict[str, dict[str, Any]]:
    """Return a fresh JSON-safe copy of the normalization contract."""
    return deepcopy(_NORMALIZATION_CONTRACT)
