from __future__ import annotations

from algal_bloom_forecast.data.normalization import build_normalization_contract


def test_normalization_contract_records_units_time_and_quality_policies() -> None:
    contract = build_normalization_contract()

    assert contract["usgs_maumee"]["model_unit"] == "m^3/s"
    assert "sentinel" in contract["ndbc_45005"]["missing_policy"]
    assert "1=pass" in contract["glerl_continuous"]["quality_policy"]
    assert contract["glerl_continuous"]["quality_flag_mapping"]["3"] == "suspect"
    assert "do not infer meanings" in contract["glerl_continuous"]["quality_flag_unmapped_policy"]
    assert contract["location_policy"]["regional_scope"] == "western Lake Erie"
    assert contract["target"]["label_observation"] == "10-day composite center date"
    assert contract["target"]["daily_horizon_status"].startswith("pending")
    assert "aggregate unit of ci_sum remains unresolved" in contract["target"]["unit_status"]
    assert "historical MERIS CI pixel values" in contract["target"]["unit_evidence"]


def test_normalization_contract_returns_independent_copies() -> None:
    first = build_normalization_contract()
    first["target"]["unit_status"] = "changed"

    second = build_normalization_contract()

    assert second["target"]["unit_status"] != "changed"
