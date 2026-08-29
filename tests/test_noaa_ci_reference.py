from __future__ import annotations

import zipfile
from pathlib import Path

from algal_bloom_forecast.data.noaa_ci_reference import profile_curated_wle_annual_ci


def _write_fixture(path: Path) -> None:
    workbook = """<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="Annual" r:id="rId1"/><sheet name="Metadata" r:id="rId2"/></sheets></workbook>"""
    relationships = """<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet2.xml"/></Relationships>"""
    annual = """<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData><row r="1"><c r="A1" t="inlineStr"><is><t>year</t></is></c><c r="B1" t="inlineStr"><is><t>WLE_CI_max</t></is></c><c r="C1" t="inlineStr"><is><t>WLE_CI_avg_30d</t></is></c><c r="D1" t="inlineStr"><is><t>WLE_area_km2_max</t></is></c></row><row r="2"><c r="A2"><v>2000</v></c><c r="B2"><v>0.5</v></c><c r="C2"><v>0.4</v></c><c r="D2"><v>12</v></c></row><row r="3"><c r="A3"><v>2025</v></c><c r="B3"><v>4.3</v></c><c r="C3"><v>3.1</v></c><c r="D3"><v>753</v></c></row></sheetData></worksheet>"""
    metadata = """<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData><row r="1"><c r="A1" t="inlineStr"><is><t>File Name:</t></is></c><c r="B1" t="inlineStr"><is><t>reference.xlsx</t></is></c></row><row r="2"><c r="A2" t="inlineStr"><is><t>Dataset Title:</t></is></c><c r="B2" t="inlineStr"><is><t>Annual reference</t></is></c></row><row r="3"><c r="A3" t="inlineStr"><is><t>Description:</t></is></c><c r="B3" t="inlineStr"><is><t>Reprocessed with new calibrations and improved algorithms.</t></is></c></row><row r="4"><c r="A4" t="inlineStr"><is><t>Data:</t></is></c></row><row r="5"><c r="B5" t="inlineStr"><is><t>WLE_CI_max</t></is></c><c r="C5" t="inlineStr"><is><t>Maximum quantity (biomass).</t></is></c></row><row r="6"><c r="B6" t="inlineStr"><is><t>WLE_area_km2_max</t></is></c><c r="C6" t="inlineStr"><is><t>Maximum area (km2).</t></is></c></row></sheetData></worksheet>"""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("xl/workbook.xml", workbook)
        archive.writestr("xl/_rels/workbook.xml.rels", relationships)
        archive.writestr("xl/worksheets/sheet1.xml", annual)
        archive.writestr("xl/worksheets/sheet2.xml", metadata)


def test_profile_curated_wle_annual_ci_preserves_reference_boundary(tmp_path: Path) -> None:
    path = tmp_path / "reference.xlsx"
    _write_fixture(path)

    profile = profile_curated_wle_annual_ci(path)

    assert profile["records"] == 2
    assert profile["observed_years"] == [2000, 2025]
    assert profile["ci_fields"] == ["WLE_CI_max", "WLE_CI_avg_30d"]
    assert profile["area_unit_status"] == "explicit km^2 in workbook metadata"
    assert profile["ci_unit_status"] == "not stated in workbook"
    assert "new calibrations" in profile["calibration_evidence"]
    assert "not interchangeable" in profile["target_interchangeability"]
