"""Profile NOAA's curated annual Western Lake Erie CI reference workbook."""

from __future__ import annotations

import posixpath
import re
import zipfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

CURATED_WLE_ANNUAL_CI_URL = (
    "https://nccospublicstor.blob.core.windows.net/hab-data/bulletins/lake-erie/2025/"
    "NOAA_NCCOS_2000to2025_Curated_WLE_Annual_CI.xlsx"
)
CURATED_WLE_ANNUAL_CI_FILENAME = "NOAA_NCCOS_2000to2025_Curated_WLE_Annual_CI.xlsx"

_MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_PACKAGE_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
_CELL_REF = re.compile(r"^([A-Z]+)")


def _xml(zfile: zipfile.ZipFile, name: str) -> ElementTree.Element:
    try:
        return ElementTree.fromstring(zfile.read(name))
    except KeyError as exc:
        raise ValueError(f"missing required workbook part: {name}") from exc


def _shared_strings(zfile: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in zfile.namelist():
        return []
    root = _xml(zfile, "xl/sharedStrings.xml")
    values: list[str] = []
    for item in root.findall(f"{{{_MAIN_NS}}}si"):
        values.append("".join(text.text or "" for text in item.iter(f"{{{_MAIN_NS}}}t")))
    return values


def _sheet_paths(zfile: zipfile.ZipFile) -> dict[str, str]:
    workbook = _xml(zfile, "xl/workbook.xml")
    relationships = _xml(zfile, "xl/_rels/workbook.xml.rels")
    targets = {
        relationship.attrib["Id"]: relationship.attrib["Target"]
        for relationship in relationships.findall(f"{{{_PACKAGE_REL_NS}}}Relationship")
    }
    sheets: dict[str, str] = {}
    for sheet in workbook.findall(f".//{{{_MAIN_NS}}}sheet"):
        relationship_id = sheet.attrib.get(f"{{{_REL_NS}}}id")
        if relationship_id not in targets:
            raise ValueError(f"missing relationship for workbook sheet {sheet.attrib.get('name')}")
        target = targets[relationship_id]
        sheets[sheet.attrib["name"]] = posixpath.normpath(
            target if target.startswith("/") else f"xl/{target}"
        ).lstrip("/")
    return sheets


def _column_index(cell_reference: str) -> int:
    match = _CELL_REF.match(cell_reference)
    if not match:
        raise ValueError(f"invalid worksheet cell reference: {cell_reference}")
    index = 0
    for character in match.group(1):
        index = index * 26 + ord(character) - ord("A") + 1
    return index


def _cell_value(cell: ElementTree.Element, shared_strings: list[str]) -> str:
    value = cell.find(f"{{{_MAIN_NS}}}v")
    raw = "" if value is None or value.text is None else value.text
    if cell.attrib.get("t") == "s":
        try:
            return shared_strings[int(raw)]
        except (IndexError, ValueError) as exc:
            raise ValueError(f"invalid shared-string index in cell {cell.attrib.get('r')}") from exc
    if cell.attrib.get("t") == "inlineStr":
        inline = cell.find(f"{{{_MAIN_NS}}}is")
        return "" if inline is None else "".join(
            text.text or "" for text in inline.iter(f"{{{_MAIN_NS}}}t")
        )
    return raw


def _sheet_rows(
    zfile: zipfile.ZipFile, sheet_path: str, shared_strings: list[str]
) -> list[dict[int, str]]:
    root = _xml(zfile, sheet_path)
    rows: list[dict[int, str]] = []
    for row in root.findall(f".//{{{_MAIN_NS}}}row"):
        values: dict[int, str] = {}
        for cell in row.findall(f"{{{_MAIN_NS}}}c"):
            values[_column_index(cell.attrib["r"])] = _cell_value(cell, shared_strings)
        rows.append(values)
    return rows


def _row_values(row: dict[int, str], width: int | None = None) -> list[str]:
    max_column = width or max(row, default=0)
    return [row.get(column, "") for column in range(1, max_column + 1)]


def _metadata_value(rows: list[dict[int, str]], label: str) -> str | None:
    for row in rows:
        values = _row_values(row)
        if values and values[0].strip() == label:
            return values[1].strip() if len(values) > 1 else None
    return None


def _as_year(value: str) -> int:
    try:
        year = float(value)
    except ValueError as exc:
        raise ValueError(f"annual CI workbook has a nonnumeric year: {value!r}") from exc
    if not year.is_integer():
        raise ValueError(f"annual CI workbook has a fractional year: {value!r}")
    return int(year)


def profile_curated_wle_annual_ci(path: Path) -> dict[str, Any]:
    """Summarize the workbook without treating it as the historical 10-day target."""
    with zipfile.ZipFile(path) as workbook:
        shared_strings = _shared_strings(workbook)
        sheets = _sheet_paths(workbook)
        data_sheet_name = next(
            (name for name in sheets if name != "Metadata"),
            None,
        )
        if data_sheet_name is None or "Metadata" not in sheets:
            raise ValueError("curated annual CI workbook must contain data and Metadata sheets")
        data_rows = _sheet_rows(workbook, sheets[data_sheet_name], shared_strings)
        metadata_rows = _sheet_rows(workbook, sheets["Metadata"], shared_strings)

    if not data_rows:
        raise ValueError("curated annual CI workbook data sheet is empty")
    headers = _row_values(data_rows[0])
    if not {"year", "WLE_CI_max", "WLE_CI_avg_30d"}.issubset(headers):
        raise ValueError(f"curated annual CI workbook has unexpected headers: {headers}")
    year_index = headers.index("year") + 1
    years = [_as_year(row[year_index]) for row in data_rows[1:] if row.get(year_index, "")]
    if years != sorted(years) or len(years) != len(set(years)):
        raise ValueError("curated annual CI workbook years are not sorted and unique")

    metadata = {
        label.rstrip(":"): _metadata_value(metadata_rows, f"{label}:")
        for label in ("File Name", "Dataset Title", "Description", "Attribution")
    }
    data_descriptions: dict[str, str] = {}
    data_section_seen = False
    for row in metadata_rows:
        values = _row_values(row, width=3)
        if values[0].strip() == "Data:":
            data_section_seen = True
            continue
        if data_section_seen and values[1].strip() and values[2].strip():
            data_descriptions[values[1].strip()] = values[2].strip()

    description = metadata["Description"] or ""
    all_data_text = " ".join(data_descriptions.values())
    calibration_evidence = (
        "workbook description records reprocessing after the 2024 Microcystis bloom with new "
        "calibrations and improved algorithms"
        if "new calibrations" in description.lower() and "improved algorithms" in description.lower()
        else "calibration update language was not found in the workbook description"
    )
    return {
        "source_filename": path.name,
        "workbook_sheets": list(sheets),
        "data_sheet": data_sheet_name,
        "fields": headers,
        "records": len(years),
        "observed_years": years,
        "observed_start_year": years[0] if years else None,
        "observed_end_year": years[-1] if years else None,
        "ci_fields": [field for field in headers if "ci" in field.lower()],
        "area_fields": [field for field in headers if "area" in field.lower()],
        "metadata": metadata,
        "data_descriptions": data_descriptions,
        "ci_unit_status": "not stated in workbook",
        "area_unit_status": (
            "explicit km^2 in workbook metadata"
            if "km" in all_data_text.lower()
            else "area unit not confirmed in workbook metadata"
        ),
        "calibration_evidence": calibration_evidence,
        "target_interchangeability": (
            "reference only; annual CI metrics are not interchangeable with historical "
            "10-day fused ci_sum"
        ),
    }
