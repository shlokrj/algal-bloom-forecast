from __future__ import annotations

from pathlib import Path

from scripts.build_feature_table import ROOT, _resolve


def test_feature_table_resolves_relative_input_paths() -> None:
    assert _resolve(Path("data/processed/input.csv")) == ROOT / "data/processed/input.csv"


def test_feature_table_preserves_absolute_input_paths() -> None:
    absolute = Path("/tmp/input.csv")

    assert _resolve(absolute) == absolute
