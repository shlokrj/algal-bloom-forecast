import gzip
from pathlib import Path
import tempfile
import unittest

from algal_bloom_forecast.data.ndbc import (
    build_standard_meteorology_url,
    parse_standard_meteorology,
)
from algal_bloom_forecast.data.usgs import DailyValuesQuery, build_daily_values_url


class DataSourceTests(unittest.TestCase):
    def test_usgs_url_contains_explicit_filters(self):
        query = DailyValuesQuery(
            monitoring_location_id="USGS-04193500",
            parameter_code="00060",
            statistic_id="00003",
            start_date="2012-01-01",
            end_date="2025-12-31",
        )
        url = build_daily_values_url(query)
        self.assertIn("monitoring_location_id=USGS-04193500", url)
        self.assertIn("parameter_code=00060", url)
        self.assertIn("datetime=2012-01-01%2F2025-12-31", url)

    def test_ndbc_url_is_year_specific(self):
        self.assertEqual(
            build_standard_meteorology_url("45005", 2024),
            "https://www.ndbc.noaa.gov/data/historical/stdmet/45005h2024.txt.gz",
        )

    def test_ndbc_parser_uses_utc_and_nulls_sentinels(self):
        content = (
            "#YY MM DD hh mm WDIR WSPD GST WVHT DPD APD MWD PRES ATMP WTMP DEWP VIS TIDE\n"
            "#yr mo dy hr mn degT m/s m/s m sec sec degT hPa degC degC degC mi ft\n"
            "2024 04 15 16 40 999 3.4 99.0 0.00 99.00 4.40 999 1038.6 24.7 99.0 17.6 99.0 99.00\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            file_path = Path(directory) / "sample.txt.gz"
            with gzip.open(file_path, "wt", encoding="utf-8") as handle:
                handle.write(content)
            records = parse_standard_meteorology(file_path)

        self.assertEqual(records[0]["timestamp"], "2024-04-15T16:40:00+00:00")
        self.assertIsNone(records[0]["WDIR"])
        self.assertEqual(records[0]["WSPD"], 3.4)
        self.assertIsNone(records[0]["WTMP"])
