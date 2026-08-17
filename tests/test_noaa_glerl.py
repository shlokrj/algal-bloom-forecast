import tempfile
import unittest
from pathlib import Path

from algal_bloom_forecast.data.noaa_glerl import (
    parse_fluoroprobe_coordinates,
    parse_fluoroprobe_dictionary,
    parse_ftp_listing,
    profile_fluoroprobe_csv,
)


class NoaaGlerlTests(unittest.TestCase):
    def test_ftp_listing_keeps_files_and_sizes(self):
        listing = (
            "-rw-rw-r--    1 0 0 123 Jan 01 2025 sample.csv\n"
            "drwxrwxr-x    2 0 0 4096 Jan 01 2025 profiles\n"
        )
        files = parse_ftp_listing(listing, base_url="ftp://example.test/data/")
        self.assertEqual(len(files), 1)
        self.assertEqual(files[0].name, "sample.csv")
        self.assertEqual(files[0].size_bytes, 123)
        self.assertEqual(files[0].url, "ftp://example.test/data/sample.csv")

    def test_dictionary_and_coordinates_ignore_blank_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            dictionary_path = Path(directory) / "dictionary.csv"
            dictionary_path.write_text(
                """measurement_numb,The number of the measurement collected by the fluoroprobe in acsending order with the downcast.
datetime,Date and time
""",
                encoding="utf-8",
            )
            coordinates_path = Path(directory) / "coordinates.csv"
            coordinates_path.write_text(
                """station,lat,long,,,,
WE12,41.703,-83.254,,,,
,,,,,,
""",
                encoding="utf-8",
            )
            dictionary = parse_fluoroprobe_dictionary(dictionary_path)
            coordinates = parse_fluoroprobe_coordinates(coordinates_path)

        self.assertEqual(dictionary["datetime"], "Date and time")
        self.assertEqual(coordinates, [{"station": "WE12", "latitude": 41.703, "longitude": -83.254}])

    def test_profile_preserves_documented_local_time_basis(self):
        content = """\
"measurement_number","datetime","green_algae","bluegreen","diatoms","cryptophyta","yellow_substances","total_concentration","transmission","depth","temperature","station","key"
1,"2017-07-11 11:48",4.71,1.35,0,1.56,1.1,7.62,97.1,0.92,25.13,"WE12","sample"
2,"2017-07-11 11:49:30",,0.27,0,1.56,1.1,7.62,97.1,1.92,24.13,"WE12","sample"
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.csv"
            path.write_text(content, encoding="utf-8")
            profile = profile_fluoroprobe_csv(path)

        self.assertEqual(profile["records"], 2)
        self.assertEqual(profile["observed_start"], "2017-07-11 11:48")
        self.assertEqual(profile["observed_end"], "2017-07-11 11:49:30")
        self.assertIn("Eastern Daylight Time", profile["time_basis"])
        self.assertEqual(profile["missing_counts"]["green_algae"], 1)
