import unittest

from algal_bloom_forecast.models.feature_groups import build_feature_groups


class FeatureGroupTests(unittest.TestCase):
    def test_groups_are_explicit_and_limited_to_available_fields(self):
        groups = build_feature_groups(
            [
                {
                    "seasonal_day_of_year_sin": 0.1,
                    "seasonal_day_of_year_cos": 0.9,
                    "usgs_maumee_discharge_m3s": 1.0,
                    "ndbc_wspd_mean": 2.0,
                    "glerl_we02_water_temperature_mean": 20.0,
                }
            ]
        )
        self.assertEqual(
            groups["seasonal_only"],
            ("seasonal_day_of_year_sin", "seasonal_day_of_year_cos"),
        )
        self.assertIn("ndbc_wspd_mean", groups["buoy_weather"])
        self.assertEqual(groups["water_quality"][-1], "glerl_we02_water_temperature_mean")
        self.assertNotIn("ndbc_gst_mean", groups["buoy_weather"])
