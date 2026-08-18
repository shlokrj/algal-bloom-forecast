import unittest

from algal_bloom_forecast.evaluation.errors import classify_event_error


class ErrorDiagnosticsTests(unittest.TestCase):
    def test_classifies_event_cases(self):
        self.assertEqual(classify_event_error(6.0, 7.0, 5.0), "true_positive")
        self.assertEqual(classify_event_error(2.0, 7.0, 5.0), "false_alarm")
        self.assertEqual(classify_event_error(6.0, 2.0, 5.0), "missed_event")
        self.assertEqual(classify_event_error(2.0, 3.0, 5.0), "true_negative")
