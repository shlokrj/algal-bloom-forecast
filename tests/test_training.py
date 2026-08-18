import unittest

import numpy as np

from algal_bloom_forecast.models.training import fit_medians


class TrainingTests(unittest.TestCase):
    def test_medians_are_fit_from_training_matrix_values(self):
        matrix = np.array([[1.0, np.nan], [3.0, 10.0], [5.0, np.nan]])

        medians = fit_medians(matrix)

        self.assertEqual(medians.tolist(), [3.0, 10.0])

    def test_all_missing_feature_uses_deterministic_zero(self):
        medians = fit_medians(np.array([[np.nan], [np.nan]]))

        self.assertEqual(medians.tolist(), [0.0])
