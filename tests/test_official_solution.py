import unittest

import numpy as np
import pandas as pd

from official_solution import weighted_pearson


class OfficialWeightedPearsonTest(unittest.TestCase):
    def test_matches_numpy_correlation_for_equal_weights(self) -> None:
        actual = pd.Series([1.0, 2.0, 4.0, 8.0])
        prediction = np.array([0.0, 3.0, 5.0, 9.0])
        score = weighted_pearson(actual, prediction, pd.Series([1.0, 1.0, 1.0, 1.0]))
        self.assertAlmostEqual(score, np.corrcoef(actual, prediction)[0, 1])

    def test_rejects_missing_prediction(self) -> None:
        with self.assertRaises(ValueError):
            weighted_pearson(pd.Series([1.0, 2.0]), np.array([0.0, np.nan]), pd.Series([1.0, 1.0]))
