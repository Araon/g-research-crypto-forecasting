import unittest

import pandas as pd

from advanced_solution import weighted_pearson


class WeightedPearsonTest(unittest.TestCase):
    def test_constant_weights_match_ordinary_correlation(self):
        actual = pd.Series([1.0, 2.0, 4.0, 8.0])
        prediction = pd.Series([0.0, 2.0, 3.0, 9.0])
        self.assertAlmostEqual(weighted_pearson(actual, prediction, pd.Series([1.0] * 4)), actual.corr(prediction))

    def test_invalid_prediction_is_rejected(self):
        with self.assertRaises(ValueError):
            weighted_pearson(pd.Series([1.0, 2.0]), pd.Series([1.0, float("nan")]), pd.Series([1.0, 1.0]))

    def test_non_positive_weight_is_rejected(self):
        with self.assertRaises(ValueError):
            weighted_pearson(pd.Series([1.0, 2.0]), pd.Series([1.0, 2.0]), pd.Series([1.0, 0.0]))


if __name__ == "__main__":
    unittest.main()
