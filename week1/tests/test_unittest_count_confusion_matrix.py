import unittest
import pandas as pd
import numpy as np
from engine.tools.select_best_threshold import count_TP_and_FP_for_df

class TestVarScenario(unittest.TestCase):
    def test_01_all_hit(self):
        test_01_df = pd.DataFrame({'Predictions': ['a', 'd', 'd', 'c', 'a', 'b', 'b', 'c', 'e', 'a', 'd', 'e', 'a'],
                                   'true_type':   ['a', 'd', 'd', 'c', 'a', 'b', 'b', 'c', 'e', 'a', 'd', 'e', 'a']})
        test_01_df['indicator'] = test_01_df['Predictions'] == test_01_df['true_type']
        TP_list, FP_list, FN_list, TN_list = count_TP_and_FP_for_df(test_01_df)

        self.assertTrue(np.array_equal([4, 3, 2, 2, 2], TP_list))
        self.assertTrue(np.array_equal([0, 0, 0, 0, 0], FP_list))
        self.assertTrue(np.array_equal([0, 0, 0, 0, 0], FN_list))
        self.assertTrue(np.array_equal([9, 10, 11, 11, 11], TN_list))

    def test_02_all_fail(self):
        test_02_df = pd.DataFrame({'Predictions': ['a', 'd', 'd', 'c', 'a', 'b', 'b', 'c', 'e', 'g', 'd', 'e', 'a'],
                                   'true_type':   ['b', 'c', 'c', 'd', 'b', 'd', 'a', 'e', 'b', 'a', 'c', 'a', 'b']})
        test_02_df['indicator'] = test_02_df['Predictions'] == test_02_df['true_type']
        TP_list, FP_list, FN_list, TN_list = count_TP_and_FP_for_df(test_02_df)

        self.assertTrue(np.array_equal([0, 0, 0, 0, 0], TP_list))
        self.assertTrue(np.array_equal([2, 2, 3, 3, 2], FP_list))
        self.assertTrue(np.array_equal([4, 3, 2, 3, 1], FN_list))
        self.assertTrue(np.array_equal([0, 0, 0, 0, 0], TN_list))

    def test_03_custom_test(self):
        test_03_df = pd.DataFrame({'Predictions': ['a', 'e', 'e', 'd', 'b', 'd', 'e', 'c', 'c', 'd', 'c', 'e', 'b'],
                                   'true_type':   ['b', 'c', 'c', 'd', 'b', 'd', 'a', 'e', 'b', 'a', 'c', 'a', 'b']})
        test_03_df['indicator'] = test_03_df['Predictions'] == test_03_df['true_type']
        TP_list, FP_list, FN_list, TN_list = count_TP_and_FP_for_df(test_03_df)
        self.assertTrue(np.array_equal([2, 1, 2, 0, 0], TP_list))
        self.assertTrue(np.array_equal([0, 2, 1, 1, 4], FP_list))
        self.assertTrue(np.array_equal([2, 2, 0, 3, 1], FN_list))
        self.assertTrue(np.array_equal([3, 4, 3, 5, 5], TN_list))

if __name__ == '__main__':
    unittest.main()
