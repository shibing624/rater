# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Test feature generate
"""

import unittest

from rater.datasets.movielens import Movielens


class TestFeature(unittest.TestCase):
    def setUp(self):
        self.dataset = Movielens(n_samples=10)
        self.raw_ratings = self.dataset.data

    def test_get_feature(self):
        features, X_idx, X_value, y, category_index, continuous_value = self.dataset.get_features()
        print(features.feature_size(), features.field_size())
        print("X_idx[0], X_value[0], y[0] :\n", X_idx[0], X_value[0], y[0])
        assert X_idx.shape[0] == 10, 'error shape'

    def test_label_binarize(self, binarize_label=True):
        y = self.raw_ratings.rating

        if binarize_label:
            def transform_y(label):
                if label > 3:
                    return 1
                else:
                    return 0

            y = y.apply(transform_y)
        print(y)
        assert len(y) == 10, 'error label size'
