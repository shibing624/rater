# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Test deepFM
"""

import unittest

from rater.datasets.movielens import Movielens
from rater.features.feature_dict import FeatureDict, process_features


class TestFeature(unittest.TestCase):
    def setUp(self):
        self.dataset = Movielens()
        self.raw_ratings = self.dataset.data

    def test_get_feature(self):

        features = FeatureDict()
        features.add_categorical_feat('user')
        features.add_categorical_feat('item')

        X = self.raw_ratings
        X_idx, X_value = self.dataset.X_idx, self.dataset.X_value

        print(X_idx, X_value, features)
        assert X_idx.shape[0] == 100000, 'error shape'

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
        assert len(y) == 100000, 'error label size'
