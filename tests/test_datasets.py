# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Test bpmf
"""

import unittest

import numpy as np
from sklearn import preprocessing

from rater.datasets.criteo import Criteo
from rater.datasets.movielens import Movielens


class TestDatasets(unittest.TestCase):
    def setUp(self):
        self.choices = list(range(1, 10))
        self.movielens_data = Movielens()
        self.criteo_data = Criteo()

    def test_movielens_data(self):
        data_file = self.movielens_data.data_file
        data_size = self.movielens_data.data.shape[0]
        print(data_file)
        print(self.movielens_data)
        assert data_size == 100000, '100k size'

    def test_criteo_data(self):
        data_file = self.criteo_data.data_file
        data_size = self.criteo_data.data.shape[0]
        print(data_file)
        print(self.criteo_data)
        assert data_size == 100000, '100k size'

    def test_min_max_scaler(self):
        x = np.array([[3., -1., 2., 613.],
                      [2., 0., 0., 232],
                      [0., 1., -1., 113],
                      [1., 2., -3., 489]])
        print(x)
        min_max_scaler = preprocessing.MinMaxScaler()
        x_minmax = min_max_scaler.fit_transform(x)
        print(x_minmax)
