# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Test bpmf
"""

import unittest

import numpy as np
import numpy.testing as np_test
from numpy.random import RandomState

from rater.metrics.rating import RMSE


class TestRMSE(unittest.TestCase):
    def test_rmse_same_input(self):
        rs = RandomState(0)
        data = rs.randn(1, 100)
        print(data)
        r = RMSE().compute(data, data)
        np_test.assert_almost_equal(r, np.sqrt(0))

    def test_rmse(self):
        np_test.assert_almost_equal(
            RMSE().compute(np.zeros(100), np.ones(100)), np.sqrt(1))

    def test_rand(self):
        arr1 = np.random.randn(2, 4)
        print(arr1)
        print('*' * 42)
        arr2 = np.random.rand(2, 4)
        print(arr2)
