# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Test deepFM
"""

import os
import unittest

from recommender.datasets import criteo
from recommender.metrics import RMSE
from recommender.models.deepfm import DeepFM

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), 'test_data')
dac_sample_file = "dac_sample_1w.txt"


class TestDeepFM(unittest.TestCase):
    def setUp(self):
        self.n_feature = 10
        self.seed = 0

    def test_model_with_random_data(self):
        file_path = os.path.join(TEST_DATA_DIR, dac_sample_file)

