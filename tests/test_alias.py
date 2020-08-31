# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import unittest

import matplotlib.pyplot as plt
import numpy as np

from rater.models.graph.alias import alias_sample, create_alias_table


def gen_prob_dist(N):
    p = np.random.randint(0, 100, N)
    return p / np.sum(p)


def simulate(N=100, k=10000):
    truth = gen_prob_dist(N)

    area_ratio = truth
    accept, alias = create_alias_table(area_ratio)

    ans = np.zeros(N)
    for _ in range(k):
        i = alias_sample(accept, alias)
        ans[i] += 1
    return ans / np.sum(ans), truth


class TestDatasets(unittest.TestCase):
    def setUp(self):
        print("start")

    def test_alias(self):
        alias, truth = simulate()
        plt.bar(list(range(len(alias))), alias, label='alias')
        plt.bar(list(range(len(truth))), truth, label='truth')
        plt.legend()
        # plt.show()
        print(alias)
        print(truth)
        assert len(alias) == 100, 'error data'
