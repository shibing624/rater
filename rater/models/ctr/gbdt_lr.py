# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
import torch.nn as nn

from .lr import LR
from ..basic.gbdt import GBDT


class GBDTLR(nn.Module):
    def __init__(self, num_leaves=31, max_depth=-1, n_estimators=100, min_data_in_leaf=20,
                 learning_rate=0.1, objective='binary'):
        super(GBDTLR, self).__init__()
        self.gbdt = GBDT(num_leaves, max_depth, n_estimators, min_data_in_leaf, learning_rate, objective)
        self.gbdt_trained = False
        self.logistic_layer = LR(num_leaves * n_estimators, out_type=objective)

    def forward(self, data):
        pred_y, feat_index = self.gbdt.pred(data)
        y = self.logistic_layer(feat_index)
        return y

    def train_gbdt(self, data, y):
        self.gbdt.train(data, y)
        self.gbdt_trained = True

    def get_gbdt_trained(self):
        return self.gbdt_trained
