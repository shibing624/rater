# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: GBDT LR model
"""
import torch.nn as nn

from .lr import LR
from ..basic.gbdt import GBDT


class GBDTLR(nn.Module):
    """
    GBDT LR model
    """

    def __init__(self, num_leaves=31, max_depth=-1, n_estimators=100, min_data_in_leaf=20,
                 learning_rate=0.1, out_type='binary'):
        """
        Init model
        :param num_leaves:
        :param max_depth:
        :param n_estimators:
        :param min_data_in_leaf:
        :param learning_rate:
        :param out_type:
        """
        super(GBDTLR, self).__init__()
        self.gbdt = GBDT(num_leaves, max_depth, n_estimators, min_data_in_leaf, learning_rate, out_type)
        self.gbdt_trained = False
        self.logistic_layer = LR(num_leaves * n_estimators, out_type=out_type)

    def forward(self, data):
        """
        Forward
        :param data: input tensor
        :return: predict y
        """
        pred_y, feat_index = self.gbdt.pred(data)
        y = self.logistic_layer(feat_index)
        return y

    def train_gbdt(self, data, y):
        """
        Train model with labeled data
        :param data:
        :param y:
        :return: model
        """
        self.gbdt.train(data, y)
        self.gbdt_trained = True

    def get_gbdt_trained(self):
        """
        Is model trained.
        :return: bool
        """
        return self.gbdt_trained
