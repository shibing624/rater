# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: OutputLayer for Predication
"""
import torch.nn as nn


class OutputLayer(nn.Module):
    def __init__(self, in_dim, out_type='binary', use_bias=True, **kwargs):
        """
        Prediction output layer
        :param in_dim: int, input tensor dim
        :param out_type: str, task name, ``"binary"`` for binary logloss, ``"multiclass"`` for multi classification
                or  ``"regression"`` for regression loss
        :param use_bias: bool, use bias or not
        """
        super(OutputLayer, self).__init__()
        self.out_type = out_type
        self.in_dim = in_dim
        self.use_bias = use_bias
        if not self.in_dim == 1:
            self.weights = nn.Linear(in_features=in_dim, out_features=1, bias=self.use_bias)
        # out type for task
        if out_type not in ["binary", "multiclass", "regression"]:
            raise ValueError("out_type must be binary, multiclass or regression")
        if self.out_type == 'binary':
            self.output_layer = nn.Sigmoid()

    def forward(self, x):
        if not self.in_dim == 1:
            y = self.weights(x)
        else:
            y = x
        if self.out_type == 'binary':
            y = self.output_layer(y)
        return y
