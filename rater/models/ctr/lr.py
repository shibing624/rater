# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: LogisticRegression
"""
import torch
import torch.nn as nn

from ..basic.output_layer import OutputLayer


class LR(nn.Module):
    """
    LR model
    """

    def __init__(self, input_size, out_type='binary'):
        """
        Init
        :param input_size:
        :param out_type:
        """
        super(LR, self).__init__()
        self.feature_size = input_size
        self.weights = nn.Embedding(num_embeddings=input_size, embedding_dim=1)
        self.bias = nn.Parameter(torch.randn(1), requires_grad=True)
        self.output_layer = OutputLayer(1, out_type)

    def forward(self, feat_index, feat_value):
        """
        Forward
        :param feat_index: index input tensor
        :param feat_value: value input tensor
        :return: predict y
        """
        weights = self.weights(feat_index)  # N * F * 1
        feat_value = torch.unsqueeze(feat_value, dim=2)  # N * F * 1
        first_order = torch.mul(feat_value, weights)  # N * F * 1
        first_order = torch.squeeze(first_order, dim=2)  # N * F
        y = torch.sum(first_order, dim=1)
        y += self.bias

        y = self.output_layer(y)
        return y
