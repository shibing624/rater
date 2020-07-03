# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: LR
"""
import torch
import torch.nn as nn

from ..basic.output_layer import OutputLayer


class LR(nn.Module):
    """
    LR model
    """

    def __init__(self, num_feats, out_type='binary'):
        """
        Init
        :param num_feats:
        :param out_type:
        """
        super(LR, self).__init__()
        self.num_feats = num_feats
        self.weights = nn.Embedding(num_embeddings=num_feats, embedding_dim=1)
        self.bias = nn.Parameter(torch.randn(1))
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
