# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com), jachin,Nie
@description: A pytorch implementation of AFM

Reference:
[1] Attentional Factorization Machines - Learning the Weight of Feature Interactions via Attention Networks (ZJU 2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..basic.functional import build_cross
from ..basic.output_layer import OutputLayer


class AFM(nn.Module):
    """
    Network
    """

    def __init__(self, feature_size, field_size, embedding_size=5, attention_size=4, out_type='binary'):
        """
        Init model
        :param feature_size: int, size of the feature dictionary
        :param field_size: int, size of the feature fields
        :param embedding_size: int, size of the feature embedding
        :param attention_size: int, size of attention weight
        :param out_type: str, output layer function, binary is Sigmoid
        """
        super(AFM, self).__init__()
        self.feature_size = feature_size
        self.field_size = field_size
        self.embedding_size = embedding_size
        self.attention_size = attention_size
        self.first_order_weights = nn.Embedding(num_embeddings=feature_size, embedding_dim=1)
        nn.init.xavier_uniform_(self.first_order_weights.weight)
        self.bias = nn.Parameter(torch.randn(1), requires_grad=True)
        self.emb_layer = nn.Embedding(num_embeddings=feature_size, embedding_dim=embedding_size)
        nn.init.xavier_uniform_(self.emb_layer.weight)
        self.num_pairs = int(field_size * (field_size - 1) / 2)

        self.attention_w = nn.Parameter(torch.Tensor(self.embedding_size, self.attention_size), requires_grad=True)
        self.attention_b = nn.Parameter(torch.Tensor(self.attention_size), requires_grad=True)
        self.projection_h = nn.Parameter(torch.Tensor(self.attention_size, 1), requires_grad=True)
        self.projection_p = nn.Parameter(torch.Tensor(self.embedding_size, 1), requires_grad=True)
        for tensor in [self.attention_w, self.projection_h, self.projection_p]:
            nn.init.xavier_normal_(tensor)
        # output layer
        self.output_layer = OutputLayer(1, out_type)

    def forward(self, feat_index, feat_value):
        """
        Forward
        :param feat_index: index input tensor
        :param feat_value: value input tensor
        :return: predict y
        """
        feat_value = feat_value.unsqueeze(2)  # N * field_size * 1
        # first order
        first_order_weight = self.first_order_weights(feat_index)  # N * field_size * 1
        y_first_order = torch.mul(first_order_weight, feat_value)  # N * field_size * 1
        y_first_order = torch.sum(y_first_order, dim=1)  # N * 1
        y_first_order = y_first_order.squeeze(1)

        feat_emb = self.emb_layer(feat_index)  # N * field_size * embedding_size
        feat_emb_value = torch.mul(feat_emb, feat_value)  # N * field_size * embedding_size

        # attention weight
        p, q = build_cross(self.field_size, feat_emb_value)  # input: N * num_pairs * embedding_size
        bi_interaction = p * q  # N * num_pairs * embedding_size
        attention_temp = F.relu(torch.tensordot(
            bi_interaction, self.attention_w, dims=([-1], [0])) + self.attention_b)
        self.normalized_att_score = F.softmax(torch.tensordot(
            attention_temp, self.projection_h, dims=([-1], [0])), dim=1)
        attention_weight = torch.sum(
            self.normalized_att_score * bi_interaction, dim=1)
        attention_weight = torch.tensordot(
            attention_weight, self.projection_p, dims=([-1], [0]))

        y = self.bias + y_first_order + attention_weight.squeeze()
        y = self.output_layer(y)
        return y
