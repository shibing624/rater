# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
import torch
import torch.nn as nn

from ..basic.functional import bi_interaction
from ..basic.output_layer import OutputLayer


class FM(nn.Module):
    def __init__(self, emb_dim, num_feats, out_type='binary'):
        super(FM, self).__init__()
        self.emb_dim = emb_dim
        self.num_feats = num_feats
        self.emb_layer = nn.Embedding(num_embeddings=num_feats, embedding_dim=emb_dim)
        nn.init.xavier_uniform_(self.emb_layer.weight)
        self.bias = nn.Parameter(torch.randn(1))
        self.first_order_weights = nn.Embedding(num_embeddings=num_feats, embedding_dim=1)
        nn.init.xavier_uniform_(self.first_order_weights.weight)
        self.output_layer = OutputLayer(1, out_type)

    def forward(self, feat_index, feat_value):
        # With single sample, it should be expanded into 1 * F * K
        # Batch_size: N
        # feat_index_dim&feat_value_dim: F
        # embedding_dim: K

        # feat_index: N * F
        # feat_value: N * F

        # compute first order
        feat_value = torch.unsqueeze(feat_value, dim=2)  # N * F * 1
        first_order_weights = self.first_order_weights(feat_index)  # N * F * 1
        first_order = torch.mul(feat_value, first_order_weights)  # N * F * 1
        first_order = torch.squeeze(first_order, dim=2)  # N * F
        y_first_order = torch.sum(first_order, dim=1)  # N * 1

        # compute second order
        # look up embedding table
        feat_emb = self.emb_layer(feat_index)  # N * F * K
        feat_emb_value = torch.mul(feat_emb, feat_value)  # N * F * K element-wise mul

        # compute sum of square
        # squared_feat_emb = torch.pow(feat_emb_value, 2)  # N * K
        # sum_of_square = torch.sum(squared_feat_emb, dim=1)  # N * K
        #
        # # compute square of sum
        # summed_feat_emb = torch.sum(feat_emb_value, dim=1)  # N * K
        # square_of_sum = torch.pow(summed_feat_emb, 2)  # N * K

        BI = bi_interaction(feat_emb_value)

        y_second_order = 0.5 * BI  # N * K
        y_second_order = torch.sum(y_second_order, dim=1)  # N * 1

        # compute y
        y = self.bias + y_first_order + y_second_order  # N * 1
        y = self.output_layer(y)
        return y

    def get_embedding(self):
        return self.emb_layer
