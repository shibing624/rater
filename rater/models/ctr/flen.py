# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
import torch
import torch.nn as nn

from ..basic.functional import build_cross, bi_interaction
from ..basic.mlp import MLP
from ..basic.output_layer import OutputLayer


class FLEN(nn.Module):
    def __init__(self, emb_dim, num_feats, num_categories, field_ranges, fc_dims=None, dropout=None, batch_norm=None,
                 out_type='binary'):
        super(FLEN, self).__init__()
        self.num_feats = num_feats
        self.emb_dim = emb_dim
        self.num_categories = num_categories
        if not field_ranges:
            field_ranges = torch.tensor(range(num_categories))
        self.field_ranges = field_ranges
        self.num_fields = len(field_ranges)

        # embedding layer
        self.emb_layer = nn.Embedding(num_embeddings=num_feats, embedding_dim=emb_dim)
        nn.init.xavier_uniform_(self.emb_layer.weight)

        # S part
        self.first_order_weights = nn.Embedding(num_embeddings=num_categories, embedding_dim=1)
        nn.init.xavier_uniform_(self.first_order_weights.weight)
        self.first_order_bias = nn.Parameter(torch.randn(1))

        # MF part
        self.num_pairs = self.num_fields * (self.num_fields - 1) / 2
        self.r_mf = nn.Parameter(torch.zeros(self.num_pairs, 1))  # num_pairs * 1
        nn.init.xavier_uniform_(self.r_mf.data)

        # FM part
        self.r_fm = nn.Parameter(torch.zeros(self.num_fields, 1))  # num_fields * 1
        nn.init.xavier_uniform_(self.r_fm.data)

        # dnn
        if not fc_dims:
            fc_dims = [32, 32, 32]
        self.fc_dims = fc_dims
        self.fc_layers = MLP(fc_dims, dropout, batch_norm)

        self.output_layer = OutputLayer(fc_dims[-1] + 1 + self.emb_dim, out_type)

    def forward(self, feat_index):
        feat_emb = self.emb_layer(feat_index)  # N * num_categories * emb_dim

        field_wise_emb_list = [
            feat_emb[:, field_range]  # N * num_categories_in_field * emb_dim
            for field_range in self.field_ranges
        ]

        field_emb_list = [
            torch.sum(field_wise_emb, dim=1).unsqueeze(dim=1)  # N * emb_dim
            for field_wise_emb in field_wise_emb_list
        ]
        field_emb = torch.cat(field_emb_list, dim=1)  # N * num_fields * emb_dim
        # S part
        y_S = self.first_order_weights(feat_index)  # N * num_categories * 1
        y_S = y_S.squeeze()  # N * num_categories
        y_S = torch.sum(y_S, dim=1)  # N
        y_S = torch.add(y_S, self.first_order_bias)  # N
        y_S = y_S.unsqueeze(dim=1)  # N * 1

        # MF part -> N * emb_dim
        p, q = build_cross(self.num_fields, field_emb)  # N * num_pairs * emb_dim
        y_MF = torch.mul(p, q)  # N * num_pairs * emb_dim
        y_MF = torch.mul(y_MF, self.r_mf)  # N * num_pairs * emb_dim
        y_MF = torch.sum(y_MF, dim=1)  # N * emb_dim

        # FM part
        field_wise_fm = [
            bi_interaction(field_wise_emb).unsqueeze(dim=1)  # N * 1 * emb_dim
            for field_wise_emb in field_wise_emb_list
        ]
        field_wise_fm = torch.cat(field_wise_fm, dim=1)  # N * num_fields * emb_dim
        y_FM = torch.mul(field_wise_fm, self.r_fm)  # N * num_fields * emb_dim
        y_FM = torch.sum(y_FM, dim=1)  # N * emb_dim

        # dnn
        fc_in = field_emb.reshape((-1, self.num_fields * self.emb_dim))
        y_dnn = self.fc_layers(fc_in)

        # output
        fwBI = y_MF + y_FM
        fwBI = torch.cat([y_S, fwBI], dim=1)  # N * (emb_dim + 1)
        y = torch.cat([fwBI, y_dnn], dim=1)  # N * (fc_dims[-1] + emb_dim + 1)
        y = self.output_layer(y)
        return y
