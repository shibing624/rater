# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:

Reference:
[arxiv 2019][FLEN: Leveraging Field for Scalable CTR Prediction](https://arxiv.org/pdf/1911.04690.pdf)

TODO fix error.
"""
import torch
import torch.nn as nn

from ..basic.functional import build_cross, bi_interaction
from ..basic.mlp import MLP
from ..basic.output_layer import OutputLayer


class FLEN(nn.Module):
    """
    FLEN Network
    """

    def __init__(self, feature_size, num_categories, field_ranges, embedding_size=5, fc_dims=[32, 32, 32],
                 dropout=0.0, is_batch_norm=False, out_type='binary'):
        """
        Init model
        :param feature_size: int, size of the feature dictionary
        :param num_categories: int, size of the category fields
        :param field_ranges:
        :param embedding_size:
        :param fc_dims:
        :param dropout:
        :param is_batch_norm:
        :param out_type:
        """
        super(FLEN, self).__init__()
        self.feature_size = feature_size
        self.embedding_size = embedding_size
        self.num_categories = num_categories
        if not field_ranges:
            field_ranges = torch.tensor(range(num_categories))
        self.field_ranges = field_ranges
        self.num_fields = len(field_ranges)

        # embedding layer
        self.emb_layer = nn.Embedding(num_embeddings=feature_size, embedding_dim=embedding_size)
        nn.init.xavier_uniform_(self.emb_layer.weight)

        # S part
        self.first_order_weights = nn.Embedding(num_embeddings=num_categories, embedding_dim=1)
        nn.init.xavier_uniform_(self.first_order_weights.weight)
        self.first_order_bias = nn.Parameter(torch.randn(1))

        # MF part
        self.num_pairs = int(self.num_fields * (self.num_fields - 1) / 2)
        self.r_mf = nn.Parameter(torch.zeros(self.num_pairs, 1))  # num_pairs * 1
        nn.init.xavier_uniform_(self.r_mf.data)

        # FM part
        self.r_fm = nn.Parameter(torch.zeros(self.num_fields, 1))  # num_fields * 1
        nn.init.xavier_uniform_(self.r_fm.data)

        # dnn
        self.fc_dims = fc_dims
        self.fc_layers = MLP(fc_dims, dropout, is_batch_norm)

        self.output_layer = OutputLayer(fc_dims[-1] + 1 + self.embedding_size, out_type)

    def forward(self, feat_index):
        """
        Forward
        :param feat_index: index input tensor
        :return: predict y
        """
        feat_emb = self.emb_layer(feat_index)  # N * num_categories * embedding_size

        field_wise_emb_list = [
            feat_emb[:, field_range]  # N * num_categories_in_field * embedding_size
            for field_range in self.field_ranges
        ]

        field_emb_list = [
            torch.sum(field_wise_emb, dim=1).unsqueeze(dim=1)  # N * embedding_size
            for field_wise_emb in field_wise_emb_list
        ]
        field_emb = torch.cat(field_emb_list, dim=1)  # N * num_fields * embedding_size
        # S part
        y_S = self.first_order_weights(feat_index)  # N * num_categories * 1
        y_S = y_S.squeeze()  # N * num_categories
        y_S = torch.sum(y_S, dim=1)  # N
        y_S = torch.add(y_S, self.first_order_bias)  # N
        y_S = y_S.unsqueeze(dim=1)  # N * 1

        # MF part -> N * embedding_size
        p, q = build_cross(self.num_fields, field_emb)  # N * num_pairs * embedding_size
        y_MF = torch.mul(p, q)  # N * num_pairs * embedding_size
        y_MF = torch.mul(y_MF, self.r_mf)  # N * num_pairs * embedding_size
        y_MF = torch.sum(y_MF, dim=1)  # N * embedding_size

        # FM part
        field_wise_fm = [
            bi_interaction(field_wise_emb).unsqueeze(dim=1)  # N * 1 * embedding_size
            for field_wise_emb in field_wise_emb_list
        ]
        field_wise_fm = torch.cat(field_wise_fm, dim=1)  # N * num_fields * embedding_size
        y_FM = torch.mul(field_wise_fm, self.r_fm)  # N * num_fields * embedding_size
        y_FM = torch.sum(y_FM, dim=1)  # N * embedding_size

        # dnn
        fc_in = field_emb.reshape((-1, self.num_fields * self.embedding_size))
        y_dnn = self.fc_layers(fc_in)

        # output
        fwBI = y_MF + y_FM
        fwBI = torch.cat([y_S, fwBI], dim=1)  # N * (embedding_size + 1)
        y = torch.cat([fwBI, y_dnn], dim=1)  # N * (fc_dims[-1] + embedding_size + 1)
        y = self.output_layer(y)
        return y
