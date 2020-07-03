# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:

Reference:
[1] Wide & Deep Learning for Recommender Systems (Google 2016)

"""
import torch
import torch.nn as nn

from ..basic.mlp import MLP
from ..basic.output_layer import OutputLayer


class WideAndDeep(nn.Module):
    """
    WideAndDeep Network
    """

    # TODO: categorical_field_size, continuous_field_size bug, fix with num_fields
    def __init__(self, feature_size, categorical_field_size, continuous_field_size, embedding_size=5, fc_dims=[32, 32],
                 dropout=0.0, is_batch_norm=False, out_type='binary'):
        """
        Init model
        :param feature_size: int, size of the feature dictionary
        :param categorical_field_size: int, size of categorical field
        :param continuous_field_size: int, size of continuous field
        :param embedding_size: int, size of the feature embedding
        :param fc_dims: range, sizes of fc dims
        :param dropout: float, dropout rate
        :param is_batch_norm: bool, use batch normalization
        :param out_type: str, output layer function, binary is Sigmoid
        """
        super(WideAndDeep, self).__init__()
        self.embedding_size = embedding_size
        self.feature_size = feature_size
        self.categorical_field_size = categorical_field_size
        self.continuous_field_size = continuous_field_size

        # first order weight for category features
        self.cate_weights = nn.Embedding(num_embeddings=feature_size - continuous_field_size, embedding_dim=1)
        nn.init.xavier_uniform_(self.cate_weights.weight)

        # first order weight for continuous features
        self.cont_weights = nn.Linear(in_features=continuous_field_size, out_features=1)
        nn.init.xavier_uniform_(self.cont_weights)

        self.wide_bias = nn.Parameter(torch.randn(1), requires_grad=True)

        fc_dims.append(1)
        self.fc_dims = fc_dims

        # embedding for deep network
        self.emb_layer = nn.Embedding(num_embeddings=feature_size - continuous_field_size, embedding_dim=embedding_size)
        nn.init.xavier_uniform_(self.emb_layer.weight)

        self.deep = MLP(continuous_field_size + categorical_field_size * embedding_size, fc_dims, dropout,
                        is_batch_norm)
        self.out_layer = OutputLayer(in_dim=1, out_type=out_type)

    def forward(self, categorical_index, continuous_value):
        """
        Forward
        :param categorical_index: index input tensor
        :param continuous_value: value input tensor
        :return: predict y
        """
        first_order_cate = self.cate_weights(categorical_index)
        first_order_cont = self.cont_weights(continuous_value)
        y_wide = first_order_cate + first_order_cont + self.wide_bias

        cate_emb_value = self.emb_layer(categorical_index)  # N * categorical_field_size * embedding_size
        # N * (categorical_field_size * embedding_size)
        cate_emb_value = cate_emb_value.reshape((-1, self.categorical_field_size * self.embedding_size))
        deep_in = torch.cat([continuous_value, cate_emb_value],
                            1)  # N * (categorical_field_size * embedding_size + continuous_field_size)
        y_deep = self.deep(deep_in)  # N * 1
        y = y_deep + y_wide
        y = self.out_layer(y)
        return y
