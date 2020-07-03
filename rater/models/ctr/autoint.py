# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: AutoInt model

Reference:

[arxiv 2018][AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks]
(https://arxiv.org/abs/1810.11921)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..basic.output_layer import OutputLayer


class AutoInt(nn.Module):
    """
    Network
    """

    def __init__(self, feature_size, field_size, embedding_size=5, projection_dim=5, num_heads=8, use_res=True,
                 out_type='binary'):
        """
        Init model
        :param feature_size: int, size of the feature dictionary
        :param field_size: int, size of the feature fields
        :param embedding_size: int, size of the feature embedding
        :param projection_dim:
        :param num_heads:
        :param use_res: bool,
        :param out_type: str, output layer function, binary is Sigmoid
        """
        super(AutoInt, self).__init__()
        self.feature_size = feature_size
        self.field_size = field_size
        self.embedding_size = embedding_size
        self.projection_dim = projection_dim
        self.num_heads = num_heads

        self.emb_layer = nn.Embedding(num_embeddings=feature_size, embedding_dim=embedding_size)
        nn.init.xavier_uniform_(self.emb_layer.weight)

        self.query_weights = nn.Parameter(torch.zeros(embedding_size, projection_dim * num_heads), requires_grad=True)
        nn.init.xavier_uniform_(self.query_weights.data)

        self.key_weights = nn.Parameter(torch.zeros(embedding_size, projection_dim * num_heads), requires_grad=True)
        nn.init.xavier_uniform_(self.key_weights.data)

        self.value_weights = nn.Parameter(torch.zeros(embedding_size, projection_dim * num_heads), requires_grad=True)
        nn.init.xavier_uniform_(self.value_weights.data)

        self.use_res = use_res
        if self.use_res:
            self.res_weights = nn.Parameter(torch.zeros(embedding_size, projection_dim * num_heads), requires_grad=True)
            nn.init.xavier_uniform_(self.res_weights.data)
        # output layer
        self.output_layer = OutputLayer(in_dim=field_size * num_heads * projection_dim, out_type=out_type)

    def forward(self, feat_index):
        """
        Forward
        :param feat_index: index input tensor
        :return: predict y
        """
        # for each field, there is a multi-head self-attention,
        # so the calculation is num_heads * field_size (inner-product),
        # and the total calculation is num_heads * field_size * field_size
        feat_emb = self.emb_layer(feat_index)  # N * field_size * embedding_size

        queries = torch.matmul(feat_emb, self.query_weights)  # N * field_size * (pro_dim * num_heads)
        queries = torch.split(queries, self.projection_dim, dim=2)  # [N * field_size * pro_dim] * num_heads
        queries = torch.stack(queries, dim=1)  # N * num_heads * field_size * pro_dim

        keys = torch.matmul(feat_emb, self.key_weights)  # N * field_size * (pro_dim * num_heads)
        keys = torch.split(keys, self.projection_dim, dim=2)  # [N * field_size * pro_dim] * num_heads
        keys = torch.stack(keys, dim=1)  # N * num_heads * field_size * pro_dim

        values = torch.matmul(feat_emb, self.value_weights)  # N * field_size * (pro_dim * num_heads)
        values = torch.split(values, self.projection_dim, dim=2)  # [N * field_size * pro_dim] * num_heads
        values = torch.stack(values, dim=1)  # N * num_heads * field_size * pro_dim

        keys = keys.transpose(2, 3)
        # the i^th row of inner-product (pro_dim * pro_dim) means the attention signal when the i^th field is the query
        inner_product_qk = torch.matmul(queries, keys)  # N * num_heads * field_size * field_size

        # here the inner-product is not scaled by sqrt(n)
        att_signal = F.softmax(inner_product_qk, dim=2)  # N * num_heads * field_size * field_size
        att_value = torch.matmul(att_signal, values)  # N * num_heads * field_size * pro_dim
        att_values = torch.split(att_value, 1, dim=1)  # [N * 1 * field_size * pro_dim] * num_heads
        att_values = torch.cat(att_values, dim=3)  # N * 1 * field_size * (pro_dim * num_heads)
        multi_head_emb = att_values.squeeze()  # N * field_size * (pro_dim * num_heads)

        if self.use_res:
            res = torch.matmul(feat_emb, self.res_weights)  # N * field_size * (pro_dim * num_heads)
            multi_head_emb = torch.add(multi_head_emb, res)  # N * field_size * (pro_dim * num_heads)

        multi_head_emb = F.relu(multi_head_emb)
        multi_head_emb = multi_head_emb.reshape((-1, self.field_size * self.num_heads * self.projection_dim))
        y = self.output_layer(multi_head_emb)
        return y
