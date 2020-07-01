# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..basic.output_layer import OutputLayer
from ..basic.mlp import MLP


class AutoInt(nn.Module):
    def __init__(self, emb_dim, projection_dim, num_heads, num_feats, num_fields, use_res=True, out_type='binary'):
        super(AutoInt, self).__init__()
        self.emb_dim = emb_dim
        self.projection_dim = projection_dim
        self.num_heads = num_heads
        self.num_feats = num_feats
        self.num_fields = num_fields

        self.emb_layer = nn.Embedding(num_embeddings=num_feats, embedding_dim=emb_dim)
        nn.init.xavier_uniform_(self.emb_layer.weight)

        self.query_weights = nn.Parameter(torch.zeros(emb_dim, projection_dim * num_heads))
        nn.init.xavier_uniform_(self.query_weights.data)

        self.key_weights = nn.Parameter(torch.zeros(emb_dim, projection_dim * num_heads))
        nn.init.xavier_uniform_(self.key_weights.data)

        self.value_weights = nn.Parameter(torch.zeros(emb_dim, projection_dim * num_heads))
        nn.init.xavier_uniform_(self.value_weights.data)

        self.use_res = use_res
        if use_res:
            self.res_weights = nn.Parameter(torch.zeros(emb_dim, projection_dim * num_heads))
            nn.init.xavier_uniform_(self.res_weights.data)

        self.output_layer = OutputLayer(in_dim=num_fields * num_heads * projection_dim, out_type=out_type)

    def forward(self, feat_index):
        # for each field, there is a multi-head self-attention,
        # so the calculation is num_heads * num_fields (inner-product),
        # and the total calculation is num_heads * num_fields * num_fields
        feat_emb = self.emb_layer(feat_index)  # N * num_fields * emb_dim

        queries = torch.matmul(feat_emb, self.query_weights)  # N * num_fields * (pro_dim * num_heads)
        queries = torch.split(queries, self.projection_dim, dim=2)  # [N * num_fields * pro_dim] * num_heads
        queries = torch.stack(queries, dim=1)  # N * num_heads * num_fields * pro_dim

        keys = torch.matmul(feat_emb, self.key_weights)  # N * num_fields * (pro_dim * num_heads)
        keys = torch.split(keys, self.projection_dim, dim=2)  # [N * num_fields * pro_dim] * num_heads
        keys = torch.stack(keys, dim=1)  # N * num_heads * num_fields * pro_dim

        values = torch.matmul(feat_emb, self.value_weights)  # N * num_fields * (pro_dim * num_heads)
        values = torch.split(values, self.projection_dim, dim=2)  # [N * num_fields * pro_dim] * num_heads
        values = torch.stack(values, dim=1)  # N * num_heads * num_fields * pro_dim

        keys = keys.transpose(2, 3)
        # the i^th row of inner-product (pro_dim * pro_dim) means the attention signal when the i^th field is the query
        inner_product_qk = torch.matmul(queries, keys)  # N * num_heads * num_fields * num_fields

        # here the inner-product is not scaled by sqrt(n)
        att_signal = F.softmax(inner_product_qk, dim=2)  # N * num_heads * num_fields * num_fields
        att_value = torch.matmul(att_signal, values)  # N * num_heads * num_fields * pro_dim
        att_values = torch.split(att_value, 1, dim=1)  # [N * 1 * num_fields * pro_dim] * num_heads
        att_values = torch.cat(att_values, dim=3)  # N * 1 * num_fields * (pro_dim * num_heads)
        multi_head_emb = att_values.squeeze()  # N * num_fields * (pro_dim * num_heads)

        if self.use_res:
            res = torch.matmul(feat_emb, self.res_weights)  # N * num_fields * (pro_dim * num_heads)
            multi_head_emb = torch.add(multi_head_emb, res)  # N * num_fields * (pro_dim * num_heads)

        multi_head_emb = F.relu(multi_head_emb)
        multi_head_emb = multi_head_emb.reshape((-1, self.num_fields * self.num_heads * self.projection_dim))
        y = self.output_layer(multi_head_emb)
        return y
