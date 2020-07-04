# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com), jachin, Nie
@description: A pytorch implementation of FNN


Reference:
[1] Product-based Neural Networks for User Response Prediction (SJTU 2016)
    Yanru Qu, Han Cai, Kan Ren, Weinan Zhang, Yong Yu Shanghai Jiao Tong University
    {kevinqu, hcai, kren, wnzhang, yyu}@apex.sjtu.edu.cn Ying Wen, Jun Wang University College London {ying.wen, j.wang}@cs.ucl.ac.uk

"""

import torch
import torch.nn as nn

from ..basic.functional import build_cross
from ..basic.mlp import MLP
from ..basic.output_layer import OutputLayer


class PNN(nn.Module):
    """
    PNN Network
    """

    def __init__(self, feature_size, field_size, embedding_size=5, fc_dims=[32, 32], dropout=0.0, is_batch_norm=False,
                 product_type='inner', out_type='binary'):
        """
        Init model
        :param feature_size: int, size of the feature dictionary
        :param field_size: int, size of the feature fields
        :param embedding_size: int, size of the feature embedding
        :param fc_dims: range, sizes of fc dims
        :param dropout: float, dropout rate
        :param is_batch_norm: bool, use batch normalization
        :param product_type: str, product type layer, inner/outer
        :param out_type: str, output layer function, binary is Sigmoid
        """
        super(PNN, self).__init__()
        self.feature_size = feature_size
        self.field_size = field_size
        # embedding layer
        self.embedding_size = embedding_size
        self.emb_layer = nn.Embedding(num_embeddings=self.feature_size,
                                      embedding_dim=self.embedding_size)
        nn.init.xavier_uniform_(self.emb_layer.weight)

        fc_dims = fc_dims if fc_dims else[32, 32]
        # linear signal layer, named l_z
        self.d1 = d1 = fc_dims[0]
        self.product_type = product_type
        if product_type == '*':
            d1 *= 2
        self.linear_signal_weights = nn.Linear(in_features=field_size * embedding_size, out_features=d1)
        nn.init.xavier_uniform_(self.linear_signal_weights.weight)

        # product layer, named l_p
        if product_type == 'inner':
            self.product_layer = InnerProductLayer(field_size, d1)
        elif product_type == 'outer':
            self.product_layer = OuterProductLayer(embedding_size, field_size, d1)
        else:
            self.product_layer = HybridProductLayer(embedding_size, field_size, d1)

        # fc layers
        # l_1=relu(l_z+l_p_b_1)
        self.l1_layer = nn.ReLU()
        self.l1_bias = nn.Parameter(torch.randn(d1), requires_grad=True)
        # l_2 to l_n
        self.fc_dims = fc_dims
        self.fc_layers = MLP(d1, fc_dims=fc_dims, dropout=dropout, is_batch_norm=is_batch_norm)

        # output layer
        self.output_layer = OutputLayer(fc_dims[-1], out_type)

    def forward(self, feat_index):
        """
        Forward
        :param feat_index: index input tensor
        :return: predict y
        """
        # feat_index: N * field_size
        feat_emb = self.emb_layer(feat_index)  # N * field_size * embedding_size

        # compute linear signal l_z
        concat_z = feat_emb.reshape(-1, self.embedding_size * self.field_size)
        linear_signal = self.linear_signal_weights(concat_z)

        # product_layer
        product_out = self.product_layer(feat_emb)

        # fc layers from l_2 to l_n
        # l_1=relu(l_z+l_p_b_1)
        l1_in = torch.add(linear_signal, self.l1_bias)
        l1_in = torch.add(l1_in, product_out)
        l1_out = self.l1_layer(l1_in)
        y = self.fc_layers(l1_out)
        y = self.output_layer(y)
        return y


class InnerProductLayer(nn.Module):
    def __init__(self, field_size, d1):
        super(InnerProductLayer, self).__init__()
        self.field_size = field_size
        self.d1 = d1
        self.num_pairs = int(field_size * (field_size - 1) / 2)
        self.product_layer_weights = nn.Linear(in_features=self.num_pairs, out_features=d1)
        nn.init.xavier_uniform_(self.product_layer_weights.weight)

    def forward(self, feat_emb):
        # feat_emb: N * field_size * embedding_size

        # p_ij=<f_i,f_j>
        # p is symmetric matrix, so only upper triangular matrix needs calculation (without diagonal)
        p, q = build_cross(self.field_size, feat_emb)
        pij = p * q  # N * num_pairs * embedding_size
        pij = torch.sum(pij, dim=2)  # N * num_pairs

        # l_p
        lp = self.product_layer_weights(pij)
        return lp


class OuterProductLayer(nn.Module):
    def __init__(self, embedding_size, field_size, d1, kernel_type='mat'):
        super(OuterProductLayer, self).__init__()
        self.embedding_size = embedding_size
        self.field_size = field_size
        self.d1 = d1
        self.num_pairs = field_size * (field_size - 1) / 2
        self.kernel_type = kernel_type
        if kernel_type == 'vec':
            kernel_shape = (self.num_pairs, embedding_size)
        elif kernel_type == 'num':
            kernel_shape = (self.num_pairs, 1)
        else:  # by default mat
            kernel_shape = (embedding_size, self.num_pairs, embedding_size)
        self.kernel_shape = kernel_shape
        self.kernel = nn.Parameter(torch.zeros(kernel_shape))
        nn.init.xavier_uniform_(self.kernel.data)
        self.num_pairs = field_size * (field_size - 1) / 2
        self.product_layer_weights = nn.Linear(in_features=field_size, out_features=d1)
        nn.init.xavier_uniform_(self.product_layer_weights.weight)

    def forward(self, feat_emb):
        p, q = build_cross(self.field_size, feat_emb)  # p, q: N * num_pairs * embedding_size

        if self.kernel_type == 'mat':
            # self.kernel: embedding_size * num_pairs * embedding_size
            p = p.unsqueeze(1)  # N * 1 * num_pairs * embedding_size
            p = p * self.kernel  # N * embedding_size * num_pairs * embedding_size
            kp = torch.sum(p, dim=-1)  # N * embedding_size * num_pairs
            kp = kp.permute(0, 2, 1)  # N * num_pairs * embedding_size
            pij = torch.sum(kp * q, -1)  # N * num_pairs
        else:
            # self.kernel: num_pairs * embedding_size/1
            kernel = self.kernel.unsqueeze(1)  # 1 * num_pairs * embedding_size/1
            pij = p * q  # N * num_pairs * embedding_size
            pij = pij * kernel  # N * num_pairs * embedding_size
            pij = torch.sum(pij, -1)  # N * num_pairs

        # l_p
        lp = self.product_layer_weights(pij)
        return lp


class HybridProductLayer(nn.Module):
    def __init__(self, embedding_size, field_size, d1):
        super(HybridProductLayer, self).__init__()
        self.field_size = field_size
        self.d1 = d1 / 2
        self.inner_product_layer = InnerProductLayer(field_size, d1)
        self.outer_product_layer = OuterProductLayer(embedding_size, field_size, d1)

    def forward(self, feat_emb):
        inner_product_out = self.inner_product_layer(feat_emb)
        outer_product_out = self.outer_product_layer(feat_emb)
        lp = torch.cat([inner_product_out, outer_product_out], dim=1)
        return lp
