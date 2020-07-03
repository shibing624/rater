# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com), jachin, Nie
@description: A pytorch implementation of NFM


Reference:
[1] Neural Factorization Machines for Sparse Predictive Analytics (NUS 2017)
    Xiangnan He,School of Computing,National University of Singapore,Singapore 117417,dcshex@nus.edu.sg
    Tat-Seng Chua,School of Computing,National University of Singapore,Singapore 117417,dcscts@nus.edu.sg

"""

import torch
import torch.nn as nn

from ..basic.functional import bi_interaction
from ..basic.mlp import MLP
from ..basic.output_layer import OutputLayer


class NFM(nn.Module):
    """
    NFM Network
    """

    def __init__(self, feature_size, field_size, embedding_size=5, fc_dims=[32, 32], dropout=0.0, is_batch_norm=False,
                 out_type='binary'):
        """
        Init model
        :param feature_size: int, size of the feature dictionary
        :param field_size: int, size of the feature fields
        :param embedding_size: int, size of the feature embedding
        :param fc_dims: range, sizes of fc dims
        :param dropout: float, dropout rate
        :param is_batch_norm: bool, use batch normalization
        :param out_type: str, output layer function, binary is Sigmoid
        """
        super(NFM, self).__init__()
        self.embedding_size = embedding_size
        self.feature_size = feature_size
        self.field_size = field_size
        self.emb_layer = nn.Embedding(num_embeddings=feature_size, embedding_dim=embedding_size)
        nn.init.xavier_uniform_(self.emb_layer.weight)

        self.bi_intaraction_layer = BiInteractionLayer()
        self.fc_dims = fc_dims
        self.fc_layers = MLP(embedding_size, fc_dims, dropout, is_batch_norm)
        self.output_layer = OutputLayer(in_dim=fc_dims[-1], out_type=out_type)

    def forward(self, feat_index, feat_value):
        """
        Forward
        :param feat_index: index input tensor
        :param feat_value: value input tensor
        :return: predict y
        """
        feat_emb = self.emb_layer(feat_index)  # N * field_size * embedding_size
        feat_value = feat_value.unsqueeze(dim=2)  # N * field_size * 1
        feat_emb_value = torch.mul(feat_emb, feat_value)  # N * field_size * embedding_size
        bi = self.bi_intaraction_layer(feat_emb_value)  # N * embedding_size

        fc_out = self.fc_layers(bi)
        out = self.output_layer(fc_out)
        return out


class BiInteractionLayer(nn.Module):
    def __init__(self):
        super(BiInteractionLayer, self).__init__()

    def forward(self, feat_emb_value):
        # square_of_sum = torch.sum(feat_emb_value, dim=1)  # N * embedding_size
        # square_of_sum = torch.mul(square_of_sum, square_of_sum)  # N * embedding_size

        # sum_of_square = torch.mul(feat_emb_value, feat_emb_value)  # N * field_size * embedding_size
        # sum_of_square = torch.sum(sum_of_square, dim=1)  # N * embedding_size

        # bi_out = square_of_sum - sum_of_square

        bi_out = bi_interaction(feat_emb_value)
        bi_out = bi_out / 2
        return bi_out
