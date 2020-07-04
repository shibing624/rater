# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com), yuwanlong, jachin,Nie
@description: A pytorch implementation of deepfm

Reference:
[1] DeepFM: A Factorization-Machine based Neural Network for CTR Prediction (HIT-Huawei 2017)
    Huifeng Guo, Ruiming Tang, Yunming Yey, Zhenguo Li, Xiuqiang He.
"""

import torch
import torch.nn as nn

from .fm import FM
from ..basic.mlp import MLP
from ..basic.output_layer import OutputLayer


class DeepFM(nn.Module):
    def __init__(self, feature_size, field_size, embedding_size=5, fc_dims=[32, 32, 32], dropout=0.0,
                 is_batch_norm=False, out_type='binary'):
        """
        Init model
        :param feature_size: int, size of the feature dictionary
        :param field_size: int, size of the feature fields
        :param embedding_size: int, size of the feature embedding
        :param fc_dims: range, fc dims range
        :param dropout: float, dropout rate
        :param is_batch_norm: bool, use batch normalization
        :param out_type: str, output layer function, binary is Sigmoid
        """
        super(DeepFM, self).__init__()
        self.field_size = field_size
        # embedding layer is embedded in the FM sub-module
        self.embedding_size = embedding_size

        # fm
        self.fm = FM(feature_size=feature_size, embedding_size=embedding_size, out_type='regression')

        # dnn
        self.fc_dims = fc_dims if fc_dims else [32, 32, 32]
        self.dnn = MLP(embedding_size * field_size, fc_dims=fc_dims, dropout=dropout, is_batch_norm=is_batch_norm)

        # output
        self.output_layer = OutputLayer(fc_dims[-1] + 1, out_type)

    def forward(self, feat_index, feat_value):
        """
        Forward
        :param feat_index: index input tensor
        :param feat_value: value input tensor
        :return: predict y
        """
        # embedding
        emb_layer = self.fm.get_embedding()
        feat_emb = emb_layer(feat_index)

        # compute y_FM
        y_fm = self.fm(feat_index, feat_value)  # N
        y_fm = y_fm.unsqueeze(1)  # N * 1

        # compute y_dnn
        # reshape the embedding matrix to a vector
        dnn_in = feat_emb.reshape(-1, self.embedding_size * self.field_size)  # N * (embedding_size * field_size)
        y_dnn = self.dnn(dnn_in)  # N * fc_dims[-1]

        # compute output
        y = torch.cat((y_fm, y_dnn), dim=1)  # N * (fc_dims[-1] + 1)
        y = self.output_layer(y)
        return y
