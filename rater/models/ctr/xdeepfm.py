# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com), Wutong Zhang
@description: PyTorch implementation of xDeepFM

Reference:
[1] xDeepFM: Combining Explicit and Implicit Feature Interactionsfor Recommender Systems,
    Jianxun Lian, Xiaohuan Zhou, Fuzheng Zhang, Zhongxia Chen, Xing Xie,and Guangzhong Sun
    https://arxiv.org/pdf/1803.05170.pdf
[2] TensorFlow implementation of xDeepFM
    https://github.com/Leavingseason/xDeepFM
[3] PaddlePaddle implemantation of xDeepFM
    https://github.com/PaddlePaddle/models/tree/develop/PaddleRec/ctr/xdeepfm
[4] PyTorch implementation of xDeepFM
    https://github.com/qian135/ctr_model_zoo/blob/master/xdeepfm.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class xDeepFM(nn.Module):
    def __init__(self, feature_size, field_size,
                 dropout_deep=[0, 0, 0, 0, 0],
                 deep_layer_sizes=[400, 400, 400, 400],
                 cin_layer_sizes=[100, 100, 50],
                 split_half=True,
                 embedding_size=5):
        super(xDeepFM, self).__init__()
        self.feature_size = feature_size
        self.field_size = field_size
        self.cin_layer_sizes = cin_layer_sizes
        self.deep_layer_sizes = deep_layer_sizes
        self.embedding_size = embedding_size
        self.dropout_deep = dropout_deep
        self.split_half = split_half

        self.input_dim = field_size * embedding_size

        # init feature embedding
        feat_embedding = nn.Embedding(feature_size, embedding_size)
        nn.init.xavier_uniform_(feat_embedding.weight)
        self.feat_embedding = feat_embedding

        # Compress Interaction Network (CIN) Part
        cin_layer_dims = [self.field_size] + cin_layer_sizes

        prev_dim, fc_input_dim = self.field_size, 0
        self.conv1ds = nn.ModuleList()
        for k in range(1, len(cin_layer_dims)):
            conv1d = nn.Conv1d(cin_layer_dims[0] * prev_dim, cin_layer_dims[k], 1)
            nn.init.xavier_uniform_(conv1d.weight)
            self.conv1ds.append(conv1d)
            if self.split_half and k != len(self.cin_layer_sizes):
                prev_dim = cin_layer_dims[k] // 2
            else:
                prev_dim = cin_layer_dims[k]
            fc_input_dim += prev_dim

        # Deep Neural Network Part
        all_dims = [self.input_dim] + deep_layer_sizes
        for i in range(len(deep_layer_sizes)):
            setattr(self, 'linear_' + str(i + 1), nn.Linear(all_dims[i], all_dims[i + 1]))
            setattr(self, 'batchNorm_' + str(i + 1), nn.BatchNorm1d(all_dims[i + 1]))
            setattr(self, 'dropout_' + str(i + 1), nn.Dropout(dropout_deep[i + 1]))

        # Linear Part
        self.linear = nn.Linear(self.input_dim, 1)

        # output Part
        self.output_layer = nn.Linear(1 + fc_input_dim + deep_layer_sizes[-1], 1)

    def forward(self, feat_index, feat_value, use_dropout=True):
        # get feat embedding
        fea_embedding = self.feat_embedding(feat_index)  # None * F * K
        x0 = fea_embedding

        # Linear Part
        linear_part = self.linear(fea_embedding.reshape(-1, self.input_dim))

        # CIN Part
        x_list = [x0]
        res = []
        for k in range(1, len(self.cin_layer_sizes) + 1):
            # Batch * H_K * D, Batch * M * D -->  Batch * H_k * M * D
            z_k = torch.einsum('bhd,bmd->bhmd', x_list[-1], x_list[0])
            z_k = z_k.reshape(x0.shape[0], x_list[-1].shape[1] * x0.shape[1], x0.shape[2])
            x_k = self.conv1ds[k - 1](z_k)
            x_k = torch.relu(x_k)

            if self.split_half and k != len(self.cin_layer_sizes):
                # x, h = torch.split(x, x.shape[1] // 2, dim=1)
                next_hidden, hi = torch.split(x_k, x_k.shape[1] // 2, 1)
            else:
                next_hidden, hi = x_k, x_k

            x_list.append(next_hidden)
            res.append(hi)

        res = torch.cat(res, dim=1)
        res = torch.sum(res, dim=2)

        # Deep NN Part
        y_deep = fea_embedding.reshape(-1, self.field_size * self.embedding_size)  # None * (F * K)
        if use_dropout:
            y_deep = nn.Dropout(self.dropout_deep[0])(y_deep)

        for i in range(1, len(self.deep_layer_sizes) + 1):
            y_deep = getattr(self, 'linear_' + str(i))(y_deep)
            y_deep = getattr(self, 'batchNorm_' + str(i))(y_deep)
            y_deep = F.relu(y_deep)
            if use_dropout:
                y_deep = getattr(self, 'dropout_' + str(i))(y_deep)

        # Output Part
        concat_input = torch.cat((linear_part, res, y_deep), dim=1)
        y = self.output_layer(concat_input)
        y = nn.Sigmoid()(y)
        return y
