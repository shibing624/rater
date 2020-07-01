# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import os

import torch
import torch.nn as nn
from torch.utils.data.dataset import TensorDataset

from rater.datasets.criteo import Criteo
from rater.models.ctr.afm import AFM
from rater.models.train import train_model

pwd_path = os.path.abspath(os.path.dirname(__file__))


def train(x_idx, x_value, label, features, out_type='binary'):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    X_idx_tensor = torch.LongTensor(x_idx).to(device)
    X_value_tensor = torch.Tensor(x_value).to(device)
    y_tensor = torch.Tensor(label).to(device)
    y_tensor = y_tensor.reshape(-1, 1)

    X = TensorDataset(X_idx_tensor, X_value_tensor, y_tensor)
    model = AFM(emb_dim=5, num_feats=features.get_num_feats(), num_fields=features.get_num_fields(),
                out_type=out_type).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    model_path = os.path.join(pwd_path, 'afm_model.pt')
    model, loss_history = train_model(model=model, model_path=model_path, dataset=X, loss_func=nn.BCELoss(),
                                      optimizer=optimizer, device=device, val_size=0.2, batch_size=32, epochs=10,
                                      shuffle=True)
    print(loss_history)


if __name__ == '__main__':
    # load criteo sample dataset
    dataset = Criteo(n_samples=10)
    X_idx, X_value, y, features, category, continues = dataset.get_features()
    print(features.get_num_feats(), features.get_num_fields(), features.get_feature_sizes())

    print(X_idx[:10], X_value[:10], y[:10])

    m = AFM(field_size=features.get_num_fields(), feature_sizes=features.get_feature_sizes(), is_shallow_dropout=False,
            use_cuda=False, weight_decay=0.00002, use_fm=True, use_ffm=False)

    m.fit(X_idx, X_value, y, early_stopping=True, save_path='afm_model_criteo.pt')
