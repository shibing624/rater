# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
import lightgbm as lgb
import numpy as np
from lightgbm.sklearn import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import train_test_split


class GBDT:
    def __init__(self, num_leaves=31, max_depth=-1, n_estimators=100, min_data_in_leaf=20,
                 learning_rate=0.1, objective='binary'):
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.min_data_in_leaf = min_data_in_leaf
        self.learning_rate = learning_rate
        self.objective = objective
        if objective == 'binary':
            self.model = LGBMClassifier
        else:
            self.model = LGBMRegressor
        self.model = self.model(num_leaves=num_leaves, max_depth=max_depth,
                                n_estimators=n_estimators, learning_rate=learning_rate,
                                min_child_samples=min_data_in_leaf)

    def train(self, data, y, val_ratio=0.2, early_stopping_rounds=5):
        X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=val_ratio)
        val_set = lgb.Dataset(X_test, label=y_test, free_raw_data=False)
        self.model.fit(X_train, y_train, eval_set=val_set, early_stopping_rounds=early_stopping_rounds)

    def pred(self, data):
        pred_y, leaf_indices = self.model.predict(data, pred_leaf=True)
        base_idx = np.arange(0, self.num_leaves * self.n_estimators, self.n_estimators)
        feat_idx = base_idx + leaf_indices
        return pred_y, feat_idx
