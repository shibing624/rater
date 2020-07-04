# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: GBDT impl with lightgbm
"""
import numpy as np
from lightgbm.sklearn import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import train_test_split


class GBDT:
    def __init__(self, num_leaves=31, max_depth=-1, n_estimators=100,
                 learning_rate=0.1, out_type='binary'):
        """
        Init model
        :param num_leaves:
        :param max_depth:
        :param n_estimators:
        :param learning_rate:
        :param out_type:
        """
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        if out_type == 'binary':
            cls = LGBMClassifier
        else:
            cls = LGBMRegressor
        self.model = cls(num_leaves=num_leaves, max_depth=max_depth,
                         n_estimators=n_estimators, learning_rate=learning_rate)

    def train(self, data, y):
        """
        Train model
        :param data:
        :param y:
        :return: object, Returns self.
        """
        # X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=val_ratio)
        # self.model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=early_stopping_rounds)
        self.model.fit(data, y)

    def pred(self, data):
        """
        Predict data
        :param data:
        :return: pred_y, feature_index
        """
        y_pred = self.model.predict(data, pred_leaf=True)

        transformed_matrix = np.zeros([len(y_pred), len(y_pred[0]) * self.num_leaves], dtype=np.int64)
        for i in range(0, len(y_pred)):
            temp = np.arange(len(y_pred[0])) * self.num_leaves - 1 + np.array(y_pred[i])
            transformed_matrix[i][temp] += 1
        return y_pred, transformed_matrix
