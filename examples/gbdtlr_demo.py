# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import os
import sys

import torch
import torch.nn as nn
from torch.utils.data.dataset import TensorDataset

sys.path.append("..")
from rater.datasets.criteo import Criteo
from rater.models.ctr.gbdt_lr import GBDTLR
from rater.models.model import train_model

pwd_path = os.path.abspath(os.path.dirname(__file__))


def train(x_idx, x_value, label):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    X_idx_tensor = torch.LongTensor(x_idx).to(device)
    X_value_tensor = torch.Tensor(x_value).to(device)
    y_tensor = torch.Tensor(label).to(device)
    y_tensor = y_tensor.reshape(-1, 1)

    X = TensorDataset(X_idx_tensor, y_tensor)
    model = GBDTLR().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    model_path = os.path.join(pwd_path, 'gbdtlr_model.pt')
    model.train_gbdt(x_idx.values.tolist(), label.values.tolist())

    model, loss_history = train_model(model=model, model_path=model_path, dataset=X, loss_func=nn.BCELoss(),
                                      optimizer=optimizer, device=device, val_size=0.2, batch_size=32, epochs=10)
    print(loss_history)


def lightgbm_demo():
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=800)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
    print('X_train[:2]:', X_train[:2])
    print('y_train[:2]:', y_train[:2])
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

    # specify your configurations as a dict
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'binary_logloss'},
        'num_leaves': 63,
        'num_trees': 100,
        'learning_rate': 0.01,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0
    }

    # number of leaves,will be used in feature transformation
    num_leaf = 63

    print('Start training...')
    # train
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=100,
                    valid_sets=lgb_eval)

    print('Save model...')
    # save model to file
    gbm.save_model('model.txt')

    print('Start predicting...')
    # predict and get data on leaves, training data
    y_pred = gbm.predict(X_train, pred_leaf=True)

    # feature transformation and write result
    print('Writing transformed training data')
    transformed_training_matrix = np.zeros([len(y_pred), len(y_pred[0]) * num_leaf], dtype=np.int64)
    for i in range(0, len(y_pred)):
        temp = np.arange(len(y_pred[0])) * num_leaf - 1 + np.array(y_pred[i])
        transformed_training_matrix[i][temp] += 1

    # for i in range(0,len(y_pred)):
    #	for j in range(0,len(y_pred[i])):
    #		transformed_training_matrix[i][j * num_leaf + y_pred[i][j]-1] = 1

    # predict and get data on leaves, testing data
    y_pred = gbm.predict(X_test, pred_leaf=True)

    # feature transformation and write result
    print('Writing transformed testing data')
    transformed_testing_matrix = np.zeros([len(y_pred), len(y_pred[0]) * num_leaf], dtype=np.int64)
    for i in range(0, len(y_pred)):
        temp = np.arange(len(y_pred[0])) * num_leaf - 1 + np.array(y_pred[i])
        transformed_testing_matrix[i][temp] += 1

    # for i in range(0,len(y_pred)):
    #	for j in range(0,len(y_pred[i])):
    #		transformed_testing_matrix[i][j * num_leaf + y_pred[i][j]-1] = 1

    print('Calculate feature importances...')
    # feature importances
    print('Feature importances:', list(gbm.feature_importance()))
    print('Feature importances:', list(gbm.feature_importance("gain")))

    # Logestic Regression Start
    print("Logestic Regression Start")

    # load or create your dataset
    print('Load data...')

    lm = LogisticRegression(penalty='l2')  # logestic model construction
    lm.fit(transformed_training_matrix, y_train)  # fitting the data

    # y_pred_label = lm.predict(transformed_training_matrix )  # For training data
    y_pred_label = lm.predict(transformed_testing_matrix)  # For testing data
    # y_pred_est = lm.predict_proba(transformed_training_matrix)   # Give the probabilty on each label
    y_pred_est = lm.predict_proba(transformed_testing_matrix)  # Give the probabilty on each label

    print('number of testing data is ' + str(len(y_pred_label)))
    print(y_pred_est)

    # calculate predict accuracy
    num = 0
    for i in range(0, len(y_pred_label)):
        if y_test[i] == y_pred_label[i]:
            if y_train[i] == y_pred_label[i]:
                num += 1
    print("prediction accuracy is " + str((num) / len(y_pred_label)))

    # Calculate the Normalized Cross-Entropy
    # for testing data
    NE = (-1) / len(y_pred_est) * sum(
        ((1 + y_test) / 2 * np.log(y_pred_est[:, 1]) + (1 - y_test) / 2 * np.log(1 - y_pred_est[:, 1])))
    # for training data
    # NE = (-1) / len(y_pred_est) * sum(((1+y_train)/2 * np.log(y_pred_est[:,1]) +  (1-y_train)/2 * np.log(1 - y_pred_est[:,1])))
    print("Normalized Cross Entropy " + str(NE))

    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_test, y_pred_label)
    print("auc:", auc)


def criteo_gdbtlr(X_idx, X_value, y):
    import numpy as np
    from sklearn.metrics import roc_auc_score, accuracy_score
    from sklearn.linear_model import LogisticRegression
    from lightgbm.sklearn import LGBMClassifier

    X_idx = X_idx.values.tolist()
    y = y.values.tolist()
    num_leaves = 31
    model = LGBMClassifier(num_leaves=num_leaves)
    model.fit(X_idx, y)
    model_path = os.path.join(pwd_path, 'gbdtlr_model1.pt')
    y_pred = model.predict(X_idx, pred_leaf=True)
    y_pred_gbdt = model.predict(X_idx, pred_leaf=False)
    acc = model.score(X_idx, y)
    print("gbdt train acc:", acc)
    s = roc_auc_score(y, y_pred_gbdt)
    print('gbdt auc:', s)
    a = accuracy_score(y, y_pred_gbdt)
    print('gbdt train acc:', a)
    import pickle  # pickle模块

    # 保存Model(注:save文件夹要预先建立，否则会报错)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    # # 读取Model
    # with open('save/clf.pickle', 'rb') as f:
    #     clf2 = pickle.load(f)

    transformed_matrix = np.zeros([len(y_pred), len(y_pred[0]) * num_leaves], dtype=np.int64)
    for i in range(0, len(y_pred)):
        temp = np.arange(len(y_pred[0])) * num_leaves - 1 + np.array(y_pred[i])
        transformed_matrix[i][temp] += 1

    lr_model = LogisticRegression()
    lr_model.fit(transformed_matrix, y)
    y_pred_lr = lr_model.predict(transformed_matrix)
    print("truth_y:", y[:100], 'y_pred_lr:', y_pred_lr[:100])

    s = roc_auc_score(y, y_pred_lr)
    print('auc:', s)


if __name__ == '__main__':
    # lightgbm_demo()
    # exit()
    # load criteo sample dataset
    dataset = Criteo(n_samples=-1)
    features, X_idx, X_value, y, category_index, continuous_value = dataset.get_features()

    print("X_idx[0], X_value[0], y[0] :\n", X_idx[0], X_value[0], y[0])

    criteo_gdbtlr(X_idx, X_value, y)
    exit()

    train(X_idx, X_value, y)
