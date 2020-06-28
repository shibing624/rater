# -*- coding:utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: load sample file
"""

import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from rater.layers.inputs import SparseFeat, DenseFeat, get_feature_names

continuous_columns = ['I' + str(i) for i in range(1, 14)]
category_columns = ['C' + str(i) for i in range(1, 27)]

def load_data(path, n_samples=-1):
    """
    Load sample file to DataFrame
    :param path:
    :param n_samples:
    :return: X
    """
    data = pd.read_csv(path, delimiter='\t', header=None)

    columns = ['y']

    columns.extend(continuous_columns)
    columns.extend(category_columns)
    data.columns = columns

    if n_samples > 0:
        data = data.sample(n=n_samples)
        data.reset_index(drop=True, inplace=True)

    # fill NaN
    data[continuous_columns] = data[continuous_columns].fillna(0)
    data[category_columns] = data[category_columns].fillna('0')
    return data


def load_criteo_file(path, n_samples=10000):
    """
    An example for load criteo dataset
    :param path: File path of criteo dataset.
    :param n_samples: Number to sample from the full dataset. n_samples <= 0 means not to sample.
    :return: X
    """
    data = load_data(path, n_samples=n_samples)
    # 1.Label Encoding for sparse features, and do simple Transformation for dense features
    for col in category_columns:
        lbe = LabelEncoder()
        data[col] = lbe.fit_transform(data[col])
    mms = MinMaxScaler(feature_range=(0, 1))
    data[continuous_columns] = mms.fit_transform(data[continuous_columns])

    # 2.Count unique features for each sparse field, and record dense feature field name
    total_feature_columns = [SparseFeat(feat, data[feat].nunique()) for feat in category_columns] + \
                             [DenseFeat(feat, 1, ) for feat in continuous_columns]

    feature_names = get_feature_names(total_feature_columns)

    # 3.generate input data for model
    train, test = train_test_split(data, test_size=0.2)

    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}

    y = data.y
    return train_model_input, test_model_input, y, train, test


if __name__ == '__main__':
    # file_path = '../../examples/criteo_sample_1w.txt'
    # data = load_criteo_file(file_path)

    import xlearn as xl

    # 训练
    ffm_model = xl.create_ffm()  # 使用FFM模型
    ffm_model.setTrain("./FFM_train.txt")  # 训练数据
    ffm_model.setTrain("../../examples/test.txt")  # 训练数据
    ffm_model.setValidate("../../examples/test.txt")  # 校验测试数据

    # param:
    #  0. binary classification
    #  1. learning rate: 0.2
    #  2. regular lambda: 0.0022
    #  3. evaluation metric: accuracy
    param = {'task': 'binary', 'lr': 0.2,
             'lambda': 0.002, 'metric': 'acc'}

    # 开始训练
    ffm_model.fit(param, './model.out')

    # 预测
    ffm_model.setTest("../../examples/test.txt")  # 测试数据
    ffm_model.setSigmoid()  # 归一化[0,1]之间

    # 开始预测

    ffm_model.predict("./model.out", "./output.txt")