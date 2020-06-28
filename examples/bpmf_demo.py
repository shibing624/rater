# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import os

import numpy as np

import rater
from rater.datasets.movielens import load_movielens_1m_ratings
from rater.metrics import RMSE
from rater.models.bpmf import BPMF
from rater.utils.get_file import get_file
from rater.utils.logger import logger

ML_1M_URL = "http://files.grouplens.org/datasets/movielens/ml-1m.zip"
rand_state = np.random.RandomState(0)

# load or download MovieLens 1M dataset
download_file = get_file(rater.movielens_1m_path, ML_1M_URL, extract=True, cache_dir=rater.USER_DIR,
                         cache_subdir=rater.USER_DATA_DIR)
logger.info(download_file)
rating_file = os.path.join(rater.movielens_1m_dir, 'ratings.dat')
ratings = load_movielens_1m_ratings(rating_file)
n_user = max(ratings[:, 0])
n_item = max(ratings[:, 1])

# shift user_id & movie_id by 1. let user_id & movie_id start from 0
ratings[:, (0, 1)] -= 1

# split data to training & testing
train_pct = 0.9
rand_state.shuffle(ratings)
train_size = int(train_pct * ratings.shape[0])
train = ratings[:train_size]
validation = ratings[train_size:]

# models settings
n_feature = 10
epochs = 20
print("n_user: %d, n_item: %d, n_feature: %d, training size: %d, validation size: %d" % (
    n_user, n_item, n_feature, train.shape[0], validation.shape[0]))
bpmf = BPMF(n_user=n_user, n_item=n_item, n_feature=n_feature,
            max_rating=5., min_rating=1., seed=0)

bpmf.fit(train, epochs=epochs)
train_preds = bpmf.predict(train[:, :2])
train_rmse = RMSE().compute(train[:, 2], train_preds)
val_preds = bpmf.predict(validation[:, :2])
val_rmse = RMSE().compute(validation[:, 2], val_preds)
print("after %d epochs, train RMSE: %.6f, validation RMSE: %.6f" %
      (epochs, train_rmse, val_rmse))
