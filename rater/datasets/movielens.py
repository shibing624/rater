# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:

data format:

user	item	rating	timestamp
1	1193	5	978300760
1	661	3	978302109
1	914	3	978301968
1	3408	4	978300275
1	2355	5	97882429
"""

from functools import partial

import numpy as np
import scipy.sparse as sparse


def make_ratings(n_users, n_items, min_rating_per_user, max_rating_per_user,
                 rating_choices, seed=None, shuffle=True):
    """Randomly generate a (user_id, item_id, rating) array

    Return
    ------
        ndarray with shape (n_samples, 3)

    """
    if not (isinstance(rating_choices, list) or
                isinstance(rating_choices, tuple)):
        raise ValueError("'rating_choices' must be a list or tuple")
    if min_rating_per_user < 0 or min_rating_per_user >= n_items:
        raise ValueError("invalid 'min_rating_per_user' invalid")
    if (min_rating_per_user > max_rating_per_user) or \
            (max_rating_per_user >= n_items):
        raise ValueError("invalid 'max_rating_per_user' invalid")

    rs = np.random.RandomState(seed=seed)
    user_arrs = []
    for user_id in range(n_users):
        item_count = rs.randint(min_rating_per_user, max_rating_per_user)
        item_ids = rs.choice(n_items, item_count, replace=False)
        ratings = rs.choice(rating_choices, item_count)
        arr = np.stack(
            [np.repeat(user_id, item_count), item_ids, ratings], axis=1)
        user_arrs.append(arr)

    ratings = np.array(np.vstack(user_arrs))
    ratings[:, 2] = ratings[:, 2].astype('float')
    if shuffle:
        rs.shuffle(ratings)
    return ratings


def load_movielens_ratings(ratings_file, separator):
    with open(ratings_file) as f:
        ratings = []
        for line in f:
            line = line.split(separator)[:3]
            line = [int(l) for l in line]
            ratings.append(line)
        ratings = np.array(ratings)
    return ratings


load_movielens_1m_ratings = partial(load_movielens_ratings, separator="::")
load_movielens_100k_ratings = partial(load_movielens_ratings, separator="\t")


def build_user_item_matrix(n_users, n_items, ratings):
    """Build user-item matrix

    Return
    ------
        sparse matrix with shape (n_users, n_items)
    """
    data = ratings[:, 2]
    row_ind = ratings[:, 0]
    col_ind = ratings[:, 1]
    shape = (n_users, n_items)
    return sparse.csr_matrix((data, (row_ind, col_ind)), shape=shape)
