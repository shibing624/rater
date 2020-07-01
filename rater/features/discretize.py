# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def discretize(data_series, disc_method, bins):
    if disc_method == 'eq_dist':
        discrete, intervals = pd.cut(data_series, bins=bins, labels=range(bins), retbins=True)
        return discrete.values, intervals
    elif disc_method == 'eq_freq':
        discrete, intervals = pd.qcut(data_series, q=bins, labels=range(bins), retbins=True, duplicates='drop')
        return discrete.values, intervals
    elif disc_method == 'cluster':
        data = np.reshape(data_series, (-1, 1))
        kmeans = KMeans(n_clusters=bins)
        ret_data = kmeans.fit_transform(data)
        return np.reshape(ret_data, -1), None
