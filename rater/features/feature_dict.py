# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .discretize import discretize
from .feature import ContinuousFeature, CategoricalFeature, MultiCategoryFeature
from ..utils.logger import logger


class FeatureDict:
    def __init__(self):
        self.continuous_feats = {}
        self.categorical_feats = {}
        self.multi_category_feats = {}
        self.feat_dict = {}

    def add_continuous_feat(self, name, transformation=None, discretize=None, discretize_bin=10):
        self.delete_feat(name)
        self.continuous_feats[name] = ContinuousFeature(name, transformation, discretize, discretize_bin)
        self.feat_dict[name] = 'continuous'

    def add_categorical_feat(self, name, all_categories=None):
        self.delete_feat(name)
        self.categorical_feats[name] = CategoricalFeature(name, all_categories)
        self.feat_dict[name] = 'categorical'

    def add_multi_category_feat(self, name, all_categories=None):
        self.delete_feat(name)
        self.multi_category_feats[name] = MultiCategoryFeature(name, all_categories)
        self.feat_dict[name] = 'multi_category'

    def delete_feat(self, name):
        if name in self.feat_dict:
            feat_type = self.feat_dict[name]
            if feat_type == 'continuous':
                del self.continuous_feats[name]
            elif feat_type == 'categorical':
                del self.categorical_feats[name]
            elif feat_type == 'multi_category':
                del self.multi_category_feats[name]

    def feature_size(self):
        total_size = 0
        total_size += len(self.continuous_feats)
        for key in self.categorical_feats:
            feat = self.categorical_feats[key]
            total_size += feat.dim

        for key in self.multi_category_feats:
            feat = self.multi_category_feats[key]
            total_size += feat.dim
        return total_size

    def field_size(self):
        """
        Num of features keys
        :return: int
        """
        return len(self.feat_dict)

    def field_range(self):
        fields = []
        for k,v in self.continuous_feats.items():
            fields.append(v.dim)
        for k,v in self.categorical_feats.items():
            fields.append(v.dim)
        for k,v in self.multi_category_feats.items():
            fields.append(v.dim)
        return fields

    def __repr__(self):
        feats_list = [self.continuous_feats, self.categorical_feats, self.multi_category_feats]
        info_strs = []
        for feats in feats_list:
            info_str = ''
            for key in feats:
                feat = feats[key]
                info_str += str(feat)
                info_str += '\n'
            info_strs.append(info_str)
        return 'Continuous Features:\n{}Categorical Features:\n{}Multi-Category Features:\n{}'.format(*info_strs)


def process_features(features: FeatureDict, data: pd.DataFrame):
    r"""Transform raw data into index and value form.
    Continuous features will be discretized, standardized, normalized or scaled according to feature meta.
    Categorical features will be encoded with a label encoder.


    :param features: The FeatureList instance that describes raw_data.
    :param data: The raw_data to be transformed.
    :return: feat_index, feat_value, category_index, continuous_value (DataFrame)
    """
    logger.info('process_features start')
    continuous_feats = features.continuous_feats
    categorical_feats = features.categorical_feats
    columns = list(continuous_feats.keys())
    columns.extend(list(categorical_feats.keys()))
    data = data[columns]
    feat_idx = pd.DataFrame()

    # transform continuous features
    logger.info('transforming continuous features')
    feat_value_continuous = pd.DataFrame()
    idx = 0
    for name in continuous_feats:
        feat = continuous_feats[name]
        feat.start_idx = idx
        if feat.discretize:
            # use discretize
            discrete_data, intervals = discretize(data[name], feat.discretize, feat.dim)
            feat.bins = intervals
            feat_idx[name] = discrete_data + idx
            feat_value_continuous[name] = pd.Series(np.ones(len(data[name])))
            idx += feat.dim
        else:
            # standardized, normalize or MinMaxScaler
            processor = feat.transformation
            col_data = np.reshape(data[name].values, (-1, 1))
            col_data = processor.fit_transform(col_data)
            col_data = np.reshape(col_data, -1)
            feat_value_continuous[name] = col_data
            feat_idx[name] = np.repeat(idx, repeats=len(data))
            idx += 1

    logger.info('transforming categorical features')
    # transform categorical features
    categorical_index = pd.DataFrame()
    for name in categorical_feats:
        categorical_feat = categorical_feats[name]
        le = LabelEncoder()
        feat_idx[name] = le.fit_transform(data[name]) + idx
        categorical_index[name] = feat_idx[name]
        categorical_feat.processor = le
        num_classes = len(le.classes_)
        categorical_feat.dim = num_classes
        categorical_feat.start_idx = idx
        idx += num_classes

    feat_idx = feat_idx.apply(lambda x: x.values, axis=1)
    categorical_index = categorical_index.apply(lambda x: x.values, axis=1)

    feat_value_category = pd.DataFrame(np.ones((len(data), len(categorical_feats))))
    feat_value = pd.concat([feat_value_continuous, feat_value_category], axis=1)
    feat_value = feat_value.apply(lambda x: x.values, axis=1)
    continuous_value = feat_value_continuous.apply(lambda x: x.values, axis=1)

    logger.info('features info:{}'.format(features))
    logger.info('process_features end')
    return feat_idx, feat_value, categorical_index, continuous_value
