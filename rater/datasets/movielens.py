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
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import sys

import pandas as pd
from six.moves import input

from . import download_builtin_dataset, BUILTIN_DATASETS
from ..features.feature_dict import FeatureDict, process_features


class Movielens:
    """`Movielens <http://files.grouplens.org/datasets/movielens/ml-100k.zip>`_ Dataset.

    If the dataset has not already been loaded, it will be downloaded and saved.

    Args:
        name(:obj:`string`): The name of the built-in dataset to load.
            Accepted values are 'ml-100k.zip', 'ml-1m.zip', and 'jester_dataset_2.zip'.
            Default is 'ml-100k.zip'.
        prompt(:obj:`bool`): Prompt before downloading if dataset is not
            already on disk.
            Default is True.

    Returns:
        A :obj:`Dataset` object.

    Raises:
        ValueError: If the ``name`` parameter is incorrect.

    """

    def __init__(self, name='ml-100k.zip', prompt=True, seed=None, shuffle=True, n_samples=-1):
        self.name = name
        self.shuffle = shuffle
        self.n_samples = n_samples

        try:
            dataset = BUILTIN_DATASETS[name]
        except KeyError:
            raise ValueError('unknown dataset ' + name +
                             '. Accepted values are ' +
                             ', '.join(BUILTIN_DATASETS.keys()) + '.')

        # if dataset does not exist, offer to download it
        if not os.path.isfile(dataset.path):
            answered = not prompt
            while not answered:
                print('Dataset ' + name + ' could not be found. Do you want '
                                          'to download it? [Y/n] ', end='')
                choice = input().lower()

                if choice in ['yes', 'y', '', 'ok', 'true']:
                    answered = True
                elif choice in ['no', 'n', 'false']:
                    print("Ok then, I'm out!")
                    sys.exit()

            download_builtin_dataset(name)

        self.line_format, self.sep = dataset.reader_params
        splitted_format = self.line_format.split()

        self.columns = ['user', 'item', 'rating']
        if 'timestamp' in splitted_format:
            self.with_timestamp = True
            self.columns.append('timestamp')
        else:
            self.with_timestamp = False

        # check that all fields are correct
        if any(field not in self.columns for field in splitted_format):
            raise ValueError('line_format parameter is incorrect.')

        self.data_file = dataset.path
        self.data = self.read_data(dataset.path)

    def read_data(self, file_name):
        """Return a list of ratings (user, item, rating, timestamp) read from file_name"""
        file_path = os.path.expanduser(file_name)
        data = pd.read_csv(file_path, delimiter=self.sep, header=None)
        data.columns = self.columns

        # Sample data
        if self.n_samples > 0:
            data = data.sample(n=self.n_samples)
            data.reset_index(drop=True, inplace=True)
        elif self.shuffle:
            data = data.sample(frac=1)
            data.reset_index(drop=True, inplace=True)

        # Add cols
        data[self.columns] = data[self.columns].fillna(0)
        return data

    def get_features(self, binarize=True):
        """
        Get feature dict
        :param binarize: bool
        :return:
        """
        # build feature instance
        features = FeatureDict()
        for column in ['user', 'item']:
            features.add_categorical_feat(column)

        X_idx, X_value = process_features(features, self.data)
        y = self.data.rating
        if binarize:
            def transform_y(label):
                if label > 3:
                    return 1
                else:
                    return 0

            y = y.apply(transform_y)
        return X_idx, X_value, y, features

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    name: {}\n'.format(self.name)
        fmt_str += '    data size: {}\n'.format(len(self.data))
        fmt_str += '    shuffle: {}\n'.format(self.shuffle)
        fmt_str += '    data file: {}\n'.format(self.data_file)
        fmt_str += '    line format: {}\n'.format(self.line_format)
        fmt_str += '    data head: {}\n'.format(self.data.head(n=1))
        return fmt_str
