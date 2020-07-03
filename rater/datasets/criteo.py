# -*- coding:utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: load sample file

data format:

label continuous_columns category_columns
0	1	1	5	0	1382	4	15	2	181	1	2		2	68fd1e64	80e26c9b	fb936136	7b4723c4

"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import sys

import pandas as pd
from six.moves import input

from . import download_builtin_dataset, BUILTIN_DATASETS
from ..features.feature_dict import FeatureDict, process_features


class Criteo:
    """`Criteo <http://labs.criteo.com/wp-content/uploads/2015/04/dac_sample.tar.gz>`_ Dataset.

    If the dataset has not already been loaded, it will be downloaded and saved.

    Args:
        name(:obj:`string`): The name of the built-in dataset to load.
            Accepted values are 'dac_sample.tar.gz', 'dac.tar.gz'.
            Default is dac_sample.tar.gz.
        prompt(:obj:`bool`): Prompt before downloading if dataset is not
            already on disk.
            Default is True.

    Returns:
        A :obj:`Dataset` object.

    Raises:
        ValueError: If the ``name`` parameter is incorrect.

    """

    def __init__(self, name='dac_sample.tar.gz', prompt=True, shuffle=False, n_samples=-1):
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

        # dac data format
        self.continuous_columns = ['I' + str(i) for i in range(1, 14)]
        self.category_columns = ['C' + str(i) for i in range(1, 27)]
        self.columns = ['label']
        self.columns.extend(self.continuous_columns)
        self.columns.extend(self.category_columns)

        self.data_file = dataset.path
        self.data = self.read_data(dataset.path)

    def read_data(self, file_name):
        """Return a list of data read from file_name"""
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

        # Fill Nan
        data[self.continuous_columns] = data[self.continuous_columns].fillna(0)
        data[self.category_columns] = data[self.category_columns].fillna('-1')

        return data

    def get_features(self, use_continuous_columns=True, use_category_columns=True, transformation=None,
                     discretize=None, discretize_bin=None, all_categories=None):
        """
        Get feature dict
        """
        # build feature instance
        features = FeatureDict()
        if use_continuous_columns:
            for column in self.continuous_columns:
                features.add_continuous_feat(column, transformation=transformation, discretize=discretize,
                                             discretize_bin=discretize_bin)
        if use_category_columns:
            for column in self.category_columns:
                features.add_categorical_feat(column, all_categories=all_categories)

        X_idx, X_value, categorical_index, continuous_value = process_features(features, self.data)
        y = self.data.label

        return features, X_idx, X_value, y, categorical_index, continuous_value

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    name: {}\n'.format(self.name)
        fmt_str += '    data size: {}\n'.format(len(self.data))
        fmt_str += '    shuffle: {}\n'.format(self.shuffle)
        fmt_str += '    data file: {}\n'.format(self.data_file)
        fmt_str += '    line format: {}\n'.format(self.line_format)
        fmt_str += '    data head: {}\n'.format(self.data.head(n=1))
        return fmt_str
