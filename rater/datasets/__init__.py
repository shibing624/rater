# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: download dataset
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
from collections import namedtuple

from rater.utils.get_file import get_file


def get_dataset_dir():
    """
    Return folder where downloaded datasets and other data are stored.
    Default folder is ~/.rater_data/
    :return:
    """
    folder = os.environ.get('RATER_DATA_FOLDER', os.path.expanduser('~') +
                            '/.rater_datasets/')
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder


# a builtin dataset has
# - an url (where to download it)
# - a path (where it is located on the filesystem)
# - the parameters of the corresponding reader
BuiltinDataset = namedtuple('BuiltinDataset', ['url', 'path', 'reader_params'])

BUILTIN_DATASETS = {
    'ml-100k.zip':
        BuiltinDataset(
            url='http://files.grouplens.org/datasets/movielens/ml-100k.zip',
            path=os.path.join(get_dataset_dir(), 'ml-100k/u.data'),
            reader_params=('user item rating timestamp', '\t')
        ),
    'ml-1m.zip':
        BuiltinDataset(
            url='http://files.grouplens.org/datasets/movielens/ml-1m.zip',
            path=os.path.join(get_dataset_dir(), 'ml-1m/ratings.dat'),
            reader_params=('user item rating timestamp', '::')
        ),
    'jester_dataset_2.zip':
        BuiltinDataset(
            url='http://eigentaste.berkeley.edu/dataset/archive/jester_dataset_2.zip',
            path=os.path.join(get_dataset_dir(), 'jester/jester_ratings.dat'),
            reader_params=('user item rating', '\t')
        ),
    'dac_sample.tar.gz':
        BuiltinDataset(
            url='http://labs.criteo.com/wp-content/uploads/2015/04/dac_sample.tar.gz',
            path=os.path.join(get_dataset_dir(), 'dac_sample.txt'),
            reader_params=('label continuous_columns category_columns', '\t')
        ),
    'dac.tar.gz':
        BuiltinDataset(
            url='https://s3-eu-west-1.amazonaws.com/kaggle-display-advertising-challenge-dataset/dac.tar.gz',
            path=os.path.join(get_dataset_dir(), 'dac.txt'),
            reader_params=('label continuous_columns category_columns', '\t')
        ),

}


def download_builtin_dataset(name):
    dataset = BUILTIN_DATASETS[name]

    print('Trying to download dataset from ' + dataset.url + '...')
    file_name = get_file(name, dataset.url, extract=True, cache_subdir='', cache_dir=get_dataset_dir())

    print('Done! Dataset ' + name + 'has been saved to ' + file_name)
