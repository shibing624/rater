# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

from abc import abstractmethod

from sklearn.preprocessing import LabelEncoder, MinMaxScaler


class Feature:
    r"""General feature description for all types of feature

    :argument
        name (str): name of the feature, also refers to the column name in data (pd.Dataframe)
        start_idx: start index of the feature when it is transformed into index and value form
        dim: number of classes when it is transformed into index and value form
        proc_type (str): type of the feature
        processor: processor used in the preprocess stage.

    """

    def __init__(self, name, start_idx=None, proc_type='continuous', processor=None):
        self.name = name
        self.proc_type = proc_type
        self.start_idx = start_idx
        self.processor = processor

    @abstractmethod
    def get_idx_and_value(self, value):
        pass

    def __repr__(self):
        return 'feature name:{0}, start index:{1}, feature type:{2}'.format(self.name, self.start_idx, self.proc_type)


class ContinuousFeature(Feature):
    r"""Feature description for continuous feature

    :argument
        discretize: method of the feature to discretize, default is None
        discretize_bins: number of bins the feature discretize into
        transformation: method to transform the feature, default is MinMaxScaler. Note that the feature will be
                        transformed only when discretize=None
    """

    def __init__(self, name, transformation=None, discretize=None, discretize_bins=10):
        super(ContinuousFeature, self).__init__(name)
        if not transformation:
            transformation = MinMaxScaler()
        if discretize:
            if discretize not in ['eq_dist', 'eq_freq', 'cluster']:
                discretize = 'eq_freq'
            self.dim = discretize_bins
            self.bins = []
        else:
            self.transformation = transformation
            self.dim = 1
        self.discretize = discretize

    def get_idx_and_value(self, value):
        return self.start_idx, value


class CategoricalFeature(Feature):
    r"""Feature description for categorical feature

    :argument
        all_categories: give all categories of the feature when the description is created. The argument should be None
                        or list of str. Once the argument is not None, processor and dim is decided. By default the
                        categories is generated after scanning the column.
    """

    def __init__(self, name, all_categories=None):
        super(CategoricalFeature, self).__init__(name, proc_type='categorical')
        if all_categories:
            self.processor = LabelEncoder()
            self.processor.fit(all_categories)
            self.dim = len(self.processor.classes_)

    def get_idx_and_value(self, value):
        value = [value]
        idx = self.processor.transform(value)[0]
        return self.start_idx + idx, 1

    def __repr__(self):
        str_basic = super(CategoricalFeature, self).__repr__()
        if self.processor:
            return str_basic + ', feature dim:{0}, category encoder'.format(self.dim)
        else:
            return str_basic + ',category encoder not generated yet.'


class MultiCategoryFeature(Feature):
    def __init__(self, name, all_categories=None):
        super(MultiCategoryFeature, self).__init__(name, proc_type='multi_category')
        if all_categories:
            self.processor = LabelEncoder()
            self.processor.fit(all_categories)
            self.dim = len(self.processor.classes_)

    def get_idx_and_value(self, value):
        value = [value]
        values = self.processor.transform(value)[0]
        ids = range(self.start_idx, self.start_idx + self.dim)
        return ids, values

    def __repr__(self):
        str_basic = super(MultiCategoryFeature, self).__repr__()
        if self.processor:
            return str_basic + ', feature dim:{0}, category encoder'.format(self.dim)
        else:
            return str_basic + ',category encoder not generated yet.'
