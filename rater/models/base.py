# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Base class
"""

from abc import ABCMeta, abstractmethod


class ModelBase(object):
    """base class of recommendations"""

    __metaclass__ = ABCMeta

    @abstractmethod
    def fit(self, train, n_iters):
        """training models"""

    @abstractmethod
    def predict(self, data):
        """save model"""
