# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

from surprise import Dataset
from surprise import SVD
from surprise.model_selection import cross_validate

# 默认载入movielens数据集
data = Dataset.load_builtin('ml-100k')
print(data)
# k折交叉验证(k=3)
# data.split(n_folds=3)
# 试一把SVD矩阵分解
algo = SVD()
# 在数据集上测试一下效果
perf = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=3)
# 输出结果
print(perf)

from surprise.model_selection import cross_validate

data = Dataset.load_builtin('ml-100k')
### 使用NormalPredictor
from surprise import NormalPredictor

algo = NormalPredictor()
perf = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=3)
print(perf)

### 使用BaselineOnly
from surprise import BaselineOnly

algo = BaselineOnly()
perf = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=3)
print(perf)

### 使用基础版协同过滤
from surprise import KNNBasic

algo = KNNBasic()
perf = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=3)
print(perf)

### 使用均值协同过滤
from surprise import KNNWithMeans

algo = KNNWithMeans()
perf = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=3)
print(perf)

### 使用协同过滤baseline
from surprise import KNNBaseline

algo = KNNBaseline()
perf = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=3)
print(perf)

### 使用SVD
from surprise import SVD

algo = SVD()
perf = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=3)
print(perf)

### 使用NMF
from surprise import NMF

algo = NMF()
perf = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=3)
print(perf)
