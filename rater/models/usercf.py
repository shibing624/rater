# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 基于用户的协同过滤算法
"""

import math
import os
import random

from sklearn.model_selection import train_test_split

import recommender
from recommender.utils.logger import timer


class Dataset:
    """
    load data and split data
    """

    def __init__(self, fp):
        # fp: data file path
        self.data = self.load_data(fp)

    @timer
    def load_data(self, fp):
        data = []
        with open(fp) as f:
            for l in f:
                data.append(tuple(map(int, l.strip().split('::')[:2])))
        return data

    @timer
    def split_data(self, seed=0):
        """
        :params: data, 加载的所有(user, item)数据条目
        :params: seed, random的种子数，对于不同的k应设置成一样的
        :return: train, test
        """
        train, test = train_test_split(self.data, test_size=0.2, random_state=seed)

        # 处理成字典的形式，user->set(items)
        def convert_dict(data):
            data_dict = {}
            for user, item in data:
                if user not in data_dict:
                    data_dict[user] = set()  # 物品集合，去重
                data_dict[user].add(item)
            data_dict = {k: list(data_dict[k]) for k in data_dict}  # 物品集合转为列表
            return data_dict

        return convert_dict(train), convert_dict(test)


# ### 2. 评价指标
# 1. Precision
# 2. Recall
# 3. Coverage
# 4. Popularity(Novelty)

# TopN推荐，不关心用户具体的评分，只预测用户是否会对某部电影评分
class Metric:
    def __init__(self, train, test, get_recommendation):
        """
        :params: train, 训练数据，在定义覆盖率、新颖度时会用到
        :params: test, 测试数据
        :params: GetRecommendation, 为某个用户获取推荐物品的接口函数
        """
        self.train = train
        self.test = test
        self.get_recommendation = get_recommendation
        self.recs = self.get_test_rec()

    # 为test中的每个用户进行推荐
    def get_test_rec(self):
        recs = {}
        for user in self.test:
            rank = self.get_recommendation(user)  # 推荐列表
            recs[user] = rank
        return recs

    # 定义精确率指标计算方式
    def precision(self):
        total, hit = 0, 0
        for user in self.test:
            test_items = set(self.test[user])  # true list
            rank = self.recs[user]  # recommend list
            for item, score in rank:
                if item in test_items:
                    hit += 1  # 命中
            total += len(rank)
        return round(hit / total, 4)

    # 定义召回率指标计算方式
    def recall(self):
        total, hit = 0, 0
        for user in self.test:
            test_items = set(self.test[user])
            rank = self.recs[user]
            for item, score in rank:
                if item in test_items:
                    hit += 1
            total += len(test_items)
        return round(hit / total, 4)

    # 定义覆盖率指标计算方式：覆盖率反映了推荐算法发掘长尾的能力
    def coverage(self):
        all_item, recom_item = set(), set()
        for user in self.test:
            for item in self.train[user]:
                all_item.add(item)  # 注意all_item只能累计训练集的item
            rank = self.recs[user]
            for item, score in rank:
                recom_item.add(item)
        return round(len(recom_item) / len(all_item), 4)

    # 定义新颖度指标计算方式:平均流行度越低，新颖性越高(物品的流行度是指 对物品产生过行为的用户总数)
    def popularity(self):
        # 计算物品的流行度
        item_pop = {}
        for user in self.train:
            for item in self.train[user]:
                if item not in item_pop:
                    item_pop[item] = 0
                item_pop[item] += 1

        num, pop = 0, 0
        for user in self.test:
            rank = self.recs[user]
            for item, score in rank:
                # 物品的流行度满足长尾分布，取对数后流行度的平均值更加稳定，防止被流行物品所主导（削弱流行物品的影响）
                pop += math.log(1 + item_pop[item])
                num += 1  # 对推荐列表计数
        return round(pop / num, 4)  # 平均流行度

    def eval(self):
        metric = {'Precision': self.precision(),
                  'Recall': self.recall(),
                  'Coverage': self.coverage(),
                  'Popularity': self.popularity()}
        print('Metric:', metric)
        return metric


# ## 二. 算法实现
# 1. Random
# 2. MostPopular
# 3. UserCF：UserCollaborationFilter
# 4. UserIIF

# In[5]:


# 1. 随机推荐
def random_alg(train, K, N):
    """
    :params: train, 训练数据集
    :params: K, 可忽略
    :params: N, 超参数，设置取TopN推荐物品数目
    :return: recommendation，推荐接口函数
    """
    items = {}
    for user in train:
        for item in train[user]:
            items[item] = 1

    def recommendation(user):
        # 随机推荐N个未见过的
        user_items = set(train[user]) if user in train else set()
        rec_items = {k: items[k] for k in items if k not in user_items}
        rec_items = list(rec_items.items())
        random.shuffle(rec_items)
        return rec_items[:N]

    return recommendation


# 2. 热门推荐
def most_popular_alg(train, K, N):
    """
    :params: train, 训练数据集
    :params: K, 可忽略
    :params: N, 超参数，设置取TopN推荐物品数目
    :return: GetRecommendation, 推荐接口函数
    """
    items = {}
    for user in train:
        for item in train[user]:
            if item not in items:
                items[item] = 0
            items[item] += 1  # 统计物品频率

    def recommendation(user):
        # 推荐N个没见过的最热门的
        user_items = set(train[user])
        rec_items = {k: items[k] for k in items if k not in user_items}
        rec_items = list(sorted(rec_items.items(), key=lambda x: x[1], reverse=True))
        return rec_items[:N]  # topN最热门

    return recommendation


# 3. 基于用户余弦相似度的推荐
def user_cf_alg(train, K, N):
    """
    :params: train, 训练数据集
    :params: K, 超参数，设置取TopK相似用户数目
    :params: N, 超参数，设置取TopN推荐物品数目
    :return: recommendation, 推荐接口函数
    """
    # 计算item->user的倒排索引
    item_users = {}
    for user in train:
        for item in train[user]:
            if item not in item_users:
                item_users[item] = set()  # 集合，去重
            item_users[item].add(user)
    item_users = {k: list(v) for k, v in item_users.items()}

    # 计算用户相似度矩阵：calculate co-rated items between users
    sim = {}
    num = {}
    for item in item_users:
        users = item_users[item]
        for i in range(len(users)):
            u = users[i]
            if u not in num:
                num[u] = 0
            num[u] += 1
            if u not in sim:
                sim[u] = {}
            for j in range(len(users)):
                if j == i: continue
                v = users[j]
                if v not in sim[u]:
                    sim[u][v] = 0
                sim[u][v] += 1
    for u in sim:
        for v in sim[u]:
            sim[u][v] /= math.sqrt(num[u] * num[v])

    # 按照相似度排序
    sorted_user_sim = {k: list(sorted(v.items(), key=lambda x: x[1], reverse=True)) for k, v in sim.items()}

    # 获取接口函数：给user推荐与其最相似的K个用户喜欢的物品i(排除掉user已见的)，按照喜欢物品i的用户u与user的累计相似度排序
    def recommendation(user):
        items = {}
        seen_items = set(train[user])
        for u, _ in sorted_user_sim[user][:K]:
            for item in train[u]:
                # 要去掉用户见过的
                if item not in seen_items:
                    if item not in items:
                        items[item] = 0
                    items[item] += sim[user][u]  # 累计用户相似度
        recs = list(sorted(items.items(), key=lambda x: x[1],
                           reverse=True))[:N]
        return recs

    return recommendation


# 4. 基于改进的用户余弦相似度的推荐：两个用户对冷门物品采取过同样的行为更能说明他们兴趣的相似度，按物品的流行度进行惩罚
# IIF：inverse item frequency
def user_iif_alg(train, K, N):
    """
    :params: train, 训练数据集
    :params: K, 超参数，设置取TopK相似用户数目
    :params: N, 超参数，设置取TopN推荐物品数目
    :return: recommendation, 推荐接口函数
    """
    # 计算item->user的倒排索引
    item_users = {}
    for user in train:
        for item in train[user]:
            if item not in item_users:
                item_users[item] = set()  # 集合，去重
            item_users[item].add(user)
    item_users = {k: list(v) for k, v in item_users.items()}

    # 计算用户相似度矩阵
    sim = {}
    num = {}
    for item in item_users:
        users = item_users[item]
        for i in range(len(users)):
            u = users[i]
            if u not in num:
                num[u] = 0
            num[u] += 1
            if u not in sim:
                sim[u] = {}
            for j in range(len(users)):
                if j == i: continue
                v = users[j]
                if v not in sim[u]:
                    sim[u][v] = 0
                # 相比UserCF，主要是改进了这里, len(users)表示u,v共同爱好的物品一共有多少人喜欢(流行度)
                # 如果该物品本身就很热门，则无法说明u,v的相似性
                # 反之，如果该物品很冷门，则更能说明u,v的相似性
                sim[u][v] += 1 / math.log(1 + len(users))
    for u in sim:
        for v in sim[u]:
            sim[u][v] /= math.sqrt(num[u] * num[v])

    # 按照相似度排序
    sorted_user_sim = {k: list(sorted(v.items(), key=lambda x: x[1], reverse=True)) for k, v in sim.items()}

    # 获取接口函数
    def recommendation(user):
        items = {}
        seen_items = set(train[user])
        for u, _ in sorted_user_sim[user][:K]:
            for item in train[u]:
                # 要去掉用户见过的
                if item not in seen_items:
                    if item not in items:
                        items[item] = 0
                    items[item] += sim[user][u]
        recs = list(sorted(items.items(), key=lambda x: x[1], reverse=True))[:N]
        return recs

    return recommendation


# ## 三. 实验
# 1. Random实验
# 2. MostPopular实验
# 3. UserCF实验，K=[5, 10, 20, 40, 80, 160]
# 4. UserIIF实验, K=80

class Experiment:
    def __init__(self, M, K, N, fp=os.path.join(recommender.movielens_1m_dir, 'ratings.dat'), name='UserCF'):
        '''
        :params: M, 进行多少次实验
        :params: K, TopK相似用户的个数
        :params: N, TopN推荐物品的个数
        :params: fp, 数据文件路径
        :params: rt, 推荐算法类型
        '''
        self.M = M
        self.K = K
        self.N = N
        self.fp = fp

        self.name = name
        self.algs = {'Random': random_alg, 'MostPopular': most_popular_alg, 'UserCF': user_cf_alg,
                     'UserIIF': user_iif_alg}
        self.dataset = Dataset(self.fp)

    # 定义单次实验
    @timer
    def worker(self, train, test):
        """
        :params: train, 训练数据集
        :params: test, 测试数据集
        :return: 各指标的值
        """
        recommendation = self.algs[self.name](train, self.K, self.N)
        metric = Metric(train, test, recommendation)
        return metric.eval()

    # 多次实验取平均
    @timer
    def run(self):
        metrics = {'Precision': 0, 'Recall': 0,
                   'Coverage': 0, 'Popularity': 0}
        for i in range(self.M):
            train, test = self.dataset.split_data(i)
            print('Experiment {}:'.format(i))
            metric = self.worker(train, test)
            metrics = {k: metrics[k] + metric[k] for k in metrics}
        metrics = {k: metrics[k] / self.M for k in metrics}
        print('Average Result (M={}, K={}, N={}): {}'.format(self.M, self.K, self.N, metrics))


if __name__ == '__main__':
    # 1. random实验：precision和recall很低，覆盖率100%
    print('*' * 42)
    print('random:')
    M, N = 1, 10
    K = 10  # 为保持一致而设置，随便填一个值
    random_exp = Experiment(M, K, N, name='Random')
    random_exp.run()  # 注意随机推荐的覆盖率应该是100%，实验结果中有的超过了100是因为在all_items中只统计了训练集

    # 2. MostPopular实验：precision和recall较高，但覆盖率很低，流行度很高
    print('*' * 42)
    print('most popular:')
    M, N = 1, 10
    K = 10  # 为保持一致而设置，随便填一个值
    mp_exp = Experiment(M, K, N, name='MostPopular')
    mp_exp.run()

    # 3. UserCF实验：注意K值的影响
    print('*' * 42)
    print('user cf:')
    M, N = 1, 10
    for K in [5, 10, 20, 40, 80]:
        cf_exp = Experiment(M, K, N, name='UserCF')
        cf_exp.run()

    # 4. UserIIF实验
    print('*' * 42)
    print('user iif:')
    M, N = 1, 10
    K = 80  # 与书中保持一致
    iif_exp = Experiment(M, K, N, name='UserIIF')
    iif_exp.run()


# ## 四. 实验结果
#
# 1. Random实验
#
#     Running time: 185.54872608184814
#
#     Average Result (M=8, K=0, N=10):
#     {'Precision': 0.61, 'Recall': 0.29,
#      'Coverage': 100.0, 'Popularity': 4.38958}
#
# 2. MostPopular实验
#
#     Running time: 103.3697898387909
#
#     Average Result (M=8, K=0, N=10):
#     {'Precision': 12.83, 'Recall': 6.16,
#     'Coverage': 2.43, 'Popularity': 7.72326}
#
# 3. UserCF实验
#
#     Running time: 1456.9617431163788
#
#     Average Result (M=8, K=5, N=10):
#     {'Precision': 16.89, 'Recall': 8.11,
#      'Coverage': 52.09, 'Popularity': 6.8192915}
#
#     Running time: 1416.0529160499573
#
#     Average Result (M=8, K=10, N=10):
#     {'Precision': 20.46, 'Recall': 9.83,
#      'Coverage': 41.64, 'Popularity': 6.979140375}
#
#     Running time: 1463.8790090084076
#
#     Average Result (M=8, K=20, N=10):
#     {'Precision': 22.99, 'Recall': 11.04,
#      'Coverage': 32.78, 'Popularity': 7.102363}
#
#     Running time: 1540.0677690505981
#
#     Average Result (M=8, K=40, N=10):
#     {'Precision': 24.54, 'Recall': 11.78,
#      'Coverage': 25.89, 'Popularity': 7.20221475}
#
#     Running time: 1643.4831750392914
#
#     Average Result (M=8, K=80, N=10):
#     {'Precision': 25.11, 'Recall': 12.06,
#      'Coverage': 20.25, 'Popularity': 7.288118125}
#
#     Running time: 1891.5019328594208
#
#     Average Result (M=8, K=160, N=10):
#     {'Precision': 24.81, 'Recall': 11.91,
#      'Coverage': 15.39, 'Popularity': 7.367559}
#
# 4. UserIIF实验
#
#     Running time: 3006.6924328804016
#
#     Average Result (M=8, K=80, N=10):
#     {'Precision': 25.22, 'Recall': 12.11,
#      'Coverage': 21.32, 'Popularity': 7.258887}
