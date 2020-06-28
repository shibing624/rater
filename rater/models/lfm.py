# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 基于隐语义模型
"""
import os

import numpy as np

import recommender
from recommender.models.usercf import Dataset, Metric
from recommender.utils.logger import timer


def lfm_alg(train, ratio, K, lr, step, lmbda, N):
    """
    :params: train, 训练数据
    :params: ratio, 负采样的正负比例
    :params: K, 隐语义个数
    :params: lr, 初始学习率
    :params: step, 迭代次数
    :params: lmbda, 正则化系数
    :params: N, 推荐TopN物品的个数
    :return: recommendation, 获取推荐结果的接口
    """
    all_items = {}  # 统计物品的流行度
    for user in train:
        for item in train[user]:
            if item not in all_items:
                all_items[item] = 0
            all_items[item] += 1
    all_items = list(all_items.items())
    items = [x[0] for x in all_items]
    pops = [x[1] for x in all_items]  # 流行度

    # 负采样函数(注意：要按照流行度进行采样)
    # 两个原则：(1) 保证正负样本的均衡 (2)负样本要选取那些很热门而用户却没有行为的物品
    def negative_sample(data, ratio, pops):
        new_data = {}
        # 正样本
        for user in data:
            if user not in new_data:
                new_data[user] = {}
            for item in data[user]:
                new_data[user][item] = 1  # 用户对正样本的兴趣为1
        # 负样本
        for user in new_data:
            seen = set(new_data[user])
            pos_num = len(seen)
            # 从items中按流行度pops采样int(pos_num * ratio * 3)个负样本
            item = np.random.choice(items, int(pos_num * ratio * 3), pops)  # 按流行度采样
            item = [x for x in item if x not in seen][:int(pos_num * ratio)]
            new_data[user].update({x: 0 for x in item})  # 用户对负样本的兴趣为0

        return new_data

    # 训练
    P, Q = {}, {}
    # 随机初始化
    for user in train:
        P[user] = np.random.random(K)  # p(u,k)度量了用户u的兴趣和第k个隐类的关系
    for item in items:
        Q[item] = np.random.random(K)  # q(u,k)度量了第k个隐类和物品i之间的关系

    for s in range(step):
        data = negative_sample(train, ratio, pops)
        for user in data:
            for item in data[user]:
                eui = data[user][item] - (P[user] * Q[item]).sum()
                # loss=1/2 (eui^2 + lmbda*P[user]^2 + lmbda*Q[item]^2 )
                # 按SGD更新参数
                P[user] += lr * (Q[item] * eui - lmbda * P[user])
                Q[item] += lr * (P[user] * eui - lmbda * Q[item])
        lr *= 0.9  # 调整学习率

    # 获取接口函数
    def recommendation(user):
        seen_items = set(train[user])
        recs = {}
        for item in items:
            if item not in seen_items:
                recs[item] = (P[user] * Q[item]).sum()  # user对item的兴趣
        recs = list(sorted(recs.items(), key=lambda x: x[1], reverse=True))[:N]
        return recs

    return recommendation


# 三. LFM实验
# M=8, N=10, ratio=[1, 2, 3, 5, 10, 20] ratio参数对LFM的性能影响最大，固定K=100, lr=0.02, step=100, lmbda=0.01，
# 只研究正负样本比例的影响
class Experiment:
    def __init__(self, M, N, ratio=1,
                 K=100, lr=0.02, step=100, lmbda=0.01,
                 fp=os.path.join(recommender.movielens_1m_dir, 'ratings.dat'), name='LFM'):
        """
        :params: M, 进行多少次实验
        :params: N, TopN推荐物品的个数
        :params: ratio, 正负样本比例
        :params: K, 隐语义个数
        :params: lr, 学习率
        :params: step, 训练步数
        :params: lmbda, 正则化系数
        :params: fp, 数据文件路径
        """
        self.M = M
        self.K = K
        self.N = N
        self.ratio = ratio
        self.lr = lr
        self.step = step
        self.lmbda = lmbda
        self.fp = fp
        self.alg = lfm_alg
        self.dataset = Dataset(self.fp)

    # 定义单次实验
    @timer
    def worker(self, train, test):
        """
        :params: train, 训练数据集
        :params: test, 测试数据集
        :return: 各指标的值
        """
        recommendation = self.alg(train, self.ratio, self.K,
                                  self.lr, self.step, self.lmbda, self.N)
        metric = Metric(train, test, recommendation)
        return metric.eval()

    # 多次实验取平均
    @timer
    def run(self):
        metrics = {'Precision': 0, 'Recall': 0,
                   'Coverage': 0, 'Popularity': 0}
        for ii in range(self.M):
            train, test = self.dataset.split_data(ii)
            print('Experiment {}:'.format(ii))
            metric = self.worker(train, test)
            metrics = {k: metrics[k] + metric[k] for k in metrics}
        metrics = {k: metrics[k] / self.M for k in metrics}
        print('Average Result (M={}, N={}, ratio={}): {}'.format(
            self.M, self.N, self.ratio, metrics))


if __name__ == '__main__':
    # LFM实验：随着负样本数的增加，precision，recall明显提高，coverage不断降低，流行度不断增加，
    # 不过当ratio达到某个值之后，precision，recall就比较稳定了
    M, N = 1, 10  # 为节省时间，取1折
    for r in [1, 2, 3, 5, 10, 20]:
        exp = Experiment(M, N, ratio=r)
        exp.run()
