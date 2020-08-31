# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE

from rater.models.graph.classify import read_node_label, Classifier
from rater.models.graph.node2vec import Node2Vec

pwd_path = os.path.abspath(os.path.dirname(__file__))
label_file = os.path.join(pwd_path, './data/wiki/wiki_labels.txt')
edge_file = os.path.join(pwd_path, './data/wiki/wiki_edgelist.txt')


def evaluate_embeddings(embeddings, label_file):
    X, Y = read_node_label(label_file, skip_head=True)
    tr_frac = 0.8
    print("Training classifier using {:.2f}% nodes...".format(
        tr_frac * 100))
    clf = Classifier(embeddings=embeddings, clf=LogisticRegression(solver='lbfgs'))
    clf.split_train_evaluate(X, Y, tr_frac)


def plot_embeddings(embeddings, label_file):
    X, Y = read_node_label(label_file, skip_head=True)

    emb_list = []
    for k in X:
        emb_list.append(embeddings[k])
    emb_list = np.array(emb_list)

    model = TSNE(n_components=2)
    node_pos = model.fit_transform(emb_list)

    color_idx = {}
    for i in range(len(X)):
        color_idx.setdefault(Y[i][0], [])
        color_idx[Y[i][0]].append(i)

    for c, idx in color_idx.items():
        plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    G = nx.read_edgelist(edge_file, create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])

    model = Node2Vec(G, walk_length=10, num_walks=80,
                     p=1, q=1, workers=1, use_rejection_sampling=0)
    model.train(window_size=5, iter=3)
    embeddings = model.get_embeddings()

    evaluate_embeddings(embeddings, label_file)
    plot_embeddings(embeddings, label_file)
