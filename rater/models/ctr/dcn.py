# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com), jachin, Nie
@description: A pytorch implementation of DCN

Reference:
[1] Deep & Cross Network for Ad Click Predictions (Stanford 2017)
    Ruoxi Wang,Stanford University,Stanford, CA,ruoxi@stanford.edu
    Bin Fu,Google Inc.,New York, NY,binfu@google.com
    Gang Fu,Google Inc.,New York, NY,thomasfu@google.com
    Mingliang Wang,Google Inc.,New York, NY,mlwang@google.com
[2] Deep Crossing - Web-Scale Modeling without Manually Crafted Combinatorial Features (Microsoft 2016)

"""

from time import time

import numpy as np
import torch
import torch.backends.cudnn
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.autograd import Variable


class DCN(torch.nn.Module):
    """
    Network
    """

    def __init__(self, field_size, feature_sizes, embedding_size=4,
                 deep_layers=[32, 32], is_deep_dropout=True, dropout_deep=[0.0, 0.5, 0.5],
                 h_cross_depth=3, inner_product_layers=[32, 32],
                 is_inner_product_dropout=True, dropout_inner_product_deep=[0.0, 0.5, 0.5],
                 deep_layers_activation='relu', learning_rate=0.003,
                 optimizer_type='adam', is_batch_norm=False, random_seed=0,
                 use_cross=True, use_inner_product=False, use_deep=True, weight_decay=0.0,
                 eval_metric=roc_auc_score, use_cuda=True, n_class=1
                 ):
        super(DCN, self).__init__()
        self.field_size = field_size
        self.feature_sizes = feature_sizes
        self.embedding_size = embedding_size
        self.deep_layers = deep_layers
        self.is_deep_dropout = is_deep_dropout
        self.dropout_deep = dropout_deep
        self.h_cross_depth = h_cross_depth
        self.inner_product_layers = inner_product_layers
        self.is_inner_product_dropout = is_inner_product_dropout
        self.dropout_inner_product_deep = dropout_inner_product_deep
        self.deep_layers_activation = deep_layers_activation
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type
        self.is_batch_norm = is_batch_norm
        self.weight_decay = weight_decay
        self.random_seed = random_seed
        self.use_cross = use_cross
        self.use_inner_product = use_inner_product
        self.use_deep = use_deep
        self.eval_metric = eval_metric
        self.use_cuda = use_cuda
        self.n_class = n_class

        torch.manual_seed(self.random_seed)

        """
            check cuda
        """
        if self.use_cuda and not torch.cuda.is_available():
            self.use_cuda = False
            print("cuda is not available, automatically changed into cpu model")

        """
            check model type
        """
        if self.use_cross and self.use_deep and self.use_inner_product:
            print("The model is (cross network + deep network + inner_product network)")
        elif self.use_cross and self.use_deep:
            print("The model is (cross network + deep network)")
        elif self.use_cross and self.use_inner_product:
            print("The model is (cross network + inner product network)")
        elif self.use_inner_product and self.use_deep:
            print("The model is (inner product network + deep network)")
        elif self.use_cross:
            print("The model is a cross network only")
        elif self.use_deep:
            print("The model is a deep network only")
        elif self.use_inner_product:
            print("The model is an inner product network only")
        else:
            print(
                "You have to choose more than one of (cross network, deep network, inner product network) models to use")
            exit(1)

        """
            embeddings
        """
        self.embeddings = nn.ModuleList(
            [nn.Embedding(feature_size, self.embedding_size) for feature_size in self.feature_sizes])

        cat_size = 0
        """
            cross part
        """
        if self.use_cross:
            print("Init cross network")
            for i in range(self.h_cross_depth):
                setattr(self, 'cross_weight_' + str(i + 1),
                        torch.nn.Parameter(torch.randn(self.field_size * self.embedding_size)))
                setattr(self, 'cross_bias_' + str(i + 1),
                        torch.nn.Parameter(torch.randn(self.field_size * self.embedding_size)))
            print("Cross network finished")
            cat_size += self.field_size * self.embedding_size

        """
            inner prodcut part
        """
        if self.use_inner_product:
            print("Init inner product network")
            if self.is_inner_product_dropout:
                self.inner_product_0_dropout = nn.Dropout(self.dropout_inner_product_deep[0])
            self.inner_product_linear_1 = nn.Linear(self.field_size * (self.field_size - 1) / 2,
                                                    self.inner_product_layers[0])
            if self.is_inner_product_dropout:
                self.inner_product_1_dropout = nn.Dropout(self.dropout_inner_product_deep[1])
            if self.is_batch_norm:
                self.inner_product_batch_norm_1 = nn.BatchNorm1d(self.inner_product_layers[0])

            for i, h in enumerate(self.inner_product_layers[1:], 1):
                setattr(self, 'inner_product_linear_' + str(i + 1),
                        nn.Linear(self.inner_product_layers[i - 1], self.inner_product_layers[i]))
                if self.is_batch_norm:
                    setattr(self, 'inner_product_batch_norm_' + str(i + 1),
                            nn.BatchNorm1d(self.inner_product_layers[i]))
                if self.is_deep_dropout:
                    setattr(self, 'inner_product_' + str(i + 1) + '_dropout',
                            nn.Dropout(self.dropout_inner_product_deep[i + 1]))
            cat_size += inner_product_layers[-1]
            print("Inner product network finished")

        """
            deep part
        """
        if self.use_deep:
            print("Init deep part")

            if self.is_deep_dropout:
                self.linear_0_dropout = nn.Dropout(self.dropout_deep[0])
            self.linear_1 = nn.Linear(self.embedding_size * self.field_size, deep_layers[0])
            if self.is_batch_norm:
                self.batch_norm_1 = nn.BatchNorm1d(deep_layers[0])
            if self.is_deep_dropout:
                self.linear_1_dropout = nn.Dropout(self.dropout_deep[1])
            for i, h in enumerate(self.deep_layers[1:], 1):
                setattr(self, 'linear_' + str(i + 1), nn.Linear(self.deep_layers[i - 1], self.deep_layers[i]))
                if self.is_batch_norm:
                    setattr(self, 'batch_norm_' + str(i + 1), nn.BatchNorm1d(deep_layers[i]))
                if self.is_deep_dropout:
                    setattr(self, 'linear_' + str(i + 1) + '_dropout', nn.Dropout(self.dropout_deep[i + 1]))
            cat_size += deep_layers[-1]
            print("Init deep part succeed")

        self.last_layer = nn.Linear(cat_size, 1)

    def forward(self, Xi, Xv):
        """
        :param Xi_train: index input tensor, batch_size * k * 1
        :param Xv_train: value input tensor, batch_size * k * 1
        :return: the last output
        """

        if self.deep_layers_activation == 'sigmoid':
            activation = torch.sigmoid
        elif self.deep_layers_activation == 'tanh':
            activation = torch.tanh
        else:
            activation = torch.relu

        """
            embeddings
        """
        emb_arr = [(torch.sum(emb(Xi[:, i, :]), 1).t() * Xv[:, i]).t() for i, emb in enumerate(self.embeddings)]
        outputs = []
        """
            cross part
        """
        if self.use_cross:
            x_0 = torch.cat(emb_arr, 1)
            x_l = x_0
            for i in range(self.h_cross_depth):
                x_l = torch.sum(x_0 * x_l, 1).view([-1, 1]) * getattr(self, 'cross_weight_' + str(i + 1)).view(
                    [1, -1]) + getattr(self, 'cross_bias_' + str(i + 1)) + x_l
            outputs.append(x_l)

        """
            inner product part
        """
        if self.use_inner_product:
            fm_arr = []
            for i in range(self.field_size):
                for j in range(i + 1, self.field_size):
                    fm_arr.append(torch.sum(emb_arr[i] * emb_arr[j], 1).view([-1, 1]))
            inner_output = torch.cat(fm_arr, 1)

            deep_emb = self.inner_product_0_dropout(inner_output)
            x_deep = self.inner_product_linear_1(deep_emb)
            if self.is_batch_norm:
                x_deep = self.inner_product_batch_norm_1(x_deep)
            x_deep = activation(x_deep)
            if self.is_inner_product_dropout:
                x_deep = self.inner_product_1_dropout(x_deep)
            for i in range(1, len(self.deep_layers)):
                x_deep = getattr(self, 'inner_product_linear_' + str(i + 1))(x_deep)
                if self.is_batch_norm:
                    x_deep = getattr(self, 'inner_product_batch_norm_' + str(i + 1))(x_deep)
                x_deep = activation(x_deep)
                if self.is_deep_dropout:
                    x_deep = getattr(self, 'inner_product_' + str(i + 1) + '_dropout')(x_deep)
            outputs.append(x_deep)

        """
            deep part
        """
        if self.use_deep:
            deep_emb = torch.cat(emb_arr, 1)

            if self.is_deep_dropout:
                deep_emb = self.linear_0_dropout(deep_emb)
            x_deep = self.linear_1(deep_emb)
            if self.is_batch_norm:
                x_deep = self.batch_norm_1(x_deep)
            x_deep = activation(x_deep)
            if self.is_deep_dropout:
                x_deep = self.linear_1_dropout(x_deep)
            for i in range(1, len(self.deep_layers)):
                x_deep = getattr(self, 'linear_' + str(i + 1))(x_deep)
                if self.is_batch_norm:
                    x_deep = getattr(self, 'batch_norm_' + str(i + 1))(x_deep)
                x_deep = activation(x_deep)
                if self.is_deep_dropout:
                    x_deep = getattr(self, 'linear_' + str(i + 1) + '_dropout')(x_deep)
            outputs.append(x_deep)

        """
            total
        """
        output = self.last_layer(torch.cat(outputs, 1))
        return torch.sum(output, 1)

    def fit(self, Xi_train, Xv_train, y_train, Xi_valid=None, Xv_valid=None,
            y_valid=None, early_stopping=False, save_path=None, epochs=10, batch_size=128):
        """
        :param Xi_train: [[ind1_1, ind1_2, ...], ...
                        indi_j is the feature index of feature field j of sample i in the training set
        :param Xv_train: [[val1_1, val1_2, ...], ...
                        vali_j is the feature value of feature field j of sample i in the training set
                        vali_j can be either binary (1/0, for binary/categorical features) or float
        :param y_train: label of each sample in the training set
        :param Xi_valid: list of list of feature indices of each sample in the validation set
        :param Xv_valid: list of list of feature values of each sample in the validation set
        :param y_valid: label of each sample in the validation set
        :param early_stopping: perform early stopping or not
        :param save_path: the path to save the model
        :param epochs:
        :param batch_size:
        :return:
        """
        is_valid = False
        Xi_train = np.array(Xi_train).reshape((-1, self.field_size, 1))
        Xv_train = np.array(Xv_train)
        y_train = np.array(y_train)
        x_size = Xi_train.shape[0]
        if Xi_valid:
            Xi_valid = np.array(Xi_valid).reshape((-1, self.field_size, 1))
            Xv_valid = np.array(Xv_valid)
            y_valid = np.array(y_valid)
            x_valid_size = Xi_valid.shape[0]
            is_valid = True

        """
            train model
        """
        model = self.train()

        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        if self.optimizer_type == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer_type == 'rmsp':
            optimizer = torch.optim.RMSprop(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer_type == 'adag':
            optimizer = torch.optim.Adagrad(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        criterion = F.binary_cross_entropy_with_logits

        train_result = []
        valid_result = []
        for epoch in range(epochs):
            total_loss = 0.0
            batch_iter = x_size // batch_size
            epoch_begin_time = time()
            batch_begin_time = time()
            for i in range(batch_iter + 1):
                offset = i * batch_size
                end = min(x_size, offset + batch_size)
                if offset == end:
                    break
                batch_xi = Variable(torch.LongTensor(Xi_train[offset:end]))
                batch_xv = Variable(torch.FloatTensor(Xv_train[offset:end]))
                batch_y = Variable(torch.FloatTensor(y_train[offset:end]))
                if self.use_cuda:
                    batch_xi, batch_xv, batch_y = batch_xi.cuda(), batch_xv.cuda(), batch_y.cuda()
                optimizer.zero_grad()
                outputs = model(batch_xi, batch_xv)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                if i % 100 == 99:  # print every 100 mini-batches
                    eval = self.evaluate(batch_xi, batch_xv, batch_y)
                    print('[%d, %5d] loss: %.6f metric: %.6f time: %.1f s' %
                          (epoch + 1, i + 1, total_loss / 100.0, eval, time() - batch_begin_time))
                    total_loss = 0.0
                    batch_begin_time = time()

            train_loss, train_metric = self.eval_by_batch(Xi_train, Xv_train, y_train, x_size)
            train_result.append(train_loss)
            print('epoch: %d/%d train_loss: %.6f train_metric: %.6f time: %.1f s' %
                  (epoch + 1, epochs, train_loss, train_metric, time() - epoch_begin_time))

            if is_valid:
                valid_loss, valid_metric = self.eval_by_batch(Xi_valid, Xv_valid, y_valid, x_valid_size)
                valid_result.append(valid_loss)
                print('            val_loss: %.6f val_metric: %.6f time: %.1f s' %
                      (valid_loss, valid_metric, time() - epoch_begin_time))
            if save_path:
                torch.save(self.state_dict(), save_path)
            if is_valid and early_stopping and self.is_training_end(valid_result):
                print("early stop at [%d] epoch!" % (epoch + 1))
                break

    def eval_by_batch(self, Xi, Xv, y, x_size):
        total_loss = 0.0
        y_pred = []
        batch_size = 16384
        batch_iter = x_size // batch_size
        criterion = F.binary_cross_entropy_with_logits
        model = self.eval()
        for i in range(batch_iter + 1):
            offset = i * batch_size
            end = min(x_size, offset + batch_size)
            if offset == end:
                break
            batch_xi = Variable(torch.LongTensor(Xi[offset:end]))
            batch_xv = Variable(torch.FloatTensor(Xv[offset:end]))
            batch_y = Variable(torch.FloatTensor(y[offset:end]))
            if self.use_cuda:
                batch_xi, batch_xv, batch_y = batch_xi.cuda(), batch_xv.cuda(), batch_y.cuda()
            outputs = model(batch_xi, batch_xv)
            pred = torch.sigmoid(outputs).cpu()
            y_pred.extend(pred.data.numpy())
            loss = criterion(outputs, batch_y)
            total_loss += loss.item() * (end - offset)
        total_metric = self.eval_metric(y, y_pred)
        return total_loss / x_size, total_metric

    def is_training_end(self, valid_result):
        if len(valid_result) > 4:
            if valid_result[-1] > valid_result[-2] > valid_result[-3] > valid_result[-4]:
                return True
        return False

    def predict(self, Xi, Xv):
        """
        :param Xi: the same as fit function
        :param Xv: the same as fit function
        :return: output, ont-dim array
        """
        Xi = np.array(Xi).reshape((-1, self.field_size, 1))
        Xi = Variable(torch.LongTensor(Xi))
        Xv = Variable(torch.FloatTensor(Xv))
        if self.use_cuda and torch.cuda.is_available():
            Xi, Xv = Xi.cuda(), Xv.cuda()

        model = self.eval()
        pred = torch.sigmoid(model(Xi, Xv)).cpu()
        return (pred.data.numpy() > 0.5)

    def predict_proba(self, Xi, Xv):
        Xi = np.array(Xi).reshape((-1, self.field_size, 1))
        Xi = Variable(torch.LongTensor(Xi))
        Xv = Variable(torch.FloatTensor(Xv))
        if self.use_cuda and torch.cuda.is_available():
            Xi, Xv = Xi.cuda(), Xv.cuda()

        model = self.eval()
        pred = torch.sigmoid(model(Xi, Xv)).cpu()
        return pred.data.numpy()

    def inner_predict(self, Xi, Xv):
        """
        :param Xi: tensor of feature index
        :param Xv: tensor of feature value
        :return: output, numpy
        """
        model = self.eval()
        pred = torch.sigmoid(model(Xi, Xv)).cpu()
        return (pred.data.numpy() > 0.5)

    def inner_predict_proba(self, Xi, Xv):
        """
        :param Xi: tensor of feature index
        :param Xv: tensor of feature value
        :return: output, numpy
        """
        model = self.eval()
        pred = torch.sigmoid(model(Xi, Xv)).cpu()
        return pred.data.numpy()

    def evaluate(self, Xi, Xv, y):
        """
        :param Xi: tensor of feature index
        :param Xv: tensor of feature value
        :param y: tensor of labels
        :return: metric of the evaluation
        """
        y_pred = self.inner_predict_proba(Xi, Xv)
        return self.eval_metric(y.cpu().data.numpy(), y_pred)
