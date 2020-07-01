# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com), yuwanlong, jachin,Nie
@description: A pytorch implementation of deepfm

Reference:
[1] DeepFM: A Factorization-Machine based Neural Network for CTR Prediction (HIT-Huawei 2017)
    Huifeng Guo, Ruiming Tang, Yunming Yey, Zhenguo Li, Xiuqiang He.
"""
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.autograd import Variable


class DeepFM(nn.Module):
    def __init__(self, field_size, feature_sizes,
                 embedding_size=4, deep_layers=[32, 32], optimizer_type='adam', random_seed=0,
                 use_fm=True, use_deep=True, use_cuda=True, deep_layers_activation='relu',
                 learning_rate=0.003, weight_decay=0.0):
        super(DeepFM, self).__init__()
        self.field_size = field_size
        self.feature_sizes = feature_sizes
        self.embedding_size = embedding_size
        self.deep_layers = deep_layers
        self.optimizer_type = optimizer_type
        self.deep_layers_activation = deep_layers_activation
        self.use_fm = use_fm
        self.use_deep = use_deep
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.use_cuda = use_cuda

        torch.manual_seed(random_seed)

        if self.use_cuda and not torch.cuda.is_available():
            self.use_cuda = False
            print("cuda is not available")

        if self.use_fm and self.use_deep:
            print("use deepFM")
        elif self.use_fm:
            print("use FM")
        elif self.use_deep:
            print("use dnn")

        self.eval_metric = roc_auc_score

        if use_fm:
            self.bias = nn.Parameter(torch.randn(1))
            self.linear_part_weight = \
                nn.ModuleList([nn.Embedding(feature_size, 1) for feature_size in self.feature_sizes])
            self.fm_cross_term_embedding = \
                nn.ModuleList([nn.Embedding(feature_size, self.embedding_size) for feature_size in self.feature_sizes])

        if use_deep:
            if not self.use_fm:
                self.fm_cross_term_embedding = nn.ModuleList([nn.Embedding(feature_size, self.embedding_size)
                                                              for feature_size in self.feature_sizes])

            self.deep_linear_1 = nn.Linear(self.field_size * self.embedding_size, deep_layers[0])
            for i, h in enumerate(self.deep_layers[1:], 1):
                setattr(self, 'linear_' + str(i + 1), nn.Linear(self.deep_layers[i - 1], self.deep_layers[i]))

    def forward(self, Xi, Xv):
        fm_cross_term_array = []
        linear_part_result = []
        fm_cross_term_result = []
        x_deep_part = []
        if self.use_fm:
            linear_part_array = [(torch.sum(weight(Xi[:, i, :]), 1).t() * Xv[:, 1]).t()
                                 for i, weight in enumerate(self.linear_part_weight)]  # filed_size * [batch_size * 1]
            linear_part_result = torch.cat(linear_part_array, 1)
            # filed_size * [batch_size * embedding_size]
            fm_cross_term_array = [(torch.sum(emb(Xi[:, i, :]), 1).t() * Xv[:, 1]).t()
                                   for i, emb in enumerate(self.fm_cross_term_embedding)]
            fm_sum_cross_term_array = sum(fm_cross_term_array)  # batch_size * embedding_size
            sum_of_square = fm_sum_cross_term_array * fm_sum_cross_term_array  # batch_size * embedding_size
            fm_square_cross_term_array = \
                [item * item for item in fm_cross_term_array]  # filed_size * [batch_size * embedding_size]
            square_of_sum = sum(fm_square_cross_term_array)
            fm_cross_term_result = 0.5 * (sum_of_square - square_of_sum)

        if self.use_deep:
            if self.use_fm:
                # field_size * [batch_size * embedding_sze] -> batch_size * (filed_size * embedding_size)
                deep_part_input = torch.cat(fm_cross_term_array, 1)
            else:
                # field_size * [batch_size * embedding_sze] -> batch_size * (filed_size * embedding_size)
                deep_part_input = torch.cat([(torch.sum(emb(Xi[:, i, :]), 1).t() * Xv[:, 1]).t()
                                             for i, emb in enumerate(self.fm_cross_term_embedding)])

            if self.deep_layers_activation == 'sigmoid':
                activation = torch.sigmoid
            elif self.deep_layers_activation == 'tanh':
                activation = torch.tanh
            else:
                activation = torch.relu

            x_deep_part = self.deep_linear_1(deep_part_input)
            x_deep_part = activation(x_deep_part)
            for i in range(1, len(self.deep_layers)):
                x_deep_part = getattr(self, 'linear_' + str(i + 1))(x_deep_part)
                x_deep_part = activation(x_deep_part)

        if self.use_fm and self.use_deep:
            total_resul = torch.sum(linear_part_result, 1) + torch.sum(fm_cross_term_result, 1) + \
                          torch.sum(x_deep_part, 1) + self.bias
        elif self.use_fm:
            total_resul = torch.sum(linear_part_result, 1) + torch.sum(fm_cross_term_result, 1) + self.bias
        else:
            total_resul = torch.sum(x_deep_part, 1)

        return total_resul

    def fit(self, Xi_train, Xv_train, y_train, Xi_valid=None, Xv_valid=None, y_valid=None, save_path='',
            early_stopping=True, epochs=2, batch_size=128):
        is_valid = False
        Xi_train = np.array(Xi_train).reshape((-1, self.field_size, 1))
        Xv_train = np.array(Xv_train)
        y_train = np.array(y_train)
        x_train_size = Xi_train.shape[0]
        if Xi_valid:
            Xi_valid = np.array(Xi_valid).reshape(-1, self.field_size, 1)
            Xv_valid = np.array(Xv_valid)
            y_valid = np.array(y_valid)
            x_valid_size = Xi_valid.shape[0]
            is_valid = True

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
            batch_iter = x_train_size // batch_size
            epoch_begin_time = time()
            for i in range(batch_iter + 1):
                offset = i * batch_size
                end = min(x_train_size, offset + batch_size)
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

            train_loss, train_metric = self.eval_by_batch(Xi_train, Xv_train, y_train, x_train_size)
            train_result.append(train_metric)
            print('epoch: %d/%d train_loss: %.6f train_metric: %.6f time: %.1f s' %
                  (epoch + 1, epochs, train_loss, train_metric, time() - epoch_begin_time))

            if is_valid:
                valid_loss, valid_metric = self.eval_by_batch(Xi_valid, Xv_valid, y_valid, x_valid_size)
                valid_result.append(valid_loss)
                print('         val_loss: %.6f val_metric: %.6f time: %.1f s' %
                      (valid_loss, valid_metric, time() - epoch_begin_time))
            if save_path:
                torch.save(self.state_dict(), save_path)
            if is_valid and early_stopping and self.is_training_end(valid_result):
                print("early stop at [%d] epoch!" % (epoch + 1))
                break

    def eval_by_batch(self, Xi, Xv, y, x_size):
        total_loss = 0.0
        y_pred = []
        batch_size = 128
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
            total_loss += loss.data * (end - offset)
        total_metric = self.eval_metric(y, y_pred)
        return total_loss / x_size, total_metric

    def inner_predict_proba(self, Xi, Xv):
        model = self.eval()
        pred = torch.sigmoid(model(Xi, Xv)).cpu()
        return pred.data.numpy()

    def evaluate(self, Xi, Xv, y):
        y_pred = self.inner_predict_proba(Xi, Xv)
        return self.eval_metric(y.cpu().data.numpy, y_pred)

    def is_training_end(self, valid_result):
        if len(valid_result) > 4:
            if valid_result[-1] > valid_result[-2] > valid_result[-3] > valid_result[-4]:
                return True
        return False

    def predict(self, Xi, Xv):
        Xi = np.array(Xi).reshape((-1, self.field_size, 1))
        Xi = Variable(torch.LongTensor(Xi))
        Xv = Variable(torch.FloatTensor(Xv))
        if self.use_cuda and torch.cuda.is_available():
            Xi, Xv = Xi.cuda(), Xv.cuda()

        model = self.eval()
        pred = torch.sigmoid(model(Xi, Xv)).cpu()
        return pred.data.numpy()


if __name__ == '__main__':
    from rater.utils import data_preprocess

    result_dict = data_preprocess.read_criteo_data('./tiny_train_input.csv', 'category_emb.csv')
    test_dict = data_preprocess.read_criteo_data('tiny_test_input.csv', 'category_emb.csv')

    print(len(result_dict['feature_sizes']))
    print(result_dict['feature_sizes'])
    deepfm = DeepFM(39, result_dict['feature_sizes'], use_cuda=False, weight_decay=0.0001,
                    use_fm=True, use_deep=True)
    deepfm.fit(result_dict['index'][0:10000], result_dict['value'][0:10000], result_dict['label'][0:10000],
               test_dict['index'][0:10000], test_dict['value'][0:10000], test_dict['label'][0:10000],
               save_path='deepfm_model.pt')
