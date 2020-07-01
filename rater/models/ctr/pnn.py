# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com), jachin, Nie
@description: A pytorch implementation of FNN


Reference:
[1] Product-based Neural Networks for User Response Prediction (SJTU 2016)
    Yanru Qu, Han Cai, Kan Ren, Weinan Zhang, Yong Yu Shanghai Jiao Tong University
    {kevinqu, hcai, kren, wnzhang, yyu}@apex.sjtu.edu.cn Ying Wen, Jun Wang University College London {ying.wen, j.wang}@cs.ucl.ac.uk

"""

from time import time

import numpy as np
import torch
import torch.backends.cudnn
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.autograd import Variable


class PNN(torch.nn.Module):
    """
    Network
    -------------
    field_size: size of the feature fields
    feature_sizes: a field_size-dim array, sizes of the feature dictionary
    embedding_size: size of the feature embedding
    h_depth: deep network's hidden layers' depth
    deep_layers: a h_depth-dim array, each element is the size of corresponding hidden layers. example:[32,32] h_depth = 2
    is_deep_dropout: bool, deep part uses dropout or not?
    dropout_deep: an array of dropout factors,example:[0.5,0.5,0.5] h_depth=2
    use_inner_product: use inner product or not?
    use_outer_product: use outter product or not?
    deep_layers_activation: relu or sigmoid etc
    learning_rate: learning_rate
    optimizer_type: optimizer_type, 'adam', 'rmsp', 'sgd', 'adag'
    is_batch_norm：bool,  use batch_norm or not ?
    weight_decay: weight decay (L2 penalty)
    random_seed: 0
    eval_metric: roc_auc_score
    use_cuda: bool use gpu or cpu?
    n_class: number of classes. is bounded to 1

    """

    def __init__(self, field_size, feature_sizes, embedding_size=4,
                 h_depth=3, deep_layers=[32, 32, 32], is_deep_dropout=True, dropout_deep=[0.5, 0.5, 0.5],
                 use_inner_product=True, use_outer_product=False,
                 deep_layers_activation='relu', learning_rate=0.003,
                 optimizer_type='adam', is_batch_norm=False, random_seed=0, weight_decay=0.0,
                 eval_metric=roc_auc_score, use_cuda=True, n_class=1
                 ):
        super(PNN, self).__init__()
        self.field_size = field_size
        self.feature_sizes = feature_sizes
        self.embedding_size = embedding_size
        self.h_depth = h_depth
        self.deep_layers = deep_layers
        self.is_deep_dropout = is_deep_dropout
        self.dropout_deep = dropout_deep
        self.use_inner_product = use_inner_product
        self.use_outer_product = use_outer_product
        self.deep_layers_activation = deep_layers_activation
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type
        self.is_batch_norm = is_batch_norm
        self.weight_decay = weight_decay
        self.random_seed = random_seed
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
            check use inner_product or outer_product
        """
        if self.use_inner_product and self.use_outer_product:
            print("The model uses both inner product and outer product")
        elif self.use_inner_product:
            print("The model uses inner product (IPNN)")
        elif self.use_ffm:
            print("The model uses outer product (OPNN)")
        else:
            print("The model is sample deep model only! Neither inner product or outer product is used")

        """
            embbedding part
        """
        print("Init embeddings")
        self.embeddings = nn.ModuleList(
            [nn.Embedding(feature_size, self.embedding_size) for feature_size in self.feature_sizes])
        print("Init embeddings finished")

        """
            first order part (linear part)
        """
        print("Init first order part")
        self.first_order_weight = nn.ModuleList([nn.ParameterList(
            [torch.nn.Parameter(torch.randn(self.embedding_size), requires_grad=True) for j in range(self.field_size)])
            for i in range(self.deep_layers[0])])
        self.bias = torch.nn.Parameter(torch.randn(self.deep_layers[0]), requires_grad=True)
        print("Init first order part finished")

        """
            second order part (quadratic part)
        """
        print("Init second order part")
        if self.use_inner_product:
            self.inner_second_weight_emb = nn.ModuleList([nn.ParameterList(
                [torch.nn.Parameter(torch.randn(self.embedding_size), requires_grad=True) for j in
                 range(self.field_size)]) for i in range(self.deep_layers[0])])

        if self.use_outer_product:
            arr = []
            for i in range(self.deep_layers[0]):
                tmp = torch.randn(self.embedding_size, self.embedding_size)
                arr.append(torch.nn.Parameter(torch.mm(tmp, tmp.t())))
            self.outer_second_weight_emb = nn.ParameterList(arr)
        print("Init second order part finished")

        print("Init nn part")
        for i, h in enumerate(self.deep_layers[1:], 1):
            setattr(self, 'linear_' + str(i), nn.Linear(self.deep_layers[i - 1], self.deep_layers[i]))
            if self.is_batch_norm:
                setattr(self, 'batch_norm_' + str(i), nn.BatchNorm1d(deep_layers[i]))
            if self.is_deep_dropout:
                setattr(self, 'linear_' + str(i) + '_dropout', nn.Dropout(self.dropout_deep[i]))
        self.deep_last_layer = nn.Linear(self.deep_layers[-1], self.n_class)
        print("Init nn part succeed")

    def forward(self, Xi, Xv):
        """
        :param Xi: index input tensor, batch_size * k * 1
        :param Xv: value input tensor, batch_size * k * 1
        :param is_pretrain: the para to decide fm pretrain or not
        :return: the last output
        """

        """
            embedding
        """
        emb_arr = [(torch.sum(emb(Xi[:, i, :]), 1).t() * Xv[:, i]).t() for i, emb in enumerate(self.embeddings)]

        """
            first order part (linear part)
        """
        first_order_arr = []
        for i, weight_arr in enumerate(self.first_order_weight):
            tmp_arr = []
            for j, weight in enumerate(weight_arr):
                tmp_arr.append(torch.sum(emb_arr[j] * weight, 1))
            first_order_arr.append(sum(tmp_arr).view([-1, 1]))
        first_order = torch.cat(first_order_arr, 1)

        """
            second order part (quadratic part)
        """
        if self.use_inner_product:
            inner_product_arr = []
            for i, weight_arr in enumerate(self.inner_second_weight_emb):
                tmp_arr = []
                for j, weight in enumerate(weight_arr):
                    tmp_arr.append(torch.sum(emb_arr[j] * weight, 1))
                sum_ = sum(tmp_arr)
                inner_product_arr.append((sum_ * sum_).view([-1, 1]))
            inner_product = torch.cat(inner_product_arr, 1)
            first_order = first_order + inner_product

        if self.use_outer_product:
            outer_product_arr = []
            emb_arr_sum = sum(emb_arr)
            emb_matrix_arr = torch.bmm(emb_arr_sum.view([-1, self.embedding_size, 1]),
                                       emb_arr_sum.view([-1, 1, self.embedding_size]))
            for i, weight in enumerate(self.outer_second_weight_emb):
                outer_product_arr.append(torch.sum(torch.sum(emb_matrix_arr * weight, 2), 1).view([-1, 1]))
            outer_product = torch.cat(outer_product_arr, 1)
            first_order = first_order + outer_product

        """
            nn part
        """
        if self.deep_layers_activation == 'sigmoid':
            activation = torch.sigmoid
        elif self.deep_layers_activation == 'tanh':
            activation = torch.tanh
        else:
            activation = torch.relu
        x_deep = first_order
        for i, h in enumerate(self.deep_layers[1:], 1):
            x_deep = getattr(self, 'linear_' + str(i))(x_deep)
            if self.is_batch_norm:
                x_deep = getattr(self, 'batch_norm_' + str(i))(x_deep)
            x_deep = activation(x_deep)
            if self.is_deep_dropout:
                x_deep = getattr(self, 'linear_' + str(i) + '_dropout')(x_deep)
        x_deep = self.deep_last_layer(x_deep)
        return torch.sum(x_deep, 1)

    def fit(self, Xi_train, Xv_train, y_train, Xi_valid=None, Xv_valid=None,
            y_valid=None, early_stopping=True, save_path=None, epochs=10, batch_size=128):
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
        batch_size = 10000
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


if __name__ == '__main__':
    from rater.utils import data_preprocess

    result_dict = data_preprocess.read_criteo_data('./tiny_train_input.csv', './category_emb.csv')
    test_dict = data_preprocess.read_criteo_data('./tiny_test_input.csv', './category_emb.csv')

    print(len(result_dict['feature_sizes']))
    print(result_dict['feature_sizes'])
    m = PNN(39, result_dict['feature_sizes'], use_cuda=False,
            weight_decay=0.00002)
    print(result_dict['index'][0:10], result_dict['value'][0:10], result_dict['label'][0:10])
    m.fit(result_dict['index'][0:10000], result_dict['value'][0:10000], result_dict['label'][0:10000],
          test_dict['index'][0:10000], test_dict['value'][0:10000], test_dict['label'][0:10000], early_stopping=True,
          save_path='pnn_model.pt')
