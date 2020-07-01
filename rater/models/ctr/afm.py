# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com), jachin,Nie
@description: A pytorch implementation of AFM

Reference:
[1] Attentional Factorization Machines - Learning the Weight of Feature Interactions via Attention Networks (ZJU 2017)
"""

from time import time

import numpy as np
import torch
import torch.backends.cudnn
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.autograd import Variable


class AFM(torch.nn.Module):
    """
    Network
    -------------
    field_size: size of the feature fields
    feature_sizes: a field_size-dim array, sizes of the feature dictionary
    embedding_size: size of the feature embedding
    attention_size: The attention netwotk's parameter
    is_shallow_dropout: bool, shallow part(fm or ffm part) uses dropout or not?
    dropout_shallow: an array of the size of 1, example:[0.5], the element is for the-first order part
    is_deep_dropout: bool, deep part uses dropout or not?
    dropout_deep: an array of dropout factors,example:[0.5,0.5,0.5] h_depth=2
    learning_rate: learning_rate
    optimizer_type: optimizer_type, 'adam', 'rmsp', 'sgd', 'adag'
    is_batch_norm：bool, use batch_norm or not
    weight_decay: weight decay (L2 penalty)
    random_seed: random_seed=0
    use_fm: bool
    use_ffm: bool
    eval_metric: roc_auc_score
    use_cuda: bool use gpu or cpu?
    n_class: number of classes. is bounded to 1


    Attention: only support logsitcs regression
    """

    def __init__(self, field_size, feature_sizes, embedding_size=4, attention_size=4, is_shallow_dropout=True,
                 dropout_shallow=[0.5], is_attention_dropout=True, dropout_attention=[0.5],
                 learning_rate=0.003, optimizer_type='adam', is_batch_norm=False, random_seed=0, weight_decay=0.0,
                 use_fm=True, use_ffm=False, eval_metric=roc_auc_score,
                 use_cuda=True, n_class=1
                 ):
        super(AFM, self).__init__()
        self.field_size = field_size
        self.feature_sizes = feature_sizes
        self.embedding_size = embedding_size
        self.attention_size = attention_size
        self.is_shallow_dropout = is_shallow_dropout
        self.dropout_shallow = dropout_shallow
        self.is_attention_dropout = is_attention_dropout
        self.dropout_attention = dropout_attention
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type
        self.is_batch_norm = is_batch_norm
        self.weight_decay = weight_decay
        self.random_seed = random_seed
        self.use_fm = use_fm
        self.use_ffm = use_ffm
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
            check use fm or ffm
        """
        if self.use_fm and self.use_ffm:
            print("only support one type only, please make sure to choose only fm or ffm part")
            exit(1)
        elif self.use_fm:
            print("The model is afm(fm+attention layers)")
        elif self.use_ffm:
            print("The model is affm(ffm+attention layers)")
        else:
            print("You have to choose more than one of (fm, ffm) models to use")
            exit(1)
        """
            bias
        """
        self.bias = torch.nn.Parameter(torch.randn(1))

        """
            fm part
        """
        if self.use_fm:
            print("Init fm part")
            self.fm_first_order_embeddings = nn.ModuleList(
                [nn.Embedding(feature_size, 1) for feature_size in self.feature_sizes])
            if self.dropout_shallow:
                self.fm_first_order_dropout = nn.Dropout(self.dropout_shallow[0])
            self.fm_second_order_embeddings = nn.ModuleList(
                [nn.Embedding(feature_size, self.embedding_size) for feature_size in self.feature_sizes])
            print("Init fm part succeed")

        """
            ffm part
        """
        if self.use_ffm:
            print("Init ffm part")
            self.ffm_first_order_embeddings = nn.ModuleList(
                [nn.Embedding(feature_size, 1) for feature_size in self.feature_sizes])
            if self.dropout_shallow:
                self.ffm_first_order_dropout = nn.Dropout(self.dropout_shallow[0])
            self.ffm_second_order_embeddings = nn.ModuleList(
                [nn.ModuleList([nn.Embedding(feature_size, self.embedding_size) for i in range(self.field_size)]) for
                 feature_size in self.feature_sizes])
            print("Init ffm part succeed")

        """
            attention part
        """
        print("Init attention part")

        if self.is_attention_dropout:
            self.attention_linear_0_dropout = nn.Dropout(self.dropout_attention[0])
        self.attention_linear_1 = nn.Linear(self.embedding_size, self.attention_size)
        self.H = torch.nn.Parameter(torch.randn(self.attention_size))
        self.P = torch.nn.Parameter(torch.randn(self.embedding_size))
        print("Init attention part succeed")

    def forward(self, Xi, Xv):
        """
        :param Xi_train: index input tensor, batch_size * k * 1
        :param Xv_train: value input tensor, batch_size * k * 1
        :return: the last output
        """
        """
            fm part
        """
        fm_arr = []
        fm_first_order = None
        if self.use_fm:
            fm_first_order_emb_arr = [(torch.sum(emb(Xi[:, i, :]), 1).t() * Xv[:, i]).t() for i, emb in
                                      enumerate(self.fm_first_order_embeddings)]
            fm_first_order = torch.cat(fm_first_order_emb_arr, 1)
            if self.is_shallow_dropout:
                fm_first_order = self.fm_first_order_dropout(fm_first_order)

            fm_second_order_emb_arr = [(torch.sum(emb(Xi[:, i, :]), 1).t() * Xv[:, i]).t() for i, emb in
                                       enumerate(self.fm_second_order_embeddings)]
            for i in range(self.field_size):
                for j in range(i + 1, self.field_size):
                    fm_arr.append(fm_second_order_emb_arr[i] * fm_second_order_emb_arr[j])

        """
            ffm part
        """
        ffm_arr = []
        ffm_first_order = None
        if self.use_ffm:
            ffm_first_order_emb_arr = [(torch.sum(emb(Xi[:, i, :]), 1).t() * Xv[:, i]).t() for i, emb in
                                       enumerate(self.ffm_first_order_embeddings)]
            ffm_first_order = torch.cat(ffm_first_order_emb_arr, 1)
            if self.is_shallow_dropout:
                ffm_first_order = self.ffm_first_order_dropout(ffm_first_order)
            ffm_second_order_emb_arr = [[(torch.sum(emb(Xi[:, i, :]), 1).t() * Xv[:, i]).t() for emb in f_embs] for
                                        i, f_embs in enumerate(self.ffm_second_order_embeddings)]
            for i in range(self.field_size):
                for j in range(i + 1, self.field_size):
                    ffm_arr.append(ffm_second_order_emb_arr[i][j] * ffm_second_order_emb_arr[j][i])

        """
            attention part
        """
        if self.use_fm:
            interaction_layer = torch.cat(fm_arr, 1)
        else:
            interaction_layer = torch.cat(ffm_arr, 1)

        if self.is_attention_dropout:
            interaction_layer = self.attention_linear_0_dropout(interaction_layer)
        attn_val = self.attention_linear_1(interaction_layer.view([-1, self.embedding_size]))
        attn_val = attn_val * self.H

        num_pairs = int(self.field_size * (self.field_size - 1) / 2)
        attn_val = torch.sum(attn_val, 1).view([-1, num_pairs])
        attention_weight = torch.nn.Softmax(dim=1)(attn_val)
        attention_output = torch.sum(interaction_layer.view([-1, self.embedding_size]) * self.P, 1).view(
            [-1, num_pairs])
        attention_output = attention_output * attention_weight

        """
            sum
        """
        if self.use_fm:
            total_sum = self.bias + torch.sum(fm_first_order, 1) + torch.sum(attention_output, 1)
        else:
            total_sum = self.bias + torch.sum(ffm_first_order, 1) + torch.sum(attention_output, 1)
        return total_sum

    def fit(self, Xi_train, Xv_train, y_train, Xi_valid=None, Xv_valid=None,y_valid=None,
            early_stopping=False, save_path=None, epochs=10, batch_size=128):
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
        :param epochs: int, num of epochs
        :param batch_size: int, batch size
        :return: None
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
        if self.use_ffm:
            batch_size = 10000 * 2
        else:
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

    @staticmethod
    def is_training_end(valid_result):
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
        return pred.data.numpy() > 0.5

    def predict_proba(self, Xi, Xv):
        Xi = np.array(Xi).reshape((-1, self.field_size, 1))
        Xi = Variable(torch.LongTensor(Xi))
        Xv = Variable(torch.FloatTensor(Xv))
        if self.use_cuda and torch.cuda.is_available():
            Xi, Xv = Xi.cuda(), Xv.cuda()

        model = self.eval()
        pred = torch.sigmoid(model(Xi, Xv)).cpu()
        return pred.data.numpy()

    def inner_predict(self, Xi, Xv, threshold=0.5):
        """
        :param Xi: tensor of feature index
        :param Xv: tensor of feature value
        :return: output, numpy
        """
        model = self.eval()
        pred = torch.sigmoid(model(Xi, Xv)).cpu()
        return pred.data.numpy() > threshold

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
    m = AFM(39, result_dict['feature_sizes'], is_shallow_dropout=False, use_cuda=False,
            weight_decay=0.00002, use_fm=True, use_ffm=False)

    print(result_dict['index'][0:10], result_dict['value'][0:10], result_dict['label'][0:10])
    m.fit(result_dict['index'][0:10], result_dict['value'][0:10], result_dict['label'][0:10], early_stopping=True,
          save_path='afm_model.pt')
