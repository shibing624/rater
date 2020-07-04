# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: GBDT LR model

Reference:
Practical Lessons from Predicting Clicks on Ads at Facebook
http://quinonero.net/Publications/predicting-clicks-facebook.pdf

"""
import torch
import torch.nn as nn

from .lr import LR
from ..basic.gbdt import GBDT


class GBDTLR(nn.Module):
    """
    GBDT LR model
    """

    def __init__(self, num_leaves=31, max_depth=-1, n_estimators=100,
                 learning_rate=0.1, out_type='binary'):
        """
        Init model
        :param num_leaves:
        :param max_depth:
        :param n_estimators:
        :param min_data_in_leaf:
        :param learning_rate:
        :param out_type:
        """
        super(GBDTLR, self).__init__()
        self.gbdt = GBDT(num_leaves=num_leaves, max_depth=max_depth, n_estimators=n_estimators,
                         learning_rate=learning_rate, out_type=out_type)
        self.gbdt_trained = False
        self.logistic_layer = LR(num_leaves * n_estimators, out_type=out_type)

    def forward(self, feat_index,feat_value):
        """
        Forward
        :param feat_index: index input tensor
        :param feat_value: value input tensor
        :return: predict y
        """
        if not self.gbdt_trained:
            raise ValueError("need train gbdt model first.")
        pred_y, transformed_matrix = self.gbdt.pred(feat_index)
        transformed_matrix = nn.Parameter(torch.Tensor(transformed_matrix), requires_grad=True)
        y = self.logistic_layer(transformed_matrix, feat_value)
        return y

    def train_gbdt(self, data, y):
        """
        Train model with labeled data
        :param data:
        :param y:
        :return: model
        """
        if isinstance(data, torch.Tensor):
            data = data.tolist()
        if isinstance(y, torch.Tensor):
            y = y.tolist()
        self.gbdt.train(data, y)
        self.gbdt_trained = True

    def get_gbdt_trained(self):
        """
        Is model trained.
        :return: bool
        """
        return self.gbdt_trained

    def train_gbdtlr(self, dataset, model_path,epochs=10, batch_size=32):
        from sklearn.metrics import roc_auc_score
        from torch.utils.data import DataLoader, random_split
        from rater.models.model import train_val_split
        from rater.utils.logger import logger

        from rater.utils.early_stopping import EarlyStopping
        train_set, val_set = train_val_split(dataset, val_size=0.2)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size)

        # train epochs
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        loss_func = nn.BCELoss()
        train_losses = []
        val_losses = []

        # initialize the early_stopping object
        early_stopping = EarlyStopping(patience=2, verbose=True)

        for epoch in range(0, epochs):
            # train the epoch
            m = self.train()
            train_loss = 0.
            batch_nums = 0
            pred_y_lst = []
            y_lst = []
            for step, tensors in enumerate(train_loader):
                y = tensors[-1]
                X = tensors[:-1]

                pred_y = m(*X)
                loss = loss_func(pred_y, y)

                # back propagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                batch_nums += 1
                pred_y_lst.extend(pred_y.data.numpy())
                y_lst.extend(y.data.numpy())

            # calculate average loss and auc during the epoch
            avg_train_loss = train_loss / batch_nums if batch_nums > 0 else 0.
            train_auc = roc_auc_score(y_lst, pred_y_lst)
            train_losses.append(avg_train_loss)

            # run validation set
            m = self.eval()
            val_loss = 0.
            batch_nums = 0
            pred_y_lst = []
            y_lst = []
            for step, tensors in enumerate(val_loader):
                y = tensors[-1]
                X = tensors[:-1]
                pred_y = m(*X)
                loss = loss_func(pred_y, y)

                val_loss += loss.item()
                batch_nums += 1
                pred_y_lst.extend(pred_y.data.numpy())
                y_lst.extend(y.data.numpy())

            # calculate valid loss and valid auc score
            avg_val_loss = val_loss / batch_nums if batch_nums > 0 else 0.
            val_auc = roc_auc_score(y_lst, pred_y_lst)
            logger.info('epoch:%d/%d, train_loss:%.4f, train_auc:%.4f, '
                        'val_loss:%.4f, val_auc:%.4f' % (
                            epoch + 1, epochs, avg_train_loss, train_auc, avg_val_loss, val_auc))
            val_losses.append(avg_val_loss)

            # save model to file
            torch.save({
                'model_state_dict': m.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
                'loss': avg_train_loss}, model_path)
            logger.debug('model saved:{}'.format(model_path))

            # early_stopping needs the validation loss to check if it has decrease,
            # and if it not, out the loop
            early_stopping(avg_val_loss, m)
            if early_stopping.early_stop:
                logger.warning("early stopped at %s epoch" % (epoch + 1))
                break