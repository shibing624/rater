# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import os

import torch
from torch.utils.data import DataLoader, random_split

from ..utils.earlystopping import EarlyStopping
from ..utils.logger import logger


def train_val_split(dataset, val_size=0.2):
    data_num = len(dataset)
    val_size = val_size if 0. < val_size < 1. else 0.2
    val_num = int(data_num * val_size)
    train_num = data_num - val_num
    train_set, val_set = random_split(dataset, [train_num, val_num])
    return train_set, val_set


def train_model(model, model_path, dataset, loss_func, optimizer, device,
                val_size=0.2, batch_size=32, epochs=10, shuffle=True, patience=2):
    logger.info('train start')
    train_set, val_set = train_val_split(dataset, val_size)
    # write training log
    logger.info('model:{}, loss_func:{}, optimizer:{}, epochs:{}, batch_size:{}, '
                'shuffle:{}, device:{}, patience:{}'.format(model, loss_func, optimizer, epochs,
                                                            batch_size, shuffle, device, patience))

    # if model_path exists, load the checkpoint and continue training
    curr_epoch = 0
    if model_path and os.path.exists(model_path):
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        curr_epoch = checkpoint['epoch']
        curr_loss = checkpoint['loss']

        logger.info('model loaded from {}'.format(model_path))
        logger.info('epochs trained:{}, current loss:{:.4f}'.format(curr_epoch, curr_loss))
        model.to(device)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    # train epochs
    history = fit(model, epochs, train_loader, val_loader, loss_func, optimizer, model_path, curr_epoch, patience)
    logger.info('train end')
    return model, history


def fit(model, epochs, train_loader, val_loader, loss_func, optimizer, model_path, curr_epoch=0, patience=2):
    train_losses = []
    val_losses = []

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    for epoch in range(curr_epoch, epochs):
        # train the epoch
        model.train()
        train_loss = 0.
        batch_nums = 0
        for step, tensors in enumerate(train_loader):
            y = tensors[-1]
            X = tensors[:-1]
            pred_y = model(*X)
            loss = loss_func(pred_y, y)

            # back propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            batch_nums += 1

        # calculate average loss during the epoch and write log
        avg_train_loss = train_loss / batch_nums if batch_nums > 0 else 0.
        train_losses.append(avg_train_loss)

        # run validation set
        model.eval()
        val_loss = 0.
        batch_nums = 0
        for step, tensors in enumerate(val_loader):
            y = tensors[-1]
            X = tensors[:-1]
            pred_y = model(*X)
            loss = loss_func(pred_y, y)

            val_loss += loss.item()
            batch_nums += 1
        avg_val_loss = val_loss / batch_nums if batch_nums > 0 else 0.
        logger.info('epoch:%d/%d, train_loss: %.4f, val_loss: %.4f' % (epoch + 1, epochs,
                                                                       avg_train_loss, avg_val_loss))
        val_losses.append(avg_val_loss)

        # save model to file
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_train_loss}, model_path)
        logger.info('model saved:{}'.format(model_path))

        # early_stopping needs the validation loss to check if it has decrease,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(avg_val_loss, model)

        if early_stopping.early_stop:
            logger.warning("Early stopping")
            break

    history = {'train_losses': train_losses, 'val_losses': val_losses}
    return history
