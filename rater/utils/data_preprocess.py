# -*- coding:utf-8 -*-

"""
Created on Dec 10, 2017
@author: jachin,Nie

This script is used to preprocess the raw data file

"""

import math

import pandas as pd


def gen_criteo_category_index(file_path):
    cate_dict = []
    for i in range(26):
        cate_dict.append({})
    for line in open(file_path, 'r'):
        datas = line.replace('\n', '').split('\t')
        for i, item in enumerate(datas[14:]):
            if not cate_dict[i].has_key(item):
                cate_dict[i][item] = len(cate_dict[i])
    return cate_dict


def write_criteo_category_index(file_path, cate_dict_arr):
    f = open(file_path, 'w')
    for i, cate_dict in enumerate(cate_dict_arr):
        for key in cate_dict:
            f.write(str(i) + ',' + key + ',' + str(cate_dict[key]) + '\n')


def load_criteo_category_index(file_path):
    f = open(file_path, 'r')
    cate_dict = []
    for i in range(39):
        cate_dict.append({})
    for line in f:
        datas = line.strip().split(',')
        cate_dict[int(datas[0])][datas[1]] = int(datas[2])
    return cate_dict


def read_raw_criteo_data(file_path, embedding_path, type):
    """
    :param file_path: string
    :param type: string (train or test)
    :return: result: dict
            result['continuous_feat']:two-dim array
            result['category_feat']:dict
            result['category_feat']['index']:two-dim array
            result['category_feat']['value']:two-dim array
            result['label']: one-dim array
    """
    begin_index = 1
    if type != 'train' and type != 'test':
        print("type error")
        return {}
    elif type == 'test':
        begin_index = 0
    cate_embedding = load_criteo_category_index(embedding_path)
    result = {'continuous_feat': [], 'category_feat': {'index': [], 'value': []}, 'label': [], 'feature_sizes': []}
    for i, item in enumerate(cate_embedding):
        result['feature_sizes'].append(len(item))
    f = open(file_path)
    for line in f:
        datas = line.replace('\n', '').split('\t')

        indexs = []
        values = []
        flag = True
        for i, item in enumerate(datas[begin_index + 13:]):
            if not cate_embedding[i].has_key(item):
                flag = False
                break
            indexs.append(cate_embedding[i][item])
            values.append(1)
        if not flag:
            continue
        result['category_feat']['index'].append(indexs)
        result['category_feat']['value'].append(values)

        if type == 'train':
            result['label'].append(int(datas[0]))
        else:
            result['label'].append(0)

        continuous_array = []
        for item in datas[begin_index:begin_index + 13]:
            if item == '':
                continuous_array.append(-10.0)
            elif float(item) < 2.0:
                continuous_array.append(float(item))
            else:
                continuous_array.append(math.log(float(item)))
        result['continuous_feat'].append(continuous_array)

    return result


def read_criteo_data(file_path, emb_file):
    result = {'label': [], 'index': [], 'value': [], 'feature_sizes': []}
    cate_dict = load_criteo_category_index(emb_file)
    for item in cate_dict:
        result['feature_sizes'].append(len(item))
    f = open(file_path, 'r')
    for line in f:
        datas = line.strip().split(',')
        result['label'].append(int(datas[0]))
        indexs = [int(item) for item in datas[1:]]
        values = [1 for i in range(39)]
        result['index'].append(indexs)
        result['value'].append(values)
    return result


def gen_criteo_category_emb_from_libffmfile(filepath, dir_path):
    fr = open(filepath)
    cate_emb_arr = [{} for i in range(39)]
    for line in fr:
        datas = line.strip().split(' ')
        for item in datas[1:]:
            [filed, index, value] = item.split(':')
            filed = int(filed)
            index = int(index)
            if not cate_emb_arr[filed].has_key(index):
                cate_emb_arr[filed][index] = len(cate_emb_arr[filed])

    with open(dir_path, 'w') as f:
        for i, item in enumerate(cate_emb_arr):
            for key in item:
                f.write(str(i) + ',' + str(key) + ',' + str(item[key]) + '\n')


def gen_emb_input_file(filepath, emb_file, dir_path):
    cate_dict = load_criteo_category_index(emb_file)
    fr = open(filepath, 'r')
    fw = open(dir_path, 'w')
    for line in fr:
        row = []
        datas = line.strip().split(' ')
        row.append(datas[0])
        for item in datas[1:]:
            [filed, index, value] = item.split(':')
            filed = int(filed)
            row.append(str(cate_dict[filed][index]))
        fw.write(','.join(row) + '\n')


def read_csv_dataset(train_csv, task='like'):
    train_dict = {}
    test_dict = {}
    train_csv = pd.read_csv(train_csv)
    if task == 'like':
        label = train_csv[task]
    elif task == 'finish':
        label = train_csv[task]
    train_dict['label'] = label[0:int(len(label) * 0.8)].to_list()
    test_dict['label'] = label[int(len(label) * 0.8) + 1:-1].to_list()

    feild = ['uid', 'user_city', 'item_id', 'author_id', 'item_city', 'channel', 'music_id', 'video_duration']
    value = [1] * len(feild)
    values = [value for i in range(len(label))]
    train_dict['value'] = values[0:int(len(label) * 0.8)]
    test_dict['value'] = values[int(len(label) * 0.8) + 1:-1]

    feature_sizes = [73974, 397, 4122689, 850308, 462, 5, 89779, 641]
    train_dict['feature_sizes'] = feature_sizes
    test_dict['feature_sizes'] = feature_sizes
    '''
    creat_time_segment=35898.1
    min_num=53015373867
    train_csv['creat_time']=train_csv['creat_time'].apply(lambda x:int((x-min_num)/creat_time_segment))
    '''

    temp = train_csv[feild].values
    train_dict['index'] = temp[0:int(len(label) * 0.8)].tolist()
    test_dict['index'] = temp[int(len(label) * 0.8) + 1:-1].tolist()

    return train_dict, test_dict


def read_csv_dataset_pred(pred_csv, task='like'):
    pred_dict = {}
    train_csv = pd.read_csv(pred_csv)
    if task == 'like':
        label = train_csv[task]
    elif task == 'finish':
        label = train_csv[task]
    pred_dict['label'] = label.to_list()

    feild = ['uid', 'user_city', 'item_id', 'author_id', 'item_city', 'channel', 'music_id', 'creat_time',
             'video_duration']
    value = [1] * len(feild)
    values = [value for i in range(len(label))]
    pred_dict['value'] = values

    feature_sizes = []
    for i in feild:
        feature_size = max(train_csv[i]) + 1
        if i == 'creat_time':
            feature_size = 2010
        feature_sizes.append(feature_size)
    pred_dict['feature_sizes'] = feature_sizes

    creat_time_segment = 35898.1
    min_num = 53015373867
    train_csv['creat_time'] = train_csv['creat_time'].apply(lambda x: int((x - min_num) / creat_time_segment))

    temp = train_csv[feild].values
    pred_dict['index'] = temp.tolist()
    return pred_dict
