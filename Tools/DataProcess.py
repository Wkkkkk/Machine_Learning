# -*- coding: utf-8 -*-
# author: Wukun
import pickle
import numpy as np
from math import pi, sqrt, ceil
import matplotlib.pyplot as plt

np.random.seed(142857)


class DataProcess:
    naive_sets = {
        "mushroom", "balloon", "mnist", "cifar", "test"
    }

    @staticmethod
    def is_naive(name):
        for naive_dataset in DataProcess.naive_sets:
            if naive_dataset in name:
                return True
        return False

    @staticmethod
    def get_dataset(name, path, train_num=None, tar_idx=None, shuffle=True,
                    quantize=False, quantized=False, one_hot=False, **kwargs):
        x = []
        with open(path, "r", encoding="utf8") as file:
            if DataProcess.is_naive(name):
                for sample in file:
                    x.append(sample.strip().split(","))
            elif name == "bank1.0":
                for sample in file:
                    sample = sample.replace('"', "")
                    x.append(list(map(lambda c: c.strip(), sample.split(";"))))
            else:
                raise NotImplementedError
        if shuffle:
            np.random.shuffle(x)
        tar_idx = -1 if tar_idx is None else tar_idx
        y = np.array([xx.pop(tar_idx) for xx in x])
        if quantized:
            x = np.array(x, dtype=np.float32)
            if one_hot:
                z = []
                y = y.astype(np.int8)
                for yy in y:
                    z.append([0 if i != yy else 1 for i in range(np.max(y) + 1)])
                y = np.array(z, dtype=np.int8)
            else:
                y = y.astype(np.int8)
        else:
            x = np.array(x)
        if quantized or not quantize:
            if train_num is None:
                return x, y
            return (x[:train_num], y[:train_num]), (x[train_num:], y[train_num:])
        x, y, continuous, features, feat_dic, label_dic = DataProcess.quantize_data(x, y, **kwargs)
        if one_hot:
            z = []
            for yy in y:
                z.append([0 if i != yy else 1 for i in range(len(label_dic))])
            y = np.array(z)
        if train_num is None:
            return x, y, continuous, features, feat_dic, label_dic
        return (
            (x[:train_num], y[:train_num]), (x[train_num:], y[train_num:]),
            continuous, features, feat_dic, label_dic
        )

    @staticmethod
    def gen_xor(size=100, scale=1, one_hot=True):
        """
        得到标准正态分布的异或数据集
        :param size:数据集大小
        :param scale:方差
        :param one_hot:是否采取one-hot编码
        :return:
        """
        x = np.random.randn(size) * scale
        y = np.random.randn(size) * scale
        z = np.zeros((size, 2))
        z[x * y >= 0, :] = [0, 1]
        z[x * y < 0, :] = [1, 0]
        if one_hot:
            return np.c_[x, y].astype(np.float32), z
        return np.c_[x, y].astype(np.float32),

    @staticmethod
    def quantize_data(x, y, continuous=None, continuous_rate=0.1, separate=False):
        """
        数值化数据集
        :param x:
        :param y:
        :param continuous:各个特征是否连续
        :param continuous_rate:判断特征连续的阈值
        :param separate:是否将连续特征分开
        :return:
        """
        if isinstance(x, list):
            xt = map(list, zip(*x))  # 列表转置
        else:
            xt = x.T

        features = [set(feat) for feat in xt]
        if continuous is None:
            continuous = np.array([len(feat) >= continuous_rate * len(y) for feat in features])
        else:
            continuous = np.array(continuous)

        # 特征字典
        feat_dic = [{f: i for i, f in enumerate(feats)} if not continuous[i] else None
                    for i, feats in enumerate(features)]

        if np.all(~continuous):
            dtype = np.int
        else:
            dtype = np.double

        # 映射
        x = np.array([[feat_dic[i][f] if not continuous[i] else f for i, f in enumerate(sample)]
                      for sample in x], dtype=dtype)

        if separate:
            x = (x[:, ~continuous].astype(np.int), x[:, continuous])  # ~表示颠倒
        
        # 同样处理y
        label_dic = {f: i for i, f in enumerate(set(y))}
        y = np.array([label_dic[yy] for yy in y], dtype=np.int8)
        label_dic = {i: f for f, i in label_dic.items()}

        return x, y, continuous, features, feat_dic, label_dic