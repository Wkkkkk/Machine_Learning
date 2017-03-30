# -*- coding: utf-8 -*-
# author: Wukun
import matplotlib.pyplot as plt
import time
from Basic import *

from Tools.DataProcess import DataProcess


class MultinomialNB(NaiveBayes):

    def feed_data(self, x, y, sample_weight=None):
        if sample_weight is not None:
            sample_weight = np.array(sample_weight)
        x, y, _, features, feat_dic, label_dic = DataProcess.quantize_data(x, y)
        cat_counter = np.bincount(y)
        n_possibilities = [len(feats) for feats in features]

        # 获取每个特征在向量中对应的下标
        labels = [y == value for value in range(len(cat_counter))]
        # 按特征组织数据
        labelled_x = [x[ci].T for ci in labels]

        self._x, self._y = x, y
        self._labelled_x, self._label_zip = labelled_x, list(zip(labels, labelled_x))
        self._cat_counter, self._n_possibilities = cat_counter, n_possibilities
        self._label_dic, self._feat_dic = label_dic, feat_dic
        self.feed_sample_weight(sample_weight)

    def feed_sample_weight(self, sample_weight=None):
        """
        利用bincount快速计数
        :param sample_weight: 样本权值
        :return:
        """
        self._con_counter = []
        for dim, _p in enumerate(self._n_possibilities):
            if sample_weight is None:
                self._con_counter.append([
                    np.bincount(xx[dim], minlength=_p) for xx in self._labelled_x])
            else:
                self._con_counter.append([
                    np.bincount(xx[dim], weights=sample_weight[label] / sample_weight[label].mean(), minlength=_p)
                    for label, xx in self._label_zip])

    def _fit(self, lb):
        """
        构建模型
        :param lb:
        :return:
        """
        n_dim = len(self._n_possibilities)
        n_category = len(self._cat_counter)
        p_category = self.get_prior_probability(lb)

        # 计算条件概率
        data = [[] for _ in range(n_dim)]
        for dim, n_possibilities in enumerate(self._n_possibilities):
            data[dim] = [
                [(self._con_counter[dim][c][p] + lb) / (self._cat_counter[c] + lb * n_possibilities)
                 for p in range(n_possibilities)] for c in range(n_category)]
        self._data = [np.array(dim_info) for dim_info in data]

        # 计算后验概率
        def func(input_x, tar_category):
            input_x = np.atleast_2d(input_x).T
            rs = np.ones(input_x.shape[1])
            for d, xx in enumerate(input_x):
                rs *= self._data[d][tar_category][xx]
            return rs * p_category[tar_category]

        return func

    def _transfer_x(self, x):
        """
        特征映射
        :param x:
        :return:
        """
        for i, sample in enumerate(x):
            for j, char in enumerate(sample):
                x[i][j] = self._feat_dic[j][char]
        return x


if __name__ == '__main__':

    train_num = 6000
    (x_train, y_train), (x_test, y_test) = DataProcess.get_dataset(
        "mushroom", "../_Data/mushroom.txt", train_num=train_num, tar_idx=0)

    learning_time = time.time()
    nb = MultinomialNB()
    nb.fit(x_train, y_train)
    learning_time = time.time() - learning_time
    evaluate_time = time.time()
    nb.evaluate(x_train, y_train)
    nb.evaluate(x_test, y_test)
    evaluate_time = time.time() - evaluate_time
    print(
        "Model building  : {:12.6} s\n"
        "Predict      : {:12.6} s\n"
        "Total           : {:12.6} s".format(
            learning_time, evaluate_time,
            learning_time + evaluate_time
        )
    )
