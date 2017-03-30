# -*- coding: utf-8 -*-
# author: Wukun
import time
from NaiveBayes.Basic import *

from Tools.DataProcess import DataProcess
from Tools.Timing import Timing


class GuassianNB(NaiveBayes):
    GaussianNBTiming = Timing()

    @GaussianNBTiming.timeit(level=1, prefix="[API] ")
    def feed_data(self, x, y, sample_weight=None):
        if sample_weight is not None:
            sample_weight = np.array(sample_weight)

        # 浮点化x，数值化y
        x = np.array([list(map(lambda c:float(c), sample)) for sample in x])
        labels = list(set(y))
        label_dic = {label: i for i, label in enumerate(labels)}
        y = np.array([label_dic[yy] for yy in y])
        label_dic = {i: label for i, label in label_dic.items()}

        # 按标签组织数据
        cat_counter = np.bincount(y)
        labels = [y == value for value in range(len(cat_counter))]
        labelled_x = [x[label].T for label in labels]

        self._x, self._y = x.T, y
        self._labelled_x, self._label_zip = labelled_x, labels
        self._cat_counter, self.label_dic = cat_counter, label_dic
        self.feed_sample_weight(sample_weight)

    @GaussianNBTiming.timeit(level=1, prefix="[Core] ")
    def feed_sample_weight(self, sample_weight=None):
        if sample_weight is not None:
            local_weight = sample_weight * len(sample_weight)
            for i, label in enumerate(self._label_zip):
                self._labelled_x[i] *= local_weight[label]

    @GaussianNBTiming.timeit(level=1, prefix="[Core] ")
    def _fit(self, lb):
        lb = 0
        n_category = len(self._cat_counter)
        p_category = self.get_prior_probability(lb)

        # 计算条件概率
        data = [
            NBFunctions.gaussian_maximum_likelihood(
                self._labelled_x, n_category, dim) for dim in range(len(self._x))]
        self._data = data

        # 计算后验概率
        def func(input_x, tar_category):
            input_x = np.atleast_2d(input_x).T
            rs = np.ones(input_x.shape[1])
            for d, xx in enumerate(input_x):
                rs *= data[d][tar_category](xx)
            return rs * p_category[tar_category]

        return func

