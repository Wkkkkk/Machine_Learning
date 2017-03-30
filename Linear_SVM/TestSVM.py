# -*- coding: utf-8 -*-
# author: Wukun
import matplotlib.pyplot as plt

from Linear_SVM.SVM import SVM
from _SKlearn.SVM import SKSVM

from Tools.DataProcess import DataProcess


def main():

    (x_train, y_train), (x_test, y_test), *_ = DataProcess.get_dataset(
        "mushroom", "../_Data/mushroom.txt", train_num=100, quantize=True, tar_idx=0)
    y_train[y_train == 0] = -1
    y_test[y_test == 0] = -1

    svm = SKSVM()
    svm.fit(x_train, y_train)
    svm.evaluate(x_train, y_train)
    svm.evaluate(x_test, y_test)

    svm = SVM()
    _logs = [_log[0] for _log in svm.fit(
        x_train, y_train, metrics=["acc"], x_test=x_test, y_test=y_test
    )]
    # svm.fit(x_train, y_train, p=12)
    svm.evaluate(x_train, y_train)
    svm.evaluate(x_test, y_test)

    plt.figure()
    plt.title(svm.title)
    plt.plot(range(len(_logs)), _logs)
    plt.show()

    svm.show_timing_log()

if __name__ == '__main__':
    main()
