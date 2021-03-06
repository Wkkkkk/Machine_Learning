# -*- coding: utf-8 -*-
# author: Wukun
from DecisionTree.Tree import *
from Tools.DataProcess import DataProcess


class RandomForest(ClassifierBase):
    RandomForestTiming = Timing()
    _cvd_trees = {
        "ID3": ID3Tree,
        "C45": C45Tree,
        "Cart": CartTree
    }

    def __init__(self):
        super(RandomForest, self).__init__()
        self._tree, self._trees = "", []

    @property
    def title(self):
        return "Tree: {}; Num: {}".format(self._tree, len(self._trees))

    @staticmethod
    @RandomForestTiming.timeit(level=2, prefix="[StaticMethod] ")
    def most_appearance(arr):
        u, c = np.unique(arr, return_counts=True)
        return u[np.argmax(c)]

    @RandomForestTiming.timeit(level=1, prefix="[API] ")
    def fit(self, x, y, sample_weight=None, tree="Cart", epoch=10, feature_bound="log", **kwargs):
        x, y = np.atleast_2d(x), np.array(y)
        n_sample = len(y)
        self._tree = tree
        for _ in range(epoch):
            tmp_tree = RandomForest._cvd_trees[tree](**kwargs)
            _indices = np.random.randint(n_sample, size=n_sample)
            if sample_weight is None:
                _local_weight = None
            else:
                _local_weight = sample_weight[_indices]
                _local_weight /= _local_weight.sum()
            tmp_tree.fit(x[_indices], y[_indices], sample_weight=_local_weight, feature_bound=feature_bound)
            self._trees.append(deepcopy(tmp_tree))

    @RandomForestTiming.timeit(level=1, prefix="[API] ")
    def predict(self, x, bound=None):
        if bound is None:
            _matrix = np.array([_tree.predict(x) for _tree in self._trees]).T
        else:
            _matrix = np.array([_tree.predict(x) for _tree in self._trees[:bound]]).T
        return np.array([RandomForest.most_appearance(rs) for rs in _matrix])

if __name__ == '__main__':
    import time

    train_num = 100
    (x_train, y_train), (x_test, y_test) = DataProcess.get_dataset(
        "mushroom", "../_Data/mushroom.txt", train_num=train_num, tar_idx=0)

    learning_time = time.time()
    forest = RandomForest()
    forest.fit(x_train, y_train)
    learning_time = time.time() - learning_time
    estimation_time = time.time()
    forest.evaluate(x_train, y_train)
    forest.evaluate(x_test, y_test)
    estimation_time = time.time() - estimation_time
    print(
        "Model building  : {:12.6} s\n"
        "Estimation      : {:12.6} s\n"
        "Total           : {:12.6} s".format(
            learning_time, estimation_time,
            learning_time + estimation_time
        )
    )
    forest.show_timing_log()
