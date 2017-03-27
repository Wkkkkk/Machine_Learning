# -*- coding: utf-8 -*-
# author: Wukun
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Tools.Timing import Timing
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


class ClassifierBase:
    clf_timing = Timing()

    def __init__(self, *args, **kwargs):
        self._title = self._name = None
        self._metrics, self._available_metrics = [], {
            "acc": ClassifierBase.acc
        }

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)

    def __getitem__(self, item):
        if isinstance(item, str):
            return getattr(self, "_" + item)

    @property
    def name(self):
        return self.__class__.__name__ if self._name is None else self._name

    @property
    def title(self):
        return str(self) if self._title is None else self._title

    @staticmethod
    def disable_timing():
        ClassifierBase.clf_timing.disable()

    @staticmethod
    def show_timing_log(level=2):
        ClassifierBase.clf_timing.show_timing_log(level)

    @staticmethod
    def acc(y, y_pred, weights=None):
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        if not isinstance(y_pred, np.ndarray):
            y_pred = np.array(y_pred)
        if weights is not None:
            return np.sum((y == y_pred) * weights) / len(y)
        return np.sum(y == y_pred) / len(y)

    # noinspection PyTypeChecker
    @staticmethod
    def f1_score(y, y_pred):
        tp = np.sum(y * y_pred)
        if tp == 0:
            return .0
        fp = np.sum((1 - y) * y_pred)
        fn = np.sum(y * (1 - y_pred))
        return 2 * tp / (2 * tp + fn + fp)

    def get_metrics(self, metrics):
        if len(metrics) == 0:
            for metric in self._metrics:
                metrics.append(metric)
            return metrics
        for i in range(len(metrics) - 1, -1, -1):
            metric = metrics[i]
            if isinstance(metric, str):
                try:
                    metrics[i] = self._available_metrics[metric]
                except AttributeError:
                    metrics.pop(i)
        return metrics

    def predict(self, x, get_raw_results=False):
        pass

    def evaluate(self, x, y, metrics=None, tar=None, prefix="Acc"):
        if metrics is None:
            metrics = ["acc"]
        self.get_metrics(metrics)
        logs, y_pred = [], self.predict(x)
        y = np.array(y)
        if y.ndim == 2:
            y = np.argmax(y, axis=1)
        for metric in metrics:
            logs.append(metric(y, y_pred))
        if tar is None:
            tar = 0
        if isinstance(tar, int):
            print(prefix + ": {:12.8}".format(logs[tar]))
        return logs

    def scatter2d(self, x, y, padding=0.5, title=None):
        axis, labels = np.array(x).T, np.array(y)

        print("=" * 30 + "\n" + str(self))
        x_min, x_max = np.min(axis[0]), np.max(axis[0])
        y_min, y_max = np.min(axis[1]), np.max(axis[1])
        x_padding = max(abs(x_min), abs(x_max)) * padding
        y_padding = max(abs(y_min), abs(y_max)) * padding
        x_min -= x_padding
        x_max += x_padding
        y_min -= y_padding
        y_max += y_padding

        if labels.ndim == 1:
            _dic = {c: i for i, c in enumerate(set(labels))}
            n_label = len(_dic)
            labels = np.array([_dic[label] for label in labels])
        else:
            n_label = labels.shape[1]
            labels = np.argmax(labels, axis=1)
        colors = plt.cm.rainbow([i / n_label for i in range(n_label)])[labels]

        if title is None:
            title = self.title

        _indices = [labels == i for i in range(np.max(labels) + 1)]
        _scatters = []
        plt.figure()
        plt.title(title)
        for _index in _indices:
            _scatters.append(plt.scatter(axis[0][_index], axis[1][_index], c=colors[_index]))
        plt.legend(_scatters, ["$c_{}$".format("{" + str(i) + "}") for i in range(len(_scatters))],
                   ncol=math.ceil(math.sqrt(len(_scatters))), fontsize=8)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.show()

        print("Done.")

    def scatter3d(self, x, y, padding=0.1, title=None):
        axis, labels = np.array(x).T, np.array(y)

        print("=" * 30 + "\n" + str(self))
        x_min, x_max = np.min(axis[0]), np.max(axis[0])
        y_min, y_max = np.min(axis[1]), np.max(axis[1])
        z_min, z_max = np.min(axis[2]), np.max(axis[2])
        x_padding = max(abs(x_min), abs(x_max)) * padding
        y_padding = max(abs(y_min), abs(y_max)) * padding
        z_padding = max(abs(z_min), abs(z_max)) * padding
        x_min -= x_padding
        x_max += x_padding
        y_min -= y_padding
        y_max += y_padding
        z_min -= z_padding
        z_max += z_padding

        def transform_arr(arr):
            if arr.ndim == 1:
                _dic = {c: i for i, c in enumerate(set(arr))}
                n_dim = len(_dic)
                arr = np.array([_dic[label] for label in arr])
            else:
                n_dim = arr.shape[1]
                arr = np.argmax(arr, axis=1)
            return arr, n_dim

        if title is None:
            try:
                title = self.title
            except AttributeError:
                title = str(self)

        labels, n_label = transform_arr(labels)
        colors = plt.cm.rainbow([i / n_label for i in range(n_label)])[labels]
        _indices = [labels == i for i in range(n_label)]
        _scatters = []
        fig = plt.figure()
        plt.title(title)
        ax = fig.add_subplot(111, projection='3d')
        for _index in _indices:
            _scatters.append(ax.scatter(axis[0][_index], axis[1][_index], axis[2][_index], c=colors[_index]))
        ax.legend(_scatters, ["$c_{}$".format("{" + str(i) + "}") for i in range(len(_scatters))],
                  ncol=math.ceil(math.sqrt(len(_scatters))), fontsize=8)
        plt.show()

    def visualize2d(self, x, y, padding=0.1, dense=200,
                    title=None, show_org=False, show_background=True, emphasize=None):
        axis, labels = np.array(x).T, np.array(y)

        print("=" * 30 + "\n" + str(self))
        decision_function = lambda _xx: self.predict(_xx)

        nx, ny, padding = dense, dense, padding
        x_min, x_max = np.min(axis[0]), np.max(axis[0])
        y_min, y_max = np.min(axis[1]), np.max(axis[1])
        x_padding = max(abs(x_min), abs(x_max)) * padding
        y_padding = max(abs(y_min), abs(y_max)) * padding
        x_min -= x_padding
        x_max += x_padding
        y_min -= y_padding
        y_max += y_padding

        def get_base(_nx, _ny):
            _xf = np.linspace(x_min, x_max, _nx)
            _yf = np.linspace(y_min, y_max, _ny)
            n_xf, n_yf = np.meshgrid(_xf, _yf)
            return _xf, _yf, np.c_[n_xf.ravel(), n_yf.ravel()]

        xf, yf, base_matrix = get_base(nx, ny)

        t = time.time()
        z = decision_function(base_matrix).reshape((nx, ny))
        print("Decision Time: {:8.6} s".format(time.time() - t))

        print("Drawing figures...")
        xy_xf, xy_yf = np.meshgrid(xf, yf, sparse=True)
        if labels.ndim == 1:
            _dic = {c: i for i, c in enumerate(set(labels))}
            n_label = len(_dic)
            labels = np.array([_dic[label] for label in labels])
        else:
            n_label = labels.shape[1]
            labels = np.argmax(labels, axis=1)
        colors = plt.cm.rainbow([i / n_label for i in range(n_label)])[labels]

        if title is None:
            title = self.title

        if show_org:
            plt.figure()
            plt.scatter(axis[0], axis[1], c=colors)
            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)
            plt.show()

        plt.figure()
        plt.title(title)
        if show_background:
            plt.pcolormesh(xy_xf, xy_yf, z, cmap=plt.cm.Paired)
        else:
            plt.contour(xf, yf, z, c='k-', levels=[0])
        plt.scatter(axis[0], axis[1], c=colors)
        if emphasize is not None:
            _indices = np.array([False] * len(axis[0]))
            _indices[np.array(emphasize)] = True
            plt.scatter(axis[0][_indices], axis[1][_indices], s=80,
                        facecolors="None", zorder=10)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.show()

        print("Done.")

    def visualize3d(self, x, y, padding=0.1, dense=100,
                    title=None, show_org=False, show_background=True, emphasize=None):
        if False:
            print(Axes3D.add_artist)
        axis, labels = np.array(x).T, np.array(y)

        print("=" * 30 + "\n" + str(self))
        decision_function = lambda _x: self.predict(_x)

        nx, ny, nz, padding = dense, dense, dense, padding
        x_min, x_max = np.min(axis[0]), np.max(axis[0])
        y_min, y_max = np.min(axis[1]), np.max(axis[1])
        z_min, z_max = np.min(axis[2]), np.max(axis[2])
        x_padding = max(abs(x_min), abs(x_max)) * padding
        y_padding = max(abs(y_min), abs(y_max)) * padding
        z_padding = max(abs(z_min), abs(z_max)) * padding
        x_min -= x_padding
        x_max += x_padding
        y_min -= y_padding
        y_max += y_padding
        z_min -= z_padding
        z_max += z_padding

        def get_base(_nx, _ny, _nz):
            _xf = np.linspace(x_min, x_max, _nx)
            _yf = np.linspace(y_min, y_max, _ny)
            _zf = np.linspace(z_min, z_max, _nz)
            n_xf, n_yf, n_zf = np.meshgrid(_xf, _yf, _zf)
            return _xf, _yf, _zf, np.c_[n_xf.ravel(), n_yf.ravel(), n_zf.ravel()]

        xf, yf, zf, base_matrix = get_base(nx, ny, nz)

        t = time.time()
        z_xyz = decision_function(base_matrix).reshape((nx, ny, nz))
        p_classes = decision_function(x).astype(np.int8)
        _, _, _, base_matrix = get_base(10, 10, 10)
        z_classes = decision_function(base_matrix).astype(np.int8)
        print("Decision Time: {:8.6} s".format(time.time() - t))

        print("Drawing figures...")
        z_xy = np.average(z_xyz, axis=2)
        z_yz = np.average(z_xyz, axis=1)
        z_xz = np.average(z_xyz, axis=0)

        xy_xf, xy_yf = np.meshgrid(xf, yf, sparse=True)
        yz_xf, yz_yf = np.meshgrid(yf, zf, sparse=True)
        xz_xf, xz_yf = np.meshgrid(xf, zf, sparse=True)

        def transform_arr(arr):
            if arr.ndim == 1:
                _dic = {c: i for i, c in enumerate(set(arr))}
                n_dim = len(_dic)
                arr = np.array([_dic[label] for label in arr])
            else:
                n_dim = arr.shape[1]
                arr = np.argmax(arr, axis=1)
            return arr, n_dim

        labels, n_label = transform_arr(labels)
        p_classes, _ = transform_arr(p_classes)
        z_classes, _ = transform_arr(z_classes)
        colors = plt.cm.rainbow([i / n_label for i in range(n_label)])

        if title is None:
            try:
                title = self.title
            except AttributeError:
                title = str(self)

        if show_org:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(axis[0], axis[1], axis[2], c=colors[labels])
            plt.show()

        fig = plt.figure(figsize=(16, 4), dpi=100)
        plt.title(title)
        ax1 = fig.add_subplot(131, projection='3d')
        ax2 = fig.add_subplot(132, projection='3d')
        ax3 = fig.add_subplot(133, projection='3d')

        ax1.set_title("Org")
        ax2.set_title("Pred")
        ax3.set_title("Boundary")

        ax1.scatter(axis[0], axis[1], axis[2], c=colors[labels])
        ax2.scatter(axis[0], axis[1], axis[2], c=colors[p_classes], s=15)
        xyz_xf, xyz_yf, xyz_zf = base_matrix[..., 0], base_matrix[..., 1], base_matrix[..., 2]
        ax3.scatter(xyz_xf, xyz_yf, xyz_zf, c=colors[z_classes], s=15)

        plt.show()
        plt.close()

        fig = plt.figure(figsize=(16, 4), dpi=100)
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)

        def _draw(_ax, _x, _xf, _y, _yf, _z):
            if show_background:
                _ax.pcolormesh(_x, _y, _z > 0, cmap=plt.cm.Paired)
            else:
                _ax.contour(_xf, _yf, _z, c='k-', levels=[0])

        def _emphasize(_ax, axis0, axis1, _c):
            _ax.scatter(axis0, axis1, c=_c)
            if emphasize is not None:
                _indices = np.array([False] * len(axis[0]))
                _indices[np.array(emphasize)] = True
                _ax.scatter(axis0[_indices], axis1[_indices], s=80,
                            facecolors="None", zorder=10)

        colors = colors[labels]

        ax1.set_title("xy figure")
        _draw(ax1, xy_xf, xf, xy_yf, yf, z_xy)
        _emphasize(ax1, axis[0], axis[1], colors)

        ax2.set_title("yz figure")
        _draw(ax2, yz_xf, yf, yz_yf, zf, z_yz)
        _emphasize(ax2, axis[1], axis[2], colors)

        ax3.set_title("xz figure")
        _draw(ax3, xz_xf, xf, xz_yf, zf, z_xz)
        _emphasize(ax3, axis[0], axis[2], colors)

        plt.show()

        print("Done.")

