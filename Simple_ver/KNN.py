# -*- coding: utf-8 -*-
"""
KNN.py

K-Nearest Neighbors (KNN) 分类器，仅依赖 numpy。

Usage:
    from KNN import KNNClassifier
    knn = KNNClassifier(n_neighbors=5, metric="euclidean")
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
"""

import numpy as np
from typing import Optional


class KNNClassifier:
    """
    简单 KNN 分类器（仅支持分类任务）。
    """

    def __init__(self, n_neighbors: int = 5, metric: str = "euclidean"):
        """
        Parameters
        ----------
        n_neighbors : int
            近邻数量 K。
        metric : str
            距离度量，目前仅支持 "euclidean"。
        """
        assert n_neighbors >= 1, "n_neighbors must be >= 1"
        assert metric in ("euclidean",), "Currently only 'euclidean' is supported."

        self.n_neighbors = n_neighbors
        self.metric = metric

        self._X_train: Optional[np.ndarray] = None
        self._y_train: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        保存训练数据（KNN 是惰性学习，没有显式训练过程）。

        Parameters
        ----------
        X : (n_samples, n_features) ndarray
        y : (n_samples,) ndarray
        """
        X = np.asarray(X)
        y = np.asarray(y)

        assert X.ndim == 2, "X must be 2D array"
        assert y.ndim == 1, "y must be 1D array"
        assert len(X) == len(y), "X and y must have same length"

        self._X_train = X
        self._y_train = y

    def _pairwise_dist(self, X: np.ndarray) -> np.ndarray:
        """
        计算预测样本与训练样本之间的欧式距离矩阵。

        Parameters
        ----------
        X : (n_test, n_features) ndarray

        Returns
        -------
        dist : (n_test, n_train) ndarray
        """
        assert self._X_train is not None, "Model has not been fitted yet."
        X_train = self._X_train

        # 使用 (x - y)^2 = x^2 + y^2 - 2xy 的向量化写法
        # X: (n_test, d), X_train: (n_train, d)
        X_norm = np.sum(X ** 2, axis=1).reshape(-1, 1)          # (n_test, 1)
        X_train_norm = np.sum(X_train ** 2, axis=1).reshape(1, -1)  # (1, n_train)

        # 广播： (n_test, 1) + (1, n_train) - 2*(n_test, d)*(d, n_train)
        dist_sq = X_norm + X_train_norm - 2.0 * np.dot(X, X_train.T)
        dist_sq = np.maximum(dist_sq, 0.0)  # 数值稳定性
        dist = np.sqrt(dist_sq)
        return dist

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        对输入样本进行类别预测。

        Parameters
        ----------
        X : (n_test, n_features) ndarray

        Returns
        -------
        y_pred : (n_test,) ndarray
        """
        X = np.asarray(X)
        assert X.ndim == 2, "X must be 2D array"
        assert self._X_train is not None, "Model has not been fitted yet."

        n_test = X.shape[0]
        n_train = self._X_train.shape[0]
        k = min(self.n_neighbors, n_train)  # 防止 K > n_train

        # (n_test, n_train)
        dists = self._pairwise_dist(X)

        # 找到每个样本的 K 个最近邻索引
        # argsort 返回从小到大的索引，取前 k 个
        knn_indices = np.argpartition(dists, kth=k - 1, axis=1)[:, :k]  # (n_test, k)
        knn_labels = self._y_train[knn_indices]  # (n_test, k)

        # 对每一行做投票
        y_pred = np.empty(n_test, dtype=self._y_train.dtype)
        for i in range(n_test):
            labels, counts = np.unique(knn_labels[i], return_counts=True)
            # 取频数最大的标签，若并列，np.unique 已排序，等价于取最小标签
            y_pred[i] = labels[np.argmax(counts)]

        return y_pred

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        计算预测准确率。

        Parameters
        ----------
        X : (n_test, n_features) ndarray
        y : (n_test,) ndarray

        Returns
        -------
        acc : float
        """
        y_pred = self.predict(X)
        y = np.asarray(y)
        assert len(y_pred) == len(y), "X and y length mismatch"
        return float(np.mean(y_pred == y))
