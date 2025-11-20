# -*- coding: utf-8 -*-
"""
QKNN.py

Simple Quantum K-Nearest Neighbors (QKNN) classifier based on angle encoding
and state fidelity as similarity measure.

依赖:
    - numpy
    - qiskit >= 1.0 (for QuantumCircuit, Statevector)

Usage:
    from QKNN import QKNNClassifier
    qknn = QKNNClassifier(n_neighbors=5)
    qknn.fit(X_train, y_train)
    y_pred = qknn.predict(X_test)
"""

import numpy as np
from typing import Optional
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


class QKNNClassifier:
    """
    简单 QKNN 分类器：
    - 将每个样本 x ∈ R^d 映射到 d 个量子比特的量子态 |phi(x)>
    - 相似度 = 量子态保真度 F = |<phi(x_test) | phi(x_train)>|^2
    - "距离" = 1 - F
    """

    def __init__(self, n_neighbors: int = 5):
        assert n_neighbors >= 1, "n_neighbors must be >= 1"
        self.n_neighbors = n_neighbors

        self._X_train: Optional[np.ndarray] = None
        self._y_train: Optional[np.ndarray] = None

        self._x_min: Optional[np.ndarray] = None
        self._x_max: Optional[np.ndarray] = None
        self._scale: Optional[np.ndarray] = None

        self._train_states: Optional[list] = None
        self._n_qubits: Optional[int] = None

    # ---------- Data / encoding utilities ----------

    def _fit_scaler(self, X: np.ndarray):
        """记录特征维度上的最小 / 最大值，用于归一化到 [0, pi]."""
        self._x_min = X.min(axis=0)
        self._x_max = X.max(axis=0)

        # 防止除以 0
        diff = self._x_max - self._x_min
        diff[diff == 0.0] = 1.0
        self._scale = diff

    def _encode_features(self, X: np.ndarray) -> np.ndarray:
        """
        将特征线性缩放到 [0, pi] 区间，作为 Ry 旋转角度。
        """
        assert self._x_min is not None and self._scale is not None
        X_norm = (X - self._x_min) / self._scale  # [0, 1]
        X_theta = X_norm * np.pi                  # [0, pi]
        return X_theta

    def _build_encoding_circuit(self, thetas: np.ndarray) -> QuantumCircuit:
        """
        给定一个样本的角度向量 thetas (d,), 构造编码电路 U(x).
        使用简单的角编码：对第 j 个 qubit 施加 Ry(theta_j).
        """
        qc = QuantumCircuit(self._n_qubits)
        for q, theta in enumerate(thetas):
            qc.ry(float(theta), q)
        return qc

    def _state_from_thetas(self, thetas: np.ndarray) -> Statevector:
        """
        从角度向量生成对应的量子态 |phi(x)>.
        """
        qc = self._build_encoding_circuit(thetas)
        # 初始态默认 |0...0>
        sv = Statevector.from_label("0" * self._n_qubits)
        sv = sv.evolve(qc)
        return sv

    # ---------- Public API ----------

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        保存训练数据，并预先计算每个训练样本对应的量子态。

        Parameters
        ----------
        X : (n_samples, n_features) ndarray
        y : (n_samples,) ndarray
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        assert X.ndim == 2, "X must be 2D array"
        assert y.ndim == 1, "y must be 1D array"
        assert len(X) == len(y), "X and y must have same length"

        self._X_train = X
        self._y_train = y

        self._n_qubits = X.shape[1]

        # 拟合缩放器，并编码所有训练样本
        self._fit_scaler(X)
        train_thetas = self._encode_features(X)

        # 预计算所有训练样本的量子态
        self._train_states = [self._state_from_thetas(thetas) for thetas in train_thetas]

    def _pairwise_quantum_distance(self, X: np.ndarray) -> np.ndarray:
        """
        利用量子态保真度计算 "距离" 矩阵:
            d_ij = 1 - |<phi(x_i) | phi(x_j)>|^2

        Parameters
        ----------
        X : (n_test, n_features) ndarray

        Returns
        -------
        dist : (n_test, n_train) ndarray
        """
        assert self._X_train is not None, "Model has not been fitted yet."
        assert self._train_states is not None, "Train states not computed."

        X = np.asarray(X, dtype=float)
        assert X.ndim == 2
        assert X.shape[1] == self._n_qubits

        # 编码测试样本
        test_thetas = self._encode_features(X)
        test_states = [self._state_from_thetas(thetas) for thetas in test_thetas]

        n_test = len(test_states)
        n_train = len(self._train_states)

        dist = np.empty((n_test, n_train), dtype=float)

        # 逐对计算保真度
        for i, sv_test in enumerate(test_states):
            psi = sv_test.data  # complex vector
            for j, sv_train in enumerate(self._train_states):
                phi = sv_train.data
                inner = np.vdot(psi, phi)         # <psi|phi>
                fidelity = np.abs(inner) ** 2      # |<psi|phi>|^2
                dist[i, j] = 1.0 - fidelity        # 越小越相似

        return dist

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        对输入样本进行类别预测（基于量子距离的 KNN 投票）。

        Parameters
        ----------
        X : (n_test, n_features) ndarray

        Returns
        -------
        y_pred : (n_test,) ndarray
        """
        X = np.asarray(X, dtype=float)
        assert X.ndim == 2, "X must be 2D array"
        assert self._X_train is not None, "Model has not been fitted yet."

        n_test = X.shape[0]
        n_train = self._X_train.shape[0]
        k = min(self.n_neighbors, n_train)

        dists = self._pairwise_quantum_distance(X)  # (n_test, n_train)

        # 找到每个样本的 K 个最近邻索引（距离最小）
        knn_indices = np.argpartition(dists, kth=k - 1, axis=1)[:, :k]
        knn_labels = self._y_train[knn_indices]  # (n_test, k)

        y_pred = np.empty(n_test, dtype=self._y_train.dtype)
        for i in range(n_test):
            labels, counts = np.unique(knn_labels[i], return_counts=True)
            y_pred[i] = labels[np.argmax(counts)]
        return y_pred

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        计算预测准确率。
        """
        y_pred = self.predict(X)
        y = np.asarray(y)
        assert len(y_pred) == len(y), "X and y length mismatch"
        return float(np.mean(y_pred == y))
