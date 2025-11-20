# -*- coding: utf-8 -*-
"""
PreDataset.py

生成简单的可控分类数据集（支持设置随机种子、样本数量、标签数量）。
默认生成高斯簇（blobs），并划分训练集 / 测试集。

Usage:
    from PreDataset import PreDataset
    X_train, y_train, X_test, y_test = PreDataset(
        n_samples=300, n_features=2, n_classes=3,
        test_ratio=0.2, random_state=42
    )
"""

import numpy as np
from typing import Tuple


def PreDataset(
    n_samples: int = 300,
    n_features: int = 2,
    n_classes: int = 2,
    test_ratio: float = 0.2,
    random_state: int = 42,
    cluster_std: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    生成一个简单的高斯簇分类数据集并划分训练 / 测试集。

    Parameters
    ----------
    n_samples : int
        总样本数量。
    n_features : int
        特征维度（VisualizationTools 只支持 n_features=2 的可视化）。
    n_classes : int
        类别数（标签从 0 到 n_classes-1）。
    test_ratio : float
        测试集占比 (0,1)。
    random_state : int
        随机种子，保证可复现。
    cluster_std : float
        每个簇的方差规模，值越大簇越“散”。

    Returns
    -------
    X_train : (n_train, n_features) ndarray
    y_train : (n_train,) ndarray
    X_test : (n_test, n_features) ndarray
    y_test : (n_test,) ndarray
    """
    assert 0 < test_ratio < 1, "test_ratio must be in (0,1)"
    assert n_classes >= 2, "n_classes must be >= 2"
    assert n_samples >= n_classes, "n_samples must be >= n_classes"

    rng = np.random.default_rng(random_state)

    # 为每个类别分配样本数量（尽量均匀）
    base = n_samples // n_classes
    remainder = n_samples % n_classes
    samples_per_class = [base + (1 if i < remainder else 0) for i in range(n_classes)]

    # 随机生成各类别中心（在 [-5, 5] 超立方体中）
    centers = rng.uniform(low=-5.0, high=5.0, size=(n_classes, n_features))

    X_list = []
    y_list = []

    for class_idx, n_c in enumerate(samples_per_class):
        # 从对应中心生成高斯分布样本
        cov = (cluster_std ** 2) * np.eye(n_features)
        # 多元正态
        samples = rng.multivariate_normal(mean=centers[class_idx], cov=cov, size=n_c)
        labels = np.full(shape=(n_c,), fill_value=class_idx, dtype=int)

        X_list.append(samples)
        y_list.append(labels)

    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    # 打乱
    perm = rng.permutation(len(X))
    X = X[perm]
    y = y[perm]

    # 划分训练 / 测试
    n_test = int(np.floor(len(X) * test_ratio))
    n_train = len(X) - n_test

    X_train = X[:n_train]
    y_train = y[:n_train]
    X_test = X[n_train:]
    y_test = y[n_train:]

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    # 简单自测
    X_tr, y_tr, X_te, y_te = PreDataset(
        n_samples=300, n_features=2, n_classes=3,
        test_ratio=0.25, random_state=0
    )
    print("Train shape:", X_tr.shape, y_tr.shape)
    print("Test shape :", X_te.shape, y_te.shape)
