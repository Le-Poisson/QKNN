# -*- coding: utf-8 -*-
"""
VisualizationTools.py

针对二维特征的数据，绘制：
1) KNN 决策边界
2) 训练 / 测试样本散点图

Usage:
    from VisualizationTools import plot_knn_decision_boundary

    plot_knn_decision_boundary(
        X_train, y_train, knn_model,
        X_test=X_test, y_test=y_test,
        title="KNN Decision Boundary (k=5)",
        save_path="figs/knn_decision_boundary.png"
    )
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


def plot_knn_decision_boundary(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model,
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    h: float = 0.05,
    figsize=(8, 6),
    title: str = "KNN Decision Boundary",
    save_path: Optional[str] = None,
):
    """
    绘制二维特征下的 KNN 决策边界和样本点。

    Parameters
    ----------
    X_train : (n_train, 2) ndarray
    y_train : (n_train,) ndarray
    model : 已经 fit 的 KNNClassifier 实例（需要有 predict 方法）
    X_test : (n_test, 2) ndarray, optional
    y_test : (n_test,) ndarray, optional
    h : float
        网格步长（越小越精细，但越慢）。
    figsize : tuple
        画布大小。
    title : str
        图标题。
    save_path : str, optional
        若不为 None，则保存图片到该路径。
    """
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)

    assert X_train.shape[1] == 2, "Visualization only supports 2D features."

    if X_test is not None:
        X_test = np.asarray(X_test)
        assert X_test.shape[1] == 2, "X_test must also be 2D for visualization."
        if y_test is not None:
            y_test = np.asarray(y_test)

    # 网格边界
    x_min, x_max = X_train[:, 0].min() - 1.0, X_train[:, 0].max() + 1.0
    y_min, y_max = X_train[:, 1].min() - 1.0, X_train[:, 1].max() + 1.0

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h)
    )
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # 网格上预测类别
    Z = model.predict(grid_points)
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=figsize)

    # 背景决策区域
    # cmap：背景软颜色；edgecolors='k' 可以让点有黑色边框更清晰
    from matplotlib.colors import ListedColormap
    light_colors = ListedColormap(["#FFEEEE", "#EEFFEE", "#EEEEFF", "#FFF5CC"])
    bold_colors = ListedColormap(["#FF0000", "#00AA00", "#0000FF", "#FFAA00"])

    plt.contourf(xx, yy, Z, alpha=0.4, cmap=light_colors)

    # 训练集点
    scatter_train = plt.scatter(
        X_train[:, 0], X_train[:, 1],
        c=y_train,
        cmap=bold_colors,
        edgecolors="k",
        marker="o",
        label="Train"
    )

    # 测试集点（若提供）
    if X_test is not None:
        if y_test is not None:
            plt.scatter(
                X_test[:, 0], X_test[:, 1],
                c=y_test,
                cmap=bold_colors,
                edgecolors="k",
                marker="^",
                label="Test"
            )
        else:
            plt.scatter(
                X_test[:, 0], X_test[:, 1],
                c="k",
                edgecolors="w",
                marker="^",
                label="Test (no labels)"
            )

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title(title)
    plt.legend(loc="best", framealpha=0.8)

    # 让图像更紧凑一些
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"[Info] Figure saved to: {save_path}")

    plt.show()
