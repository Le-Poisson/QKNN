# -*- coding: utf-8 -*-
"""
main.py

精简入口脚本：
1) 调用 PreDataset 生成数据集（可通过命令行参数控制）
2) 训练 KNN / QKNN 分类器
3) 打印训练 / 测试集准确率
4) 若特征维度为 2，则绘制决策边界图
"""

import argparse

from PreDataset import PreDataset
from KNN import KNNClassifier
from QKNN import QKNNClassifier
from VisualizationTools import plot_knn_decision_boundary


def parse_args():
    parser = argparse.ArgumentParser(description="Classical KNN vs Quantum KNN (QKNN) demo")

    parser.add_argument("--n-samples", type=int, default=300,
                        help="Total number of samples.")
    parser.add_argument("--n-features", type=int, default=2,
                        help="Number of features. (2D is needed for visualization.)")
    parser.add_argument("--n-classes", type=int, default=3,
                        help="Number of classes (labels).")
    parser.add_argument("--test-ratio", type=float, default=0.2,
                        help="Test set ratio in (0,1).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument("--k", type=int, default=5,
                        help="Number of neighbors (K in KNN/QKNN).")
    parser.add_argument("--cluster-std", type=float, default=1.0,
                        help="Cluster std; larger gives more overlap between classes.")
    parser.add_argument("--no-plot", action="store_true",
                        help="Disable plotting even if n_features == 2.")
    parser.add_argument("--fig-path", type=str, default="figs/decision_boundary.png",
                        help="Where to save the decision boundary figure.")
    parser.add_argument("--model", type=str, default="qknn",
                        choices=["knn", "qknn"],
                        help="Choose 'knn' for classical KNN, 'qknn' for quantum KNN.")

    return parser.parse_args()


def main():
    args = parse_args()

    print("========== KNN / QKNN Demo ==========")
    print(f"model      = {args.model}")
    print(f"n_samples  = {args.n_samples}")
    print(f"n_features = {args.n_features}")
    print(f"n_classes  = {args.n_classes}")
    print(f"test_ratio = {args.test_ratio}")
    print(f"seed       = {args.seed}")
    print(f"k          = {args.k}")
    print("=====================================")

    # 1. 生成数据
    X_train, y_train, X_test, y_test = PreDataset(
        n_samples=args.n_samples,
        n_features=args.n_features,
        n_classes=args.n_classes,
        test_ratio=args.test_ratio,
        random_state=args.seed,
        cluster_std=args.cluster_std,
    )

    # 2. 初始化模型
    if args.model == "knn":
        model = KNNClassifier(n_neighbors=args.k, metric="euclidean")
    else:
        model = QKNNClassifier(n_neighbors=args.k)

    # 3. "训练"（KNN / QKNN 都是惰性学习，只是存数据 / 预计算）
    model.fit(X_train, y_train)

    # 4. 评估
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Test  accuracy: {test_acc:.4f}")

    # 5. 可视化（仅在二维特征时有效）
    if args.n_features == 2 and not args.no_plot:
        title = f"{args.model.upper()} Decision Boundary (k={args.k}) | Train acc={train_acc:.2f}, Test acc={test_acc:.2f}"
        plot_knn_decision_boundary(
            X_train, y_train, model,
            X_test=X_test, y_test=y_test,
            title=title,
            save_path=args.fig_path,
        )
    else:
        print("[Info] Visualization skipped (either n_features != 2 or --no-plot is set).")


if __name__ == "__main__":
    main()
