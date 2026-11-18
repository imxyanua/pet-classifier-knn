# code by xyanua - github.com/imxyanua
"""Đánh giá k-NN: chọn (khoảng cách, k) bằng cross-validation trên train, báo cáo trên test."""
import argparse
from collections import Counter

import numpy as np

from config import DISPLAY_NAMES, K_CHOICES, SELECTED_BREEDS
from dataset import load_or_extract
from features import FEATURE_GROUPS
from knn import (
    accuracy_score,
    confusion_matrix,
    knn_predict_batch,
    per_class_metrics,
    select_params,
    standardize_apply,
    standardize_fit,
    stratified_split,
)

METRICS = ["euclidean", "manhattan"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42, help="seed chia train/test cho báo cáo chi tiết")
    parser.add_argument("--folds", type=int, default=5, help="số fold cross-validation trên tập train")
    parser.add_argument(
        "--repeats",
        type=int,
        default=10,
        help="lặp toàn bộ quy trình với seed 0..N-1 để ước lượng độ dao động (0 để bỏ qua)",
    )
    parser.add_argument(
        "--features",
        default=",".join(FEATURE_GROUPS),
        help=f"nhóm đặc trưng, phân tách bằng dấu phẩy: {', '.join(FEATURE_GROUPS)}",
    )
    return parser.parse_args(argv)


def select_columns(X, groups):
    unknown = [g for g in groups if g not in FEATURE_GROUPS]
    if unknown:
        raise SystemExit(f"Nhóm đặc trưng không hợp lệ: {', '.join(unknown)}")
    return np.hstack([X[:, FEATURE_GROUPS[g]] for g in groups])


def run_once(X, y, seed, n_folds):
    """Chia train/test, chọn tham số bằng CV trên train, đánh giá đúng một lần trên test."""
    train_idx, test_idx = stratified_split(y, test_ratio=0.2, seed=seed)
    (metric, k, cv_acc), grid = select_params(
        X[train_idx], y[train_idx], K_CHOICES, METRICS, n_folds=n_folds, seed=seed
    )
    mean, std = standardize_fit(X[train_idx])
    X_train = standardize_apply(X[train_idx], mean, std)
    X_test = standardize_apply(X[test_idx], mean, std)
    preds = knn_predict_batch(X_test, X_train, y[train_idx], k=k, metric=metric)
    return {
        "metric": metric,
        "k": k,
        "cv_acc": cv_acc,
        "grid": grid,
        "test_acc": accuracy_score(y[test_idx], preds),
        "cm": confusion_matrix(y[test_idx], preds, SELECTED_BREEDS),
        "n_train": len(train_idx),
        "n_test": len(test_idx),
    }


def print_detailed(result, seed, n_folds):
    print(f"\n=== Chi tiết với seed={seed} ===")
    print(f"Train {result['n_train']} · Test {result['n_test']}")
    print(f"\nAccuracy cross-validation {n_folds} fold trên tập train:")
    print(f"{'k':>4}  " + "  ".join(f"{m:>10}" for m in METRICS))
    for k in K_CHOICES:
        row = "  ".join(f"{result['grid'][(m, k)] * 100:9.2f}%" for m in METRICS)
        print(f"{k:>4}  {row}")

    print(
        f"\nChọn theo CV: {result['metric']}, k={result['k']} "
        f"(CV {result['cv_acc'] * 100:.2f}%)"
    )
    print(f"Accuracy trên test: {result['test_acc'] * 100:.2f}%")

    cm = result["cm"]
    print("\nConfusion matrix (hàng = thực tế, cột = dự đoán)")
    print(" " * 16 + " ".join(f"{c[:10]:>10}" for c in SELECTED_BREEDS))
    for i, cls in enumerate(SELECTED_BREEDS):
        print(f"{cls[:16]:<16}" + " ".join(f"{n:10d}" for n in cm[i]))

    precision, recall, f1 = per_class_metrics(cm)
    print(f"\n{'lớp':<16}{'precision':>10}{'recall':>10}{'F1':>10}")
    for i, cls in enumerate(SELECTED_BREEDS):
        print(
            f"{DISPLAY_NAMES[cls]:<16}{precision[i] * 100:9.1f}%"
            f"{recall[i] * 100:9.1f}%{f1[i] * 100:9.1f}%"
        )
    print(f"{'trung bình':<16}{precision.mean() * 100:9.1f}%{recall.mean() * 100:9.1f}%{f1.mean() * 100:9.1f}%")


def print_repeated(X, y, repeats, n_folds):
    print(f"\n=== Lặp {repeats} lần (seed 0..{repeats - 1}) ===")
    accs = []
    chosen = Counter()
    for seed in range(repeats):
        result = run_once(X, y, seed, n_folds)
        accs.append(result["test_acc"])
        chosen[(result["metric"], result["k"])] += 1
        print(
            f"seed {seed:>2}: chọn {result['metric']:<9} k={result['k']:<2}  "
            f"CV {result['cv_acc'] * 100:5.1f}%  test {result['test_acc'] * 100:5.1f}%"
        )
    accs = np.array(accs)
    print(
        f"\nAccuracy trên test: {accs.mean() * 100:.1f}% ± {accs.std() * 100:.1f}% "
        f"(thấp nhất {accs.min() * 100:.1f}%, cao nhất {accs.max() * 100:.1f}%)"
    )
    print("Cấu hình được chọn: " + ", ".join(f"{m} k={k} ×{n}" for (m, k), n in chosen.most_common()))


def main(argv=None):
    args = parse_args(argv)
    groups = [g.strip() for g in args.features.split(",") if g.strip()]

    print("Đang nạp / trích đặc trưng…")
    X, y, _paths = load_or_extract(
        progress=lambda i, n, p: print(f"\r{i}/{n}", end="", flush=True)
    )
    X = select_columns(X, groups)
    print(f"{len(y)} mẫu · {X.shape[1]} chiều ({', '.join(groups)})")

    print_detailed(run_once(X, y, args.seed, args.folds), args.seed, args.folds)
    if args.repeats > 0:
        print_repeated(X, y, args.repeats, args.folds)


if __name__ == "__main__":
    main()
