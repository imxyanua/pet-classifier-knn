# code by xyanua - github.com/imxyanua
"""Khảo sát k / khoảng cách, in accuracy và confusion matrix."""
from config import K_CHOICES, SELECTED_BREEDS
from dataset import load_or_extract
from knn import (
    accuracy_score,
    confusion_matrix,
    knn_predict_batch,
    standardize_apply,
    standardize_fit,
    stratified_split,
)

METRICS = ["euclidean", "manhattan"]


def main():
    print("Đang nạp / trích đặc trưng…")
    X, y, _paths = load_or_extract(
        progress=lambda i, n, p: print(f"\r{i}/{n}", end="", flush=True)
    )
    print(f"\n{len(y)} mẫu.")
    train_idx, test_idx = stratified_split(y, test_ratio=0.2, seed=42)
    mean, std = standardize_fit(X[train_idx])
    X_train = standardize_apply(X[train_idx], mean, std)
    X_test = standardize_apply(X[test_idx], mean, std)
    y_train, y_test = y[train_idx], y[test_idx]

    print(f"Train {len(train_idx)} · Test {len(test_idx)}\n")
    print(f"{'metric':<12} {'k':>4}  accuracy")
    best = (None, None, -1.0)
    for metric in METRICS:
        for k in K_CHOICES:
            preds = knn_predict_batch(X_test, X_train, y_train, k=k, metric=metric)
            acc = accuracy_score(y_test, preds)
            print(f"{metric:<12} {k:>4}  {acc * 100:6.2f}%")
            if acc > best[2]:
                best = (metric, k, acc)

    metric, k, acc = best
    print(f"\nTốt nhất: {metric}, k={k}, accuracy={acc * 100:.2f}%")
    preds = knn_predict_batch(X_test, X_train, y_train, k=k, metric=metric)
    cm = confusion_matrix(y_test, preds, SELECTED_BREEDS)
    header = " " * 16 + " ".join(f"{c[:10]:>10}" for c in SELECTED_BREEDS)
    print("\nConfusion matrix (hàng = thực tế, cột = dự đoán)")
    print(header)
    for i, cls in enumerate(SELECTED_BREEDS):
        row = " ".join(f"{n:10d}" for n in cm[i])
        print(f"{cls[:16]:<16}{row}")


if __name__ == "__main__":
    main()
