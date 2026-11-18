# code by xyanua - github.com/imxyanua
"""Chọn tham số k-NN bằng cross-validation trên tập train."""
import numpy as np

from .knn import (
    accuracy_score,
    knn_predict_batch,
    standardize_apply,
    standardize_fit,
)


def stratified_k_fold(y, n_folds=5, seed=0):
    """Trả về danh sách (train_idx, val_idx), mỗi fold giữ tỷ lệ lớp."""
    y = np.asarray(y)
    rng = np.random.RandomState(seed)
    fold_of = np.empty(len(y), dtype=int)
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        fold_of[idx] = np.arange(len(idx)) % n_folds
    folds = []
    for f in range(n_folds):
        val_idx = np.where(fold_of == f)[0]
        train_idx = np.where(fold_of != f)[0]
        folds.append((train_idx, val_idx))
    return folds


def cross_val_accuracy(X, y, k, metric, folds):
    """Accuracy trung bình qua các fold; chuẩn hóa được fit riêng trên phần train của từng fold."""
    scores = []
    for train_idx, val_idx in folds:
        mean, std = standardize_fit(X[train_idx])
        X_tr = standardize_apply(X[train_idx], mean, std)
        X_val = standardize_apply(X[val_idx], mean, std)
        preds = knn_predict_batch(X_val, X_tr, y[train_idx], k=k, metric=metric)
        scores.append(accuracy_score(y[val_idx], preds))
    return float(np.mean(scores))


def select_params(X, y, k_choices, metrics, n_folds=5, seed=0):
    """Chọn (metric, k) có accuracy cross-validation cao nhất. Hòa thì ưu tiên k nhỏ hơn."""
    folds = stratified_k_fold(y, n_folds=n_folds, seed=seed)
    grid = {}
    best = None
    for metric in metrics:
        for k in k_choices:
            score = cross_val_accuracy(X, y, k, metric, folds)
            grid[(metric, k)] = score
            if best is None or score > best[2]:
                best = (metric, k, score)
    return best, grid
