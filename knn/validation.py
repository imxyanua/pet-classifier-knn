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


def prepare_features(X_train, X_other, metric, hist_mask):
    """χ² giữ histogram thô; các metric khác z-score toàn bộ."""
    skip = hist_mask if metric == "chi_square" else None
    mean, std = standardize_fit(X_train, skip_mask=skip)
    return (
        standardize_apply(X_train, mean, std),
        standardize_apply(X_other, mean, std),
        mean,
        std,
    )


def cross_val_accuracy(X, y, k, metric, folds, vote="uniform", hist_mask=None):
    """Accuracy trung bình qua các fold; chuẩn hóa được fit riêng trên phần train của từng fold."""
    scores = []
    for train_idx, val_idx in folds:
        X_tr, X_val, _, _ = prepare_features(X[train_idx], X[val_idx], metric, hist_mask)
        preds = knn_predict_batch(
            X_val,
            X_tr,
            y[train_idx],
            k=k,
            metric=metric,
            vote=vote,
            hist_mask=hist_mask,
        )
        scores.append(accuracy_score(y[val_idx], preds))
    return float(np.mean(scores))


def select_params(
    X,
    y,
    k_choices,
    metrics,
    n_folds=5,
    seed=0,
    votes=("uniform",),
    hist_mask=None,
):
    """Chọn (metric, vote, k) có accuracy CV cao nhất. Hòa thì ưu tiên k nhỏ hơn, rồi metric đứng trước trong danh sách."""
    folds = stratified_k_fold(y, n_folds=n_folds, seed=seed)
    grid = {}
    best = None
    for metric in metrics:
        for vote in votes:
            for k in k_choices:
                score = cross_val_accuracy(
                    X, y, k, metric, folds, vote=vote, hist_mask=hist_mask
                )
                grid[(metric, vote, k)] = score
                if best is None or score > best[3]:
                    best = (metric, vote, k, score)
    return best, grid
