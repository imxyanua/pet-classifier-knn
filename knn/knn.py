# code by xyanua - github.com/imxyanua
"""k-NN, chuẩn hóa, chia tập và đánh giá — tự cài, không dùng sklearn."""
import numpy as np


def euclidean_distances(X_train, x):
    return np.sqrt(np.sum((X_train - x) ** 2, axis=1))


def manhattan_distances(X_train, x):
    return np.sum(np.abs(X_train - x), axis=1)


DISTANCES = {
    "euclidean": euclidean_distances,
    "manhattan": manhattan_distances,
}


def standardize_fit(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean, std


def standardize_apply(X, mean, std):
    return (X - mean) / std


def stratified_split(y, test_ratio=0.2, seed=42):
    rng = np.random.RandomState(seed)
    train_idx = []
    test_idx = []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n_test = max(1, int(round(len(idx) * test_ratio)))
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])
    return np.array(train_idx, dtype=int), np.array(test_idx, dtype=int)


def _majority_vote(neighbor_labels, neighbor_dists):
    best_label = neighbor_labels[0]
    best_count = -1
    best_dist = np.inf
    for lab in np.unique(neighbor_labels):
        mask = neighbor_labels == lab
        count = int(np.sum(mask))
        mean_d = float(np.mean(neighbor_dists[mask]))
        if count > best_count or (count == best_count and mean_d < best_dist):
            best_label = lab
            best_count = count
            best_dist = mean_d
    return best_label


def knn_predict_one(x, X_train, y_train, k=5, metric="euclidean"):
    if k < 1:
        raise ValueError("k phải >= 1")
    if metric not in DISTANCES:
        raise ValueError(f"Khoảng cách không hỗ trợ: {metric}")
    k = min(k, len(X_train))
    dists = DISTANCES[metric](X_train, x)
    nn = np.argpartition(dists, k - 1)[:k]
    order = np.argsort(dists[nn])
    nn = nn[order]
    labels = y_train[nn]
    pred = _majority_vote(labels, dists[nn])
    return pred, dists[nn], labels


def knn_predict_batch(X, X_train, y_train, k=5, metric="euclidean"):
    preds = []
    for x in X:
        pred, _, _ = knn_predict_one(x, X_train, y_train, k=k, metric=metric)
        preds.append(pred)
    return np.array(preds)


def accuracy_score(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true, y_pred, classes):
    index = {cls: i for i, cls in enumerate(classes)}
    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in index and p in index:
            matrix[index[t], index[p]] += 1
    return matrix
