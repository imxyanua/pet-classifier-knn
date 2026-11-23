import numpy as np
import pytest

from knn import (
    accuracy_score,
    chi_square_distances,
    confusion_matrix,
    cosine_distances,
    knn_predict_one,
    per_class_metrics,
    select_params,
    standardize_apply,
    standardize_fit,
    stratified_k_fold,
    stratified_split,
)


def two_clusters(n_per_class=30, seed=0):
    rng = np.random.RandomState(seed)
    X = np.vstack([rng.normal(0, 0.3, (n_per_class, 3)), rng.normal(5, 0.3, (n_per_class, 3))])
    y = np.array(["a"] * n_per_class + ["b"] * n_per_class)
    return X, y


@pytest.mark.parametrize("metric", ["euclidean", "manhattan"])
@pytest.mark.parametrize("vote", ["uniform", "distance"])
def test_separated_clusters_are_classified_perfectly(metric, vote):
    X, y = two_clusters()
    for x, label in zip(X, y):
        pred, _, _ = knn_predict_one(x, X, y, k=3, metric=metric, vote=vote)
        assert pred == label


@pytest.mark.parametrize("vote", ["uniform", "distance"])
def test_cosine_classifies_separated_positive_clusters(vote):
    # Cosine phụ thuộc hướng từ gốc; hai cụm cách xa gốc theo hướng khác nhau.
    rng = np.random.RandomState(0)
    X = np.vstack(
        [
            rng.normal([4.0, 0.0, 0.0], 0.2, (25, 3)),
            rng.normal([0.0, 4.0, 0.0], 0.2, (25, 3)),
        ]
    )
    y = np.array(["a"] * 25 + ["b"] * 25)
    for x, label in zip(X, y):
        pred, _, _ = knn_predict_one(x, X, y, k=3, metric="cosine", vote=vote)
        assert pred == label


def test_neighbors_are_sorted_by_distance():
    X_train = np.array([[3.0], [1.0], [2.0], [10.0]])
    y_train = np.array(["c", "a", "b", "d"])
    _, dists, labels = knn_predict_one(np.array([0.0]), X_train, y_train, k=3)
    assert list(labels) == ["a", "b", "c"]
    assert np.all(np.diff(dists) >= 0)


def test_tie_is_broken_by_mean_distance():
    X_train = np.array([[0.8], [1.0], [1.5], [1.1]])
    y_train = np.array(["pug", "bombay", "pug", "bombay"])
    pred, _, _ = knn_predict_one(np.array([0.0]), X_train, y_train, k=4)
    assert pred == "bombay"


def test_distance_vote_prefers_closer_neighbor_class():
    # 2 pug xa, 1 bombay rất gần → uniform chọn pug, distance chọn bombay
    X_train = np.array([[0.1], [2.0], [2.1]])
    y_train = np.array(["bombay", "pug", "pug"])
    uniform, _, _ = knn_predict_one(np.array([0.0]), X_train, y_train, k=3, vote="uniform")
    weighted, _, _ = knn_predict_one(np.array([0.0]), X_train, y_train, k=3, vote="distance")
    assert uniform == "pug"
    assert weighted == "bombay"


def test_k_larger_than_train_set_is_clamped():
    X, y = two_clusters(n_per_class=2)
    pred, dists, _ = knn_predict_one(X[0], X, y, k=50)
    assert len(dists) == len(X)
    assert pred in ("a", "b")


def test_invalid_arguments_raise():
    X, y = two_clusters(n_per_class=2)
    with pytest.raises(ValueError):
        knn_predict_one(X[0], X, y, k=0)
    with pytest.raises(ValueError):
        knn_predict_one(X[0], X, y, k=1, metric="hamming")
    with pytest.raises(ValueError):
        knn_predict_one(X[0], X, y, k=1, vote="soft")


def test_cosine_and_chi_square_distances_basic():
    X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    x = np.array([1.0, 0.0])
    cos = cosine_distances(X, x)
    assert cos[0] == pytest.approx(0.0)
    assert cos[1] == pytest.approx(1.0)
    chi = chi_square_distances(np.array([[0.5, 0.5], [1.0, 0.0]]), np.array([0.5, 0.5]))
    assert chi[0] == pytest.approx(0.0)
    assert chi[1] > 0


def test_chi_square_hybrid_uses_histogram_mask():
    # hist = cột 1; rest = cột 0. Cùng hist → χ²=0, khác rest → L1 quyết định
    X_train = np.array([[0.0, 0.5], [3.0, 0.5]])
    y_train = np.array(["near", "far"])
    hist_mask = np.array([False, True])
    pred, dists, _ = knn_predict_one(
        np.array([0.1, 0.5]),
        X_train,
        y_train,
        k=1,
        metric="chi_square",
        hist_mask=hist_mask,
    )
    assert pred == "near"
    assert dists[0] == pytest.approx(0.1)


def test_standardize_uses_train_statistics_and_handles_constant_column():
    X = np.array([[1.0, 5.0], [3.0, 5.0]])
    mean, std = standardize_fit(X)
    Z = standardize_apply(X, mean, std)
    np.testing.assert_allclose(Z[:, 0], [-1.0, 1.0])
    np.testing.assert_allclose(Z[:, 1], [0.0, 0.0])


def test_standardize_skip_mask_keeps_columns_raw():
    X = np.array([[1.0, 0.25], [3.0, 0.75]])
    mean, std = standardize_fit(X, skip_mask=np.array([False, True]))
    Z = standardize_apply(X, mean, std)
    np.testing.assert_allclose(Z[:, 0], [-1.0, 1.0])
    np.testing.assert_allclose(Z[:, 1], [0.25, 0.75])


def test_stratified_split_keeps_class_ratio_and_is_disjoint():
    y = np.array(["a"] * 50 + ["b"] * 50)
    train_idx, test_idx = stratified_split(y, test_ratio=0.2, seed=1)
    assert set(train_idx).isdisjoint(test_idx)
    assert len(train_idx) + len(test_idx) == len(y)
    assert np.sum(y[test_idx] == "a") == 10
    assert np.sum(y[test_idx] == "b") == 10


def test_stratified_k_fold_covers_every_sample_once():
    y = np.array(["a"] * 20 + ["b"] * 15)
    folds = stratified_k_fold(y, n_folds=5, seed=3)
    seen = np.concatenate([val for _, val in folds])
    assert sorted(seen) == list(range(len(y)))
    for train_idx, val_idx in folds:
        assert set(train_idx).isdisjoint(val_idx)
        assert np.sum(y[val_idx] == "a") == 4
        assert np.sum(y[val_idx] == "b") == 3


def test_select_params_finds_perfect_config_on_separated_data():
    X, y = two_clusters()
    (metric, vote, k, score), grid = select_params(
        X, y, [1, 3], ["euclidean", "manhattan"], n_folds=3, votes=("uniform", "distance")
    )
    assert score == 1.0
    assert len(grid) == 8
    assert vote in ("uniform", "distance")
    assert metric == "euclidean"
    assert k == 1


def test_accuracy_and_confusion_matrix():
    y_true = ["a", "a", "b", "b", "c"]
    y_pred = ["a", "b", "b", "b", "a"]
    assert accuracy_score(y_true, y_pred) == pytest.approx(0.6)
    cm = confusion_matrix(y_true, y_pred, ["a", "b", "c"])
    np.testing.assert_array_equal(cm, [[1, 1, 0], [0, 2, 0], [1, 0, 0]])


def test_per_class_metrics_from_confusion_matrix():
    cm = np.array([[1, 1, 0], [0, 2, 0], [1, 0, 0]])
    precision, recall, f1 = per_class_metrics(cm)
    np.testing.assert_allclose(precision, [0.5, 2 / 3, 0.0])
    np.testing.assert_allclose(recall, [0.5, 1.0, 0.0])
    np.testing.assert_allclose(f1, [0.5, 0.8, 0.0])
