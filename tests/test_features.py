import cv2
import numpy as np
import pytest

from features import FEATURE_GROUPS, select_feature_groups
from features.features import (
    COLOR_DIMS,
    GEOMETRY_DIMS,
    HU_DIMS,
    extract_features,
    grabcut_init_mask,
)
from features.segmentation_eval import failure_type, mask_scores, split_samples


def test_grabcut_init_mask_layout():
    mask = grabcut_init_mask(100, 200)
    assert mask[0, 0] == cv2.GC_BGD
    assert mask[3, 100] == cv2.GC_PR_BGD
    assert mask[50, 100] == cv2.GC_FGD
    assert mask[50, 40] == cv2.GC_PR_FGD


def test_mask_scores_ignores_border_pixels():
    trimap = np.array([[1, 1, 2, 3]])
    mask = np.array([[255, 0, 255, 255]], dtype=np.uint8)
    iou, precision, recall = mask_scores(mask, trimap)
    assert iou == pytest.approx(1 / 3)
    assert precision == pytest.approx(0.5)
    assert recall == pytest.approx(0.5)


@pytest.mark.parametrize(
    "scores, expected",
    [
        ((0.8, 0.9, 0.9), None),
        ((0.3, 0.9, 0.3), "missing"),
        ((0.3, 0.3, 0.9), "background"),
        ((0.2, 0.4, 0.4), "both"),
    ],
)
def test_failure_type(scores, expected):
    assert failure_type(*scores) == expected


def test_split_samples_matches_stratified_split():
    samples = [(f"img{i}.jpg", "a" if i < 50 else "b") for i in range(100)]
    train = split_samples(samples, "train")
    test = split_samples(samples, "test")
    assert len(train) == 80 and len(test) == 20
    assert set(train).isdisjoint(test)
    assert split_samples(samples, "all") == samples


TOTAL_DIMS = COLOR_DIMS + GEOMETRY_DIMS + HU_DIMS


def test_feature_groups_cover_the_vector_without_overlap():
    covered = []
    for sl in FEATURE_GROUPS.values():
        covered.extend(range(sl.start, sl.stop))
    assert sorted(covered) == list(range(TOTAL_DIMS))


def test_extract_features_length_matches_groups():
    img = np.zeros((60, 80, 3), np.uint8)
    img[:, :, 2] = 200
    mask = np.zeros((60, 80), np.uint8)
    cv2.rectangle(mask, (20, 15), (60, 45), 255, thickness=-1)
    vector, _, _ = extract_features(img, mask)
    assert vector.shape == (TOTAL_DIMS,)
    assert np.all(np.isfinite(vector))


def test_achromatic_ratio_and_hue_ignore_gray_pixels():
    # Nửa ảnh xám (S=0), nửa đỏ bão hòa → tỷ lệ không màu ~0.5; hist H chỉ từ nửa đỏ.
    img = np.zeros((40, 40, 3), np.uint8)
    img[:, :20] = (128, 128, 128)  # BGR xám
    img[:, 20:] = (0, 0, 220)  # BGR đỏ thuần
    mask = np.full((40, 40), 255, dtype=np.uint8)
    vector, _, _ = extract_features(img, mask)
    achromatic_ratio = vector[6]
    hist_h = vector[7 : 7 + 16]
    assert achromatic_ratio == pytest.approx(0.5, abs=0.05)
    assert hist_h.sum() == pytest.approx(1.0)
    # Đỏ OpenCV H ≈ 0 → bin đầu chiếm ưu thế
    assert hist_h[0] == pytest.approx(1.0, abs=0.15)


def test_select_feature_groups_on_matrix_and_vector():
    X = np.arange(2 * TOTAL_DIMS).reshape(2, TOTAL_DIMS)
    used = COLOR_DIMS + GEOMETRY_DIMS
    selected = select_feature_groups(X, ["color", "geometry"])
    assert selected.shape == (2, used)
    np.testing.assert_array_equal(selected[0], np.arange(used))

    vector = select_feature_groups(X[1], ["hu"])
    np.testing.assert_array_equal(vector, X[1, used:TOTAL_DIMS])


def test_select_feature_groups_rejects_unknown_group():
    with pytest.raises(ValueError):
        select_feature_groups(np.zeros(TOTAL_DIMS), ["shape"])
