import cv2
import numpy as np
import pytest

from features import FEATURE_GROUPS, select_feature_groups
from features.features import grabcut_init_mask
from features.segmentation_eval import mask_scores


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


def test_feature_groups_cover_the_34_dim_vector_without_overlap():
    covered = []
    for sl in FEATURE_GROUPS.values():
        covered.extend(range(sl.start, sl.stop))
    assert sorted(covered) == list(range(34))


def test_select_feature_groups_on_matrix_and_vector():
    X = np.arange(2 * 34).reshape(2, 34)
    selected = select_feature_groups(X, ["color", "geometry"])
    assert selected.shape == (2, 27)
    np.testing.assert_array_equal(selected[0], np.arange(27))

    vector = select_feature_groups(X[1], ["hu"])
    np.testing.assert_array_equal(vector, X[1, 27:34])


def test_select_feature_groups_rejects_unknown_group():
    with pytest.raises(ValueError):
        select_feature_groups(np.zeros(34), ["shape"])
