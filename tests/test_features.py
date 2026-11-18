import numpy as np
import pytest

from features import FEATURE_GROUPS, select_feature_groups


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
