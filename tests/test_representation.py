import numpy as np
from number_geometry.representation import pca_dimension, subspace_overlap, svcca_similarity


def test_pca_dimension_is_valid():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(45, 20))
    k = pca_dimension(x, 0.95)
    assert 1 <= k <= 20


def test_self_overlap_retains_requested_variance():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(45, 30))
    score = subspace_overlap(x, x, variance_threshold=0.95)
    assert 0.94 <= score <= 1.001


def test_svcca_identical_representation_is_high():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(45, 20))
    score = svcca_similarity(x, x, variance_threshold=0.95, standardize=True)
    assert score > 0.99
