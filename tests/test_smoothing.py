"""Phase-space smoothing: kNN lengths and Epanechnikov kernel draws."""

import numpy as np
import pytest
from scipy import stats

from snapdragons.smoothing import (
    estimate_smoothing_lengths, knn_smoothing_length, epanechnikov_offsets,
    sample_epanechnikov_3d,
)


def test_knn_length_on_a_lattice():
    # unit cubic lattice: the 6 nearest neighbours of an interior point are at 1
    g = np.arange(8.0)
    pts = np.array(np.meshgrid(g, g, g, indexing="ij")).reshape(3, -1).T
    h = knn_smoothing_length(pts, 6)
    interior = np.all((pts > 0) & (pts < 7), axis=1)
    assert np.allclose(h[interior], 1.0 / 5 ** (1 / 3))


def test_knn_length_recovers_a_uniform_density(rng):
    n, k = 40_000, 64
    pts = rng.uniform(0, 1, (n, 3))
    h = knn_smoothing_length(pts, k)
    interior = np.all((pts > 0.15) & (pts < 0.85), axis=1)
    # h = r_k / (k-1)^(1/3) with (4/3) pi r_k^3 n ~ k
    expected = (3 * k / (4 * np.pi * n)) ** (1 / 3) / (k - 1) ** (1 / 3)
    assert np.median(h[interior]) == pytest.approx(expected, rel=0.03)


def test_knn_length_scales_with_the_coordinates(rng):
    pts = rng.normal(size=(2000, 3))
    assert np.allclose(knn_smoothing_length(3.0 * pts, 16), 3.0 * knn_smoothing_length(pts, 16))


def test_knn_length_grows_with_k_and_in_sparse_regions(rng):
    pts = rng.normal(size=(5000, 3))
    h16, h64 = knn_smoothing_length(pts, 16), knn_smoothing_length(pts, 64)
    r = np.linalg.norm(pts, axis=1)
    assert np.median(h64[r > 2.5]) > 2 * np.median(h64[r < 0.5])
    # r_k / (k-1)^(1/3) is a spacing estimate: roughly independent of k
    assert np.median(h64) == pytest.approx(np.median(h16), rel=0.15)
    assert np.all(h16 > 0)


def test_knn_single_thread_matches_all_threads(rng):
    pts = rng.normal(size=(1000, 3))
    assert np.array_equal(knn_smoothing_length(pts, 8, workers=1),
                          knn_smoothing_length(pts, 8, workers=-1))


def test_estimate_smoothing_lengths_is_separate_in_position_and_velocity(rng):
    pos = rng.normal(size=(3000, 3))
    vel = 50.0 * rng.normal(size=(3000, 3))
    h_pos, h_vel = estimate_smoothing_lengths(pos, vel, n_ngb=32)
    assert h_pos.shape == h_vel.shape == (3000,)
    assert np.allclose(h_pos, knn_smoothing_length(pos, 32))
    assert np.allclose(h_vel, knn_smoothing_length(vel, 32))
    h_pos2, h_vel2 = estimate_smoothing_lengths(pos, vel, n_ngb=32, scale=2.5)
    assert np.allclose(h_pos2, 2.5 * h_pos) and np.allclose(h_vel2, 2.5 * h_vel)


def test_epanechnikov_sample_has_compact_support_and_the_right_profile(rng):
    x = sample_epanechnikov_3d(200_000, rng)
    assert x.shape == (200_000, 3)
    r = np.linalg.norm(x, axis=1)
    assert r.max() <= 1.0
    # density ~ 1 - r^2 in the unit ball: radial CDF r^3 (5 - 3 r^2) / 2
    assert stats.kstest(r, lambda q: q**3 * (5 - 3 * q**2) / 2).pvalue > 1e-3
    assert np.mean(r**2) == pytest.approx(3 / 7, rel=0.01)


def test_epanechnikov_sample_is_isotropic(rng):
    x = sample_epanechnikov_3d(200_000, rng)
    assert np.allclose(x.mean(axis=0), 0.0, atol=0.005)
    assert np.allclose(x.var(axis=0), 1 / 7, rtol=0.02)       # <x^2> = <r^2> / 3
    assert np.allclose(np.corrcoef(x.T), np.eye(3), atol=0.01)


def test_epanechnikov_sample_small_and_empty(rng):
    assert sample_epanechnikov_3d(1, rng).shape == (1, 3)
    assert sample_epanechnikov_3d(0, rng).shape == (0, 3)


def test_offsets_are_scaled_per_parent_in_parent_order(rng):
    h = np.array([0.1, 5.0, 0.0, 1.0])
    n_stars = np.array([1000, 2000, 10, 0])
    off = epanechnikov_offsets(h, n_stars, rng)
    assert off.shape == (3010, 3)
    r = np.linalg.norm(off, axis=1)
    assert r[:1000].max() <= 0.1 and r[:1000].max() > 0.08
    assert r[1000:3000].max() <= 5.0 and r[1000:3000].max() > 4.0
    assert np.all(r[3000:] == 0.0)


def test_offsets_with_no_stars(rng):
    assert epanechnikov_offsets(np.array([1.0, 2.0]), np.array([0, 0]), rng).shape == (0, 3)


def test_offsets_are_reproducible():
    h, n = np.array([1.0, 2.0]), np.array([5, 7])
    a = epanechnikov_offsets(h, n, np.random.default_rng(3))
    b = epanechnikov_offsets(h, n, np.random.default_rng(3))
    assert np.array_equal(a, b)
