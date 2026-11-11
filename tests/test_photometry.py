"""G_RVS from G and G_RP (Sartoretti et al. 2023, eqs 9-10)."""

import numpy as np
import pytest

from snapdragons.photometry import grvs_from_g_rp, GRVS_COLOUR_MIN, GRVS_COLOUR_MAX


def test_reference_values():
    # G_RVS - G_RP evaluated by hand from the published coefficients
    # blue branch, G - G_RP = 0.5
    assert grvs_from_g_rp(12.5, 12.0) - 12.0 == pytest.approx(-0.2013875, abs=1e-9)
    # red branch, G - G_RP = 1.5
    assert grvs_from_g_rp(13.5, 12.0) - 12.0 == pytest.approx(-0.5984125, abs=1e-9)


def test_depends_on_colour_only_up_to_an_offset():
    g = np.array([8.0, 12.0, 16.0])
    rp = g - 0.8
    out = grvs_from_g_rp(g, rp)
    assert np.allclose(out - rp, (out - rp)[0])
    # absolute or apparent: adding a distance modulus to both shifts G_RVS by it
    assert np.allclose(grvs_from_g_rp(g + 10.0, rp + 10.0), out + 10.0)


def test_nan_outside_the_calibrated_colour_range():
    assert np.isnan(grvs_from_g_rp(12.0, 12.0 - (GRVS_COLOUR_MAX + 0.05)))
    assert np.isnan(grvs_from_g_rp(12.0, 12.0 - (GRVS_COLOUR_MIN - 0.05)))
    assert np.isfinite(grvs_from_g_rp(12.0, 12.0 - (GRVS_COLOUR_MAX - 0.01)))
    assert np.isfinite(grvs_from_g_rp(12.0, 12.0 - (GRVS_COLOUR_MIN + 0.01)))


def test_nan_inputs_give_nan():
    out = grvs_from_g_rp(np.array([np.nan, 12.0, 12.0]),
                         np.array([11.0, np.nan, 11.2]))
    assert np.isnan(out[0]) and np.isnan(out[1]) and np.isfinite(out[2])


def test_branches_join_within_the_quoted_scatter():
    eps = 1e-6
    blue = grvs_from_g_rp(12.0, 12.0 - (1.2 - eps))
    red = grvs_from_g_rp(12.0, 12.0 - (1.2 + eps))
    assert abs(float(blue) - float(red)) < 0.02


def test_grvs_is_brighter_than_g_for_normal_stars():
    x = np.linspace(0.0, 1.6, 50)
    grvs = grvs_from_g_rp(12.0 + 0 * x, 12.0 - x)
    assert np.all(grvs < 12.0)
    # and G - G_RVS grows with colour
    assert np.all(np.diff(12.0 - grvs) > 0)


def test_shapes_and_scalars():
    assert np.ndim(grvs_from_g_rp(12.0, 11.2)) == 0
    assert grvs_from_g_rp(np.full((3, 4), 12.0), np.full((3, 4), 11.2)).shape == (3, 4)
    assert grvs_from_g_rp([12.0, 13.0], [11.2, 12.2]).shape == (2,)


def test_agrees_with_pygaia():
    transformations = pytest.importorskip("pygaia.photometry.transformations")
    x = np.linspace(-0.1, 1.65, 60)
    rp = np.full_like(x, 11.0)
    ours = grvs_from_g_rp(rp + x, rp)
    theirs = transformations.grvs_from_gmingrp(x) + rp
    assert np.allclose(ours, theirs, atol=1e-6)
