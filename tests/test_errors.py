"""Gaia error model: PyGaia wrapper, error application, fallbacks."""

import numpy as np
import pytest

from snapdragons import Config
from snapdragons.errors import GaiaErrorModel, GAIA_MAG_BRIGHT, GAIA_MAG_FAINT
from snapdragons.photometry import grvs_from_g_rp

pygaia = pytest.importorskip("pygaia")


def make_catalogue(n, rng, g_lo=8.0, g_hi=13.5):
    g = rng.uniform(g_lo, g_hi, n)
    rp = g - 0.6
    true = {
        "ra": rng.uniform(0, 360, n), "dec": rng.uniform(-60, 60, n),
        "l": rng.uniform(0, 360, n), "b": rng.uniform(-60, 60, n),
        "parallax": rng.uniform(0.2, 5.0, n),
        "pmra": rng.normal(0, 5, n), "pmdec": rng.normal(0, 5, n),
        "rv": rng.normal(0, 40, n),
        "Gmag_app": g, "G_BPmag_app": g + 0.4, "G_RPmag_app": rp,
        "G_RVSmag_app": grvs_from_g_rp(g, rp),
    }
    stellar = {"logTe": np.full(n, np.log10(5000.0)), "logg": np.full(n, 4.3),
               "Mini": rng.uniform(0.5, 2.0, n)}
    return {"true": true, "stellar": stellar}


def model(release="dr3"):
    return GaiaErrorModel(Config(gaia_release=release))


# --- uncertainty curves --------------------------------------------------------

def test_units_and_magnitudes_of_the_astrometric_uncertainties():
    m = model("dr3")
    g = np.array([12.0, 15.0, 18.0, 20.0])
    plx = m.parallax_uncertainty(g)
    # DR3: ~0.02-0.03 mas at G = 15, ~0.5 mas at G = 20
    assert 0.01 < plx[1] < 0.05 and 0.3 < plx[3] < 1.0
    assert np.all(np.diff(plx) > 0)
    pmra, pmdec = m.proper_motion_uncertainty(g)
    assert 0.01 < pmra[1] < 0.06 and 0.005 < pmdec[1] < 0.06


def test_photometric_uncertainties_are_in_mag():
    """Regression: PyGaia returns mmag; these were once used as mag, making
    the photometric errors 1000 times too large."""
    from pygaia.errors.photometric import magnitude_uncertainty
    g = np.array([12.0, 15.0, 18.0])
    g_err, bp_err, rp_err = model("dr3").photometric_uncertainty(g)
    # Gaia DR3: G to ~0.3 mmag at G = 15, ~1 mmag at G = 18
    assert np.all(g_err < 0.01)
    assert g_err[1] == pytest.approx(3e-4, rel=0.5)
    assert np.allclose(g_err, magnitude_uncertainty("g", g, release="dr3") * 1e-3)
    assert np.all(np.diff(g_err) > 0)
    assert np.all(bp_err > g_err) and np.all(rp_err > g_err)


def test_bp_and_rp_uncertainties_use_their_own_magnitudes():
    from pygaia.errors.photometric import magnitude_uncertainty
    m = model("dr3")
    g = np.array([14.0, 16.0, 18.0])
    bp, rp = g + 1.2, g - 0.9
    g_err, bp_err, rp_err = m.photometric_uncertainty(g, bp, rp)
    assert np.allclose(bp_err, magnitude_uncertainty("bp", bp, release="dr3") * 1e-3)
    assert np.allclose(rp_err, magnitude_uncertainty("rp", rp, release="dr3") * 1e-3)
    # a red star: fainter in BP, so a larger BP error than at its G magnitude
    _, bp_at_g, rp_at_g = m.photometric_uncertainty(g)
    assert np.all(bp_err > bp_at_g) and np.all(rp_err < rp_at_g)


def test_band_magnitudes_outside_pygaias_range_are_clipped_not_fatal():
    m = model("dr3")
    g = np.array([20.5, 5.0, 15.0])
    bp = np.array([24.0, 6.0, np.nan])        # beyond the table; missing
    rp = np.array([19.0, 3.2, 14.0])
    g_err, bp_err, rp_err = m.photometric_uncertainty(g, bp, rp)
    assert np.all(np.isfinite(bp_err)) and np.all(np.isfinite(rp_err))
    at_edge = m.photometric_uncertainty(np.array([21.0]), np.array([21.0]), np.array([4.0]))
    assert bp_err[0] == pytest.approx(at_edge[1][0])
    assert rp_err[1] == pytest.approx(at_edge[2][0])
    assert bp_err[2] == pytest.approx(m.photometric_uncertainty(np.array([15.0]))[1][0])


def test_observed_photometry_is_close_to_the_truth():
    cat = make_catalogue(5000, np.random.default_rng(0))
    out = model().apply_errors(cat, rng=np.random.default_rng(1))
    for obs, true in (("Gmag", "Gmag_app"), ("G_BPmag", "G_BPmag_app"), ("G_RPmag", "G_RPmag_app")):
        assert np.abs(out["observed"][obs] - out["true"][true]).max() < 0.05
    # and the catalogue's BP/RP uncertainties are those at the BP/RP magnitudes
    m = model()
    _, bp_err, rp_err = m.photometric_uncertainty(
        out["true"]["Gmag_app"], out["true"]["G_BPmag_app"], out["true"]["G_RPmag_app"])
    assert np.allclose(out["uncertainties"]["G_BPmag"], bp_err)
    assert np.allclose(out["uncertainties"]["G_RPmag"], rp_err)


def test_uncertainties_are_in_mas_not_microarcsec():
    from pygaia.errors.astrometric import parallax_uncertainty
    g = np.array([10.0, 14.0, 17.0])
    assert np.allclose(model("dr3").parallax_uncertainty(g),
                       parallax_uncertainty(g, release="dr3") / 1000.0)


def test_later_releases_are_more_precise():
    g = np.array([13.0, 16.0, 19.0])
    plx = {r: model(r).parallax_uncertainty(g) for r in ("dr2", "dr3", "dr4", "dr5")}
    assert np.all(plx["dr2"] > plx["dr3"])
    assert np.all(plx["dr3"] > plx["dr4"])
    assert np.all(plx["dr4"] > plx["dr5"])
    pm = {r: model(r).proper_motion_uncertainty(g)[0] for r in ("dr3", "dr4")}
    assert np.all(pm["dr3"] > 2 * pm["dr4"])       # proper motions gain faster than parallaxes


def test_dr2_is_scaled_from_dr3():
    g = np.array([14.0, 17.0])
    assert np.allclose(model("dr2").parallax_uncertainty(g), 1.5 * model("dr3").parallax_uncertainty(g))
    assert np.allclose(model("dr2").proper_motion_uncertainty(g)[0],
                       2.0 * model("dr3").proper_motion_uncertainty(g)[0])
    for band in range(3):
        assert np.allclose(model("dr2").photometric_uncertainty(g)[band],
                           1.3 * model("dr3").photometric_uncertainty(g)[band])


# --- radial velocities -----------------------------------------------------------

def test_rv_uncertainty_is_called_with_grvs_teff_logg(monkeypatch):
    """Regression: PyGaia's model takes (G_RVS, T_eff, log g).  It was once
    called with (G, T_eff), failed, and fell back silently to a toy model."""
    import pygaia.errors.spectroscopic as spec
    seen = {}

    def spy(grvs, teff, logg, release="dr4"):
        seen.update(grvs=np.array(grvs), teff=np.array(teff), logg=np.array(logg), release=release)
        return np.full(len(grvs), 1.0)

    monkeypatch.setattr(spec, "radial_velocity_uncertainty", spy)
    cat = make_catalogue(50, np.random.default_rng(0))
    grvs = cat["true"]["G_RVSmag_app"].copy()
    out = model("dr3").apply_errors(cat, rng=np.random.default_rng(1))
    assert np.allclose(seen["grvs"], grvs)
    assert not np.allclose(seen["grvs"], out["true"]["Gmag_app"])
    assert np.allclose(seen["teff"], 5000.0)
    assert np.allclose(seen["logg"], 4.3)
    assert seen["release"] == "dr3"
    assert np.allclose(out["uncertainties"]["rv"], 1.0)


def test_rv_uncertainty_values_and_grid_edges():
    m = model("dr3")
    teff, logg = np.full(4, 5000.0), np.full(4, 4.3)
    err = m.radial_velocity_uncertainty(np.array([8.0, 11.0, 13.5, 15.5]), teff, logg)
    assert 0.11 <= err[0] < 1.0
    assert err[0] < err[1] < err[2] < 10.0
    assert np.isnan(err[3])                           # fainter than the DR3 RVS grid
    # DR4 reaches fainter
    assert np.isfinite(model("dr4").radial_velocity_uncertainty(
        np.array([15.5]), np.array([5000.0]), np.array([4.3]))[0])
    # stars with no G_RVS have no RV
    assert np.isnan(m.radial_velocity_uncertainty(
        np.array([np.nan]), np.array([5000.0]), np.array([4.3]))[0])


def test_rv_uncertainty_has_a_floor():
    err = model("dr3").radial_velocity_uncertainty(
        np.array([5.0, 6.0]), np.array([5000.0, 5000.0]), np.array([4.3, 2.5]))
    assert np.all(err[np.isfinite(err)] >= 0.11)


def test_rv_uncertainty_defaults_when_teff_or_logg_are_missing(capsys):
    err = model("dr3").radial_velocity_uncertainty(np.array([10.0, 12.0]))
    assert np.all(np.isfinite(err))
    assert "assuming 5500 K" in capsys.readouterr().out


def test_grvs_is_derived_when_the_catalogue_lacks_it():
    cat = make_catalogue(200, np.random.default_rng(0))
    with_col = model().apply_errors(
        {k: dict(v) for k, v in cat.items()}, rng=np.random.default_rng(1))
    del cat["true"]["G_RVSmag_app"]
    without = model().apply_errors(cat, rng=np.random.default_rng(1))
    assert np.allclose(with_col["uncertainties"]["rv"], without["uncertainties"]["rv"], equal_nan=True)


def test_stars_outside_the_rv_grid_get_nan_velocities(capsys):
    cat = make_catalogue(500, np.random.default_rng(0), g_lo=13.0, g_hi=16.5)
    out = model("dr3").apply_errors(cat, rng=np.random.default_rng(1))
    unc, obs = out["uncertainties"]["rv"], out["observed"]["rv"]
    assert np.any(np.isnan(unc)) and np.any(np.isfinite(unc))
    assert np.array_equal(np.isnan(unc), np.isnan(obs))
    assert "observed rv set to NaN" in capsys.readouterr().out
    # everything else is still observed
    assert np.all(np.isfinite(out["observed"]["parallax"]))


# --- apply_errors ---------------------------------------------------------------

def test_apply_errors_structure():
    cat = make_catalogue(100, np.random.default_rng(0))
    out = model().apply_errors(cat, rng=np.random.default_rng(1))
    assert set(out) == {"true", "stellar", "observed", "uncertainties"}
    for k in ("parallax", "pmra", "pmdec", "rv", "ra", "dec", "Gmag", "G_BPmag", "G_RPmag"):
        assert k in out["observed"] and k in out["uncertainties"], k
        assert len(out["observed"][k]) == 100
    # Galactic coordinates are carried through unperturbed
    assert np.array_equal(out["observed"]["l"], out["true"]["l"])
    assert np.array_equal(out["observed"]["b"], out["true"]["b"])


def test_observed_values_scatter_by_their_quoted_uncertainty():
    n = 40_000
    cat = make_catalogue(n, np.random.default_rng(0))
    out = model().apply_errors(cat, rng=np.random.default_rng(1))
    true, obs, unc = out["true"], out["observed"], out["uncertainties"]
    pairs = [("parallax", "parallax"), ("pmra", "pmra"), ("pmdec", "pmdec"), ("rv", "rv"),
             ("Gmag", "Gmag_app"), ("G_BPmag", "G_BPmag_app"), ("G_RPmag", "G_RPmag_app")]
    for name, tname in pairs:
        pull = (obs[name] - true[tname]) / unc[name]
        pull = pull[np.isfinite(pull)]
        assert len(pull) > 0.9 * n
        assert pull.mean() == pytest.approx(0.0, abs=0.02), name
        assert pull.std() == pytest.approx(1.0, abs=0.02), name


def test_position_uncertainties_are_in_arcsec():
    n = 40_000
    cat = make_catalogue(n, np.random.default_rng(0))
    out = model().apply_errors(cat, rng=np.random.default_rng(1))
    true, obs, unc = out["true"], out["observed"], out["uncertainties"]
    for k in ("ra", "dec"):
        pull = (obs[k] - true[k]) * 3600.0 / unc[k]
        assert pull.std() == pytest.approx(1.0, abs=0.02)
        assert np.all(unc[k] < 1e-3)                  # sub-mas


def test_errors_are_independent_between_quantities():
    cat = make_catalogue(40_000, np.random.default_rng(0))
    out = model().apply_errors(cat, rng=np.random.default_rng(1))
    true, obs, unc = out["true"], out["observed"], out["uncertainties"]
    pulls = np.array([(obs[k] - true[k]) / unc[k] for k in ("parallax", "pmra", "pmdec")])
    assert np.allclose(np.corrcoef(pulls), np.eye(3), atol=0.03)


def test_stars_outside_gaias_magnitude_range_are_removed(capsys):
    cat = make_catalogue(100, np.random.default_rng(0))
    cat["true"]["Gmag_app"][:3] = [3.5, 21.5, 3.9]
    kept_mini = cat["stellar"]["Mini"][3:].copy()
    out = model().apply_errors(cat, rng=np.random.default_rng(1))
    assert len(out["true"]["Gmag_app"]) == 97
    assert out["true"]["Gmag_app"].min() >= GAIA_MAG_BRIGHT
    assert out["true"]["Gmag_app"].max() <= GAIA_MAG_FAINT
    assert np.array_equal(out["stellar"]["Mini"], kept_mini)       # stellar stays aligned
    assert all(len(v) == 97 for v in out["observed"].values())
    text = capsys.readouterr().out
    assert "2 stars too bright" in text and "1 stars too faint" in text


def test_no_valid_stars_gives_empty_groups():
    cat = make_catalogue(5, np.random.default_rng(0))
    cat["true"]["Gmag_app"][:] = 2.0
    out = model().apply_errors(cat, rng=np.random.default_rng(1))
    assert out["observed"] == {} and out["uncertainties"] == {}


def test_empty_catalogue_is_returned_unchanged():
    cat = {"true": {}, "stellar": {}}
    assert model().apply_errors(cat) is cat


def test_only_available_quantities_are_observed():
    cat = make_catalogue(50, np.random.default_rng(0))
    for k in ("ra", "dec", "pmra", "pmdec", "rv", "l", "b"):
        del cat["true"][k]
    out = model().apply_errors(cat, rng=np.random.default_rng(1))
    assert set(out["observed"]) == {"parallax", "Gmag", "G_BPmag", "G_RPmag"}
    assert "pmra" in out["uncertainties"]            # the uncertainty is still reported


def test_apply_errors_is_reproducible():
    a = model().apply_errors(make_catalogue(200, np.random.default_rng(0)), rng=np.random.default_rng(9))
    b = model().apply_errors(make_catalogue(200, np.random.default_rng(0)), rng=np.random.default_rng(9))
    for k in a["observed"]:
        assert np.array_equal(a["observed"][k], b["observed"][k], equal_nan=True)


# --- fallback without PyGaia -----------------------------------------------------

def test_simple_fallback_model():
    m = model()
    m._has_pygaia = False
    g = np.array([12.0, 15.0, 17.0])
    assert m.parallax_uncertainty(g)[1] == pytest.approx(0.02)
    assert np.all(np.diff(m.parallax_uncertainty(g)) > 0)
    pmra, pmdec = m.proper_motion_uncertainty(g)
    assert np.array_equal(pmra, pmdec)
    assert len(m.photometric_uncertainty(g)) == 3
    rv = m.radial_velocity_uncertainty(g)
    assert rv[0] == pytest.approx(0.3) and rv[1] > rv[0] and np.isnan(rv[2])
    out = m.apply_errors(make_catalogue(100, np.random.default_rng(0)), rng=np.random.default_rng(1))
    assert len(out["observed"]["parallax"]) == 100


def test_missing_pygaia_is_reported(monkeypatch, capsys):
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pygaia":
            raise ImportError("no pygaia")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    m = model()
    assert m._has_pygaia is False
    assert "PyGaia not installed" in capsys.readouterr().out
