"""Population synthesis: mass sampling, magnitude limits, binaries, smoothing."""

import numpy as np
import pytest
from scipy import integrate

from snapdragons.extinction import EXTINCTION_COEFFS
from snapdragons.imf import KroupaIMF, SalpeterIMF
from snapdragons.photometry import grvs_from_g_rp
from snapdragons.population import (
    _mass_integral, _vectorised_count_sample, _vectorised_mass_budget,
    synthesise_population,
)
from snapdragons.smoothing import estimate_smoothing_lengths

from conftest import ConstantExtinction, build_mock_snapshot


def kroupa_mass_pdf(m):
    return m * (m ** -1.3 if m < 0.5 else 0.5 * m ** -2.3)


# --- mass integrals and sampling helpers -------------------------------------

@pytest.mark.parametrize("lo,hi", [(0.08, 100.0), (0.08, 0.3), (0.3, 2.0), (1.0, 50.0)])
def test_mass_integral_kroupa(lo, hi):
    pts = [0.5] if lo < 0.5 < hi else None
    expected = integrate.quad(kroupa_mass_pdf, lo, hi, points=pts)[0]
    assert _mass_integral(KroupaIMF(), lo, hi) == pytest.approx(expected, rel=1e-8)


def test_mass_integral_salpeter():
    expected = integrate.quad(lambda m: m ** -1.35, 0.5, 20.0)[0]
    assert _mass_integral(SalpeterIMF(), 0.5, 20.0) == pytest.approx(expected, rel=1e-8)


def _budget_draw(particle_masses, m_trunc, seed=0):
    imf = KroupaIMF()
    rng = np.random.default_rng(seed)
    full = _mass_integral(imf, imf.m_min, imf.m_max)
    masses, counts = _vectorised_mass_budget(
        imf, particle_masses, m_trunc, imf.m_max, rng, full, {})
    budget = particle_masses * _mass_integral(imf, m_trunc, imf.m_max) / full
    return masses, counts, budget


@pytest.mark.parametrize("m_trunc", [0.08, 0.5, 1.0, 3.0])
@pytest.mark.parametrize("particle_mass", [30.0, 1e3, 3e4])
def test_mass_budget_is_unbiased(m_trunc, particle_mass):
    """The stars drawn above m_trunc carry, on average, the particle's mass
    in that part of the IMF: the boundary-star rule (keep the star that
    overshoots with probability remaining / its mass) makes the total
    unbiased whatever the particle mass."""
    n = int(max(400, 4e6 / particle_mass))
    pm = np.full(n, particle_mass)
    masses, counts, budget = _budget_draw(pm, m_trunc)
    assert counts.shape == (n,) and counts.sum() == len(masses)
    assert np.all(masses >= m_trunc) and np.all(masses <= 100.0)
    per_particle = np.bincount(np.repeat(np.arange(n), counts), weights=masses, minlength=n)
    err = per_particle.std() / np.sqrt(n)
    assert abs(per_particle.mean() - budget[0]) < 4 * err


def test_mass_budget_per_particle_overshoot_is_at_most_one_star():
    pm = np.full(2000, 5e3)
    masses, counts, budget = _budget_draw(pm, 0.3)
    starts = np.cumsum(counts) - counts
    total = np.add.reduceat(masses, starts)
    last = masses[starts + counts - 1]
    # without its last star every particle is below budget; with it, it
    # overshoots by less than that star
    assert np.all(total - last <= budget + 1e-9)
    assert np.all(total >= budget - 100.0)


def test_mass_budget_scales_with_particle_mass():
    pm = np.concatenate([np.full(3000, 1e3), np.full(3000, 4e3)])
    masses, counts, _ = _budget_draw(pm, 0.5)
    per = np.bincount(np.repeat(np.arange(len(pm)), counts), weights=masses, minlength=len(pm))
    assert per[3000:].mean() / per[:3000].mean() == pytest.approx(4.0, rel=0.05)


def test_mass_budget_zero_mass_particles_get_no_stars():
    pm = np.array([0.0, 1e4, 0.0])
    masses, counts, _ = _budget_draw(pm, 0.5)
    assert counts[0] == 0 and counts[2] == 0 and counts[1] > 0
    assert len(masses) == counts[1]


def test_mass_budget_truncation_at_the_upper_limit_gives_nothing():
    masses, counts, _ = _budget_draw(np.full(10, 1e4), 100.0)
    assert len(masses) == 0 and counts.sum() == 0


def test_mass_budget_is_reproducible():
    a = _budget_draw(np.full(50, 1e3), 0.4, seed=5)
    b = _budget_draw(np.full(50, 1e3), 0.4, seed=5)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_count_sampling_poisson_and_deterministic(rng):
    imf = KroupaIMF()
    pm = np.full(4000, 2e3)
    expected = imf.expected_count(2e3, 1.0)
    masses, counts = _vectorised_count_sample(imf, pm, 1.0, "poisson", rng)
    assert counts.sum() == len(masses)
    assert counts.mean() == pytest.approx(expected, rel=0.02)
    assert counts.var() == pytest.approx(expected, rel=0.1)
    assert masses.min() >= 1.0
    masses, counts = _vectorised_count_sample(imf, pm, 1.0, "deterministic", rng)
    assert np.all(counts == int(np.round(expected)))
    masses, counts = _vectorised_count_sample(imf, pm, 100.0, "poisson", rng)
    assert len(masses) == 0 and counts.sum() == 0


# --- the synthesis as a whole ---------------------------------------------------

def synth(snapshot, iso_grid, cfg, **kw):
    return synthesise_population(snapshot, iso_grid, cfg, **kw)


def test_output_columns_and_lengths(mock_snapshot, iso_grid, make_config):
    cat = synth(mock_snapshot, iso_grid, make_config())
    n = len(cat["Mini"])
    assert n > 1000
    for key in ("pos", "vel", "Mini", "logage", "MH", "parent_id", "dist_mod", "A_V",
                "Gmag", "G_BPmag", "G_RPmag", "G_RVSmag", "logTe", "logg", "Mass",
                "Gmag_app", "G_BPmag_app", "G_RPmag_app", "G_RVSmag_app"):
        assert key in cat, key
        assert len(cat[key]) == n, key
    assert cat["pos"].shape == (n, 3) and cat["vel"].shape == (n, 3)
    assert cat["parent_id"].dtype == np.int64
    assert "is_binary" not in cat


def test_stars_inherit_their_parent_particle(mock_snapshot, iso_grid, make_config):
    cat = synth(mock_snapshot, iso_grid, make_config())
    pid = cat["parent_id"]
    assert pid.min() >= 0 and pid.max() < mock_snapshot.n_star
    assert np.array_equal(cat["pos"], mock_snapshot.star_pos[pid])
    assert np.array_equal(cat["vel"], mock_snapshot.star_vel[pid])
    assert np.allclose(cat["logage"], np.log10(mock_snapshot.star_age[pid] * 1e9))
    assert np.allclose(cat["MH"], mock_snapshot.star_metallicity[pid])
    # every particle of this nearby patch contributes
    assert len(np.unique(pid)) == mock_snapshot.n_star


def test_photometry_is_self_consistent(mock_snapshot, iso_grid, make_config):
    cfg = make_config()
    cat = synth(mock_snapshot, iso_grid, cfg)
    d = np.linalg.norm(cat["pos"] - np.asarray(cfg.observer_position), axis=1)
    dm = 5 * np.log10(d * 1000) - 5
    assert np.allclose(cat["dist_mod"], dm)
    for band in ("Gmag", "G_BPmag", "G_RPmag"):
        assert np.allclose(cat[f"{band}_app"], cat[band] + dm)
    assert np.allclose(cat["G_RVSmag_app"], grvs_from_g_rp(cat["Gmag_app"], cat["G_RPmag_app"]))
    assert np.allclose(cat["G_RVSmag"], grvs_from_g_rp(cat["Gmag"], cat["G_RPmag"]), equal_nan=True)
    assert np.all(cat["A_V"] == 0)


def test_stellar_properties_come_from_the_parent_isochrone(mock_snapshot, iso_grid, make_config):
    cat = synth(mock_snapshot, iso_grid, make_config())
    cells = iso_grid.cell_of(cat["logage"], cat["MH"])
    for ci in np.unique(cells)[:4]:
        s = cells == ci
        la, mh = iso_grid.grid_points[ci]
        props = iso_grid.interpolate_stars(la, mh, cat["Mini"][s])
        for col in ("Gmag", "G_RPmag", "logTe", "logg"):
            assert np.allclose(cat[col][s], props[col])
    assert np.all(cat["Mini"] >= 0.08) and np.all(cat["Mini"] <= 100.0)
    # current mass never exceeds the initial mass (to the table's 1e-3 rounding)
    assert np.all(cat["Mass"] <= cat["Mini"] + 2e-3)


@pytest.mark.parametrize("band,column", [("G_RVS", "G_RVSmag_app"), ("G", "Gmag_app")])
@pytest.mark.parametrize("limit", [10.0, 12.0, 14.0])
def test_faint_limit_applies_in_the_configured_band(mock_snapshot, iso_grid, make_config,
                                                    band, column, limit):
    cfg = make_config(mag_limit=limit, mag_limit_band=band)
    cat = synth(mock_snapshot, iso_grid, cfg)
    assert np.all(np.isfinite(cat[column]))
    assert cat[column].max() <= limit
    assert cat[column].max() > limit - 0.1             # the sample reaches the limit
    assert cat["Gmag_app"].min() >= cfg.mag_bright_limit


def test_g_rvs_limit_is_not_a_g_limit(mock_snapshot, iso_grid, make_config):
    cat = synth(mock_snapshot, iso_grid, make_config(mag_limit=12.0, mag_limit_band="G_RVS"))
    # G_RVS is brighter than G: a G_RVS-limited sample reaches fainter in G
    assert cat["Gmag_app"].max() > 12.3
    cat_g = synth(mock_snapshot, iso_grid, make_config(mag_limit=12.0, mag_limit_band="G"))
    assert len(cat["Mini"]) > len(cat_g["Mini"])


def test_bright_limit_is_in_g(mock_snapshot, iso_grid, make_config):
    cat = synth(mock_snapshot, iso_grid, make_config(mag_bright_limit=9.0))
    assert cat["Gmag_app"].min() >= 9.0
    assert cat["G_RVSmag_app"].min() < 9.0


def test_deeper_limits_give_more_stars(mock_snapshot, iso_grid, make_config):
    n = [len(synth(mock_snapshot, iso_grid, make_config(mag_limit=lim))["Mini"])
         for lim in (9.0, 11.0, 13.0)]
    assert n[0] < n[1] < n[2]


def test_limiting_mass_truncation_loses_no_stars(iso_grid, make_config):
    """Sampling only above the limiting mass must give the same stars
    brighter than the limit as sampling far deeper and cutting afterwards."""
    snap = build_mock_snapshot(n=300, seed=11, mass=2e4)
    shallow = synth(snap, iso_grid, make_config(mag_limit=11.0, random_seed=1))
    deep = synth(snap, iso_grid, make_config(mag_limit=15.0, random_seed=2))
    n_shallow = len(shallow["Mini"])
    n_deep_cut = int(np.sum(deep["G_RVSmag_app"] <= 11.0))
    assert n_shallow > 20_000
    assert abs(n_shallow - n_deep_cut) < 5 * np.sqrt(n_shallow + n_deep_cut)
    # and the same mass function above the limit
    q = [10, 50, 90, 99]
    assert np.allclose(np.percentile(shallow["Mini"], q),
                       np.percentile(deep["Mini"][deep["G_RVSmag_app"] <= 11.0], q), rtol=0.03)


def test_star_counts_scale_with_particle_mass(iso_grid, make_config):
    light = build_mock_snapshot(n=300, seed=3, mass=2e3)
    heavy = build_mock_snapshot(n=300, seed=3, mass=8e3)
    n_light = len(synth(light, iso_grid, make_config())["Mini"])
    n_heavy = len(synth(heavy, iso_grid, make_config())["Mini"])
    assert n_heavy / n_light == pytest.approx(4.0, rel=0.05)


@pytest.mark.parametrize("sampling", ["poisson", "deterministic"])
def test_sampling_modes_agree_on_the_star_count(mock_snapshot, iso_grid, make_config, sampling):
    n_budget = len(synth(mock_snapshot, iso_grid, make_config(sampling="mass_budget"))["Mini"])
    n_other = len(synth(mock_snapshot, iso_grid, make_config(sampling=sampling))["Mini"])
    assert n_other == pytest.approx(n_budget, rel=0.05)


def test_salpeter_imf_runs_and_differs(mock_snapshot, iso_grid, make_config):
    n_k = len(synth(mock_snapshot, iso_grid, make_config(imf="kroupa"))["Mini"])
    n_s = len(synth(mock_snapshot, iso_grid, make_config(imf="salpeter"))["Mini"])
    # Salpeter puts more of the mass into unobservable dwarfs
    assert 0.3 * n_k < n_s < 0.9 * n_k


def test_same_seed_same_catalogue(mock_snapshot, iso_grid, make_config):
    a = synth(mock_snapshot, iso_grid, make_config(random_seed=5))
    b = synth(mock_snapshot, iso_grid, make_config(random_seed=5))
    c = synth(mock_snapshot, iso_grid, make_config(random_seed=6))
    assert a.keys() == b.keys()
    for k in a:
        assert np.array_equal(a[k], b[k], equal_nan=True), k
    assert len(a["Mini"]) != len(c["Mini"]) or not np.array_equal(a["Mini"], c["Mini"])


def test_fsample_downsamples_the_particles(mock_snapshot, iso_grid, make_config):
    full = synth(mock_snapshot, iso_grid, make_config())
    part = synth(mock_snapshot, iso_grid, make_config(fsample=0.2))
    assert len(np.unique(part["parent_id"])) <= int(mock_snapshot.n_star * 0.2)
    assert len(part["Mini"]) == pytest.approx(0.2 * len(full["Mini"]), rel=0.25)
    assert part["parent_id"].max() < mock_snapshot.n_star


def test_nothing_observable_returns_none(mock_snapshot, iso_grid, make_config):
    assert synth(mock_snapshot, iso_grid, make_config(mag_limit=-5.0)) is None


def test_particles_outside_the_age_metallicity_grid_are_clipped(iso_grid, make_config):
    snap = build_mock_snapshot(n=50, seed=2)
    snap.star_age = np.full(50, 13.5)             # log age 10.13 > config maximum 10.0
    snap.star_metallicity = np.full(50, -3.0)
    cfg = make_config(logage_min=8.0, logage_max=10.0, mh_min=-1.0, mh_max=0.0)
    cat = synth(snap, iso_grid, cfg)
    assert np.allclose(cat["logage"], 10.0)
    assert np.allclose(cat["MH"], -1.0)


# --- extinction inside the synthesis --------------------------------------------

def test_extinction_dims_the_stars_and_shrinks_the_sample(mock_snapshot, iso_grid, make_config):
    cfg = make_config()
    clear = synth(mock_snapshot, iso_grid, cfg)
    dusty = synth(mock_snapshot, iso_grid, cfg, extinction_calc=ConstantExtinction(1.5))
    assert np.all(dusty["A_V"] == 1.5)
    for band in ("Gmag", "G_BPmag", "G_RPmag"):
        assert np.allclose(dusty[f"{band}_app"],
                           dusty[band] + dusty["dist_mod"] + 1.5 * EXTINCTION_COEFFS[band])
    assert dusty["G_RVSmag_app"].max() <= cfg.mag_limit
    assert len(dusty["Mini"]) < 0.8 * len(clear["Mini"])
    # the limiting mass moved up with the extinction
    assert dusty["Mini"].min() >= clear["Mini"].min()


def test_extinction_is_evaluated_at_the_particles(mock_snapshot, iso_grid, make_config):
    class Recorder:
        def compute(self, l_deg, b_deg, dist_kpc, star_pos=None, observer_pos=None):
            self.n = len(l_deg)
            self.star_pos = star_pos
            self.observer_pos = observer_pos
            return 0.3 * dist_kpc

    rec = Recorder()
    cfg = make_config()
    cat = synth(mock_snapshot, iso_grid, cfg, extinction_calc=rec)
    assert rec.n == mock_snapshot.n_star            # once per particle, not per star
    assert np.array_equal(rec.star_pos, mock_snapshot.star_pos)
    assert np.allclose(rec.observer_pos, cfg.observer_position)
    d = np.linalg.norm(cat["pos"] - np.asarray(cfg.observer_position), axis=1)
    assert np.allclose(cat["A_V"], 0.3 * d)


def test_calculators_without_position_arguments_are_supported(mock_snapshot, iso_grid, make_config):
    class Simple:
        def compute(self, l_deg, b_deg, dist_kpc):
            return np.full(len(l_deg), 0.5)

    cat = synth(mock_snapshot, iso_grid, make_config(), extinction_calc=Simple())
    assert np.all(cat["A_V"] == 0.5)


# --- unresolved binaries ---------------------------------------------------------

@pytest.fixture(scope="module")
def binary_cat(iso_grid, tmp_path_factory):
    from snapdragons import Config
    snap = build_mock_snapshot(n=300, seed=7, mass=2e4)
    cfg = Config(isochrone_cache_dir=str(tmp_path_factory.mktemp("iso")),
                 extinction_mode="none", random_seed=42, mag_limit=12.0,
                 unresolved_binaries=True, f_binary=0.5, q_min=0.1)
    return synthesise_population(snap, iso_grid, cfg), cfg


def test_binary_columns(binary_cat):
    cat, cfg = binary_cat
    n = len(cat["Mini"])
    for key in ("is_binary", "q", "Mini_secondary"):
        assert len(cat[key]) == n
    b = cat["is_binary"] > 0
    assert set(np.unique(cat["is_binary"])) == {0.0, 1.0}
    assert np.all(cat["q"][~b] == 0) and np.all(cat["Mini_secondary"][~b] == 0)
    assert cat["q"][b].min() >= cfg.q_min and cat["q"][b].max() <= 1.0
    assert np.all(cat["Mini_secondary"][b] <= cat["Mini"][b] + 1e-9)
    # secondary mass is q x primary unless that falls below the isochrone
    plain = cat["Mini_secondary"][b] > 0.0901
    assert np.allclose(cat["Mini_secondary"][b][plain], (cat["q"] * cat["Mini"])[b][plain])


def primary_grvs_app(cat, iso_grid):
    """Apparent G_RVS each star would have without a companion."""
    out = np.full(len(cat["Mini"]), np.nan)
    cells = iso_grid.cell_of(cat["logage"], cat["MH"])
    for ci in np.unique(cells):
        s = cells == ci
        la, mh = iso_grid.grid_points[ci]
        props = iso_grid.interpolate_stars(la, mh, cat["Mini"][s])
        out[s] = grvs_from_g_rp(props["Gmag"] + cat["dist_mod"][s],
                                props["G_RPmag"] + cat["dist_mod"][s])
    return out


def test_binary_fraction_rises_with_primary_mass(binary_cat, iso_grid):
    """f_bin(m) = clip(f_binary + 0.25 log10 m, 0, 1), among the stars drawn.

    Checked on primaries that pass the magnitude limit on their own, so the
    companion's extra light plays no part in whether the star is in the
    sample (it would otherwise favour binaries near the limit)."""
    cat, cfg = binary_cat
    b = cat["is_binary"] > 0
    alone = primary_grvs_app(cat, iso_grid)
    safe = np.isfinite(alone) & (alone < cfg.mag_limit - 0.05)
    fractions = []
    for lo, hi in [(0.8, 1.2), (1.5, 2.5), (3.0, 6.0)]:
        s = safe & (cat["Mini"] >= lo) & (cat["Mini"] < hi)
        assert s.sum() > 500
        expected = np.mean(np.clip(cfg.f_binary + 0.25 * np.log10(cat["Mini"][s]), 0, 1))
        err = np.sqrt(expected * (1 - expected) / s.sum())
        assert abs(b[s].mean() - expected) < 4 * err
        fractions.append(b[s].mean())
    assert fractions[0] < fractions[1] < fractions[2]


def test_binary_light_is_the_sum_of_both_stars(binary_cat, iso_grid):
    cat, _ = binary_cat
    b = cat["is_binary"] > 0
    cells = iso_grid.cell_of(cat["logage"], cat["MH"])
    ci = np.bincount(cells[b]).argmax()
    s = b & (cells == ci)
    la, mh = iso_grid.grid_points[ci]
    for band in ("Gmag", "G_RPmag", "G_BPmag"):
        m1 = iso_grid.interpolate_stars(la, mh, cat["Mini"][s])[band]
        m2 = iso_grid.interpolate_stars(la, mh, cat["Mini_secondary"][s])[band]
        ok = np.isfinite(m2)
        both = -2.5 * np.log10(10 ** (-0.4 * m1[ok]) + 10 ** (-0.4 * m2[ok]))
        assert np.allclose(cat[band][s][ok], both)
        # brighter than either star, by at most 2.5 log 2 over the brighter one
        # (the companion can be the brighter star when the primary has
        # evolved past its peak luminosity)
        brighter = np.minimum(m1, m2)[ok]
        assert np.all(cat[band][s][ok] <= brighter + 1e-9)
        assert np.all(cat[band][s][ok] >= brighter - 2.5 * np.log10(2.0) - 1e-9)
    # single stars are untouched
    s1 = ~b & (cells == ci)
    assert np.allclose(cat["Gmag"][s1], iso_grid.interpolate_stars(la, mh, cat["Mini"][s1])["Gmag"])


def test_binary_spectroscopic_properties_are_the_primarys(binary_cat, iso_grid):
    cat, _ = binary_cat
    b = cat["is_binary"] > 0
    cells = iso_grid.cell_of(cat["logage"], cat["MH"])
    ci = np.bincount(cells[b]).argmax()
    s = b & (cells == ci)
    la, mh = iso_grid.grid_points[ci]
    props = iso_grid.interpolate_stars(la, mh, cat["Mini"][s])
    assert np.allclose(cat["logTe"][s], props["logTe"])
    assert np.allclose(cat["logg"][s], props["logg"])


def test_binaries_add_stars_near_the_limit(mock_snapshot, iso_grid, make_config):
    single = synth(mock_snapshot, iso_grid, make_config(unresolved_binaries=False))
    binary = synth(mock_snapshot, iso_grid, make_config(unresolved_binaries=True))
    assert len(binary["Mini"]) >= 0.98 * len(single["Mini"])
    assert np.mean(binary["is_binary"]) > 0.2


def test_zero_binary_fraction_at_low_f(mock_snapshot, iso_grid, make_config):
    cat = synth(mock_snapshot, iso_grid, make_config(unresolved_binaries=True, f_binary=-5.0))
    assert np.all(cat["is_binary"] == 0)


# --- phase-space smoothing --------------------------------------------------------

def test_smoothing_spreads_stars_within_the_parent_kernel(mock_snapshot, iso_grid, make_config):
    cfg = make_config(phase_space_smoothing=True, smoothing_ngb=16, smoothing_scale=1.0)
    cat = synth(mock_snapshot, iso_grid, cfg)
    h_pos, h_vel = estimate_smoothing_lengths(
        mock_snapshot.star_pos, mock_snapshot.star_vel, n_ngb=16)
    pid = cat["parent_id"]
    dpos = np.linalg.norm(cat["pos"] - mock_snapshot.star_pos[pid], axis=1)
    dvel = np.linalg.norm(cat["vel"] - mock_snapshot.star_vel[pid], axis=1)
    assert np.all(dpos <= h_pos[pid] * (1 + 1e-9))
    assert np.all(dvel <= h_vel[pid] * (1 + 1e-9))
    assert np.median(dpos / h_pos[pid]) == pytest.approx(0.62, abs=0.05)   # kernel median radius
    assert np.median(dvel / h_vel[pid]) == pytest.approx(0.62, abs=0.05)
    # siblings do not share a position any more
    assert len(np.unique(cat["pos"][:, 0])) == len(cat["Mini"])


def test_smoothing_updates_distances_and_apparent_magnitudes(mock_snapshot, iso_grid, make_config):
    cfg = make_config(phase_space_smoothing=True, smoothing_ngb=16)
    cat = synth(mock_snapshot, iso_grid, cfg)
    d = np.linalg.norm(cat["pos"] - np.asarray(cfg.observer_position), axis=1)
    dm = 5 * np.log10(d * 1000) - 5
    assert np.allclose(cat["dist_mod"], dm)
    assert np.allclose(cat["Gmag_app"], cat["Gmag"] + dm)
    assert np.allclose(cat["G_RVSmag_app"], grvs_from_g_rp(cat["Gmag_app"], cat["G_RPmag_app"]),
                       equal_nan=True)
    # a star sitting exactly on the edge of the G_RVS colour range can lose
    # its G_RVS to rounding when the distance modulus changes: a handful at most
    assert np.mean(np.isnan(cat["G_RVSmag_app"])) < 1e-3
    # stars moved outward can now be fainter than the limit; build_catalogue
    # re-applies the cut (see test_catalogue.py)
    assert np.nanmax(cat["G_RVSmag_app"]) > cfg.mag_limit


def test_smoothing_scale_and_zero_scale(mock_snapshot, iso_grid, make_config):
    zero = synth(mock_snapshot, iso_grid, make_config(phase_space_smoothing=True, smoothing_scale=0.0))
    assert np.allclose(zero["pos"], mock_snapshot.star_pos[zero["parent_id"]])
    one = synth(mock_snapshot, iso_grid, make_config(phase_space_smoothing=True, smoothing_ngb=16))
    three = synth(mock_snapshot, iso_grid, make_config(phase_space_smoothing=True, smoothing_ngb=16,
                                                       smoothing_scale=3.0))
    r1 = np.linalg.norm(one["pos"] - mock_snapshot.star_pos[one["parent_id"]], axis=1)
    r3 = np.linalg.norm(three["pos"] - mock_snapshot.star_pos[three["parent_id"]], axis=1)
    assert np.median(r3) == pytest.approx(3 * np.median(r1), rel=0.05)


def test_smoothing_does_not_change_which_stars_are_drawn(mock_snapshot, iso_grid, make_config):
    plain = synth(mock_snapshot, iso_grid, make_config())
    smooth = synth(mock_snapshot, iso_grid, make_config(phase_space_smoothing=True))
    assert np.array_equal(plain["Mini"], smooth["Mini"])
    assert np.array_equal(plain["parent_id"], smooth["parent_id"])
