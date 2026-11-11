"""Extinction: coefficients, calculators, simulation-gas ray march."""

import numpy as np
import pytest

from snapdragons import Config
from snapdragons.extinction import (
    EXTINCTION_COEFFS, ExtinctionCalculator, MWDustExtinction, NoExtinction,
    SimulationExtinction, get_extinction_calculator,
)
from snapdragons.simulation import SimulationSnapshot


def test_coefficients():
    # Casagrande & VandenBerg (2018) at R_V = 3.1; RVS taken as RP
    assert EXTINCTION_COEFFS["Gmag"] == pytest.approx(0.789)
    assert EXTINCTION_COEFFS["G_BPmag"] == pytest.approx(1.002)
    assert EXTINCTION_COEFFS["G_RPmag"] == pytest.approx(0.589)
    assert EXTINCTION_COEFFS["G_RVSmag"] == EXTINCTION_COEFFS["G_RPmag"]
    # blue is extinguished more than red
    assert (EXTINCTION_COEFFS["G_BPmag"] > EXTINCTION_COEFFS["Gmag"]
            > EXTINCTION_COEFFS["G_RPmag"])


def test_no_extinction():
    calc = NoExtinction(Config())
    assert np.array_equal(calc.compute(np.zeros(5), np.zeros(5), np.ones(5)), np.zeros(5))
    # accepts the keyword arguments the synthesis passes
    out = calc.compute(np.zeros(3), np.zeros(3), np.ones(3),
                       star_pos=np.zeros((3, 3)), observer_pos=np.zeros(3))
    assert np.array_equal(out, np.zeros(3))


def test_base_class_compute_is_abstract():
    with pytest.raises(NotImplementedError):
        ExtinctionCalculator(Config()).compute(np.zeros(1), np.zeros(1), np.ones(1))


def test_apply_adds_band_extinction():
    calc = NoExtinction(Config(rv_extinction=3.1))
    mags = {"Gmag": np.array([10.0, 12.0]), "G_BPmag": np.array([10.5, 12.5]),
            "other": np.array([1.0, 1.0])}
    a_v = np.array([1.0, 2.0])
    out = calc.apply(mags, a_v)
    assert np.allclose(out["Gmag"], [10.789, 13.578])
    assert np.allclose(out["G_BPmag"], [11.502, 14.504])
    assert np.allclose(out["other"], [1.0, 1.0])      # unknown band untouched
    assert np.allclose(out["E_BV"], a_v / 3.1)
    assert np.allclose(out["A_Gmag"], 0.789 * a_v)
    assert np.array_equal(out["A_V"], a_v)


# --- factory -----------------------------------------------------------------

def test_factory_modes(mock_snapshot, mock_snapshot_gas):
    assert isinstance(get_extinction_calculator(Config(extinction_mode="none")), NoExtinction)
    assert isinstance(get_extinction_calculator(Config(extinction_mode="NONE")), NoExtinction)
    assert isinstance(get_extinction_calculator(Config(extinction_mode="mwdust")), MWDustExtinction)
    sim = get_extinction_calculator(Config(extinction_mode="simulation"), mock_snapshot_gas)
    assert isinstance(sim, SimulationExtinction)
    with pytest.raises(ValueError):
        get_extinction_calculator(Config(extinction_mode="simulation"))            # no snapshot
    with pytest.raises(ValueError):
        get_extinction_calculator(Config(extinction_mode="simulation"), mock_snapshot)  # no gas
    with pytest.raises(ValueError):
        get_extinction_calculator(Config(extinction_mode="schlegel"))


def test_mwdust_map_is_loaded_lazily_and_unknown_maps_are_rejected():
    pytest.importorskip("mwdust")
    calc = MWDustExtinction(Config(mwdust_map="NoSuchMap"))
    assert calc._dustmap is None          # nothing loaded at construction
    with pytest.raises(ValueError):
        calc.compute(np.array([10.0]), np.array([0.0]), np.array([1.0]))


@pytest.mark.external
def test_mwdust_combined19():
    pytest.importorskip("mwdust")
    calc = MWDustExtinction(Config(mwdust_map="Combined19"))
    l = np.array([30.0, 30.0, 30.0, 0.0])
    b = np.array([0.0, 0.0, 0.0, 90.0])
    d = np.array([0.5, 2.0, 6.0, 2.0])
    a_v = calc.compute(l, b, d)
    assert a_v.shape == (4,)
    assert np.all(a_v >= 0)
    assert a_v[0] < a_v[1] <= a_v[2]       # grows along a plane sightline
    assert a_v[3] < 0.3                    # the pole is nearly clear
    assert a_v[2] > 1.0


# --- simulation gas ----------------------------------------------------------

RHO = 1e7       # Msun / kpc^3 = 0.01 Msun / pc^3


def gas_slab(spacing=0.05, h=1.0, rho=RHO):
    """Uniform lattice of gas over 0 <= x <= 2, |y|, |z| <= 0.2 kpc.  With
    h much larger than the spacing the kernel weight is ~1 everywhere, so
    the sampled density is the particle density."""
    gx, gy, gz = np.meshgrid(np.arange(0.0, 2.0 + 1e-9, spacing),
                             np.arange(-0.2, 0.2 + 1e-9, spacing),
                             np.arange(-0.2, 0.2 + 1e-9, spacing), indexing="ij")
    pos = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
    n = len(pos)
    return pos, np.full(n, rho), np.full(n, h)


def make_sim_ext(dust_to_gas=1.0, rho=RHO, **kw):
    pos, dens, h = gas_slab(rho=rho, **kw)
    return SimulationExtinction(Config(dust_to_gas_ratio=dust_to_gas), pos, dens, h)


def expected_av(rho, length_kpc, rv=3.1):
    msun, kpc, mp = 1.98892e33, 3.085677581e21, 1.6726219e-24
    n_h = rho * msun / kpc**3 / (1.4 * mp)
    return n_h * length_kpc * kpc / 5.8e21 * rv


def test_rho_to_nh_units():
    calc = make_sim_ext()
    # 1 Msun / pc^3 = 1e9 Msun / kpc^3 is 28.9 H nuclei per cm^3 for mu = 1.4
    assert calc._rho_to_nh(1e9) == pytest.approx(28.9, rel=5e-3)


def test_uniform_medium_gives_the_analytic_column():
    calc = make_sim_ext()
    obs = np.array([0.2, 0.0, 0.0])
    stars = np.array([[1.8, 0.0, 0.0], [1.0, 0.0, 0.0], [0.6, 0.1, 0.05]])
    a_v = calc.compute(None, None, None, star_pos=stars, observer_pos=obs)
    lengths = np.linalg.norm(stars - obs, axis=1)
    assert np.allclose(a_v, expected_av(RHO, lengths), rtol=0.03)
    # sanity on the scale: 0.29 cm^-3 over 1.6 kpc is A_V ~ 0.76
    assert a_v[0] == pytest.approx(0.76, abs=0.04)


def test_scales_with_density_dust_ratio_and_rv():
    obs = np.array([0.2, 0.0, 0.0])
    star = np.array([[1.8, 0.0, 0.0]])
    base = make_sim_ext().compute(None, None, None, star_pos=star, observer_pos=obs)[0]
    half_dust = make_sim_ext(dust_to_gas=0.5).compute(None, None, None, star_pos=star, observer_pos=obs)[0]
    double_rho = make_sim_ext(rho=2 * RHO).compute(None, None, None, star_pos=star, observer_pos=obs)[0]
    assert half_dust == pytest.approx(0.5 * base)
    assert double_rho == pytest.approx(2.0 * base)
    pos, dens, h = gas_slab()
    rv5 = SimulationExtinction(Config(rv_extinction=5.0), pos, dens, h).compute(
        None, None, None, star_pos=star, observer_pos=obs)[0]
    assert rv5 == pytest.approx(base * 5.0 / 3.1)


def test_increases_along_a_sightline():
    calc = make_sim_ext()
    obs = np.array([0.1, 0.0, 0.0])
    stars = np.column_stack([np.linspace(0.3, 1.9, 9), np.zeros(9), np.zeros(9)])
    a_v = calc.compute(None, None, None, star_pos=stars, observer_pos=obs)
    assert np.all(np.diff(a_v) > 0)


def test_no_gas_within_a_smoothing_length_is_transparent():
    pos, dens, _ = gas_slab()
    calc = SimulationExtinction(Config(), pos, dens, np.full(len(pos), 0.01))
    obs = np.array([0.0, 5.0, 5.0])
    stars = np.array([[1.0, 5.0, 5.0], [2.0, 6.0, 5.0]])
    assert np.array_equal(
        calc.compute(None, None, None, star_pos=stars, observer_pos=obs), [0.0, 0.0])


def test_star_at_the_observer_and_batching():
    calc = make_sim_ext()
    obs = np.array([0.5, 0.0, 0.0])
    stars = np.tile(np.array([[1.5, 0.0, 0.0]]), (5200, 1))   # more than one 5000-star batch
    stars[0] = obs
    a_v = calc.compute(None, None, None, star_pos=stars, observer_pos=obs)
    assert a_v[0] == 0.0
    assert np.allclose(a_v[1:], a_v[1])
    assert a_v[1] > 0


def test_more_samples_converge():
    calc = make_sim_ext()
    obs = np.array([0.2, 0.0, 0.0])
    star = np.array([[1.8, 0.05, 0.0]])
    coarse = calc.compute(None, None, None, star_pos=star, observer_pos=obs, n_sample=16)[0]
    fine = calc.compute(None, None, None, star_pos=star, observer_pos=obs, n_sample=256)[0]
    assert coarse == pytest.approx(fine, rel=0.02)


def test_requires_positions():
    calc = make_sim_ext()
    with pytest.raises(ValueError):
        calc.compute(np.zeros(1), np.zeros(1), np.ones(1))


def test_snapshot_gas_feeds_the_calculator(mock_snapshot_gas):
    calc = get_extinction_calculator(Config(extinction_mode="simulation"), mock_snapshot_gas)
    obs = np.asarray(Config().observer_position)
    a_v = calc.compute(None, None, None, star_pos=mock_snapshot_gas.star_pos, observer_pos=obs)
    assert a_v.shape == (mock_snapshot_gas.n_star,)
    assert np.all(a_v >= 0) and np.median(a_v) > 0
    d = np.linalg.norm(mock_snapshot_gas.star_pos - obs, axis=1)
    assert np.corrcoef(d, a_v)[0, 1] > 0.8        # further stars are more extinguished
