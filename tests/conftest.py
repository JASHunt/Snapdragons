"""Shared fixtures for the Snapdragons test suite.

The suite is self-contained: no simulation file, no network, no pynbody.

* ``tests/data/parsec_subset.h5`` is a 12-cell subset (log age 8, 9, 9.7, 10;
  [M/H] -1, -0.5, 0) of the PARSEC grid Snapdragons caches, in the cache
  format, so the real isochrone code paths run on real isochrones.
* ``mock_snapshot`` is a 300-particle disc patch around the default observer
  whose ages and metallicities sit on those cells.

Tests that need downloaded third-party data (the mwdust maps, the
gaiaunlimited selection function) are marked ``external`` and skipped unless
pytest is run with ``--run-external``.

``--update-golden`` rewrites ``tests/data/golden.json`` from the current code
(see test_regression.py).
"""

import os
import shutil

import numpy as np
import pytest

from snapdragons import Config, IsochroneGrid
from snapdragons.simulation import SimulationSnapshot

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
PARSEC_SUBSET = os.path.join(DATA_DIR, "parsec_subset.h5")
SUBSET_LOGAGES = (8.0, 9.0, 9.7, 10.0)
SUBSET_MHS = (-1.0, -0.5, 0.0)

ALL_COORDS = ["equatorial", "galactic", "cartesian"]


def pytest_addoption(parser):
    parser.addoption(
        "--run-external", action="store_true", default=False,
        help="run tests that need downloaded data (mwdust, gaiaunlimited)")
    parser.addoption(
        "--update-golden", action="store_true", default=False,
        help="rewrite tests/data/golden.json from the current code")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-external"):
        return
    skip = pytest.mark.skip(reason="needs downloaded data: use --run-external")
    for item in items:
        if "external" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def rng():
    return np.random.default_rng(12345)


def build_mock_snapshot(n=300, seed=7, mass=5e3, with_gas=False,
                        rotation="clockwise"):
    """Disc patch within ~2 kpc of the default observer at (-8.2, 0, 0.02).

    Rotation at 232.8 km/s, clockwise seen from +z by default (the Milky
    Way's sense) or "counterclockwise", plus a velocity ellipsoid; ages
    and [M/H] drawn from the cells of the PARSEC subset.  The positions
    and the random part of the velocities do not depend on the sense.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(-10.2, -6.2, n)
    y = rng.uniform(-2.0, 2.0, n)
    z = rng.normal(0.0, 0.2, n)
    phi = np.arctan2(y, x)
    sense = -1.0 if rotation == "clockwise" else 1.0
    vel = (sense * 232.8 * np.column_stack([-np.sin(phi), np.cos(phi), np.zeros(n)])
           + rng.normal(0.0, [30.0, 20.0, 15.0], (n, 3)))
    age = rng.choice([0.1, 1.0, 5.0, 10.0], n)
    mh = rng.choice(list(SUBSET_MHS), n)
    kwargs = {}
    if with_gas:
        # uniform lattice of gas through the patch, n_H ~ 0.29 cm^-3
        gx, gy, gz = np.meshgrid(np.arange(-10.5, -5.9, 0.25),
                                 np.arange(-2.25, 2.3, 0.25),
                                 np.arange(-0.5, 0.55, 0.25), indexing="ij")
        gas_pos = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
        ng = len(gas_pos)
        kwargs = dict(gas_pos=gas_pos, gas_density=np.full(ng, 1e7),
                      gas_metallicity=np.zeros(ng),
                      gas_smooth=np.full(ng, 1.0))
    return SimulationSnapshot.from_arrays(
        np.column_stack([x, y, z]), vel, np.full(n, mass), age, mh, **kwargs)


@pytest.fixture(scope="session")
def mock_snapshot():
    return build_mock_snapshot()


@pytest.fixture(scope="session")
def mock_snapshot_gas():
    return build_mock_snapshot(with_gas=True)


@pytest.fixture(scope="session")
def iso_grid(tmp_path_factory):
    """The PARSEC subset loaded through IsochroneGrid.load_cache."""
    cfg = Config(isochrone_cache_dir=str(tmp_path_factory.mktemp("iso")))
    return IsochroneGrid(cfg).load_cache(PARSEC_SUBSET)


def _base_config_kwargs(tmp_path):
    return dict(
        isochrone_cache_dir=str(tmp_path / "isochrones"),
        output_path=str(tmp_path / "catalogue.h5"),
        plot_dir=str(tmp_path / "plots"),
        extinction_mode="none",
        random_seed=42,
        mag_limit=12.0,
        mag_limit_band="G_RVS",
        output_coords=list(ALL_COORDS),
    )


@pytest.fixture
def make_config(tmp_path):
    """Config factory with test defaults (no extinction, seed 42,
    G_RVS < 12, all coordinate systems, everything written under tmp_path)."""
    def _make(**kwargs):
        base = _base_config_kwargs(tmp_path)
        base.update(kwargs)
        return Config(**base)
    return _make


@pytest.fixture
def make_pipeline_config(make_config):
    """As make_config, but also places the PARSEC subset where
    ``IsochroneGrid.load_or_download`` looks for its cache, so a full
    ``Pipeline.run`` needs no network."""
    def _make(**kwargs):
        cfg = make_config(**kwargs)
        shutil.copy(PARSEC_SUBSET, IsochroneGrid(cfg)._cache_path())
        return cfg
    return _make


class ConstantExtinction:
    """Extinction calculator stub: the same A_V for every star."""

    def __init__(self, a_v):
        self.a_v = a_v

    def compute(self, l_deg, b_deg, dist_kpc, **kwargs):
        return np.full(len(l_deg), float(self.a_v))


def n_rows(group):
    """Length of the arrays in a catalogue group (0 if empty)."""
    for v in group.values():
        if isinstance(v, np.ndarray):
            return len(v)
    return 0
