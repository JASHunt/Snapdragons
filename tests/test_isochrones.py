"""Isochrone grid: cache loading, lookup, interpolation, CMD parsing."""

import os
import shutil

import h5py
import numpy as np
import pytest

from snapdragons import Config, IsochroneGrid
from snapdragons import isochrones as iso_module
from snapdragons.isochrones import REQUIRED_COLUMNS, _turnoff_mass

from conftest import PARSEC_SUBSET, SUBSET_LOGAGES, SUBSET_MHS


# --- the shipped subset --------------------------------------------------------

def test_subset_grid_points(iso_grid):
    pts = iso_grid.grid_points
    assert pts.shape == (12, 2)
    expected = {(la, mh) for la in SUBSET_LOGAGES for mh in SUBSET_MHS}
    assert {tuple(p) for p in pts} == expected


def test_cells_have_the_required_columns_sorted_by_mass(iso_grid):
    for i in range(len(iso_grid.grid_points)):
        cell = iso_grid.cell(i)
        for col in REQUIRED_COLUMNS:
            assert col in cell
        n = len(cell["Mini"])
        assert n > 50
        assert all(len(v) == n for v in cell.values())
        assert np.all(np.diff(cell["Mini"]) >= 0)
        la, mh = iso_grid.grid_points[i]
        assert np.allclose(cell["logAge"], la, atol=1e-3)
        assert np.allclose(cell["MH"], mh, atol=1e-3)


def test_subset_is_physically_sensible(iso_grid):
    young = iso_grid.cell(iso_grid.cell_of(8.0, 0.0)[0])
    old = iso_grid.cell(iso_grid.cell_of(10.0, 0.0)[0])
    # older populations have lost their massive stars
    assert old["Mini"].max() < 1.2 < 4.0 < young["Mini"].max()
    # a solar-metallicity 0.8 Msun dwarf at 10 Gyr: M_G ~ 6
    props = iso_grid.interpolate_stars(10.0, 0.0, np.array([0.8]))
    assert 5.0 < props["Gmag"][0] < 7.0
    assert 3.6 < props["logTe"][0] < 3.8
    assert 4.3 < props["logg"][0] < 4.8


# --- lookup --------------------------------------------------------------------

def test_cell_of_finds_the_nearest_cell(iso_grid):
    assert tuple(iso_grid.grid_points[iso_grid.cell_of(9.02, -0.48)[0]]) == (9.0, -0.5)
    assert tuple(iso_grid.grid_points[iso_grid.cell_of(9.9, 0.2)[0]]) == (10.0, 0.0)
    # out-of-grid values go to the edge cell
    assert tuple(iso_grid.grid_points[iso_grid.cell_of(6.0, -2.5)[0]]) == (8.0, -1.0)


def test_metallicity_is_weighted_five_times_age_in_the_lookup(iso_grid):
    # 0.3 dex in age to the 9.0 cell against 0.4 to 9.7: 9.0 wins
    assert tuple(iso_grid.grid_points[iso_grid.cell_of(9.3, -0.5)[0]]) == (9.0, -0.5)
    # (9.3, -0.3): staying at age 9.0 costs 0.3 + moving 0.2 dex in [M/H] costs 1.0
    assert tuple(iso_grid.grid_points[iso_grid.cell_of(9.3, -0.3)[0]])[1] == -0.5
    # 0.2 in age + 0.05 x 5 in [M/H] to (9.0, 0.0), against 0.2 + 0.45 x 5 to (9.0, -0.5)
    assert tuple(iso_grid.grid_points[iso_grid.cell_of(9.2, -0.05)[0]]) == (9.0, 0.0)


def test_cell_of_is_vectorised_and_matches_get_nearest(iso_grid, rng):
    la = rng.uniform(7.5, 10.3, 200)
    mh = rng.uniform(-1.4, 0.3, 200)
    idx = iso_grid.cell_of(la, mh)
    assert idx.shape == (200,)
    for i in range(0, 200, 17):
        _, key = iso_grid.get_nearest(la[i], mh[i])
        assert key == tuple(iso_grid.grid_points[idx[i]])
        assert iso_grid.cell(idx[i]) is iso_grid.get_nearest(la[i], mh[i])[0]


# --- interpolation -------------------------------------------------------------

def test_interpolation_reproduces_grid_points(iso_grid):
    cell = iso_grid.cell(iso_grid.cell_of(9.0, 0.0)[0])
    mini = cell["Mini"]
    unique = np.concatenate([[True], np.diff(mini) > 0]) & np.concatenate([np.diff(mini) > 0, [True]])
    m = mini[unique][::7]
    props = iso_grid.interpolate_stars(9.0, 0.0, m)
    for col in ("Gmag", "G_BPmag", "G_RPmag", "logTe", "logg", "Mass"):
        assert np.allclose(props[col], cell[col][unique][::7], atol=1e-10)
    assert np.array_equal(props["Mini"], m)
    assert "logAge" not in props and "MH" not in props


def test_interpolation_between_grid_points_is_linear(iso_grid):
    cell = iso_grid.cell(iso_grid.cell_of(9.0, 0.0)[0])
    i = np.searchsorted(cell["Mini"], 0.6)
    m0, m1 = cell["Mini"][i], cell["Mini"][i + 1]
    assert m1 > m0
    props = iso_grid.interpolate_stars(9.0, 0.0, np.array([0.5 * (m0 + m1)]))
    assert props["Gmag"][0] == pytest.approx(0.5 * (cell["Gmag"][i] + cell["Gmag"][i + 1]))


def test_masses_outside_the_isochrone_are_nan(iso_grid):
    cell = iso_grid.cell(iso_grid.cell_of(10.0, 0.0)[0])
    props = iso_grid.interpolate_stars(
        10.0, 0.0, np.array([0.05, 0.5, cell["Mini"].max() + 0.5]))
    assert np.isnan(props["Gmag"][0]) and np.isnan(props["Gmag"][2])   # too light / dead
    assert np.isfinite(props["Gmag"][1])


# --- minimum observable mass ---------------------------------------------------

def test_min_observable_mass_definition(iso_grid):
    cell = iso_grid.cell(iso_grid.cell_of(9.0, 0.0)[0])
    d, lim = 1.0, 14.0
    m = iso_grid.min_observable_mass(9.0, 0.0, d, lim)
    app = cell["Gmag"] + 5 * np.log10(d * 1000) - 5
    assert m == cell["Mini"][app <= lim].min()
    # extinction pushes the limiting mass up
    assert iso_grid.min_observable_mass(9.0, 0.0, d, lim, extinction_G=1.5) > m


def test_min_observable_mass_trends(iso_grid):
    d = np.array([0.1, 0.5, 2.0, 8.0])
    m = [iso_grid.min_observable_mass(9.0, 0.0, di, 15.0) for di in d]
    assert np.all(np.diff(m) >= 0) and m[-1] > m[0]
    lims = [10.0, 13.0, 16.0, 20.0]
    m = [iso_grid.min_observable_mass(9.0, 0.0, 1.0, lim) for lim in lims]
    assert np.all(np.diff(m) <= 0) and m[-1] < m[0]
    assert iso_grid.min_observable_mass(9.0, 0.0, 1000.0, 5.0) == np.inf


# --- cache handling ------------------------------------------------------------

def test_cache_path_depends_on_the_grid_settings(tmp_path):
    base = dict(isochrone_cache_dir=str(tmp_path))
    p0 = IsochroneGrid(Config(**base))._cache_path()
    assert p0 == IsochroneGrid(Config(**base))._cache_path()
    assert p0.startswith(str(tmp_path)) and p0.endswith(".h5")
    for change in (dict(logage_step=0.05), dict(mh_min=-2.5), dict(logage_max=10.0),
                   dict(photometric_system="other")):
        assert IsochroneGrid(Config(**base, **change))._cache_path() != p0
    # settings unrelated to the grid do not invalidate the cache
    assert IsochroneGrid(Config(**base, mag_limit=12.0, imf="salpeter"))._cache_path() == p0


def test_cache_directory_is_created(tmp_path):
    target = tmp_path / "a" / "b"
    IsochroneGrid(Config(isochrone_cache_dir=str(target)))
    assert target.is_dir()


def test_load_or_download_uses_an_existing_cache(tmp_path, monkeypatch):
    grid = IsochroneGrid(Config(isochrone_cache_dir=str(tmp_path)))
    shutil.copy(PARSEC_SUBSET, grid._cache_path())

    def no_network(*a, **k):
        raise AssertionError("tried to download with a cache present")

    monkeypatch.setattr(iso_module, "_http_post", no_network)
    monkeypatch.setattr(iso_module, "_http_get", no_network)
    assert grid.load_or_download() is grid
    assert len(grid.grid_points) == 12


def test_load_cache_ignores_the_config_grid(tmp_path):
    cfg = Config(isochrone_cache_dir=str(tmp_path), logage_min=9.5, logage_max=9.6)
    grid = IsochroneGrid(cfg).load_cache(PARSEC_SUBSET)
    assert len(grid.grid_points) == 12


# --- CMD download path, offline ------------------------------------------------

CMD_TABLE = """# File generated by CMD 3.9 (http://stev.oapd.inaf.it/cmd) on test
# PARSEC release v1.2S
# Zini     MH   logAge Mini        int_IMF         Mass   logL    logTe  logg  label   mbolmag  Gmag    G_BPmag  G_RPmag
  0.0152  {mh:.1f}  9.0  0.10  1.08  0.10  -2.9  3.50  5.3  0  12.0  12.9  14.4  11.7
  0.0152  {mh:.1f}  9.0  0.50  1.60  0.50  -1.4  3.58  4.8  1   8.3   8.6   9.3   7.8
  0.0152  {mh:.1f}  9.0  1.00  2.10  1.00   0.0  3.76  4.4  1   4.7   4.7   5.0   4.2
# isochrone terminated
  0.0152  {mh:.1f}  9.5  0.10  1.08  0.10  -2.9  3.50  5.3  0  12.0  12.9  14.4  11.7
  0.0152  {mh:.1f}  9.5  0.90  2.00  0.90  -0.2  3.74  4.5  1   5.2   5.2   5.5   4.7
#isochrone terminated
"""


def test_parse_cmd_output(tmp_path):
    grid = IsochroneGrid(Config(isochrone_cache_dir=str(tmp_path)))
    df = grid._parse_cmd_output(CMD_TABLE.format(mh=-0.5))
    assert list(df.columns)[:4] == ["Zini", "MH", "logAge", "Mini"]
    assert len(df) == 5
    assert df["Gmag"].tolist() == [12.9, 8.6, 4.7, 12.9, 5.2]
    assert set(df["logAge"]) == {9.0, 9.5}
    assert grid._parse_cmd_output("no table here") is None
    assert grid._parse_cmd_output("# Zini MH logAge Mini int_IMF Mass\n") is None


def test_download_path_without_network(tmp_path, monkeypatch):
    """The CMD round trip with the two HTTP calls replaced: one POST per
    metallicity, the table link followed, the result cached and loaded."""
    cfg = Config(isochrone_cache_dir=str(tmp_path), logage_min=9.0, logage_max=9.5,
                 logage_step=0.5, mh_min=-0.5, mh_max=0.0, mh_step=0.5)
    posts, gets = [], []

    def fake_post(url, data, timeout=120):
        posts.append(data)
        return f'<a href="../tmp/output{len(posts)}.dat">output</a>'

    def fake_get(url, timeout=120):
        gets.append(url)
        return CMD_TABLE.format(mh=float(posts[-1]["isoc_metlow"]))

    monkeypatch.setattr(iso_module, "_http_post", fake_post)
    monkeypatch.setattr(iso_module, "_http_get", fake_get)
    monkeypatch.setattr(iso_module.time, "sleep", lambda s: None)

    grid = IsochroneGrid(cfg).load_or_download()
    assert [p["isoc_metlow"] for p in posts] == ["-0.5", "0.0"]
    assert all(p["isoc_lagelow"] == "9.0" and p["isoc_lageupp"] == "9.5" for p in posts)
    assert all("gaiaEDR3" in p["photsys_file"] for p in posts)
    assert gets == ["https://stev.oapd.inaf.it/tmp/output1.dat",
                    "https://stev.oapd.inaf.it/tmp/output2.dat"]
    assert {tuple(p) for p in grid.grid_points} == {(9.0, -0.5), (9.5, -0.5), (9.0, 0.0), (9.5, 0.0)}
    assert os.path.exists(grid._cache_path())
    with h5py.File(grid._cache_path()) as h:
        assert h.attrs["n_rows"] == 10
        assert "Gmag" in h and "Zini" not in h      # only the columns the code uses

    # a second grid with the same settings reads the cache
    monkeypatch.setattr(iso_module, "_http_post",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("network")))
    again = IsochroneGrid(cfg).load_or_download()
    assert len(again.grid_points) == 4


def test_download_failure_raises_a_clear_error(tmp_path, monkeypatch):
    def broken(*a, **k):
        raise OSError("no route to host")

    monkeypatch.setattr(iso_module, "_http_post", broken)
    monkeypatch.setattr(iso_module.time, "sleep", lambda s: None)
    cfg = Config(isochrone_cache_dir=str(tmp_path), mh_min=0.0, mh_max=0.0)
    with pytest.raises(RuntimeError, match="Failed to download"):
        IsochroneGrid(cfg).load_or_download()
    assert not os.path.exists(IsochroneGrid(cfg)._cache_path())


def test_load_from_files(tmp_path):
    table = tmp_path / "iso.dat"
    lines = [l for l in CMD_TABLE.format(mh=0.0).splitlines() if not l.startswith("#")]
    header = "Zini MH logAge Mini int_IMF Mass logL logTe logg label mbolmag Gmag G_BPmag G_RPmag"
    table.write_text(header + "\n" + "\n".join(lines) + "\n")
    cfg = Config(isochrone_cache_dir=str(tmp_path / "cache"))
    grid = IsochroneGrid(cfg).load_from_files([str(table)])
    assert {tuple(p) for p in grid.grid_points} == {(9.0, 0.0), (9.5, 0.0)}
    assert os.path.exists(grid._cache_path())
    assert grid.interpolate_stars(9.0, 0.0, np.array([0.75]))["Gmag"][0] == pytest.approx(6.65)


# --- synthetic grid ------------------------------------------------------------

def test_synthetic_grid(tmp_path):
    cfg = Config(isochrone_cache_dir=str(tmp_path), logage_min=8.0, logage_max=10.0,
                 logage_step=0.5, mh_min=-1.0, mh_max=0.0, mh_step=0.5)
    grid = IsochroneGrid.generate_synthetic(cfg)
    assert len(grid.grid_points) == 5 * 3
    cell, key = grid.get_nearest(9.0, -0.5)
    assert key == (9.0, -0.5)
    assert np.all(np.diff(cell["Mini"]) > 0)
    assert np.all(np.isfinite(cell["Gmag"]))
    # brighter at the top than at the bottom of the mass range.  (Not
    # monotonic: the two mass-luminosity slopes are not joined at 0.43 Msun,
    # so the synthetic relation steps down there.  Test-only grid.)
    assert cell["Gmag"][-1] < cell["Gmag"][0] - 5.0
    for col in REQUIRED_COLUMNS:
        assert col in cell
    # older cells end at lower mass
    assert grid.get_nearest(10.0, 0.0)[0]["Mini"].max() < grid.get_nearest(8.0, 0.0)[0]["Mini"].max()


def test_turnoff_mass():
    assert _turnoff_mass(10.0) == pytest.approx(1.0)
    assert _turnoff_mass(1.0) > _turnoff_mass(5.0) > _turnoff_mass(12.0)
    assert _turnoff_mass(0.0) == 100.0
    assert _turnoff_mass(1e-9) == 100.0
