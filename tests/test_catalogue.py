"""Catalogue assembly, selection function, HDF5 / FITS output and the reader."""

import h5py
import numpy as np
import pytest

from snapdragons import Config, read_catalogue
from snapdragons import catalogue as catalogue_module
from snapdragons.catalogue import (
    apply_selection_function, build_catalogue, rvs_selection_probability,
    write_catalogue,
)
from snapdragons.coordinates import observed_to_cartesian
from snapdragons.errors import GaiaErrorModel
from snapdragons.population import synthesise_population
from snapdragons.reader import Catalogue, CatalogueData

from conftest import ConstantExtinction, build_mock_snapshot, n_rows

pytest.importorskip("pygaia")


@pytest.fixture(scope="module")
def raw_cat(mock_snapshot, iso_grid, tmp_path_factory):
    """Raw synthesis output: G_RVS < 12, no extinction, seed 42."""
    cfg = Config(isochrone_cache_dir=str(tmp_path_factory.mktemp("iso")),
                 extinction_mode="none", random_seed=42, mag_limit=12.0)
    return synthesise_population(mock_snapshot, iso_grid, cfg)


def build(raw, cfg, errors=True, seed=1):
    model = GaiaErrorModel(cfg) if errors else None
    return build_catalogue(raw, cfg, error_model=model, rng=np.random.default_rng(seed))


# --- frame ---------------------------------------------------------------------

def test_catalogue_uses_the_frame_of_the_synthesis(raw_cat, make_config):
    assert raw_cat.frame.rotation == "clockwise"
    cat = build(raw_cat, make_config(), errors=False)
    assert cat.frame is raw_cat.frame


def test_plain_dict_input_needs_the_rotation_or_a_frame(raw_cat, make_config):
    plain = dict(raw_cat)
    with pytest.raises(ValueError, match="needs the snapshot"):
        build_catalogue(plain, make_config())
    ref = build(raw_cat, make_config(), errors=False)["true"]
    by_setting = build_catalogue(plain, make_config(rotation="clockwise"))["true"]
    by_frame = build_catalogue(plain, make_config(), frame=raw_cat.frame)["true"]
    assert np.array_equal(by_setting["l"], ref["l"]) and np.array_equal(by_frame["l"], ref["l"])


def test_an_explicit_frame_takes_precedence(raw_cat, make_config):
    from snapdragons import GalacticFrame
    other = GalacticFrame(rotation="counterclockwise")
    cat = build_catalogue(raw_cat, make_config(), frame=other)
    ref = build(raw_cat, make_config(), errors=False)["true"]
    assert cat.frame is other
    # the same stars seen with north toward -z: l and b change sign
    assert np.allclose((cat["true"]["l"] + ref["l"]) % 360, 0, atol=1e-7) or \
        np.allclose(np.cos(np.radians(cat["true"]["l"] + ref["l"])), 1.0)
    assert np.allclose(cat["true"]["b"], -ref["b"], atol=1e-3)


# --- structure -----------------------------------------------------------------

def test_no_stars_in_no_catalogue_out(make_config):
    assert build_catalogue(None, make_config()) is None


def test_groups_are_aligned(raw_cat, make_config):
    cat = build(raw_cat, make_config())
    assert set(cat) == {"true", "observed", "uncertainties", "stellar"}
    n = n_rows(cat["true"])
    assert n > 1000
    for group in cat.values():
        for key, arr in group.items():
            assert len(arr) == n, key


@pytest.mark.parametrize("coords,present,absent", [
    (["equatorial"], {"ra", "dec", "parallax", "pmra", "pmdec", "rv"}, {"l", "b", "x", "vx"}),
    (["galactic"], {"l", "b", "mul_cosb", "mub"}, {"ra", "parallax", "x"}),
    (["cartesian"], {"x", "y", "z", "vx", "vy", "vz"}, {"ra", "l"}),
    (["equatorial", "galactic", "cartesian"],
     {"ra", "dec", "parallax", "pmra", "pmdec", "rv", "l", "b", "mul_cosb", "mub", "x", "vz"}, set()),
])
def test_true_columns_follow_output_coords(raw_cat, make_config, coords, present, absent):
    cat = build(raw_cat, make_config(output_coords=coords), errors=False)
    cols = set(cat["true"])
    assert present <= cols
    assert not (absent & cols)
    always = {"dist", "A_V", "E_BV", "Gmag_app", "Gmag_abs", "G_BPmag_app", "G_RPmag_app",
              "G_RVSmag_app", "G_RVSmag_abs", "A_Gmag", "A_G_RVSmag"}
    assert always <= cols


def test_stellar_columns(raw_cat, make_config):
    cat = build(raw_cat, make_config(), errors=False)
    assert {"Mini", "Mass", "logage", "MH", "logL", "logTe", "logg", "label",
            "parent_id"} <= set(cat["stellar"])
    assert "is_binary" not in cat["stellar"]
    assert cat["stellar"]["parent_id"].dtype == np.int64


def test_binary_columns_are_written_when_present(iso_grid, make_config):
    cfg = make_config(unresolved_binaries=True)
    raw = synthesise_population(build_mock_snapshot(n=60), iso_grid, cfg)
    cat = build(raw, cfg, errors=False)
    assert {"is_binary", "q", "Mini_secondary"} <= set(cat["stellar"])


def test_true_values_are_the_synthesised_ones(raw_cat, make_config):
    cfg = make_config()
    cat = build(raw_cat, cfg, errors=False)
    t = cat["true"]
    obs = np.asarray(cfg.observer_position)
    d = np.linalg.norm(raw_cat["pos"] - obs, axis=1)
    assert np.allclose(t["dist"], d)
    assert np.allclose(t["parallax"], 1.0 / d)
    assert np.array_equal(t["x"], raw_cat["pos"][:, 0])
    assert np.array_equal(t["vz"], raw_cat["vel"][:, 2])
    assert np.array_equal(t["Gmag_app"], raw_cat["Gmag_app"])
    assert np.array_equal(t["Gmag_abs"], raw_cat["Gmag"])
    assert np.allclose(t["Gmag_app"] - t["Gmag_abs"], 5 * np.log10(d * 1000) - 5)
    assert np.array_equal(cat["stellar"]["Mini"], raw_cat["Mini"])


def test_extinction_columns(mock_snapshot, iso_grid, make_config):
    cfg = make_config(rv_extinction=3.1)
    raw = synthesise_population(mock_snapshot, iso_grid, cfg, extinction_calc=ConstantExtinction(0.9))
    t = build(raw, cfg, errors=False)["true"]
    assert np.allclose(t["A_V"], 0.9)
    assert np.allclose(t["E_BV"], 0.9 / 3.1)
    assert np.allclose(t["A_Gmag"], 0.9 * 0.789)
    assert np.allclose(t["A_G_BPmag"], 0.9 * 1.002)
    assert np.allclose(t["A_G_RPmag"], 0.9 * 0.589)


# --- magnitude cuts ------------------------------------------------------------

def test_faint_cut_is_reapplied_in_the_configured_band(raw_cat, make_config):
    cat = build(raw_cat, make_config(mag_limit=11.0, mag_limit_band="G_RVS"), errors=False)
    assert cat["true"]["G_RVSmag_app"].max() <= 11.0
    assert n_rows(cat["true"]) == int(np.sum(raw_cat["G_RVSmag_app"] <= 11.0))
    cat = build(raw_cat, make_config(mag_limit=11.5, mag_limit_band="G"), errors=False)
    assert cat["true"]["Gmag_app"].max() <= 11.5
    assert n_rows(cat["true"]) == int(np.sum(raw_cat["Gmag_app"] <= 11.5))


def test_bright_cut_is_in_g(raw_cat, make_config):
    cat = build(raw_cat, make_config(mag_bright_limit=9.5), errors=False)
    assert cat["true"]["Gmag_app"].min() >= 9.5
    assert n_rows(cat["true"]) == int(np.sum(raw_cat["Gmag_app"] >= 9.5))


def test_stars_without_g_rvs_are_dropped_from_a_g_rvs_limited_catalogue(raw_cat, make_config):
    raw = type(raw_cat)(raw_cat)
    raw.frame = raw_cat.frame
    raw["G_RVSmag_app"] = raw_cat["G_RVSmag_app"].copy()
    raw["G_RVSmag_app"][:25] = np.nan
    n_all = len(raw_cat["Mini"])
    assert n_rows(build(raw, make_config(), errors=False)["true"]) == n_all - 25
    # but kept when the limit is in G
    cfg_g = make_config(mag_limit=20.0, mag_limit_band="G")
    assert n_rows(build(raw, cfg_g, errors=False)["true"]) == n_all


def test_smoothed_stars_beyond_the_limit_are_cut(mock_snapshot, iso_grid, make_config):
    cfg = make_config(phase_space_smoothing=True, smoothing_ngb=16)
    raw = synthesise_population(mock_snapshot, iso_grid, cfg)
    assert np.nanmax(raw["G_RVSmag_app"]) > cfg.mag_limit
    cat = build(raw, cfg, errors=False)
    assert cat["true"]["G_RVSmag_app"].max() <= cfg.mag_limit
    assert n_rows(cat["true"]) < len(raw["Mini"])


# --- errors --------------------------------------------------------------------

def test_without_errors_the_observed_groups_are_empty(raw_cat, make_config):
    cat = build(raw_cat, make_config(), errors=False)
    assert cat["observed"] == {} and cat["uncertainties"] == {}
    cfg = make_config(apply_errors=False)
    cat = build_catalogue(raw_cat, cfg, error_model=GaiaErrorModel(cfg),
                          rng=np.random.default_rng(1))
    assert cat["observed"] == {} and cat["uncertainties"] == {}


def test_observed_cartesian_follows_output_coords(raw_cat, make_config):
    with_cart = build(raw_cat, make_config())["observed"]
    assert {"x", "y", "z", "vx", "vy", "vz"} <= set(with_cart)
    without = build(raw_cat, make_config(output_coords=["equatorial", "galactic"]))["observed"]
    assert "x" not in without and "parallax" in without


def test_observed_cartesian_is_the_inversion_of_the_observed_astrometry(raw_cat, make_config):
    cfg = make_config()
    cat = build(raw_cat, cfg)
    o = cat["observed"]
    back = observed_to_cartesian(o["ra"], o["dec"], o["parallax"], o["pmra"], o["pmdec"], o["rv"],
                                 frame=cat.frame)
    for k in ("x", "y", "z", "vx", "vy", "vz"):
        assert np.allclose(o[k], back[k], equal_nan=True)
    # positions exist wherever the parallax is positive; velocities also need an RV
    good = o["parallax"] > 0
    assert np.all(np.isfinite(o["x"][good]))
    assert np.array_equal(np.isfinite(o["vx"]), good & np.isfinite(o["rv"]))


def test_observed_cartesian_is_close_to_the_truth_for_this_bright_nearby_sample(raw_cat, make_config):
    cat = build(raw_cat, make_config())
    t, o = cat["true"], cat["observed"]
    dpos = np.sqrt((o["x"] - t["x"])**2 + (o["y"] - t["y"])**2 + (o["z"] - t["z"])**2)
    dvel = np.sqrt((o["vx"] - t["vx"])**2 + (o["vy"] - t["vy"])**2 + (o["vz"] - t["vz"])**2)
    assert np.nanmedian(dpos) < 0.05          # kpc
    assert np.nanmedian(dvel) < 5.0           # km/s


def test_error_draws_follow_the_rng(raw_cat, make_config):
    a = build(raw_cat, make_config(), seed=3)["observed"]
    b = build(raw_cat, make_config(), seed=3)["observed"]
    c = build(raw_cat, make_config(), seed=4)["observed"]
    assert np.array_equal(a["parallax"], b["parallax"])
    assert not np.array_equal(a["parallax"], c["parallax"])


# --- selection function --------------------------------------------------------

def fake_probability(value, calls=None):
    def _p(l_deg, b_deg, gmag, g_rp, mag_shift=0.0, **kwargs):
        if calls is not None:
            calls.append(dict(l=np.array(l_deg), b=np.array(b_deg), g=np.array(gmag),
                              c=np.array(g_rp), mag_shift=mag_shift))
        return np.full(len(gmag), value) if np.isscalar(value) else value(gmag)
    return _p


def test_selection_function_thins_by_the_probability(raw_cat, make_config, monkeypatch):
    calls = []
    monkeypatch.setattr(catalogue_module, "rvs_selection_probability", fake_probability(0.4, calls))
    n_all = len(raw_cat["Mini"])
    cat = build(raw_cat, make_config(selection_function="dr3_rvs"), errors=False)
    n = n_rows(cat["true"])
    assert n == pytest.approx(0.4 * n_all, abs=5 * np.sqrt(0.24 * n_all))
    assert np.allclose(cat["true"]["p_selection"], 0.4)
    assert n_rows(cat["stellar"]) == n
    assert len(calls) == 1 and len(calls[0]["g"]) == n_all
    assert calls[0]["mag_shift"] == 0.0
    # queried with G and G - G_RP at the stars' Galactic coordinates
    assert np.allclose(calls[0]["g"], raw_cat["Gmag_app"])
    assert np.allclose(calls[0]["c"], raw_cat["Gmag_app"] - raw_cat["G_RPmag_app"])
    assert np.all((calls[0]["l"] >= 0) & (calls[0]["l"] < 360))


def test_selection_function_keeps_all_or_nothing_at_the_extremes(raw_cat, make_config, monkeypatch):
    monkeypatch.setattr(catalogue_module, "rvs_selection_probability", fake_probability(1.0))
    assert n_rows(build(raw_cat, make_config(selection_function="dr3_rvs"), errors=False)["true"]) \
        == len(raw_cat["Mini"])
    monkeypatch.setattr(catalogue_module, "rvs_selection_probability", fake_probability(0.0))
    assert n_rows(build(raw_cat, make_config(selection_function="dr3_rvs"), errors=False)["true"]) == 0


def test_selection_function_magnitude_dependence_is_respected(raw_cat, make_config, monkeypatch):
    monkeypatch.setattr(catalogue_module, "rvs_selection_probability",
                        fake_probability(lambda g: (g < 11.0).astype(float)))
    cat = build(raw_cat, make_config(selection_function="dr3_rvs"), errors=False)
    assert cat["true"]["Gmag_app"].max() < 11.0
    assert n_rows(cat["true"]) == int(np.sum(raw_cat["Gmag_app"] < 11.0))


def test_selection_function_shift_for_deeper_releases(raw_cat, make_config, monkeypatch):
    calls = []
    monkeypatch.setattr(catalogue_module, "rvs_selection_probability", fake_probability(1.0, calls))
    build(raw_cat, make_config(selection_function="dr3_rvs", gaia_release="dr4"), errors=False)
    build(raw_cat, make_config(selection_function="dr3_rvs", gaia_release="dr4", sf_mag_shift=1.5),
          errors=False)
    assert [c["mag_shift"] for c in calls] == [2.5, 1.5]


def test_selection_function_works_without_galactic_output(raw_cat, make_config, monkeypatch):
    calls = []
    monkeypatch.setattr(catalogue_module, "rvs_selection_probability", fake_probability(1.0, calls))
    build(raw_cat, make_config(selection_function="dr3_rvs", output_coords=["cartesian"]), errors=False)
    ref = build(raw_cat, make_config(output_coords=["galactic"]), errors=False)["true"]
    assert np.allclose(calls[0]["l"], ref["l"]) and np.allclose(calls[0]["b"], ref["b"])


def test_selection_function_then_errors_stay_aligned(raw_cat, make_config, monkeypatch):
    monkeypatch.setattr(catalogue_module, "rvs_selection_probability", fake_probability(0.5))
    cat = build(raw_cat, make_config(selection_function="dr3_rvs"))
    n = n_rows(cat["true"])
    assert all(len(v) == n for g in cat.values() for v in g.values())
    assert len(cat["true"]["p_selection"]) == n


def test_selection_function_input_checks(make_config):
    cfg = make_config(selection_function="dr3_rvs")
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="apparent G"):
        apply_selection_function({"true": {"l": np.zeros(2), "b": np.zeros(2)}}, cfg, rng)
    true = {"Gmag_app": np.full(2, 12.0), "G_RPmag_app": np.full(2, 11.0)}
    with pytest.raises(ValueError, match="star_pos"):
        apply_selection_function({"true": true}, cfg, rng)
    empty = {"true": {"Gmag_app": np.array([]), "G_RPmag_app": np.array([])}}
    assert apply_selection_function(empty, cfg, rng) is empty


class FakeSelectionFunction:
    """Stands in for gaiaunlimited's DR3RVSSelectionFunction: records the
    queries and returns 1 for g < 13, NaN (no data) for g > 16, else 0.25."""

    def __init__(self):
        self.calls = []

    def query(self, coords, g=None, c=None):
        self.calls.append(dict(l=coords.l.deg, b=coords.b.deg, g=np.array(g), c=np.array(c)))
        return np.where(g < 13.0, 1.0, np.where(g > 16.0, np.nan, 0.25))


def test_selection_probability_queries_the_selection_function():
    sfn = FakeSelectionFunction()
    l = np.array([10.0, 200.0, 359.0]); b = np.array([-5.0, 40.0, 0.0])
    g = np.array([12.0, 14.0, 17.0]); c = np.array([0.5, 0.8, 1.1])
    p = rvs_selection_probability(l, b, g, c, sfn=sfn)
    assert np.allclose(p, [1.0, 0.25, 0.0])               # cells without data count as 0
    call = sfn.calls[0]
    assert np.allclose(call["l"], l) and np.allclose(call["b"], b)
    assert np.allclose(call["g"], g) and np.allclose(call["c"], c)


def test_selection_probability_magnitude_shift():
    sfn = FakeSelectionFunction()
    g = np.array([14.0, 15.4, 18.0])
    p = rvs_selection_probability(np.zeros(3), np.zeros(3), g, np.full(3, 0.8), mag_shift=2.5, sfn=sfn)
    assert np.allclose(sfn.calls[0]["g"], g - 2.5)         # evaluated at G - shift
    assert np.allclose(sfn.calls[0]["c"], 0.8)             # the colour is not shifted
    assert np.allclose(p, [1.0, 1.0, 0.25])


def test_selection_probability_is_chunked():
    sfn = FakeSelectionFunction()
    n = 2500
    g = np.linspace(10.0, 17.5, n)
    p = rvs_selection_probability(np.full(n, 30.0), np.full(n, 10.0), g, np.full(n, 0.7),
                                  sfn=sfn, chunk=1000)
    assert [len(c["g"]) for c in sfn.calls] == [1000, 1000, 500]
    assert np.array_equal(p, np.where(g < 13.0, 1.0, np.where(g > 16.0, 0.0, 0.25)))
    assert rvs_selection_probability([], [], [], [], sfn=sfn).shape == (0,)


@pytest.mark.external
def test_real_dr3_rvs_selection_function():
    pytest.importorskip("gaiaunlimited")
    l = np.array([30.0, 30.0, 30.0, 200.0])
    b = np.array([20.0, 20.0, 20.0, -40.0])
    g = np.array([10.0, 13.5, 17.0, 12.5])
    c = np.array([0.8, 0.8, 0.8, 0.6])
    p = rvs_selection_probability(l, b, g, c)
    assert p.shape == (4,) and np.all((p >= 0) & (p <= 1))
    assert p[1] > 0.9 and p[3] > 0.9          # G ~ 12-14: nearly complete
    assert p[0] > 0.5                         # bright bins are sparse, so noisier
    assert p[2] < 0.05                        # G = 17 is beyond DR3's RVS
    # the stand-in for a deeper release: G = 16 evaluated as if 13.5
    shifted = rvs_selection_probability(l[:1], b[:1], np.array([16.0]), c[:1], mag_shift=2.5)
    assert shifted[0] == pytest.approx(p[1])


# --- output files and the reader ------------------------------------------------

def test_hdf5_round_trip(raw_cat, make_config):
    cfg = make_config(gaia_release="dr3", selection_function="none")
    cat = build(raw_cat, cfg)
    write_catalogue(cat, cfg)
    back = read_catalogue(cfg.output_path)
    assert isinstance(back, Catalogue)
    assert back.n_stars == len(back) == n_rows(cat["true"])
    for group in ("true", "observed", "uncertainties", "stellar"):
        data = getattr(back, group)
        assert set(data.columns) == set(cat[group])
        for key, arr in cat[group].items():
            assert np.array_equal(getattr(data, key), arr, equal_nan=True), (group, key)


def test_hdf5_metadata(raw_cat, make_config):
    cfg = make_config(gaia_release="dr4", mag_limit=11.5, selection_function="none")
    cat = build(raw_cat, cfg)
    write_catalogue(cat, cfg)
    meta = read_catalogue(cfg.output_path).metadata
    assert meta["gaia_release"] == "dr4"
    assert meta["mag_limit"] == 11.5 and meta["mag_limit_band"] == "G_RVS"
    assert meta["imf"] == "kroupa" and meta["extinction_mode"] == "none"
    assert meta["selection_function"] == "none" and meta["sf_mag_shift"] == 2.5
    assert meta["observer_position"] == pytest.approx(list(cfg.observer_position))
    assert meta["observer_velocity"] == pytest.approx(list(cat.frame.obs_vel))
    assert meta["observer_velocity"] == pytest.approx([11.1, 245.04, 7.25])
    assert meta["rotation"] == "clockwise" and meta["rotation_source"] == "detected"
    assert meta["galactic_north"] == "+z"
    assert np.allclose(meta["frame_axes"], np.eye(3))
    assert list(meta["output_coords"]) == cfg.output_coords
    with h5py.File(cfg.output_path) as h:
        assert h["metadata"].attrs["n_stars"] == h["true/ra"].shape[0]


def test_output_directory_is_created(raw_cat, make_config, tmp_path):
    cfg = make_config(output_path=str(tmp_path / "deep" / "er" / "cat.h5"))
    write_catalogue(build(raw_cat, cfg, errors=False), cfg)
    assert (tmp_path / "deep" / "er" / "cat.h5").exists()


def test_fits_round_trip(raw_cat, make_config, tmp_path):
    cfg = make_config(output_format="fits", output_path=str(tmp_path / "cat.fits"))
    cat = build(raw_cat, cfg)
    write_catalogue(cat, cfg)
    back = read_catalogue(cfg.output_path)
    assert back.n_stars == n_rows(cat["true"])
    assert np.allclose(back.true.ra, cat["true"]["ra"])
    assert np.allclose(back.observed.parallax, cat["observed"]["parallax"])
    assert np.allclose(back.uncertainties.pmra, cat["uncertainties"]["pmra"])
    assert np.allclose(back.stellar.Mini, cat["stellar"]["Mini"])
    assert set(back.true.columns) == set(cat["true"])
    assert set(back.observed.columns) == set(cat["observed"])
    assert set(back.uncertainties.columns) == set(cat["uncertainties"])
    assert set(back.stellar.columns) == set(cat["stellar"])


def test_unknown_output_format(raw_cat, make_config):
    cfg = make_config(output_format="parquet")
    with pytest.raises(ValueError, match="Unknown output format"):
        write_catalogue(build(raw_cat, cfg, errors=False), cfg)


def test_reader_interface(raw_cat, make_config):
    cfg = make_config()
    write_catalogue(build(raw_cat, cfg), cfg)
    cat = read_catalogue(cfg.output_path)
    assert "stars" in repr(cat) and "dr3" in repr(cat)
    assert "ra" in cat.true and "nonexistent" not in cat.true
    with pytest.raises(AttributeError, match="No column"):
        cat.true.nonexistent
    assert isinstance(cat.true, CatalogueData)
    assert "columns" in repr(cat.true)
    assert set(cat.true.to_dict()) == set(cat.true.columns)


def test_reader_dataframes(raw_cat, make_config):
    pytest.importorskip("pandas")
    cfg = make_config()
    write_catalogue(build(raw_cat, cfg), cfg)
    cat = read_catalogue(cfg.output_path)
    df = cat.to_dataframe()
    assert len(df) == cat.n_stars
    assert {"true_ra", "observed_parallax", "Mini"} <= set(df.columns)
    assert not any(c.startswith("uncertainties_") for c in df.columns)
    df_unc = cat.to_dataframe(include=("uncertainties",))
    assert "uncertainties_parallax" in df_unc.columns
    assert list(cat.stellar.to_dataframe().columns) == cat.stellar.columns


def test_reader_falls_back_on_content_for_unknown_extensions(raw_cat, make_config, tmp_path):
    cfg = make_config(output_path=str(tmp_path / "catalogue.dat"))
    cat = build(raw_cat, cfg, errors=False)
    write_catalogue(cat, cfg)
    assert read_catalogue(cfg.output_path).n_stars == n_rows(cat["true"])


def test_reader_on_a_catalogue_without_errors(raw_cat, make_config):
    cfg = make_config()
    write_catalogue(build(raw_cat, cfg, errors=False), cfg)
    cat = read_catalogue(cfg.output_path)
    assert cat.observed.columns == [] and cat.uncertainties.columns == []
    assert cat.n_stars > 0
