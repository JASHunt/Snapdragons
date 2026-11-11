"""Pipeline: end-to-end runs on the mock snapshot, snapshot loading, CLI."""

import os
import sys

import numpy as np
import pytest

from snapdragons import Config, Pipeline, read_catalogue
from snapdragons import pipeline as pipeline_module
from snapdragons.simulation import SimulationSnapshot

from conftest import n_rows

pytest.importorskip("pygaia")


def test_end_to_end(mock_snapshot, make_pipeline_config):
    cfg = make_pipeline_config()
    cat = Pipeline(cfg).run(snapshot=mock_snapshot)
    n = n_rows(cat["true"])
    assert n > 1000
    assert os.path.exists(cfg.output_path)
    back = read_catalogue(cfg.output_path)
    assert back.n_stars == n
    assert back.true.G_RVSmag_app.max() <= cfg.mag_limit
    assert np.all(np.isfinite(back.observed.parallax))
    assert np.all(back.uncertainties.parallax > 0)
    assert back.metadata["n_stars"] == n
    # the stars are where the snapshot's particles are
    assert back.true.x.min() >= mock_snapshot.star_pos[:, 0].min()
    assert back.true.x.max() <= mock_snapshot.star_pos[:, 0].max()


def test_pipeline_accepts_config_keywords(tmp_path):
    p = Pipeline(mag_limit=13.0, extinction_mode="none", output_path=str(tmp_path / "c.h5"))
    assert isinstance(p.config, Config)
    assert p.config.mag_limit == 13.0


def test_no_snapshot_is_an_error(make_pipeline_config):
    with pytest.raises(ValueError, match="snapshot"):
        Pipeline(make_pipeline_config()).run()


def test_nothing_observable_returns_none_and_writes_nothing(mock_snapshot, make_pipeline_config):
    cfg = make_pipeline_config(mag_limit=-5.0)
    assert Pipeline(cfg).run(snapshot=mock_snapshot) is None
    assert not os.path.exists(cfg.output_path)


def test_same_seed_same_catalogue(mock_snapshot, make_pipeline_config):
    a = Pipeline(make_pipeline_config(random_seed=7)).run(snapshot=mock_snapshot)
    b = Pipeline(make_pipeline_config(random_seed=7)).run(snapshot=mock_snapshot)
    c = Pipeline(make_pipeline_config(random_seed=8)).run(snapshot=mock_snapshot)
    for group in ("true", "observed", "uncertainties", "stellar"):
        for key in a[group]:
            assert np.array_equal(a[group][key], b[group][key], equal_nan=True), (group, key)
    assert n_rows(c["true"]) != n_rows(a["true"]) or not np.array_equal(
        c["observed"]["parallax"], a["observed"]["parallax"])


def test_no_errors(mock_snapshot, make_pipeline_config):
    cfg = make_pipeline_config(apply_errors=False)
    cat = Pipeline(cfg).run(snapshot=mock_snapshot)
    assert cat["observed"] == {} and cat["uncertainties"] == {}
    assert read_catalogue(cfg.output_path).observed.columns == []


def test_fits_output(mock_snapshot, make_pipeline_config, tmp_path):
    cfg = make_pipeline_config(output_format="fits", output_path=str(tmp_path / "cat.fits"))
    cat = Pipeline(cfg).run(snapshot=mock_snapshot)
    assert read_catalogue(cfg.output_path).n_stars == n_rows(cat["true"])


def test_simulation_extinction_end_to_end(mock_snapshot_gas, make_pipeline_config):
    clear = Pipeline(make_pipeline_config()).run(snapshot=mock_snapshot_gas)
    dusty = Pipeline(make_pipeline_config(extinction_mode="simulation")).run(snapshot=mock_snapshot_gas)
    a_v = dusty["true"]["A_V"]
    assert np.all(a_v >= 0) and np.median(a_v) > 0.05
    assert np.all(clear["true"]["A_V"] == 0)
    assert n_rows(dusty["true"]) < n_rows(clear["true"])
    assert np.corrcoef(dusty["true"]["dist"], a_v)[0, 1] > 0.7
    half = Pipeline(make_pipeline_config(extinction_mode="simulation", dust_to_gas_ratio=0.5)).run(
        snapshot=mock_snapshot_gas)
    assert np.median(half["true"]["A_V"]) < np.median(a_v)


def test_all_options_together(mock_snapshot_gas, make_pipeline_config):
    cfg = make_pipeline_config(
        extinction_mode="simulation", unresolved_binaries=True, phase_space_smoothing=True,
        smoothing_ngb=16, gaia_release="dr4", mag_limit=13.0, sampling="poisson", fsample=0.5)
    cat = Pipeline(cfg).run(snapshot=mock_snapshot_gas)
    n = n_rows(cat["true"])
    assert n > 500
    assert all(len(v) == n for g in cat.values() for v in g.values())
    assert cat["true"]["G_RVSmag_app"].max() <= 13.0
    assert {"is_binary", "q"} <= set(cat["stellar"])


def test_diagnostic_plots(mock_snapshot_gas, make_pipeline_config):
    pytest.importorskip("matplotlib")
    cfg = make_pipeline_config(make_plots=True, extinction_mode="simulation")
    Pipeline(cfg).run(snapshot=mock_snapshot_gas)
    files = sorted(os.listdir(cfg.plot_dir))
    expected = {"sky_scatter.png", "sky_density.png", "gaia_photometry_sky.png", "cartesian_xy.png",
                "cartesian_xz.png", "cmd.png", "parallax.png", "distance_mag.png",
                "extinction_map.png"}
    assert expected <= set(files)
    assert all(os.path.getsize(os.path.join(cfg.plot_dir, f)) > 2000 for f in files)


def test_plots_without_errors_or_extra_coordinates(mock_snapshot, make_pipeline_config):
    pytest.importorskip("matplotlib")
    cfg = make_pipeline_config(make_plots=True, apply_errors=False, output_coords=["equatorial"])
    Pipeline(cfg).run(snapshot=mock_snapshot)
    files = set(os.listdir(cfg.plot_dir))
    assert {"cmd.png", "parallax.png", "distance_mag.png"} <= files
    assert "cartesian_xy.png" not in files and "sky_density.png" not in files


# --- snapshot loading -----------------------------------------------------------

@pytest.mark.parametrize("mode,expect_gas", [("none", False), ("simulation", True)])
def test_snapshot_is_loaded_with_the_configured_options(mock_snapshot_gas, make_pipeline_config,
                                                       monkeypatch, mode, expect_gas):
    seen = {}

    def fake_from_pynbody(filepath, **kwargs):
        seen.update(filepath=filepath, **kwargs)
        return mock_snapshot_gas

    monkeypatch.setattr(SimulationSnapshot, "from_pynbody", staticmethod(fake_from_pynbody))
    cfg = make_pipeline_config(snapshot_path="run.000100", param_file="run.param",
                               extinction_mode=mode, apply_tform_correction=False)
    Pipeline(cfg).run()
    assert seen["filepath"] == "run.000100"
    assert seen["param_file"] == "run.param"
    assert seen["load_gas"] is expect_gas
    assert seen["apply_tform_correction"] is False


# --- command line ---------------------------------------------------------------

def run_cli(monkeypatch, argv):
    captured = {}

    def fake_run(self, snapshot=None):
        captured["config"] = self.config

    monkeypatch.setattr(Pipeline, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["snapdragons"] + argv)
    pipeline_module.main()
    return captured["config"]


def test_cli_defaults_match_the_config_defaults(monkeypatch):
    cfg = run_cli(monkeypatch, ["snap.000100"])
    ref = Config()
    assert cfg.snapshot_path == "snap.000100"
    for field in ("mag_limit", "mag_limit_band", "gaia_release", "extinction_mode", "mwdust_map",
                  "imf", "sampling", "output_coords", "output_format", "output_path", "fsample",
                  "observer_position", "selection_function", "sf_mag_shift", "apply_errors",
                  "apply_tform_correction", "make_plots", "plot_dir", "random_seed", "param_file"):
        assert getattr(cfg, field) == getattr(ref, field), field


def test_cli_options_reach_the_config(monkeypatch):
    cfg = run_cli(monkeypatch, [
        "snap.000100", "-o", "out/c.fits", "--format", "fits", "--release", "dr4",
        "--param", "p.param", "--mag-limit", "14.5", "--mag-band", "G",
        "--selection-function", "dr3_rvs", "--sf-shift", "1.0", "--extinction", "simulation",
        "--dust-map", "Green19", "--imf", "salpeter", "--sampling", "poisson",
        "--coords", "galactic", "cartesian", "--no-errors", "--no-tform-correction",
        "--plots", "--plot-dir", "figs", "--seed", "11", "--fsample", "0.25",
        "--obs-x", "-8.0", "--obs-y", "1.0", "--obs-z", "0.0"])
    assert cfg.output_path == "out/c.fits" and cfg.output_format == "fits"
    assert cfg.gaia_release == "dr4" and cfg.param_file == "p.param"
    assert cfg.mag_limit == 14.5 and cfg.mag_limit_band == "G"
    assert cfg.selection_function == "dr3_rvs" and cfg.sf_mag_shift == 1.0
    assert cfg.extinction_mode == "simulation" and cfg.mwdust_map == "Green19"
    assert cfg.imf == "salpeter" and cfg.sampling == "poisson"
    assert cfg.output_coords == ["galactic", "cartesian"]
    assert cfg.apply_errors is False and cfg.apply_tform_correction is False
    assert cfg.make_plots is True and cfg.plot_dir == "figs"
    assert cfg.random_seed == 11 and cfg.fsample == 0.25
    assert cfg.observer_position == (-8.0, 1.0, 0.0)


def test_cli_rejects_unknown_choices(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["snapdragons", "snap", "--release", "dr9"])
    with pytest.raises(SystemExit):
        pipeline_module.main()
    monkeypatch.setattr(sys, "argv", ["snapdragons"])
    with pytest.raises(SystemExit):
        pipeline_module.main()
