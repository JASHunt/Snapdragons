"""Config: defaults, derived observer velocity, validation, summary."""

import numpy as np
import pytest

from snapdragons import Config


def test_defaults():
    c = Config()
    assert c.observer_position == (-8.2, 0.0, 0.02)
    assert c.imf == "kroupa"
    assert c.sampling == "mass_budget"
    assert c.mag_limit == 16.0 and c.mag_limit_band == "G_RVS"
    assert c.gaia_release == "dr3"
    assert c.extinction_mode == "mwdust"
    assert c.apply_tform_correction is True
    assert c.unresolved_binaries is False
    assert c.phase_space_smoothing is False
    assert c.selection_function == "none"
    assert c.output_coords == ["equatorial"]
    assert c.rotation == "auto"


def test_observer_velocity_is_left_to_the_frame():
    # it depends on which way the disc turns, which is only known once the
    # snapshot is loaded: see test_frame.py
    assert Config().observer_velocity is None


def test_rotation_option():
    assert Config().rotation == "auto"
    assert Config(rotation="Clockwise").rotation == "clockwise"
    assert Config(rotation="counterclockwise").rotation == "counterclockwise"
    with pytest.raises(ValueError):
        Config(rotation="anticlockwise")


def test_frame_shortcut(mock_snapshot):
    frame = Config().frame(mock_snapshot)
    assert frame.rotation == "clockwise"
    assert frame.obs_vel == pytest.approx((11.1, 232.8 + 12.24, 7.25))
    assert Config(rotation="counterclockwise").frame().north_sign == -1


def test_explicit_observer_velocity_is_kept():
    c = Config(observer_velocity=(10, 200, 5))
    assert c.observer_velocity == (10.0, 200.0, 5.0)
    assert all(isinstance(v, float) for v in c.observer_velocity)
    assert all(isinstance(v, float) for v in Config(observer_position=(-8, 0, 0)).observer_position)


def test_output_coords_validation():
    Config(output_coords=["equatorial", "galactic", "cartesian"])
    with pytest.raises(ValueError):
        Config(output_coords=["ecliptic"])


def test_output_coords_default_is_not_shared_between_instances():
    a, b = Config(), Config()
    a.output_coords.append("galactic")
    assert b.output_coords == ["equatorial"]


def test_mag_limit_band():
    assert Config(mag_limit_band="G_RVS").limit_band_column == "G_RVSmag"
    assert Config(mag_limit_band="G").limit_band_column == "Gmag"
    with pytest.raises(ValueError):
        Config(mag_limit_band="RP")


def test_selection_function_validation_and_shift_defaults():
    with pytest.raises(ValueError):
        Config(selection_function="dr2_rvs")
    assert Config(gaia_release="dr3").sf_mag_shift == 0.0
    assert Config(gaia_release="dr2").sf_mag_shift == 0.0
    assert Config(gaia_release="dr4").sf_mag_shift == 2.5
    assert Config(gaia_release="DR5").sf_mag_shift == 2.5
    assert Config(gaia_release="dr4", sf_mag_shift=1.0).sf_mag_shift == 1.0
    assert Config(gaia_release="dr4", sf_mag_shift=0.0).sf_mag_shift == 0.0


def test_print_summary_reports_the_run_settings(capsys):
    Config(mag_limit=14.0, mag_limit_band="G_RVS", gaia_release="dr4",
           selection_function="dr3_rvs", unresolved_binaries=True,
           phase_space_smoothing=True, extinction_mode="simulation",
           apply_tform_correction=False).print_summary()
    out = capsys.readouterr().out
    assert "G_RVS < 14.0" in out
    assert "DR4" in out
    assert "dr3_rvs" in out and "G - 2.5" in out
    assert "f_bin" in out
    assert "Neighbours" in out
    assert "simulation" in out
    assert "tform correction : OFF" in out
    assert "Disc rotation    : auto" in out and "detected" in out
    assert "reported with the Galactic frame" in out


def test_print_summary_minimal(capsys):
    Config(mag_limit_band="G", extinction_mode="mwdust", rotation="clockwise",
           observer_velocity=(10, 240, 7)).print_summary()
    out = capsys.readouterr().out
    assert "Disc rotation    : clockwise" in out and "as set" in out
    assert "(10.00, 240.00, 7.00) km/s  [as given]" in out
    assert "G < 16.0" in out
    assert "Combined19" in out
    assert "tform correction : ON" in out
