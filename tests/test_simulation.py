"""Simulation snapshot container and the ChaNGa-specific helpers.

``from_pynbody`` itself is not exercised: it needs pynbody and a snapshot
file.  The pieces of it that can go wrong quietly (metallicity units, the
tform remap, the .param link) are tested on stand-ins.
"""

import os

import numpy as np
import pytest

from snapdragons.simulation import (
    SimulationSnapshot, _apply_tform_correction, _link_param_file,
)


class FakeFamily(dict):
    """Stands in for a pynbody family: keyed arrays plus a particle count."""

    def __init__(self, n, **arrays):
        super().__init__(**arrays)
        self._n = n

    def __len__(self):
        return self._n


class FakeSim:
    def __init__(self, tform):
        self.star = {"tform": tform}


def test_from_arrays_stars_only():
    snap = SimulationSnapshot.from_arrays(
        star_pos=[[1, 2, 3], [4, 5, 6]], star_vel=[[0, 0, 0], [1, 1, 1]],
        star_mass=[1e4, 2e4], star_age=[1.0, 5.0], star_metallicity=[0.0, -0.5])
    assert snap.n_star == 2 and snap.n_gas == 0
    assert snap.star_pos.shape == (2, 3) and snap.star_pos.dtype == np.float64
    assert snap.star_mass.dtype == np.float64
    assert snap.gas_pos is None and snap.gas_density is None


def test_from_arrays_with_gas():
    snap = SimulationSnapshot.from_arrays(
        star_pos=np.zeros((1, 3)), star_vel=np.zeros((1, 3)), star_mass=[1.0],
        star_age=[1.0], star_metallicity=[0.0],
        gas_pos=np.zeros((4, 3)), gas_density=np.ones(4),
        gas_metallicity=np.zeros(4), gas_smooth=np.full(4, 0.1))
    assert snap.n_gas == 4
    assert snap.gas_smooth.shape == (4,)


def test_empty_snapshot_defaults():
    snap = SimulationSnapshot()
    assert snap.n_star == 0 and snap.star_pos is None and snap.n_gas == 0


# --- levelling the disc ------------------------------------------------------------

def flat_disc(sense, n=2000, seed=0):
    """An exactly planar disc: its angular momentum is exactly along z."""
    rng = np.random.default_rng(seed)
    R, phi = rng.uniform(1, 10, n), rng.uniform(0, 2 * np.pi, n)
    pos = np.column_stack([R * np.cos(phi), R * np.sin(phi), np.zeros(n)])
    sign = -1.0 if sense == "clockwise" else 1.0
    vel = sign * 200.0 * np.column_stack([-np.sin(phi), np.cos(phi), np.zeros(n)])
    return pos, vel


def tilt_matrix(angle_deg, node_azimuth_deg):
    """Rotation by angle about an axis in the xy plane at the given azimuth."""
    a, p = np.radians(angle_deg), np.radians(node_azimuth_deg)
    k = np.array([np.cos(p), np.sin(p), 0.0])
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * (K @ K)


def snapshot_of(pos, vel, gas_pos=None):
    n = len(pos)
    kw = {}
    if gas_pos is not None:
        ng = len(gas_pos)
        kw = dict(gas_pos=gas_pos, gas_density=np.ones(ng), gas_metallicity=np.zeros(ng),
                  gas_smooth=np.full(ng, 0.1))
    return SimulationSnapshot.from_arrays(pos, vel, np.ones(n), np.ones(n), np.zeros(n), **kw)


@pytest.mark.parametrize("sense", ["clockwise", "counterclockwise"])
@pytest.mark.parametrize("angle,node", [(30.0, 0.0), (70.0, 125.0), (3.0, -40.0)])
def test_level_disc_undoes_a_tilt_exactly(sense, angle, node, capsys):
    """Levelling is the smallest rotation back into the plane, so a disc
    tilted about a line of nodes comes back to its original coordinates:
    same sense of rotation, no turn in azimuth."""
    pos, vel = flat_disc(sense)
    T = tilt_matrix(angle, node)
    gas = pos[:50] * 0.5
    snap = snapshot_of(pos @ T.T, vel @ T.T, gas_pos=gas @ T.T)
    tilt = snap.level_disc()
    assert tilt == pytest.approx(angle, abs=1e-6)
    assert np.allclose(snap.star_pos, pos, atol=1e-9)
    assert np.allclose(snap.star_vel, vel, atol=1e-9)
    assert np.allclose(snap.gas_pos, gas, atol=1e-9)
    out = capsys.readouterr().out
    assert f"tilted {angle:.2f} deg" in out and "levelled" in out
    assert ("clockwise seen from +z" in out) and (("counter-clockwise" in out) == (sense != "clockwise"))


@pytest.mark.parametrize("sense", ["clockwise", "counterclockwise"])
def test_level_disc_leaves_a_level_disc_untouched(sense, capsys):
    pos, vel = flat_disc(sense)
    T = tilt_matrix(0.2, 30.0)                 # below the 0.5 deg threshold
    snap = snapshot_of(pos @ T.T, vel @ T.T)
    before = snap.star_pos.copy()
    tilt = snap.level_disc()
    assert tilt == pytest.approx(0.2, abs=1e-6)
    assert np.array_equal(snap.star_pos, before)
    assert "coordinates left as they are" in capsys.readouterr().out


def test_level_disc_never_turns_a_disc_over():
    # an upside-down (clockwise) disc tilted by 20 deg stays clockwise
    pos, vel = flat_disc("clockwise")
    T = tilt_matrix(20.0, 10.0)
    snap = snapshot_of(pos @ T.T, vel @ T.T)
    snap.level_disc()
    lz = np.sum(snap.star_pos[:, 0] * snap.star_vel[:, 1] - snap.star_pos[:, 1] * snap.star_vel[:, 0])
    assert lz < 0
    assert np.allclose(snap.star_pos[:, 2], 0.0, atol=1e-9)


def test_level_disc_threshold_and_radius_options():
    pos, vel = flat_disc("clockwise")
    T = tilt_matrix(0.2, 30.0)
    snap = snapshot_of(pos @ T.T, vel @ T.T)
    snap.level_disc(min_tilt_deg=0.0)
    assert np.allclose(snap.star_pos, pos, atol=1e-9)
    # only the stars inside r_max define the plane
    outer = np.linalg.norm(pos, axis=1) > 6
    pos2, vel2 = pos.copy(), vel.copy()
    W = tilt_matrix(40.0, 0.0)
    pos2[outer], vel2[outer] = pos[outer] @ W.T, vel[outer] @ W.T    # a warped outer disc
    snap = snapshot_of(pos2, vel2)
    assert snap.level_disc(r_max=5.0) == pytest.approx(0.0, abs=1e-6)


def test_metal_mass_fraction_is_converted_to_dex():
    z = np.array([0.0152, 0.00152, 0.0304])
    mh = SimulationSnapshot._get_metallicity(FakeFamily(3, metals=z))
    assert np.allclose(mh, [0.0, -1.0, np.log10(2.0)])


def test_zero_metallicity_is_floored_not_infinite():
    mh = SimulationSnapshot._get_metallicity(FakeFamily(2, metals=np.array([0.0, 0.0152])))
    assert np.isfinite(mh[0]) and mh[0] == pytest.approx(np.log10(1e-6 / 0.0152))


def test_feh_is_used_when_there_is_no_metals_array():
    feh = np.array([-0.3, 0.1])
    assert np.array_equal(SimulationSnapshot._get_metallicity(FakeFamily(2, feh=feh)), feh)


def test_no_metallicity_information_gives_solar():
    assert np.array_equal(SimulationSnapshot._get_metallicity(FakeFamily(4)), np.zeros(4))


def test_tform_correction_remaps_only_the_flagged_stars(capsys):
    raw = np.array([100.0, 597.0, 1e6 + 10.0, 1e6 + 0.5, 0.39e6])
    sim = FakeSim(raw.copy())
    _apply_tform_correction(sim)
    out = sim.star["tform"]
    assert np.array_equal(out[[0, 1, 4]], raw[[0, 1, 4]])
    assert out[2] == pytest.approx(10.0 * 4.5 + 597.53125)
    assert out[3] == pytest.approx(0.5 * 4.5 + 597.53125)
    assert "2 / 5" in capsys.readouterr().out


def test_tform_correction_is_a_no_op_without_flagged_stars(capsys):
    raw = np.array([1.0, 300.0, 597.0])
    sim = FakeSim(raw.copy())
    _apply_tform_correction(sim)
    assert np.array_equal(sim.star["tform"], raw)
    assert "no particles needed correction" in capsys.readouterr().out


def test_tform_correction_is_continuous_in_formation_order():
    # the remapped stars continue from t = 597.53125 at the simulation's 4.5 x rate
    raw = np.array([1e6, 1e6 + 1.0, 1e6 + 2.0])
    sim = FakeSim(raw.copy())
    _apply_tform_correction(sim)
    assert np.allclose(np.diff(sim.star["tform"]), 4.5)
    assert sim.star["tform"][0] == pytest.approx(597.53125)


def test_param_file_is_linked_under_the_name_pynbody_expects(tmp_path):
    snap = tmp_path / "run.000100"
    snap.write_bytes(b"")
    param = tmp_path / "other.param"
    param.write_text("dKpcUnit = 1\n")
    _link_param_file(str(snap), str(param))
    expected = tmp_path / "run.param"
    assert expected.exists()
    assert expected.read_text() == "dKpcUnit = 1\n"


def test_existing_param_file_is_left_alone(tmp_path):
    snap = tmp_path / "run.000100"
    snap.write_bytes(b"")
    (tmp_path / "run.param").write_text("original\n")
    other = tmp_path / "other.param"
    other.write_text("different\n")
    _link_param_file(str(snap), str(other))
    assert (tmp_path / "run.param").read_text() == "original\n"
    assert not os.path.islink(tmp_path / "run.param")


def test_missing_param_file_warns_and_creates_nothing(tmp_path, capsys):
    snap = tmp_path / "run.000100"
    snap.write_bytes(b"")
    _link_param_file(str(snap), str(tmp_path / "absent.param"))
    assert not (tmp_path / "run.param").exists()
    assert "not found" in capsys.readouterr().out
