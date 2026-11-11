"""Galactic frame: rotation detection, axes, observer velocity, and the
invariance of the mock sky to how the simulation is set up."""

import astropy.coordinates as coord
import astropy.units as u
import numpy as np
import pytest

from snapdragons import Config, GalacticFrame
from snapdragons.catalogue import build_catalogue
from snapdragons.coordinates import compute_observables
from snapdragons.frame import (
    CLOCKWISE, COUNTERCLOCKWISE, WithFrame, angular_momentum, detect_rotation,
    levelling_rotation,
)
from snapdragons.population import synthesise_population
from snapdragons.simulation import SimulationSnapshot

from conftest import build_mock_snapshot, n_rows

K = 4.740470463533349
SENSES = [CLOCKWISE, COUNTERCLOCKWISE]
AZIMUTHS = [180.0, 0.0, 90.0, -37.0, 123.0]


def rotation_direction(pos, sense):
    """Unit vector of circular motion at pos for a disc turning `sense`."""
    phi = np.arctan2(pos[1], pos[0])
    e_phi = np.array([-np.sin(phi), np.cos(phi), 0.0])      # counter-clockwise
    return e_phi if sense == COUNTERCLOCKWISE else -e_phi


def observer_at(azimuth_deg, R=8.2, z=0.02):
    a = np.radians(azimuth_deg)
    return (R * np.cos(a), R * np.sin(a), z)


def make_disc(n=6000, seed=0, sense=CLOCKWISE, v=220.0, sigma=15.0):
    rng = np.random.default_rng(seed)
    R = rng.uniform(2.0, 14.0, n)
    phi = rng.uniform(0, 2 * np.pi, n)
    pos = np.column_stack([R * np.cos(phi), R * np.sin(phi), rng.normal(0, 0.2, n)])
    sign = 1.0 if sense == COUNTERCLOCKWISE else -1.0
    vel = sign * v * np.column_stack([-np.sin(phi), np.cos(phi), np.zeros(n)])
    vel += rng.normal(0, sigma, (n, 3))
    return pos, vel, np.full(n, 1e4)


def rot_x(angle_deg):
    a = np.radians(angle_deg)
    return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])


def rot_z(angle_deg):
    a = np.radians(angle_deg)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


def transformed(snapshot, matrix):
    """The same galaxy in rotated coordinates (a proper rotation)."""
    kw = {}
    if snapshot.gas_pos is not None:
        kw = dict(gas_pos=snapshot.gas_pos @ matrix.T, gas_density=snapshot.gas_density,
                  gas_metallicity=snapshot.gas_metallicity, gas_smooth=snapshot.gas_smooth)
    return SimulationSnapshot.from_arrays(
        snapshot.star_pos @ matrix.T, snapshot.star_vel @ matrix.T, snapshot.star_mass,
        snapshot.star_age, snapshot.star_metallicity, **kw)


# --- detecting the rotation ---------------------------------------------------

@pytest.mark.parametrize("sense", SENSES)
def test_detects_the_sense_of_rotation(sense):
    det = detect_rotation(*make_disc(sense=sense))
    assert det["sense"] == sense
    assert det["lz_fraction"] > 0.98
    assert det["tilt_deg"] < 1.0
    assert det["v_rot"] == pytest.approx(220.0, abs=3.0)
    assert det["n_used"] == 6000 and det["n_annulus"] > 100


def test_clockwise_means_negative_lz():
    pos, vel, mass = make_disc(sense=CLOCKWISE)
    assert angular_momentum(pos, vel, mass)[2] < 0
    pos, vel, mass = make_disc(sense=COUNTERCLOCKWISE)
    assert angular_momentum(pos, vel, mass)[2] > 0


def test_detection_reports_weak_net_rotation():
    p1, v1, m1 = make_disc(n=3000, seed=1, sense=CLOCKWISE)
    p2, v2, m2 = make_disc(n=3000, seed=2, sense=COUNTERCLOCKWISE)
    det = detect_rotation(np.vstack([p1, p2]), np.vstack([v1, v2]), np.concatenate([m1, m2]))
    assert det["lz_fraction"] < 0.1


def test_detection_reports_the_tilt():
    pos, vel, mass = make_disc()
    M = rot_x(20.0)
    det = detect_rotation(pos @ M.T, vel @ M.T, mass)
    assert det["tilt_deg"] == pytest.approx(20.0, abs=0.5)
    assert det["sense"] == CLOCKWISE


def test_detection_follows_the_mass():
    p1, v1, m1 = make_disc(n=3000, seed=1, sense=CLOCKWISE)
    p2, v2, m2 = make_disc(n=3000, seed=2, sense=COUNTERCLOCKWISE)
    det = detect_rotation(np.vstack([p1, p2]), np.vstack([v1, v2]),
                          np.concatenate([m1, 5 * m2]))
    assert det["sense"] == COUNTERCLOCKWISE


def test_detection_falls_back_to_all_stars():
    pos, vel, mass = make_disc()
    det = detect_rotation(100.0 * pos, vel, mass, r_obs=8.2)      # nothing near the observer
    assert det["sense"] == CLOCKWISE and det["n_used"] == 6000
    assert np.isnan(det["v_rot"])


# --- levelling ------------------------------------------------------------------

def test_levelling_leaves_an_aligned_disc_alone_and_keeps_its_sense():
    for L in ([0, 0, 5.0], [0, 0, -5.0]):
        M, tilt = levelling_rotation(L)
        assert np.allclose(M, np.eye(3)) and tilt == pytest.approx(0.0, abs=1e-12)
    M, tilt = levelling_rotation([0, 0, 0])
    assert np.allclose(M, np.eye(3)) and tilt == 0.0


@pytest.mark.parametrize("L", [[0.3, -0.2, 1.0], [0.3, -0.2, -1.0], [1.0, 0.1, 0.2], [-0.5, 2.0, -0.4]])
def test_levelling_is_the_smallest_proper_rotation(L):
    L = np.array(L, dtype=float)
    M, tilt = levelling_rotation(L)
    assert np.allclose(M @ M.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(M) == pytest.approx(1.0)
    target = np.array([0, 0, np.sign(L[2])])
    assert np.allclose(M @ (L / np.linalg.norm(L)), target, atol=1e-12)
    assert tilt == pytest.approx(np.degrees(np.arccos(abs(L[2]) / np.linalg.norm(L))))
    # the line of nodes is the rotation axis: points on it do not move
    nodes = np.cross(L, [0, 0, 1.0])
    assert np.allclose(M @ nodes, nodes, atol=1e-12)
    # and the rotation angle is the tilt, no more
    assert np.degrees(np.arccos((np.trace(M) - 1) / 2)) == pytest.approx(tilt)


# --- axes -------------------------------------------------------------------------

def test_default_frame_is_the_standard_galactocentric_one():
    f = GalacticFrame()
    assert np.allclose(f.obs_pos, (-8.2, 0.0, 0.02))
    assert f.rotation == CLOCKWISE and f.north_sign == 1
    assert np.allclose(f.axes, np.eye(3))                 # x, y, z of the simulation
    assert np.allclose(f.obs_vel, (11.1, 232.8 + 12.24, 7.25))
    assert f.height_north == pytest.approx(0.02)
    assert f.observer_radius == pytest.approx(8.2)
    assert f.observer_azimuth_deg == pytest.approx(180.0)


def test_counterclockwise_disc_puts_north_at_minus_z():
    f = GalacticFrame(rotation=COUNTERCLOCKWISE)
    assert f.north_sign == -1
    assert np.allclose(f.axes, [[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    assert np.allclose(f.obs_vel, (11.1, -(232.8 + 12.24), -7.25))
    assert f.height_north == pytest.approx(-0.02)
    assert np.allclose(f.obs_pos, (-8.2, 0.0, 0.02))      # the observer is not moved


@pytest.mark.parametrize("sense", SENSES)
@pytest.mark.parametrize("azimuth", AZIMUTHS)
def test_axes_are_right_handed_and_follow_the_galaxy(sense, azimuth):
    obs = observer_at(azimuth)
    f = GalacticFrame(observer_position=obs, rotation=sense)
    assert np.allclose(f.axes @ f.axes.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(f.axes) == pytest.approx(1.0)    # a rotation, never a mirror
    # l = 0 toward the centre, l = 90 along the rotation, north where the disc turns clockwise
    assert np.allclose(f.axes[0], -np.array([obs[0], obs[1], 0]) / 8.2)
    assert np.allclose(f.axes[1], rotation_direction(obs, sense))
    assert np.allclose(f.axes[2], [0, 0, 1 if sense == CLOCKWISE else -1])


@pytest.mark.parametrize("sense", SENSES)
@pytest.mark.parametrize("azimuth", AZIMUTHS)
def test_sky_directions(sense, azimuth):
    obs = np.array(observer_at(azimuth, z=0.0))
    f = GalacticFrame(observer_position=obs, rotation=sense)
    north = np.array([0, 0, 1.0 if sense == CLOCKWISE else -1.0])
    points = np.array([
        [0.0, 0.0, 0.0],                                  # the galactic centre
        obs + 0.5 * rotation_direction(obs, sense),       # ahead along the rotation
        obs * 1.2,                                        # the anticentre
        obs - 0.5 * rotation_direction(obs, sense),       # behind
        obs + 0.3 * north,                                # the north pole
    ])
    g = f.galactic(points)
    assert np.allclose(np.cos(np.radians(g["l"][:4])), [1, 0, -1, 0], atol=1e-9)
    assert np.allclose(np.sin(np.radians(g["l"][:4])), [0, 1, 0, -1], atol=1e-9)
    assert np.allclose(g["b"], [0, 0, 0, 0, 90], atol=1e-9)
    assert g["dist"][0] == pytest.approx(8.2)


@pytest.mark.parametrize("sense", SENSES)
@pytest.mark.parametrize("azimuth", AZIMUTHS)
def test_observer_velocity_is_circular_motion_plus_solar_motion(sense, azimuth):
    obs = np.array(observer_at(azimuth))
    f = GalacticFrame(observer_position=obs, rotation=sense, solar_motion=(10.0, 5.0, 7.0),
                      v_circular=220.0)
    to_centre = -np.array([obs[0], obs[1], 0]) / 8.2
    north = np.array([0, 0, 1.0 if sense == CLOCKWISE else -1.0])
    assert f.obs_vel @ to_centre == pytest.approx(10.0)
    assert f.obs_vel @ rotation_direction(obs, sense) == pytest.approx(225.0)
    assert f.obs_vel @ north == pytest.approx(7.0)
    assert np.linalg.norm(f.obs_vel) == pytest.approx(np.sqrt(10**2 + 225**2 + 7**2))
    assert f.velocity_source == "derived"


def test_explicit_observer_velocity_is_used_as_given():
    f = GalacticFrame(observer_velocity=(1.0, 2.0, 3.0), rotation=COUNTERCLOCKWISE)
    assert np.allclose(f.obs_vel, (1.0, 2.0, 3.0))
    assert f.velocity_source == "given"


def test_round_trip_between_simulation_and_galactic_axes(rng):
    f = GalacticFrame(observer_position=observer_at(-37.0), rotation=COUNTERCLOCKWISE)
    pos, vel = rng.normal(0, 5, (50, 3)), rng.normal(0, 100, (50, 3))
    d, dv = f.to_galactic_cartesian(pos, vel)
    back_pos, back_vel = f.to_simulation(d, dv)
    assert np.allclose(back_pos, pos) and np.allclose(back_vel, vel)
    assert np.allclose(np.linalg.norm(d, axis=1), np.linalg.norm(pos - f.obs_pos, axis=1))
    assert np.allclose(f.to_simulation(f.to_galactic_cartesian(pos)), pos)
    # the observer is the origin and at rest in its own frame
    d0, dv0 = f.to_galactic_cartesian(f.obs_pos[None], f.obs_vel[None])
    assert np.allclose(d0, 0) and np.allclose(dv0, 0)


def test_invalid_rotation():
    with pytest.raises(ValueError):
        GalacticFrame(rotation="auto")
    with pytest.raises(ValueError):
        GalacticFrame(rotation="retrograde")


def test_observer_on_the_axis_is_flagged():
    f = GalacticFrame(observer_position=(0.0, 0.0, 1.0))
    assert f.at_centre
    assert np.allclose(f.axes[0], [1, 0, 0])
    assert any("arbitrary" in w for w in f.warnings())


# --- the Milky Way's numbers ------------------------------------------------------

@pytest.mark.parametrize("sense", SENSES)
@pytest.mark.parametrize("azimuth", [180.0, 60.0])
def test_sgr_a_star_moves_as_it_does_on_the_real_sky(sense, azimuth):
    """A source at rest at the galactic centre reflects the Sun's motion:
    mu_l = -(V_c + V) / (K R0) ~ -6.3 mas/yr, v_r = -U, mu_b < 0."""
    f = GalacticFrame(observer_position=observer_at(azimuth, z=0.0), rotation=sense)
    g = compute_observables(np.zeros((1, 3)), np.zeros((1, 3)), f, ("galactic",))["galactic"]
    assert g["mul"][0] == pytest.approx(-(232.8 + 12.24) / (K * 8.2), rel=1e-6)
    assert g["mul"][0] == pytest.approx(-6.3, abs=0.05)
    assert g["vr"][0] == pytest.approx(-11.1)
    assert g["mub"][0] == pytest.approx(-7.25 / (K * 8.2), rel=1e-6)


def test_clockwise_frame_agrees_with_astropy_galactocentric(rng):
    """For a clockwise disc with the Sun on the -x axis the simulation's
    coordinates are astropy's Galactocentric ones.  (astropy also tilts
    the frame by z_sun / R0 and uses the measured position of Sgr A*, so
    the two skies agree to ~0.2 deg, not exactly.)"""
    f = GalacticFrame()
    n = 400
    pos = f.obs_pos + rng.uniform(-4, 4, (n, 3)) * [1, 1, 0.2]
    vel = rng.normal(0, 40, (n, 3)) + [0, 220, 0]
    ours = f.galactic(pos, vel)
    gc = coord.Galactocentric(
        galcen_distance=np.hypot(8.2, 0.02) * u.kpc, z_sun=0.02 * u.kpc,
        galcen_v_sun=coord.CartesianDifferential(f.obs_vel * u.km / u.s))
    sky = coord.SkyCoord(x=pos[:, 0] * u.kpc, y=pos[:, 1] * u.kpc, z=pos[:, 2] * u.kpc,
                         v_x=vel[:, 0] * u.km / u.s, v_y=vel[:, 1] * u.km / u.s,
                         v_z=vel[:, 2] * u.km / u.s, frame=gc).transform_to(coord.Galactic())
    mine = coord.SkyCoord(l=ours["l"] * u.deg, b=ours["b"] * u.deg, frame="galactic")
    assert mine.separation(sky).deg.max() < 0.25
    assert np.allclose(ours["dist"], sky.distance.kpc, rtol=1e-6)
    assert np.allclose(ours["vr"], sky.radial_velocity.value, atol=1e-6)
    far = ours["dist"] > 1.0
    assert np.allclose(ours["mul"][far], sky.pm_l_cosb.value[far], atol=0.1)
    assert np.allclose(ours["mub"][far], sky.pm_b.value[far], atol=0.1)
    assert np.allclose(np.hypot(ours["mul"], ours["mub"]),
                       np.hypot(sky.pm_l_cosb.value, sky.pm_b.value), rtol=1e-6)


@pytest.mark.parametrize("sense", SENSES)
def test_local_disc_has_the_milky_ways_mean_proper_motion(sense, iso_grid, make_config):
    """Differential rotation of a flat rotation curve gives
    mu_l = A (cos 2l - 1) / K <= 0: about -3 mas/yr averaged round the sky.
    With the frame the wrong way up it would be +3."""
    snap = build_mock_snapshot(n=600, seed=5, rotation=sense)
    cat = build_catalogue(synthesise_population(snap, iso_grid, make_config()), make_config())
    t = cat["true"]
    assert -4.5 < np.mean(t["mul_cosb"]) < -1.5


# --- building the frame for a run -----------------------------------------------

@pytest.mark.parametrize("sense", SENSES)
def test_auto_detects_from_the_snapshot(sense, make_config):
    snap = build_mock_snapshot(rotation=sense)
    f = GalacticFrame.from_config(make_config(), snap)
    assert f.rotation == sense and f.rotation_source == "detected"
    assert f.detection["sense"] == sense
    assert f.warnings() == []
    assert make_config().frame(snap).rotation == sense


def test_auto_without_a_snapshot_is_an_error(make_config):
    with pytest.raises(ValueError, match="needs the snapshot"):
        GalacticFrame.from_config(make_config())
    with pytest.raises(ValueError, match="needs the snapshot"):
        make_config().frame()


@pytest.mark.parametrize("sense", SENSES)
def test_explicit_rotation_needs_no_snapshot(sense, make_config):
    f = GalacticFrame.from_config(make_config(rotation=sense))
    assert f.rotation == sense and f.rotation_source == "set" and f.detection is None


def test_explicit_rotation_overrides_the_detection_with_a_warning(make_config):
    snap = build_mock_snapshot(rotation=CLOCKWISE)
    f = GalacticFrame.from_config(make_config(rotation="counterclockwise"), snap)
    assert f.rotation == COUNTERCLOCKWISE and f.detection["sense"] == CLOCKWISE
    assert any("l = 270" in w for w in f.warnings())
    assert "WARNING" in f.report()
    # agreeing with the detection is silent
    assert GalacticFrame.from_config(make_config(rotation="clockwise"), snap).warnings() == []


def test_config_values_reach_the_frame(make_config):
    cfg = make_config(observer_position=(0.0, -8.0, 0.1), rotation="clockwise",
                      solar_U=10.0, solar_V=5.0, solar_W=7.0, v_circular=220.0)
    f = GalacticFrame.from_config(cfg)
    assert np.allclose(f.obs_pos, (0.0, -8.0, 0.1))
    assert np.allclose(f.obs_vel, (-225.0, 10.0, 7.0))
    cfg = make_config(rotation="clockwise", observer_velocity=(1, 2, 3))
    assert np.allclose(GalacticFrame.from_config(cfg).obs_vel, (1, 2, 3))


def test_weak_rotation_and_tilt_are_warned_about(make_config):
    p1, v1, m1 = make_disc(n=3000, seed=1, sense=CLOCKWISE)
    p2, v2, m2 = make_disc(n=2900, seed=2, sense=COUNTERCLOCKWISE)
    age, mh = np.full(5900, 5.0), np.zeros(5900)
    mixed = SimulationSnapshot.from_arrays(np.vstack([p1, p2]), np.vstack([v1, v2]),
                                           np.concatenate([m1, m2]), age, mh)
    assert any("weak" in w for w in GalacticFrame.from_config(make_config(), mixed).warnings())
    pos, vel, mass = make_disc()
    M = rot_x(15.0)
    tilted = SimulationSnapshot.from_arrays(pos @ M.T, vel @ M.T, mass, np.full(6000, 5.0),
                                            np.zeros(6000))
    w = GalacticFrame.from_config(make_config(), tilted).warnings()
    assert any("tilted 15.0 deg" in x and "level_disc" in x for x in w)


def test_circular_velocity_mismatch_is_warned_about(make_config):
    pos, vel, mass = make_disc(v=180.0)
    slow = SimulationSnapshot.from_arrays(pos, vel, mass, np.full(6000, 5.0), np.zeros(6000))
    w = GalacticFrame.from_config(make_config(), slow).warnings()
    assert any("v_circular is 232.8 km/s" in x and "slower" in x for x in w)
    # consistent settings are silent, as is a small asymmetric drift
    assert GalacticFrame.from_config(make_config(v_circular=185.0), slow).warnings() == []
    pos, vel, mass = make_disc(v=220.0)
    mw = SimulationSnapshot.from_arrays(pos, vel, mass, np.full(6000, 5.0), np.zeros(6000))
    assert GalacticFrame.from_config(make_config(), mw).warnings() == []
    # a velocity given by hand is the user's business
    cfg = make_config(observer_velocity=(11.1, 245.0, 7.25))
    assert GalacticFrame.from_config(cfg, slow).warnings() == []


# --- the printed report ------------------------------------------------------------

def test_report_for_a_clockwise_disc(make_config):
    text = GalacticFrame.from_config(make_config(), build_mock_snapshot(rotation=CLOCKWISE)).report()
    assert "Disc rotation    : clockwise seen from the simulation's +z" in text
    assert "detected:" in text and "star particles" in text
    assert "Galactic north   : +z of the simulation" in text
    assert "used as they are" in text
    assert "(-8.200, 0.000, 0.020) kpc" in text and "0.020 kpc north of the plane" in text
    assert "l = 0            : toward the galactic centre, along (+1.000, +0.000, +0.000)" in text
    assert "l = 90           : along the rotation, (+0.000, +1.000, +0.000)" in text
    assert "(11.10, 245.04, 7.25) km/s" in text
    assert "For reference" in text
    assert "Cartesian columns stay in simulation coordinates" in text
    assert "WARNING" not in text


def test_report_for_a_counterclockwise_disc(make_config):
    text = GalacticFrame.from_config(make_config(),
                                     build_mock_snapshot(rotation=COUNTERCLOCKWISE)).report()
    assert "Disc rotation    : counterclockwise seen from the simulation's +z" in text
    assert "Galactic north   : -z of the simulation" in text
    assert "opposite way to the Milky Way" in text
    assert "No particle was moved or mirrored" in text
    assert "0.020 kpc south of the plane" in text
    assert "change its sign" in text
    assert "l = 90           : along the rotation, (+0.000, -1.000, +0.000)" in text
    assert "(11.10, -245.04, -7.25) km/s" in text


def test_report_for_set_rotation_and_given_velocity(make_config):
    cfg = make_config(rotation="clockwise", observer_velocity=(10.0, 240.0, 7.0))
    text = GalacticFrame.from_config(cfg).report()
    assert "set by the rotation option" in text
    assert "(10.00, 240.00, 7.00) km/s in simulation coordinates, as given" in text
    assert "For reference" not in text


def test_metadata():
    m = GalacticFrame(rotation=COUNTERCLOCKWISE).metadata()
    assert m["rotation"] == COUNTERCLOCKWISE and m["galactic_north"] == "-z"
    assert np.allclose(m["frame_axes"], [[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    assert np.allclose(m["observer_velocity"], (11.1, -245.04, -7.25))
    assert "GalacticFrame" in repr(GalacticFrame())


def test_with_frame_is_a_dict():
    d = WithFrame(a=1)
    assert d == {"a": 1} and d.frame is None


# --- the sky does not depend on how the simulation is set up ----------------------

SKY = ("l", "b", "ra", "dec", "parallax", "pmra", "pmdec", "rv", "mul_cosb", "mub",
       "dist", "Gmag_app", "G_RVSmag_app", "A_V")


def observe(snapshot, cfg, iso_grid):
    from snapdragons.errors import GaiaErrorModel
    raw = synthesise_population(snapshot, iso_grid, cfg)
    return build_catalogue(raw, cfg, error_model=GaiaErrorModel(cfg),
                           rng=np.random.default_rng(3))


def star_keys(cat):
    return np.rec.fromarrays([cat["stellar"]["parent_id"], cat["stellar"]["Mini"]],
                             names="parent,mini")


def assert_same_sky(a, b, exact=True):
    """Same stars with the same sky coordinates.  With exact=False a star
    whose magnitude sits on the limit to within rounding may be in one
    catalogue and not the other (the two inputs differ by rounding in the
    last digit of every distance): the rest must still match."""
    ka, kb = star_keys(a), star_keys(b)
    assert len(ka) > 1000
    if exact or (len(ka) == len(kb) and np.array_equal(ka, kb)):
        assert np.array_equal(ka, kb)
        ia = ib = np.arange(len(ka))
        for k in ("ra", "dec", "parallax", "pmra", "pmdec", "rv"):
            assert np.allclose(a["observed"][k], b["observed"][k], atol=1e-8, equal_nan=True), k
    else:
        _, ia, ib = np.intersect1d(ka, kb, return_indices=True)
        assert len(ia) >= 0.999 * max(len(ka), len(kb))
    for k in SKY:
        assert np.allclose(a["true"][k][ia], b["true"][k][ib], atol=1e-8), k
    return ia, ib


def test_same_sky_whichever_way_the_simulation_rotates(iso_grid, make_config):
    """The same galaxy written with its z axis the other way up (so that it
    rotates the other way about +z), observed from the same place, gives
    the same catalogue star for star."""
    pytest.importorskip("pygaia")
    snap = build_mock_snapshot(n=300, seed=7, rotation=CLOCKWISE)
    flip = rot_x(180.0)                                   # y -> -y, z -> -z
    obs = np.array([-8.2, 0.0, 0.02])
    a = observe(snap, make_config(observer_position=tuple(obs)), iso_grid)
    b = observe(transformed(snap, flip), make_config(observer_position=tuple(flip @ obs)), iso_grid)
    assert a.frame.rotation == CLOCKWISE and b.frame.rotation == COUNTERCLOCKWISE
    assert_same_sky(a, b)
    # Cartesian columns are each simulation's own
    assert np.allclose(b["true"]["x"], a["true"]["x"])
    assert np.allclose(b["true"]["y"], -a["true"]["y"])
    assert np.allclose(b["true"]["z"], -a["true"]["z"])
    assert np.allclose(b["true"]["vy"], -a["true"]["vy"])
    assert np.allclose(b["observed"]["y"], -a["observed"]["y"], equal_nan=True)


@pytest.mark.parametrize("sense", SENSES)
@pytest.mark.parametrize("angle", [90.0, -143.0])
def test_same_sky_wherever_the_observer_is_put(sense, angle, iso_grid, make_config):
    """Turning the simulation and the observer together about z changes
    nothing on the sky."""
    pytest.importorskip("pygaia")
    snap = build_mock_snapshot(n=300, seed=7, rotation=sense)
    turn = rot_z(angle)
    obs = np.array([-8.2, 0.0, 0.02])
    a = observe(snap, make_config(observer_position=tuple(obs)), iso_grid)
    b = observe(transformed(snap, turn), make_config(observer_position=tuple(turn @ obs)), iso_grid)
    ia, ib = assert_same_sky(a, b, exact=False)
    assert np.allclose(np.column_stack([b["true"][k][ib] for k in "xyz"]),
                       np.column_stack([a["true"][k][ia] for k in "xyz"]) @ turn.T)


def test_simulation_extinction_is_frame_independent(iso_grid, make_config):
    from snapdragons.extinction import get_extinction_calculator
    snap = build_mock_snapshot(n=200, seed=7, with_gas=True)
    flip = rot_x(180.0)
    snap2 = transformed(snap, flip)
    obs = np.array([-8.2, 0.0, 0.02])
    out = []
    for s, o in ((snap, obs), (snap2, flip @ obs)):
        cfg = make_config(observer_position=tuple(o), extinction_mode="simulation")
        raw = synthesise_population(s, iso_grid, cfg,
                                    extinction_calc=get_extinction_calculator(cfg, s))
        out.append(raw)
    assert np.allclose(out[0]["A_V"], out[1]["A_V"])
    assert np.median(out[0]["A_V"]) > 0


def test_sky_based_extinction_sees_the_galactic_coordinates(iso_grid, make_config):
    """A dust map is queried at the frame's (l, b): a calculator that
    reddens only l < 180 must redden the same stars either way up."""
    class HalfSky:
        def compute(self, l_deg, b_deg, dist_kpc):
            return np.where(l_deg < 180.0, 1.0, 0.0)

    snap = build_mock_snapshot(n=300, seed=7)
    flip = rot_x(180.0)
    obs = np.array([-8.2, 0.0, 0.02])
    a = synthesise_population(snap, iso_grid, make_config(observer_position=tuple(obs)),
                              extinction_calc=HalfSky())
    b = synthesise_population(transformed(snap, flip), iso_grid,
                              make_config(observer_position=tuple(flip @ obs)),
                              extinction_calc=HalfSky())
    assert np.array_equal(a["A_V"], b["A_V"])
    assert 0.2 < np.mean(a["A_V"]) < 0.8
    # l < 180 is the side the disc rotates toward: +y for the clockwise original
    assert np.all((a["pos"][:, 1] > 0) == (a["A_V"] > 0))
