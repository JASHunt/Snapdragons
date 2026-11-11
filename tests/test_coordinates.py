"""Coordinates: Cartesian -> Galactic -> equatorial and back."""

import astropy.coordinates as coord
import astropy.units as u
import numpy as np
import pytest

from snapdragons import GalacticFrame
from snapdragons.coordinates import (
    cartesian_to_galactic, galactic_to_equatorial, observed_to_cartesian,
    compute_observables,
)

K = 4.740470463533349          # km/s per (mas/yr x kpc)


def random_stars(rng, n=500):
    d = rng.uniform(-3.0, 3.0, (n, 3))
    dv = rng.normal(0.0, 60.0, (n, 3))
    return d, dv


# --- cartesian_to_galactic -------------------------------------------------

def test_axis_directions():
    out = cartesian_to_galactic(np.array([1.0, 0.0, -1.0, 0.0, 0.0]),
                                np.array([0.0, 1.0, 0.0, -1.0, 0.0]),
                                np.array([0.0, 0.0, 0.0, 0.0, 2.0]))
    assert np.allclose(out["l"][:4], [0.0, 90.0, 180.0, 270.0])
    assert np.allclose(out["b"], [0.0, 0.0, 0.0, 0.0, 90.0])
    assert np.allclose(out["dist"], [1.0, 1.0, 1.0, 1.0, 2.0])
    assert set(out) == {"l", "b", "dist"}


def test_longitude_is_in_0_360(rng):
    d, _ = random_stars(rng)
    out = cartesian_to_galactic(d[:, 0], d[:, 1], d[:, 2])
    assert np.all((out["l"] >= 0) & (out["l"] < 360))
    assert np.all(np.abs(out["b"]) <= 90)


def test_proper_motion_units():
    # 1 kpc away along x, moving at K km/s along +y / +z: 1 mas/yr
    one, zero = np.array([1.0]), np.array([0.0])
    out = cartesian_to_galactic(one, zero, zero, zero, np.array([K]), zero)
    assert out["mul"][0] == pytest.approx(1.0, rel=1e-6)
    assert out["mub"][0] == pytest.approx(0.0, abs=1e-12)
    assert out["vr"][0] == pytest.approx(0.0, abs=1e-12)
    out = cartesian_to_galactic(one, zero, zero, zero, zero, np.array([K]))
    assert out["mub"][0] == pytest.approx(1.0, rel=1e-6)
    out = cartesian_to_galactic(one, zero, zero, np.array([30.0]), zero, zero)
    assert out["vr"][0] == pytest.approx(30.0)
    assert out["mul"][0] == pytest.approx(0.0, abs=1e-12)


def test_matches_astropy_galactic_frame(rng):
    """(dx, dy, dz) are astropy's Galactic (u, v, w): same l, b, proper
    motions (mul is mu_l cos b) and radial velocity."""
    d, dv = random_stars(rng)
    out = cartesian_to_galactic(*d.T, *dv.T)
    gal = coord.Galactic(
        u=d[:, 0] * u.kpc, v=d[:, 1] * u.kpc, w=d[:, 2] * u.kpc,
        U=dv[:, 0] * u.km / u.s, V=dv[:, 1] * u.km / u.s, W=dv[:, 2] * u.km / u.s,
        representation_type=coord.CartesianRepresentation,
        differential_type=coord.CartesianDifferential)
    sph = gal.represent_as(coord.SphericalRepresentation,
                           coord.SphericalCosLatDifferential)
    dsph = sph.differentials["s"]
    assert np.allclose(out["l"], sph.lon.wrap_at(360 * u.deg).deg, atol=1e-9)
    assert np.allclose(out["b"], sph.lat.deg, atol=1e-9)
    assert np.allclose(out["dist"], sph.distance.kpc)
    with u.set_enabled_equivalencies(u.dimensionless_angles()):
        assert np.allclose(out["mul"], dsph.d_lon_coslat.to(u.mas / u.yr).value, rtol=1e-6)
        assert np.allclose(out["mub"], dsph.d_lat.to(u.mas / u.yr).value, rtol=1e-6)
    assert np.allclose(out["vr"], dsph.d_distance.to(u.km / u.s).value)


def test_velocity_decomposition_preserves_the_speed(rng):
    d, dv = random_stars(rng)
    out = cartesian_to_galactic(*d.T, *dv.T)
    speed2 = (out["vr"]**2 + (K * out["mul"] * out["dist"])**2
              + (K * out["mub"] * out["dist"])**2)
    assert np.allclose(speed2, np.sum(dv**2, axis=1), rtol=1e-6)


def test_zero_distance_does_not_blow_up():
    out = cartesian_to_galactic(np.zeros(1), np.zeros(1), np.zeros(1),
                                np.ones(1), np.ones(1), np.ones(1))
    assert all(np.all(np.isfinite(v)) for v in out.values())


# --- galactic_to_equatorial ------------------------------------------------

def test_known_directions():
    # Galactic centre and north Galactic pole in ICRS
    out = galactic_to_equatorial(np.array([0.0, 0.0]), np.array([0.0, 90.0]),
                                 np.array([8.0, 1.0]))
    assert out["ra"][0] == pytest.approx(266.405, abs=1e-2)
    assert out["dec"][0] == pytest.approx(-28.936, abs=1e-2)
    assert out["ra"][1] == pytest.approx(192.8595, abs=1e-2)
    assert out["dec"][1] == pytest.approx(27.1283, abs=1e-2)
    assert set(out) == {"ra", "dec", "parallax"}


def test_parallax_is_inverse_distance_in_mas():
    d = np.array([0.1, 1.0, 4.0])
    out = galactic_to_equatorial(np.full(3, 30.0), np.full(3, 10.0), d)
    assert np.allclose(out["parallax"], [10.0, 1.0, 0.25])


def test_rotation_to_icrs_preserves_pm_modulus_and_rv(rng):
    d, dv = random_stars(rng)
    gal = cartesian_to_galactic(*d.T, *dv.T)
    eq = galactic_to_equatorial(gal["l"], gal["b"], gal["dist"],
                                gal["mul"], gal["mub"], gal["vr"])
    assert np.allclose(np.hypot(eq["pmra"], eq["pmdec"]),
                       np.hypot(gal["mul"], gal["mub"]), rtol=1e-8)
    assert np.allclose(eq["rv"], gal["vr"], atol=1e-8)


# --- compute_observables ---------------------------------------------------

FRAMES = [
    GalacticFrame(),
    GalacticFrame(rotation="counterclockwise"),
    GalacticFrame(observer_position=(3.0, -7.0, -0.1), rotation="clockwise"),
    GalacticFrame(observer_position=(5.0, 6.0, 0.3), rotation="counterclockwise"),
]


def test_default_observer_sees_the_galactic_centre_at_l0():
    frame = GalacticFrame()
    out = compute_observables(np.zeros((1, 3)), np.zeros((1, 3)), frame,
                              coord_systems=("galactic",))
    g = out["galactic"]
    assert g["l"][0] == pytest.approx(0.0, abs=1e-9)
    assert g["b"][0] == pytest.approx(-np.degrees(np.arcsin(0.02 / g["dist"][0])))
    assert g["dist"][0] == pytest.approx(np.hypot(8.2, 0.02))


@pytest.mark.parametrize("frame", FRAMES)
def test_comoving_star_has_no_relative_motion(frame):
    pos = frame.obs_pos[None] + [[0.5, 0.3, 0.1]]
    out = compute_observables(pos, frame.obs_vel[None], frame,
                              coord_systems=("galactic", "equatorial"))
    for k in ("mul", "mub", "vr"):
        assert out["galactic"][k][0] == pytest.approx(0.0, abs=1e-9)
    for k in ("pmra", "pmdec", "rv"):
        assert out["equatorial"][k][0] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("systems", [
    ("equatorial",), ("galactic",), ("cartesian",),
    ("equatorial", "galactic"), ("equatorial", "galactic", "cartesian"),
])
def test_requested_systems_and_distance(systems, rng):
    frame = GalacticFrame()
    pos = frame.obs_pos + rng.uniform(-2, 2, (50, 3))
    vel = rng.normal(0, 50, (50, 3))
    out = compute_observables(pos, vel, frame, coord_systems=systems)
    assert set(out) == set(systems) | {"_dist"}
    expected = np.linalg.norm(pos - frame.obs_pos, axis=1)
    assert np.allclose(out["_dist"], expected)
    if "cartesian" in systems:
        assert np.array_equal(out["cartesian"]["x"], pos[:, 0])
        assert np.array_equal(out["cartesian"]["vz"], vel[:, 2])
        assert np.allclose(out["cartesian"]["dist"], expected)


def test_cartesian_output_is_in_simulation_coordinates_whatever_the_frame(rng):
    pos, vel = rng.uniform(-9, 9, (30, 3)), rng.normal(0, 100, (30, 3))
    for frame in FRAMES:
        c = compute_observables(pos, vel, frame, coord_systems=("cartesian",))["cartesian"]
        assert np.array_equal(np.column_stack([c["x"], c["y"], c["z"]]), pos)
        assert np.array_equal(np.column_stack([c["vx"], c["vy"], c["vz"]]), vel)


# --- observed_to_cartesian -------------------------------------------------

@pytest.mark.parametrize("frame", FRAMES)
def test_round_trip_recovers_positions_and_velocities(frame, rng):
    pos = frame.obs_pos + rng.uniform(-4, 4, (300, 3))
    vel = frame.obs_vel + rng.normal(0, 60, (300, 3))
    eq = compute_observables(pos, vel, frame)["equatorial"]
    back = observed_to_cartesian(eq["ra"], eq["dec"], eq["parallax"],
                                 eq["pmra"], eq["pmdec"], eq["rv"], frame=frame)
    got_pos = np.column_stack([back["x"], back["y"], back["z"]])
    got_vel = np.column_stack([back["vx"], back["vy"], back["vz"]])
    assert np.allclose(got_pos, pos, atol=1e-8)
    assert np.allclose(got_vel, vel, atol=1e-6)


def test_round_trip_positions_only(rng):
    # without a frame: observer-centred, along the Galactic axes
    pos = rng.uniform(-4, 4, (20, 3))
    gal = cartesian_to_galactic(*pos.T)
    eq = galactic_to_equatorial(gal["l"], gal["b"], gal["dist"])
    back = observed_to_cartesian(eq["ra"], eq["dec"], eq["parallax"])
    assert np.allclose(np.column_stack([back["x"], back["y"], back["z"]]), pos, atol=1e-8)
    assert np.all(np.isnan(back["vx"]))


def test_non_positive_parallax_gives_nan():
    ra = np.array([10.0, 20.0, 30.0])
    dec = np.array([5.0, -5.0, 0.0])
    plx = np.array([1.0, 0.0, -0.3])
    pm = np.zeros(3)
    for frame in (None, GalacticFrame(rotation="counterclockwise")):
        out = observed_to_cartesian(ra, dec, plx, pm, pm, pm, frame=frame)
        for k in ("x", "y", "z", "vx", "vy", "vz"):
            assert np.isfinite(out[k][0])
            assert np.isnan(out[k][1]) and np.isnan(out[k][2])
    allbad = observed_to_cartesian(ra, dec, -np.ones(3))
    assert all(np.all(np.isnan(v)) for v in allbad.values())
