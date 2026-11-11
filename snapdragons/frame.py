"""The Galactic frame of the mock observer.

The Milky Way, seen from its north Galactic pole, rotates clockwise: with
the Sun on the -x axis of a right-handed Galactocentric frame whose z axis
points north, the Sun moves toward +y, which is Galactic longitude l = 90.

A simulation can rotate either way about its z axis, and the observer can
be put anywhere in its plane.  This module works out, for any such input,
the frame in which the mock sky has the Milky Way's conventions:

* l = 0 points from the observer to the galactic centre (the origin of
  the simulation), projected on the disc plane;
* Galactic north is the side from which the disc is seen to turn
  clockwise: the simulation's +z for a clockwise disc, its -z for a
  counter-clockwise one;
* l = 90 is then the direction of rotation at the observer.

Nothing in the simulation is moved or mirrored to achieve this.  The
observer sits where it was put, in simulation coordinates, and only the
directions in which it measures longitude and latitude are chosen.  The
Cartesian columns of the catalogue stay in simulation coordinates.

The sense of rotation is detected from the angular momentum of the star
particles (``rotation="auto"``, the default) or set by hand.
"""

import numpy as np

CLOCKWISE = "clockwise"
COUNTERCLOCKWISE = "counterclockwise"
ROTATION_CHOICES = ("auto", CLOCKWISE, COUNTERCLOCKWISE)

# below this share of co-rotating angular momentum the sense is called ambiguous
WEAK_ROTATION = 0.5
# above this tilt of the disc from the xy plane the frame warns [deg]
TILT_WARNING = 5.0
# above this fractional difference between v_circular and the rotation of the
# stars at the observer's radius the frame warns (asymmetric drift is ~5%)
VCIRC_WARNING = 0.15


class WithFrame(dict):
    """A dict (of catalogue arrays or groups) that remembers the frame it
    was built in, as ``.frame``."""

    frame = None


# --- rotation of the disc ---------------------------------------------------

def angular_momentum(pos, vel, mass, r_max=None):
    """Total angular momentum vector of the particles (within r_max)."""
    pos = np.asarray(pos, dtype=np.float64)
    vel = np.asarray(vel, dtype=np.float64)
    mass = np.asarray(mass, dtype=np.float64)
    if r_max is not None:
        keep = np.sum(pos**2, axis=1) < r_max**2
        if np.any(keep):
            pos, vel, mass = pos[keep], vel[keep], mass[keep]
    return np.sum(mass[:, None] * np.cross(pos, vel), axis=0)


def detect_rotation(pos, vel, mass, r_obs=8.2):
    """Sense of rotation of a disc about the z axis, from its stars.

    Uses the stars inside a cylinder of radius 2.5 r_obs and half-height
    r_obs (all stars if none are inside).

    Returns a dict:
      sense        "clockwise" or "counterclockwise", seen from +z
      lz_fraction  |sum L_z| / sum |L_z|: 1 if every star turns the same way
      tilt_deg     angle between the angular momentum and the z axis
      n_used, r_max
      v_rot        mass-weighted mean rotation speed of the stars within
                   10 per cent of r_obs in radius [km/s], NaN if there are none
      n_annulus
    """
    pos = np.asarray(pos, dtype=np.float64)
    vel = np.asarray(vel, dtype=np.float64)
    mass = np.asarray(mass, dtype=np.float64)
    scale = max(float(r_obs), 1.0)
    R = np.hypot(pos[:, 0], pos[:, 1])
    use = (R < 2.5 * scale) & (np.abs(pos[:, 2]) < scale)
    if not np.any(use):
        use = np.ones(len(R), dtype=bool)

    lz = mass * (pos[:, 0] * vel[:, 1] - pos[:, 1] * vel[:, 0])
    L = angular_momentum(pos[use], vel[use], mass[use])
    Lz = float(np.sum(lz[use]))
    denom = float(np.sum(np.abs(lz[use])))
    Lnorm = float(np.linalg.norm(L))

    ann = use & (np.abs(R - r_obs) < 0.1 * scale) & (R > 0)
    if np.any(ann):
        vphi = (pos[ann, 0] * vel[ann, 1] - pos[ann, 1] * vel[ann, 0]) / R[ann]
        v_rot = abs(float(np.sum(mass[ann] * vphi) / np.sum(mass[ann])))
    else:
        v_rot = np.nan

    return dict(
        sense=CLOCKWISE if Lz < 0 else COUNTERCLOCKWISE,
        lz_fraction=abs(Lz) / denom if denom > 0 else 0.0,
        tilt_deg=float(np.degrees(np.arccos(min(1.0, abs(L[2]) / Lnorm)))) if Lnorm > 0 else 0.0,
        n_used=int(np.sum(use)),
        r_max=2.5 * scale,
        v_rot=v_rot,
        n_annulus=int(np.sum(ann)),
    )


def levelling_rotation(L):
    """Smallest rotation that puts the angular momentum L along the z axis.

    L is taken to +z or -z, whichever is nearer, so the sense of rotation
    is kept; the rotation is about the line of nodes, so azimuths in the
    plane are not turned.

    Returns (matrix, tilt_deg) with new = matrix @ old.
    """
    L = np.asarray(L, dtype=np.float64)
    norm = np.linalg.norm(L)
    if norm == 0:
        return np.eye(3), 0.0
    lhat = L / norm
    target = np.array([0.0, 0.0, 1.0 if lhat[2] >= 0 else -1.0])
    cos_t = float(np.clip(np.dot(lhat, target), -1.0, 1.0))
    axis = np.cross(lhat, target)
    sin_t = float(np.linalg.norm(axis))
    tilt = float(np.degrees(np.arctan2(sin_t, cos_t)))
    if sin_t < 1e-15:
        return np.eye(3), tilt
    k = axis / sin_t
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + sin_t * K + (1.0 - cos_t) * (K @ K), tilt


# --- the frame ----------------------------------------------------------------

class GalacticFrame:
    """Where the observer is, how it moves, and which way is north.

    Parameters
    ----------
    observer_position : (3,) [kpc], simulation coordinates
    rotation : "clockwise" or "counterclockwise"
        Sense of rotation of the disc seen from the simulation's +z.
    observer_velocity : (3,) [km/s], simulation coordinates, optional
        If None: v_circular along the rotation plus the solar motion
        (U toward the galactic centre, V with the rotation, W to the north).
    solar_motion : (U, V, W) [km/s]
    v_circular : float [km/s]

    Attributes
    ----------
    obs_pos, obs_vel : (3,) arrays, simulation coordinates
    axes : (3, 3) array
        Rows are the Galactic axes at the observer in simulation
        coordinates: toward l = 0 (the galactic centre), toward l = 90
        (the rotation), toward the north Galactic pole.
    north_sign : +1 or -1
        Galactic north is this times the simulation's z axis.
    """

    def __init__(self, observer_position=(-8.2, 0.0, 0.02), rotation=CLOCKWISE,
                 observer_velocity=None, solar_motion=(11.1, 12.24, 7.25),
                 v_circular=232.8):
        if rotation not in (CLOCKWISE, COUNTERCLOCKWISE):
            raise ValueError(f"rotation must be '{CLOCKWISE}' or "
                             f"'{COUNTERCLOCKWISE}', not '{rotation}'")
        self.rotation = rotation
        self.rotation_source = "set"
        self.detection = None
        self.north_sign = 1 if rotation == CLOCKWISE else -1
        self.obs_pos = np.asarray(observer_position, dtype=np.float64)
        self.solar_motion = tuple(float(v) for v in solar_motion)
        self.v_circular = float(v_circular)

        z_hat = np.array([0.0, 0.0, float(self.north_sign)])
        R = float(np.hypot(self.obs_pos[0], self.obs_pos[1]))
        self.at_centre = R < 1e-9
        if self.at_centre:
            x_hat = np.array([1.0, 0.0, 0.0])
        else:
            x_hat = np.array([-self.obs_pos[0] / R, -self.obs_pos[1] / R, 0.0])
        y_hat = np.cross(z_hat, x_hat)
        self.axes = np.array([x_hat, y_hat, z_hat]) + 0.0     # no negative zeros
        x_hat, y_hat, z_hat = self.axes

        if observer_velocity is None:
            U, V, W = self.solar_motion
            self.obs_vel = U * x_hat + (self.v_circular + V) * y_hat + W * z_hat
            self.velocity_source = "derived"
        else:
            self.obs_vel = np.asarray(observer_velocity, dtype=np.float64)
            self.velocity_source = "given"

    # --- construction from a run ------------------------------------------

    @classmethod
    def from_config(cls, config, snapshot=None):
        """Frame for a run: the rotation is detected from the snapshot's
        stars if ``config.rotation`` is "auto", otherwise taken as set."""
        detection = None
        r_obs = float(np.hypot(config.observer_position[0],
                               config.observer_position[1]))
        if snapshot is not None and getattr(snapshot, "n_star", 0) > 0:
            detection = detect_rotation(snapshot.star_pos, snapshot.star_vel,
                                        snapshot.star_mass, r_obs=r_obs)
        setting = str(config.rotation).lower()
        if setting == "auto":
            if detection is None:
                raise ValueError(
                    "rotation='auto' needs the snapshot's star particles to "
                    "detect which way the disc turns.  Pass the snapshot (or "
                    "the frame the synthesis used), or set rotation to "
                    "'clockwise' or 'counterclockwise' (seen from +z).")
            rotation = detection["sense"]
            source = "detected"
        else:
            rotation = setting
            source = "set"
        frame = cls(observer_position=config.observer_position,
                    rotation=rotation,
                    observer_velocity=config.observer_velocity,
                    solar_motion=(config.solar_U, config.solar_V, config.solar_W),
                    v_circular=config.v_circular)
        frame.rotation_source = source
        frame.detection = detection
        return frame

    # --- geometry -----------------------------------------------------------

    @property
    def observer_radius(self):
        return float(np.hypot(self.obs_pos[0], self.obs_pos[1]))

    @property
    def observer_azimuth_deg(self):
        return float(np.degrees(np.arctan2(self.obs_pos[1], self.obs_pos[0])))

    @property
    def height_north(self):
        """Observer's height above the z = 0 plane toward Galactic north [kpc]."""
        return float(self.obs_pos[2] * self.north_sign)

    def to_galactic_cartesian(self, pos, vel=None):
        """Simulation coordinates -> observer-centred Cartesian along the
        Galactic axes (x to l = 0, y to l = 90, z to the north pole).
        Velocities are relative to the observer."""
        d = (np.asarray(pos, dtype=np.float64) - self.obs_pos) @ self.axes.T
        if vel is None:
            return d
        dv = (np.asarray(vel, dtype=np.float64) - self.obs_vel) @ self.axes.T
        return d, dv

    def to_simulation(self, d, dv=None):
        """Inverse of ``to_galactic_cartesian``."""
        pos = np.asarray(d, dtype=np.float64) @ self.axes + self.obs_pos
        if dv is None:
            return pos
        vel = np.asarray(dv, dtype=np.float64) @ self.axes + self.obs_vel
        return pos, vel

    def galactic(self, pos, vel=None):
        """l, b [deg], dist [kpc] (and mul, mub [mas/yr], vr [km/s]) of
        simulation-frame positions (and velocities)."""
        from .coordinates import cartesian_to_galactic
        if vel is None:
            d = self.to_galactic_cartesian(pos)
            return cartesian_to_galactic(d[:, 0], d[:, 1], d[:, 2])
        d, dv = self.to_galactic_cartesian(pos, vel)
        return cartesian_to_galactic(d[:, 0], d[:, 1], d[:, 2],
                                     dv[:, 0], dv[:, 1], dv[:, 2])

    # --- reporting ----------------------------------------------------------

    def warnings(self):
        out = []
        det = self.detection
        if det is not None:
            if self.rotation_source == "set" and det["sense"] != self.rotation:
                out.append(
                    f"rotation is set to '{self.rotation}' but the stars turn "
                    f"{det['sense']} seen from +z.  As set, the mock sky has "
                    "the disc rotating toward l = 270, opposite to the Milky "
                    "Way.  Use rotation='auto' unless this is intended.")
            if det["lz_fraction"] < WEAK_ROTATION:
                out.append(
                    f"net rotation is weak (only {100 * det['lz_fraction']:.0f}% "
                    "of the stars' |L_z| is co-rotating), so the detected "
                    "sense is uncertain.  Set rotation explicitly if you know it.")
            if det["tilt_deg"] > TILT_WARNING:
                out.append(
                    f"the stellar disc is tilted {det['tilt_deg']:.1f} deg "
                    "from the xy plane.  The frame takes the disc to lie in "
                    "the xy plane: level it first (SimulationSnapshot."
                    "level_disc(); automatic when loading through pynbody).")
            if (self.velocity_source == "derived" and np.isfinite(det["v_rot"])
                    and self.v_circular > 0
                    and abs(det["v_rot"] - self.v_circular) > VCIRC_WARNING * self.v_circular):
                diff = det["v_rot"] - self.v_circular
                out.append(
                    f"v_circular is {self.v_circular:g} km/s but the stars at the "
                    f"observer's radius rotate at {det['v_rot']:.1f} km/s "
                    f"({abs(diff):.0f} km/s {'slower' if diff < 0 else 'faster'}).  "
                    "The observer moves at v_circular, so it streams through "
                    "the local disc at about that difference and every proper "
                    "motion and radial velocity carries it.  If the "
                    "simulation's rotation curve is not the Milky Way's, set "
                    "v_circular to its circular velocity at the observer.")
        if self.at_centre:
            out.append("the observer is on the z axis, so the direction of "
                       "l = 0 is arbitrary: the simulation's +x is used.")
        return out

    def report(self):
        """Multi-line description of what the frame is and how it was chosen."""
        ax = self.axes
        north = "+z" if self.north_sign > 0 else "-z"
        det = self.detection
        lines = ["  Galactic frame"]

        how = "set by the rotation option" if self.rotation_source == "set" \
            else "detected"
        if det is not None and self.rotation_source == "detected":
            how = (f"detected: {100 * det['lz_fraction']:.0f}% of the |L_z| of "
                   f"{det['n_used']:,} star particles inside R < "
                   f"{det['r_max']:.1f} kpc; disc tilted "
                   f"{det['tilt_deg']:.2f} deg from the xy plane")
        lines.append(f"    Disc rotation    : {self.rotation} seen from the "
                     f"simulation's +z  ({how})")

        if self.north_sign > 0:
            lines.append(f"    Galactic north   : {north} of the simulation.  "
                         "This is the Milky Way's sense already (clockwise "
                         "from the north pole): the simulation's axes are used "
                         "as they are.")
        else:
            lines.append(f"    Galactic north   : {north} of the simulation.  "
                         "The disc turns the opposite way to the Milky Way "
                         "about +z, so the sky is built with north toward -z; "
                         "seen from there the disc turns clockwise, as the "
                         "Milky Way does.")
            lines.append("                       No particle was moved or "
                         "mirrored: only the directions of longitude and "
                         "latitude were chosen.")

        p = self.obs_pos
        side = "north" if self.height_north >= 0 else "south"
        lines.append(f"    Observer         : (x, y, z) = ({p[0]:.3f}, {p[1]:.3f}, "
                     f"{p[2]:.3f}) kpc in simulation coordinates; R = "
                     f"{self.observer_radius:.3f} kpc, "
                     f"{abs(self.height_north):.3f} kpc {side} of the plane")
        if self.north_sign < 0 and p[2] != 0:
            lines.append(f"                       (z = {p[2]:+.3f} is {side} of "
                         "the plane in this frame; change its sign to put the "
                         "observer on the other side)")
        lines.append(f"    l = 0            : toward the galactic centre, along "
                     f"({ax[0, 0]:+.3f}, {ax[0, 1]:+.3f}, {ax[0, 2]:+.3f}) in "
                     "simulation coordinates")
        lines.append(f"    l = 90           : along the rotation, "
                     f"({ax[1, 0]:+.3f}, {ax[1, 1]:+.3f}, {ax[1, 2]:+.3f})")
        v = self.obs_vel
        U, V, W = self.solar_motion
        if self.velocity_source == "derived":
            lines.append(f"    Observer velocity: ({v[0]:.2f}, {v[1]:.2f}, {v[2]:.2f}) "
                         f"km/s in simulation coordinates = V_circ "
                         f"{self.v_circular:g} along the rotation + solar motion "
                         f"(U, V, W) = ({U:g}, {V:g}, {W:g}) toward the centre / "
                         "with the rotation / to the north")
        else:
            lines.append(f"    Observer velocity: ({v[0]:.2f}, {v[1]:.2f}, {v[2]:.2f}) "
                         "km/s in simulation coordinates, as given")
        if det is not None and np.isfinite(det["v_rot"]):
            lines.append(f"    For reference    : the {det['n_annulus']:,} star "
                         f"particles within 10% of the observer's radius "
                         f"rotate at a mean {det['v_rot']:.1f} km/s "
                         f"(v_circular is {self.v_circular:g})")
        lines.append("    Catalogue        : sky coordinates and proper motions "
                     "are in this frame; Cartesian columns stay in simulation "
                     "coordinates")
        for w in self.warnings():
            lines.append(f"    WARNING: {w}")
        return "\n".join(lines)

    def print_report(self):
        print(self.report())

    def metadata(self):
        """Frame description for the catalogue file's metadata."""
        return dict(
            rotation=self.rotation,
            rotation_source=self.rotation_source,
            galactic_north="+z" if self.north_sign > 0 else "-z",
            frame_axes=self.axes.tolist(),
            observer_position=self.obs_pos.tolist(),
            observer_velocity=self.obs_vel.tolist(),
        )

    def __repr__(self):
        return (f"GalacticFrame(observer={tuple(np.round(self.obs_pos, 3))}, "
                f"rotation={self.rotation!r}, north="
                f"{'+z' if self.north_sign > 0 else '-z'})")
