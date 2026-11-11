"""Coordinate transformations between Cartesian, Galactic, and Equatorial.

All functions operate on arrays for vectorised performance.
Positions in kpc, velocities in km/s, angles in radians (converted to
degrees in the output catalogue).
"""

import numpy as np
from astropy.coordinates import Galactocentric, Galactic, ICRS
import astropy.units as u
import astropy.coordinates as coord


def cartesian_to_galactic(dx, dy, dz, dvx=None, dvy=None, dvz=None):
    """Convert observer-relative Cartesian to Galactic (l, b, d).

    Parameters
    ----------
    dx, dy, dz : array
        Relative position [kpc], observer-centred along the Galactic axes
        (x toward l = 0, y toward l = 90, z toward the north pole): see
        GalacticFrame.to_galactic_cartesian.
    dvx, dvy, dvz : array, optional
        Relative velocity [km/s].

    Returns
    -------
    dict with keys: l, b [deg], dist [kpc],
    and if velocities provided: mul, mub [mas/yr], vr [km/s].
    """
    dist = np.sqrt(dx**2 + dy**2 + dz**2)
    dist = np.maximum(dist, 1e-10)

    b = np.degrees(np.arcsin(dz / dist))
    l = np.degrees(np.arctan2(dy, dx)) % 360.0

    result = {"l": l, "b": b, "dist": dist}

    if dvx is not None:
        vr = (dx * dvx + dy * dvy + dz * dvz) / dist

        dist_xy = np.sqrt(dx**2 + dy**2)
        dist_xy = np.maximum(dist_xy, 1e-10)

        vl = (-dvx * dy + dvy * dx) / dist_xy
        vb = (dvz * dist_xy - dz * (dx * dvx + dy * dvy) / dist_xy) / dist

        kpc_to_km = 3.085677581e16
        yr_to_s = 3.15576e7
        mas_per_rad = 206264806.24709636
        dist_km = dist * kpc_to_km
        mul = (vl / dist_km) * (yr_to_s * mas_per_rad)
        mub = (vb / dist_km) * (yr_to_s * mas_per_rad)

        result.update({"mul": mul, "mub": mub, "vr": vr})

    return result


def galactic_to_equatorial(l_deg, b_deg, dist_kpc,
                           mul=None, mub=None, vr=None):
    """Convert Galactic (l, b, d) to Equatorial (RA, Dec) via astropy.

    Parameters
    ----------
    l_deg, b_deg : array  [degrees]
    dist_kpc : array [kpc]
    mul, mub : array [mas/yr], optional
    vr : array [km/s], optional

    Returns
    -------
    dict with keys: ra, dec [deg], parallax [mas],
    and if PM/RV provided: pmra, pmdec [mas/yr], rv [km/s].
    """
    has_vel = mul is not None

    if has_vel:
        gal = Galactic(
            l=l_deg * u.deg,
            b=b_deg * u.deg,
            distance=dist_kpc * u.kpc,
            pm_l_cosb=mul * u.mas / u.yr,
            pm_b=mub * u.mas / u.yr,
            radial_velocity=vr * u.km / u.s,
        )
    else:
        gal = Galactic(
            l=l_deg * u.deg,
            b=b_deg * u.deg,
            distance=dist_kpc * u.kpc,
        )

    icrs = gal.transform_to(ICRS())

    result = {
        "ra": icrs.ra.deg,
        "dec": icrs.dec.deg,
        "parallax": (1.0 / dist_kpc),  # mas (dist in kpc → parallax in mas)
    }

    if has_vel:
        result.update({
            "pmra": icrs.pm_ra_cosdec.to(u.mas / u.yr).value,
            "pmdec": icrs.pm_dec.to(u.mas / u.yr).value,
            "rv": icrs.radial_velocity.to(u.km / u.s).value,
        })

    return result


def observed_to_cartesian(ra_deg, dec_deg, parallax_mas,
                          pmra=None, pmdec=None, rv=None, frame=None):
    """Convert observed equatorial coordinates back to Cartesian.

    Inverse of the forward chain: ICRS -> Galactic (l, b, d) ->
    observer-centred Cartesian along the Galactic axes -> simulation
    coordinates.  Stars with non-positive observed parallax get NaN.

    Parameters
    ----------
    ra_deg, dec_deg : array [degrees]
    parallax_mas : array [mas]
    pmra, pmdec : array [mas/yr], optional
    rv : array [km/s], optional
    frame : GalacticFrame, optional
        The observer's frame (position, velocity, Galactic axes).  Without
        it the result is observer-centred along the Galactic axes (x to
        l = 0, y to l = 90, z to the north pole) with velocities relative
        to the observer.

    Returns
    -------
    dict with keys x, y, z [kpc], vx, vy, vz [km/s].
    """
    n = len(ra_deg)

    result = {k: np.full(n, np.nan) for k in
              ("x", "y", "z", "vx", "vy", "vz")}

    valid = parallax_mas > 0
    if not np.any(valid):
        return result

    dist_kpc = 1.0 / parallax_mas[valid]

    has_vel = pmra is not None and rv is not None
    if has_vel:
        icrs = ICRS(
            ra=ra_deg[valid] * u.deg,
            dec=dec_deg[valid] * u.deg,
            distance=dist_kpc * u.kpc,
            pm_ra_cosdec=pmra[valid] * u.mas / u.yr,
            pm_dec=pmdec[valid] * u.mas / u.yr,
            radial_velocity=rv[valid] * u.km / u.s,
        )
    else:
        icrs = ICRS(
            ra=ra_deg[valid] * u.deg,
            dec=dec_deg[valid] * u.deg,
            distance=dist_kpc * u.kpc,
        )

    gal = icrs.transform_to(Galactic())
    l_rad = gal.l.rad
    b_rad = gal.b.rad
    d = gal.distance.kpc

    cosb, sinb = np.cos(b_rad), np.sin(b_rad)
    cosl, sinl = np.cos(l_rad), np.sin(l_rad)

    rel = np.column_stack([d * cosb * cosl, d * cosb * sinl, d * sinb])
    rel_v = None

    if has_vel:
        kpc_to_km = 3.085677581e16
        yr_to_s = 3.15576e7
        mas_per_rad = 206264806.24709636

        mul_cosb = gal.pm_l_cosb.to(u.mas / u.yr).value
        mub = gal.pm_b.to(u.mas / u.yr).value
        vr_gal = gal.radial_velocity.to(u.km / u.s).value

        d_km = d * kpc_to_km
        vl = mul_cosb * d_km / (yr_to_s * mas_per_rad)
        vb = mub * d_km / (yr_to_s * mas_per_rad)

        rel_v = np.column_stack([
            vr_gal * cosb * cosl - vl * sinl - vb * sinb * cosl,
            vr_gal * cosb * sinl + vl * cosl - vb * sinb * sinl,
            vr_gal * sinb + vb * cosb,
        ])

    if frame is not None:
        if rel_v is not None:
            pos, vel = frame.to_simulation(rel, rel_v)
        else:
            pos, vel = frame.to_simulation(rel), None
    else:
        pos, vel = rel, rel_v

    for i, k in enumerate(("x", "y", "z")):
        result[k][valid] = pos[:, i]
    if vel is not None:
        for i, k in enumerate(("vx", "vy", "vz")):
            result[k][valid] = vel[:, i]

    return result


def compute_observables(star_pos, star_vel, frame,
                        coord_systems=("equatorial",)):
    """Compute all requested coordinate systems for a set of stars.

    Parameters
    ----------
    star_pos : (N, 3) array  [kpc], simulation coordinates
    star_vel : (N, 3) array  [km/s], simulation coordinates
    frame : GalacticFrame
        The observer's position, velocity and Galactic axes.
    coord_systems : tuple of str

    Returns
    -------
    dict of dicts, keyed by coordinate system name.  Sky coordinates and
    proper motions are in the Galactic frame of the observer; the
    "cartesian" entry is the input, in simulation coordinates.
    """
    star_pos = np.asarray(star_pos)
    star_vel = np.asarray(star_vel)
    rel, rel_v = frame.to_galactic_cartesian(star_pos, star_vel)
    dx, dy, dz = rel[:, 0], rel[:, 1], rel[:, 2]
    dvx, dvy, dvz = rel_v[:, 0], rel_v[:, 1], rel_v[:, 2]

    result = {}
    gal = None

    if "galactic" in coord_systems or "equatorial" in coord_systems:
        gal = cartesian_to_galactic(dx, dy, dz, dvx, dvy, dvz)
        if "galactic" in coord_systems:
            result["galactic"] = gal

    if "equatorial" in coord_systems:
        eq = galactic_to_equatorial(
            gal["l"], gal["b"], gal["dist"],
            gal["mul"], gal["mub"], gal["vr"],
        )
        result["equatorial"] = eq

    dist = gal["dist"] if gal is not None else np.sqrt(dx**2 + dy**2 + dz**2)

    if "cartesian" in coord_systems:
        result["cartesian"] = {
            "x": star_pos[:, 0], "y": star_pos[:, 1], "z": star_pos[:, 2],
            "vx": star_vel[:, 0], "vy": star_vel[:, 1], "vz": star_vel[:, 2],
            "dist": dist,
        }

    result["_dist"] = dist
    return result
