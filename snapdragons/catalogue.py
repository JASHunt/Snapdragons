"""Catalogue assembly and output.

Collects true positions, applies extinction, computes observed
coordinates, applies magnitude cuts, and writes HDF5 or FITS.
"""

import numpy as np
import h5py
import os

from .coordinates import compute_observables
from .extinction import EXTINCTION_COEFFS


def build_catalogue(raw_cat, config, error_model=None, rng=None,
                    extinction_calc=None):
    """Assemble the final catalogue from raw population synthesis output.

    Extinction is already applied to apparent magnitudes in the raw
    catalogue (computed per particle during synthesis).

    Parameters
    ----------
    raw_cat : dict from synthesise_population
    config : Config
    error_model : GaiaErrorModel
    rng : numpy Generator
    extinction_calc : ignored (kept for backwards compatibility)

    Returns
    -------
    dict with "true", "observed", "uncertainties", "stellar" sub-dicts.
    """
    if raw_cat is None:
        return None

    if rng is None:
        rng = np.random.default_rng(config.random_seed)

    obs_pos = np.asarray(config.observer_position)
    obs_vel = np.asarray(config.observer_velocity)

    coords = compute_observables(
        raw_cat["pos"], raw_cat["vel"],
        obs_pos, obs_vel,
        coord_systems=tuple(config.output_coords),
    )
    dist = coords.pop("_dist")

    true = {}

    if "equatorial" in coords:
        eq = coords["equatorial"]
        true["ra"] = eq["ra"]
        true["dec"] = eq["dec"]
        true["parallax"] = eq["parallax"]
        if "pmra" in eq:
            true["pmra"] = eq["pmra"]
            true["pmdec"] = eq["pmdec"]
            true["rv"] = eq["rv"]

    if "galactic" in coords:
        gal = coords["galactic"]
        true["l"] = gal["l"]
        true["b"] = gal["b"]
        if "mul" in gal:
            true["mul_cosb"] = gal["mul"]
            true["mub"] = gal["mub"]

    if "cartesian" in coords:
        cart = coords["cartesian"]
        for k in ("x", "y", "z", "vx", "vy", "vz"):
            true[k] = cart[k]

    true["dist"] = dist

    for band in ["Gmag", "G_BPmag", "G_RPmag"]:
        if f"{band}_app" in raw_cat:
            true[f"{band}_app"] = raw_cat[f"{band}_app"]
        if band in raw_cat:
            true[f"{band}_abs"] = raw_cat[band]

    a_v = raw_cat.get("A_V", np.zeros(len(dist)))
    true["A_V"] = a_v
    true["E_BV"] = a_v / config.rv_extinction
    for band, coeff in EXTINCTION_COEFFS.items():
        true[f"A_{band}"] = coeff * a_v

    mag_mask = np.ones(len(dist), dtype=bool)
    if "Gmag_app" in true:
        mag_mask &= (true["Gmag_app"] <= config.mag_limit)
        mag_mask &= (true["Gmag_app"] >= config.mag_bright_limit)

    n_before = len(dist)
    n_after = np.sum(mag_mask)
    print(f"  Magnitude cut: {n_before:,} → {n_after:,} stars "
          f"(G < {config.mag_limit})")

    true = {k: v[mag_mask] if isinstance(v, np.ndarray) else v
            for k, v in true.items()}

    stellar = {}
    for key in ["Mini", "Mass", "logage", "MH", "logL", "logTe", "logg",
                "label", "parent_id", "is_binary", "q", "Mini_secondary"]:
        if key in raw_cat:
            arr = raw_cat[key]
            stellar[key] = arr[mag_mask] if isinstance(arr, np.ndarray) else arr

    catalogue = {"true": true, "stellar": stellar}

    if config.apply_errors and error_model is not None:
        catalogue = error_model.apply_errors(catalogue, rng=rng)

        if "cartesian" in config.output_coords:
            obs = catalogue.get("observed", {})
            if "ra" in obs and "parallax" in obs:
                from .coordinates import observed_to_cartesian
                obs_cart = observed_to_cartesian(
                    obs["ra"], obs["dec"], obs["parallax"],
                    obs.get("pmra"), obs.get("pmdec"), obs.get("rv"),
                    np.asarray(config.observer_position),
                    np.asarray(config.observer_velocity),
                )
                for k, v in obs_cart.items():
                    obs[k] = v
                n_neg = int(np.sum(obs["parallax"] <= 0))
                if n_neg > 0:
                    print(f"  {n_neg} stars with non-positive observed parallax "
                          f"(NaN in observed Cartesian)")
    else:
        catalogue["observed"] = {}
        catalogue["uncertainties"] = {}

    return catalogue


def write_hdf5(catalogue, filepath, config):
    """Write catalogue to HDF5."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

    with h5py.File(filepath, "w") as hf:
        for group_name in ["true", "observed", "uncertainties", "stellar"]:
            if group_name not in catalogue:
                continue
            grp = hf.create_group(group_name)
            for key, val in catalogue[group_name].items():
                if isinstance(val, np.ndarray):
                    grp.create_dataset(key, data=val, compression="gzip",
                                       compression_opts=4)

        meta = hf.create_group("metadata")
        meta.attrs["gaia_release"] = config.gaia_release
        meta.attrs["mag_limit"] = config.mag_limit
        meta.attrs["imf"] = config.imf
        meta.attrs["extinction_mode"] = config.extinction_mode
        meta.attrs["observer_position"] = list(config.observer_position)
        meta.attrs["observer_velocity"] = list(config.observer_velocity)
        meta.attrs["output_coords"] = config.output_coords
        meta.attrs["photometric_system"] = config.photometric_system

        if "true" in catalogue:
            n = 0
            for v in catalogue["true"].values():
                if isinstance(v, np.ndarray):
                    n = len(v)
                    break
            meta.attrs["n_stars"] = n

    print(f"  Wrote {filepath}")


def write_fits(catalogue, filepath, config):
    """Write catalogue to FITS."""
    try:
        from astropy.table import Table
        from astropy.io import fits as pyfits
    except ImportError:
        raise ImportError("astropy is required for FITS output.")

    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

    columns = {}
    for group_name, prefix in [("true", "true_"), ("observed", "obs_"),
                                ("uncertainties", "err_"),
                                ("stellar", "")]:
        if group_name not in catalogue:
            continue
        for key, val in catalogue[group_name].items():
            if isinstance(val, np.ndarray):
                col_name = f"{prefix}{key}" if prefix else key
                columns[col_name] = val

    table = Table(columns)
    table.write(filepath, format="fits", overwrite=True)
    print(f"  Wrote {filepath}")


def write_catalogue(catalogue, config):
    """Write catalogue in the configured format."""
    fmt = config.output_format.lower()
    if fmt == "hdf5":
        write_hdf5(catalogue, config.output_path, config)
    elif fmt == "fits":
        write_fits(catalogue, config.output_path, config)
    else:
        raise ValueError(f"Unknown output format '{fmt}'. "
                         "Choose 'hdf5' or 'fits'.")
