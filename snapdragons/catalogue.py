"""Catalogue assembly and output.

Collects true positions, applies extinction, computes observed
coordinates, applies magnitude cuts, and writes HDF5 or FITS.
"""

import numpy as np
import h5py
import os

from .coordinates import compute_observables
from .extinction import EXTINCTION_COEFFS
from .frame import GalacticFrame, WithFrame


def build_catalogue(raw_cat, config, error_model=None, rng=None,
                    extinction_calc=None, frame=None):
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
    frame : GalacticFrame, optional
        The observer's frame.  Defaults to the one the synthesis used
        (``raw_cat.frame``), else one built from the config (which then
        needs an explicit ``rotation``).

    Returns
    -------
    dict with "true", "observed", "uncertainties", "stellar" sub-dicts,
    with the frame attached as ``.frame``.
    """
    if raw_cat is None:
        return None

    if rng is None:
        rng = np.random.default_rng(config.random_seed)

    if frame is None:
        frame = getattr(raw_cat, "frame", None)
    if frame is None:
        frame = GalacticFrame.from_config(config)

    coords = compute_observables(
        raw_cat["pos"], raw_cat["vel"], frame,
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

    for band in ["Gmag", "G_BPmag", "G_RPmag", "G_RVSmag"]:
        if f"{band}_app" in raw_cat:
            true[f"{band}_app"] = raw_cat[f"{band}_app"]
        if band in raw_cat:
            true[f"{band}_abs"] = raw_cat[band]

    a_v = raw_cat.get("A_V", np.zeros(len(dist)))
    true["A_V"] = a_v
    true["E_BV"] = a_v / config.rv_extinction
    for band, coeff in EXTINCTION_COEFFS.items():
        true[f"A_{band}"] = coeff * a_v

    # Faint limit in the configured band, bright limit in G.  The synthesis
    # already applied both before smoothing; after smoothing the distance
    # moduli have changed slightly, so re-apply here.
    lim_col = f"{config.limit_band_column}_app"
    mag_mask = np.ones(len(dist), dtype=bool)
    with np.errstate(invalid="ignore"):
        if lim_col in true:
            mag_mask &= np.isfinite(true[lim_col])
            mag_mask &= (true[lim_col] <= config.mag_limit)
        if "Gmag_app" in true:
            mag_mask &= (true["Gmag_app"] >= config.mag_bright_limit)

    n_before = len(dist)
    n_after = np.sum(mag_mask)
    print(f"  Magnitude cut: {n_before:,} → {n_after:,} stars "
          f"({config.mag_limit_band} < {config.mag_limit})")

    true = {k: v[mag_mask] if isinstance(v, np.ndarray) else v
            for k, v in true.items()}

    stellar = {}
    for key in ["Mini", "Mass", "logage", "MH", "logL", "logTe", "logg",
                "label", "parent_id", "is_binary", "q", "Mini_secondary"]:
        if key in raw_cat:
            arr = raw_cat[key]
            stellar[key] = arr[mag_mask] if isinstance(arr, np.ndarray) else arr

    catalogue = WithFrame({"true": true, "stellar": stellar})
    catalogue.frame = frame

    if config.selection_function != "none":
        catalogue = apply_selection_function(
            catalogue, config, rng, star_pos=raw_cat["pos"][mag_mask],
            frame=frame)

    if config.apply_errors and error_model is not None:
        catalogue = error_model.apply_errors(catalogue, rng=rng)

        if "cartesian" in config.output_coords:
            obs = catalogue.get("observed", {})
            if "ra" in obs and "parallax" in obs:
                from .coordinates import observed_to_cartesian
                obs_cart = observed_to_cartesian(
                    obs["ra"], obs["dec"], obs["parallax"],
                    obs.get("pmra"), obs.get("pmdec"), obs.get("rv"),
                    frame=frame,
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


def load_rvs_selection_function():
    """gaiaunlimited's DR3 RVS selection function object (downloads and
    caches its data, ~95 MB, on first use)."""
    try:
        from gaiaunlimited.selectionfunctions import DR3RVSSelectionFunction
    except ImportError:
        raise ImportError(
            "The DR3 RVS selection function needs gaiaunlimited: "
            "pip install gaiaunlimited"
        )
    return DR3RVSSelectionFunction()


def rvs_selection_probability(l_deg, b_deg, gmag, g_rp, mag_shift=0.0,
                              sfn=None, chunk=500_000):
    """P(has RV | l, b, G, G - G_RP) from the Gaia DR3 RVS selection function
    (Castro-Ginard et al. 2023, via gaiaunlimited), vectorised.

    ``mag_shift`` is subtracted from G before the query: the DR3 function
    evaluated at G - 2.5 is the stand-in for the unpublished DR4 one (same
    sky pattern, limit shifted with the depth).  Sky cells and
    colour-magnitude bins without data give P = 0.
    """
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    if sfn is None:
        sfn = load_rvs_selection_function()
    l_deg = np.asarray(l_deg, dtype=np.float64)
    b_deg = np.asarray(b_deg, dtype=np.float64)
    gmag = np.asarray(gmag, dtype=np.float64)
    g_rp = np.asarray(g_rp, dtype=np.float64)
    n = len(gmag)
    prob = np.zeros(n)
    for i in range(0, n, chunk):
        sl = slice(i, min(i + chunk, n))
        coords = SkyCoord(l=l_deg[sl] * u.deg, b=b_deg[sl] * u.deg,
                          frame="galactic")
        p = sfn.query(coords, g=gmag[sl] - mag_shift, c=g_rp[sl])
        prob[sl] = np.nan_to_num(np.asarray(p, dtype=np.float64), nan=0.0)
    return prob


def apply_selection_function(catalogue, config, rng, star_pos=None,
                             frame=None):
    """Thin the catalogue with the Gaia DR3 RVS selection function.

    Each star is kept with probability ``rvs_selection_probability`` (a
    Bernoulli draw); the probability is stored as ``true/p_selection``.
    Needs l, b (taken from the catalogue if present, else computed from
    ``star_pos`` in the observer's frame), apparent G and G_RP.
    """
    true = catalogue["true"]
    gmag = true.get("Gmag_app")
    rpmag = true.get("G_RPmag_app")
    if gmag is None or rpmag is None:
        raise ValueError("Selection function needs apparent G and G_RP.")
    n = len(gmag)
    if n == 0:
        return catalogue

    if "l" in true and "b" in true:
        l_deg, b_deg = true["l"], true["b"]
    else:
        if star_pos is None:
            raise ValueError("Selection function needs l, b or star_pos.")
        if frame is None:
            frame = getattr(catalogue, "frame", None)
        if frame is None:
            frame = GalacticFrame.from_config(config)
        gal = frame.galactic(star_pos)
        l_deg, b_deg = gal["l"], gal["b"]

    print(f"  Selection function: DR3 RVS (gaiaunlimited)"
          + (f", evaluated at G - {config.sf_mag_shift}"
             if config.sf_mag_shift else ""))
    prob = rvs_selection_probability(l_deg, b_deg, gmag, gmag - rpmag,
                                     mag_shift=config.sf_mag_shift)

    keep = rng.uniform(size=n) < prob
    n_keep = int(np.sum(keep))
    print(f"  Selection function: {n:,} → {n_keep:,} stars "
          f"(mean P = {np.mean(prob):.3f})")

    catalogue["true"] = {k: v[keep] if isinstance(v, np.ndarray) and len(v) == n
                         else v for k, v in true.items()}
    if "stellar" in catalogue:
        catalogue["stellar"] = {
            k: v[keep] if isinstance(v, np.ndarray) and len(v) == n else v
            for k, v in catalogue["stellar"].items()}
    catalogue["true"]["p_selection"] = prob[keep]
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
        meta.attrs["mag_limit_band"] = config.mag_limit_band
        meta.attrs["selection_function"] = config.selection_function
        meta.attrs["sf_mag_shift"] = config.sf_mag_shift
        meta.attrs["imf"] = config.imf
        meta.attrs["extinction_mode"] = config.extinction_mode
        meta.attrs["observer_position"] = list(config.observer_position)
        frame = getattr(catalogue, "frame", None)
        if frame is not None:
            fm = frame.metadata()
            meta.attrs["observer_velocity"] = fm["observer_velocity"]
            meta.attrs["rotation"] = fm["rotation"]
            meta.attrs["rotation_source"] = fm["rotation_source"]
            meta.attrs["galactic_north"] = fm["galactic_north"]
            meta.attrs["frame_axes"] = np.array(fm["frame_axes"])
        elif config.observer_velocity is not None:
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
