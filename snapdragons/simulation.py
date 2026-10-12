"""Load and prepare simulation snapshots via pynbody.

Reads ChaNGa/Tipsy (or any pynbody-supported format), extracts star
and gas particle data, and optionally centres/orients the galaxy.
"""

import os
import glob
import numpy as np


class SimulationSnapshot:
    """Container for simulation particle data.

    Attributes
    ----------
    star_pos : (N_star, 3) array [kpc]
    star_vel : (N_star, 3) array [km/s]
    star_mass : (N_star,) array [Msun]
    star_age : (N_star,) array [Gyr]
    star_metallicity : (N_star,) array [M/H] (dex, solar-scaled)
    gas_pos : (N_gas, 3) array [kpc]  (if loaded)
    gas_density : (N_gas,) array [Msun/kpc^3]
    gas_metallicity : (N_gas,) array
    gas_smooth : (N_gas,) array [kpc]
    """

    def __init__(self):
        self.star_pos = None
        self.star_vel = None
        self.star_mass = None
        self.star_age = None
        self.star_metallicity = None
        self.n_star = 0

        self.gas_pos = None
        self.gas_density = None
        self.gas_metallicity = None
        self.gas_smooth = None
        self.n_gas = 0

    @classmethod
    def from_pynbody(cls, filepath, center=True, orient=True,
                     load_gas=True, param_file=None,
                     apply_tform_correction=True):
        """Load a simulation snapshot using pynbody.

        Parameters
        ----------
        filepath : str
            Path to the snapshot file (Tipsy, Gadget, etc.).
        center : bool
            Centre the snapshot on the main halo.
        orient : bool
            Rotate to face-on (angular momentum along z).
        load_gas : bool
            Also load gas particles (needed for simulation-based extinction).
        param_file : str, optional
            Path to the ChaNGa .param file. If None, searches for any
            .param file in the snapshot's directory.
        apply_tform_correction : bool
            Apply the ChaNGa tform correction for stars with
            tform > 0.4e6 (default True).
        """
        try:
            import pynbody
        except ImportError:
            raise ImportError(
                "pynbody is required to read simulation snapshots. "
                "Install with: pip install pynbody"
            )

        if param_file is not None:
            _link_param_file(filepath, param_file)

        print(f"  Loading snapshot: {filepath}")
        sim = pynbody.load(filepath)

        if apply_tform_correction:
            _apply_tform_correction(sim)

        sim.physical_units()

        if center or orient:
            try:
                halo = sim
                if center:
                    pynbody.analysis.halo.center(halo)
                    print("  Centred on main halo")
                if orient:
                    pynbody.analysis.angmom.faceon(halo)
                    print("  Rotated to face-on orientation")
            except Exception as e:
                print(f"  Warning: centering/orientation failed ({e}), "
                      f"using raw coordinates")

        snap = cls()
        stars = sim.star

        snap.star_pos = np.array(stars["pos"].in_units("kpc"))
        snap.star_vel = np.array(stars["vel"].in_units("km s**-1"))
        snap.star_mass = np.array(stars["mass"].in_units("Msol"))
        snap.n_star = len(snap.star_mass)

        snap.star_age = cls._get_age(stars)
        snap.star_metallicity = cls._get_metallicity(stars)

        print(f"  Loaded {snap.n_star} star particles")

        if load_gas and len(sim.gas) > 0:
            gas = sim.gas
            snap.gas_pos = np.array(gas["pos"].in_units("kpc"))
            snap.gas_density = np.array(gas["rho"].in_units("Msol kpc**-3"))
            snap.n_gas = len(snap.gas_density)

            try:
                snap.gas_smooth = np.array(gas["smooth"].in_units("kpc"))
            except KeyError:
                snap.gas_smooth = np.full(snap.n_gas, 0.1)

            snap.gas_metallicity = cls._get_metallicity(gas)
            print(f"  Loaded {snap.n_gas} gas particles")

        return snap

    @staticmethod
    def _get_age(family):
        """Extract stellar age in Gyr, handling different naming conventions."""
        try:
            tform = family["tform"]
            age = family.properties.get("time", None)
            if age is not None:
                age_arr = np.array(
                    (age - tform).in_units("Gyr")
                )
            else:
                age_arr = np.abs(np.array(tform.in_units("Gyr")))
        except (KeyError, Exception):
            try:
                age_arr = np.array(family["age"].in_units("Gyr"))
            except KeyError:
                raise ValueError(
                    "Cannot determine stellar ages. Snapshot must have "
                    "'tform' or 'age' for star particles."
                )

        age_arr = np.maximum(age_arr, 1e-4)
        return age_arr

    @staticmethod
    def _get_metallicity(family):
        """Extract metallicity as [M/H] (dex).

        Handles both linear Z and [Fe/H] conventions.
        """
        Z_sun = 0.0152

        try:
            met = np.array(family["metals"])
        except KeyError:
            try:
                met = np.array(family["feh"])
                return met
            except KeyError:
                return np.zeros(len(family))

        if np.nanmedian(met) < 1.0:
            met = np.maximum(met, 1e-6)
            return np.log10(met / Z_sun)
        else:
            return met

    @classmethod
    def from_arrays(cls, star_pos, star_vel, star_mass, star_age,
                    star_metallicity, gas_pos=None, gas_density=None,
                    gas_metallicity=None, gas_smooth=None):
        """Create a snapshot from raw numpy arrays."""
        snap = cls()
        snap.star_pos = np.asarray(star_pos, dtype=np.float64)
        snap.star_vel = np.asarray(star_vel, dtype=np.float64)
        snap.star_mass = np.asarray(star_mass, dtype=np.float64)
        snap.star_age = np.asarray(star_age, dtype=np.float64)
        snap.star_metallicity = np.asarray(star_metallicity, dtype=np.float64)
        snap.n_star = len(snap.star_mass)

        if gas_pos is not None:
            snap.gas_pos = np.asarray(gas_pos, dtype=np.float64)
            snap.gas_density = np.asarray(gas_density, dtype=np.float64)
            snap.gas_metallicity = np.asarray(gas_metallicity, dtype=np.float64)
            snap.gas_smooth = np.asarray(gas_smooth, dtype=np.float64)
            snap.n_gas = len(snap.gas_density)

        print(f"  Loaded {snap.n_star} star particles from arrays")
        return snap


def _link_param_file(snapshot_path, param_path):
    """Ensure pynbody can find the .param file.

    Pynbody looks for <snapshot_basename>.param in the snapshot directory.
    If the param file has a different name, create a symlink.
    """
    snap_dir = os.path.dirname(os.path.abspath(snapshot_path))
    snap_base = os.path.basename(snapshot_path)
    base_name = snap_base.split(".")[0]
    expected = os.path.join(snap_dir, f"{base_name}.param")

    if os.path.exists(expected):
        return

    param_abs = os.path.abspath(param_path)
    if not os.path.exists(param_abs):
        print(f"  Warning: param file not found: {param_path}")
        return

    try:
        os.symlink(param_abs, expected)
        print(f"  Linked {param_path} -> {expected}")
    except OSError:
        import shutil
        shutil.copy2(param_abs, expected)
        print(f"  Copied {param_path} -> {expected}")


def _apply_tform_correction(sim):
    """Apply ChaNGa tform correction for AMNH-type simulations.

    Stars with raw tform > 0.4e6 (in simulation units) need remapping:
        tform_new = (tform_old - 1e6) * 4.5 + 597.53125

    This corrects for a time-encoding issue in certain ChaNGa runs.
    """
    tform = sim.star["tform"]
    raw = np.array(tform)
    mask = raw > 0.4e6
    n_corrected = np.sum(mask)

    if n_corrected > 0:
        raw[mask] = (raw[mask] - 1e6) * 4.5 + 597.53125
        tform[:] = raw
        print(f"  Applied tform correction to {n_corrected} / "
              f"{len(raw)} star particles")
    else:
        print("  tform correction: no particles needed correction")
