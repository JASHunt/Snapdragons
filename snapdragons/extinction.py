"""Dust extinction: mwdust 3D maps or simulation-based gas→dust.

Two modes
---------
1. **mwdust** — queries a 3D Milky Way dust map (e.g. Combined19,
   Marshall06, Green19) at each star's (l, b, d).  The mwdust package
   handles its own data downloads and caching, avoiding the need
   for a large binary extinction table.

2. **simulation** — estimates the dust column density from the gas
   particles in the simulation along the line of sight from the
   observer to each star, converting gas density to dust via a
   configurable dust-to-gas ratio.

Extinction coefficients
-----------------------
Gaia band coefficients from Casagrande & VandenBerg (2018) assuming
R_V = 3.1 by default:

    A_G / A_V   ≈ 0.789
    A_BP / A_V  ≈ 1.002
    A_RP / A_V  ≈ 0.589
    A_V / E(B-V) = R_V = 3.1
"""

import numpy as np
from scipy.spatial import cKDTree


EXTINCTION_COEFFS = {
    "Gmag":    0.789,
    "G_BPmag": 1.002,
    "G_RPmag": 0.589,
}


class ExtinctionCalculator:
    """Base class for extinction calculations."""

    def __init__(self, config):
        self.config = config
        self.rv = config.rv_extinction

    def compute(self, l_deg, b_deg, dist_kpc):
        """Return A_V for each star. Subclasses must implement."""
        raise NotImplementedError

    def apply(self, apparent_mags, a_v):
        """Add extinction to apparent magnitudes.

        Parameters
        ----------
        apparent_mags : dict
            Keys are band names (e.g. "Gmag"), values are arrays.
        a_v : array
            Visual extinction for each star.

        Returns
        -------
        dict of extincted apparent magnitudes, plus A_V and E(B-V) arrays.
        """
        result = {}
        for band, mag in apparent_mags.items():
            coeff = EXTINCTION_COEFFS.get(band, 0.0)
            result[band] = mag + coeff * a_v
        result["A_V"] = a_v
        result["E_BV"] = a_v / self.rv
        for band in EXTINCTION_COEFFS:
            result[f"A_{band}"] = EXTINCTION_COEFFS[band] * a_v
        return result


class MWDustExtinction(ExtinctionCalculator):
    """3D Milky Way dust extinction using the mwdust package.

    The mwdust package downloads and caches map data on first use
    (~200 MB–1 GB depending on the map), stored in ~/.mwdust/.
    No external binary table is needed.
    """

    def __init__(self, config):
        super().__init__(config)
        self._dustmap = None

    def _load_map(self):
        if self._dustmap is not None:
            return
        try:
            import mwdust
        except ImportError:
            raise ImportError(
                "mwdust is required for 3D dust maps. "
                "Install with: pip install mwdust"
            )

        map_name = self.config.mwdust_map
        print(f"  Loading mwdust map: {map_name}")
        try:
            MapClass = getattr(mwdust, map_name)
        except AttributeError:
            raise ValueError(
                f"Unknown mwdust map '{map_name}'. Available maps include: "
                "Combined19, Marshall06, Green19, Drimmel03, Sale14"
            )
        self._dustmap = MapClass(filter="2MASS H")

    def compute(self, l_deg, b_deg, dist_kpc):
        """Query the 3D dust map for E(B-V), convert to A_V.

        Parameters
        ----------
        l_deg, b_deg : arrays [degrees]
        dist_kpc : array [kpc]

        Returns
        -------
        a_v : array — visual extinction
        """
        self._load_map()

        n = len(l_deg)
        ebv = np.zeros(n)

        chunk = 10000
        for i in range(0, n, chunk):
            sl = slice(i, min(i + chunk, n))
            ebv[sl] = self._dustmap(l_deg[sl], b_deg[sl], dist_kpc[sl])

        a_v = ebv * self.rv
        return a_v


class SimulationExtinction(ExtinctionCalculator):
    """Estimate extinction from the gas particles in the simulation.

    For each star, ray-marches from the observer through the gas,
    sampling the SPH density field to compute the hydrogen column
    density N_H, then converts to A_V via:

        N_H [cm^-2] → E(B-V) = N_H / 5.8e21  (Bohlin+ 1978)
        A_V = R_V × E(B-V)

    The Bohlin relation assumes a standard MW dust-to-gas ratio.
    The ``dust_to_gas_ratio`` config parameter acts as a multiplicative
    calibration factor (default 1.0 = solar neighbourhood).  Set it
    below 1 for sub-solar metallicity gas.
    """

    _MU = 1.4               # mean molecular weight per H nucleus
    _MSUN_G = 1.98892e33    # solar mass in grams
    _KPC_CM = 3.085677581e21  # kpc in cm
    _MP_G = 1.6726219e-24   # proton mass in grams
    _NH_TO_EBV = 1.0 / 5.8e21  # Bohlin+ 1978

    def __init__(self, config, gas_pos, gas_density, gas_smooth,
                 gas_metallicity=None):
        super().__init__(config)
        self.gas_pos = np.asarray(gas_pos)
        self.gas_density = np.asarray(gas_density)
        self.gas_smooth = np.asarray(gas_smooth)
        self.gas_metallicity = gas_metallicity
        self.dust_to_gas = config.dust_to_gas_ratio
        self._tree = cKDTree(self.gas_pos)

    def _rho_to_nh(self, rho_msun_kpc3):
        """Convert gas density [Msol kpc^-3] to n_H [cm^-3]."""
        rho_cgs = rho_msun_kpc3 * self._MSUN_G / self._KPC_CM**3
        return rho_cgs / (self._MU * self._MP_G)

    def compute(self, l_deg, b_deg, dist_kpc,
                star_pos=None, observer_pos=None, n_sample=64):
        """Compute extinction from simulation gas along lines of sight.

        Ray-marches from the observer to each star, sampling the gas
        density at ``n_sample`` points and integrating N_H numerically.
        Processes stars in vectorised batches for speed.

        Parameters
        ----------
        l_deg, b_deg, dist_kpc : arrays
            Galactic coordinates (used only for the interface).
        star_pos : (N, 3) array [kpc]
            Cartesian positions of stars in the simulation frame.
        observer_pos : (3,) array [kpc]
            Observer position.
        n_sample : int
            Number of sample points per line of sight.

        Returns
        -------
        a_v : array
        """
        if star_pos is None or observer_pos is None:
            raise ValueError(
                "simulation-based extinction requires star_pos and observer_pos"
            )

        observer_pos = np.asarray(observer_pos, dtype=np.float64)
        star_pos = np.asarray(star_pos, dtype=np.float64)
        n_stars = len(star_pos)
        a_v = np.zeros(n_stars)

        t_mid = ((np.arange(n_sample) + 0.5) / n_sample).astype(np.float64)
        chunk = 5000

        for start in range(0, n_stars, chunk):
            end = min(start + chunk, n_stars)
            rays = star_pos[start:end] - observer_pos
            ray_lens = np.linalg.norm(rays, axis=1)

            valid = ray_lens > 1e-10
            if not np.any(valid):
                continue

            v_rays = rays[valid]
            v_lens = ray_lens[valid]
            V = len(v_rays)

            pts = observer_pos + t_mid[:, None, None] * v_rays[None, :, :]
            pts_flat = pts.reshape(-1, 3)

            dists_nn, idx_nn = self._tree.query(pts_flat, k=1)
            dists_nn = dists_nn.reshape(n_sample, V)
            idx_nn = idx_nn.reshape(n_sample, V)

            h_nn = self.gas_smooth[idx_nn]
            q = dists_nn / h_nn
            close = q < 1.0

            rho = np.where(close, self.gas_density[idx_nn], 0.0)

            w = np.zeros_like(q)
            m1 = (q < 0.5) & close
            m2 = (q >= 0.5) & close
            w[m1] = 1.0 - 6.0 * q[m1]**2 + 6.0 * q[m1]**3
            w[m2] = 2.0 * (1.0 - q[m2])**3

            n_h = self._rho_to_nh(rho * w)

            ds_kpc = v_lens / n_sample
            column_nh = np.sum(n_h, axis=0) * ds_kpc * self._KPC_CM

            ebv = column_nh * self._NH_TO_EBV * self.dust_to_gas

            batch_av = np.zeros(end - start)
            batch_av[valid] = ebv * self.rv
            a_v[start:end] = batch_av

        return a_v


class NoExtinction(ExtinctionCalculator):
    """Null extinction — returns zero for all stars."""

    def compute(self, l_deg, b_deg, dist_kpc, **kwargs):
        return np.zeros(len(l_deg))


def get_extinction_calculator(config, snapshot=None):
    """Factory function for extinction calculators."""
    mode = config.extinction_mode.lower()
    if mode == "none":
        return NoExtinction(config)
    elif mode == "mwdust":
        return MWDustExtinction(config)
    elif mode == "simulation":
        if snapshot is None or snapshot.gas_pos is None:
            raise ValueError(
                "Simulation-based extinction requires gas particles. "
                "Ensure the snapshot is loaded with load_gas=True."
            )
        return SimulationExtinction(
            config,
            gas_pos=snapshot.gas_pos,
            gas_density=snapshot.gas_density,
            gas_smooth=snapshot.gas_smooth,
            gas_metallicity=snapshot.gas_metallicity,
        )
    else:
        raise ValueError(
            f"Unknown extinction mode '{mode}'. "
            "Choose 'mwdust', 'simulation', or 'none'."
        )
