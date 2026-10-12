"""Phase-space smoothing for population synthesis.

Estimates per-particle smoothing lengths from the local phase-space
density (separately in position and velocity), then spreads synthetic
stars around their parent particles using an Epanechnikov kernel.

The density estimation uses k-nearest-neighbour distances via
scipy's cKDTree (C/Cython under the hood).  The Epanechnikov kernel
has compact support, preventing unphysically large displacements.

References
----------
- Sharma & Steinmetz (2006), MNRAS 373, 1293 (EnBid algorithm)
- Sanderson et al. (2020), ApJS 246, 6 (Ananke framework)
- Nguyen et al. (2023), MNRAS 533, 143 (bandwidth calibration)
"""

import numpy as np
from scipy.spatial import cKDTree


def estimate_smoothing_lengths(pos, vel, n_ngb=64, scale=1.0):
    """Compute per-particle smoothing lengths in position and velocity.

    Uses kNN density estimation: rho_i = (k-1) / (4/3 pi r_k^3)
    then h_i = (3 / (4 pi rho_i))^(1/3) = r_k / (k-1)^(1/3).

    Parameters
    ----------
    pos : (N, 3) array [kpc]
    vel : (N, 3) array [km/s]
    n_ngb : int
        Number of nearest neighbours (default 64).
    scale : float
        Multiplicative calibration factor applied to both h_pos
        and h_vel (default 1.0).

    Returns
    -------
    h_pos : (N,) array [kpc]
    h_vel : (N,) array [km/s]
    """
    h_pos = _knn_smoothing_length(pos, n_ngb) * scale
    h_vel = _knn_smoothing_length(vel, n_ngb) * scale
    return h_pos, h_vel


def _knn_smoothing_length(coords, k):
    """Smoothing length from the k-th nearest-neighbour distance.

    h = r_k / (k-1)^(1/3), which is equivalent to inverting
    rho = (k-1) / (4/3 pi r_k^3) then h = (3/(4 pi rho))^(1/3).
    """
    tree = cKDTree(coords)
    dists, _ = tree.query(coords, k=k + 1)
    r_k = dists[:, -1]
    r_k = np.maximum(r_k, 1e-30)
    return r_k / (k - 1) ** (1.0 / 3.0)


def epanechnikov_offsets(h, n_stars, rng):
    """Draw 3D displacement vectors from an Epanechnikov kernel.

    K(r) ~ (1 - r^2) for r <= 1.  The radial CDF in 3D is sampled
    via the inverse-CDF method: r = h * u^(1/5) where u ~ Uniform(0,1).

    The exponent 1/5 comes from the normalised 3D Epanechnikov radial
    CDF: F(r) = r^3 (5/2 - 3/2 r^2) simplified to the quantile of
    a Beta(5/2, 2) variable raised to 1/3 — but a fast, exact method
    is rejection sampling from a uniform ball.

    Parameters
    ----------
    h : (N_parent,) array
        Smoothing length per parent particle.
    n_stars : (N_parent,) int array
        Number of stars per parent particle.
    rng : numpy Generator

    Returns
    -------
    offsets : (n_total, 3) array
    """
    n_total = int(np.sum(n_stars))
    if n_total == 0:
        return np.zeros((0, 3))

    h_per_star = np.repeat(h, n_stars)

    offsets = _sample_epanechnikov_3d(n_total, rng)
    offsets *= h_per_star[:, None]
    return offsets


def _sample_epanechnikov_3d(n, rng):
    """Sample n points from a unit 3D Epanechnikov kernel via rejection.

    Proposal: uniform in the unit ball.
    Accept with probability (1 - r^2).
    Expected acceptance rate = 5/8 = 62.5%.
    """
    result = np.empty((n, 3))
    filled = 0

    while filled < n:
        need = int((n - filled) * 1.7) + 64
        candidates = rng.uniform(-1, 1, size=(need, 3))
        r2 = np.sum(candidates**2, axis=1)
        in_ball = r2 <= 1.0
        candidates = candidates[in_ball]
        r2 = r2[in_ball]

        accept = rng.uniform(size=len(r2)) < (1.0 - r2)
        accepted = candidates[accept]

        take = min(len(accepted), n - filled)
        result[filled:filled + take] = accepted[:take]
        filled += take

    return result
