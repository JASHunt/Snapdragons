"""Gaia photometric transformations.

G_RVS from G and G_RP
---------------------
Sartoretti et al. (2023, A&A 674, A6), eqs 9-10: G_RVS - G_RP as a cubic in
G - G_RP, with a blue branch (G - G_RP <= 1.2, rms 0.04 mag) and a red one
(1.2 < G - G_RP <= 1.7, rms 0.09 mag).  Outside -0.15 <= G - G_RP <= 1.7 the
relation is not calibrated and NaN is returned.
"""

import numpy as np


GRVS_COLOUR_MIN = -0.15
GRVS_COLOUR_MAX = 1.7


def grvs_from_g_rp(gmag, rpmag):
    """G_RVS from G and G_RP (Sartoretti+ 2023 eqs 9-10).

    Works on absolute or apparent magnitudes alike; apply it to extincted
    apparent magnitudes to get the apparent G_RVS.  NaN where G - G_RP is
    outside the calibrated colour range.
    """
    gmag = np.asarray(gmag, dtype=np.float64)
    rpmag = np.asarray(rpmag, dtype=np.float64)
    x = gmag - rpmag
    with np.errstate(invalid="ignore"):
        blue = -0.0397 - 0.2852 * x - 0.0330 * x**2 - 0.0867 * x**3
        red = -4.0618 + 10.0187 * x - 9.0532 * x**2 + 2.6089 * x**3
        out = np.where(x <= 1.2, blue, red) + rpmag
        bad = (x < GRVS_COLOUR_MIN) | (x > GRVS_COLOUR_MAX) | ~np.isfinite(x)
    return np.where(bad, np.nan, out)
