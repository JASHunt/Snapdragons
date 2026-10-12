"""Gaia observational error models.

Wraps PyGaia to add realistic astrometric, photometric, and
spectroscopic uncertainties for a chosen Gaia data release.

Supported releases: DR2, DR3, DR4, DR5.
"""

import numpy as np

GAIA_MAG_BRIGHT = 4.0
GAIA_MAG_FAINT = 21.0

RELEASE_MAP = {
    "dr2": "dr3",  # PyGaia uses dr3 as minimum; DR2 approximated via scaling
    "dr3": "dr3",
    "dr4": "dr4",
    "dr5": "dr5",
}

DR2_SCALE = {
    "parallax": 1.5,
    "pm": 2.0,
    "phot": 1.3,
}


class GaiaErrorModel:
    """Apply Gaia observational errors to a synthetic catalogue."""

    def __init__(self, config):
        self.config = config
        self.release = config.gaia_release.lower()
        self._check_pygaia()

    def _check_pygaia(self):
        try:
            import pygaia
            self._has_pygaia = True
        except ImportError:
            self._has_pygaia = False
            print("  Warning: PyGaia not installed. "
                  "Using simplified error model.")

    def parallax_uncertainty(self, gmag):
        """Parallax uncertainty in mas."""
        if self._has_pygaia:
            return self._pygaia_parallax(gmag)
        return self._simple_parallax(gmag)

    def proper_motion_uncertainty(self, gmag):
        """Proper motion uncertainty in mas/yr. Returns (pmra_err, pmdec_err)."""
        if self._has_pygaia:
            return self._pygaia_pm(gmag)
        err = self._simple_pm(gmag)
        return err, err

    def photometric_uncertainty(self, gmag):
        """Photometric uncertainty in G, G_BP, G_RP [mag]."""
        if self._has_pygaia:
            return self._pygaia_phot(gmag)
        return self._simple_phot(gmag)

    def radial_velocity_uncertainty(self, gmag, teff=None):
        """Radial velocity uncertainty in km/s."""
        if self._has_pygaia:
            return self._pygaia_rv(gmag, teff)
        return self._simple_rv(gmag)

    def apply_errors(self, catalogue, rng=None):
        """Add Gaussian errors to all observables in the catalogue.

        Stars outside PyGaia's valid magnitude range [4, 21] are removed
        (too bright for Gaia's detectors, or too faint to observe).

        Parameters
        ----------
        catalogue : dict
            Must contain 'true' sub-dict with astrometric/photometric data.
        rng : numpy Generator

        Returns
        -------
        dict with 'observed' and 'uncertainties' sub-dicts added.
        Stars outside Gaia's magnitude range are filtered out.
        """
        if rng is None:
            rng = np.random.default_rng()

        true = catalogue["true"]
        gmag = true.get("Gmag_app", true.get("Gmag", np.array([])))
        if len(gmag) == 0:
            return catalogue

        gaia_valid = (gmag >= GAIA_MAG_BRIGHT) & (gmag <= GAIA_MAG_FAINT)
        n_too_bright = int(np.sum(gmag < GAIA_MAG_BRIGHT))
        n_too_faint = int(np.sum(gmag > GAIA_MAG_FAINT))
        n_valid = int(np.sum(gaia_valid))

        if n_too_bright > 0:
            print(f"  {n_too_bright} stars too bright for Gaia (G < {GAIA_MAG_BRIGHT}), removed")
        if n_too_faint > 0:
            print(f"  {n_too_faint} stars too faint for Gaia (G > {GAIA_MAG_FAINT}), removed")

        if n_valid == 0:
            catalogue["observed"] = {}
            catalogue["uncertainties"] = {}
            return catalogue

        true = {k: v[gaia_valid] if isinstance(v, np.ndarray) and len(v) == len(gmag)
                else v for k, v in true.items()}
        catalogue["true"] = true

        if "stellar" in catalogue:
            catalogue["stellar"] = {
                k: v[gaia_valid] if isinstance(v, np.ndarray) and len(v) == len(gmag)
                else v for k, v in catalogue["stellar"].items()
            }

        gmag = true["Gmag_app"]
        n = len(gmag)
        obs = {}
        unc = {}

        plx_err = self.parallax_uncertainty(gmag)
        unc["parallax"] = plx_err
        if "parallax" in true:
            obs["parallax"] = true["parallax"] + rng.normal(0, plx_err)

        pmra_err, pmdec_err = self.proper_motion_uncertainty(gmag)
        unc["pmra"] = pmra_err
        unc["pmdec"] = pmdec_err
        if "pmra" in true:
            obs["pmra"] = true["pmra"] + rng.normal(0, pmra_err)
        if "pmdec" in true:
            obs["pmdec"] = true["pmdec"] + rng.normal(0, pmdec_err)

        g_err, bp_err, rp_err = self.photometric_uncertainty(gmag)
        unc["Gmag"] = g_err
        unc["G_BPmag"] = bp_err
        unc["G_RPmag"] = rp_err
        if "Gmag_app" in true:
            obs["Gmag"] = true["Gmag_app"] + rng.normal(0, g_err)
        if "G_BPmag_app" in true:
            obs["G_BPmag"] = true["G_BPmag_app"] + rng.normal(0, bp_err)
        if "G_RPmag_app" in true:
            obs["G_RPmag"] = true["G_RPmag_app"] + rng.normal(0, rp_err)

        teff = true.get("logTe", None)
        if teff is not None:
            teff_linear = 10.0 ** teff
        else:
            teff_linear = None
        rv_err = self.radial_velocity_uncertainty(gmag, teff_linear)
        unc["rv"] = rv_err
        if "rv" in true:
            obs["rv"] = true["rv"] + rng.normal(0, np.maximum(rv_err, 1e-10))

        if "ra" in true:
            ra_err = plx_err / 1000.0
            unc["ra"] = ra_err
            obs["ra"] = true["ra"] + rng.normal(0, ra_err / 3600.0)
        if "dec" in true:
            dec_err = plx_err / 1000.0
            unc["dec"] = dec_err
            obs["dec"] = true["dec"] + rng.normal(0, dec_err / 3600.0)

        if "l" in true:
            obs["l"] = true["l"]
            obs["b"] = true["b"]

        catalogue["observed"] = obs
        catalogue["uncertainties"] = unc
        return catalogue

    # --- PyGaia backends ---

    def _pygaia_parallax(self, gmag):
        from pygaia.errors.astrometric import parallax_uncertainty
        release = RELEASE_MAP.get(self.release, "dr3")
        err = parallax_uncertainty(gmag, release=release) * 1e-3  # μas → mas
        if self.release == "dr2":
            err *= DR2_SCALE["parallax"]
        return err

    def _pygaia_pm(self, gmag):
        from pygaia.errors.astrometric import proper_motion_uncertainty
        release = RELEASE_MAP.get(self.release, "dr3")
        pmra_err, pmdec_err = proper_motion_uncertainty(gmag, release=release)
        pmra_err = pmra_err * 1e-3  # μas/yr → mas/yr
        pmdec_err = pmdec_err * 1e-3
        if self.release == "dr2":
            pmra_err *= DR2_SCALE["pm"]
            pmdec_err *= DR2_SCALE["pm"]
        return pmra_err, pmdec_err

    def _pygaia_phot(self, gmag):
        from pygaia.errors.photometric import magnitude_uncertainty
        release = RELEASE_MAP.get(self.release, "dr3")
        g_err = magnitude_uncertainty("g", gmag, release=release)
        bp_err = magnitude_uncertainty("bp", gmag, release=release)
        rp_err = magnitude_uncertainty("rp", gmag, release=release)
        if self.release == "dr2":
            g_err *= DR2_SCALE["phot"]
            bp_err *= DR2_SCALE["phot"]
            rp_err *= DR2_SCALE["phot"]
        return g_err, bp_err, rp_err

    def _pygaia_rv(self, gmag, teff=None):
        try:
            from pygaia.errors.spectroscopic import radial_velocity_uncertainty
            release = RELEASE_MAP.get(self.release, "dr3")
            if teff is not None:
                err = radial_velocity_uncertainty(gmag, teff, release=release)
            else:
                err = radial_velocity_uncertainty(
                    gmag, np.full_like(gmag, 5500.0), release=release
                )
            return np.maximum(err, 0.11)
        except Exception:
            return self._simple_rv(gmag)

    # --- Simple fallback models ---

    def _simple_parallax(self, gmag):
        return 0.02 * 10.0 ** (0.2 * (gmag - 15.0))

    def _simple_pm(self, gmag):
        return 0.02 * 10.0 ** (0.2 * (gmag - 15.0))

    def _simple_phot(self, gmag):
        base = 0.001 * 10.0 ** (0.2 * (gmag - 15.0))
        return base, base * 1.2, base * 0.9

    def _simple_rv(self, gmag):
        return np.where(gmag < 13.0, 0.3,
                        np.where(gmag < 16.0,
                                 0.3 * 10.0 ** (0.15 * (gmag - 13.0)),
                                 np.nan))
