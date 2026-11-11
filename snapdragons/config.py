"""Configuration for Snapdragons population synthesis."""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class Config:
    """All parameters for a Snapdragons run.

    Attributes are grouped by function. Defaults match standard Solar
    values and Gaia DR3 settings.
    """

    # --- Input ---
    snapshot_path: Optional[str] = None
    param_file: Optional[str] = None  # ChaNGa .param file (auto-detected if None)

    # --- Observer position [kpc], in the simulation's own coordinates ---
    # The galactic centre is the origin and the disc lies in the xy plane.
    # The observer is used exactly where it is put; the Galactic frame
    # (l = 0 to the centre, north = the side from which the disc turns
    # clockwise) is built around it: see snapdragons/frame.py.
    observer_position: tuple = (-8.2, 0.0, 0.02)

    # Sense of rotation of the disc seen from the simulation's +z:
    #   "auto"              -- detected from the star particles (default)
    #   "clockwise"         -- the Milky Way's sense with z to the north
    #   "counterclockwise"
    rotation: str = "auto"

    # --- Solar peculiar motion [km/s] (Schoenrich et al. 2010) ---
    solar_U: float = 11.1    # toward Galactic centre
    solar_V: float = 12.24   # in direction of rotation (peculiar component)
    solar_W: float = 7.25    # toward North Galactic Pole

    # Circular velocity at the Solar radius [km/s]
    v_circular: float = 232.8

    # Full observer velocity [km/s] in simulation coordinates.
    # If None, it is v_circular along the disc's rotation plus the solar
    # motion (U to the centre, V with the rotation, W to Galactic north),
    # worked out once the sense of rotation is known (GalacticFrame).
    observer_velocity: Optional[tuple] = None

    # --- IMF ---
    imf: str = "kroupa"          # "kroupa" or "salpeter"
    m_min: float = 0.08          # minimum stellar mass [Msun]
    m_max: float = 100.0         # maximum stellar mass [Msun]

    # --- Isochrones ---
    isochrone_source: str = "parsec"
    isochrone_cache_dir: str = "~/.snapdragons/isochrones"
    photometric_system: str = "YBC_vegamag_GaiaEDR3"
    logage_min: float = 6.0     # log10(age/yr) grid lower bound
    logage_max: float = 10.2    # log10(age/yr) grid upper bound
    logage_step: float = 0.1    # grid spacing
    mh_min: float = -2.0        # [M/H] grid lower bound
    mh_max: float = 0.5         # [M/H] grid upper bound
    mh_step: float = 0.1        # grid spacing

    # --- Gaia ---
    gaia_release: str = "dr3"    # "dr2", "dr3", "dr4", "dr5"
    mag_limit: float = 16.0     # faint limit in the band given by mag_limit_band
    mag_limit_band: str = "G_RVS"  # "G_RVS" (Sartoretti+ 2023 from G, G_RP) or "G"
    mag_bright_limit: float = 3.0  # bright limit, always in G
    apply_errors: bool = True

    # RVS selection function (gaiaunlimited, Castro-Ginard+ 2023):
    #   "none"     -- keep every star brighter than mag_limit
    #   "dr3_rvs"  -- Bernoulli draw from P(has RV | l, b, G, G - G_RP) of DR3
    # sf_mag_shift evaluates the DR3 function at G - shift, a stand-in for the
    # deeper releases; None -> 2.5 for dr4/dr5, 0 otherwise.
    selection_function: str = "none"
    sf_mag_shift: Optional[float] = None

    # --- Unresolved binaries ---
    unresolved_binaries: bool = False    # combine companion flux into primary
    f_binary: float = 0.5               # binary fraction at 1 Msun (mass-dependent)
    q_min: float = 0.1                  # minimum mass ratio q = m2/m1

    # --- Phase-space smoothing ---
    phase_space_smoothing: bool = False   # spread stars around parent particles
    smoothing_ngb: int = 64              # neighbours for density estimation
    smoothing_scale: float = 1.0         # bandwidth calibration multiplier

    # --- Extinction ---
    extinction_mode: str = "mwdust"   # "mwdust", "simulation", or "none"
    mwdust_map: str = "Combined19"    # any mwdust map class name
    dust_to_gas_ratio: float = 1.0    # calibration factor for simulation extinction (1.0 = solar)
    rv_extinction: float = 3.1        # total-to-selective extinction ratio

    # --- Simulation-specific corrections ---
    apply_tform_correction: bool = True  # ChaNGa tform correction for AMNH-type sims

    # --- Output ---
    output_path: str = "output/snapdragons_catalogue.h5"
    output_format: str = "hdf5"       # "hdf5" or "fits"
    output_coords: list = field(default_factory=lambda: ["equatorial"])
    make_plots: bool = False           # generate diagnostic plots
    plot_dir: str = "output/plots"     # directory for diagnostic plots
    # Valid entries: "equatorial", "galactic", "cartesian"

    # --- Processing ---
    chunk_size: int = 1000       # star particles processed per batch
    random_seed: Optional[int] = None
    sampling: str = "mass_budget"  # "mass_budget", "poisson", or "deterministic"
    fsample: float = 1.0         # down-sampling fraction (0 < fsample <= 1)

    def __post_init__(self):
        self.observer_position = tuple(float(x) for x in self.observer_position)
        if self.observer_velocity is not None:
            self.observer_velocity = tuple(float(x) for x in self.observer_velocity)
        self.rotation = str(self.rotation).lower()
        if self.rotation not in ("auto", "clockwise", "counterclockwise"):
            raise ValueError(f"Unknown rotation '{self.rotation}'. Choose "
                             "'auto', 'clockwise' or 'counterclockwise'")
        valid_coords = {"equatorial", "galactic", "cartesian"}
        for c in self.output_coords:
            if c not in valid_coords:
                raise ValueError(f"Unknown coordinate system '{c}'. "
                                 f"Choose from {valid_coords}")
        if self.mag_limit_band not in ("G", "G_RVS"):
            raise ValueError(f"Unknown mag_limit_band '{self.mag_limit_band}'. "
                             "Choose 'G' or 'G_RVS'")
        if self.selection_function not in ("none", "dr3_rvs"):
            raise ValueError(f"Unknown selection_function "
                             f"'{self.selection_function}'. "
                             "Choose 'none' or 'dr3_rvs'")
        if self.sf_mag_shift is None:
            self.sf_mag_shift = 2.5 if self.gaia_release.lower() in ("dr4", "dr5") \
                else 0.0

    @property
    def limit_band_column(self):
        """Catalogue column that mag_limit applies to ('Gmag' or 'G_RVSmag')."""
        return "G_RVSmag" if self.mag_limit_band == "G_RVS" else "Gmag"

    def frame(self, snapshot=None):
        """The Galactic frame of this run's observer (see frame.py).  With
        rotation="auto" the snapshot is needed to detect the rotation."""
        from .frame import GalacticFrame
        return GalacticFrame.from_config(self, snapshot)

    def print_summary(self):
        """Print a formatted summary of all run parameters."""
        lines = [
            "",
            "=" * 64,
            "  SNAPDRAGONS v2.0 -- Population Synthesis Configuration",
            "=" * 64,
            "",
            "  Observer",
            f"    Position (x,y,z) : ({self.observer_position[0]:.2f}, "
            f"{self.observer_position[1]:.2f}, "
            f"{self.observer_position[2]:.3f}) kpc  [simulation coordinates]",
            ("    Velocity (vx,vy,vz): "
             f"({self.observer_velocity[0]:.2f}, {self.observer_velocity[1]:.2f}, "
             f"{self.observer_velocity[2]:.2f}) km/s  [as given]"
             if self.observer_velocity is not None else
             "    Velocity         : V_circular along the disc rotation + solar "
             "motion (reported with the Galactic frame)"),
            f"    Disc rotation    : {self.rotation}"
            + ("  (detected from the star particles)" if self.rotation == "auto"
               else "  (seen from +z; as set)"),
            f"    Solar motion (U,V,W): ({self.solar_U}, {self.solar_V}, "
            f"{self.solar_W}) km/s  [Schoenrich+ 2010]",
            f"    V_circular       : {self.v_circular} km/s",
            "",
            "  IMF",
            f"    Type             : {self.imf}",
            f"    Mass range       : [{self.m_min}, {self.m_max}] Msun",
            f"    Sampling         : {self.sampling}",
            "",
            "  Isochrones",
            f"    Source           : {self.isochrone_source}",
            f"    Photometric sys  : {self.photometric_system}",
            f"    log(age) grid    : [{self.logage_min}, {self.logage_max}], "
            f"step {self.logage_step}",
            f"    [M/H] grid       : [{self.mh_min}, {self.mh_max}], "
            f"step {self.mh_step}",
            "",
            "  Gaia",
            f"    Data release     : {self.gaia_release.upper()}",
            f"    Magnitude limit  : {self.mag_limit_band} < {self.mag_limit}"
            f"  (G > {self.mag_bright_limit})",
            f"    Selection fn     : {self.selection_function}"
            + (f"  (DR3 RVS evaluated at G - {self.sf_mag_shift})"
               if self.selection_function != "none" and self.sf_mag_shift
               else ""),
            f"    Apply errors     : {self.apply_errors}",
            "",
            "  Unresolved binaries",
            f"    Enabled          : {'ON' if self.unresolved_binaries else 'OFF'}",
        ]
        if self.unresolved_binaries:
            lines += [
                f"    f_bin(1 Msun)    : {self.f_binary}  [mass-dependent, Moe+ 2017]",
                f"    Min mass ratio   : {self.q_min}",
            ]
        lines += [
            "",
            "  Phase-space smoothing",
            f"    Enabled          : {'ON' if self.phase_space_smoothing else 'OFF'}",
        ]
        if self.phase_space_smoothing:
            lines += [
                f"    Neighbours       : {self.smoothing_ngb}",
                f"    Scale factor     : {self.smoothing_scale}",
            ]
        lines += [
            "",
            "  Extinction",
            f"    Mode             : {self.extinction_mode}",
        ]
        if self.extinction_mode == "mwdust":
            lines.append(f"    Dust map         : {self.mwdust_map}")
        elif self.extinction_mode == "simulation":
            lines.append(f"    Dust calibration : {self.dust_to_gas_ratio} (1.0 = solar)")
        lines += [
            f"    R_V              : {self.rv_extinction}",
            "",
            "  Simulation corrections",
            f"    tform correction : {'ON' if self.apply_tform_correction else 'OFF'}",
            "",
            "  Output",
            f"    Path             : {self.output_path}",
            f"    Format           : {self.output_format}",
            f"    Coordinates      : {', '.join(self.output_coords)}",
            f"    Down-sample frac : {self.fsample}",
            f"    Diagnostic plots : {'ON -> ' + self.plot_dir if self.make_plots else 'OFF'}",
            "",
            "=" * 64,
            "",
        ]
        print("\n".join(lines))
