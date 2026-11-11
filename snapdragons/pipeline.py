"""Top-level pipeline orchestrator.

Ties together all modules into a single ``run()`` call, with
configuration logging and progress reporting.
"""

import time
import argparse
import numpy as np

from .config import Config
from .simulation import SimulationSnapshot
from .isochrones import IsochroneGrid
from .population import synthesise_population
from .extinction import get_extinction_calculator
from .errors import GaiaErrorModel
from .catalogue import build_catalogue, write_catalogue
from .diagnostics import make_diagnostic_plots
from .frame import GalacticFrame


class Pipeline:
    """Snapdragons population synthesis pipeline."""

    def __init__(self, config=None, **kwargs):
        if config is None:
            config = Config(**kwargs)
        self.config = config

    def run(self, snapshot=None):
        """Execute the full pipeline.

        Parameters
        ----------
        snapshot : SimulationSnapshot, optional
            If None, loads from config.snapshot_path.

        Returns
        -------
        catalogue : dict
        """
        t0 = time.time()

        self.config.print_summary()

        rng = np.random.default_rng(self.config.random_seed)

        # --- Load simulation ---
        print("Step 1/5: Loading simulation snapshot")
        if snapshot is None:
            if self.config.snapshot_path is None:
                raise ValueError("No snapshot_path configured and no "
                                 "snapshot provided.")
            snapshot = SimulationSnapshot.from_pynbody(
                self.config.snapshot_path,
                load_gas=(self.config.extinction_mode == "simulation"),
                param_file=self.config.param_file,
                apply_tform_correction=self.config.apply_tform_correction,
            )

        # --- Galactic frame of the observer ---
        print()
        frame = GalacticFrame.from_config(self.config, snapshot)
        frame.print_report()

        # --- Load isochrones ---
        print("\nStep 2/5: Loading isochrones")
        iso_grid = IsochroneGrid(self.config)
        iso_grid.load_or_download()

        # --- Set up extinction ---
        print("\nStep 3/5: Setting up extinction")
        ext_calc = get_extinction_calculator(self.config, snapshot)
        print(f"  Extinction mode: {self.config.extinction_mode}")

        # --- Population synthesis ---
        print("\nStep 4/5: Population synthesis")
        raw_cat = synthesise_population(
            snapshot, iso_grid, self.config,
            extinction_calc=ext_calc, frame=frame,
        )

        if raw_cat is None:
            print("\n  No stars generated. Exiting.")
            return None

        # --- Build and write catalogue ---
        print("\nStep 5/5: Building catalogue and applying errors")
        error_model = GaiaErrorModel(self.config) if self.config.apply_errors \
            else None

        catalogue = build_catalogue(
            raw_cat, self.config,
            error_model=error_model,
            rng=rng, frame=frame,
        )

        if catalogue is not None:
            write_catalogue(catalogue, self.config)

            if self.config.make_plots:
                print("\nStep 6: Generating diagnostic plots")
                make_diagnostic_plots(
                    catalogue, self.config,
                    output_dir=self.config.plot_dir,
                )

            n = 0
            for v in catalogue.get("true", {}).values():
                if isinstance(v, np.ndarray):
                    n = len(v)
                    break
            elapsed = time.time() - t0
            print(f"\n{'=' * 64}")
            print(f"  Complete: {n:,} stars in {elapsed:.1f}s")
            print(f"  Output: {self.config.output_path}")
            if self.config.make_plots:
                print(f"  Plots:  {self.config.plot_dir}/")
            print(f"{'=' * 64}\n")

        return catalogue


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Snapdragons: population synthesis from galaxy simulations"
    )
    parser.add_argument("snapshot", help="Path to simulation snapshot file")
    parser.add_argument("-o", "--output", default="output/snapdragons_catalogue.h5",
                        help="Output catalogue path")
    parser.add_argument("--format", default="hdf5",
                        choices=["hdf5", "fits"],
                        help="Output format")
    parser.add_argument("--release", default="dr3",
                        choices=["dr2", "dr3", "dr4", "dr5"],
                        help="Gaia data release for error model")
    parser.add_argument("--param", default=None,
                        help="Path to ChaNGa .param file")
    parser.add_argument("--mag-limit", type=float, default=16.0,
                        help="Apparent faint magnitude limit (default 16)")
    parser.add_argument("--mag-band", default="G_RVS",
                        choices=["G_RVS", "G"],
                        help="Band the faint limit applies to (default G_RVS)")
    parser.add_argument("--selection-function", default="none",
                        choices=["none", "dr3_rvs"],
                        help="RVS selection function (gaiaunlimited DR3)")
    parser.add_argument("--sf-shift", type=float, default=None,
                        help="Evaluate the DR3 selection function at G - shift "
                             "(default 2.5 for dr4/dr5, else 0)")
    parser.add_argument("--extinction", default="mwdust",
                        choices=["mwdust", "simulation", "none"],
                        help="Extinction mode")
    parser.add_argument("--dust-map", default="Combined19",
                        help="mwdust map name")
    parser.add_argument("--imf", default="kroupa",
                        choices=["kroupa", "salpeter"],
                        help="Initial mass function")
    parser.add_argument("--sampling", default="mass_budget",
                        choices=["mass_budget", "poisson", "deterministic"],
                        help="Star sampling method")
    parser.add_argument("--coords", nargs="+",
                        default=["equatorial"],
                        choices=["equatorial", "galactic", "cartesian"],
                        help="Output coordinate systems")
    parser.add_argument("--no-errors", action="store_true",
                        help="Skip Gaia error convolution")
    parser.add_argument("--no-tform-correction", action="store_true",
                        help="Disable ChaNGa tform correction")
    parser.add_argument("--plots", action="store_true",
                        help="Generate diagnostic plots")
    parser.add_argument("--plot-dir", default="output/plots",
                        help="Directory for diagnostic plots")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--fsample", type=float, default=1.0,
                        help="Down-sampling fraction")
    parser.add_argument("--obs-x", type=float, default=-8.2,
                        help="Observer x position [kpc]")
    parser.add_argument("--obs-y", type=float, default=0.0,
                        help="Observer y position [kpc]")
    parser.add_argument("--obs-z", type=float, default=0.02,
                        help="Observer z position [kpc]")
    parser.add_argument("--rotation", default="auto",
                        choices=["auto", "clockwise", "counterclockwise"],
                        help="Sense of disc rotation seen from +z "
                             "(default: detected from the stars)")

    args = parser.parse_args()

    config = Config(
        snapshot_path=args.snapshot,
        param_file=args.param,
        output_path=args.output,
        output_format=args.format,
        gaia_release=args.release,
        mag_limit=args.mag_limit,
        mag_limit_band=args.mag_band,
        selection_function=args.selection_function,
        sf_mag_shift=args.sf_shift,
        extinction_mode=args.extinction,
        mwdust_map=args.dust_map,
        imf=args.imf,
        sampling=args.sampling,
        output_coords=args.coords,
        apply_errors=not args.no_errors,
        apply_tform_correction=not args.no_tform_correction,
        make_plots=args.plots,
        plot_dir=args.plot_dir,
        random_seed=args.seed,
        fsample=args.fsample,
        observer_position=(args.obs_x, args.obs_y, args.obs_z),
        rotation=args.rotation,
    )

    pipeline = Pipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
