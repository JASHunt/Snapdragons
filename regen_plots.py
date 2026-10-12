"""Regenerate diagnostic plots from stored HDF5 output files."""

import h5py
import numpy as np
from snapdragons.config import Config
from snapdragons.diagnostics import make_diagnostic_plots


def load_catalogue(h5_path):
    cat = {}
    with h5py.File(h5_path, "r") as f:
        for group in f:
            cat[group] = {}
            for key in f[group]:
                cat[group][key] = f[group][key][:]
    return cat


RUNS = [
    ("output/full_simulation.h5", "simulation", "output/plots_full_sim"),
    ("output/full_mwdust.h5",     "mwdust",     "output/plots_full_mwd"),
]

for h5_path, ext_mode, plot_dir in RUNS:
    print(f"\n{'=' * 60}")
    print(f"Regenerating plots: {ext_mode} from {h5_path}")
    print(f"{'=' * 60}")

    cat = load_catalogue(h5_path)
    n = len(next(iter(cat["true"].values())))
    print(f"  Loaded {n:,} stars")

    config = Config(
        extinction_mode=ext_mode,
        output_coords=["equatorial", "galactic", "cartesian"],
        apply_errors=True,
        make_plots=True,
        plot_dir=plot_dir,
    )

    make_diagnostic_plots(cat, config, output_dir=plot_dir)
    del cat

print("\nDone.")
