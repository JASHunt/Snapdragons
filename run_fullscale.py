"""Full-scale fsample=1 test for simulation and mwdust extinction."""

import time
import numpy as np
from snapdragons.pipeline import Pipeline
from snapdragons.config import Config

MODES = [
    ("simulation", "output/full_simulation.h5", "output/plots_full_sim"),
    ("mwdust",     "output/full_mwdust.h5",     "output/plots_full_mwd"),
]

for mode, outpath, plotdir in MODES:
    print(f"\n{'#' * 72}")
    print(f"# FULL-SCALE RUN: extinction={mode}")
    print(f"{'#' * 72}\n")

    t0 = time.time()

    config = Config(
        snapshot_path="data_examples/AMNH.010000",
        param_file="data_examples/AMNH.param",
        extinction_mode=mode,
        output_path=outpath,
        output_format="hdf5",
        output_coords=["equatorial", "galactic", "cartesian"],
        apply_errors=True,
        make_plots=True,
        plot_dir=plotdir,
        fsample=1.0,
        random_seed=42,
    )

    pipeline = Pipeline(config)
    cat = pipeline.run()

    elapsed = time.time() - t0

    if cat and "true" in cat:
        av = cat["true"].get("A_V", np.array([]))
        n = len(av)
        print(f"\n>>> {mode}: {n:,} final stars in {elapsed:.0f}s")
        print(f"    A_V: median={np.median(av):.3f}, "
              f"mean={np.mean(av):.3f}, max={np.max(av):.2f}")
        if "Gmag_app" in cat["true"]:
            g = cat["true"]["Gmag_app"]
            print(f"    Gmag_app: median={np.median(g):.2f}, "
                  f"min={np.min(g):.2f}, max={np.max(g):.2f}")
    else:
        print(f"\n>>> {mode}: FAILED or no stars. Elapsed {elapsed:.0f}s")

    del cat, pipeline
    import gc; gc.collect()

print("\n\nDone.")
