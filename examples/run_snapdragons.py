"""Example: run Snapdragons on a simulation snapshot.

Usage
-----
    python run_snapdragons.py /path/to/snapshot.tipsy

Or from Python:

    from snapdragons import Config, Pipeline

    config = Config(snapshot_path="snapshot.tipsy")
    pipeline = Pipeline(config)
    catalogue = pipeline.run()
"""

from snapdragons import Config, Pipeline, read_catalogue


def run_basic(snapshot_path):
    """Run with default settings (Gaia DR3, Kroupa IMF, mwdust extinction)."""
    config = Config(
        snapshot_path=snapshot_path,
        output_path="catalogue_dr3.h5",
    )
    pipeline = Pipeline(config)
    catalogue = pipeline.run()
    return catalogue


def run_custom(snapshot_path):
    """Run with custom settings."""
    config = Config(
        snapshot_path=snapshot_path,

        # Observer at custom position
        observer_position=(-8.2, 0.0, 0.02),
        solar_U=11.1,
        solar_V=12.24,
        solar_W=7.25,
        v_circular=232.8,

        # Gaia DR4 with deeper magnitude limit
        gaia_release="dr4",
        mag_limit=20.7,

        # Output all coordinate systems
        output_coords=["equatorial", "galactic", "cartesian"],
        output_format="hdf5",
        output_path="catalogue_dr4_full.h5",

        # Use simulation-based extinction from gas particles
        extinction_mode="simulation",
        dust_to_gas_ratio=0.01,

        # Reproducible
        random_seed=42,
    )
    pipeline = Pipeline(config)
    catalogue = pipeline.run()
    return catalogue


def run_no_extinction(snapshot_path):
    """Run without extinction (useful for testing)."""
    config = Config(
        snapshot_path=snapshot_path,
        extinction_mode="none",
        output_path="catalogue_noext.h5",
    )
    pipeline = Pipeline(config)
    return pipeline.run()


def read_and_inspect(filepath="catalogue_dr3.h5"):
    """Read a previously generated catalogue."""
    cat = read_catalogue(filepath)
    print(cat)
    print(f"\nTrue columns: {cat.true.columns}")
    print(f"Observed columns: {cat.observed.columns}")
    print(f"Stellar columns: {cat.stellar.columns}")

    # Access data as attributes
    print(f"\nRA range: [{cat.true.ra.min():.2f}, {cat.true.ra.max():.2f}] deg")
    print(f"G mag range: [{cat.true.Gmag_app.min():.2f}, "
          f"{cat.true.Gmag_app.max():.2f}]")
    print(f"Parallax range: [{cat.true.parallax.min():.4f}, "
          f"{cat.true.parallax.max():.4f}] mas")

    # Convert to pandas
    df = cat.to_dataframe()
    print(f"\nDataFrame shape: {df.shape}")
    print(df.describe())

    return cat


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python run_snapdragons.py <snapshot_path>")
        print("       python run_snapdragons.py --read <catalogue_path>")
        sys.exit(1)

    if sys.argv[1] == "--read":
        read_and_inspect(sys.argv[2])
    else:
        run_basic(sys.argv[1])
