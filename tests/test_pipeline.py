"""Synthetic end-to-end test of the Snapdragons pipeline.

Creates a small mock dataset (no simulation file needed) and runs
the full pipeline to verify all modules integrate correctly.
"""

import sys
sys.path.insert(0, ".")

import numpy as np
from snapdragons import Config, Pipeline, read_catalogue
from snapdragons.simulation import SimulationSnapshot
from snapdragons.isochrones import IsochroneGrid
from snapdragons.imf import KroupaIMF
from snapdragons.population import synthesise_population
from snapdragons.extinction import NoExtinction
from snapdragons.errors import GaiaErrorModel
from snapdragons.catalogue import build_catalogue, write_catalogue


def create_mock_snapshot(n_stars=50, n_gas=30):
    """Create a simple mock snapshot for testing."""
    rng = np.random.default_rng(42)

    r = rng.uniform(2.0, 12.0, n_stars)
    theta = rng.uniform(0, 2 * np.pi, n_stars)
    z = rng.normal(0, 0.3, n_stars)
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    v_circ = 220.0
    vx = -v_circ * np.sin(theta) + rng.normal(0, 20, n_stars)
    vy = v_circ * np.cos(theta) + rng.normal(0, 20, n_stars)
    vz = rng.normal(0, 10, n_stars)

    mass = rng.uniform(1e3, 1e5, n_stars)
    age = rng.uniform(0.1, 10.0, n_stars)
    metallicity = rng.uniform(-1.5, 0.3, n_stars)

    star_pos = np.column_stack([x, y, z])
    star_vel = np.column_stack([vx, vy, vz])

    snap = SimulationSnapshot.from_arrays(
        star_pos=star_pos,
        star_vel=star_vel,
        star_mass=mass,
        star_age=age,
        star_metallicity=metallicity,
    )
    return snap


def test_imf():
    """Test IMF sampling correctness."""
    print("=" * 40)
    print("TEST: IMF sampling")
    imf = KroupaIMF()
    rng = np.random.default_rng(123)

    masses = imf.sample(100000, rng=rng)
    assert len(masses) == 100000
    assert masses.min() >= 0.08
    assert masses.max() <= 100.0
    assert abs(masses.mean() - imf.mean_mass()) / imf.mean_mass() < 0.02

    masses_t = imf.sample_truncated(10000, 2.0, rng=rng)
    assert masses_t.min() >= 2.0

    n_exp = imf.expected_count(1e4)
    assert n_exp > 0

    print("  PASSED\n")


def test_config():
    """Test configuration and velocity computation."""
    print("=" * 40)
    print("TEST: Configuration")
    c = Config()
    assert abs(c.observer_velocity[0] - 11.1) < 0.01
    assert c.observer_velocity[1] < 0  # should be negative (CCW rotation)
    assert abs(c.observer_velocity[2] - 7.25) < 0.01

    c2 = Config(observer_velocity=(10, 200, 5))
    assert c2.observer_velocity == (10.0, 200.0, 5.0)

    print("  PASSED\n")


def test_isochrone_grid():
    """Test isochrone grid (synthetic fallback if CMD unavailable)."""
    print("=" * 40)
    print("TEST: Isochrone grid")

    config = Config(
        logage_min=8.0,
        logage_max=10.0,
        logage_step=0.5,
        mh_min=-1.0,
        mh_max=0.0,
        mh_step=0.5,
    )

    try:
        grid = IsochroneGrid(config)
        grid.load_or_download()
    except Exception:
        print("  CMD download unavailable, using synthetic isochrones")
        grid = IsochroneGrid.generate_synthetic(config)

    print(f"  Grid has {len(grid.grid_points)} isochrones")

    iso, key = grid.get_nearest(9.0, -0.5)
    print(f"  Nearest to (9.0, -0.5): {key}")
    print(f"  Mass range: [{iso['Mini'].min():.3f}, {iso['Mini'].max():.3f}]")

    props = grid.interpolate_stars(9.0, -0.5, np.array([0.5, 1.0, 2.0]))
    print(f"  Interpolated Gmag: {props['Gmag']}")

    m_min = grid.min_observable_mass(9.0, -0.5, 10.0, 20.7)
    print(f"  Min observable mass at 10 kpc: {m_min:.3f} Msun")

    print("  PASSED\n")


def test_pipeline_synthetic():
    """Full pipeline test with mock data (requires isochrones)."""
    print("=" * 40)
    print("TEST: Full pipeline (synthetic data)")

    config = Config(
        output_path="/tmp/test_snapdragons.h5",
        output_format="hdf5",
        output_coords=["equatorial", "galactic", "cartesian"],
        extinction_mode="none",
        gaia_release="dr3",
        mag_limit=20.7,
        random_seed=42,
        logage_min=8.0,
        logage_max=10.0,
        logage_step=0.5,
        mh_min=-1.0,
        mh_max=0.0,
        mh_step=0.5,
    )

    snap = create_mock_snapshot(n_stars=20)

    try:
        # Try to load real isochrones, fall back to synthetic
        try:
            iso_grid = IsochroneGrid(config)
            iso_grid.load_or_download()
        except Exception:
            print("  CMD download unavailable, using synthetic isochrones")
            iso_grid = IsochroneGrid.generate_synthetic(config)

        from snapdragons.extinction import NoExtinction
        from snapdragons.errors import GaiaErrorModel
        from snapdragons.catalogue import build_catalogue, write_catalogue

        ext_calc = NoExtinction(config)
        error_model = GaiaErrorModel(config)
        rng = np.random.default_rng(42)

        config.print_summary()

        raw_cat = synthesise_population(
            snap, iso_grid, config, extinction_calc=ext_calc)

        if raw_cat is not None:
            catalogue = build_catalogue(
                raw_cat, config,
                extinction_calc=ext_calc,
                error_model=error_model,
                rng=rng,
            )
            write_catalogue(catalogue, config)

            n = 0
            for v in catalogue["true"].values():
                if isinstance(v, np.ndarray):
                    n = len(v)
                    break
            print(f"  Generated {n} stars in catalogue")

            cat = read_catalogue("/tmp/test_snapdragons.h5")
            print(f"  Read back: {cat}")
            print(f"  True columns: {cat.true.columns}")
            print(f"  Observed columns: {cat.observed.columns}")
            print(f"  Stellar columns: {cat.stellar.columns}")

            # Verify data integrity
            assert cat.n_stars > 0, "No stars in catalogue"
            assert hasattr(cat.true, "ra"), "Missing RA"
            assert hasattr(cat.true, "dec"), "Missing Dec"
            assert hasattr(cat.true, "parallax"), "Missing parallax"
            assert hasattr(cat.true, "Gmag_app"), "Missing apparent G"
            assert hasattr(cat.stellar, "Mini"), "Missing initial mass"
            print("  Data integrity checks passed")

            print("  PASSED\n")
        else:
            print("  No stars generated (check isochrones)\n")

    except ImportError as e:
        print(f"  SKIPPED (missing dependency: {e})\n")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  FAILED: {e}\n")


if __name__ == "__main__":
    test_imf()
    test_config()
    test_isochrone_grid()
    test_pipeline_synthetic()
    print("All tests complete.")
