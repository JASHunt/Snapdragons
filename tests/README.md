# Snapdragons tests

    pip install -e ".[test]"
    python -m pytest                      # ~20 s, no network, no simulation file
    python -m pytest --run-external       # also the tests that need downloaded data
    python -m pytest tests/test_imf.py -k truncated

The suite is self-contained.  `data/parsec_subset.h5` is a 12-cell subset of the PARSEC grid
(log age 8, 9, 9.7, 10; [M/H] −1, −0.5, 0) in Snapdragons' own cache format, and `conftest.py`
builds a 300-particle disc patch around the default observer whose ages and metallicities sit on
those cells.  A full synthesis on it takes under 0.1 s, so most tests run the real code on real
isochrones rather than on stubs.

| file | what it pins down |
|---|---|
| `test_imf.py` | Kroupa / Salpeter CDF, inverse, mean mass and expected counts against numerical integrals; samples against the CDF (KS) |
| `test_photometry.py` | Sartoretti+23 G_RVS: hand-computed values, colour range, agreement with PyGaia |
| `test_config.py` | defaults, observer velocity from the solar motion, validation, run summary |
| `test_frame.py` | rotation sense detected from the stars, Galactic axes and observer velocity for either sense and any observer azimuth, Sgr A*'s proper motion, agreement with astropy's Galactocentric frame, the printed report, and that the mock sky is identical star for star however the simulation is oriented |
| `test_coordinates.py` | Cartesian → Galactic → ICRS against astropy, units of proper motion, round trip back to simulation coordinates in any frame |
| `test_extinction.py` | band coefficients, gas ray-march against the analytic column of a uniform medium, scalings |
| `test_isochrones.py` | cache loading, nearest-cell lookup, interpolation, limiting mass, CMD parsing and the download path with the HTTP calls replaced |
| `test_smoothing.py` | kNN lengths against a known density, Epanechnikov draws against the kernel |
| `test_population.py` | mass-budget sampling is unbiased, magnitude limits in G and G_RVS, truncation loses no stars, extinction, binaries, smoothing, reproducibility |
| `test_errors.py` | PyGaia units (mas, mag, km/s), the arguments of the RV call, pull distributions, magnitude range, fallback model |
| `test_catalogue.py` | columns per coordinate system, re-applied cuts, selection function, HDF5 / FITS round trips, the reader |
| `test_simulation.py` | array container, metallicity units, tform remap, .param link, levelling a tilted disc without turning it over |
| `test_pipeline.py` | end-to-end runs, plots, snapshot loading options, command line ↔ Config |
| `test_regression.py` | golden baseline: every output column of two fixed-seed runs |

## When you add or change something

1. Run the suite.  A change that is not meant to alter the catalogue must leave everything green.
2. If `test_regression.py::test_golden` fails, its report lists each output column that moved and
   by how much.  If that is what you intended, regenerate the baseline and commit it with the
   code:

       python -m pytest tests/test_regression.py --update-golden

   The baseline also depends on numpy's random streams and PyGaia's error tables (versions are
   stored in `data/golden.json`), so a major upgrade of either can need the same step.
3. Add tests for the new behaviour next to the module's existing ones.  The fixtures to reach for:
   `make_config(**overrides)` (no extinction, seed 42, G_RVS < 12, output under a temp directory),
   `iso_grid`, `mock_snapshot` / `mock_snapshot_gas` (clockwise; `build_mock_snapshot(rotation="counterclockwise")` for the other sense), `make_pipeline_config` (as `make_config`,
   with the isochrone cache pre-seeded so `Pipeline.run` needs no network), `ConstantExtinction`.
4. A known bug that is not being fixed yet goes in as a test of the correct behaviour marked
   `@pytest.mark.xfail(strict=True, reason=...)`: the suite stays green, and turns red the day
   the bug is fixed, which is the reminder to remove the mark.

## Not covered

- `SimulationSnapshot.from_pynbody` on a real snapshot (needs pynbody and a file); its helpers are.
- The live CMD web form (the request and parsing are tested against a canned reply).
- mwdust maps and the gaiaunlimited selection function unless `--run-external` is given (they
  download ~1 GB and ~95 MB on first use).
- Plot contents: the plots are only checked to be written.
