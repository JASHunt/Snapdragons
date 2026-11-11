"""Golden regression: fixed-seed pipeline runs against stored summaries.

Two scenarios are run end to end on the mock snapshot and every output
column is reduced to (count of finite values, mean, min, max).  The result
must match ``tests/data/golden.json``.

This is the test that answers "did my change alter the catalogue?".  It is
expected to fail whenever the output changes on purpose (a bug fix, a new
column, a different order of random draws).  When it does:

1. read the report: it names every column that moved and by how much;
2. if the change is intended, regenerate the baseline and commit it with
   the code change::

       python -m pytest tests/test_regression.py --update-golden

The stored numbers also depend on the random streams of numpy and on the
PyGaia error tables, so a major upgrade of either can need a regeneration
with no change to Snapdragons.  The versions used are recorded in the file.
"""

import json
import os

import numpy as np
import pytest

from snapdragons import Pipeline

from conftest import DATA_DIR

pygaia = pytest.importorskip("pygaia")

GOLDEN = os.path.join(DATA_DIR, "golden.json")
RTOL = 1e-7

SCENARIOS = {
    # the plain path: G_RVS-limited DR3 mock, no extinction
    "baseline": dict(snapshot="plain", config=dict(
        gaia_release="dr3", mag_limit=12.0, mag_limit_band="G_RVS", random_seed=42)),
    # everything switched on
    "all_options": dict(snapshot="gas", config=dict(
        gaia_release="dr4", mag_limit=12.5, mag_limit_band="G", random_seed=42,
        extinction_mode="simulation", unresolved_binaries=True,
        phase_space_smoothing=True, smoothing_ngb=16)),
}


def summarise(catalogue):
    out = {}
    for group in ("true", "observed", "uncertainties", "stellar"):
        for key, arr in sorted(catalogue.get(group, {}).items()):
            a = np.asarray(arr, dtype=np.float64)
            fin = a[np.isfinite(a)]
            out[f"{group}/{key}"] = dict(
                n=int(len(a)), n_finite=int(len(fin)),
                mean=float(fin.mean()) if len(fin) else None,
                min=float(fin.min()) if len(fin) else None,
                max=float(fin.max()) if len(fin) else None)
    return out


def versions():
    import importlib.metadata as md
    import scipy
    import snapdragons
    return dict(snapdragons=snapdragons.__version__, numpy=np.__version__,
                scipy=scipy.__version__, pygaia=md.version("pygaia"))


def compare(name, got, ref):
    problems = []
    for col in sorted(set(ref) - set(got)):
        problems.append(f"{name}: column {col} disappeared")
    for col in sorted(set(got) - set(ref)):
        problems.append(f"{name}: new column {col}")
    for col in sorted(set(got) & set(ref)):
        g, r = got[col], ref[col]
        for stat in ("n", "n_finite"):
            if g[stat] != r[stat]:
                problems.append(f"{name}: {col} {stat} {r[stat]} -> {g[stat]}")
        for stat in ("mean", "min", "max"):
            a, b = g[stat], r[stat]
            if (a is None) != (b is None):
                problems.append(f"{name}: {col} {stat} {b} -> {a}")
            elif a is not None and not np.isclose(a, b, rtol=RTOL, atol=1e-10):
                problems.append(f"{name}: {col} {stat} {b:.10g} -> {a:.10g}")
    return problems


@pytest.fixture(scope="module")
def results(request, mock_snapshot, mock_snapshot_gas, tmp_path_factory):
    import shutil
    from snapdragons import Config, IsochroneGrid
    from conftest import ALL_COORDS, PARSEC_SUBSET
    snaps = {"plain": mock_snapshot, "gas": mock_snapshot_gas}
    out = {}
    for name, sc in SCENARIOS.items():
        tmp = tmp_path_factory.mktemp(name)
        kwargs = dict(isochrone_cache_dir=str(tmp / "iso"), output_path=str(tmp / "cat.h5"),
                      extinction_mode="none", output_coords=list(ALL_COORDS))
        kwargs.update(sc["config"])
        cfg = Config(**kwargs)
        shutil.copy(PARSEC_SUBSET, IsochroneGrid(cfg)._cache_path())
        out[name] = summarise(Pipeline(cfg).run(snapshot=snaps[sc["snapshot"]]))
    return out


def test_golden(request, results):
    if request.config.getoption("--update-golden"):
        with open(GOLDEN, "w") as f:
            json.dump(dict(versions=versions(), scenarios=results), f, indent=1, sort_keys=True)
            f.write("\n")
        pytest.skip(f"golden baseline rewritten: {GOLDEN}")
    if not os.path.exists(GOLDEN):
        pytest.fail("tests/data/golden.json is missing: run with --update-golden")
    with open(GOLDEN) as f:
        stored = json.load(f)
    problems = []
    for name in SCENARIOS:
        if name not in stored["scenarios"]:
            problems.append(f"{name}: no stored baseline")
            continue
        problems += compare(name, results[name], stored["scenarios"][name])
    if problems:
        head = (f"{len(problems)} differences from the golden baseline "
                f"(stored with {stored['versions']}, now {versions()}).  "
                "If intended: pytest tests/test_regression.py --update-golden\n")
        pytest.fail(head + "\n".join(problems[:60])
                    + ("\n..." if len(problems) > 60 else ""), pytrace=False)


def test_scenarios_are_not_trivial(results):
    base, full = results["baseline"], results["all_options"]
    assert base["true/ra"]["n"] > 10_000 and full["true/ra"]["n"] > 5_000
    assert base["true/A_V"]["max"] == 0.0 and full["true/A_V"]["mean"] > 0.05
    assert "stellar/is_binary" in full and "stellar/is_binary" not in base
    assert 0.2 < full["stellar/is_binary"]["mean"] < 0.8
