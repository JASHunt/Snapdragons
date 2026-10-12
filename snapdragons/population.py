"""Core population synthesis engine.

Takes simulation star particles and splits each into a constituent
population of individual stars by:
  1. Assigning particles to isochrone grid cells
  2. Batch-computing minimum observable masses per cell
  3. Sampling stellar masses from the truncated IMF
  4. Batch-interpolating isochrone properties per cell
"""

import numpy as np
from collections import defaultdict
from scipy.interpolate import interp1d
from tqdm import tqdm

from .imf import get_imf
from .coordinates import cartesian_to_galactic
from .extinction import EXTINCTION_COEFFS


def _vectorised_mass_budget(imf, particle_masses, m_min_trunc, m_max,
                            rng, full_integral, integral_cache):
    """Draw stars from the IMF for a batch of particles sharing m_min_trunc.

    Draws one large pool from the truncated IMF, assigns consecutive
    chunks to particles, and uses cumsum + searchsorted to find where
    each particle's mass budget is exhausted.
    """
    n_particles = len(particle_masses)
    counts = np.zeros(n_particles, dtype=np.intp)

    cache_key = round(m_min_trunc, 8)
    if cache_key not in integral_cache:
        trunc_integral = _mass_integral(imf, m_min_trunc, m_max)
        mean_m = imf.mean_mass(m_min_trunc, m_max)
        integral_cache[cache_key] = (trunc_integral, mean_m)

    trunc_integral, mean_m = integral_cache[cache_key]

    if trunc_integral <= 0 or mean_m <= 0:
        return np.array([]), counts

    ratio = trunc_integral / full_integral
    budgets = particle_masses * ratio
    valid = budgets > 0
    n_valid = int(np.sum(valid))
    if n_valid == 0:
        return np.array([]), counts

    v_idx = np.where(valid)[0]
    v_budgets = budgets[valid]

    n_per = np.maximum((v_budgets / mean_m * 1.5).astype(np.intp) + 10, 20)
    total_pool = int(np.sum(n_per))

    pool = imf.sample_truncated(total_pool, m_min_trunc, rng=rng)
    if len(pool) == 0:
        return np.array([]), counts

    pool_cs = np.cumsum(pool)

    starts = np.empty(n_valid, dtype=np.intp)
    starts[0] = 0
    if n_valid > 1:
        np.cumsum(n_per[:-1], out=starts[1:])

    cs_offsets = np.where(starts > 0, pool_cs[starts - 1], 0.0)
    thresholds = cs_offsets + v_budgets

    cutoffs = np.searchsorted(pool_cs, thresholds, side='right')
    ends = starts + n_per
    cutoffs = np.minimum(cutoffs, ends)

    n_clean = cutoffs - starts

    has_boundary = cutoffs < ends
    boundary_gidx = np.clip(cutoffs, 0, len(pool) - 1)

    prev_cs = np.where(
        n_clean > 0,
        pool_cs[np.clip(cutoffs - 1, 0, len(pool_cs) - 1)],
        cs_offsets,
    )
    remaining = v_budgets - (prev_cs - cs_offsets)
    remaining = np.maximum(remaining, 0.0)

    boundary_mass = pool[boundary_gidx]
    accept_prob = np.where(
        has_boundary & (boundary_mass > 0),
        np.minimum(remaining / boundary_mass, 1.0),
        0.0,
    )
    accept = rng.uniform(size=n_valid) < accept_prob

    n_boundary = (accept & has_boundary).astype(np.intp)
    n_total = n_clean + n_boundary

    total_stars = int(np.sum(n_total))
    if total_stars == 0:
        return np.array([]), counts

    result = np.empty(total_stars)
    result_offsets = np.cumsum(n_total) - n_total

    total_clean = int(np.sum(n_clean))
    if total_clean > 0:
        within = np.arange(total_clean) - np.repeat(
            np.cumsum(n_clean) - n_clean, n_clean)
        pool_idx = np.repeat(starts, n_clean) + within
        result_idx = np.repeat(result_offsets, n_clean) + within
        result[result_idx] = pool[pool_idx]

    if int(np.sum(n_boundary)) > 0:
        amask = accept & has_boundary
        result[result_offsets[amask] + n_clean[amask]] = pool[boundary_gidx[amask]]

    counts[v_idx] = n_total
    return result, counts


def _vectorised_count_sample(imf, particle_masses, m_min_trunc, sampling, rng):
    """Draw stars for a batch using Poisson or deterministic counts."""
    n_particles = len(particle_masses)

    count_per_mass = imf.expected_count(1.0, m_min_trunc=m_min_trunc)
    if count_per_mass <= 0:
        return np.array([]), np.zeros(n_particles, dtype=np.intp)

    n_expected = particle_masses * count_per_mass

    if sampling == "poisson":
        n_stars = rng.poisson(n_expected).astype(np.intp)
    else:
        n_stars = np.round(n_expected).astype(np.intp)

    n_stars = np.maximum(n_stars, 0)
    total_n = int(np.sum(n_stars))
    if total_n == 0:
        return np.array([]), n_stars

    masses = imf.sample_truncated(total_n, m_min_trunc, rng=rng)
    return masses, n_stars


def _mass_integral(imf, m_lo, m_hi):
    """Total mass in stars between m_lo and m_hi under the IMF."""
    boundaries = sorted(set([m_lo, m_hi]))
    if hasattr(imf, 'm_break'):
        boundaries = sorted(set([m_lo, imf.m_break, m_hi]))
    boundaries = [b for b in boundaries if m_lo <= b <= m_hi]

    total = 0.0
    for i in range(len(boundaries) - 1):
        lo, hi = boundaries[i], boundaries[i + 1]
        if lo >= hi:
            continue
        if hasattr(imf, '_k1'):
            k = imf._k1 if lo < imf.m_break else imf._k2
            alpha = imf.alpha1 if lo < imf.m_break else imf.alpha2
        else:
            k = 1.0
            alpha = imf.alpha
        total += imf._segment_mass_integral(lo, hi, k, alpha)
    return total


def synthesise_population(snapshot, isochrone_grid, config,
                          extinction_calc=None):
    """Generate individual stars from simulation star particles.

    Groups particles by isochrone grid cell (~1k cells) instead of
    looping over individual particles (~millions), giving large
    speedups from batch interpolation and vectorised magnitude cuts.

    Parameters
    ----------
    snapshot : SimulationSnapshot
    isochrone_grid : IsochroneGrid
    config : Config
    extinction_calc : ExtinctionCalculator, optional

    Returns
    -------
    dict of arrays — the raw stellar catalogue (before error convolution).
    """
    imf = get_imf(config.imf, m_min=config.m_min, m_max=config.m_max)
    rng = np.random.default_rng(config.random_seed)

    obs_pos = np.asarray(config.observer_position)
    obs_vel = np.asarray(config.observer_velocity)

    n_particles = snapshot.n_star

    if config.fsample < 1.0:
        n_sample = max(1, int(n_particles * config.fsample))
        particle_indices = np.sort(
            rng.choice(n_particles, size=n_sample, replace=False))
        print(f"\n  Population synthesis: {n_particles} star particles "
              f"(sampling {n_sample}, fsample={config.fsample})")
    else:
        particle_indices = np.arange(n_particles)
        print(f"\n  Population synthesis: {n_particles} star particles")

    print(f"  IMF: {imf.name}, mass range [{config.m_min}, {config.m_max}] Msun")
    print(f"  Magnitude limit: G < {config.mag_limit}")

    n_use = len(particle_indices)

    # --- Precompute per-particle quantities ---
    part_pos = snapshot.star_pos[particle_indices]
    part_vel = snapshot.star_vel[particle_indices]
    part_mass = snapshot.star_mass[particle_indices]
    part_age = snapshot.star_age[particle_indices]
    part_met = snapshot.star_metallicity[particle_indices]

    part_logage = np.clip(np.log10(part_age * 1e9),
                          config.logage_min, config.logage_max)
    part_mh = np.clip(part_met, config.mh_min, config.mh_max)

    rel_all = part_pos - obs_pos
    part_dist = np.sqrt(np.sum(rel_all**2, axis=1))
    part_dist = np.maximum(part_dist, 1e-6)
    part_dist_mod = 5.0 * np.log10(part_dist * 1000.0) - 5.0

    # --- Phase-space smoothing lengths ---
    h_pos = h_vel = None
    if config.phase_space_smoothing:
        import time as _time
        from .smoothing import estimate_smoothing_lengths
        t_sm = _time.time()
        print(f"  Computing smoothing lengths (n_ngb={config.smoothing_ngb}, "
              f"scale={config.smoothing_scale}) ...")
        h_pos, h_vel = estimate_smoothing_lengths(
            part_pos, part_vel,
            n_ngb=config.smoothing_ngb,
            scale=config.smoothing_scale,
        )
        print(f"  Smoothing lengths done in {_time.time() - t_sm:.1f}s — "
              f"h_pos median={np.median(h_pos):.4f} kpc, "
              f"h_vel median={np.median(h_vel):.2f} km/s")

    # --- Precompute extinction ---
    particle_av = np.zeros(n_use)
    if extinction_calc is not None:
        import time as _time
        t_ext = _time.time()
        print(f"  Precomputing extinction for {n_use:,} particles ...")
        gal_all = cartesian_to_galactic(rel_all[:, 0], rel_all[:, 1],
                                        rel_all[:, 2])
        try:
            particle_av = extinction_calc.compute(
                gal_all["l"], gal_all["b"], gal_all["dist"],
                star_pos=part_pos, observer_pos=obs_pos,
            )
        except TypeError:
            particle_av = extinction_calc.compute(
                gal_all["l"], gal_all["b"], gal_all["dist"],
            )
        del gal_all
        print(f"  Extinction done in {_time.time() - t_ext:.1f}s — "
              f"median A_V={np.median(particle_av):.3f}, "
              f"max={np.max(particle_av):.2f}")

    part_ag = particle_av * EXTINCTION_COEFFS.get("Gmag", 0.789)

    # --- Assign particles to isochrone cells (one batch query) ---
    queries = np.column_stack([part_logage, part_mh]) * isochrone_grid._scale
    _, cell_ids = isochrone_grid._tree.query(queries)
    unique_cells, inverse = np.unique(cell_ids, return_inverse=True)
    n_cells = len(unique_cells)
    print(f"  {n_cells} unique isochrone cells")

    # Precompute full IMF integral (denominator, constant for all particles)
    full_integral = _mass_integral(imf, config.m_min, config.m_max)
    integral_cache = {}

    # --- Synthesis by isochrone cell ---
    all_mini = []
    all_pidx = []
    prop_lists = defaultdict(list)
    n_total_stars = 0
    n_skipped = 0

    pbar = tqdm(total=n_use, desc="  Synthesising", unit="part")

    for ci in range(n_cells):
        cell_particles = np.where(inverse == ci)[0]
        n_cell = len(cell_particles)

        key = tuple(isochrone_grid._grid_points[unique_cells[ci]])
        iso = isochrone_grid._grid[key]
        iso_mini = iso["Mini"]
        iso_gmag = iso["Gmag"]

        # Vectorised min_observable_mass via sorted lookup
        dm_eff = part_dist_mod[cell_particles] + part_ag[cell_particles]
        threshold = config.mag_limit - dm_eff

        sorted_order = np.argsort(iso_gmag)
        sorted_gmag = iso_gmag[sorted_order]
        sorted_mini = iso_mini[sorted_order]
        running_min = np.minimum.accumulate(sorted_mini)

        k_idx = np.searchsorted(sorted_gmag, threshold, side='right')
        m_min_obs = np.where(
            k_idx > 0,
            running_min[np.clip(k_idx - 1, 0, len(running_min) - 1)],
            np.inf,
        )

        valid = np.isfinite(m_min_obs) & (m_min_obs < config.m_max)
        n_skipped += n_cell - int(np.sum(valid))

        if not np.any(valid):
            pbar.update(n_cell)
            continue

        valid_local = np.where(valid)[0]

        # Vectorised mass sampling — group by m_min_trunc
        m_min_samples = np.maximum(m_min_obs[valid_local], config.m_min)
        group_pidx_all = cell_particles[valid_local]
        group_pmass_all = part_mass[group_pidx_all]

        cache_keys = np.round(m_min_samples, 8)
        unique_keys, group_inv = np.unique(cache_keys, return_inverse=True)

        cell_masses = []
        cell_pidx = []

        for gi in range(len(unique_keys)):
            mmin = float(unique_keys[gi])
            mask = group_inv == gi
            g_pmass = group_pmass_all[mask]
            g_pidx = group_pidx_all[mask]

            if config.sampling == "mass_budget":
                masses, cnts = _vectorised_mass_budget(
                    imf, g_pmass, mmin, config.m_max,
                    rng, full_integral, integral_cache)
            else:
                masses, cnts = _vectorised_count_sample(
                    imf, g_pmass, mmin, config.sampling, rng)

            if len(masses) == 0:
                n_skipped += len(g_pmass)
                continue

            pidx = np.repeat(g_pidx, cnts)

            ok = (masses >= config.m_min) & (masses <= config.m_max)
            if not np.all(ok):
                masses = masses[ok]
                pidx = pidx[ok]

            if len(masses) == 0:
                n_skipped += len(g_pmass)
                continue

            n_skipped += int(np.sum(cnts == 0))
            cell_masses.append(masses)
            cell_pidx.append(pidx)

        if not cell_masses:
            pbar.update(n_cell)
            continue

        # Concatenate all masses for this cell
        batch_masses = np.concatenate(cell_masses)
        batch_pidx = np.concatenate(cell_pidx)
        del cell_masses, cell_pidx

        # Batch interpolation — ONE set of interpolators for this cell
        props = {}
        interpolators = {}
        for col in iso:
            if col in ("logAge", "MH", "Mini"):
                continue
            try:
                fi = interp1d(iso_mini, iso[col], kind="linear",
                              bounds_error=False, fill_value=np.nan)
                interpolators[col] = fi
                props[col] = fi(batch_masses)
            except Exception:
                props[col] = np.full(len(batch_masses), np.nan)

        # --- Unresolved binaries: draw companions and combine fluxes ---
        if config.unresolved_binaries:
            n_batch = len(batch_masses)
            f_bin_m = np.clip(
                config.f_binary + 0.25 * np.log10(batch_masses), 0.0, 1.0)
            is_binary = rng.uniform(size=n_batch) < f_bin_m
            n_bin = int(np.sum(is_binary))
            q_arr = np.zeros(n_batch)
            mini_sec = np.zeros(n_batch)
            if n_bin > 0:
                q_vals = rng.uniform(config.q_min, 1.0, size=n_bin)
                q_arr[is_binary] = q_vals
                m2 = q_vals * batch_masses[is_binary]
                m2 = np.clip(m2, iso_mini[0], iso_mini[-1])
                mini_sec[is_binary] = m2
                for col, fi in interpolators.items():
                    if "mag" in col or "Mag" in col:
                        sec_mag = fi(m2)
                        pri_mag = props[col][is_binary]
                        valid = np.isfinite(sec_mag) & np.isfinite(pri_mag)
                        combined = pri_mag.copy()
                        f1 = 10.0 ** (-0.4 * pri_mag[valid])
                        f2 = 10.0 ** (-0.4 * sec_mag[valid])
                        combined[valid] = -2.5 * np.log10(f1 + f2)
                        props[col][is_binary] = combined
            props["is_binary"] = is_binary.astype(np.float32)
            props["q"] = q_arr
            props["Mini_secondary"] = mini_sec

        # Vectorised validity + magnitude cut
        g_valid = np.isfinite(
            props.get("Gmag", np.full(len(batch_masses), np.nan)))
        gmag_app = props["Gmag"] + part_dist_mod[batch_pidx] + part_ag[batch_pidx]
        mag_ok = (g_valid
                  & (gmag_app <= config.mag_limit)
                  & (gmag_app >= config.mag_bright_limit))

        if np.any(mag_ok):
            n_ok = int(np.sum(mag_ok))
            all_mini.append(batch_masses[mag_ok])
            all_pidx.append(batch_pidx[mag_ok])
            for col in props:
                prop_lists[col].append(props[col][mag_ok])
            n_total_stars += n_ok

        pbar.update(n_cell)

    pbar.close()

    if n_total_stars == 0:
        print("  Warning: no stars generated! Check magnitude limit and "
              "particle distances.")
        return None

    print(f"\n  Generated {n_total_stars:,} individual stars "
          f"({n_skipped} particles skipped)")

    # --- Concatenate and reconstruct ---
    cat_mini = np.concatenate(all_mini); del all_mini
    cat_pidx = np.concatenate(all_pidx); del all_pidx

    cat_pos = part_pos[cat_pidx]
    cat_vel = part_vel[cat_pidx]
    cat_logage = part_logage[cat_pidx]
    cat_mh = part_mh[cat_pidx]
    cat_av = particle_av[cat_pidx]
    cat_dist_mod = part_dist_mod[cat_pidx]
    cat_parent = particle_indices[cat_pidx].astype(np.int64)

    # --- Apply phase-space smoothing offsets ---
    if config.phase_space_smoothing and h_pos is not None:
        from .smoothing import epanechnikov_offsets
        import time as _time
        t_sm2 = _time.time()
        print(f"  Applying phase-space smoothing to {n_total_stars:,} stars ...")

        parent_counts = np.bincount(cat_pidx, minlength=n_use)
        used = parent_counts > 0

        pos_offsets = epanechnikov_offsets(
            h_pos[used], parent_counts[used], rng)
        vel_offsets = epanechnikov_offsets(
            h_vel[used], parent_counts[used], rng)

        order = np.argsort(cat_pidx, kind="stable")
        cat_pos[order] += pos_offsets
        cat_vel[order] += vel_offsets
        del pos_offsets, vel_offsets, order

        cat_rel = cat_pos - obs_pos
        cat_dist = np.sqrt(np.sum(cat_rel**2, axis=1))
        cat_dist = np.maximum(cat_dist, 1e-6)
        cat_dist_mod = 5.0 * np.log10(cat_dist * 1000.0) - 5.0
        del cat_rel, cat_dist

        print(f"  Smoothing applied in {_time.time() - t_sm2:.1f}s")

    del cat_pidx

    merged_props = {}
    for key in list(prop_lists.keys()):
        merged_props[key] = np.concatenate(prop_lists[key])
        del prop_lists[key]
    del prop_lists

    catalogue = {
        "pos": cat_pos,
        "vel": cat_vel,
        "Mini": cat_mini,
        "logage": cat_logage,
        "MH": cat_mh,
        "parent_id": cat_parent,
        "dist_mod": cat_dist_mod,
        "A_V": cat_av,
    }
    catalogue.update(merged_props)

    for band in ["Gmag", "G_BPmag", "G_RPmag"]:
        if band in catalogue:
            a_band = EXTINCTION_COEFFS.get(band, 0.0) * cat_av
            catalogue[f"{band}_app"] = catalogue[band] + cat_dist_mod + a_band

    return catalogue
