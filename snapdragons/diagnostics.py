"""Diagnostic plots for Snapdragons catalogues."""

import os
import numpy as np


def make_diagnostic_plots(catalogue, config, output_dir="output/plots"):
    """Generate and save diagnostic plots from a catalogue.

    Produces:
      1. Cartesian (x, y) density — true vs observed
      2. Cartesian (x, z) density — true vs observed
      3. Galactic (l, b) sky density (Mollweide) — true vs observed
      4. Galactic (l, b) sky density (Cartesian) — true vs observed
      5. CMD density (colour-magnitude diagram)
      6. Parallax comparison (true vs observed)
      7. Distance vs apparent magnitude density
      8. Extinction map (if simulation extinction mode)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    true = catalogue["true"]
    observed = catalogue.get("observed", {})
    stellar = catalogue.get("stellar", {})
    has_obs = len(observed) > 0 and "parallax" in observed
    has_obs_cart = has_obs and "x" in observed

    obs_pos = np.asarray(config.observer_position)

    has_cart = "cartesian" in config.output_coords
    has_gal = "l" in true and "b" in true

    plot_files = []

    if has_gal:
        _plot_sky_scatter(true, observed, has_obs, output_dir, plot_files)
        _plot_sky_density(true, observed, has_obs, output_dir, plot_files)

    if (has_gal and all(k in true
                        for k in ("Gmag_app", "G_BPmag_app", "G_RPmag_app"))):
        _plot_gaia_photometry_sky(true, output_dir, plot_files)

    if has_cart:
        _plot_cartesian_xy(true, observed, has_obs_cart, obs_pos,
                           output_dir, plot_files)
        _plot_cartesian_xz(true, observed, has_obs_cart, obs_pos,
                           output_dir, plot_files)

    if "Gmag_app" in true and "G_BPmag_app" in true and "G_RPmag_app" in true:
        _plot_cmd(true, observed, has_obs, output_dir, plot_files)

    if "parallax" in true:
        _plot_parallax(true, observed, has_obs, output_dir, plot_files)

    if "dist" in true and "Gmag_app" in true:
        _plot_distance_mag(true, output_dir, plot_files)

    if has_gal and "A_V" in true and config.extinction_mode == "simulation":
        _plot_extinction_map(true, output_dir, plot_files)

    plt.close("all")
    print(f"  Saved {len(plot_files)} diagnostic plots to {output_dir}/")
    for f in plot_files:
        print(f"    {os.path.basename(f)}")

    return plot_files


def _plot_sky_scatter(true, observed, has_obs, output_dir, plot_files):
    import matplotlib.pyplot as plt

    n_stars = len(true["l"])
    max_plot = min(n_stars, 500_000)
    if n_stars > max_plot:
        idx = np.random.default_rng(0).choice(n_stars, max_plot, replace=False)
    else:
        idx = slice(None)

    l_true = np.deg2rad(true["l"][idx])
    b_true = np.deg2rad(true["b"][idx])
    l_true = np.where(l_true > np.pi, l_true - 2 * np.pi, l_true)

    ncols = 2 if has_obs else 1
    fig = plt.figure(figsize=(8 * ncols, 4))

    ax0 = fig.add_subplot(1, ncols, 1, projection="mollweide")
    ax0.scatter(l_true, b_true, s=0.1, alpha=0.3, c="steelblue",
                rasterized=True)
    ax0.set_title("True positions")
    ax0.grid(True, alpha=0.3)
    ax0.set_xlabel("l")
    ax0.set_ylabel("b")

    if has_obs and "l" in observed:
        ax1 = fig.add_subplot(1, ncols, 2, projection="mollweide")
        l_obs = np.deg2rad(observed["l"][idx])
        b_obs = np.deg2rad(observed["b"][idx])
        l_obs = np.where(l_obs > np.pi, l_obs - 2 * np.pi, l_obs)
        ax1.scatter(l_obs, b_obs, s=0.1, alpha=0.3, c="firebrick",
                    rasterized=True)
        ax1.set_title("Observed positions")
        ax1.grid(True, alpha=0.3)
        ax1.set_xlabel("l")
        ax1.set_ylabel("b")

    fig.suptitle("Sky distribution (Galactic coordinates)", fontsize=14)
    fig.tight_layout()
    path = os.path.join(output_dir, "sky_scatter.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plot_files.append(path)
    plt.close(fig)


def _plot_sky_density(true, observed, has_obs, output_dir, plot_files):
    import matplotlib.pyplot as plt

    ncols = 2 if has_obs else 1
    fig, axes = plt.subplots(1, ncols, figsize=(8 * ncols, 4))
    if ncols == 1:
        axes = [axes]

    l_true = true["l"]
    b_true = true["b"]

    axes[0].hist2d(l_true, b_true, bins=[360, 180], range=[[0, 360], [-90, 90]],
                   cmap="Blues", cmin=1)
    axes[0].set_xlabel("l [deg]")
    axes[0].set_ylabel("b [deg]")
    axes[0].set_title("True — stellar density")
    axes[0].set_aspect("equal")

    if has_obs and "l" in observed:
        l_obs = observed["l"]
        b_obs = observed["b"]
        axes[1].hist2d(l_obs, b_obs, bins=[360, 180],
                       range=[[0, 360], [-90, 90]], cmap="Reds", cmin=1)
        axes[1].set_xlabel("l [deg]")
        axes[1].set_ylabel("b [deg]")
        axes[1].set_title("Observed — stellar density")
        axes[1].set_aspect("equal")

    fig.suptitle("Sky density (Galactic coordinates)", fontsize=14)
    fig.tight_layout()
    path = os.path.join(output_dir, "sky_density.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plot_files.append(path)
    plt.close(fig)


def _plot_cartesian_xy(true, observed, has_obs, obs_pos,
                       output_dir, plot_files):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    x = true["x"]
    y = true["y"]

    ncols = 2 if (has_obs and "x" in observed) else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 6))
    if ncols == 1:
        axes = [axes]

    nbins = 500
    h, xe, ye = np.histogram2d(x, y, bins=nbins)
    h = np.ma.masked_where(h == 0, h)
    axes[0].pcolormesh(xe, ye, h.T, cmap="Blues", norm=LogNorm(),
                       rasterized=True)
    axes[0].plot(obs_pos[0], obs_pos[1], "r*", ms=12, label="Observer",
                 zorder=5)
    axes[0].set_xlabel("x [kpc]")
    axes[0].set_ylabel("y [kpc]")
    axes[0].set_title("True")
    axes[0].legend()
    axes[0].set_aspect("equal")
    xlim = (xe[0], xe[-1])
    ylim = (ye[0], ye[-1])

    if ncols > 1:
        h2, _, _ = np.histogram2d(observed["x"], observed["y"],
                                  bins=[xe, ye])
        h2 = np.ma.masked_where(h2 == 0, h2)
        axes[1].pcolormesh(xe, ye, h2.T, cmap="Reds", norm=LogNorm(),
                           rasterized=True)
        axes[1].plot(obs_pos[0], obs_pos[1], "r*", ms=12, label="Observer",
                     zorder=5)
        axes[1].set_xlabel("x [kpc]")
        axes[1].set_ylabel("y [kpc]")
        axes[1].set_title("Observed")
        axes[1].legend()
        axes[1].set_aspect("equal")
        axes[1].set_xlim(xlim)
        axes[1].set_ylim(ylim)

    fig.suptitle("Cartesian distribution (x, y)", fontsize=14)
    fig.tight_layout()
    path = os.path.join(output_dir, "cartesian_xy.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plot_files.append(path)
    plt.close(fig)


def _plot_cartesian_xz(true, observed, has_obs, obs_pos,
                       output_dir, plot_files):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    x = true["x"]
    z = true["z"]

    ncols = 2 if (has_obs and "x" in observed) else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 4))
    if ncols == 1:
        axes = [axes]

    nbins = 500
    h, xe, ze = np.histogram2d(x, z, bins=nbins)
    h = np.ma.masked_where(h == 0, h)
    axes[0].pcolormesh(xe, ze, h.T, cmap="Blues", norm=LogNorm(),
                       rasterized=True)
    axes[0].plot(obs_pos[0], obs_pos[2], "r*", ms=12, label="Observer",
                 zorder=5)
    axes[0].set_xlabel("x [kpc]")
    axes[0].set_ylabel("z [kpc]")
    axes[0].set_title("True")
    axes[0].legend()
    xlim = (xe[0], xe[-1])
    zlim = (ze[0], ze[-1])

    if ncols > 1:
        h2, _, _ = np.histogram2d(observed["x"], observed["z"],
                                  bins=[xe, ze])
        h2 = np.ma.masked_where(h2 == 0, h2)
        axes[1].pcolormesh(xe, ze, h2.T, cmap="Reds", norm=LogNorm(),
                           rasterized=True)
        axes[1].plot(obs_pos[0], obs_pos[2], "r*", ms=12, label="Observer",
                     zorder=5)
        axes[1].set_xlabel("x [kpc]")
        axes[1].set_ylabel("z [kpc]")
        axes[1].set_title("Observed")
        axes[1].legend()
        axes[1].set_xlim(xlim)
        axes[1].set_ylim(zlim)

    fig.suptitle("Cartesian distribution (x, z)", fontsize=14)
    fig.tight_layout()
    path = os.path.join(output_dir, "cartesian_xz.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plot_files.append(path)
    plt.close(fig)


def _plot_cmd(true, observed, has_obs, output_dir, plot_files):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    bp_rp_true = true["G_BPmag_app"] - true["G_RPmag_app"]
    g_true = true["Gmag_app"]

    ncols = 2 if (has_obs and "Gmag" in observed) else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 7))
    if ncols == 1:
        axes = [axes]

    nbins = 500
    h, xe, ye = np.histogram2d(bp_rp_true, g_true, bins=nbins)
    h = np.ma.masked_where(h == 0, h)
    axes[0].pcolormesh(xe, ye, h.T, cmap="Blues", norm=LogNorm(),
                       rasterized=True)
    axes[0].set_xlabel("G_BP - G_RP [mag]")
    axes[0].set_ylabel("G [mag]")
    axes[0].set_title("True")
    axes[0].invert_yaxis()

    if ncols > 1:
        bp_rp_obs = observed["G_BPmag"] - observed["G_RPmag"]
        g_obs = observed["Gmag"]
        h2, xe2, ye2 = np.histogram2d(bp_rp_obs, g_obs, bins=nbins)
        h2 = np.ma.masked_where(h2 == 0, h2)
        axes[1].pcolormesh(xe2, ye2, h2.T, cmap="Reds", norm=LogNorm(),
                           rasterized=True)
        axes[1].set_xlabel("G_BP - G_RP [mag]")
        axes[1].set_ylabel("G [mag]")
        axes[1].set_title("Observed")
        axes[1].invert_yaxis()

    fig.suptitle("Colour-magnitude diagram (apparent)", fontsize=14)
    fig.tight_layout()
    path = os.path.join(output_dir, "cmd.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plot_files.append(path)
    plt.close(fig)


def _plot_parallax(true, observed, has_obs, output_dir, plot_files):
    import matplotlib.pyplot as plt

    plx_true = true["parallax"]

    if not has_obs:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(plx_true, bins=100, alpha=0.7, color="steelblue",
                label="True")
        ax.set_xlabel("Parallax [mas]")
        ax.set_ylabel("Count")
        ax.set_title("Parallax distribution")
        ax.legend()
    else:
        plx_obs = observed["parallax"]
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].hist(plx_true, bins=100, alpha=0.6, color="steelblue",
                     label="True")
        axes[0].hist(plx_obs, bins=100, alpha=0.6, color="firebrick",
                     label="Observed")
        axes[0].set_xlabel("Parallax [mas]")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Parallax distribution")
        axes[0].legend()

        resid = plx_obs - plx_true
        axes[1].hist(resid, bins=100, alpha=0.7, color="grey")
        axes[1].axvline(0, color="k", ls="--", lw=0.8)
        axes[1].set_xlabel("Parallax residual (obs - true) [mas]")
        axes[1].set_ylabel("Count")
        axes[1].set_title(f"Residuals: median={np.median(resid):.4f}, "
                          f"std={np.std(resid):.4f} mas")

    fig.tight_layout()
    path = os.path.join(output_dir, "parallax.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plot_files.append(path)
    plt.close(fig)


def _plot_distance_mag(true, output_dir, plot_files):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    dist = true["dist"]
    gmag = true["Gmag_app"]

    fig, ax = plt.subplots(figsize=(7, 5))
    nbins = 500
    h, xe, ye = np.histogram2d(dist, gmag, bins=nbins)
    h = np.ma.masked_where(h == 0, h)
    ax.pcolormesh(xe, ye, h.T, cmap="Blues", norm=LogNorm(), rasterized=True)
    ax.set_xlabel("Distance [kpc]")
    ax.set_ylabel("G [mag]")
    ax.set_title("Apparent magnitude vs distance")
    ax.invert_yaxis()

    fig.tight_layout()
    path = os.path.join(output_dir, "distance_mag.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plot_files.append(path)
    plt.close(fig)


def _plot_extinction_map(true, output_dir, plot_files):
    import matplotlib.pyplot as plt
    from scipy.stats import binned_statistic_2d

    l = true["l"]
    b = true["b"]
    a_v = true["A_V"]

    l_rad = np.deg2rad(l)
    l_rad = np.where(l_rad > np.pi, l_rad - 2 * np.pi, l_rad)
    b_rad = np.deg2rad(b)

    vmax = np.percentile(a_v[a_v > 0], 99) if np.any(a_v > 0) else 1.0

    fig = plt.figure(figsize=(16, 5))

    # Panel 1: binned median A_V in Mollweide projection
    ax_mol = fig.add_subplot(1, 2, 1, projection="mollweide")
    l_edges_rad = np.linspace(-np.pi, np.pi, 181)
    b_edges_rad = np.linspace(-np.pi / 2, np.pi / 2, 91)
    med_av_mol, _, _, _ = binned_statistic_2d(
        l_rad, b_rad, a_v, statistic="median",
        bins=[l_edges_rad, b_edges_rad],
    )
    med_av_mol = np.where(np.isnan(med_av_mol), 0, med_av_mol)
    L_mol, B_mol = np.meshgrid(l_edges_rad, b_edges_rad)
    im1 = ax_mol.pcolormesh(L_mol, B_mol, med_av_mol.T, cmap="inferno",
                            vmin=0, vmax=vmax, rasterized=True)
    ax_mol.grid(True, alpha=0.3)
    ax_mol.set_title("Median A$_V$ (Mollweide, 2° bins)")
    cb1 = fig.colorbar(im1, ax=ax_mol, shrink=0.7, pad=0.08)
    cb1.set_label("A$_V$ [mag]")

    # Panel 2: binned median A_V in Cartesian (l, b)
    ax2 = fig.add_subplot(1, 2, 2)
    n_lbins, n_bbins = 180, 90
    l_edges = np.linspace(0, 360, n_lbins + 1)
    b_edges = np.linspace(-90, 90, n_bbins + 1)
    med_av, _, _, _ = binned_statistic_2d(
        l, b, a_v, statistic="median",
        bins=[l_edges, b_edges],
    )
    med_av = np.where(np.isnan(med_av), 0, med_av)
    L, B = np.meshgrid(l_edges, b_edges)
    im2 = ax2.pcolormesh(L, B, med_av.T, cmap="inferno",
                         vmin=0, vmax=vmax, rasterized=True)
    ax2.set_xlabel("l [deg]")
    ax2.set_ylabel("b [deg]")
    ax2.set_title("Median A$_V$ (2° bins)")
    ax2.set_aspect("equal")
    cb2 = fig.colorbar(im2, ax=ax2, shrink=0.7)
    cb2.set_label("A$_V$ [mag]")

    fig.suptitle("Simulation extinction map", fontsize=14)
    fig.tight_layout()
    path = os.path.join(output_dir, "extinction_map.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plot_files.append(path)
    plt.close(fig)


def _plot_gaia_photometry_sky(true, output_dir, plot_files):
    """Gaia-style all-sky colour map in Mollweide projection.

    Each pixel's hue encodes the flux-weighted mean BP-RP colour
    (blue → white → yellow → orange → red) and brightness encodes
    the total G-band flux on a log stretch.
    """
    import matplotlib.pyplot as plt
    from scipy.interpolate import interp1d

    l = true["l"]
    b = true["b"]
    gmag = true["Gmag_app"]
    bp_rp = true["G_BPmag_app"] - true["G_RPmag_app"]

    flux = 10.0 ** (-0.4 * gmag)

    n_l, n_b = 720, 360
    l_edges = np.linspace(0, 360, n_l + 1)
    b_edges = np.linspace(-90, 90, n_b + 1)

    total_flux, _, _ = np.histogram2d(
        l, b, bins=[l_edges, b_edges], weights=flux)
    weighted_color, _, _ = np.histogram2d(
        l, b, bins=[l_edges, b_edges], weights=flux * bp_rp)
    del flux

    occupied = total_flux > 0
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_color = np.where(occupied, weighted_color / total_flux, 1.0)
    del weighted_color

    ctrl = np.array([
        [-0.5, 0.65, 0.73, 1.00],
        [ 0.0, 0.78, 0.84, 1.00],
        [ 0.4, 0.94, 0.95, 1.00],
        [ 0.8, 1.00, 0.98, 0.95],
        [ 1.2, 1.00, 0.96, 0.90],
        [ 1.8, 1.00, 0.92, 0.82],
        [ 2.5, 1.00, 0.86, 0.72],
        [ 4.0, 1.00, 0.78, 0.58],
    ])
    f_r = interp1d(ctrl[:, 0], ctrl[:, 1], bounds_error=False,
                   fill_value=(ctrl[0, 1], ctrl[-1, 1]))
    f_g = interp1d(ctrl[:, 0], ctrl[:, 2], bounds_error=False,
                   fill_value=(ctrl[0, 2], ctrl[-1, 2]))
    f_b = interp1d(ctrl[:, 0], ctrl[:, 3], bounds_error=False,
                   fill_value=(ctrl[0, 3], ctrl[-1, 3]))

    rgb = np.stack([f_r(mean_color), f_g(mean_color), f_b(mean_color)],
                   axis=-1)
    del mean_color

    log_flux = np.log10(np.maximum(total_flux, 1e-30))
    if np.any(occupied):
        vmin = np.percentile(log_flux[occupied], 1)
        vmax = np.percentile(log_flux[occupied], 99.8)
    else:
        vmin, vmax = -10, 0
    bright = np.clip((log_flux - vmin) / (vmax - vmin), 0, 1)
    bright[~occupied] = 0
    bright = bright ** 0.45
    del log_flux, total_flux

    desat = np.clip(bright * 0.6, 0, 0.6)
    rgb = rgb * (1 - desat[..., None]) + desat[..., None]
    rgb *= bright[..., None]
    rgb = np.clip(rgb, 0, 1)
    del bright, desat

    img_w, img_h = 2000, 1000
    xp = np.linspace(-2 * np.sqrt(2), 2 * np.sqrt(2), img_w)
    yp = np.linspace(-np.sqrt(2), np.sqrt(2), img_h)
    X, Y = np.meshgrid(xp, yp)

    inside = (X / (2 * np.sqrt(2)))**2 + (Y / np.sqrt(2))**2 <= 1

    theta = np.arcsin(np.clip(Y / np.sqrt(2), -1, 1))
    cos_th = np.cos(theta)
    cos_th = np.where(np.abs(cos_th) < 1e-10, 1e-10, cos_th)
    lam = np.pi * X / (2 * np.sqrt(2) * cos_th)
    phi = np.arcsin(np.clip(
        (2 * theta + np.sin(2 * theta)) / np.pi, -1, 1))

    l_deg = np.degrees(-lam) % 360
    b_deg = np.degrees(phi)

    il_f = (l_deg / 360 * n_l) % n_l
    ib_f = np.clip((b_deg + 90) / 180 * n_b, 0, n_b - 1e-6)

    il0 = np.floor(il_f).astype(np.intp)
    il1 = (il0 + 1) % n_l
    ib0 = np.clip(np.floor(ib_f).astype(np.intp), 0, n_b - 1)
    ib1 = np.clip(ib0 + 1, 0, n_b - 1)

    wl = (il_f - il0)[..., None]
    wb = (ib_f - np.floor(ib_f))[..., None]

    img = (rgb[il0, ib0] * (1 - wl) * (1 - wb) +
           rgb[il1, ib0] * wl * (1 - wb) +
           rgb[il0, ib1] * (1 - wl) * wb +
           rgb[il1, ib1] * wl * wb)
    img[~inside] = 0
    del rgb, X, Y, il0, il1, ib0, ib1, wl, wb

    fig = plt.figure(figsize=(14, 7), facecolor='black')
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.92])
    ax.set_facecolor('black')
    ax.imshow(img, origin='lower', aspect='auto',
              extent=[-2 * np.sqrt(2), 2 * np.sqrt(2),
                      -np.sqrt(2), np.sqrt(2)],
              interpolation='bilinear')
    ax.axis('off')
    ax.set_title("Gaia photometric sky map "
                 "(flux-weighted BP$-$RP colour)",
                 color='white', fontsize=14, pad=10)

    path = os.path.join(output_dir, "gaia_photometry_sky.png")
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='black')
    plot_files.append(path)
    plt.close(fig)
