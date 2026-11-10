"""PARSEC isochrone grid: download, cache, and interpolation.

Downloads isochrone tables directly from the CMD web interface,
caches them locally as an HDF5 file, and provides fast interpolation
over (log_age, [M/H], M_ini) -> stellar properties.

Also supports loading pre-existing isochrone tables from CSV files.
"""

import os
import io
import re
import ssl
import time
import hashlib
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial import cKDTree


REQUIRED_COLUMNS = [
    "logAge", "MH", "Mini", "Mass", "logL", "logTe", "logg",
    "Gmag", "G_BPmag", "G_RPmag", "label",
]

CMD_URL = "https://stev.oapd.inaf.it/cgi-bin/cmd"


def _get_ssl_context():
    """Build an SSL context, trying certifi then falling back to unverified."""
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
        import urllib.request
        test = urllib.request.build_opener(
            urllib.request.HTTPSHandler(context=ctx))
        test.open("https://stev.oapd.inaf.it/", timeout=10).close()
        return ctx
    except Exception:
        pass
    return ssl._create_unverified_context()


def _http_post(url, data, timeout=120):
    """POST form data, handling SSL and redirects."""
    import urllib.request
    import urllib.parse

    ctx = _get_ssl_context()
    encoded = urllib.parse.urlencode(data).encode("utf-8")

    opener = urllib.request.build_opener(
        urllib.request.HTTPSHandler(context=ctx))
    req = urllib.request.Request(url, data=encoded)
    req.add_header("Content-Type", "application/x-www-form-urlencoded")
    req.add_header("User-Agent",
                   "Snapdragons/2.0 (stellar population synthesis)")

    with opener.open(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8")


def _http_get(url, timeout=120):
    """GET a URL, handling SSL."""
    import urllib.request

    ctx = _get_ssl_context()
    opener = urllib.request.build_opener(
        urllib.request.HTTPSHandler(context=ctx))
    req = urllib.request.Request(url)
    req.add_header("User-Agent",
                   "Snapdragons/2.0 (stellar population synthesis)")

    with opener.open(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8")


class IsochroneGrid:
    """A grid of PARSEC isochrones for interpolation.

    The grid is indexed by (log_age, [M/H]).  For each grid point a
    1-D interpolator in M_ini gives absolute magnitudes and stellar
    parameters.
    """

    def __init__(self, config):
        self.config = config
        self.cache_dir = os.path.expanduser(config.isochrone_cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)
        self._grid = {}
        self._tree = None
        self._grid_points = None

    def _cache_path(self):
        tag = hashlib.md5(
            f"{self.config.photometric_system}_{self.config.logage_min}_"
            f"{self.config.logage_max}_{self.config.logage_step}_"
            f"{self.config.mh_min}_{self.config.mh_max}_"
            f"{self.config.mh_step}".encode()
        ).hexdigest()[:12]
        return os.path.join(self.cache_dir, f"parsec_{tag}.h5")

    def load_or_download(self):
        """Load cached isochrones or download from CMD."""
        cache = self._cache_path()
        if os.path.exists(cache):
            self._load_cache(cache)
        else:
            self._download_and_cache(cache)
        self._build_tree()
        return self

    def load_cache(self, cache_path):
        """Load a specific isochrone cache file (flat PARSEC columns in HDF5,
        as written by this class), ignoring the config's grid settings."""
        self._load_cache(cache_path)
        self._build_tree()
        return self

    def cell_of(self, logage, mh):
        """Index into ``grid_points`` of the nearest isochrone for each
        (log age, [M/H]) pair, vectorised."""
        queries = np.column_stack([np.atleast_1d(logage),
                                   np.atleast_1d(mh)]) * self._scale
        _, idx = self._tree.query(queries)
        return idx

    def cell(self, idx):
        """Isochrone data dict for a ``grid_points`` index."""
        return self._grid[tuple(self._grid_points[idx])]

    def _download_and_cache(self, cache_path):
        """Download PARSEC isochrones from CMD web interface."""
        import h5py

        logage_vals = np.arange(
            self.config.logage_min,
            self.config.logage_max + 0.5 * self.config.logage_step,
            self.config.logage_step,
        )
        mh_vals = np.arange(
            self.config.mh_min,
            self.config.mh_max + 0.5 * self.config.mh_step,
            self.config.mh_step,
        )

        print(f"  Downloading PARSEC isochrones: "
              f"{len(logage_vals)} ages x {len(mh_vals)} metallicities ...")
        print(f"  Photometric system: {self.config.photometric_system}")

        all_data = []
        for mh_val in mh_vals:
            form_data = {
                "cmd_version": "3.9",
                "track_parsec": "parsec_CAF09_v1.2S",
                "track_colibri": "parsec_CAF09_v1.2S_S_LMC_08_web",
                "track_postagb": "no",
                "n_inTPC": "10",
                "eta_reimers": "0.2",
                "kind_interp": "1",
                "kind_postagb": "-1",
                "photsys_file": (
                    "YBC_tab_mag_odfnew/tab_mag_gaiaEDR3.dat"
                ),
                "photsys_version": "YBCnewVega",
                "dust_sourceM": "dpmod60alox40",
                "dust_sourceC": "AMCSIC15",
                "kind_mag": "2",
                "kind_dust": "0",
                "extinction_av": "0.0",
                "extinction_coeff": "constant",
                "extinction_curve": "cardelli",
                "kind_LPV": "4",
                "imf_file": "tab_imf/imf_kroupa_orig.dat",
                "isoc_isagelog": "1",
                "isoc_agelow": "1.0e9",
                "isoc_ageupp": "1.0e10",
                "isoc_dage": "0.0",
                "isoc_lagelow": str(self.config.logage_min),
                "isoc_lageupp": str(self.config.logage_max),
                "isoc_dlage": str(self.config.logage_step),
                "isoc_ismetlog": "1",
                "isoc_zlow": "0.0152",
                "isoc_zupp": "0.03",
                "isoc_dz": "0.0",
                "isoc_metlow": str(mh_val),
                "isoc_metupp": str(mh_val),
                "isoc_dmet": "0.1",
                "output_kind": "0",
                "output_evstage": "1",
                "lf_maginf": "-15",
                "lf_magsup": "20",
                "lf_deltamag": "0.5",
                "sim_mtot": "1.0e4",
                "output_gzip": "0",
                "submit_form": "Submit",
            }

            try:
                html = _http_post(CMD_URL, form_data)

                match = re.search(r'(output\w+\.dat)', html)

                if match:
                    dat_name = match.group(1)
                    dat_url = (f"https://stev.oapd.inaf.it"
                               f"/tmp/{dat_name}")

                    table_text = _http_get(dat_url)
                    df_chunk = self._parse_cmd_output(table_text)
                    if df_chunk is not None and len(df_chunk) > 0:
                        all_data.append(df_chunk)
                        print(f"    [M/H]={mh_val:+.1f}: "
                              f"{len(df_chunk)} rows")
                    else:
                        print(f"    [M/H]={mh_val:+.1f}: "
                              f"could not parse output")
                else:
                    print(f"    [M/H]={mh_val:+.1f}: no output link found")

                time.sleep(1.0)

            except Exception as e:
                print(f"    [M/H]={mh_val:+.1f}: download failed ({e})")
                continue

        if not all_data:
            raise RuntimeError(
                "Failed to download any isochrones from CMD. "
                "Check your internet connection, or provide pre-downloaded "
                "isochrone tables via load_from_files()."
            )

        import pandas as pd
        full_df = pd.concat(all_data, ignore_index=True)

        keep = [c for c in REQUIRED_COLUMNS if c in full_df.columns]
        mag_cols = [c for c in full_df.columns
                    if c.endswith("mag") or c.endswith("Mag")]
        for mc in mag_cols:
            if mc not in keep:
                keep.append(mc)

        with h5py.File(cache_path, "w") as hf:
            for col in keep:
                hf.create_dataset(col, data=full_df[col].values,
                                  compression="gzip")
            hf.attrs["photometric_system"] = self.config.photometric_system
            hf.attrs["n_rows"] = len(full_df)

        print(f"  Cached {len(full_df)} isochrone rows -> {cache_path}")
        self._parse_arrays({col: full_df[col].values for col in keep})

    def _parse_cmd_output(self, text):
        """Parse the text output from the CMD web interface."""
        import pandas as pd

        lines = text.strip().split("\n")
        header_idx = None
        for i, line in enumerate(lines):
            if line.startswith("#") and "Mini" in line:
                header_idx = i
                break

        if header_idx is None:
            for i, line in enumerate(lines):
                if line.startswith("#"):
                    cols = line.lstrip("#").split()
                    if len(cols) > 5:
                        header_idx = i
                        break

        if header_idx is None:
            return None

        header_line = lines[header_idx].lstrip("#").strip()
        columns = header_line.split()

        data_lines = []
        for line in lines[header_idx + 1:]:
            if line.startswith("#") or not line.strip():
                continue
            data_lines.append(line)

        if not data_lines:
            return None

        df = pd.read_csv(
            io.StringIO("\n".join(data_lines)),
            sep=r"\s+",
            names=columns,
            comment="#",
        )
        return df

    def load_from_files(self, file_paths):
        """Load isochrones from local CSV/text files.

        Parameters
        ----------
        file_paths : list of str
            Paths to isochrone table files (space-separated, with header
            lines starting with #).
        """
        import pandas as pd

        all_data = []
        for fp in file_paths:
            print(f"  Loading isochrone file: {fp}")
            df = pd.read_csv(fp, sep=r"\s+", comment="#")
            all_data.append(df)

        full_df = pd.concat(all_data, ignore_index=True)

        keep = [c for c in REQUIRED_COLUMNS if c in full_df.columns]
        for c in full_df.columns:
            if (c.endswith("mag") or c.endswith("Mag")) and c not in keep:
                keep.append(c)

        self._parse_arrays({col: full_df[col].values for col in keep})
        self._build_tree()

        cache = self._cache_path()
        import h5py
        with h5py.File(cache, "w") as hf:
            for col in keep:
                hf.create_dataset(col, data=full_df[col].values,
                                  compression="gzip")
            hf.attrs["n_rows"] = len(full_df)

        print(f"  Loaded {len(full_df)} isochrone rows from "
              f"{len(file_paths)} files")
        return self

    def _load_cache(self, cache_path):
        """Load pre-cached isochrone HDF5 file."""
        import h5py

        print(f"  Loading cached isochrones from {cache_path}")
        with h5py.File(cache_path, "r") as hf:
            data = {key: hf[key][:] for key in hf.keys()}
        self._parse_arrays(data)

    def _parse_arrays(self, data):
        """Build internal grid from column arrays."""
        logage = data["logAge"]
        mh = data["MH"]

        logage_r = np.round(logage, 4)
        mh_r = np.round(mh, 4)
        unique_pairs = set(zip(logage_r, mh_r))

        self._grid = {}
        for la, m in unique_pairs:
            mask = (logage_r == la) & (mh_r == m)
            entry = {key: arr[mask] for key, arr in data.items()}
            sort_idx = np.argsort(entry["Mini"])
            entry = {key: arr[sort_idx] for key, arr in entry.items()}
            self._grid[(la, m)] = entry

    def _build_tree(self):
        """Build KDTree over (log_age, [M/H]) for nearest-neighbour lookup."""
        points = np.array(list(self._grid.keys()))
        scale = np.array([1.0, 5.0])
        self._grid_points = points
        self._tree = cKDTree(points * scale)
        self._scale = scale

    def get_nearest(self, logage, mh):
        """Return the isochrone data dict nearest to (logage, mh)."""
        query = np.array([logage, mh]) * self._scale
        _, idx = self._tree.query(query)
        key = tuple(self._grid_points[idx])
        return self._grid[key], key

    def interpolate_stars(self, logage, mh, masses):
        """Interpolate stellar properties for given masses on nearest isochrone.

        Parameters
        ----------
        logage : float   -- log10(age/yr)
        mh : float       -- [M/H]
        masses : 1-D array of initial masses [Msun]

        Returns
        -------
        dict with arrays for each property, or NaN where out of range.
        """
        iso, _ = self.get_nearest(logage, mh)
        mini = iso["Mini"]

        props = {}
        for key in iso:
            if key in ("logAge", "MH", "Mini"):
                continue
            try:
                interp = interp1d(
                    mini, iso[key],
                    kind="linear",
                    bounds_error=False,
                    fill_value=np.nan,
                )
                props[key] = interp(masses)
            except Exception:
                props[key] = np.full_like(masses, np.nan)

        props["Mini"] = masses.copy()
        return props

    def min_observable_mass(self, logage, mh, dist_kpc,
                            mag_limit, extinction_G=0.0):
        """Find the minimum initial mass observable at a given distance.

        Stars fainter than mag_limit in apparent G are unobservable.
        Returns m_min or np.inf if nothing is observable.
        """
        iso, _ = self.get_nearest(logage, mh)
        mini = iso["Mini"]
        abs_G = iso["Gmag"]
        dist_mod = 5.0 * np.log10(dist_kpc * 1000.0) - 5.0
        app_G = abs_G + dist_mod + extinction_G

        observable = app_G <= mag_limit
        if not np.any(observable):
            return np.inf

        return mini[observable].min()

    @classmethod
    def generate_synthetic(cls, config):
        """Generate a simplified synthetic isochrone grid for testing.

        Uses analytical approximations for main-sequence stars:
          logL ~ 3.5 * log10(M) for M > 0.43
          logL ~ 2.3 * log10(M) for M <= 0.43
          logTe ~ 3.76 + 0.5 * logL - 0.1 * logL^2
          Mbol = 4.74 - 2.5 * logL
          BC_G ~ -0.2  (rough bolometric correction)

        This is adequate for pipeline testing but should not be used
        for science.
        """
        print("  Generating synthetic isochrone grid (for testing)")
        grid = cls(config)

        logage_vals = np.arange(
            config.logage_min,
            config.logage_max + 0.5 * config.logage_step,
            config.logage_step,
        )
        mh_vals = np.arange(
            config.mh_min,
            config.mh_max + 0.5 * config.mh_step,
            config.mh_step,
        )

        for la in logage_vals:
            age_gyr = 10.0 ** (la - 9.0)
            m_to = _turnoff_mass(age_gyr)

            for mh in mh_vals:
                masses = np.logspace(np.log10(config.m_min),
                                     np.log10(min(m_to, config.m_max)),
                                     200)
                n = len(masses)

                logL = np.where(masses > 0.43,
                                3.5 * np.log10(masses),
                                2.3 * np.log10(masses))
                logL += 0.1 * mh

                logTe = 3.76 + 0.12 * logL - 0.01 * logL**2
                logg_val = np.log10(masses) - 2.0 * logTe + 4.44 + logL * 0.1

                Mbol = 4.74 - 2.5 * logL
                Gmag = Mbol + 0.2 - 0.05 * (logTe - 3.76)
                G_BPmag = Gmag + 0.3 * (logTe - 3.76)
                G_RPmag = Gmag - 0.3 * (logTe - 3.76)

                la_round = round(la, 4)
                mh_round = round(mh, 4)

                grid._grid[(la_round, mh_round)] = {
                    "logAge": np.full(n, la_round),
                    "MH": np.full(n, mh_round),
                    "Mini": masses,
                    "Mass": masses * 0.99,
                    "logL": logL,
                    "logTe": logTe,
                    "logg": logg_val,
                    "Gmag": Gmag,
                    "G_BPmag": G_BPmag,
                    "G_RPmag": G_RPmag,
                    "label": np.ones(n, dtype=int),
                }

        grid._build_tree()
        print(f"  Created {len(grid._grid)} synthetic isochrones")
        return grid

    @property
    def grid_points(self):
        """Array of (log_age, [M/H]) grid points."""
        return self._grid_points


def _turnoff_mass(age_gyr):
    """Approximate main-sequence turnoff mass for a given age."""
    if age_gyr <= 0:
        return 100.0
    return min(100.0, (age_gyr / 10.0) ** (-0.4) * 1.0)
