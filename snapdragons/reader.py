"""Reader for Snapdragons output catalogues.

Provides a convenient interface to access true, observed, and
uncertainty data from HDF5 or FITS output files.
"""

import numpy as np


class CatalogueData:
    """Attribute-access wrapper around a dict of arrays."""

    def __init__(self, data):
        self._data = data

    def __getattr__(self, name):
        if name.startswith("_"):
            return super().__getattribute__(name)
        if name in self._data:
            return self._data[name]
        raise AttributeError(f"No column '{name}' in catalogue")

    def __contains__(self, name):
        return name in self._data

    def __repr__(self):
        cols = list(self._data.keys())
        return f"CatalogueData({len(cols)} columns: {cols})"

    @property
    def columns(self):
        return list(self._data.keys())

    def to_dict(self):
        return dict(self._data)

    def to_dataframe(self):
        import pandas as pd
        return pd.DataFrame(self._data)


class Catalogue:
    """A Snapdragons output catalogue.

    Attributes
    ----------
    true : CatalogueData
        True (noise-free) astrometry, photometry, coordinates.
    observed : CatalogueData
        Observed values with Gaia errors applied.
    uncertainties : CatalogueData
        1-sigma uncertainties for each observed quantity.
    stellar : CatalogueData
        Intrinsic stellar properties (mass, age, metallicity, etc.).
    metadata : dict
        Run configuration metadata.
    n_stars : int
        Number of stars in the catalogue.
    """

    def __init__(self, true, observed, uncertainties, stellar, metadata):
        self.true = CatalogueData(true)
        self.observed = CatalogueData(observed)
        self.uncertainties = CatalogueData(uncertainties)
        self.stellar = CatalogueData(stellar)
        self.metadata = metadata

        self.n_stars = 0
        for v in true.values():
            if isinstance(v, np.ndarray):
                self.n_stars = len(v)
                break

    def __repr__(self):
        return (f"Catalogue({self.n_stars:,} stars, "
                f"release={self.metadata.get('gaia_release', '?')})")

    def __len__(self):
        return self.n_stars

    def to_dataframe(self, include=("true", "observed", "stellar")):
        """Convert to a single pandas DataFrame.

        Parameters
        ----------
        include : tuple of str
            Which sub-catalogues to include. Column names are prefixed
            with the group name (e.g. true_ra, obs_Gmag, Mini).
        """
        import pandas as pd

        dfs = {}
        for group in include:
            data = getattr(self, group)
            prefix = "" if group == "stellar" else f"{group}_"
            for col in data.columns:
                dfs[f"{prefix}{col}"] = getattr(data, col)
        return pd.DataFrame(dfs)


def read_catalogue(filepath):
    """Read a Snapdragons catalogue from HDF5 or FITS.

    Parameters
    ----------
    filepath : str
        Path to the catalogue file.

    Returns
    -------
    Catalogue
    """
    if filepath.endswith((".h5", ".hdf5", ".he5")):
        return _read_hdf5(filepath)
    elif filepath.endswith((".fits", ".fit")):
        return _read_fits(filepath)
    else:
        try:
            return _read_hdf5(filepath)
        except Exception:
            return _read_fits(filepath)


def _read_hdf5(filepath):
    """Read HDF5 catalogue."""
    import h5py

    groups = {}
    metadata = {}

    with h5py.File(filepath, "r") as hf:
        for grp_name in ["true", "observed", "uncertainties", "stellar"]:
            if grp_name in hf:
                groups[grp_name] = {
                    key: hf[grp_name][key][:] for key in hf[grp_name].keys()
                }
            else:
                groups[grp_name] = {}

        if "metadata" in hf:
            for key in hf["metadata"].attrs:
                val = hf["metadata"].attrs[key]
                if isinstance(val, bytes):
                    val = val.decode()
                elif isinstance(val, np.ndarray):
                    val = val.tolist()
                metadata[key] = val

    return Catalogue(
        true=groups["true"],
        observed=groups["observed"],
        uncertainties=groups["uncertainties"],
        stellar=groups["stellar"],
        metadata=metadata,
    )


def _read_fits(filepath):
    """Read FITS catalogue."""
    from astropy.table import Table

    table = Table.read(filepath, format="fits")

    true = {}
    observed = {}
    uncertainties = {}
    stellar = {}

    for col in table.colnames:
        arr = np.array(table[col])
        if col.startswith("true_"):
            true[col[5:]] = arr
        elif col.startswith("obs_"):
            observed[col[4:]] = arr
        elif col.startswith("err_"):
            uncertainties[col[4:]] = arr
        else:
            stellar[col] = arr

    return Catalogue(
        true=true,
        observed=observed,
        uncertainties=uncertainties,
        stellar=stellar,
        metadata={},
    )
