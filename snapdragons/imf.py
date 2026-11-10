"""Initial Mass Function sampling.

Supports Kroupa (2001) and Salpeter (1955) IMFs with efficient
inverse-CDF sampling, including truncated sampling above a minimum mass.
"""

import numpy as np


class KroupaIMF:
    """Kroupa (2001) three-segment broken power-law IMF.

    dN/dm ∝ m^{-alpha_i} with continuity at break points.

    Segments:
        alpha_1 = 1.3  for  0.08 <= m < 0.50 Msun
        alpha_2 = 2.3  for  0.50 <= m          Msun

    Brown dwarfs (m < 0.08) are excluded by default since they
    are unobservable by Gaia.
    """

    def __init__(self, m_min=0.08, m_max=100.0):
        self.m_min = m_min
        self.m_max = m_max
        self.m_break = 0.50
        self.alpha1 = 1.3
        self.alpha2 = 2.3
        self.name = "Kroupa (2001)"

        self._k1 = 1.0
        self._k2 = self._k1 * self.m_break ** (self.alpha2 - self.alpha1)

        self._build_cdf(m_min, m_max)
        self._trunc_cache = {}

    def _segment_number_integral(self, m_lo, m_hi, k, alpha):
        """∫ k * m^{-alpha} dm from m_lo to m_hi."""
        if alpha == 1.0:
            return k * np.log(m_hi / m_lo)
        exp = 1.0 - alpha
        return k / exp * (m_hi ** exp - m_lo ** exp)

    def _segment_mass_integral(self, m_lo, m_hi, k, alpha):
        """∫ k * m^{1-alpha} dm from m_lo to m_hi."""
        if alpha == 2.0:
            return k * np.log(m_hi / m_lo)
        exp = 2.0 - alpha
        return k / exp * (m_hi ** exp - m_lo ** exp)

    def _build_cdf(self, m_lo, m_hi):
        """Precompute CDF boundaries for inverse-CDF sampling."""
        self._m_lo = m_lo
        self._m_hi = m_hi

        self._segments = []
        boundaries = sorted(set(
            [m_lo] + [self.m_break] + [m_hi]
        ))
        boundaries = [b for b in boundaries if m_lo <= b <= m_hi]
        if boundaries[0] != m_lo:
            boundaries.insert(0, m_lo)
        if boundaries[-1] != m_hi:
            boundaries.append(m_hi)

        for i in range(len(boundaries) - 1):
            lo, hi = boundaries[i], boundaries[i + 1]
            if lo >= hi:
                continue
            if lo < self.m_break:
                k, alpha = self._k1, self.alpha1
            else:
                k, alpha = self._k2, self.alpha2
            n_int = self._segment_number_integral(lo, hi, k, alpha)
            self._segments.append((lo, hi, k, alpha, n_int))

        self._total_n = sum(s[4] for s in self._segments)
        cum = 0.0
        self._cdf_bounds = []
        for lo, hi, k, alpha, n_int in self._segments:
            f_lo = cum / self._total_n
            cum += n_int
            f_hi = cum / self._total_n
            self._cdf_bounds.append((f_lo, f_hi))

    def _invert_segment(self, u, m_lo, k, alpha):
        """Invert the CDF within a single power-law segment."""
        if alpha == 1.0:
            return m_lo * np.exp(u / k)
        exp = 1.0 - alpha
        return (m_lo ** exp + u * exp / k) ** (1.0 / exp)

    def sample(self, n, rng=None):
        """Draw n masses from the IMF via inverse-CDF."""
        if rng is None:
            rng = np.random.default_rng()
        u = rng.uniform(0.0, 1.0, size=n)
        masses = np.empty(n)

        for (lo, hi, k, alpha, n_int), (f_lo, f_hi) in zip(
                self._segments, self._cdf_bounds):
            mask = (u >= f_lo) & (u < f_hi)
            if not np.any(mask):
                continue
            u_seg = (u[mask] - f_lo) / (f_hi - f_lo) * n_int
            masses[mask] = self._invert_segment(u_seg, lo, k, alpha)

        return masses

    def sample_truncated(self, n, m_min_trunc, rng=None):
        """Draw n masses from the IMF truncated below m_min_trunc."""
        if m_min_trunc <= self._m_lo:
            return self.sample(n, rng=rng)
        if m_min_trunc >= self._m_hi:
            return np.array([])

        key = round(m_min_trunc, 6)
        if key not in self._trunc_cache:
            trunc = KroupaIMF.__new__(KroupaIMF)
            trunc.m_min = m_min_trunc
            trunc.m_max = self.m_max
            trunc.m_break = self.m_break
            trunc.alpha1 = self.alpha1
            trunc.alpha2 = self.alpha2
            trunc._k1 = self._k1
            trunc._k2 = self._k2
            trunc._trunc_cache = {}
            trunc._build_cdf(m_min_trunc, self._m_hi)
            self._trunc_cache[key] = trunc
        return self._trunc_cache[key].sample(n, rng=rng)

    def mean_mass(self, m_lo=None, m_hi=None):
        """Mean stellar mass <m> = ∫ m ξ(m) dm / ∫ ξ(m) dm."""
        m_lo = m_lo or self.m_min
        m_hi = m_hi or self.m_max
        boundaries = sorted(set([m_lo, self.m_break, m_hi]))
        boundaries = [b for b in boundaries if m_lo <= b <= m_hi]

        total_n = 0.0
        total_m = 0.0
        for i in range(len(boundaries) - 1):
            lo, hi = boundaries[i], boundaries[i + 1]
            if lo >= hi:
                continue
            k, alpha = (self._k1, self.alpha1) if lo < self.m_break \
                else (self._k2, self.alpha2)
            total_n += self._segment_number_integral(lo, hi, k, alpha)
            total_m += self._segment_mass_integral(lo, hi, k, alpha)
        return total_m / total_n if total_n > 0 else 0.0

    # --- Vectorised number CDF and its inverse (for per-particle truncation) ---

    def cdf(self, m):
        """Number CDF F(m) = N(< m) / N_total over [m_min, m_max], vectorised."""
        m = np.clip(np.asarray(m, dtype=np.float64), self.m_min, self.m_max)
        out = np.zeros_like(m)
        cum = 0.0
        for lo, hi, k, alpha, n_int in self._segments:
            inseg = (m > lo) & (m <= hi)
            above = m > hi
            out[inseg] = cum + self._segment_number_integral(lo, m[inseg], k, alpha)
            out[above] = cum + n_int
            cum += n_int
        return out / self._total_n

    def ppf(self, u):
        """Inverse number CDF: mass at quantile u in [0, 1], vectorised.

        Uniform u between cdf(m_lo) and cdf(m_hi) gives masses distributed
        as the IMF truncated to [m_lo, m_hi].
        """
        u = np.asarray(u, dtype=np.float64)
        out = np.empty_like(u)
        for (lo, hi, k, alpha, n_int), (f_lo, f_hi) in zip(
                self._segments, self._cdf_bounds):
            mask = (u >= f_lo) & (u < f_hi)
            if not np.any(mask):
                continue
            out[mask] = self._invert_segment(
                (u[mask] - f_lo) * self._total_n, lo, k, alpha)
        out[u >= self._cdf_bounds[-1][1]] = self._m_hi
        out[u < self._cdf_bounds[0][0]] = self._m_lo
        return out

    def stars_per_msun_between(self, m_lo, m_hi):
        """Number of stars with initial mass in [m_lo, m_hi] per solar mass
        of population formed over the full IMF range, vectorised."""
        frac = np.clip(self.cdf(m_hi) - self.cdf(m_lo), 0.0, None)
        return frac / self.mean_mass()

    def number_fraction_above(self, m_min_trunc):
        """Fraction of stars (by number) with mass >= m_min_trunc."""
        if m_min_trunc <= self.m_min:
            return 1.0
        if m_min_trunc >= self.m_max:
            return 0.0
        n_above = 0.0
        boundaries = sorted(set([m_min_trunc, self.m_break, self.m_max]))
        boundaries = [b for b in boundaries if m_min_trunc <= b <= self.m_max]
        for i in range(len(boundaries) - 1):
            lo, hi = boundaries[i], boundaries[i + 1]
            if lo >= hi:
                continue
            k, alpha = (self._k1, self.alpha1) if lo < self.m_break \
                else (self._k2, self.alpha2)
            n_above += self._segment_number_integral(lo, hi, k, alpha)
        return n_above / self._total_n

    def expected_count(self, particle_mass, m_min_trunc=None):
        """Expected number of stars above m_min_trunc from a particle.

        N = M_particle * ∫_{m_min_trunc}^{m_max} ξ(m) dm
                       / ∫_{m_min}^{m_max} m ξ(m) dm
        """
        m_lo = m_min_trunc or self.m_min
        if m_lo >= self.m_max:
            return 0.0

        mass_norm = 0.0
        boundaries_full = sorted(set([self.m_min, self.m_break, self.m_max]))
        boundaries_full = [b for b in boundaries_full
                           if self.m_min <= b <= self.m_max]
        for i in range(len(boundaries_full) - 1):
            lo, hi = boundaries_full[i], boundaries_full[i + 1]
            if lo >= hi:
                continue
            k, alpha = (self._k1, self.alpha1) if lo < self.m_break \
                else (self._k2, self.alpha2)
            mass_norm += self._segment_mass_integral(lo, hi, k, alpha)

        n_above = 0.0
        boundaries_trunc = sorted(set([m_lo, self.m_break, self.m_max]))
        boundaries_trunc = [b for b in boundaries_trunc
                            if m_lo <= b <= self.m_max]
        for i in range(len(boundaries_trunc) - 1):
            lo, hi = boundaries_trunc[i], boundaries_trunc[i + 1]
            if lo >= hi:
                continue
            k, alpha = (self._k1, self.alpha1) if lo < self.m_break \
                else (self._k2, self.alpha2)
            n_above += self._segment_number_integral(lo, hi, k, alpha)

        return particle_mass * n_above / mass_norm


class SalpeterIMF:
    """Salpeter (1955) single power-law IMF: dN/dm ∝ m^{-2.35}."""

    def __init__(self, m_min=0.08, m_max=100.0):
        self.m_min = m_min
        self.m_max = m_max
        self.m_break = m_max + 1.0
        self.alpha = 2.35
        self.alpha1 = self.alpha
        self.alpha2 = self.alpha
        self._k1 = 1.0
        self._k2 = 1.0
        self.name = "Salpeter (1955)"
        self._exp = 1.0 - self.alpha

    def _segment_mass_integral(self, m_lo, m_hi, k, alpha):
        """∫ k * m^{1-alpha} dm from m_lo to m_hi."""
        if alpha == 2.0:
            return k * np.log(m_hi / m_lo)
        exp = 2.0 - alpha
        return k / exp * (m_hi ** exp - m_lo ** exp)

    def sample(self, n, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        u = rng.uniform(0.0, 1.0, size=n)
        lo_exp = self.m_min ** self._exp
        hi_exp = self.m_max ** self._exp
        return (lo_exp + u * (hi_exp - lo_exp)) ** (1.0 / self._exp)

    def sample_truncated(self, n, m_min_trunc, rng=None):
        if m_min_trunc <= self.m_min:
            return self.sample(n, rng=rng)
        if m_min_trunc >= self.m_max:
            return np.array([])
        if rng is None:
            rng = np.random.default_rng()
        u = rng.uniform(0.0, 1.0, size=n)
        lo_exp = m_min_trunc ** self._exp
        hi_exp = self.m_max ** self._exp
        return (lo_exp + u * (hi_exp - lo_exp)) ** (1.0 / self._exp)

    def mean_mass(self, m_lo=None, m_hi=None):
        m_lo = m_lo or self.m_min
        m_hi = m_hi or self.m_max
        exp_n = self._exp
        exp_m = 2.0 - self.alpha
        n_int = (m_hi ** exp_n - m_lo ** exp_n) / exp_n
        m_int = (m_hi ** exp_m - m_lo ** exp_m) / exp_m
        return m_int / n_int

    def cdf(self, m):
        """Number CDF over [m_min, m_max], vectorised."""
        m = np.clip(np.asarray(m, dtype=np.float64), self.m_min, self.m_max)
        lo_exp = self.m_min ** self._exp
        hi_exp = self.m_max ** self._exp
        return (m ** self._exp - lo_exp) / (hi_exp - lo_exp)

    def ppf(self, u):
        """Inverse number CDF, vectorised."""
        u = np.clip(np.asarray(u, dtype=np.float64), 0.0, 1.0)
        lo_exp = self.m_min ** self._exp
        hi_exp = self.m_max ** self._exp
        return (lo_exp + u * (hi_exp - lo_exp)) ** (1.0 / self._exp)

    def stars_per_msun_between(self, m_lo, m_hi):
        """Stars with initial mass in [m_lo, m_hi] per solar mass formed."""
        frac = np.clip(self.cdf(m_hi) - self.cdf(m_lo), 0.0, None)
        return frac / self.mean_mass()

    def expected_count(self, particle_mass, m_min_trunc=None):
        m_lo = m_min_trunc or self.m_min
        if m_lo >= self.m_max:
            return 0.0
        exp_n = self._exp
        exp_m = 2.0 - self.alpha
        mass_norm = (self.m_max ** exp_m - self.m_min ** exp_m) / exp_m
        n_above = (self.m_max ** exp_n - m_lo ** exp_n) / exp_n
        return particle_mass * n_above / mass_norm


def get_imf(name, m_min=0.08, m_max=100.0):
    """Factory function to get an IMF by name."""
    name = name.lower()
    if name == "kroupa":
        return KroupaIMF(m_min=m_min, m_max=m_max)
    elif name == "salpeter":
        return SalpeterIMF(m_min=m_min, m_max=m_max)
    else:
        raise ValueError(f"Unknown IMF '{name}'. Choose 'kroupa' or 'salpeter'.")
