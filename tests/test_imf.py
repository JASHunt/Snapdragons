"""IMF: normalisation, CDF / inverse, sampling, truncation, expected counts."""

import numpy as np
import pytest
from scipy import integrate, stats

from snapdragons.imf import KroupaIMF, SalpeterIMF, get_imf


def kroupa_pdf(m):
    """Unnormalised Kroupa (2001) dN/dm, written independently of the code:
    slope 1.3 below 0.5 Msun, 2.3 above, continuous at the break."""
    return m ** -1.3 if m < 0.5 else 0.5 * m ** -2.3


def salpeter_pdf(m):
    return m ** -2.35


IMFS = [(KroupaIMF, kroupa_pdf), (SalpeterIMF, salpeter_pdf)]


def _quad(f, lo, hi):
    pts = [0.5] if lo < 0.5 < hi else None
    return integrate.quad(f, lo, hi, points=pts, limit=200)[0]


@pytest.mark.parametrize("cls,pdf", IMFS)
def test_cdf_matches_numerical_integral(cls, pdf):
    imf = cls()
    norm = _quad(pdf, imf.m_min, imf.m_max)
    m = np.array([0.08, 0.1, 0.3, 0.5, 0.8, 1.0, 5.0, 30.0, 100.0])
    expected = np.array([_quad(pdf, imf.m_min, mi) / norm for mi in m])
    assert np.allclose(imf.cdf(m), expected, atol=1e-9)
    assert imf.cdf(np.array([imf.m_min]))[0] == pytest.approx(0.0, abs=1e-12)
    assert imf.cdf(np.array([imf.m_max]))[0] == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("cls,pdf", IMFS)
def test_cdf_clips_outside_mass_range(cls, pdf):
    imf = cls()
    assert np.allclose(imf.cdf(np.array([0.01, 500.0])), [0.0, 1.0])


@pytest.mark.parametrize("cls,pdf", IMFS)
def test_ppf_inverts_cdf(cls, pdf):
    imf = cls()
    m = np.logspace(np.log10(0.081), np.log10(99.0), 200)
    assert np.allclose(imf.ppf(imf.cdf(m)), m, rtol=1e-9)
    assert np.allclose(imf.ppf(np.array([0.0, 1.0])), [imf.m_min, imf.m_max])


@pytest.mark.parametrize("cls,pdf", IMFS)
def test_cdf_is_monotonic(cls, pdf):
    imf = cls()
    m = np.logspace(np.log10(0.08), 2, 500)
    assert np.all(np.diff(imf.cdf(m)) > 0)


@pytest.mark.parametrize("cls,pdf", IMFS)
def test_mean_mass_matches_numerical_integral(cls, pdf):
    imf = cls()
    num = _quad(lambda m: m * pdf(m), imf.m_min, imf.m_max)
    den = _quad(pdf, imf.m_min, imf.m_max)
    assert imf.mean_mass() == pytest.approx(num / den, rel=1e-8)
    # and over a sub-range
    num = _quad(lambda m: m * pdf(m), 0.3, 5.0)
    den = _quad(pdf, 0.3, 5.0)
    assert imf.mean_mass(0.3, 5.0) == pytest.approx(num / den, rel=1e-8)


def test_kroupa_mean_mass_value():
    # 0.08-100 Msun, slopes 1.3 / 2.3: pinned so a change of the IMF
    # definition (range, break, slopes) is noticed
    assert KroupaIMF().mean_mass() == pytest.approx(0.5738648, rel=1e-6)


@pytest.mark.parametrize("cls,pdf", IMFS)
def test_sample_follows_the_imf(cls, pdf, rng):
    imf = cls()
    m = imf.sample(200_000, rng=rng)
    assert m.shape == (200_000,)
    assert m.min() >= imf.m_min and m.max() <= imf.m_max
    assert stats.kstest(m, imf.cdf).pvalue > 1e-3
    # the mean is dominated by rare massive stars: 3 sigma of the sample mean
    assert abs(m.mean() - imf.mean_mass()) < 3 * m.std() / np.sqrt(len(m))


@pytest.mark.parametrize("cls,pdf", IMFS)
@pytest.mark.parametrize("m_trunc", [0.2, 0.5, 0.9, 3.0])
def test_sample_truncated_follows_the_truncated_imf(cls, pdf, m_trunc, rng):
    imf = cls()
    m = imf.sample_truncated(100_000, m_trunc, rng=rng)
    assert m.min() >= m_trunc and m.max() <= imf.m_max
    f0 = imf.cdf(np.array([m_trunc]))[0]

    def trunc_cdf(x):
        return (imf.cdf(x) - f0) / (1.0 - f0)

    assert stats.kstest(m, trunc_cdf).pvalue > 1e-3


@pytest.mark.parametrize("cls,pdf", IMFS)
def test_sample_truncated_edge_cases(cls, pdf, rng):
    imf = cls()
    # truncation below the lower limit is the full IMF
    assert len(imf.sample_truncated(10, 0.01, rng=rng)) == 10
    # truncation at or above the upper limit gives nothing
    assert len(imf.sample_truncated(10, imf.m_max, rng=rng)) == 0
    assert len(imf.sample_truncated(10, 500.0, rng=rng)) == 0


@pytest.mark.parametrize("cls,pdf", IMFS)
def test_ppf_between_cdf_bounds_samples_a_mass_interval(cls, pdf, rng):
    """u ~ U(cdf(lo), cdf(hi)) -> ppf(u) is the IMF restricted to [lo, hi]:
    the per-particle truncation used by external mock layers."""
    imf = cls()
    lo, hi = 0.7, 4.0
    u = rng.uniform(imf.cdf(np.array([lo]))[0], imf.cdf(np.array([hi]))[0],
                    50_000)
    m = imf.ppf(u)
    assert m.min() >= lo and m.max() <= hi
    f_lo, f_hi = imf.cdf(np.array([lo, hi]))
    assert stats.kstest(m, lambda x: (imf.cdf(x) - f_lo) / (f_hi - f_lo)).pvalue > 1e-3


@pytest.mark.parametrize("cls,pdf", IMFS)
def test_expected_count_matches_numerical_integral(cls, pdf):
    """Stars above m_trunc per unit mass formed over the full IMF."""
    imf = cls()
    mass_norm = _quad(lambda m: m * pdf(m), imf.m_min, imf.m_max)
    for m_trunc in (None, 0.3, 1.0, 8.0):
        lo = imf.m_min if m_trunc is None else m_trunc
        expected = 1e4 * _quad(pdf, lo, imf.m_max) / mass_norm
        assert imf.expected_count(1e4, m_trunc) == pytest.approx(expected, rel=1e-8)
    assert imf.expected_count(1e4, imf.m_max) == 0.0
    assert imf.expected_count(1e4, 500.0) == 0.0


@pytest.mark.parametrize("cls,pdf", IMFS)
def test_stars_per_msun_between(cls, pdf):
    imf = cls()
    lo = np.array([0.08, 0.5, 1.0, 2.0])
    hi = np.array([100.0, 2.0, 8.0, 1.0])     # last one inverted -> 0
    got = imf.stars_per_msun_between(lo, hi)
    assert got[0] == pytest.approx(1.0 / imf.mean_mass(), rel=1e-10)
    assert got[1] == pytest.approx(
        imf.expected_count(1.0, 0.5) - imf.expected_count(1.0, 2.0), rel=1e-9)
    assert got[2] == pytest.approx(
        imf.expected_count(1.0, 1.0) - imf.expected_count(1.0, 8.0), rel=1e-9)
    assert got[3] == 0.0


def test_kroupa_number_fraction_above():
    imf = KroupaIMF()
    for m in (0.2, 0.5, 1.0, 10.0):
        assert imf.number_fraction_above(m) == pytest.approx(
            1.0 - imf.cdf(np.array([m]))[0], rel=1e-10)
    assert imf.number_fraction_above(0.01) == 1.0
    assert imf.number_fraction_above(200.0) == 0.0


def test_kroupa_is_continuous_at_the_break():
    imf = KroupaIMF()
    below = imf._k1 * imf.m_break ** -imf.alpha1
    above = imf._k2 * imf.m_break ** -imf.alpha2
    assert below == pytest.approx(above, rel=1e-12)


def test_custom_mass_range():
    imf = KroupaIMF(m_min=0.1, m_max=50.0)
    m = imf.sample(20_000, rng=np.random.default_rng(1))
    assert m.min() >= 0.1 and m.max() <= 50.0
    assert imf.cdf(np.array([0.1, 50.0])) == pytest.approx([0.0, 1.0])


def test_get_imf_factory():
    assert isinstance(get_imf("kroupa"), KroupaIMF)
    assert isinstance(get_imf("Salpeter", m_min=0.1), SalpeterIMF)
    assert get_imf("kroupa", m_min=0.1, m_max=50.0).m_max == 50.0
    with pytest.raises(ValueError):
        get_imf("chabrier")
