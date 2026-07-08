"""Scalar Einstein-Boltzmann perturbation-equation tests.

Ported from the DISCO2 ``tests/test_perturbations.py`` suite, keeping the
analytical (pure-function) checks of the perturbation equations, initial
conditions, and power-spectrum mapping. The full ODE-solve comparison (DISCO2
validated a numba-CUDA solve against CAMB) is intentionally excluded here; the
end-to-end solve is covered by ``tests/test_perturbations_system.py`` against
CLASS.
"""

import math

import numpy as np
import pytest

from discoeb.background import (
    grhob,
    grhoc,
    grhog,
    grhornomass,
    grhov,
    hubble_a,
)
import discoeb.matter_power_spectrum as M
import discoeb.perturbations as P

from cosmologies import BENCHMARK_COSMOLOGIES


DEFAULT_COSMOLOGY = BENCHMARK_COSMOLOGIES["planck_2018_flat_lcdm"]

COSMOLOGY_CASES = [
    pytest.param(DEFAULT_COSMOLOGY, id="planck_2018_flat_lcdm"),
]


def density_coefficients(cosmology) -> tuple[float, float, float, float, float]:
    """Return the ``grho`` density coefficients ``(g, r, c, b, v)`` for a cosmology."""

    grhog_value = grhog(cosmology.T_cmb)
    return (
        grhog_value,
        grhornomass(grhog_value, cosmology.Neff_massless),
        grhoc(cosmology.omega_c_h2),
        grhob(cosmology.omega_b_h2),
        grhov(
            cosmology.omega_b_h2,
            cosmology.omega_c_h2,
            cosmology.h,
            cosmology.Neff_massless,
            cosmology.T_cmb,
        ),
    )


# ============================================================
# ANALYTICAL TESTS (pure functions, no integration)
# ============================================================


def test_state_layout_is_consistent_with_config():
    """Check that the state-vector indices tile the hierarchy without gaps."""

    assert P.IX_ETAK == 0
    assert P.IX_G == P.IX_VB + 1
    assert P.IX_POL == P.IX_G + P.LMAX_G + 1
    assert P.IX_R == P.IX_POL + (P.LMAX_POL - 1)
    assert P.NVAR == P.IX_R + P.LMAX_NR + 1


def test_comoving_densities_scale_with_species_dilution():
    """Check the ``8*pi*G*rho_i a^2`` coefficients dilute with the correct powers of a."""

    g, r, c, b, v = 1.0, 2.0, 3.0, 4.0, 5.0
    a1, a2 = 0.1, 0.4
    g1, r1, c1, b1, v1 = P.comoving_densities(a1, g, r, c, b, v)
    g2, r2, c2, b2, v2 = P.comoving_densities(a2, g, r, c, b, v)

    # Radiation ~ a^-2, matter ~ a^-1, Lambda ~ a^2 in these comoving coefficients.
    assert g2 / g1 == pytest.approx((a1 / a2) ** 2)
    assert r2 / r1 == pytest.approx((a1 / a2) ** 2)
    assert c2 / c1 == pytest.approx(a1 / a2)
    assert b2 / b1 == pytest.approx(a1 / a2)
    assert v2 / v1 == pytest.approx((a2 / a1) ** 2)


@pytest.mark.parametrize("cosmology", COSMOLOGY_CASES)
def test_expansion_rate_matches_background_hubble(cosmology):
    """Check ``adotoa = a H(a)`` against the independent background module."""

    dens = density_coefficients(cosmology)
    for a in (1.0e-4, 1.0e-2, 0.5, 1.0):
        adotoa = P.expansion_rate(*P.comoving_densities(a, *dens))
        assert adotoa == pytest.approx(a * hubble_a(a, *dens), rel=1.0e-13)


def test_metric_constraints_invert_their_definitions():
    """Check the metric helpers ``z`` and ``sigma`` satisfy their defining algebra."""

    dgrho, etak, adotoa, k, dgq = 1.3e-6, -0.2, 3.0e-4, 0.05, 4.0e-7
    z = P.metric_z(dgrho, etak, adotoa, k)
    sigma = P.shear_sigma(z, dgq, k)

    assert z * adotoa == pytest.approx(0.5 * dgrho / k + etak)
    assert sigma - z == pytest.approx(1.5 * dgq / k**2)


@pytest.mark.parametrize("cosmology", COSMOLOGY_CASES)
def test_adiabatic_initial_conditions_super_horizon_ratios(cosmology):
    """Check the adiabatic mode obeys its defining ratios as ``k tau -> 0``.

    Deep in radiation domination the growing adiabatic mode has
    ``delta_c = delta_b = (3/4) delta_gamma``, ``delta_nu = delta_gamma``,
    ``v_b = (3/4) q_gamma``, and the metric perturbation ``etak -> -k``.
    """

    g, r, c, b, v = density_coefficients(cosmology)
    k = 0.2
    tau = 1.0e-3 / k  # k tau = 1e-3, deep super-horizon
    y = P.adiabatic_initial_conditions(k, tau, g, r, c, b)

    clxg = y[P.IX_G]
    assert clxg > 0.0
    assert y[P.IX_CLXC] == pytest.approx(0.75 * clxg, rel=1.0e-12)
    assert y[P.IX_CLXB] == pytest.approx(0.75 * clxg, rel=1.0e-12)
    assert y[P.IX_VB] == pytest.approx(0.75 * y[P.IX_G + 1], rel=1.0e-12)
    assert y[P.IX_R] == pytest.approx(clxg, rel=1.0e-12)
    assert clxg == pytest.approx((k * tau) ** 2 / 3.0, rel=1.0e-3)
    assert y[P.IX_ETAK] == pytest.approx(-k, rel=1.0e-4)
    assert y[P.IX_POL] == 0.0
    assert y[P.IX_R + 4] == 0.0


@pytest.mark.parametrize("cosmology", COSMOLOGY_CASES)
def test_neutrino_radiation_fraction_is_physical(cosmology):
    """Check the neutrino radiation fraction lies in (0, 1) and matches the ratio."""

    g, r = density_coefficients(cosmology)[:2]
    Rv = P.neutrino_radiation_fraction(g, r)
    assert 0.0 < Rv < 1.0
    assert Rv == pytest.approx(r / (g + r), rel=1.0e-14)


def test_multipole_recurrence_coefficients():
    """Anchor the free-streaming recurrence coefficients against hand computation."""

    k, opacity = 0.1, 0.5
    assert P.photon_temperature_multipole_derivative(
        5, 1.0, 2.0, 3.0, k, opacity
    ) == pytest.approx(k * 5 / 11 * 1.0 - k * 6 / 11 * 3.0 - opacity * 2.0)
    assert P.neutrino_multipole_derivative(5, 1.0, 2.0, 3.0, k) == pytest.approx(
        k * 5 / 11 * 1.0 - k * 6 / 11 * 3.0
    )
    polfac = 8 * 4 / 6
    assert P.polarization_multipole_derivative(
        5, 1.0, 2.0, 3.0, k, opacity
    ) == pytest.approx(-opacity * 2.0 + k * 5 / 11 * 1.0 - polfac * k / 11 * 3.0)


def test_matter_power_spectrum_normalization():
    """Check the primordial and matter power-spectrum normalizations."""

    A_s, n_s, k_pivot = 2.1e-9, 0.96, 0.05
    assert M.primordial_curvature_power(k_pivot, A_s, n_s, k_pivot) == pytest.approx(
        A_s
    )
    k, delta_m = 0.1, 12.3
    expected = (
        2.0
        * math.pi**2
        / k**3
        * M.primordial_curvature_power(k, A_s, n_s, k_pivot)
        * delta_m**2
    )
    assert M.matter_power_spectrum(k, delta_m, A_s, n_s, k_pivot) == pytest.approx(
        expected
    )


def test_total_matter_contrast_is_density_weighted():
    """Check the total-matter contrast reduces to the density-weighted mean."""

    assert M.total_matter_density_contrast(8.0, 1.0, 2.0, 6.0) == pytest.approx(
        (8.0 * 1.0 + 2.0 * 6.0) / (8.0 + 2.0)
    )


@pytest.mark.parametrize("cosmology", COSMOLOGY_CASES)
def test_boltzmann_rhs_metric_and_matter_wiring(cosmology):
    """Check the RHS wires the metric, CDM, and baryon equations to the helpers."""

    dens = density_coefficients(cosmology)
    k, a, tau = 0.1, 1.0e-3, 250.0
    rng = np.random.default_rng(1)
    y = rng.standard_normal(P.NVAR) * 1.0e-3
    dy = np.asarray(P.boltzmann_rhs(tau, y, k, a, 50.0, 1.0e-10, *dens))

    assert dy.shape == (P.NVAR,)

    g, r, c, b, _ = P.comoving_densities(a, *dens)
    adotoa = P.expansion_rate(*P.comoving_densities(a, *dens))
    dgrho = P.density_perturbation(
        b, y[P.IX_CLXB], c, y[P.IX_CLXC], g, y[P.IX_G], r, y[P.IX_R]
    )
    dgq = P.momentum_perturbation(b, y[P.IX_VB], g, y[P.IX_G + 1], r, y[P.IX_R + 1])
    z = P.metric_z(dgrho, y[P.IX_ETAK], adotoa, k)

    assert dy[P.IX_ETAK] == pytest.approx(0.5 * dgq)
    assert dy[P.IX_CLXC] == pytest.approx(-k * z)
    assert dy[P.IX_CLXB] == pytest.approx(-k * (z + y[P.IX_VB]))
    assert dy[P.IX_G] == pytest.approx(-k * (4.0 / 3.0 * z + y[P.IX_G + 1]))


@pytest.mark.parametrize("cosmology", COSMOLOGY_CASES)
def test_boltzmann_rhs_evolves_full_photon_hierarchy(cosmology):
    """Check that the full photon/polarization hierarchy evolves at high opacity.

    Without a tight-coupling approximation, the higher multipoles are driven by
    the exact Thomson scattering terms even deep in the early universe, so their
    derivatives are non-zero whenever the corresponding multipoles are.
    """

    dens = density_coefficients(cosmology)
    k, a, tau, opacity = 0.1, 1.0e-6, 5.0, 1.0e7
    rng = np.random.default_rng(2)
    y = rng.standard_normal(P.NVAR) * 1.0e-3
    dy = P.boltzmann_rhs(tau, y, k, a, opacity, 1.0e-9, *dens)

    for ell in range(3, P.LMAX_G):
        assert dy[P.IX_G + ell] == pytest.approx(
            P.photon_temperature_multipole_derivative(
                ell,
                y[P.IX_G + ell - 1],
                y[P.IX_G + ell],
                y[P.IX_G + ell + 1],
                k,
                opacity,
            )
        )
    for idx in range(P.IX_POL + 1, P.IX_R):
        assert dy[idx] != 0.0

    g, r, c, b, _ = P.comoving_densities(a, *dens)
    adotoa = P.expansion_rate(g, r, c, b, P.comoving_densities(a, *dens)[4])
    dgrho = P.density_perturbation(
        b, y[P.IX_CLXB], c, y[P.IX_CLXC], g, y[P.IX_G], r, y[P.IX_R]
    )
    dgq = P.momentum_perturbation(b, y[P.IX_VB], g, y[P.IX_G + 1], r, y[P.IX_R + 1])
    z = P.metric_z(dgrho, y[P.IX_ETAK], adotoa, k)
    sigma = P.shear_sigma(z, dgq, k)
    polter = P.polarization_source(y[P.IX_G + 2], y[P.IX_POL])
    expected_pig = P.photon_quadrupole_derivative(
        y[P.IX_G + 1], y[P.IX_G + 3], y[P.IX_G + 2], polter, sigma, k, opacity
    )
    assert dy[P.IX_G + 2] == pytest.approx(expected_pig)
