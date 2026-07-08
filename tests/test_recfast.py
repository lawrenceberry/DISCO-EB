"""Scalar-primitive and RHS-anchor tests for :mod:`discoeb.recfast`.

Ported from the DISCO2 ``tests/test_recfast.py`` suite, keeping the scalar-helper
and right-hand-side checks. The end-to-end ODE-solve test (DISCO2 validated a
CUDA solve against CAMB) is intentionally excluded here; the fused
background+RECFAST solve is covered separately by
``tests/test_background_system.py`` against CLASS.
"""

import math

import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from discoeb.background import (
    grhob,
    grhoc,
    grhog,
    grhornomass,
    radiation_neutrino_factor,
)
from discoeb.background_system import recfast_parameters
from discoeb.constants import M_H, MPC_IN_M, RHO_CRIT_100_SI
from discoeb.recfast import (
    escape_probability,
    find_recfast_ode_start,
    helium_recombination_rate,
    helium_triplet_recombination_rate,
    hydrogen_photoionization_rate,
    hydrogen_recombination_rate,
    initial_thermal_state,
    recfast_rhs,
    recfast_rhs_with_tau,
    saha_he1,
    saha_he2,
)

from cosmologies import BENCHMARK_COSMOLOGIES


DEFAULT_COSMOLOGY = BENCHMARK_COSMOLOGIES["planck_2018_flat_lcdm"]

COSMOLOGY_CASES = [
    pytest.param(DEFAULT_COSMOLOGY, id="planck_2018_flat_lcdm"),
]


def _param_row(cosmology):
    """Pack a cosmology into the RECFAST parameter row expected by ``recfast_rhs``."""

    a = recfast_parameters(cosmology)
    return jnp.asarray(
        [
            a["T_cmb"],
            a["f_He"],
            a["Nnow"],
            a["H0_SI"],
            a["omega_m"],
            a["z_eq"],
            a["grhog"],
            a["grhornomass"],
            a["grhoc"],
            a["grhob"],
            a["grhov"],
        ],
        dtype=jnp.float64,
    )


@pytest.mark.parametrize("cosmology", COSMOLOGY_CASES)
def test_recfast_setup_helpers_match_direct_algebra(cosmology):
    """Check scalar setup helpers against their direct algebraic definitions."""

    args = recfast_parameters(cosmology)
    grhog_value = grhog(cosmology.T_cmb)
    grhornomass_value = grhornomass(grhog_value, cosmology.Neff_massless)
    grhoc_value = grhoc(cosmology.omega_c_h2)
    grhob_value = grhob(cosmology.omega_b_h2)

    assert args["Nnow"] > 0.0
    assert args["Nnow"] == pytest.approx(
        (1.0 - cosmology.Y_He) * (cosmology.omega_b_h2 * RHO_CRIT_100_SI) / M_H,
        rel=1.0e-15,
    )
    assert args["H0_SI"] > 0.0
    assert args["H0_SI"] == pytest.approx(
        100.0 * cosmology.h * 1.0e3 / MPC_IN_M, rel=1.0e-15
    )
    assert args["omega_m"] > 0.0
    assert args["omega_m"] == pytest.approx(
        (cosmology.omega_b_h2 + cosmology.omega_c_h2) / cosmology.h**2
    )
    assert args["z_eq"] > 0.0
    assert args["z_eq"] == pytest.approx(
        1.0 / ((grhog_value + grhornomass_value) / (grhoc_value + grhob_value)) - 1.0
    )


@pytest.mark.parametrize("cosmology", COSMOLOGY_CASES)
def test_saha_helium_fractions_are_bounded(cosmology):
    """Check Saha helium initial-condition helpers remain physically bounded."""

    args = recfast_parameters(cosmology)
    for z in (6000.0, 3500.0, 2500.0):
        he2 = saha_he2(z, args["T_cmb"], args["Nnow"], args["f_He"])
        he1 = saha_he1(z, args["T_cmb"], args["Nnow"], args["f_He"])
        assert 0.0 <= he2 <= 1.0 + 2.0 * args["f_He"]
        assert 0.0 <= he1 <= 1.0


def test_recombination_rates_are_finite_and_positive():
    """Check recombination/photoionization rate helpers on representative temperatures."""

    for T_mat in (500.0, 3000.0, 10000.0):
        h_recomb = hydrogen_recombination_rate(T_mat)
        assert math.isfinite(h_recomb)
        assert h_recomb > 0.0
        assert math.isfinite(hydrogen_photoionization_rate(T_mat, h_recomb))
        assert hydrogen_photoionization_rate(T_mat, h_recomb) > 0.0
        assert math.isfinite(helium_recombination_rate(T_mat))
        assert helium_recombination_rate(T_mat) > 0.0
        assert math.isfinite(helium_triplet_recombination_rate(T_mat))
        assert helium_triplet_recombination_rate(T_mat) > 0.0


def test_escape_probability_uses_correct_small_and_large_tau_forms():
    """Check the Sobolev escape probability in its series and exact branches."""

    tiny_tau = 1.0e-9
    ordinary_tau = 0.25
    assert escape_probability(tiny_tau) == pytest.approx(1.0 - tiny_tau / 2.0)
    assert escape_probability(ordinary_tau) == pytest.approx(
        (1.0 - math.exp(-ordinary_tau)) / ordinary_tau
    )


def test_recfast_rhs_regression_values():
    """Check JAX RHS values against fixed regression anchors (DISCO2 reference)."""

    p = _param_row(DEFAULT_COSMOLOGY)
    cases = [
        (3000.0, (1.0, 0.8, 8176.5)),
        (1200.0, (0.9, 1.0e-4, 3270.6)),
        (800.0, (0.1, 1.0e-8, 2100.0)),
    ]
    expected = [
        (2.258659206196e00, -1.921451734784e-02, 2.727707033967e00),
        (4.791649204017e-02, 9.420143681200e-05, 2.725728529924e00),
        (3.445970659691e-02, 3.540003019902e-09, 2.748095898251e00),
    ]
    for (z, y), exp in zip(cases, expected):
        got = np.asarray(recfast_rhs(z, jnp.asarray(y), p))
        assert got == pytest.approx(exp, rel=2.0e-8, abs=1.0e-12)


@pytest.mark.parametrize("cosmology", COSMOLOGY_CASES)
def test_initial_thermal_state_uses_saha_helium_when_ode_starts(cosmology):
    """Check that initial conditions switch from full ionization to Saha helium."""

    args = recfast_parameters(cosmology)
    z_early = 4000.0
    z_ode = find_recfast_ode_start(args["T_cmb"], args["Nnow"], args["f_He"])

    assert 1500.0 < z_ode < 3500.0
    assert initial_thermal_state(
        z_early, args["T_cmb"], args["Nnow"], args["f_He"]
    ) == (
        1.0,
        1.0,
        args["T_cmb"] * (1.0 + z_early),
    )
    x_H, x_He, T_mat = initial_thermal_state(
        z_ode,
        args["T_cmb"],
        args["Nnow"],
        args["f_He"],
    )
    assert x_H == pytest.approx(1.0)
    assert x_He < 0.99
    assert T_mat == pytest.approx(args["T_cmb"] * (1.0 + z_ode))


@pytest.mark.parametrize("cosmology", COSMOLOGY_CASES)
def test_recfast_rhs_with_tau_extends_thermal_rhs(cosmology):
    """Check that the augmented RHS preserves thermal derivatives and adds dtau/dz."""

    p = _param_row(cosmology)
    z = 1200.0
    y = (0.9, 1.0e-4, 3270.6)
    thermal = np.asarray(recfast_rhs(z, jnp.asarray(y), p))
    augmented = np.asarray(recfast_rhs_with_tau(z, jnp.asarray((*y, 0.0)), p))

    assert augmented[:3] == pytest.approx(thermal, rel=1.0e-15)
    assert augmented[3] < 0.0
