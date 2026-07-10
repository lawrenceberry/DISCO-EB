
####################################################################################################################
# This file is not licensed under the GNU GPL license!
# Large portions of this code, which is part of the DISCO-EB module have been adapted from the recfast code base.
# The original recfast code can be found at: https://www.astro.ubc.ca/people/scott/recfast.html
# The recfast code base is distributed under the following license:
####################################################################################################################
 # Integrator for Cosmic Recombination of Hydrogen and Helium,
 # developed by Douglas Scott (dscott@astro.ubc.ca)
 # based on calculations in the papers Seager, Sasselov & Scott
 # (ApJ, 523, L1, 1999; ApJS, 128, 407, 2000)
 # and "fudge" updates in Wong, Moss & Scott (2008).
 #
 # Permission to use, copy, modify and distribute without fee or royalty at
 # any tier, this software and its documentation, for any purpose and without
 # fee or royalty is hereby granted, provided that you agree to comply with
 # the following copyright notice and statements, including the disclaimer,
 # and that the same appear on ALL copies of the software and documentation,
 # including modifications that you make for internal use or for distribution:
 #
 # Copyright 1999-2010 by University of British Columbia.  All rights reserved.
####################################################################################################################
# This file contains a JAX-compatible version of the recfast code base, which has been adapted for use in the
# DISCO-EB module by Oliver Hahn. The DISCO-EB module is distributed under the GNU GPL license.
####################################################################################################################

"""Scalar RECFAST thermodynamic equations with a JAX right-hand side.

The physics is ported from the DISCO2 RECFAST prototype, which splits into a
scalar, standard-Python module (``disco2/src/recfast.py``) and a thin
JAX-specific right-hand side (``disco2/src/jax/recfast.py``). This module keeps
the same split:

1. **Scalar primitives** -- setup helpers (number densities, equality redshift),
   the Saha initial conditions, and the individual recombination / escape /
   temperature terms. These are plain Python (operators + :mod:`math`), so they
   are readable, testable, and evaluate identically on Python floats. The purely
   arithmetic ones (``+ - * / **`` only, no :mod:`math` transcendentals) are also
   valid on JAX tracers and are reused by the JAX right-hand side below.

2. **JAX right-hand side** -- :func:`recfast_rhs` mirrors the scalar equations
   but replaces the Python ``if`` branches with ``jnp.where`` and the
   :mod:`math` transcendentals with ``jnp`` ones, so it can be traced, ``jit``-ed,
   ``vmap``-ed, and differentiated for the DISCO-EB pipeline.

All physical constants live in :mod:`discoeb.constants`.
"""

import math

import jax
import jax.numpy as jnp

from .constants import (
    A2P_S,
    A2P_T,
    AGAUSS1,
    AGAUSS2,
    BFACT,
    CB1_HE1,
    CB1_HE2,
    CB1_HE2ST,
    CDB,
    CDB_HE,
    CK,
    CK_HE,
    CL,
    CL_HE,
    CL_HE_2ST,
    CL_PST,
    CR,
    CT,
    C_SI,
    K_B_SI as K_B,
    LAMBDA_2S1S,
    LAMBDA_HE,
    L_HE_2P,
    L_HE_2PT,
    M_H,
    M_HE4_OVER_M_H,
    MPC_IN_M,
    PI,
    RECFAST_FUDGE,
    RHO_CRIT_100_SI,
    SIGMA_HE_2PS,
    SIGMA_HE_2PT,
    SQRT_PI,
    WGAUSS1,
    WGAUSS2,
    ZGAUSS1,
    ZGAUSS2,
)
from .background import (
    _dark_energy_density_ratio_jax,
    dtau_dz,
    get_neutrino_momentum_bins,
    hubble_constant_mpc,
    hubble_z,
)

# Momentum quadrature for the massive-neutrino background density. Massive
# neutrinos are still relativistic at recombination (a*amnu/q << 1), so they
# contribute to the radiation budget that sets H(z) and the equality redshift;
# ignoring them biases H(z) by several percent for a 0.06 eV species. 15 bins
# integrate the Fermi-Dirac background to ~1e-6, and the ODE cost is negligible.
NQMAX_RECFAST = 15
_Q_NU, _W_NU = (jnp.asarray(a, dtype=jnp.float64) for a in get_neutrino_momentum_bins(NQMAX_RECFAST))


def massive_neutrino_density_ratio(a, amnu):
    """Return ``rho_nu(a) / rho_nu,massless`` for the degenerate massive species.

    ``rho = sum_i w_i sqrt(1 + (a amnu / q_i)^2)``, which tends to ``1`` (fully
    relativistic) as ``a -> 0`` and grows like ``a`` once ``a amnu >> q``. Pure
    JAX arithmetic, so it evaluates on floats and on tracers.
    """

    return jnp.sum(_W_NU * jnp.sqrt(1.0 + (a * amnu / _Q_NU) ** 2))


# =============================================================================
# 1. Scalar setup helpers (standard Python)
# =============================================================================


def present_hydrogen_number_density(omega_b_h2: float, Y_He: float) -> float:
    """Return the present-day hydrogen number density in m^-3.

    The baryon mass density is ``rho_b,0 = (Omega_b h^2) rho_crit,100``.
    Multiplying by the hydrogen mass fraction ``1 - Y_He`` and dividing by the
    hydrogen mass gives ``Nnow``, the hydrogen number density used by RECFAST.
    """

    return (1.0 - Y_He) * (omega_b_h2 * RHO_CRIT_100_SI) / M_H


def hubble_constant_si(h: float) -> float:
    """Return ``H0`` in s^-1 from the dimensionless Hubble parameter ``h``."""

    return hubble_constant_mpc(h) * C_SI / MPC_IN_M


def matter_density_fraction(omega_b_h2: float, omega_c_h2: float, h: float) -> float:
    """Return the non-relativistic matter fraction ``Omega_m``."""

    return (omega_b_h2 + omega_c_h2) / h**2


def equality_redshift(
    grhog: float,
    grhornomass: float,
    grhoc: float,
    grhob: float,
    grhomnu: float = 0.0,
) -> float:
    """Return the radiation-matter equality redshift from density coefficients.

    Massive neutrinos, if present, are counted as radiation via ``grhomnu``: they
    are still relativistic at equality, so omitting them would place equality too
    early (e.g. z_eq ~ 3930 instead of ~3405 for the Planck 0.06 eV convention).
    """

    a_eq = (grhog + grhornomass + grhomnu) / (grhoc + grhob)
    return 1.0 / a_eq - 1.0


def hubble_z_si(
    z: float,
    grhog: float,
    grhornomass: float,
    grhoc: float,
    grhob: float,
    grhov: float,
    *,
    grhok: float = 0.0,
    grhomnu: float = 0.0,
    rhonu: float = 1.0,
    rho_de: float = 1.0,
) -> float:
    """Return ``H(z)`` in s^-1 from CAMB-style background coefficients.

    The keyword arguments carry the extensions beyond flat ``LambdaCDM``: spatial
    curvature (``grhok``), the massive-neutrino density (``grhomnu`` scaled by the
    density ratio ``rhonu``), and dynamical dark energy (``rho_de``); see
    :func:`discoeb.background.hubble_z`.
    """

    return (
        hubble_z(
            z, grhog, grhornomass, grhoc, grhob, grhov,
            grhok=grhok, grhomnu=grhomnu, rhonu=rhonu, rho_de=rho_de,
        )
        * C_SI
        / MPC_IN_M
    )


# =============================================================================
# 2. Saha initial conditions (standard Python)
# =============================================================================


def saha_he2(z: float, T_cmb: float, Nnow: float, f_He: float) -> float:
    """Return the Saha-equilibrium abundance for singly ionized helium.

    This is the high-redshift algebraic equilibrium used before the full
    helium recombination ODE becomes active.
    """

    T = T_cmb * (1.0 + z)
    rhs = (CR * T_cmb / (1.0 + z)) ** 1.5 * math.exp(-CB1_HE2 / T) / Nnow
    return 0.5 * (
        math.sqrt((rhs - 1.0 - f_He) ** 2 + 4.0 * (1.0 + 2.0 * f_He) * rhs)
        - (rhs - 1.0 - f_He)
    )


def saha_he1(z: float, T_cmb: float, Nnow: float, f_He: float) -> float:
    """Return the Saha-equilibrium abundance for neutral-helium recombination."""

    T = T_cmb * (1.0 + z)
    rhs = 4.0 * (CR * T_cmb / (1.0 + z)) ** 1.5 * math.exp(-CB1_HE1 / T) / Nnow
    x0 = 0.5 * (math.sqrt((rhs - 1.0) ** 2 + 4.0 * (1.0 + f_He) * rhs) - (rhs - 1.0))
    return min((x0 - 1.0) / f_He, 1.0)


def initial_thermal_state(
    z: float,
    T_cmb: float,
    Nnow: float,
    f_He: float,
) -> tuple[float, float, float]:
    """Return RECFAST initial conditions ``(x_H, x_He, T_mat)`` at redshift ``z``."""

    if z > 3500.0:
        return (1.0, 1.0, T_cmb * (1.0 + z))
    return (1.0, saha_he1(z, T_cmb, Nnow, f_He), T_cmb * (1.0 + z))


def find_recfast_ode_start(
    T_cmb: float,
    Nnow: float,
    f_He: float,
    z_start: float = 10000.0,
) -> float:
    """Return the redshift where RECFAST switches from Saha to ODE evolution."""

    dz = 0.5
    n = int(z_start / dz)
    for i in range(n + 1):
        z = z_start - i * dz
        if 0.0 < z <= 3500.0 and saha_he1(z, T_cmb, Nnow, f_He) < 0.99:
            return z
    return 0.0


# =============================================================================
# 3. Scalar thermodynamic primitives (standard Python)
#
# The purely arithmetic helpers here (``+ - * / **`` only) evaluate identically
# on Python floats and JAX tracers, and are reused verbatim by the JAX
# right-hand side in section 4. Helpers that use :mod:`math` transcendentals or
# Python branches are scalar-only reference implementations.
# =============================================================================


def free_electron_fraction(x_H: float, x_He: float, f_He: float) -> float:
    """Return the total free-electron fraction per hydrogen nucleus."""

    return x_H + f_He * x_He


def radiation_temperature(z: float, T_cmb: float) -> float:
    """Return the photon temperature ``T_rad = T_cmb (1+z)``."""

    return T_cmb * (1.0 + z)


def hydrogen_number_density(z: float, Nnow: float) -> float:
    """Return the physical hydrogen number density ``n_H(z)`` in m^-3."""

    return Nnow * (1.0 + z) ** 3


def hydrogen_recombination_rate(T_mat):
    """Return the RECFAST case-B hydrogen recombination coefficient.

    Pure ``+ - * / **`` arithmetic, so it evaluates on floats and JAX tracers.
    """

    t4 = T_mat / 1.0e4
    return 1.0e-19 * 4.309 * t4 ** (-0.6166) / (1.0 + 0.6703 * t4**0.5300)


def hydrogen_photoionization_rate(T_mat: float, Rdown: float) -> float:
    """Return the hydrogen photoionization coefficient from detailed balance."""

    return Rdown * (CR * T_mat) ** 1.5 * math.exp(-CDB / T_mat)


def hydrogen_escape_factor(z: float, Hz: float) -> float:
    """Return the RECFAST hydrogen Ly-alpha escape correction factor ``K``."""

    log1pz = math.log(1.0 + z)
    correction = (
        1.0
        + AGAUSS1 * math.exp(-(((log1pz - ZGAUSS1) / WGAUSS1) ** 2))
        + AGAUSS2 * math.exp(-(((log1pz - ZGAUSS2) / WGAUSS2) ** 2))
    )
    return CK * correction / Hz


def hydrogen_ionization_derivative(
    z: float,
    x_H: float,
    x_total: float,
    T_mat: float,
    n_H: float,
    Hz: float,
) -> float:
    """Return ``dx_H/dz`` from the RECFAST effective hydrogen atom equation."""

    Rdown = hydrogen_recombination_rate(T_mat)
    Rup = hydrogen_photoionization_rate(T_mat, Rdown)
    K = hydrogen_escape_factor(z, Hz)
    n_1s = n_H * max(1.0 - x_H, 1.0e-30)
    numerator = (
        x_total * x_H * n_H * Rdown - Rup * (1.0 - x_H) * math.exp(-CL / T_mat)
    ) * (1.0 + K * LAMBDA_2S1S * n_1s)
    denominator = (
        Hz
        * (1.0 + z)
        * (
            1.0 / RECFAST_FUDGE
            + K * LAMBDA_2S1S * n_1s / RECFAST_FUDGE
            + K * Rup * n_1s
        )
    )
    return numerator / denominator


def helium_recombination_rate(T_mat):
    """Return the neutral-helium singlet recombination coefficient.

    Pure ``+ - * / **`` arithmetic (``** 0.5`` rather than :func:`math.sqrt`), so
    it evaluates on floats and JAX tracers.
    """

    T_0 = 10.0**0.477121
    T_1 = 10.0**5.114
    sq_0 = (T_mat / T_0) ** 0.5
    sq_1 = (T_mat / T_1) ** 0.5
    return 10.0 ** (-16.744) / (sq_0 * (1.0 + sq_0) ** 0.289 * (1.0 + sq_1) ** 1.711)


def helium_triplet_recombination_rate(T_mat):
    """Return the neutral-helium triplet recombination coefficient.

    Pure arithmetic (see :func:`helium_recombination_rate`).
    """

    T_0 = 10.0**0.477121
    T_1 = 10.0**5.114
    sq_0 = (T_mat / T_0) ** 0.5
    sq_1 = (T_mat / T_1) ** 0.5
    a_trip = 10.0 ** (-16.306)
    b_trip = 0.761
    return a_trip / (
        sq_0 * (1.0 + sq_0) ** (1.0 - b_trip) * (1.0 + sq_1) ** (1.0 + b_trip)
    )


def escape_probability(tau: float) -> float:
    """Return the Sobolev escape probability with a small-tau series fallback."""

    if tau > 1.0e-7:
        return (1.0 - math.exp(-tau)) / tau
    return 1.0 - tau / 2.0


def helium_2p_doppler_width(T_mat, line_wavenumber):
    """Return the thermal Doppler width of a neutral-helium 2P line in s^-1.

    Pure arithmetic, so it evaluates on floats and JAX tracers.
    """

    return (
        C_SI
        * line_wavenumber
        * (2.0 * K_B * T_mat / (M_H * M_HE4_OVER_M_H * C_SI**2)) ** 0.5
    )


def helium_2p_sobolev_gamma(
    a2p,
    cross_section,
    line_wavenumber,
    doppler,
    f_He,
    x_He,
    one_minus_x_H,
):
    """Return the Sobolev parameter ``gamma_2P`` of a neutral-helium 2P branch.

    ``one_minus_x_H`` is supplied pre-clamped by the caller (each backend applies
    its own ``max``), keeping this helper free of branch/clamp operations. Pure
    arithmetic, so it evaluates on floats and JAX tracers.
    """

    return (
        3.0
        * a2p
        * f_He
        * (1.0 - x_He)
        * C_SI**2
        / (
            SQRT_PI
            * cross_section
            * 8.0
            * PI
            * doppler
            * one_minus_x_H
            * (C_SI * line_wavenumber) ** 2
        )
    )


def helium_feedback_kernel_singlet(
    x_H: float,
    x_He: float,
    T_mat: float,
    n_He_ground: float,
    f_He: float,
    Hz: float,
    pHe_s: float,
) -> float:
    """Return the effective helium singlet escape kernel ``K_He``."""

    if x_H < 0.9999999:
        doppler = helium_2p_doppler_width(T_mat, L_HE_2P)
        gamma_2Ps = helium_2p_sobolev_gamma(
            A2P_S, SIGMA_HE_2PS, L_HE_2P, doppler, f_He, x_He, max(1.0 - x_H, 1.0e-30)
        )
        continuum_escape = A2P_S / (1.0 + 0.36 * gamma_2Ps**0.86)
        return 1.0 / max(
            (A2P_S * pHe_s + continuum_escape) * 3.0 * n_He_ground, 1.0e-300
        )
    return 1.0 / max(A2P_S * pHe_s * 3.0 * n_He_ground, 1.0e-300)


def helium_singlet_ionization_derivative(
    z: float,
    x_H: float,
    x_He: float,
    x_total: float,
    T_mat: float,
    n_H: float,
    n_He: float,
    f_He: float,
    Hz: float,
) -> float:
    """Return the singlet contribution to ``dx_He/dz``."""

    Rdown_He = helium_recombination_rate(T_mat)
    Rup_He = 4.0 * Rdown_He * (CR * T_mat) ** 1.5 * math.exp(-CDB_HE / T_mat)
    He_Boltz = math.exp(min(BFACT / T_mat, 500.0))
    n_He_ground = n_He * max(1.0 - x_He, 1.0e-30)
    tauHe_s = A2P_S * CK_HE * 3.0 * n_He_ground / Hz
    pHe_s = escape_probability(tauHe_s)
    K_He = helium_feedback_kernel_singlet(
        x_H, x_He, T_mat, n_He_ground, f_He, Hz, pHe_s
    )
    return (
        (
            x_total * x_He * n_H * Rdown_He
            - Rup_He * (1.0 - x_He) * math.exp(-CL_HE / T_mat)
        )
        * (1.0 + K_He * LAMBDA_HE * n_He_ground * He_Boltz)
        / (
            Hz
            * (1.0 + z)
            * (1.0 + K_He * (LAMBDA_HE + Rup_He) * n_He_ground * He_Boltz)
        )
    )


def helium_triplet_escape_factor(
    x_H: float,
    x_He: float,
    T_mat: float,
    n_He_ground: float,
    f_He: float,
    Hz: float,
) -> float:
    """Return the triplet correction factor used in helium recombination."""

    tauHe_t = A2P_T * n_He_ground * 3.0 / (8.0 * math.pi * Hz * L_HE_2PT**3)
    pHe_t = escape_probability(tauHe_t)
    if x_H < 0.99999:
        doppler = helium_2p_doppler_width(T_mat, L_HE_2PT)
        gamma_2Pt = helium_2p_sobolev_gamma(
            A2P_T, SIGMA_HE_2PT, L_HE_2PT, doppler, f_He, x_He, max(1.0 - x_H, 1.0e-30)
        )
        continuum_escape = A2P_T / (1.0 + 0.66 * gamma_2Pt**0.9) / 3.0
        factor = (A2P_T * pHe_t + continuum_escape) * math.exp(-CL_PST / T_mat)
    else:
        factor = A2P_T * pHe_t * math.exp(-CL_PST / T_mat)
    return factor


def helium_triplet_ionization_derivative(
    z: float,
    x_H: float,
    x_He: float,
    x_total: float,
    T_mat: float,
    n_H: float,
    n_He: float,
    f_He: float,
    Hz: float,
) -> float:
    """Return the triplet contribution to ``dx_He/dz``."""

    if x_He <= 5.0e-9:
        return 0.0
    n_He_ground = n_He * max(1.0 - x_He, 1.0e-30)
    Rdown_trip = helium_triplet_recombination_rate(T_mat)
    Rup_trip = (
        (4.0 / 3.0) * Rdown_trip * (CR * T_mat) ** 1.5 * math.exp(-CB1_HE2ST / T_mat)
    )
    escape_factor = helium_triplet_escape_factor(
        x_H, x_He, T_mat, n_He_ground, f_He, Hz
    )
    denom = Rup_trip + escape_factor
    correction = escape_factor / denom if denom > 1.0e-300 else 0.0
    return (
        (
            x_total * x_He * n_H * Rdown_trip
            - (1.0 - x_He) * 3.0 * Rup_trip * math.exp(-CL_HE_2ST / T_mat)
        )
        * correction
        / (Hz * (1.0 + z))
    )


def helium_ionization_derivative(
    z: float,
    x_H: float,
    x_He: float,
    x_total: float,
    T_mat: float,
    n_H: float,
    f_He: float,
    Hz: float,
) -> float:
    """Return ``dx_He/dz`` from the RECFAST helium equations."""

    if x_He < 1.0e-15:
        return 0.0
    n_He = f_He * n_H
    singlet = helium_singlet_ionization_derivative(
        z, x_H, x_He, x_total, T_mat, n_H, n_He, f_He, Hz
    )
    triplet = helium_triplet_ionization_derivative(
        z, x_H, x_He, x_total, T_mat, n_H, n_He, f_He, Hz
    )
    return singlet + triplet


def compton_cooling_epsilon(Hz, x_total, x_safe, f_He, T_rad):
    """Return the Compton-cooling factor ``epsilon`` of the tight branch.

    Pure arithmetic, so it evaluates on floats and JAX tracers.
    """

    return Hz * (1.0 + x_total + f_He) / (CT * T_rad**3 * x_safe)


def hubble_redshift_derivative(z, Hz, H0_SI, omega_m, z_eq):
    """Return ``dH/dz`` for the matter+radiation background (tight branch).

    Pure arithmetic, so it evaluates on floats and JAX tracers.
    """

    return (
        (H0_SI**2 / (2.0 * Hz))
        * omega_m
        * (4.0 * (1.0 + z) ** 3 / (1.0 + z_eq) + 3.0 * (1.0 + z) ** 2)
    )


def matter_temperature_tight(
    z,
    T_cmb,
    f_He,
    x_total,
    x_safe,
    epsilon,
    dHdz,
    Hz,
    dxH_dz,
    dxHe_dz,
):
    """Return ``dT_mat/dz`` in the tightly Compton-coupled (early) regime.

    Pure arithmetic, so it evaluates on floats and JAX tracers.
    """

    return (
        T_cmb
        + epsilon
        * (1.0 + f_He)
        / (1.0 + f_He + x_total)
        * (dxH_dz + f_He * dxHe_dz)
        / x_safe
        - epsilon * dHdz / Hz
        + 3.0 * epsilon / (1.0 + z)
    )


def matter_temperature_loose(z, T_mat, T_rad, x_total, x_safe, f_He, Hz):
    """Return ``dT_mat/dz`` in the loosely coupled (late) regime.

    Pure arithmetic, so it evaluates on floats and JAX tracers.
    """

    return CT * T_rad**4 * x_safe / (1.0 + x_total + f_He) * (T_mat - T_rad) / (
        Hz * (1.0 + z)
    ) + 2.0 * T_mat / (1.0 + z)


def matter_temperature_derivative(
    z: float,
    x_total: float,
    f_He: float,
    T_mat: float,
    T_rad: float,
    Hz: float,
    H0_SI: float,
    omega_m: float,
    z_eq: float,
    dxH_dz: float,
    dxHe_dz: float,
    T_cmb: float,
) -> float:
    """Return ``dT_mat/dz`` from Compton coupling and adiabatic cooling."""

    x_safe = max(x_total, 1.0e-30)
    timeTh = (1.0 / (CT * T_rad**4)) * (1.0 + x_total + f_He) / x_safe
    timeH = 2.0 / (3.0 * H0_SI * (1.0 + z) ** 1.5)
    if timeTh < 1.0e-3 * timeH:
        epsilon = compton_cooling_epsilon(Hz, x_total, x_safe, f_He, T_rad)
        dHdz = hubble_redshift_derivative(z, Hz, H0_SI, omega_m, z_eq)
        return matter_temperature_tight(
            z, T_cmb, f_He, x_total, x_safe, epsilon, dHdz, Hz, dxH_dz, dxHe_dz
        )
    return matter_temperature_loose(z, T_mat, T_rad, x_total, x_safe, f_He, Hz)


# =============================================================================
# 4. JAX right-hand side
#
# Mirrors the scalar equations above, reusing the pure-arithmetic primitives and
# replacing the Python ``if`` branches with ``jnp.where`` and the :mod:`math`
# transcendentals with ``jnp`` ones. The background/thermodynamic context is
# supplied as a packed parameter row ``p`` in the order
#     p = (T_cmb, f_He, Nnow, H0_SI, omega_m, z_eq,
#          grhog, grhornomass, grhoc, grhob, grhov,
#          grhok, grhomnu, amnu, w_DE_0, w_DE_a).
# The last five entries carry the extensions beyond flat massless LambdaCDM
# (curvature, massive neutrinos, dynamical dark energy) and are all zero /
# LambdaCDM-inert for the baseline model.
# =============================================================================


def _escape_probability_jax(tau):
    """JAX Sobolev escape probability (``jnp.where`` form of :func:`escape_probability`)."""

    return jnp.where(tau > 1.0e-7, (1.0 - jnp.exp(-tau)) / tau, 1.0 - tau / 2.0)


def recfast_rhs(z, y, p):
    """Return the JAX RECFAST RHS ``(dx_H/dz, dx_He/dz, dT_mat/dz)``.

    Mirrors the scalar RECFAST equations, reusing the pure-arithmetic primitives
    above and replacing the Python ``if`` branches with ``jnp.where``.

    Args:
        z: redshift (scalar).
        y: state ``(x_H, x_He, T_mat)``.
        p: packed parameter row ``(T_cmb, f_He, Nnow, H0_SI, omega_m, z_eq,
            grhog, grhornomass, grhoc, grhob, grhov, grhok, grhomnu, amnu,
            w_DE_0, w_DE_a)``.

    Returns:
        jax.Array: ``(dx_H/dz, dx_He/dz, dT_mat/dz)``.
    """

    x_H = jnp.maximum(y[0], 0.0)
    x_He = jnp.maximum(y[1], 0.0)
    T_mat = jnp.maximum(y[2], 0.5)
    T_cmb = p[0]
    f_He = p[1]
    Nnow = p[2]
    H0_SI = p[3]
    omega_m = p[4]
    z_eq = p[5]
    grhog, grhornomass, grhoc, grhob, grhov = p[6], p[7], p[8], p[9], p[10]
    grhok, grhomnu, amnu, w_DE_0, w_DE_a = p[11], p[12], p[13], p[14], p[15]

    a = 1.0 / (1.0 + z)
    rhonu = massive_neutrino_density_ratio(a, amnu)
    rho_de = _dark_energy_density_ratio_jax(a, w_DE_0, w_DE_a)
    Hz = hubble_z_si(
        z, grhog, grhornomass, grhoc, grhob, grhov,
        grhok=grhok, grhomnu=grhomnu, rhonu=rhonu, rho_de=rho_de,
    )
    x_total = x_H + f_He * x_He
    T_rad = T_cmb * (1.0 + z)
    n_H = Nnow * (1.0 + z) ** 3
    one_minus_x_H = jnp.maximum(1.0 - x_H, 1.0e-30)

    # --- Hydrogen ---
    Rdown = hydrogen_recombination_rate(T_mat)
    Rup = Rdown * (CR * T_mat) ** 1.5 * jnp.exp(-CDB / T_mat)
    log1pz = jnp.log(1.0 + z)
    correction = (
        1.0
        + AGAUSS1 * jnp.exp(-(((log1pz - ZGAUSS1) / WGAUSS1) ** 2))
        + AGAUSS2 * jnp.exp(-(((log1pz - ZGAUSS2) / WGAUSS2) ** 2))
    )
    K = CK * correction / Hz
    n_1s = n_H * one_minus_x_H
    dxH_dz = (
        (x_total * x_H * n_H * Rdown - Rup * (1.0 - x_H) * jnp.exp(-CL / T_mat))
        * (1.0 + K * LAMBDA_2S1S * n_1s)
        / (
            Hz
            * (1.0 + z)
            * (
                1.0 / RECFAST_FUDGE
                + K * LAMBDA_2S1S * n_1s / RECFAST_FUDGE
                + K * Rup * n_1s
            )
        )
    )

    # --- Helium singlet ---
    n_He = f_He * n_H
    n_He_ground = n_He * jnp.maximum(1.0 - x_He, 1.0e-30)
    Rdown_He = helium_recombination_rate(T_mat)
    Rup_He = 4.0 * Rdown_He * (CR * T_mat) ** 1.5 * jnp.exp(-CDB_HE / T_mat)
    He_Boltz = jnp.exp(jnp.minimum(BFACT / T_mat, 500.0))
    tauHe_s = A2P_S * CK_HE * 3.0 * n_He_ground / Hz
    pHe_s = _escape_probability_jax(tauHe_s)
    doppler_s = helium_2p_doppler_width(T_mat, L_HE_2P)
    gamma_2Ps = helium_2p_sobolev_gamma(
        A2P_S, SIGMA_HE_2PS, L_HE_2P, doppler_s, f_He, x_He, one_minus_x_H
    )
    continuum_s = A2P_S / (1.0 + 0.36 * gamma_2Ps**0.86)
    K_He_feedback = jnp.where(
        x_H < 0.9999999,
        1.0 / jnp.maximum((A2P_S * pHe_s + continuum_s) * 3.0 * n_He_ground, 1.0e-300),
        1.0 / jnp.maximum(A2P_S * pHe_s * 3.0 * n_He_ground, 1.0e-300),
    )
    singlet = (
        (
            x_total * x_He * n_H * Rdown_He
            - Rup_He * (1.0 - x_He) * jnp.exp(-CL_HE / T_mat)
        )
        * (1.0 + K_He_feedback * LAMBDA_HE * n_He_ground * He_Boltz)
        / (
            Hz
            * (1.0 + z)
            * (1.0 + K_He_feedback * (LAMBDA_HE + Rup_He) * n_He_ground * He_Boltz)
        )
    )

    # --- Helium triplet ---
    Rdown_trip = helium_triplet_recombination_rate(T_mat)
    Rup_trip = (
        (4.0 / 3.0) * Rdown_trip * (CR * T_mat) ** 1.5 * jnp.exp(-CB1_HE2ST / T_mat)
    )
    tauHe_t = A2P_T * n_He_ground * 3.0 / (8.0 * jnp.pi * Hz * L_HE_2PT**3)
    pHe_t = _escape_probability_jax(tauHe_t)
    doppler_t = helium_2p_doppler_width(T_mat, L_HE_2PT)
    gamma_2Pt = helium_2p_sobolev_gamma(
        A2P_T, SIGMA_HE_2PT, L_HE_2PT, doppler_t, f_He, x_He, one_minus_x_H
    )
    continuum_t = A2P_T / (1.0 + 0.66 * gamma_2Pt**0.9) / 3.0
    escape_factor = jnp.where(
        x_H < 0.99999,
        (A2P_T * pHe_t + continuum_t) * jnp.exp(-CL_PST / T_mat),
        A2P_T * pHe_t * jnp.exp(-CL_PST / T_mat),
    )
    denom_trip = Rup_trip + escape_factor
    correction_trip = jnp.where(denom_trip > 1.0e-300, escape_factor / denom_trip, 0.0)
    triplet = (
        (
            x_total * x_He * n_H * Rdown_trip
            - (1.0 - x_He) * 3.0 * Rup_trip * jnp.exp(-CL_HE_2ST / T_mat)
        )
        * correction_trip
        / (Hz * (1.0 + z))
    )
    dxHe_dz = jnp.where(
        x_He < 1.0e-15, 0.0, jnp.where(x_He <= 5.0e-9, singlet, singlet + triplet)
    )

    # --- Matter temperature ---
    x_safe = jnp.maximum(x_total, 1.0e-30)
    timeTh = (1.0 / (CT * T_rad**4)) * (1.0 + x_total + f_He) / x_safe
    timeH = 2.0 / (3.0 * H0_SI * (1.0 + z) ** 1.5)
    epsilon = compton_cooling_epsilon(Hz, x_total, x_safe, f_He, T_rad)
    dHdz = hubble_redshift_derivative(z, Hz, H0_SI, omega_m, z_eq)
    tight = matter_temperature_tight(
        z, T_cmb, f_He, x_total, x_safe, epsilon, dHdz, Hz, dxH_dz, dxHe_dz
    )
    loose = matter_temperature_loose(z, T_mat, T_rad, x_total, x_safe, f_He, Hz)
    dTmat_dz = jnp.where(timeTh < 1.0e-3 * timeH, tight, loose)

    return jnp.asarray((dxH_dz, dxHe_dz, dTmat_dz), dtype=jnp.float64)


def recfast_rhs_with_tau(z, y, p):
    """Return the JAX RECFAST RHS augmented with conformal time ``dtau/dz``.

    Args:
        z: redshift (scalar).
        y: state ``(x_H, x_He, T_mat, tau)`` (``tau`` is unused by the RHS).
        p: packed parameter row (see :func:`recfast_rhs`).

    Returns:
        jax.Array: ``(dx_H/dz, dx_He/dz, dT_mat/dz, dtau/dz)``.
    """

    dxH_dz, dxHe_dz, dTmat_dz = recfast_rhs(z, y, p)
    grhog, grhornomass, grhoc, grhob, grhov = p[6], p[7], p[8], p[9], p[10]
    grhok, grhomnu, amnu, w_DE_0, w_DE_a = p[11], p[12], p[13], p[14], p[15]
    a = 1.0 / (1.0 + z)
    dtau = dtau_dz(
        z, grhog, grhornomass, grhoc, grhob, grhov,
        grhok=grhok, grhomnu=grhomnu,
        rhonu=massive_neutrino_density_ratio(a, amnu),
        rho_de=_dark_energy_density_ratio_jax(a, w_DE_0, w_DE_a),
    )
    return jnp.asarray((dxH_dz, dxHe_dz, dTmat_dz, dtau), dtype=jnp.float64)


def recfast_rhs_eval_sum_jax(z, states, params, row_index):
    """Return the summed RECFAST RHS over a batch (mirrors the DISCO2 helper)."""

    selected_params = params[row_index]
    dydz = jax.vmap(recfast_rhs)(z, states, selected_params)
    return jnp.sum(dydz)
