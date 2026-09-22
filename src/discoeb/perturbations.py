import math
import os
from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy import integrate, interpolate

from .background import (
    batch_dimensions,
    dtauda,
    evolve_background,
    get_neutrino_momentum_bins,
    setup_background_evolution,
)
from .eb_sparsity import perturbation_sparsity
from .thermodynamics_recfast import evaluate_thermo



# Hierarchy truncations. These are module-level integers (not runtime arguments)
# so the state-vector layout and the accelerator-compiled right-hand side treat
# them as compile-time constants. They are launch settings: each reads its
# ``DISCO_LMAX_*`` environment variable at import, and a caller may also rebind
# it before the first solve, since the GPU pipeline builds its layout from them
# at solve time (:func:`_layout_for`). At the defaults the flat-LambdaCDM +
# massless-neutrino layout has NVAR == 50: an 11-variable densely coupled core
# bordered by three 13-variable free-streaming hierarchies, which is the
# structure :mod:`discoeb.eb_sparsity` declares and the sparse direct solver
# exploits.
LMAX_G = int(os.environ.get("DISCO_LMAX_G", "15"))
"""Photon temperature hierarchy truncation (``Theta_0 ... Theta_LMAX_G``)."""

LMAX_POL = int(os.environ.get("DISCO_LMAX_POL", "15"))
"""Photon E-mode polarization hierarchy truncation (``E_2 ... E_LMAX_POL``)."""

LMAX_NR = int(os.environ.get("DISCO_LMAX_NR", "15"))
"""Massless-neutrino hierarchy truncation (``N_0 ... N_LMAX_NR``)."""

LMAX_NU = int(os.environ.get("DISCO_LMAX_NU", "15"))
"""Massive-neutrino hierarchy truncation (``psi_0 ... psi_LMAX_NU`` per momentum bin)."""

IX_ETAK = 0
"""State-vector index of the metric perturbation ``etak = k eta``."""

IX_CLXC = 1
"""State-vector index of the CDM density contrast ``clxc``."""

IX_CLXB = 2
"""State-vector index of the baryon density contrast ``clxb``."""

IX_VB = 3
"""State-vector index of the baryon velocity ``vb``."""

IX_G = 4
"""Base index of the photon temperature hierarchy (``Theta_l`` at ``IX_G + l``)."""

IX_POL = IX_G + LMAX_G + 1
"""Base index of the photon polarization hierarchy (``E_l`` at ``IX_POL + (l-2)``)."""

IX_R = IX_POL + LMAX_POL - 1
"""Base index of the massless-neutrino hierarchy (``N_l`` at ``IX_R + l``)."""

NVAR = IX_R + LMAX_NR + 1
"""Total length of the perturbation state vector."""


# ============================================================
# EINSTEIN CONSTRAINTS AND BACKGROUND TERMS
# ============================================================


def comoving_densities(
    a: float,
    grhog: float,
    grhornomass: float,
    grhoc: float,
    grhob: float,
    grhov: float,
) -> tuple[float, float, float, float, float]:
    """Return the ``8*pi*G*rho_i a^2`` density coefficients at scale factor ``a``.

    The CAMB ``grho`` coefficients satisfy ``grhoa4 = 8*pi*G*rho_i a^4``. Dividing
    by ``a^2`` gives the combinations that appear in the synchronous-gauge
    Einstein equations:

    ``rho_gamma, rho_nu ~ a^-4`` give ``grho_i a^2 = grho_i / a^2``;
    ``rho_c, rho_b ~ a^-3`` give ``grho_i a^2 = grho_i / a``;
    ``rho_Lambda ~ a^0`` gives ``grhov a^2 = grhov * a^2``.

    Returns ``(grhog_t, grhor_t, grhoc_t, grhob_t, grhov_t)``.
    """

    a2 = a * a
    return (grhog / a2, grhornomass / a2, grhoc / a, grhob / a, grhov * a2)


def expansion_rate(
    grhog_t: float,
    grhor_t: float,
    grhoc_t: float,
    grhob_t: float,
    grhov_t: float,
) -> float:
    """Return the conformal Hubble rate ``adotoa = a'/a = aH``.

    ``adotoa = sqrt((Sum grho_i a^2) / 3)``. Written as ``** 0.5`` so the helper
    traces unchanged on Python floats and CUDA device values.
    """

    return ((grhog_t + grhor_t + grhoc_t + grhob_t + grhov_t) / 3.0) ** 0.5


def density_perturbation(
    grhob_t: float,
    clxb: float,
    grhoc_t: float,
    clxc: float,
    grhog_t: float,
    clxg: float,
    grhor_t: float,
    clxr: float,
) -> float:
    """Return the total density perturbation ``dgrho = Sum grho_i a^2 delta_i``."""

    return grhob_t * clxb + grhoc_t * clxc + grhog_t * clxg + grhor_t * clxr


def momentum_perturbation(
    grhob_t: float,
    vb: float,
    grhog_t: float,
    qg: float,
    grhor_t: float,
    qr: float,
) -> float:
    """Return the total momentum-density source ``dgq = Sum grho_i a^2 (rho+p) v_i``."""

    return grhob_t * vb + grhog_t * qg + grhor_t * qr


def anisotropic_stress(
    grhog_t: float,
    pig: float,
    grhor_t: float,
    pir: float,
) -> float:
    """Return the total anisotropic-stress source ``dgpi``."""

    return grhog_t * pig + grhor_t * pir


def metric_z(dgrho: float, etak: float, adotoa: float, k: float) -> float:
    """Return the synchronous-gauge metric variable ``z = (dgrho/(2k) + k eta)/aH``."""

    return (0.5 * dgrho / k + etak) / adotoa


def shear_sigma(z: float, dgq: float, k: float) -> float:
    """Return the metric shear ``sigma = z + (3/2) dgq / k^2``."""

    return z + 1.5 * dgq / (k * k)


def newtonian_potential(
    dgrho: float,
    dgq: float,
    dgpi: float,
    adotoa: float,
    k: float,
) -> float:
    """Return the Newtonian-gauge curvature potential ``Phi`` (diagnostic).

    ``Phi = -[(dgrho + 3 aH dgq / k) + dgpi] / (2 k^2)``.
    """

    return -((dgrho + 3.0 * dgq * adotoa / k) + dgpi) / (2.0 * k * k)


# ============================================================
# ADIABATIC INITIAL CONDITIONS
# Deep in radiation domination (k tau << 1), following CAMB's initial().
# ============================================================


def neutrino_radiation_fraction(grhog: float, grhornomass: float) -> float:
    """Return the neutrino fraction of the radiation density ``R_nu``."""

    return grhornomass / (grhog + grhornomass)


def matter_radiation_parameter(
    grhob: float,
    grhoc: float,
    grhog: float,
    grhornomass: float,
) -> float:
    """Return the matter-to-radiation initial-condition parameter ``om``."""

    return (grhob + grhoc) / (3.0 * (grhog + grhornomass)) ** 0.5


def adiabatic_photon_density(k: float, tau: float, om: float) -> float:
    """Return the initial photon density contrast ``delta_gamma``."""

    x2 = (k * tau) ** 2
    return x2 / 3.0 * (1.0 - om * tau / 5.0)


def adiabatic_photon_velocity(k: float, tau: float, om: float) -> float:
    """Return the initial photon dipole ``q_gamma``."""

    x = k * tau
    return x**3 / 27.0 * (1.0 - om * tau / 5.0)


def adiabatic_initial_conditions(
    k: float,
    tau: float,
    grhog: float,
    grhornomass: float,
    grhoc: float,
    grhob: float,
) -> tuple[float, ...]:
    """Return the adiabatic initial state vector deep in radiation domination.

    Implements CAMB's ``initial()`` series for the adiabatic growing mode at
    ``k tau << 1``, normalized so the comoving curvature perturbation is unity.
    The returned tuple has length :data:`NVAR`; all multipoles above those set
    explicitly start at zero.
    """

    x = k * tau
    x2 = x * x

    Rv = neutrino_radiation_fraction(grhog, grhornomass)
    Rp15 = 4.0 * Rv + 15.0
    om = matter_radiation_parameter(grhob, grhoc, grhog, grhornomass)
    omtau = om * tau

    clxg = adiabatic_photon_density(k, tau, om)
    qg = adiabatic_photon_velocity(k, tau, om)

    y0 = [0.0] * NVAR

    # Metric perturbation: etak = k eta ~ -k at leading order.
    y0[IX_ETAK] = -k * (1.0 - x2 / 12.0 * (-10.0 / Rp15 + 1.0))

    # Photon monopole and dipole.
    y0[IX_G] = clxg
    y0[IX_G + 1] = qg

    # CDM and baryons share the photon density (3/4 factor for adiabatic mode).
    y0[IX_CLXC] = 0.75 * clxg
    y0[IX_CLXB] = 0.75 * clxg
    y0[IX_VB] = 0.75 * qg

    # Massless neutrinos.
    y0[IX_R] = clxg
    y0[IX_R + 1] = (4.0 * Rv + 23.0) / Rp15 * x2 * x / 27.0
    y0[IX_R + 2] = (
        -4.0
        / 3.0
        * x2
        / Rp15
        * (1.0 + omtau / 4.0 * (4.0 * Rv - 5.0) / (2.0 * Rv + 15.0))
    )
    if LMAX_NR >= 3:
        y0[IX_R + 3] = -4.0 / 21.0 / Rp15 * x2 * x

    return tuple(y0)


# ============================================================
# FLUID AND METRIC DERIVATIVES
# ============================================================


def metric_etak_derivative(dgq: float) -> float:
    """Return ``etak' = (1/2) dgq`` from the momentum Einstein constraint."""

    return 0.5 * dgq


def cdm_density_derivative(z: float, k: float) -> float:
    """Return ``clxc' = -k z`` (CDM is pressureless, at rest in synchronous gauge)."""

    return -k * z


def baryon_density_derivative(z: float, vb: float, k: float) -> float:
    """Return ``clxb' = -k (z + v_b)`` from baryon number conservation."""

    return -k * (z + vb)


def baryon_velocity_derivative(
    adotoa: float,
    vb: float,
    k: float,
    delta_p_b: float,
    photbar: float,
    opacity: float,
    qg: float,
) -> float:
    """Return ``v_b'`` from the full (non-tight-coupling) baryon Euler equation.

    ``v_b' = -aH v_b + k delta_p_b - (rho_gamma/rho_b) kappa' (4/3 v_b - q_gamma)``.
    """

    return -adotoa * vb + k * delta_p_b - photbar * opacity * (4.0 / 3.0 * vb - qg)


def photon_monopole_derivative(z: float, qg: float, k: float) -> float:
    """Return ``Theta_0' = -k (4/3 z + q_gamma)`` (photon continuity)."""

    return -k * (4.0 / 3.0 * z + qg)


def photon_dipole_derivative(
    vbdot: float,
    adotoa: float,
    vb: float,
    k: float,
    delta_p_b: float,
    pb43: float,
    clxg: float,
    pig: float,
) -> float:
    """Return the photon dipole derivative ``Theta_1' = q_gamma'``.

    ``q_gamma' = (4/3)(-v_b' - aH v_b + k delta_p_b)/R_gamma + (k/3) delta_gamma
    - (2k/3) pi_gamma``, with ``pb43 = R_gamma = (4/3) rho_gamma/rho_b``.
    """

    return (
        4.0 / 3.0 * (-vbdot - adotoa * vb + k * delta_p_b) / pb43
        + k / 3.0 * clxg
        - 2.0 * k / 3.0 * pig
    )


def polarization_source(pig: float, e2: float) -> float:
    """Return the polarization source ``Pi = pi_gamma/10 + (9/15) E_2``."""

    return pig / 10.0 + 9.0 / 15.0 * e2


def photon_quadrupole_derivative(
    qg: float,
    theta3: float,
    pig: float,
    polter: float,
    sigma: float,
    k: float,
    opacity: float,
) -> float:
    """Return ``Theta_2' = (2k/5) q - (3k/5) Theta_3 - kappa'(pi - Pi) + (8/15) k sigma``."""

    return (
        2.0 * k / 5.0 * qg
        - 3.0 * k / 5.0 * theta3
        - opacity * (pig - polter)
        + 8.0 / 15.0 * k * sigma
    )


def photon_temperature_multipole_derivative(
    l: int,
    theta_lm1: float,
    theta_l: float,
    theta_lp1: float,
    k: float,
    opacity: float,
) -> float:
    """Return ``Theta_l'`` for an interior multipole ``3 <= l < LMAX_G``."""

    return (
        k * l / (2 * l + 1) * theta_lm1
        - k * (l + 1) / (2 * l + 1) * theta_lp1
        - opacity * theta_l
    )


def photon_temperature_truncation(
    theta_lm1: float,
    theta_l: float,
    k: float,
    opacity: float,
    tau: float,
    lmax: int,
) -> float:
    """Return the truncated photon temperature derivative at ``l = LMAX_G``."""

    return k * theta_lm1 - (lmax + 1) / tau * theta_l - opacity * theta_l


def polarization_quadrupole_derivative(
    e2: float,
    e3: float,
    polter: float,
    k: float,
    opacity: float,
) -> float:
    """Return ``E_2' = -kappa'(E_2 - Pi) - (k/3) E_3``."""

    return -opacity * (e2 - polter) - k / 3.0 * e3


def polarization_multipole_derivative(
    l: int,
    e_lm1: float,
    e_l: float,
    e_lp1: float,
    k: float,
    opacity: float,
) -> float:
    """Return ``E_l'`` for an interior multipole ``3 <= l < LMAX_POL``."""

    polfac = (l + 3) * (l - 1) / (l + 1)
    return (
        -opacity * e_l + k * l / (2 * l + 1) * e_lm1 - polfac * k / (2 * l + 1) * e_lp1
    )


def polarization_truncation(
    e_lm1: float,
    e_l: float,
    k: float,
    opacity: float,
    tau: float,
    lmax: int,
) -> float:
    """Return the truncated E-mode derivative at ``l = LMAX_POL``."""

    return -opacity * e_l + k * lmax / (2 * lmax + 1) * e_lm1 - (lmax + 3) / tau * e_l


def neutrino_monopole_derivative(z: float, qr: float, k: float) -> float:
    """Return ``N_0' = -k (4/3 z + q_nu)`` (neutrino continuity)."""

    return -k * (4.0 / 3.0 * z + qr)


def neutrino_dipole_derivative(clxr: float, pir: float, k: float) -> float:
    """Return ``N_1' = (k/3)(delta_nu - 2 pi_nu)`` (neutrino momentum)."""

    return k / 3.0 * (clxr - 2.0 * pir)


def neutrino_quadrupole_derivative(
    qr: float,
    n3: float,
    sigma: float,
    k: float,
) -> float:
    """Return ``N_2' = (2k/5) q_nu - (3k/5) N_3 + (8/15) k sigma``."""

    return 2.0 * k / 5.0 * qr - 3.0 * k / 5.0 * n3 + 8.0 / 15.0 * k * sigma


def neutrino_multipole_derivative(
    l: int,
    n_lm1: float,
    n_l: float,
    n_lp1: float,
    k: float,
) -> float:
    """Return ``N_l'`` for an interior multipole ``3 <= l < LMAX_NR``."""

    return k * l / (2 * l + 1) * n_lm1 - k * (l + 1) / (2 * l + 1) * n_lp1


def neutrino_truncation(
    n_lm1: float,
    n_l: float,
    k: float,
    tau: float,
    lmax: int,
) -> float:
    """Return the truncated neutrino derivative ``N_L' = k N_{L-1} - (L+1) N_L / tau``."""

    return k * n_lm1 - (lmax + 1) / tau * n_l


# ============================================================
# DYNAMICAL DARK ENERGY (CPL fluid perturbations)
#
# Optional extension (enabled when w_0 != -1 or w_a != 0). The DE fluid is a
# clustering fluid with rest-frame sound speed cs2_DE, carrying two extra state
# variables clxq (density contrast) and thetaq (velocity divergence) that join
# the densely-coupled metric core. Synchronous-gauge equations follow Ballesteros
# & Lesgourgues 2010; the metric enters through ``0.5 h' = k z`` (see
# :func:`cdm_density_derivative`, where ``clxc' = -k z = -h'/2``).
# ============================================================


def dark_energy_equation_of_state(a: float, w_DE_0: float, w_DE_a: float) -> float:
    """Return the CPL equation of state ``w_Q(a) = w_0 + w_a (1 - a)``."""

    return w_DE_0 + w_DE_a * (1.0 - a)


def dark_energy_adiabatic_sound_speed(
    w_Q: float, w_Q_prime: float, adotoa: float
) -> float:
    """Return the adiabatic sound speed ``c_a^2 = w - w'/(3(1+w) aH)``."""

    return w_Q - w_Q_prime / (3.0 * (1.0 + w_Q) * adotoa)


def dark_energy_density_coefficient(grhov: float, rho_Q: float, a: float) -> float:
    """Return the comoving DE density coefficient ``8*pi*G*rho_Q a^2``.

    ``grhov = grhom * OmegaDE`` is the present DE density coefficient and
    ``rho_Q = rho_Q(a)/rho_Q(1)`` its normalized evolution, so the comoving
    coefficient is ``grhov * rho_Q * a^2`` (reducing to the flat ``grhov a^2``
    when ``rho_Q = 1``).
    """

    return grhov * rho_Q * a * a


def dark_energy_fluid_derivatives(
    clxq: float,
    thetaq: float,
    z: float,
    adotoa: float,
    k: float,
    a: float,
    w_DE_0: float,
    w_DE_a: float,
    cs2_Q: float,
) -> tuple[float, float]:
    """Return the DE fluid derivatives ``(clxq', thetaq')`` in synchronous gauge.

    ``clxq' = -(1+w)(thetaq + k z) - 3(cs2 - w) aH clxq
              - 9(1+w)(cs2 - c_a^2)(aH)^2 thetaq / k^2``,
    ``thetaq' = -(1 - 3 cs2) aH thetaq + cs2 k^2 clxq / (1+w)``,

    with ``w = w(a)``, ``w' = -w_a aH a``, and ``c_a^2`` the adiabatic sound
    speed. Valid only for ``w != -1`` (a cosmological constant carries no fluid
    perturbations and is excluded from the state layout).
    """

    w_Q = dark_energy_equation_of_state(a, w_DE_0, w_DE_a)
    w_Q_prime = -w_DE_a * adotoa * a
    ca2_Q = dark_energy_adiabatic_sound_speed(w_Q, w_Q_prime, adotoa)

    clxq_prime = (
        -(1.0 + w_Q) * (thetaq + k * z)
        - 3.0 * (cs2_Q - w_Q) * adotoa * clxq
        - 9.0 * (1.0 + w_Q) * (cs2_Q - ca2_Q) * adotoa**2 / k**2 * thetaq
    )
    thetaq_prime = (
        -(1.0 - 3.0 * cs2_Q) * adotoa * thetaq + cs2_Q / (1.0 + w_Q) * k**2 * clxq
    )
    return clxq_prime, thetaq_prime


# ============================================================
# FULL RIGHT-HAND SIDE
# ============================================================


def boltzmann_rhs(
    tau: float,
    y,
    k: float,
    a: float,
    opacity: float,
    cs2_b: float,
    grhog: float,
    grhornomass: float,
    grhoc: float,
    grhob: float,
    grhov: float,
) -> tuple[float, ...]:
    """Return the Einstein-Boltzmann right-hand side ``dy/dtau``.

    Assembles the synchronous-gauge hierarchy from the scalar helpers above. The
    background scale factor ``a``, Thomson opacity ``kappa' = opacity``, and
    baryon sound speed ``cs2_b`` are supplied as scalars evaluated at ``tau`` by
    the caller (typically from precomputed splines). The full photon,
    polarization, and neutrino hierarchies are integrated at all times; the
    early-time photon-baryon scattering enters through the exact Thomson drag
    terms, so a stiff ODE solver handles the high-opacity regime directly without
    a tight-coupling approximation.

    ``y`` is indexed positionally; the returned tuple has length :data:`NVAR`.
    """

    grhog_t, grhor_t, grhoc_t, grhob_t, grhov_t = comoving_densities(
        a, grhog, grhornomass, grhoc, grhob, grhov
    )
    adotoa = expansion_rate(grhog_t, grhor_t, grhoc_t, grhob_t, grhov_t)

    etak = y[IX_ETAK]
    clxc = y[IX_CLXC]
    clxb = y[IX_CLXB]
    vb = y[IX_VB]
    clxg = y[IX_G]
    qg = y[IX_G + 1]
    pig = y[IX_G + 2]
    clxr = y[IX_R]
    qr = y[IX_R + 1]
    pir = y[IX_R + 2]
    e2 = y[IX_POL] if LMAX_POL >= 2 else 0.0

    dgrho = density_perturbation(
        grhob_t, clxb, grhoc_t, clxc, grhog_t, clxg, grhor_t, clxr
    )
    dgq = momentum_perturbation(grhob_t, vb, grhog_t, qg, grhor_t, qr)
    z = metric_z(dgrho, etak, adotoa, k)
    sigma = shear_sigma(z, dgq, k)

    photbar = grhog_t / grhob_t
    pb43 = 4.0 / 3.0 * photbar
    delta_p_b = cs2_b * clxb

    dy = [0.0] * NVAR

    # Metric, CDM, baryon density.
    dy[IX_ETAK] = metric_etak_derivative(dgq)
    dy[IX_CLXC] = cdm_density_derivative(z, k)
    dy[IX_CLXB] = baryon_density_derivative(z, vb, k)

    polter = polarization_source(pig, e2)
    vbdot = baryon_velocity_derivative(adotoa, vb, k, delta_p_b, photbar, opacity, qg)
    dy[IX_VB] = vbdot
    dy[IX_G] = photon_monopole_derivative(z, qg, k)
    dy[IX_G + 1] = photon_dipole_derivative(
        vbdot, adotoa, vb, k, delta_p_b, pb43, clxg, pig
    )

    theta3 = y[IX_G + 3] if LMAX_G >= 3 else 0.0
    dy[IX_G + 2] = photon_quadrupole_derivative(
        qg, theta3, pig, polter, sigma, k, opacity
    )
    for l in range(3, LMAX_G):
        dy[IX_G + l] = photon_temperature_multipole_derivative(
            l, y[IX_G + l - 1], y[IX_G + l], y[IX_G + l + 1], k, opacity
        )
    dy[IX_G + LMAX_G] = photon_temperature_truncation(
        y[IX_G + LMAX_G - 1], y[IX_G + LMAX_G], k, opacity, tau, LMAX_G
    )

    # Photon polarization.
    e3 = y[IX_POL + 1] if LMAX_POL >= 3 else 0.0
    dy[IX_POL] = polarization_quadrupole_derivative(e2, e3, polter, k, opacity)
    for l in range(3, LMAX_POL):
        idx = IX_POL + l - 2
        dy[idx] = polarization_multipole_derivative(
            l, y[idx - 1], y[idx], y[idx + 1], k, opacity
        )
    idx_last = IX_POL + LMAX_POL - 2
    dy[idx_last] = polarization_truncation(
        y[idx_last - 1], y[idx_last], k, opacity, tau, LMAX_POL
    )

    # Massless neutrinos (collisionless).
    dy[IX_R] = neutrino_monopole_derivative(z, qr, k)
    dy[IX_R + 1] = neutrino_dipole_derivative(clxr, pir, k)
    n3 = y[IX_R + 3] if LMAX_NR >= 3 else 0.0
    dy[IX_R + 2] = neutrino_quadrupole_derivative(qr, n3, sigma, k)
    for l in range(3, LMAX_NR):
        dy[IX_R + l] = neutrino_multipole_derivative(
            l, y[IX_R + l - 1], y[IX_R + l], y[IX_R + l + 1], k
        )
    dy[IX_R + LMAX_NR] = neutrino_truncation(
        y[IX_R + LMAX_NR - 1], y[IX_R + LMAX_NR], k, tau, LMAX_NR
    )

    return tuple(dy)
def total_matter_density_contrast(
    grhoc_t: float,
    clxc: float,
    grhob_t: float,
    clxb: float,
) -> float:
    """Return the total-matter density contrast ``delta_m``.

    Density-weighted combination of the CDM and baryon contrasts,

    ``delta_m = (rho_c delta_c + rho_b delta_b) / (rho_c + rho_b)``,

    using the comoving density coefficients (the ``a`` factors cancel).
    """

    return (grhoc_t * clxc + grhob_t * clxb) / (grhoc_t + grhob_t)


def primordial_curvature_power(
    k: float,
    A_s: float,
    n_s: float,
    k_pivot: float,
) -> float:
    """Return the dimensionless primordial curvature power ``P_R(k)``.

    Power-law spectrum ``P_R(k) = A_s (k / k_pivot)^(n_s - 1)``.
    """

    return A_s * (k / k_pivot) ** (n_s - 1.0)


def matter_power_spectrum(
    k: float,
    delta_m: float,
    A_s: float,
    n_s: float,
    k_pivot: float,
) -> float:
    """Return the linear matter power spectrum ``P(k)`` in Mpc^3.

    With initial conditions normalized to unit comoving curvature, the evolved
    total-matter contrast ``delta_m(k)`` acts as a transfer function and

    ``P(k) = (2 pi^2 / k^3) P_R(k) delta_m(k)^2``,

    where ``P_R(k)`` is the dimensionless primordial curvature power. The contrast
    ``delta_m`` must be evaluated at the output time (e.g. ``a = 1``).
    """

    return (
        2.0
        * math.pi**2
        / k**3
        * primordial_curvature_power(k, A_s, n_s, k_pivot)
        * delta_m**2
    )
@dataclass(frozen=True)
class PerturbationLayout:
    """Enabled components and state-vector indices of the perturbation hierarchy.

    Attributes
    ----------
    lmaxg, lmaxpol, lmaxr : int
        Photon-temperature, E-mode-polarization, and massless-neutrino hierarchy
        truncations.
    nqmax : int
        Number of massive-neutrino momentum bins. ``0`` disables massive
        neutrinos entirely (no state variables allocated).
    lmaxnu : int
        Massive-neutrino multipole-hierarchy truncation (used only when
        ``nqmax > 0``).
    enable_dark_energy : bool
        Whether the dynamical dark-energy fluid perturbations are evolved.
    """

    lmaxg: int = 15
    lmaxpol: int = 15
    lmaxr: int = 15
    nqmax: int = 0
    lmaxnu: int = 15
    enable_dark_energy: bool = False

    def __post_init__(self):
        if self.lmaxg < 2 or self.lmaxpol < 2 or self.lmaxr < 2:
            raise ValueError("lmaxg, lmaxpol, lmaxr must each be >= 2")
        if self.nqmax < 0:
            raise ValueError("nqmax must be non-negative")
        if self.nqmax > 0 and self.lmaxnu < 2:
            raise ValueError("lmaxnu must be >= 2 when massive neutrinos are enabled")

    # --- Core (always present) -------------------------------------------------

    @property
    def ix_etak(self) -> int:
        """Index of the synchronous-gauge metric perturbation ``etak = k eta``."""
        return 0

    @property
    def ix_clxc(self) -> int:
        """Index of the CDM density contrast."""
        return 1

    @property
    def ix_clxb(self) -> int:
        """Index of the baryon density contrast."""
        return 2

    @property
    def ix_vb(self) -> int:
        """Index of the baryon velocity."""
        return 3

    @property
    def ix_g(self) -> int:
        """Base index of the photon temperature hierarchy (``Theta_l`` at ix_g + l)."""
        return 4

    @property
    def ix_pol(self) -> int:
        """Base index of the polarization hierarchy (``E_l`` at ix_pol + (l - 2))."""
        return self.ix_g + self.lmaxg + 1

    @property
    def ix_r(self) -> int:
        """Base index of the massless-neutrino hierarchy (``N_l`` at ix_r + l)."""
        return self.ix_pol + self.lmaxpol - 1

    @property
    def _ix_after_massless(self) -> int:
        """First index after the always-present core hierarchies."""
        return self.ix_r + self.lmaxr + 1

    # --- Optional: dynamical dark energy --------------------------------------

    @property
    def has_dark_energy(self) -> bool:
        return self.enable_dark_energy

    @property
    def ix_clxq(self) -> int:
        """Index of the dark-energy density contrast (only if dark energy enabled)."""
        if not self.enable_dark_energy:
            raise AttributeError("dark energy is not enabled in this layout")
        return self._ix_after_massless

    @property
    def ix_thetaq(self) -> int:
        """Index of the dark-energy velocity (only if dark energy enabled)."""
        if not self.enable_dark_energy:
            raise AttributeError("dark energy is not enabled in this layout")
        return self._ix_after_massless + 1

    @property
    def _ix_after_dark_energy(self) -> int:
        return self._ix_after_massless + (2 if self.enable_dark_energy else 0)

    # --- Optional: massive neutrinos ------------------------------------------

    @property
    def has_massive_neutrinos(self) -> bool:
        return self.nqmax > 0

    @property
    def ix_massive_nu(self) -> int:
        """Base index of the massive-neutrino momentum-bin hierarchies.

        The multipole ``psi_l`` of momentum bin ``q`` is stored at
        ``ix_massive_nu + q * (lmaxnu + 1) + l`` (**bin-major, multipole-minor**),
        so that each bin's free-streaming tail ``psi_3 ... psi_lmaxnu`` is
        contiguous and is one tridiagonal block of the sparsity pattern
        (:mod:`discoeb.eb_sparsity`).
        """
        if self.nqmax == 0:
            raise AttributeError("massive neutrinos are not enabled in this layout")
        return self._ix_after_dark_energy

    def ix_psi_base(self, q: int) -> int:
        """Return the state index of ``psi_0`` for momentum bin ``q``."""
        if self.nqmax == 0:
            raise AttributeError("massive neutrinos are not enabled in this layout")
        if not (0 <= q < self.nqmax):
            raise IndexError(f"momentum bin q={q} out of range [0, {self.nqmax})")
        return self.ix_massive_nu + q * (self.lmaxnu + 1)

    def ix_psi(self, l: int, q: int) -> int:
        """Return the state index of massive-neutrino multipole ``psi_l`` of bin ``q``."""
        if not (0 <= l <= self.lmaxnu):
            raise IndexError(f"multipole l={l} out of range [0, {self.lmaxnu}]")
        return self.ix_psi_base(q) + l

    # --- Total size -----------------------------------------------------------

    @property
    def nvar(self) -> int:
        """Total length of the trimmed perturbation state vector."""
        n = self._ix_after_dark_energy
        if self.nqmax > 0:
            n += self.nqmax * (self.lmaxnu + 1)
        return n

    @classmethod
    def from_cosmology(
        cls,
        cosmology,
        *,
        lmaxg: int = 15,
        lmaxpol: int = 15,
        lmaxr: int = 15,
        lmaxnu: int = 15,
        nqmax: int = 5,
    ) -> "PerturbationLayout":
        """Return the trimmed layout implied by a cosmology's enabled components.

        Dynamical dark energy is enabled when ``w_DE_0 != -1`` or ``w_DE_a != 0``;
        massive neutrinos are enabled (with ``nqmax`` momentum bins) when
        ``num_massive_neutrinos > 0``. Curvature does not affect the layout.
        """

        enable_de = (cosmology.w_DE_0 != -1.0) or (cosmology.w_DE_a != 0.0)
        nqmax_massive = nqmax if cosmology.num_massive_neutrinos > 0 else 0
        return cls(
            lmaxg=lmaxg,
            lmaxpol=lmaxpol,
            lmaxr=lmaxr,
            nqmax=nqmax_massive,
            lmaxnu=lmaxnu,
            enable_dark_energy=enable_de,
        )


FLAT_MASSLESS_LAYOUT = PerturbationLayout()
"""Default flat-LambdaCDM + massless-neutrino layout (NVAR == 50).

Reproduces the fixed index constants of :mod:`discoeb.perturbations`.
"""
class _ParamCosmology(NamedTuple):
    """Hashable view of the historical DISCO-EB parameter dictionary."""

    Omegam: float
    Omegab: float
    w_DE_0: float
    w_DE_a: float
    cs2_DE: float
    Omegak: float
    A_s: float
    n_s: float
    H0: float
    T_cmb: float
    Y_He: float
    Neff_massless: float
    num_massive_neutrinos: float
    mnu: float
    k_pivot: float

    @property
    def h(self):
        return self.H0 / 100.0

    @property
    def Omegac(self):
        return self.Omegam - self.Omegab

    @property
    def omega_b_h2(self):
        return self.Omegab * self.h**2

    @property
    def omega_c_h2(self):
        return self.Omegac * self.h**2

    @property
    def tau_reion(self):
        return 0.0

    @property
    def N_eff(self):
        return self.Neff_massless + self.num_massive_neutrinos

    def to_background_params(self):
        return {
            "Omegam": self.Omegam,
            "Omegab": self.Omegab,
            "w_DE_0": self.w_DE_0,
            "w_DE_a": self.w_DE_a,
            "cs2_DE": self.cs2_DE,
            "Omegak": self.Omegak,
            "A_s": self.A_s,
            "n_s": self.n_s,
            "H0": self.H0,
            "Tcmb": self.T_cmb,
            "YHe": self.Y_He,
            "Neff": self.Neff_massless,
            "Nmnu": self.num_massive_neutrinos,
            "mnu": self.mnu,
        }


_BACKGROUND_PARAM_FIELDS = {
    "Omegam": "Omegam",
    "Omegab": "Omegab",
    "w_DE_0": "w_DE_0",
    "w_DE_a": "w_DE_a",
    "cs2_DE": "cs2_DE",
    "Omegak": "Omegak",
    "A_s": "A_s",
    "n_s": "n_s",
    "H0": "H0",
    "Tcmb": "T_cmb",
    "YHe": "Y_He",
    "Neff": "Neff_massless",
    "Nmnu": "num_massive_neutrinos",
    "mnu": "mnu",
}
"""Attribute behind each :meth:`_ParamCosmology.to_background_params` key.

What lets :func:`_build_thermo_tables_batch` stack the same parameters over
cosmologies. :func:`_check_background_param_fields` pins the two together so the
map cannot drift from the method."""


def _check_background_param_fields(cosmology) -> None:
    """Raise if :data:`_BACKGROUND_PARAM_FIELDS` has drifted from the method."""

    expected = cosmology.to_background_params()
    got = {key: getattr(cosmology, attr) for key, attr in _BACKGROUND_PARAM_FIELDS.items()}
    if got != expected:
        raise AssertionError(
            "_BACKGROUND_PARAM_FIELDS is out of sync with "
            f"to_background_params: {got} != {expected}"
        )


def _as_cosmology(cosmology):
    """Normalize a legacy parameter dictionary for the accelerator path."""

    if not isinstance(cosmology, dict):
        return cosmology
    return _ParamCosmology(
        Omegam=float(cosmology["Omegam"]),
        Omegab=float(cosmology["Omegab"]),
        w_DE_0=float(cosmology.get("w_DE_0", -1.0)),
        w_DE_a=float(cosmology.get("w_DE_a", 0.0)),
        cs2_DE=float(cosmology.get("cs2_DE", 1.0)),
        Omegak=float(cosmology.get("Omegak", 0.0)),
        A_s=float(cosmology["A_s"]),
        n_s=float(cosmology["n_s"]),
        H0=float(cosmology["H0"]),
        T_cmb=float(cosmology["Tcmb"]),
        Y_He=float(cosmology["YHe"]),
        Neff_massless=float(cosmology["Neff"]),
        num_massive_neutrinos=float(cosmology.get("Nmnu", 0.0)),
        mnu=float(cosmology.get("mnu", 0.0)),
        k_pivot=float(cosmology.get("k_p", 0.05)),
    )

# Solver and grid presets for the accelerated matter-power path.
N_THERMO_GRID = 2048
N_BACKGROUND_GRID = 4096
THERMO_BATCH_CHUNK = 32
"""Cosmologies per vmapped background solve in :func:`build_thermo_tables_batch`.

Bounds the working set of one batched RECFAST solve, and keeps repeated batches
on one compiled program: any batch of 32 or more reuses the same executable, so
only a trailing partial chunk can trigger a second compilation.
"""
PERTURB_RTOL = 1.0e-4
PERTURB_ATOL = 1.0e-4
PERTURB_FIRST_STEP = 1.0e-2
PERTURB_MAX_STEPS = 20000
TAU_START = 0.1

# Packed-parameter row layout consumed by the device rhs/jac.
#   p = (grhog, grhornomass, grhoc, grhob, grhov,
#        tau_start, tau_end, k, cosmology_idx, save_index,
#        w_DE_0, w_DE_a, cs2_DE, grhok, grhor_nu, n_mnu, amnu,
#        eps, dir_grhog, dir_grhornomass, dir_grhoc, dir_grhob, dir_grhov)
# grhok = grhom * Omega_k is the (constant) comoving curvature density
# coefficient 8*pi*G*rho_K a^2; it is 0 for a flat cosmology.
# The rhs reads density ``c`` as ``p[c] + p[IX_EPS] * p[IX_DIR + c]``. ``eps``
# is zero in every primal solve, so the trajectory is what it always was; its
# derivative with respect to ``eps`` alone is the directional derivative along
# the row's ``dir`` vector. That is what lets a trajectory carry a single
# sensitivity column for a direction of its own -- see
# :func:`matter_power_spectrum_jax`.
IX_TAU_START = 5
IX_TAU_END = 6
IX_K = 7
IX_COSMOLOGY = 8
IX_SAVE_INDEX = 9
IX_W_DE_0 = 10
IX_W_DE_A = 11
IX_CS2_DE = 12
IX_GRHOK = 13
IX_GRHOR_NU = 14
IX_NMNU = 15
IX_AMNU = 16
IX_EPS = 17
IX_DIR = 18
N_PARAM = IX_DIR + 5

THERMO_TABLE_DTYPE = np.float32
TRAJECTORIES_PER_BLOCK = 32
"""Wave modes packed into one CUDA block by modax's thread-local Rodas5P kernel."""

CUDA_MAX_REGISTERS = int(os.environ.get("DISCO_CUDA_MAX_REGISTERS", "168"))
"""Per-thread register cap for that kernel: fewer registers, more resident warps."""

NQMAX = 3
"""Massive-neutrino momentum bins used by both the background and the hierarchy.

Each bin adds ``lmaxnu + 1 == 16`` state variables and the generated dense
Jacobian scales as ``nvar^2``, so this sets the numba compile time: 3 bins give
``nvar = 98``, 5 bins give ``nvar = 130`` and compile ~4x slower for no gain in
P(k) accuracy against CLASS.
"""


def neutrino_momentum_bins(nqmax: int = NQMAX):
    """Return ``(q, w, dlnf0_dlnq)`` momentum bins as NumPy arrays.

    ``w`` are normalized so a single massless flavour integrates to unit density,
    and ``dlnf0/dlnq = -q/(1 + exp(-q))`` is the log-derivative of the
    Fermi-Dirac background distribution that sources the ``psi`` hierarchy.
    """

    q, w = get_neutrino_momentum_bins(nqmax)
    q = np.asarray(q, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    dlfdlq = -q / (1.0 + np.exp(-q))
    return q, w, dlfdlq


def massive_neutrino_density_ratio(a, amnu: float, q, w) -> np.ndarray:
    """Return ``rho_nu(a)/rho_nu0_massless`` on the same momentum quadrature.

    ``rho = sum_i w_i / v_i`` with ``v_i = 1/sqrt(1 + (a amnu/q_i)^2)``, i.e.
    ``rho = sum_i w_i sqrt(1 + (a amnu/q_i)^2)``. It tends to 1 (relativistic)
    as ``a -> 0``. Using the *same* quadrature as the perturbation hierarchy
    keeps ``drho_nu/rho_nu`` consistent.
    """

    a = np.atleast_1d(np.asarray(a, dtype=np.float64))[:, None]
    return np.sum(w[None, :] * np.sqrt(1.0 + (a * amnu / q[None, :]) ** 2), axis=1)


def massive_neutrino_parameters(cosmology) -> tuple[float, float, float]:
    """Return ``(grhor_nu, N_mnu, amnu)`` for the massive-neutrino species."""

    grhor_nu = 3.39739477e-14 * cosmology.T_cmb**4
    n_mnu = float(cosmology.num_massive_neutrinos)
    amnu = 1.62581581e4 * cosmology.mnu / cosmology.T_cmb if n_mnu > 0.0 else 0.0
    return grhor_nu, n_mnu, amnu


def density_coefficients(cosmology) -> tuple[float, float, float, float, float]:
    """Return the background ``grho`` density coefficients for ``cosmology``.

    ``grhornomass`` uses the *massless* neutrino count. The dark-energy
    coefficient is fixed by closing the density budget at ``a = 1``,

    ``grhov = grhom - (grhog + grhornomass + grho_mnu(1) + grhoc + grhob + grhok)``,

    which reproduces the flat massless value and automatically accounts for
    spatial curvature and the massive-neutrino density.
    """

    grhom = 3.33795017e-11 * cosmology.H0**2
    grhog_value = 1.49594245e-13 * cosmology.T_cmb**4
    grhornomass_value = 3.39739477e-14 * cosmology.T_cmb**4 * cosmology.Neff_massless
    grhoc_value = grhom * cosmology.Omegac
    grhob_value = grhom * cosmology.Omegab
    grhok = grhom * cosmology.Omegak

    grhor_nu, n_mnu, amnu = massive_neutrino_parameters(cosmology)
    if n_mnu > 0.0:
        q, w, _ = neutrino_momentum_bins()
        grho_mnu_today = grhor_nu * n_mnu * float(
            massive_neutrino_density_ratio(1.0, amnu, q, w)[0]
        )
    else:
        grho_mnu_today = 0.0

    grhov_value = grhom - (
        grhog_value
        + grhornomass_value
        + grho_mnu_today
        + grhoc_value
        + grhob_value
        + grhok
    )
    return (grhog_value, grhornomass_value, grhoc_value, grhob_value, grhov_value)


_THERMO_TABLE_CACHE: dict = {}


def build_thermo_tables(cosmology, n_grid: int = N_THERMO_GRID):
    """Cached wrapper around :func:`_build_thermo_tables`.

    Each call runs a full RECFAST recombination solve plus the background
    quadrature, so it is far too expensive to repeat per evaluation -- a CMB batch
    needs every cosmology's ``tau0`` up front *and* again inside the solve
    preparation. The tables depend only on ``(cosmology, n_grid)``, so memoize
    them.
    """

    key = (cosmology, n_grid)
    cached = _THERMO_TABLE_CACHE.get(key)
    if cached is None:
        cached = _build_thermo_tables(cosmology, n_grid)
        _THERMO_TABLE_CACHE[key] = cached
    return cached


def build_thermo_tables_batch(cosmologies, n_grid: int = N_THERMO_GRID):
    """Return one :func:`build_thermo_tables` tuple per cosmology, batched.

    The same tables as mapping :func:`build_thermo_tables` over ``cosmologies``,
    memoized in the same cache, but the misses are solved together: the RECFAST
    recombination history and the background quadrature are ``jax.vmap``-ed over
    the batch, turning 128 sequential solves of ~0.8 s each into one compiled
    program and four launches -- about 100 s down to 10 s.

    ``vmap`` rather than the parameter batch axis :func:`batch_dimensions`
    threads through the splines. Under ``vmap`` each cosmology keeps its own
    adaptive step sequence; a batch axis would instead put all of them under the
    single scalar error norm of the thermodynamics ``PIDController``, tying every
    cosmology's step size to the worst one in the batch.

    Not bit-identical to the serial builder. ``tau_grid`` and ``tau0`` match
    exactly, but batching re-associates the arithmetic inside the RECFAST solve,
    and at its default ``rtol = 1e-5`` that is enough to flip step-acceptance
    decisions: ``xe`` moves by up to ~2e-3 at early times, in cosmologies where
    the solve is barely converged to begin with. That is a property of the
    tolerance, not of batching -- the serial solve is itself only good to ~8e-2
    there against a converged reference. It does not reach the observable:
    ``P(k)`` shifts by at most ~7e-6, against a test gate of 5e-3.
    """

    order = []
    todo = []
    for cosmology in cosmologies:
        key = (cosmology, n_grid)
        if key not in _THERMO_TABLE_CACHE and key not in todo:
            todo.append(key)
        order.append(key)

    for start in range(0, len(todo), THERMO_BATCH_CHUNK):
        chunk = todo[start : start + THERMO_BATCH_CHUNK]
        tables = _build_thermo_tables_batch([key[0] for key in chunk], n_grid)
        _THERMO_TABLE_CACHE.update(zip(chunk, tables))

    return tuple(_THERMO_TABLE_CACHE[key] for key in order)


@partial(jax.jit, static_argnames=("n_grid",))
def _solve_backgrounds(stacked, n_grid: int):
    """Solve ``evolve_background`` for a stack of cosmologies, one per vmap lane."""

    return jax.vmap(
        lambda params: evolve_background(
            param=params, thermo_module="RECFAST", num_thermo=n_grid
        )
    )(stacked)


@partial(jax.jit, static_argnames=("n_grid",))
def _solve_thermal_histories(stacked, n_grid: int):
    """The RECFAST solve of ``evolve_background`` alone, one vmap lane per cosmology."""

    def one(params):
        param = setup_background_evolution(
            amin=1e-9, amax=1.01, param=batch_dimensions(dict(params))
        )
        return evaluate_thermo(param=param, num_thermo=n_grid, rtol=1e-5, atol=1e-7)

    return jax.vmap(one)(stacked)


def thermal_history_batch(cosmologies, n_grid: int = N_THERMO_GRID) -> dict:
    """Solve the recombination history of a batch of cosmologies in one launch.

    Returns host arrays: ``a`` of shape ``(n_grid + 1,)`` (the adaptive RECFAST
    grid), and ``xe``, ``Tm`` and ``cs2`` of shape ``(n_cosmologies, n_grid + 1)``:
    the free-electron fraction ``n_e / n_H``, the matter temperature in K and the
    baryon sound speed squared. It is the thermodynamics half of
    :func:`build_thermo_tables_batch`, exposed so the recombination solve can be
    checked and timed on its own.
    """

    cosmologies = tuple(_as_cosmology(cosmology) for cosmology in cosmologies)
    _check_background_param_fields(cosmologies[0])
    stacked = {
        key: jnp.asarray([getattr(c, attr) for c in cosmologies], dtype=jnp.float64)
        for key, attr in _BACKGROUND_PARAM_FIELDS.items()
    }
    a, cs2, Tm, mu, xe, dxeda = _solve_thermal_histories(stacked, n_grid)
    return {
        "a": np.asarray(a)[0],
        "xe": np.asarray(xe)[:, :, 0],
        "Tm": np.asarray(Tm)[:, :, 0],
        "cs2": np.asarray(cs2)[:, :, 0],
    }


def _build_thermo_tables_batch(cosmologies, n_grid: int = N_THERMO_GRID):
    """Uncached :func:`build_thermo_tables_batch`, for one chunk of cosmologies."""

    _check_background_param_fields(cosmologies[0])
    stacked = {
        key: jnp.asarray([getattr(c, attr) for c in cosmologies], dtype=jnp.float64)
        for key, attr in _BACKGROUND_PARAM_FIELDS.items()
    }
    backgrounds = _solve_backgrounds(stacked, n_grid)
    # Lane i has the same leaves and shapes the serial solve returns for
    # cosmologies[i], down to the width-1 spline batch axis, so the resampling
    # tail below is the unmodified serial one.
    return tuple(
        _thermo_tables_from_background(
            jax.tree_util.tree_map(lambda leaf, j=i: leaf[j], backgrounds),
            cosmology,
            n_grid,
        )
        for i, cosmology in enumerate(cosmologies)
    )


def _build_thermo_tables(cosmology, n_grid: int = N_THERMO_GRID):
    """Return ``(tau_grid, values, seconds, tau0)`` thermodynamics spline tables.

    ``values`` has rows ``(a, kappa', c_s,b^2, rho_nu)`` sampled on a uniform
    ``log(tau)`` grid over ``[TAU_START, tau0]``. The background and RECFAST
    histories come directly from :func:`discoeb.background.evolve_background`;
    this function only resamples its existing splines into the layout required by
    the numba-CUDA callbacks. ``seconds`` contains the matching cubic second
    derivatives with respect to ``log(tau)``.
    """

    return _thermo_tables_from_background(
        evolve_background(
            param=cosmology.to_background_params(),
            thermo_module="RECFAST",
            num_thermo=n_grid,
        ),
        cosmology,
        n_grid,
    )


def _thermo_tables_from_background(background, cosmology, n_grid: int):
    """Resample a solved background into the device tables.

    The half of :func:`_build_thermo_tables` after the RECFAST solve, split out
    so the serial and batched builders share it verbatim. ``background`` is one
    cosmology's ``evolve_background`` dict, i.e. with its spline batch axis at
    width 1 -- either straight from the solver, or sliced out of a vmapped batch
    by :func:`_build_thermo_tables_batch`.
    """

    a_background = np.geomspace(1.0e-9, 1.0, N_BACKGROUND_GRID)
    dtauda_values = np.asarray(
        dtauda(jnp.asarray(a_background), background)
    )[:, 0]
    tau_background = integrate.cumulative_trapezoid(
        dtauda_values, a_background, initial=0.0
    )
    tau_background += float(np.asarray(background["taumin"]).reshape(-1)[0])
    tau0 = float(tau_background[-1])
    log_tau_grid = np.linspace(np.log(TAU_START), np.log(tau0), n_grid)
    tau_grid = np.exp(log_tau_grid)
    a_vals = np.interp(tau_grid, tau_background, a_background)
    loga_eval = jnp.log(jnp.asarray(a_vals))

    xe_vals = np.asarray(background["xe_of_loga_spline"].evaluate(loga_eval))[:, 0]
    a_recfast_start = 1.0 / 3501.0
    xe_recfast_start = float(
        np.asarray(
            background["xe_of_loga_spline"].evaluate(
                jnp.log(jnp.asarray([a_recfast_start]))
            )
        ).reshape(-1)[0]
    )
    xe_vals = np.where(a_vals < a_recfast_start, xe_recfast_start, xe_vals)
    akthom = (
        2.3038921003709498e-9
        * (1.0 - cosmology.Y_He)
        * cosmology.Omegab
        * cosmology.H0**2
    )
    opacity_vals = xe_vals * akthom / a_vals**2
    cs2a_vals = np.asarray(
        background["cs2a_of_loga_spline"].evaluate(loga_eval)
    )[:, 0]
    cs2_vals = cs2a_vals / a_vals
    rhonu_vals = np.exp(
        np.asarray(
            background["logrhonu_of_loga_spline"].evaluate(jnp.log(a_vals))
        )[:, 0]
    )

    values = np.stack((a_vals, opacity_vals, cs2_vals, rhonu_vals))
    seconds = np.stack(
        [interpolate.CubicSpline(log_tau_grid, row)(log_tau_grid, 2) for row in values]
    )
    return tau_grid, values, seconds, tau0


def make_params(cosmology, k_values: np.ndarray, tau_end: float) -> np.ndarray:
    """Return per-mode packed perturbation parameters for the device rhs."""

    densities = density_coefficients(cosmology)
    n = len(k_values)
    rows = np.repeat(np.asarray(densities)[None, :], n, axis=0)
    tau_start = np.full(n, TAU_START, dtype=np.float64)
    tau_end_col = np.full(n, tau_end, dtype=np.float64)
    cosmology_idx = np.zeros(n, dtype=np.float64)
    save_index = np.ones(n, dtype=np.float64)
    w_de_0 = np.full(n, cosmology.w_DE_0, dtype=np.float64)
    w_de_a = np.full(n, cosmology.w_DE_a, dtype=np.float64)
    cs2_de = np.full(n, cosmology.cs2_DE, dtype=np.float64)
    grhok = 3.33795017e-11 * cosmology.H0**2 * cosmology.Omegak
    grhok_col = np.full(n, grhok, dtype=np.float64)
    grhor_nu, n_mnu, amnu = massive_neutrino_parameters(cosmology)
    grhor_nu_col = np.full(n, grhor_nu, dtype=np.float64)
    n_mnu_col = np.full(n, n_mnu, dtype=np.float64)
    amnu_col = np.full(n, amnu, dtype=np.float64)
    # No sensitivity direction: eps and dir are zero, so the densities the rhs
    # sees are exactly the five in ``rows``.
    direction = np.zeros((n, N_PARAM - IX_EPS), dtype=np.float64)
    return np.column_stack(
        (rows, tau_start, tau_end_col, np.asarray(k_values, dtype=np.float64),
         cosmology_idx, save_index, w_de_0, w_de_a, cs2_de, grhok_col,
         grhor_nu_col, n_mnu_col, amnu_col, direction)
    )


def make_initial_states(cosmology, k_values: np.ndarray, layout) -> np.ndarray:
    """Return adiabatic initial states for every wave mode, shape ``(n_k, nvar)``.

    The always-present core (metric + fluid + photon/polarization/massless-nu
    hierarchies) is set by :func:`adiabatic_initial_conditions`; when dark energy
    is enabled, the DE fluid is given its adiabatic value
    ``clxq = (1 + w_0) clxc``, ``thetaq = 0``.
    """

    densities = density_coefficients(cosmology)
    y0 = np.zeros((len(k_values), layout.nvar), dtype=np.float64)
    if layout.has_massive_neutrinos:
        _, _, dlfdlq = neutrino_momentum_bins(layout.nqmax)
    for i, k in enumerate(k_values):
        core = adiabatic_initial_conditions(float(k), TAU_START, *densities[:4])
        # ``core`` is laid out by the module's index constants; place its
        # non-zero blocks at the layout's indices.
        y0[i, layout.ix_etak : layout.ix_vb + 1] = core[IX_ETAK : IX_VB + 1]
        y0[i, layout.ix_g : layout.ix_g + 2] = core[IX_G : IX_G + 2]
        y0[i, layout.ix_r : layout.ix_r + 4] = core[IX_R : IX_R + 4]
        if layout.enable_dark_energy:
            y0[i, layout.ix_clxq] = (1.0 + cosmology.w_DE_0) * core[IX_CLXC]
            y0[i, layout.ix_thetaq] = 0.0
        if layout.has_massive_neutrinos:
            # Deep in radiation domination the massive neutrinos are still
            # relativistic (a*amnu/q << 1, v -> 1), so their phase-space
            # perturbation reduces to the massless hierarchy:
            #   psi_l = -(1/4) N_l dln f0/dln q      (Ma & Bertschinger)
            for qi in range(layout.nqmax):
                for l in range(0, min(3, LMAX_NR) + 1):
                    y0[i, layout.ix_psi(l, qi)] = -0.25 * core[IX_R + l] * dlfdlq[qi]
    return y0


def build_numba_rhs(layout, tau_min, inv_dtau, values, seconds):
    """Return the numba-CUDA right-hand side ``f(y, tau, p) -> tuple`` for a layout.

    The thermodynamics tables are uploaded once and read through a uniform
    ``log(tau)`` cubic-spline device evaluator. The right-hand side is written
    directly in the form Enzyme can differentiate for the Jacobian and
    ``df/dt``: it takes and returns fixed-size tuples of scalars and indexes
    ``y`` and ``p`` only at build-time constants, because a run-time index into
    a tuple makes numba emit a bounds check Enzyme refuses. The same function
    evaluated over array rows serves the kernel's primal stage evaluations.

    The multipole hierarchies are therefore not looped over. Each multipole
    above the quadrupole is its own one-line device function with its state
    index and ``l`` baked in, and the multipoles of a hierarchy are folded into
    one function at build time by tuple concatenation
    (:func:`hierarchy_tail`), so the truncations are read from ``layout`` and
    the state-vector layout, the initial conditions and the sparsity pattern
    change together. The dynamical-dark-energy fluid equations and the
    massive-neutrino hierarchies are included when the layout enables them;
    both flags are compile-time constants, so numba prunes the disabled
    branches entirely (the flat-LambdaCDM kernel is unchanged). Each
    massive-neutrino momentum bin is likewise its own device function with the
    bin's index base and quadrature node baked in, folded over the bins the
    same way.

    The tables are stacked over cosmologies: ``values`` and ``seconds`` have
    shape ``(n_cosmologies, n_channels, n_grid)`` and ``tau_min`` / ``inv_dtau``
    are ``(n_cosmologies,)``. The device evaluator selects a cosmology's table by
    the ``IX_COSMOLOGY`` tag on each trajectory's parameter row, so one compiled
    kernel solves an arbitrary batch of cosmologies. A single-cosmology solve is
    just ``n_cosmologies == 1``.
    """

    from numba_cuda_mlir import cuda

    ix_g, lmaxg = layout.ix_g, layout.lmaxg
    ix_pol, lmaxpol = layout.ix_pol, layout.lmaxpol
    ix_r, lmaxr = layout.ix_r, layout.lmaxr

    ENABLE_DE = layout.enable_dark_energy
    IX_CLXQ = layout.ix_clxq if ENABLE_DE else 0
    IX_THETAQ = layout.ix_thetaq if ENABLE_DE else 0

    ENABLE_MNU = layout.has_massive_neutrinos
    NQ = layout.nqmax if ENABLE_MNU else 0
    lmaxnu = layout.lmaxnu
    IX_MNU = layout.ix_massive_nu if ENABLE_MNU else 0

    # The quadrupole equations below couple to l = 3, so every hierarchy needs
    # at least an octupole to truncate at.
    truncations = {"lmaxg": lmaxg, "lmaxpol": lmaxpol, "lmaxr": lmaxr}
    if ENABLE_MNU:
        truncations["lmaxnu"] = lmaxnu
    if any(lmax < 3 for lmax in truncations.values()):
        raise ValueError(f"every hierarchy truncation must be >= 3; got {truncations}")

    n_grid = values.shape[-1]
    tau_min_dev = cuda.to_device(np.ascontiguousarray(tau_min, dtype=np.float64))
    inv_dtau_dev = cuda.to_device(np.ascontiguousarray(inv_dtau, dtype=np.float64))
    values_dev = cuda.to_device(np.ascontiguousarray(values, dtype=THERMO_TABLE_DTYPE))
    seconds_dev = cuda.to_device(np.ascontiguousarray(seconds, dtype=THERMO_TABLE_DTYPE))

    @cuda.jit(device=True)
    def spline_eval(x, p, channel):
        cosmology_idx = int(p[IX_COSMOLOGY])
        x_min = tau_min_dev[cosmology_idx]
        inv_dx = inv_dtau_dev[cosmology_idx]
        idx = int((x - x_min) * inv_dx)
        if idx < 0:
            idx = 0
        if idx > n_grid - 2:
            idx = n_grid - 2
        h = 1.0 / inv_dx
        x0 = x_min + idx * h
        x1 = x0 + h
        left = x1 - x
        right = x - x0
        y0 = values_dev[cosmology_idx, channel, idx]
        y1 = values_dev[cosmology_idx, channel, idx + 1]
        m0 = seconds_dev[cosmology_idx, channel, idx]
        m1 = seconds_dev[cosmology_idx, channel, idx + 1]
        return (
            m0 * left**3 / (6.0 * h)
            + m1 * right**3 / (6.0 * h)
            + (y0 - m0 * h**2 / 6.0) * left / h
            + (y1 - m1 * h**2 / 6.0) * right / h
        )

    # --- Free-streaming hierarchies -------------------------------------------
    # Interior multipoles and the truncated top multipole of a Boltzmann
    # hierarchy, for temperature-like (photons, massless and massive neutrinos)
    # and E-mode couplings. ``damping`` is the Thomson opacity for photons and
    # zero for neutrinos; ``k`` is ``k v`` for a massive-neutrino momentum bin.

    @cuda.jit(device=True)
    def stream(ell, lower, this, upper, k, damping):
        """``psi_l' = k (l psi_{l-1} - (l+1) psi_{l+1}) / (2l+1) - damping psi_l``."""
        return k * (ell * lower - (ell + 1.0) * upper) / (2.0 * ell + 1.0) - damping * this

    @cuda.jit(device=True)
    def truncate(ell, lower, this, k, damping, tau):
        """``psi_L' = k psi_{L-1} - (L+1) psi_L / tau - damping psi_L``."""
        return k * lower - (ell + 1.0) * this / tau - damping * this

    @cuda.jit(device=True)
    def stream_pol(ell, lower, this, upper, k, opacity):
        """``E_l'``, whose coupling upwards carries ``(l+3)(l-1)/(l+1)``."""
        polfac = (ell + 3.0) * (ell - 1.0) / (ell + 1.0)
        return k * (ell * lower - polfac * upper) / (2.0 * ell + 1.0) - opacity * this

    @cuda.jit(device=True)
    def truncate_pol(ell, lower, this, k, opacity, tau):
        """``E_L' = k L E_{L-1} / (2L+1) - (L+3) E_L / tau - opacity E_L``."""
        return k * ell * lower / (2.0 * ell + 1.0) - (ell + 3.0) * this / tau - opacity * this

    def hierarchy_tail(slot, lmax, interior, top):
        """The derivatives of multipoles ``3 .. lmax`` of one hierarchy, as a tuple.

        ``slot(l)`` is the state index of multipole ``l``. Each multipole is a
        device function with its index and ``l`` baked in that appends its own
        derivative to those of the multipoles below it, so nothing indexes
        ``y`` at run time and the chain returns the tail in ascending ``l``.
        """

        def extend(below, ell):
            lower, this, upper = slot(ell - 1), slot(ell), slot(ell + 1)
            ell = float(ell)
            if ell < lmax and below is None:

                @cuda.jit(device=True)
                def tail(y, k, damping, tau):
                    return (interior(ell, y[lower], y[this], y[upper], k, damping),)

            elif ell < lmax:

                @cuda.jit(device=True)
                def tail(y, k, damping, tau):
                    return below(y, k, damping, tau) + (
                        interior(ell, y[lower], y[this], y[upper], k, damping),
                    )

            elif below is None:  # a hierarchy truncated right at the octupole

                @cuda.jit(device=True)
                def tail(y, k, damping, tau):
                    return (top(ell, y[lower], y[this], k, damping, tau),)

            else:

                @cuda.jit(device=True)
                def tail(y, k, damping, tau):
                    return below(y, k, damping, tau) + (
                        top(ell, y[lower], y[this], k, damping, tau),
                    )

            return tail

        tail = None
        for ell in range(3, lmax + 1):
            tail = extend(tail, ell)
        return tail

    # Theta_l at ix_g + l, E_l at ix_pol + (l - 2), N_l at ix_r + l.
    photon_tail = hierarchy_tail(lambda l: ix_g + l, lmaxg, stream, truncate)
    polarization_tail = hierarchy_tail(lambda l: ix_pol + l - 2, lmaxpol, stream_pol, truncate_pol)
    neutrino_tail = hierarchy_tail(lambda l: ix_r + l, lmaxr, stream, truncate)

    # --- Massive neutrinos -----------------------------------------------------
    if ENABLE_MNU:
        q_np, w_np, dlf_np = neutrino_momentum_bins(NQ)

        def make_bin(qi):
            """The two device functions of momentum bin ``qi``, constants baked in."""
            b = IX_MNU + qi * (lmaxnu + 1)  # psi_l of this bin at b + l
            q = float(q_np[qi])
            w = float(w_np[qi])
            dl = float(dlf_np[qi])
            tail = hierarchy_tail(lambda l: b + l, lmaxnu, stream, truncate)

            @cuda.jit(device=True)
            def moments(y, a, amnu):
                # The bin's share of the momentum integrals
                #   drho_nu = sum_i w_i psi0_i / v_i ,   f_nu = sum_i w_i psi1_i
                vq = 1.0 / math.sqrt(1.0 + (a * amnu / q) ** 2)
                return (w * y[b] / vq, w * y[b + 1])

            @cuda.jit(device=True)
            def derivatives(y, a, amnu, k, z, sigma, tau):
                # Massive-neutrino phase-space hierarchy (Ma & Bertschinger; CLASS
                # perturbations.c, synchronous gauge with metric_continuity = h'/2,
                # metric_euler = 0, metric_shear = k*sigma, and h' = 2 k z):
                #   psi0' = -k v psi1 + (k z / 3) dlnf0
                #   psi1' = (k v / 3) (psi0 - 2 psi2)
                #   psi2' = (k v / 5) (2 psi1 - 3 psi3) - (2/15) k sigma dlnf0
                #   psi_l' = (k v / (2l+1)) (l psi_{l-1} - (l+1) psi_{l+1})
                #   psi_L' = k v psi_{L-1} - (L+1) psi_L / tau
                vq = 1.0 / math.sqrt(1.0 + (a * amnu / q) ** 2)
                kv = k * vq
                psi0, psi1, psi2, psi3 = y[b], y[b + 1], y[b + 2], y[b + 3]
                return (
                    -kv * psi1 + (k * z / 3.0) * dl,
                    kv / 3.0 * (psi0 - 2.0 * psi2),
                    kv / 5.0 * (2.0 * psi1 - 3.0 * psi3) - (2.0 / 15.0) * k * sigma * dl,
                ) + tail(y, kv, 0.0, tau)

            return moments, derivatives

        def summed(fns):
            """Fold the bins' ``moments`` into one device function of their sums."""
            if len(fns) == 1:
                return fns[0]
            rest, last = summed(fns[:-1]), fns[-1]

            @cuda.jit(device=True)
            def total(y, a, amnu):
                so_far = rest(y, a, amnu)
                this_bin = last(y, a, amnu)
                return (so_far[0] + this_bin[0], so_far[1] + this_bin[1])

            return total

        def joined_bins(fns):
            """Fold the bins' ``derivatives`` into one device function, in bin order."""
            if len(fns) == 1:
                return fns[0]
            rest, last = joined_bins(fns[:-1]), fns[-1]

            @cuda.jit(device=True)
            def joined(y, a, amnu, k, z, sigma, tau):
                return rest(y, a, amnu, k, z, sigma, tau) + last(
                    y, a, amnu, k, z, sigma, tau
                )

            return joined

        bins = [make_bin(qi) for qi in range(NQ)]
        mnu_moments = summed([moments for moments, _ in bins])
        mnu_derivatives = joined_bins([derivatives for _, derivatives in bins])
    else:
        mnu_moments = mnu_derivatives = None

    def perturbation_rhs(y, tau, p):
        k = p[IX_K]
        log_tau = math.log(tau)
        a = spline_eval(log_tau, p, 0)
        opacity = spline_eval(log_tau, p, 1)
        cs2 = spline_eval(log_tau, p, 2)
        if opacity < 1.0e-30:
            opacity = 1.0e-30
        if cs2 < 0.0:
            cs2 = 0.0

        a2 = a * a
        # The densities are read through the row's sensitivity direction: eps
        # is zero in a primal solve and d/d(eps) is the derivative along dir.
        eps = p[IX_EPS]
        grhog_t = (p[0] + eps * p[IX_DIR]) / a2
        grhor_t = (p[1] + eps * p[IX_DIR + 1]) / a2
        grhoc_t = (p[2] + eps * p[IX_DIR + 2]) / a
        grhob_t = (p[3] + eps * p[IX_DIR + 3]) / a
        grhov = p[4] + eps * p[IX_DIR + 4]
        if ENABLE_DE:
            w0 = p[IX_W_DE_0]
            wa = p[IX_W_DE_A]
            w_Q = w0 + wa * (1.0 - a)
            rho_Q = a ** (-3.0 * (1.0 + w0 + wa)) * math.exp(3.0 * (a - 1.0) * wa)
            grhov_t = grhov * rho_Q * a2
        else:
            grhov_t = grhov * a2
        if ENABLE_MNU:
            grhor_nu = p[IX_GRHOR_NU]
            n_mnu = p[IX_NMNU]
            amnu = p[IX_AMNU]
            rhonu = spline_eval(log_tau, p, 3)
            grho_mnu_t = grhor_nu * n_mnu * rhonu / a2
        else:
            grho_mnu_t = 0.0
        grhok = p[IX_GRHOK]
        adotoa = math.sqrt(
            (grhog_t + grhor_t + grho_mnu_t + grhoc_t + grhob_t + grhov_t + grhok) / 3.0
        )

        etak = y[IX_ETAK]
        clxc = y[IX_CLXC]
        clxb = y[IX_CLXB]
        vb = y[IX_VB]
        clxg, qg, pig, theta3 = y[ix_g], y[ix_g + 1], y[ix_g + 2], y[ix_g + 3]
        e2, e3 = y[ix_pol], y[ix_pol + 1]
        clxr, qr, pir, n3 = y[ix_r], y[ix_r + 1], y[ix_r + 2], y[ix_r + 3]

        dgrho = grhob_t * clxb + grhoc_t * clxc + grhog_t * clxg + grhor_t * clxr
        dgq = grhob_t * vb + grhog_t * qg + grhor_t * qr
        if ENABLE_DE:
            clxq = y[IX_CLXQ]
            thetaq = y[IX_THETAQ]
            dgrho = dgrho + grhov_t * clxq
            dgq = dgq + grhov_t * (1.0 + w_Q) * thetaq / k
        if ENABLE_MNU:
            drhonu, fnu = mnu_moments(y, a, amnu)
            dgrho = dgrho + grhor_nu * n_mnu * drhonu / a2
            dgq = dgq + grhor_nu * n_mnu * fnu / a2
        # Curved-geometry metric relations (CLASS perturbations.c, synchronous):
        #   h'    = (k^2 s2^2 eta + 1.5 a^2 delta_rho)/(0.5 aH),   z = h'/(2k)
        #   eta'  = (1.5 a^2 (rho+p)theta + 0.5 K h')/(k^2 s2^2)
        #   sigma = k alpha = (z + 1.5 dgq/k^2)/s2^2
        # with K = -grhok/3 and s2^2 = 1 - 3K/k^2 = 1 + grhok/k^2 (= 1 when flat).
        s2 = 1.0 + grhok / (k * k)
        Kcurv = -grhok / 3.0
        z = (0.5 * dgrho / k + s2 * etak) / adotoa
        sigma = (z + 1.5 * dgq / (k * k)) / s2
        photbar = grhog_t / grhob_t
        pb43 = 4.0 / 3.0 * photbar
        delta_p_b = cs2 * clxb
        polter = pig / 10.0 + 9.0 * e2 / 15.0
        vbdot = -adotoa * vb + k * delta_p_b - photbar * opacity * (4.0 * vb / 3.0 - qg)

        metric_and_matter = (
            (0.5 * dgq + Kcurv * z) / s2,  # etak
            -k * z,  # clxc
            -k * (z + vb),  # clxb
            vbdot,  # vb
        )
        photons = (
            -k * (4.0 * z / 3.0 + qg),
            4.0 * (-vbdot - adotoa * vb + k * delta_p_b) / (3.0 * pb43)
            + k * clxg / 3.0
            - 2.0 * k * pig / 3.0,
            2.0 * k * qg / 5.0
            - 3.0 * k * theta3 / 5.0
            - opacity * (pig - polter)
            + 8.0 * k * sigma / 15.0,
        ) + photon_tail(y, k, opacity, tau)
        polarization = (-opacity * (e2 - polter) - k * e3 / 3.0,) + polarization_tail(
            y, k, opacity, tau
        )
        massless_neutrinos = (
            -k * (4.0 * z / 3.0 + qr),
            k * (clxr - 2.0 * pir) / 3.0,
            2.0 * k * qr / 5.0 - 3.0 * k * n3 / 5.0 + 8.0 * k * sigma / 15.0,
        ) + neutrino_tail(y, k, 0.0, tau)

        if ENABLE_DE:
            cs2_Q = p[IX_CS2_DE]
            w_Q_prime = -wa * adotoa * a
            ca2_Q = w_Q - w_Q_prime / (3.0 * (1.0 + w_Q) * adotoa)
            dark_energy = (
                -(1.0 + w_Q) * (thetaq + k * z)
                - 3.0 * (cs2_Q - w_Q) * adotoa * clxq
                - 9.0 * (1.0 + w_Q) * (cs2_Q - ca2_Q) * adotoa**2 / k**2 * thetaq,
                -(1.0 - 3.0 * cs2_Q) * adotoa * thetaq
                + cs2_Q / (1.0 + w_Q) * k**2 * clxq,
            )
        else:
            dark_energy = ()

        if ENABLE_MNU:
            massive_neutrinos = mnu_derivatives(y, a, amnu, k, z, sigma, tau)
        else:
            massive_neutrinos = ()

        return (
            metric_and_matter
            + photons
            + polarization
            + massless_neutrinos
            + dark_energy
            + massive_neutrinos
        )

    return perturbation_rhs


class _PreparedSolve(NamedTuple):
    """Compiled artifacts shared by every cosmology of a (possibly batched) solve."""

    cosmologies: tuple
    layout: object
    tables: tuple  # per-cosmology (tau_grid, values, seconds, tau0)
    tau0: np.ndarray  # (n_cosmologies,)
    ode_fn: object
    sparsity: object


_PREPARED_SOLVE_CACHE: dict = {}
_PACKED_BATCH_CACHE: dict = {}


def _stack_thermo_tables(tables):
    """Stack per-cosmology thermo tables for the multi-cosmology device evaluator.

    Returns ``(tau_min, inv_dtau, values, seconds, tau0)`` where ``values`` and
    ``seconds`` are ``(n_cosmologies, n_channels, n_grid)`` and the rest are
    ``(n_cosmologies,)``. The device spline evaluator selects a cosmology's row
    by the per-trajectory ``IX_COSMOLOGY`` tag.
    """

    tau_min = np.array([np.log(t[0][0]) for t in tables], dtype=np.float64)
    inv_dtau = np.array(
        [(t[0].shape[0] - 1) / (np.log(t[0][-1]) - np.log(t[0][0])) for t in tables],
        dtype=np.float64,
    )
    values = np.stack([t[1] for t in tables])
    seconds = np.stack([t[2] for t in tables])
    tau0 = np.array([t[3] for t in tables], dtype=np.float64)
    return tau_min, inv_dtau, values, seconds, tau0


def _layout_for(cosmology) -> PerturbationLayout:
    """The perturbation layout the module presets imply for a cosmology."""

    return PerturbationLayout.from_cosmology(
        cosmology,
        lmaxg=LMAX_G,
        lmaxpol=LMAX_POL,
        lmaxr=LMAX_NR,
        lmaxnu=LMAX_NU,
        nqmax=NQMAX,
    )


def _prepare_solve(cosmologies) -> _PreparedSolve:
    """Return (and cache) the compiled solve artifacts for a batch of cosmologies.

    The thermodynamics tables, the numba-CUDA right-hand side device function and
    the Jacobian sparsity pattern depend only on the cosmologies (and the
    module-level grid and quadrature presets), not on the wave modes, the save
    times, or the solver tolerances. They are also what triggers the *slow*
    numba-CUDA kernel compilation: the returned ``ode_fn`` closure and the
    sparsity pattern are the identity keys of modax's
    ``modax.rodas5P`` compiled-kernel cache, so
    reusing the same objects across solves of the same batch turns a repeated
    solve from a full recompile (tens of seconds) into a bare kernel launch.

    All cosmologies in a batch must share a perturbation layout (same dark-energy
    and massive-neutrino settings); one compiled kernel then solves them all,
    selecting each cosmology's thermodynamics table by the ``IX_COSMOLOGY`` tag.

    The cache is unbounded and holds device memory (one thermodynamics table set
    plus the compiled kernel per distinct batch); sweeping many batches
    accumulates GPU allocations. Call :func:`clear_prepared_solve_cache` to
    release them.
    """

    cosmologies = tuple(_as_cosmology(cosmology) for cosmology in cosmologies)
    # Include the presets that change the compiled artifacts, so a caller that
    # rebinds them (e.g. NQMAX) does not get a stale kernel.
    key = (
        cosmologies,
        NQMAX,
        LMAX_G,
        LMAX_POL,
        LMAX_NR,
        LMAX_NU,
        N_THERMO_GRID,
        TAU_START,
        TRAJECTORIES_PER_BLOCK,
    )
    cached = _PREPARED_SOLVE_CACHE.get(key)
    if cached is not None:
        return cached

    layout = _layout_for(cosmologies[0])
    for cosmology in cosmologies[1:]:
        if _layout_for(cosmology) != layout:
            raise ValueError(
                "all cosmologies in a batch must share a perturbation layout "
                "(same dark-energy / massive-neutrino settings)"
            )

    tables = build_thermo_tables_batch(cosmologies)
    tau_min, inv_dtau, values, seconds, tau0 = _stack_thermo_tables(tables)
    ode_fn = build_numba_rhs(layout, tau_min, inv_dtau, values, seconds)
    # The pattern is all modax needs: it colours it for the Enzyme sweeps, and
    # orders it with AMD, factorises it symbolically and compiles a sparse LU
    # and sparse triangular solves for that exact structure.
    prepared = _PreparedSolve(
        cosmologies,
        layout,
        tables,
        tau0,
        ode_fn,
        perturbation_sparsity(layout),
    )
    _PREPARED_SOLVE_CACHE[key] = prepared
    return prepared


def _pack_batch(prepared, k_values, tau_save=None):
    """Return (and cache) the packed ``(params, y0, t_span, order, k_sorted)``.

    Building the packed inputs means one :func:`make_initial_states` per cosmology
    -- i.e. ``n_cosmologies * n_k`` scalar :func:`adiabatic_initial_conditions`
    calls, tens of milliseconds for a large batch. That work depends only on the
    cosmologies and the wave modes (not the solver tolerances), so it is cached
    on ``(cosmologies, k, tau_save)``; a repeated solve of the same batch then
    skips straight to the kernel launch.

    ``tau_save`` is the shared grid of conformal times at which every trajectory
    is recorded (the CMB line-of-sight integration needs a dense one). It
    defaults to just the two endpoints, i.e. the final state only.
    """

    k_arr = np.ascontiguousarray(np.asarray(k_values, dtype=np.float64))
    tau_key = None if tau_save is None else np.asarray(tau_save, np.float64).tobytes()
    key = (prepared.cosmologies, k_arr.shape, k_arr.tobytes(), tau_key)
    cached = _PACKED_BATCH_CACHE.get(key)
    if cached is not None:
        return cached

    layout = prepared.layout
    n_cosmo = len(prepared.cosmologies)
    order = np.argsort(k_arr, kind="stable")
    k_sorted = k_arr[order]
    n_k = len(k_sorted)

    # Pack trajectories as row = k_idx * n_cosmo + cosmology_idx, so a block of
    # TRAJECTORIES_PER_BLOCK consecutive rows shares one wave mode when
    # n_cosmo >= TRAJECTORIES_PER_BLOCK: neighbouring threads then take near-identical
    # adaptive steps, minimizing warp divergence.
    n_traj = n_k * n_cosmo
    params = np.empty((n_traj, N_PARAM), dtype=np.float64)
    y0 = np.empty((n_traj, layout.nvar), dtype=np.float64)
    for ci, cosmology in enumerate(prepared.cosmologies):
        base = make_params(cosmology, k_sorted, float(prepared.tau0[ci]))
        base[:, IX_COSMOLOGY] = float(ci)
        params[ci::n_cosmo] = base
        y0[ci::n_cosmo] = make_initial_states(cosmology, k_sorted, layout)
    params = np.ascontiguousarray(params)
    y0 = np.ascontiguousarray(y0)

    # Each trajectory stops at its own tau0 (IX_TAU_END); the shared save grid
    # runs to the largest tau0 and the trailing save slot holds each trajectory's
    # final state.
    if tau_save is None:
        t_span = np.asarray((TAU_START, float(np.max(prepared.tau0))), dtype=np.float64)
    else:
        t_span = np.ascontiguousarray(np.asarray(tau_save, dtype=np.float64))

    packed = (params, y0, t_span, order, k_sorted)
    _PACKED_BATCH_CACHE[key] = packed
    return packed


def clear_prepared_solve_cache() -> None:
    """Drop all cached prepared and packed solves, releasing their device memory."""

    _PREPARED_SOLVE_CACHE.clear()
    _PACKED_BATCH_CACHE.clear()
    _THERMO_TABLE_CACHE.clear()


def solve_perturbation_history(
    k_values,
    cosmology,
    *,
    tau_save=None,
    as_jax: bool = False,
    rtol: float = PERTURB_RTOL,
    atol: float = PERTURB_ATOL,
    first_step: float = PERTURB_FIRST_STEP,
    max_steps: int = PERTURB_MAX_STEPS,
):
    """Solve the perturbation hierarchy and return the saved state history.

    Parameters
    ----------
    tau_save : array_like, optional
        Strictly increasing conformal times at which to record the state. The
        first entry must be ``TAU_START`` and the last ``tau0``. Defaults to just
        the two endpoints, i.e. the final state only. The line-of-sight CMB
        integration needs a dense grid through recombination instead.
    as_jax : bool, optional
        Return the history as a device-resident JAX array instead of copying it
        back to host NumPy. The solver's output already lives on the GPU (the
        numba kernel is invoked through a JAX FFI custom call), so this keeps the
        whole downstream pipeline -- e.g. the CMB source construction and
        line-of-sight projection -- on device with no host round-trip.

    Returns
    -------
    (history, layout, tables)
        ``history`` has shape ``(n_k, len(tau_save), nvar)`` in the caller's
        original ``k`` order; ``tables`` is the ``build_thermo_tables`` tuple.
    """

    from modax.rodas5P import solve as rodas5P_solve

    prepared = _prepare_solve((cosmology,))
    cosmology = prepared.cosmologies[0]
    layout = prepared.layout
    tau0 = float(prepared.tau0[0])

    k_arr = np.ascontiguousarray(np.asarray(k_values, dtype=np.float64))
    order = np.argsort(k_arr, kind="stable")
    k_sorted = k_arr[order]

    params = np.ascontiguousarray(make_params(cosmology, k_sorted, tau0))
    y0 = np.ascontiguousarray(make_initial_states(cosmology, k_sorted, layout))
    if tau_save is None:
        t_span = np.asarray((TAU_START, tau0), dtype=np.float64)
    else:
        t_span = np.ascontiguousarray(np.asarray(tau_save, dtype=np.float64))

    sol = rodas5P_solve(
        prepared.ode_fn,
        y0,
        t_span,
        params,
        sparsity=prepared.sparsity,
        lu_precision="fp32",
        rtol=rtol,
        atol=atol,
        first_step=first_step,
        max_steps=max_steps,
        pcoeff=0.3,
        icoeff=0.4,
        trajectories_per_block=TRAJECTORIES_PER_BLOCK,
        tf_index=IX_TAU_END,
        max_registers=CUDA_MAX_REGISTERS,
    )
    # The solver returns trajectories in ascending-k order; undo the sort. The
    # inverse permutation is a host-side constant, so the reorder is a plain
    # gather that works equally on device (JAX) or host (NumPy).
    inv_order = np.argsort(order)
    if as_jax:
        import jax.numpy as jnp

        return jnp.asarray(sol)[jnp.asarray(inv_order)], layout, prepared.tables[0]

    hist_sorted = np.asarray(sol)  # (n_k, n_save, nvar)
    return hist_sorted[inv_order], layout, prepared.tables[0]


def solve_perturbation_history_batch(
    k_values,
    cosmologies,
    *,
    tau_save,
    rtol: float = PERTURB_RTOL,
    atol: float = PERTURB_ATOL,
    first_step: float = PERTURB_FIRST_STEP,
    max_steps: int = PERTURB_MAX_STEPS,
):
    """Solve a batch of cosmologies on a shared ``tau_save`` grid in one launch.

    All ``n_cosmologies * n_k`` trajectories are integrated by a single
    modax Rodas5P launch. Each trajectory still stops at its *own* ``tau0``
    (via ``IX_TAU_END``), so save times beyond a cosmology's ``tau0`` simply hold
    its frozen final state -- consumers must mask them (as
    :mod:`discoeb.cmb` does through the opacity and the source channels).

    Returns
    -------
    (history, k_sorted, prepared)
        ``history`` is a device-resident JAX array of shape
        ``(n_k, n_cosmologies, len(tau_save), nvar)``, with the wave modes in
        **ascending** order (``k_sorted``).

        The history is deliberately left in this layout: it is the solver's own
        packing, so the reshape is a free view. Transposing to a cosmology-major
        layout, or gathering the wave modes back into the caller's order, would
        each need a full second copy of the array -- 6.5 GB for a 128-cosmology,
        1000-save-point batch, enough to exhaust a 12 GB GPU on its own. Ascending
        ``k`` is what the downstream spline and the ``dln k`` quadrature want
        anyway.
    """

    import jax.numpy as jnp

    from modax.rodas5P import solve as rodas5P_solve

    prepared = _prepare_solve(tuple(cosmologies))
    layout = prepared.layout
    n_cosmo = len(prepared.cosmologies)

    params, y0, t_span, order, k_sorted = _pack_batch(prepared, k_values, tau_save)
    n_k = len(k_sorted)

    sol = rodas5P_solve(
        prepared.ode_fn,
        y0,
        t_span,
        params,
        sparsity=prepared.sparsity,
        lu_precision="fp32",
        rtol=rtol,
        atol=atol,
        first_step=first_step,
        max_steps=max_steps,
        pcoeff=0.3,
        icoeff=0.4,
        trajectories_per_block=TRAJECTORIES_PER_BLOCK,
        tf_index=IX_TAU_END,
        max_registers=CUDA_MAX_REGISTERS,
    )
    # Rows are packed k-major (row = k_idx * n_cosmo + cosmology_idx), so this
    # reshape is a free view -- no copy of the (potentially multi-GB) history.
    hist = jnp.asarray(sol).reshape(n_k, n_cosmo, len(t_span), layout.nvar)
    return hist, k_sorted, prepared


def solve_perturbations(k_values, cosmology, **solve_kwargs):
    """Solve the perturbation hierarchy for each ``k`` and return final states.

    Returns
    -------
    np.ndarray
        Final perturbation state for each wave mode, shape ``(n_k, nvar)``,
        evaluated at ``tau0`` (today).
    """

    solve_kwargs.pop("tau_save", None)
    hist, _, _ = solve_perturbation_history(k_values, cosmology, **solve_kwargs)
    return hist[:, -1, :]


def solve_matter_power_spectrum(
    k_values,
    cosmology,
    **solve_kwargs,
) -> np.ndarray:
    """Return the linear matter power spectrum ``P(k)`` in Mpc^3 for ``cosmology``."""

    cosmology = _as_cosmology(cosmology)
    k_arr = np.ascontiguousarray(np.asarray(k_values, dtype=np.float64))
    y_final = solve_perturbations(k_arr, cosmology, **solve_kwargs)
    densities = density_coefficients(cosmology)
    grhoc_val, grhob_val = densities[2], densities[3]

    pk = np.empty(len(k_arr), dtype=np.float64)
    for i, k in enumerate(k_arr):
        delta_m = total_matter_density_contrast(
            grhoc_val, y_final[i, IX_CLXC], grhob_val, y_final[i, IX_CLXB]
        )
        pk[i] = matter_power_spectrum(
            float(k), delta_m, cosmology.A_s, cosmology.n_s, cosmology.k_pivot
        )
    return pk


PHYSICAL_DENSITY_COLUMNS = (0, 1, 2, 3, 4)
"""Packed-parameter columns carrying the background densities.

``(grhog, grhornomass, grhoc, grhob, grhov)`` -- the entries of a parameter row
that are physical quantities one would differentiate. The rest of the row is
the wave number, two integration bounds, two integer selectors and the
dark-energy and massive-neutrino settings, none of which a gradient with
respect to "the cosmology" wants, and two of which are not differentiable at
all.
"""


def matter_power_spectrum_jax(
    k_values,
    cosmology,
    densities=None,
    *,
    rtol: float = PERTURB_RTOL,
    atol: float = PERTURB_ATOL,
    first_step: float = PERTURB_FIRST_STEP,
    max_steps: int = PERTURB_MAX_STEPS,
    one_column_per_trajectory: bool = True,
):
    """``P(k)`` as a JAX array, differentiable in the background densities.

    ``densities`` is the five-vector of :data:`PHYSICAL_DENSITY_COLUMNS`,
    defaulting to the cosmology's own. Differentiating with respect to it runs
    modax's continuous forward-sensitivity system alongside the state. The
    densities enter twice -- through the hierarchy, and through the
    matter-weighting of ``delta_m`` -- and both paths are differentiated, so
    this is the total derivative and not just the ODE's part of it. What is
    held fixed is the thermodynamics: the recombination history is tabulated on
    the host and uploaded, so this is a derivative at fixed ionisation history.

    The same sensitivity system can be laid out over the threads two ways.

    ``one_column_per_trajectory=True`` (the default)
        Every wave mode is replicated once per density, and replica ``j``
        carries one sensitivity column, along the unit direction ``e_j``,
        through the ``IX_EPS`` / ``IX_DIR`` columns of its parameter row. A
        thread then integrates ``2 * nvar`` components instead of ``6 * nvar``,
        and the five columns run side by side on threads the launch would
        otherwise leave idle -- 128 modes are four blocks of 32 on a GPU with
        dozens of SMs. The state and the Jacobian are recomputed by each
        replica, which costs nothing while the card has SMs to spare. Measured
        on an RTX 4070 SUPER for the 128 test modes (``jax.value_and_grad``,
        ``test_matter_power_spectrum_derivatives``): 0.28 s for the plain
        solve, 1.92 s for five columns in one thread, 0.59 s for one column
        per thread. A backward adjoint solve would need at least the forward
        solve plus a second pass of comparable cost, so this is already at the
        adjoint's floor with no trajectory storage.

    ``one_column_per_trajectory=False``
        One trajectory per mode carrying all five columns
        (``sens_param_columns=PHYSICAL_DENSITY_COLUMNS``). Five times less
        thread-time in total, since the state and the Jacobian are formed once
        per mode rather than once per column, so it is the arrangement to
        prefer once a batch is large enough to saturate the GPU by itself.

    Either way the sensitivities are kept out of step-size control
    (``sens_error_control=False``): they run to ~1e17 while the state is O(1),
    so letting them into the weighted error norm hands them the step size and
    costs 20x the solve for no accuracy the tolerances asked for. The LU is
    fp64, unlike the plain solve: the Rosenbrock-W property lets an fp32
    factorisation carry the *state* at full order, but the sensitivity rows are
    driven by that same matrix and measure ~25% error from it, against ~5e-4
    in fp64.
    """
    import jax.numpy as jnp

    cosmology = _as_cosmology(cosmology)
    prepared = _prepare_solve((cosmology,))
    cosmology = prepared.cosmologies[0]

    history, densities, k_sorted, order = _history_jax(
        prepared,
        k_values,
        densities,
        rtol=rtol,
        atol=atol,
        first_step=first_step,
        max_steps=max_steps,
        one_column_per_trajectory=one_column_per_trajectory,
    )
    y_final = history[:, -1, :]

    grhoc_val = densities[2]
    grhob_val = densities[3]
    delta_m = (
        grhoc_val * y_final[:, IX_CLXC] + grhob_val * y_final[:, IX_CLXB]
    ) / (grhoc_val + grhob_val)
    ks = jnp.asarray(k_sorted)
    pk_sorted = (
        2.0
        * jnp.pi**2
        / ks**3
        * cosmology.A_s
        * (ks / cosmology.k_pivot) ** (cosmology.n_s - 1.0)
        * delta_m**2
    )
    return pk_sorted[jnp.asarray(np.argsort(order))]


def _history_jax(
    prepared,
    k_values,
    densities=None,
    tau_save=None,
    *,
    rtol: float = PERTURB_RTOL,
    atol: float = PERTURB_ATOL,
    first_step: float = PERTURB_FIRST_STEP,
    max_steps: int = PERTURB_MAX_STEPS,
    one_column_per_trajectory: bool = True,
    sens_error_control: bool = False,
):
    """The saved history of one cosmology's modes, differentiable in the densities.

    The solve behind :func:`matter_power_spectrum_jax` and
    :func:`discoeb.cmb.cl_power_spectrum_jax`, which documents the two thread
    layouts (``one_column_per_trajectory``) and the solver settings. ``prepared``
    is the :func:`_prepare_solve` of the single cosmology; ``tau_save`` is the
    save grid (default: the two endpoints, i.e. the final state only).

    Returns ``(history, densities, k_sorted, order)``: ``history`` is
    ``(n_k, n_save, nvar)`` in ascending ``k`` as a JAX array whose derivative
    with respect to ``densities`` is modax's forward sensitivity; ``densities``
    is the resolved five-vector; ``k_sorted`` and ``order`` are the sorted wave
    numbers and the sort of the caller's ``k_values``.
    """
    import jax
    import jax.numpy as jnp

    from modax.rodas5P import solve as rodas5P_solve

    params, y0, t_span, order, k_sorted = _pack_batch(prepared, k_values, tau_save)
    columns = list(PHYSICAL_DENSITY_COLUMNS)
    n_dir = len(columns)
    n_k = params.shape[0]
    if densities is None:
        densities = params[0, columns]
    densities = jnp.asarray(densities)

    settings = dict(
        sparsity=prepared.sparsity,
        lu_precision="fp64",
        rtol=rtol,
        atol=atol,
        first_step=first_step,
        max_steps=max_steps,
        pcoeff=0.3,
        icoeff=0.4,
        trajectories_per_block=TRAJECTORIES_PER_BLOCK,
        tf_index=IX_TAU_END,
        max_registers=CUDA_MAX_REGISTERS,
        sens_error_control=sens_error_control,
    )
    t_span_j = jnp.asarray(t_span)

    def with_densities(base, d):
        return jnp.asarray(base).at[:, jnp.asarray(columns)].set(
            jnp.broadcast_to(d, (base.shape[0], n_dir))
        )

    if not one_column_per_trajectory:
        history = rodas5P_solve(
            prepared.ode_fn,
            jnp.asarray(y0),
            t_span_j,
            with_densities(params, densities),
            sens_param_columns=PHYSICAL_DENSITY_COLUMNS,
            **settings,
        )
    else:
        # Replica j of mode i is row i * n_dir + j, so a block of consecutive
        # rows still spans neighbouring wave modes and their similar step
        # sequences; each replica's direction is the unit vector e_j.
        params_rep = np.repeat(params, n_dir, axis=0)
        params_rep[:, IX_DIR : IX_DIR + n_dir] = np.tile(np.eye(n_dir), (n_k, 1))
        y0_rep = jnp.asarray(np.repeat(y0, n_dir, axis=0))
        y0_j = jnp.asarray(y0)

        @jax.custom_jvp
        def saved_history(d):
            return rodas5P_solve(
                prepared.ode_fn, y0_j, t_span_j, with_densities(params, d), **settings
            )

        @saved_history.defjvp
        def saved_history_jvp(primals, tangents):
            (d,), (dd,) = primals, tangents
            p_rep = with_densities(params_rep, d)

            def along_directions(eps):
                return rodas5P_solve(
                    prepared.ode_fn,
                    y0_rep,
                    t_span_j,
                    p_rep.at[:, IX_EPS].set(eps),
                    sens_param_columns=(IX_EPS,),
                    **settings,
                )

            # d/d(eps) at eps = 0 along every replica at once: one joint launch
            # of n_k * n_dir trajectories, each carrying its own single column.
            n_rep = p_rep.shape[0]
            sol, dsol = jax.jvp(
                along_directions, (jnp.zeros(n_rep),), (jnp.ones(n_rep),)
            )
            # Every replica of a mode integrates the same state; keep the first.
            history = sol[::n_dir]
            # sens[i, j] = d y_i(tau) / d density_j at every save. The
            # contraction with the input tangent is linear, so JAX transposes
            # it and jax.grad falls out of the same rule with no adjoint solve.
            sens = dsol.reshape(n_k, n_dir, *dsol.shape[1:])
            return history, jnp.einsum("ijtv,j->itv", sens, dd)

        history = saved_history(densities)

    return history, densities, k_sorted, order



def solve_matter_power_spectrum_batch(
    k_values,
    cosmologies,
    *,
    rtol: float = PERTURB_RTOL,
    atol: float = PERTURB_ATOL,
    first_step: float = PERTURB_FIRST_STEP,
    max_steps: int = PERTURB_MAX_STEPS,
) -> np.ndarray:
    """Return ``P(k)`` for a batch of cosmologies in one GPU solve.

    All cosmologies must share a perturbation layout. The ``n_cosmologies * n_k``
    trajectories are integrated by a single modax Rodas5P launch, so the whole
    batch amortizes one kernel compilation and one set of device buffers.

    Returns
    -------
    np.ndarray
        ``P(k)`` in Mpc^3, shape ``(n_cosmologies, n_k)``, in the caller's
        original cosmology and ``k`` order.
    """

    from modax.rodas5P import solve as rodas5P_solve

    prepared = _prepare_solve(tuple(cosmologies))
    cosmologies = prepared.cosmologies
    layout = prepared.layout
    n_cosmo = len(cosmologies)

    params, y0, t_span, order, k_sorted = _pack_batch(prepared, k_values)
    n_k = len(k_sorted)

    sol = rodas5P_solve(
        prepared.ode_fn,
        y0,
        t_span,
        params,
        sparsity=prepared.sparsity,
        lu_precision="fp32",
        rtol=rtol,
        atol=atol,
        first_step=first_step,
        max_steps=max_steps,
        pcoeff=0.3,
        icoeff=0.4,
        trajectories_per_block=TRAJECTORIES_PER_BLOCK,
        tf_index=IX_TAU_END,
        max_registers=CUDA_MAX_REGISTERS,
    )
    y_final = np.asarray(sol)[:, -1, :].reshape(n_k, n_cosmo, layout.nvar)

    pk_sorted = np.empty((n_cosmo, n_k), dtype=np.float64)
    for ci, cosmology in enumerate(cosmologies):
        densities = density_coefficients(cosmology)
        grhoc_val, grhob_val = densities[2], densities[3]
        for ki in range(n_k):
            yv = y_final[ki, ci]
            delta_m = total_matter_density_contrast(
                grhoc_val, yv[IX_CLXC], grhob_val, yv[IX_CLXB]
            )
            pk_sorted[ci, ki] = matter_power_spectrum(
                float(k_sorted[ki]), delta_m, cosmology.A_s, cosmology.n_s, cosmology.k_pivot
            )

    pk = np.empty_like(pk_sorted)
    pk[:, order] = pk_sorted
    return pk
