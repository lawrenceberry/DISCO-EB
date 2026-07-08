"""Scalar Einstein-Boltzmann perturbation equations (synchronous gauge).

Ported from the DISCO2 prototype. This module defines the linear cosmological
perturbation right-hand side, adiabatic initial conditions, and the Einstein
constraint terms only. Like :mod:`discoeb.recfast`, it deliberately avoids solver
wrappers, grid construction, and line-of-sight integration so the hot equations
can be compiled to numba-cuda with scalar arguments. The matter power-spectrum
mapping lives in :mod:`discoeb.matter_power_spectrum`; the ODE solve lives in
:mod:`discoeb.perturbations_system`.

Conventions follow CAMB and :mod:`discoeb.background`:

    * Distances and conformal time are in Mpc with ``c = 1``.
    * Wavenumbers ``k`` are in Mpc^-1.
    * Densities use the CAMB ``grho`` convention. The functions here work with
      ``grho_i a^2 = 8*pi*G*rho_i a^2`` ("comoving" density coefficients), because
      the synchronous-gauge Einstein equations naturally pair ``8*pi*G*rho_i a^2``
      with the dimensionless density contrasts.
    * The state evolves in conformal time ``tau``; ``adotoa = a'/a = aH`` is the
      conformal Hubble rate.

The perturbation variables follow Ma & Bertschinger / CAMB:

    * ``etak = k eta`` -- synchronous-gauge metric perturbation,
    * ``clxc``, ``clxb`` -- CDM and baryon density contrasts,
    * ``vb`` -- baryon velocity,
    * ``Theta_l`` -- photon temperature multipoles (``clxg = Theta_0``,
      ``qg = Theta_1``, ``pig = Theta_2``),
    * ``E_l`` -- photon E-mode polarization multipoles,
    * ``N_l`` -- massless-neutrino multipoles (``clxr = N_0``, ``qr = N_1``,
      ``pir = N_2``).

The state-vector layout is fixed at import time from the truncation constants
:data:`LMAX_G`, :data:`LMAX_POL`, :data:`LMAX_NR` below, so the index helpers are
compile-time constants. These are the knobs that a numba-cuda kernel specializes
on; changing them re-specializes the layout and the (matching) block-LU solver.
"""

import numpy as np

# Hierarchy truncations. These are module-level integers (not runtime arguments)
# so the state-vector layout and the accelerator-compiled right-hand side treat
# them as compile-time constants. The current flat-LambdaCDM + massless-neutrino
# layout has NVAR == 50 (11-var dense core + 3x13 free-streaming hierarchies),
# matching the hand-tuned Schur-EB LU solver in :mod:`discoeb.schur_eb`.
LMAX_G = 15
"""Photon temperature hierarchy truncation (``Theta_0 ... Theta_LMAX_G``)."""

LMAX_POL = 15
"""Photon E-mode polarization hierarchy truncation (``E_2 ... E_LMAX_POL``)."""

LMAX_NR = 15
"""Massless-neutrino hierarchy truncation (``N_0 ... N_LMAX_NR``)."""

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
