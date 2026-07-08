"""Cosmological background evolution.

This module computes the homogeneous expansion history that the rest of
DISCO-EB is built on: the Hubble rate ``H(a)``, the conformal time ``tau(a)``,
the neutrino background density, and the recombination-related optical depth
and visibility functions.

Conventions
-----------
    * Distances and conformal time are measured in Mpc with ``c = 1``.
    * Hubble rates are measured in inverse Mpc.
    * Density-like quantities use the CAMB ``grho`` convention:
      ``grhoa4 = 8*pi*G*rho_total * a^4`` (a small polynomial in ``a`` for the
      radiation/matter/Lambda terms), and ``grhoa2 = 8*pi*G*rho_total * a^2``,
      which equals ``3*(a'/a)^2`` where ``a' = da/dtau``. The two differ only by
      a factor ``a^2``; ``grhoa4`` is the natural quantity for conformal-time
      evolution because ``a' = da/dtau = sqrt(grhoa4/3)``.

Structure
---------
The module is split into two layers:

1. **General background model** -- small, dependency-free *standard-Python*
   functions (no JAX) that take explicit, well-defined arguments and cover the
   full DISCO-EB model: radiation, matter, dynamical dark energy
   (``w_0``/``w_a``), spatial curvature, and massive neutrinos. The
   flat-LambdaCDM limit is recovered with the default keyword arguments, and is
   unit-tested against CLASS. These functions use only Python operators and the
   :mod:`math` module, so they are readable, portable, and easy to reason about.

2. **Compatibility layer** -- the historical ``param``-dictionary API
   (``evolve_background``, ``get_aprimeoa``, ``nu_background``, ...) built on
   JAX for the differentiable pipeline used by the thermodynamics, perturbation,
   and CMB modules. It assembles the ``grho`` coefficients from ``param`` and
   evaluates the general-model functions; because those use only Python
   operators, they are equally valid on JAX arrays under ``jit``/``vmap``/``grad``.
"""

import math
from functools import partial

import jax
import jax.numpy as jnp

from .constants import (
    C_KM_S,
    FERMI_DIRAC_CONST,
    GRHO_CRITICAL_H2,
    M_H,
    M_HE4_OVER_M_H,
    MPC_IN_M,
    NEUTRINO_MASS_KELVIN_PER_EV,
    RHO_CRIT_100_SI,
    SIGMA_SB,
    SIGMA_T,
)
from .thermodynamics_recfast import evaluate_thermo as evaluate_thermo_recfast
from .thermodynamics_mb95 import compute_thermo as compute_thermo_mb95

from .spline_interpolation import spline_interpolation
from .util import generalized_gauss_laguerre_weights, integrate_trapz


# =============================================================================
# 1. General background model (standard Python, no JAX)
#
# A self-contained scalar description of the homogeneous background covering
# radiation, matter, dynamical dark energy, spatial curvature, and massive
# neutrinos. The flat-LambdaCDM limit is recovered by leaving the optional
# keyword arguments (``grhok``, ``grhomnu``, ``rhonu``, ``rho_de``) at their
# defaults. Everything here is plain Python (operators + :mod:`math`).
# =============================================================================


def hubble_constant_mpc(h: float) -> float:
    """Return ``H0`` in Mpc^-1 from the dimensionless Hubble parameter ``h``.

    Cosmology inputs usually quote ``H0 = 100*h km/s/Mpc``. With ``c = 1`` and
    distances in Mpc, the code stores

    ``H0 = 100*h / c_km_s``.
    """

    return 100.0 * h / C_KM_S


def omega_gamma_h2(T_cmb: float) -> float:
    """Return the physical photon density ``Omega_gamma h^2``.

    Black-body radiation has energy density ``rho_gamma = 4*sigma_SB T^4 / c^3``
    in the SI-compatible units used here. Dividing by the constant
    ``rho_crit,100`` gives ``Omega_gamma h^2``.
    """

    rho_gamma = 4.0 * SIGMA_SB / (C_KM_S * 1.0e3) ** 3 * T_cmb**4
    return rho_gamma / RHO_CRIT_100_SI


def radiation_neutrino_factor(N_eff: float) -> float:
    """Return the massless-neutrino/photon density ratio.

    For relativistic neutrinos after electron-positron annihilation,

    ``rho_nu / rho_gamma = (7/8) * (4/11)^(4/3) * N_eff``.
    """

    return 7.0 / 8.0 * (4.0 / 11.0) ** (4.0 / 3.0) * N_eff


# --- Density coefficients in CAMB ``grho`` units (1/Mpc^2) -------------------


def critical_density_grho(H0: float) -> float:
    """Return the critical-density coefficient ``grhom`` in CAMB ``grho`` units.

    This normalizes the matter, dark-energy, and curvature contributions to the
    Friedmann equation. With ``H0 = 100*h km/s/Mpc``,

    ``grhom = 8*pi*G*rho_crit,0 / c^2 = 3*(H0/c)^2 = GRHO_CRITICAL_H2 * h^2``.

    Numerically it reproduces the historical DISCO-EB literal
    ``3.33795017e-11 * H0^2`` to ~1e-9 relative.
    """

    return GRHO_CRITICAL_H2 * (H0 / 100.0) ** 2


def grhog(T_cmb: float) -> float:
    """Return the photon density coefficient in CAMB ``grho`` units.

    This is the photon contribution to

    ``grhoa4(a) = grhog + grhornomass + (grhoc + grhob) a + grhov a^4``.

    Photons scale as ``rho_gamma ~ a^-4``, so after multiplying
    ``8*pi*G*rho_total`` by ``a^4`` this coefficient is independent of ``a``.
    Numerically it reproduces the historical DISCO-EB literal
    ``1.49594245e-13 * T_cmb^4`` to ~1e-9 relative.
    """

    return GRHO_CRITICAL_H2 * omega_gamma_h2(T_cmb)


def grhornomass(grhog: float, N_eff: float) -> float:
    """Return the massless-neutrino density coefficient in CAMB ``grho`` units.

    Massless neutrinos also scale as ``rho_nu ~ a^-4``. Their coefficient is the
    photon coefficient multiplied by

    ``rho_nu / rho_gamma = (7/8) * (4/11)^(4/3) * N_eff``.
    """

    return grhog * radiation_neutrino_factor(N_eff)


def neutrino_density_grho(T_cmb: float) -> float:
    """Return the per-flavour neutrino density coefficient ``grhor``.

    A single relativistic neutrino flavour contributes ``(7/8)*(4/11)^(4/3)``
    times the photon density, so

    ``grhor = grhog * (7/8) * (4/11)^(4/3) = grhornomass(grhog, N_eff=1)``.
    """

    return grhog(T_cmb) * radiation_neutrino_factor(1.0)


def grhoc(omega_c_h2: float) -> float:
    """Return the cold-dark-matter density coefficient in CAMB ``grho`` units.

    Cold dark matter scales as ``rho_c ~ a^-3``, so its contribution to
    ``grhoa4`` is ``grhoc*a``.
    """

    return GRHO_CRITICAL_H2 * omega_c_h2


def grhob(omega_b_h2: float) -> float:
    """Return the baryon density coefficient in CAMB ``grho`` units.

    Baryons scale as ``rho_b ~ a^-3``, so their contribution to ``grhoa4`` is
    ``grhob*a``.
    """

    return GRHO_CRITICAL_H2 * omega_b_h2


def grhov(
    omega_b_h2: float,
    omega_c_h2: float,
    h: float,
    N_eff: float,
    T_cmb: float,
) -> float:
    """Return the cosmological-constant coefficient in CAMB ``grho`` units.

    For flat LCDM, the vacuum density is set by flatness:

    ``Omega_Lambda = 1 - Omega_m - Omega_r``.

    Since ``rho_Lambda ~ a^0``, its contribution to ``grhoa4`` is ``grhov*a^4``.
    For dynamical dark energy this coefficient is instead scaled by the density
    ratio :func:`dark_energy_density_ratio` (see the ``rho_de`` argument of
    :func:`grhoa4`).
    """

    omega_gamma = omega_gamma_h2(T_cmb)
    omega_m = (omega_c_h2 + omega_b_h2) / h**2
    omega_r = omega_gamma * (1.0 + radiation_neutrino_factor(N_eff)) / h**2
    return GRHO_CRITICAL_H2 * (1.0 - omega_m - omega_r) * h**2


def radiation_hubble_rate(
    grhog: float, grhor: float, N_eff: float, N_mnu: float
) -> float:
    """Return the conformal Hubble rate ``a'/a`` deep in radiation domination.

    In the radiation era all species are relativistic and ``grhoa4`` is
    dominated by the constant photon + neutrino term ``grhog + grhor*(N_eff +
    N_mnu)``. The scale factor then grows as ``a ~ adotrad * tau`` with

    ``adotrad = sqrt((grhog + grhor*(N_eff + N_mnu)) / 3)``.
    """

    return ((grhog + grhor * (N_eff + N_mnu)) / 3.0) ** 0.5


# --- Dark energy (CPL parametrization) --------------------------------------


def dark_energy_density_ratio(a, w_DE_0, w_DE_a):
    """Return the dark-energy density normalized to its present value.

    For the CPL parametrization ``w(a) = w_0 + w_a (1 - a)`` the continuity
    equation integrates to

    ``rho_Q(a) / rho_Q(1) = a^{-3(1 + w_0 + w_a)} exp[3(a - 1) w_a]``.

    For a cosmological constant (``w_0 = -1``, ``w_a = 0``) this reduces to
    ``1``.
    """

    return a ** (-3.0 * (1.0 + w_DE_0 + w_DE_a)) * math.exp(3.0 * (a - 1.0) * w_DE_a)


def dark_energy_equation_of_state(a, w_DE_0, w_DE_a):
    """Return the dark-energy equation of state ``w(a) = w_0 + w_a (1 - a)``."""

    return w_DE_0 + w_DE_a * (1.0 - a)


# --- Massive neutrinos ------------------------------------------------------


def massive_neutrino_density(a, amnu, q, w):
    """Return the density and pressure ratios of one massive-neutrino flavour.

    Evaluates, for a single scale factor ``a`` and mass parameter ``amnu``, the
    momentum integrals of one massive-neutrino flavour in units of the mean
    density of one massless flavour. With the dimensionless velocity

    ``v(q) = 1 / sqrt(1 + (a amnu / q)^2)``,

    the density, pressure, and pseudo-pressure ratios are

    ``rho = sum_i w_i / v(q_i)``, ``p = sum_i w_i v(q_i) / 3``,
    ``pp = sum_i w_i v(q_i)^3 / 3``.

    Args:
        a: scale factor (scalar).
        amnu: neutrino mass in units of the neutrino temperature (scalar).
        q, w: sequences of momentum bins and weights (e.g. from
            :func:`get_neutrino_momentum_bins`).

    Returns:
        tuple[float, float, float]: ``rho_nu/rho_nu0``, ``p_nu/p_nu0``,
        ``pp_nu/pp_nu0``.
    """

    rhonu = 0.0
    pnu = 0.0
    ppnu = 0.0
    for qi, wi in zip(q, w):
        v = 1.0 / (1.0 + (a * amnu / qi) ** 2) ** 0.5  # = (1/aq)/sqrt(1+1/aq**2)
        rhonu += wi / v
        pnu += wi * v / 3.0
        ppnu += wi * v**3 / 3.0
    return rhonu, pnu, ppnu


# --- Friedmann quantities (full model) --------------------------------------


def grhoa4(
    a,
    grhog,
    grhornomass,
    grhoc,
    grhob,
    grhov,
    *,
    grhok=0.0,
    grhomnu=0.0,
    rhonu=1.0,
    rho_de=1.0,
):
    """Return ``8*pi*G*rho_total(a) a^4`` for the general DISCO-EB model.

    Grouping species by their scaling with ``a`` (after the overall ``a^4``
    factor),

    ``grhoa4 = grhog + grhornomass + grhomnu * rhonu   (radiation + neutrinos)``
    ``       + (grhoc + grhob) a                        (matter,  rho ~ a^-3)``
    ``       + grhok a^2                                (curvature, rho ~ a^-2)``
    ``       + grhov rho_de a^4                         (dark energy)``.

    The required positional arguments reproduce the flat-``LambdaCDM``
    polynomial ``grhog + grhornomass + (grhoc + grhob) a + grhov a^4``. The full
    model is switched on through the keyword arguments:

    * ``grhok = grhom * Omega_k`` -- spatial curvature.
    * ``grhomnu = grhor * N_mnu`` with ``rhonu = rho_massive_nu(a) /
      rho_massless_nu(a)`` from :func:`massive_neutrino_density` -- massive
      neutrinos. Their relativistic contribution is already carried by
      ``grhornomass``; ``grhomnu * rhonu`` adds the massive flavours.
    * ``rho_de = rho_Q(a) / rho_Q(1)`` from :func:`dark_energy_density_ratio` --
      dynamical dark energy (``rho_de = 1`` for a cosmological constant).

    The conformal expansion factor obeys ``a' = da/dtau = sqrt(grhoa4 / 3)``.
    """

    return (
        grhog + grhornomass + grhomnu * rhonu
        + (grhoc + grhob) * a
        + grhok * a**2
        + grhov * rho_de * a**4
    )


def hubble_a(
    a,
    grhog,
    grhornomass,
    grhoc,
    grhob,
    grhov,
    *,
    grhok=0.0,
    grhomnu=0.0,
    rhonu=1.0,
    rho_de=1.0,
):
    """Return the physical Hubble rate ``H(a)`` in Mpc^-1.

    The Friedmann equation is ``H(a)^2 = 8*pi*G*rho_total / 3``. Because
    ``grhoa4 = 8*pi*G*rho_total a^4``, this becomes

    ``H(a) = sqrt(grhoa4 / 3) / a^2``.

    Accepts the same arguments as :func:`grhoa4`.
    """

    return (
        grhoa4(a, grhog, grhornomass, grhoc, grhob, grhov,
               grhok=grhok, grhomnu=grhomnu, rhonu=rhonu, rho_de=rho_de) / 3.0
    ) ** 0.5 / a**2


def conformal_hubble(
    a,
    grhog,
    grhornomass,
    grhoc,
    grhob,
    grhov,
    *,
    grhok=0.0,
    grhomnu=0.0,
    rhonu=1.0,
    rho_de=1.0,
):
    """Return the conformal Hubble rate ``a'/a = a H(a)`` in Mpc^-1.

    With ``grhoa2 = grhoa4 / a^2 = 8*pi*G*rho_total a^2`` the Friedmann equation
    is ``(a'/a)^2 = grhoa2 / 3``, so

    ``a'/a = sqrt(grhoa4 / 3) / a``.

    Accepts the same arguments as :func:`grhoa4`.
    """

    return (
        grhoa4(a, grhog, grhornomass, grhoc, grhob, grhov,
               grhok=grhok, grhomnu=grhomnu, rhonu=rhonu, rho_de=rho_de) / 3.0
    ) ** 0.5 / a


def dtau_da(
    a,
    grhog,
    grhornomass,
    grhoc,
    grhob,
    grhov,
    *,
    grhok=0.0,
    grhomnu=0.0,
    rhonu=1.0,
    rho_de=1.0,
):
    """Return the conformal-time derivative ``dtau/da``.

    With ``c = 1``, conformal time obeys ``dtau/da = 1 / (a^2 H(a))``. Since
    ``H(a) = sqrt(grhoa4 / 3) / a^2``, the derivative simplifies to

    ``dtau/da = sqrt(3 / grhoa4)``.

    Accepts the same arguments as :func:`grhoa4`.
    """

    return (
        3.0
        / grhoa4(a, grhog, grhornomass, grhoc, grhob, grhov,
                 grhok=grhok, grhomnu=grhomnu, rhonu=rhonu, rho_de=rho_de)
    ) ** 0.5


def hubble_z(
    z,
    grhog,
    grhornomass,
    grhoc,
    grhob,
    grhov,
    *,
    grhok=0.0,
    grhomnu=0.0,
    rhonu=1.0,
    rho_de=1.0,
):
    """Return the physical Hubble rate ``H(z)`` in Mpc^-1.

    This evaluates :func:`hubble_a` at ``a = 1 / (1 + z)``.
    """

    return hubble_a(
        1.0 / (1.0 + z), grhog, grhornomass, grhoc, grhob, grhov,
        grhok=grhok, grhomnu=grhomnu, rhonu=rhonu, rho_de=rho_de,
    )


def dtau_dz(
    z,
    grhog,
    grhornomass,
    grhoc,
    grhob,
    grhov,
    *,
    grhok=0.0,
    grhomnu=0.0,
    rhonu=1.0,
    rho_de=1.0,
):
    """Return the conformal-time derivative ``dtau/dz``.

    We already have ``dtau/da`` from :func:`dtau_da`; the redshift derivative
    follows from the chain rule ``dtau/dz = (dtau/da) (da/dz)``.

    Redshift and scale factor are related by ``a = 1 / (1 + z)``. Differentiating
    with respect to ``z`` gives the Jacobian of the change of variable,

    ``da/dz = d/dz [ (1 + z)^-1 ] = -(1 + z)^-2 = -1 / (1 + z)^2``.

    The sign is negative because increasing redshift corresponds to decreasing
    scale factor (looking further back in time). Substituting ``a = 1/(1+z)``
    into ``dtau/da = sqrt(3 / grhoa4(a))`` and multiplying by ``da/dz`` gives

    ``dtau/dz = (dtau/da) (da/dz) = -sqrt(3 / grhoa4(a)) / (1 + z)^2``,

    so ``dtau/dz`` is negative (conformal time decreases with increasing
    redshift). In code this is exactly ``dtau_da(a) * da/dz`` evaluated at
    ``a = 1/(1+z)``.
    """

    a = 1.0 / (1.0 + z)
    return -dtau_da(
        a, grhog, grhornomass, grhoc, grhob, grhov,
        grhok=grhok, grhomnu=grhomnu, rhonu=rhonu, rho_de=rho_de,
    ) / (1.0 + z) ** 2


# --- Recombination-related normalizations -----------------------------------


def neutrino_mass_parameter(mnu: float, T_cmb: float) -> float:
    """Return the dimensionless neutrino mass parameter ``amnu``.

    ``amnu = m_nu c^2 / (k_B T_nu0)`` with ``T_nu0 = (4/11)^(1/3) T_cmb`` the
    neutrino temperature today. Using the packaged conversion constant,

    ``amnu = m_nu[eV] * NEUTRINO_MASS_KELVIN_PER_EV / T_cmb[K]``.
    """

    return mnu * NEUTRINO_MASS_KELVIN_PER_EV / T_cmb


def thomson_normalization(omega_b_h2: float, Y_He: float) -> float:
    """Return the Thomson opacity normalization ``akthom``.

    The name ``akthom`` is inherited from CAMB-style Boltzmann-code naming. It
    packages the present-day hydrogen number density and Thomson cross section
    in Mpc units:

    ``akthom = sigma_T n_H,0``.

    Given the free-electron fraction ``x_e``, the conformal-time opacity rate is
    later evaluated as ``kappa_dot = x_e akthom / a^2``. This reproduces the
    historical DISCO-EB literal
    ``2.3038921003709498e-9 * (1 - Y_He) * Omega_b * H0^2`` to ~1e-10 relative.
    """

    rho_b_SI = omega_b_h2 * RHO_CRIT_100_SI
    n_H_Mpc = (1.0 - Y_He) * rho_b_SI / M_H * MPC_IN_M**3
    return (SIGMA_T / MPC_IN_M**2) * n_H_Mpc


def helium_number_fraction(Y_He: float) -> float:
    """Return helium abundance by number relative to hydrogen."""

    return Y_He / (M_HE4_OVER_M_H * (1.0 - Y_He))


# =============================================================================
# 2. Compatibility layer (JAX pipeline, historical ``param``-dictionary API)
#
# These functions carry the differentiable (jit/vmap/grad) pipeline used by the
# thermodynamics, perturbation, and CMB modules. They assemble the ``grho``
# coefficients from ``param`` and evaluate the general-model functions above
# (which, using only Python operators, are equally valid on JAX arrays). The
# only genuinely JAX-specific pieces are the neutrino momentum quadrature and
# the dynamical-dark-energy exponential, provided here as JAX helpers.
# =============================================================================


def _dark_energy_density_ratio_jax(a, w_DE_0, w_DE_a):
    """JAX/array version of :func:`dark_energy_density_ratio` (uses ``jnp.exp``)."""

    return a ** (-3.0 * (1.0 + w_DE_0 + w_DE_a)) * jnp.exp(3.0 * (a - 1.0) * w_DE_a)


def get_neutrino_momentum_bins(nqmax: int) -> tuple[jax.Array, jax.Array]:
    """Get the momentum bins and integral kernel weights for neutrinos.

    The neutrino momentum integrals are evaluated as sums over ``nqmax`` bins,
    ``integral f(q) dq ~ sum_i w_i f(q_i)``, with the weights normalized so that
    a single massless flavour integrates to unit density.

    Args:
        nqmax (int): Number of momentum bins.

    Returns:
        tuple[jax.Array, jax.Array]: comoving momenta ``q`` (in units of
        ``k_B T_nu0 / c``) and their integration weights ``w``.
    """

    # nqmax = 3,4,5 are from high accuracy formulas from CAMB, higher values resort to modified Gauss-Laguerre,
    # which is not pre-computed however
    if nqmax == 3:
        q = jnp.array([0.913201, 3.37517, 7.79184])
        dlfdlq = -q / (1 + jnp.exp(-q))
        w = jnp.array([0.0687359, 3.31435, 2.29911]) / (-0.25 * dlfdlq)
    elif nqmax == 4:
        q = jnp.array([0.7, 2.62814, 5.90428, 12.0])
        dlfdlq = -q / (1 + jnp.exp(-q))
        w = jnp.array([0.0200251, 1.84539, 3.52736, 0.289427]) / (-0.25 * dlfdlq)
    elif nqmax == 5:
        q = jnp.array([0.583165, 2.0, 4.0, 7.26582, 13.0])
        dlfdlq = -q / (1 + jnp.exp(-q))
        w = jnp.array([0.0081201, 0.689407, 2.8063, 2.05156, 0.12681]) / (
            -0.25 * dlfdlq
        )
    else:
        alpha = 1
        q, w = generalized_gauss_laguerre_weights(nqmax, alpha)
        w *= q**3 / (1 + jnp.exp(-q)) * q**-alpha

    return q, w / FERMI_DIRAC_CONST


def nu_background(a, amnu, nq: int = 8):
    """Compute the neutrino density and pressure of one massive flavour.

    JAX-vectorized counterpart of :func:`massive_neutrino_density`: broadcasts
    over a batch of scale factors and neutrino masses. Results are in units of
    the mean density of one flavour of massless neutrinos.

    Args:
        a (N,): scale factor.
        amnu (B,): neutrino mass in units of neutrino temperature
            (``m_nu c^2 / (k_B T_nu0)``).
        nq (int, optional): number of integration points. Defaults to 8.

    Returns:
        tuple[jax.Array, jax.Array, jax.Array]: ``rho_nu/rho_nu0``,
        ``p_nu/p_nu0``, ``pp_nu/pp_nu0``, each of shape ``(N, B)``.
    """

    q, w = get_neutrino_momentum_bins(nq)

    a = a[:, None, None]
    amnu = amnu[None, :, None]
    q = q[None, None, :]
    w = w[None, None, :]

    v = 1.0 / jnp.sqrt(1.0 + (a * amnu / q) ** 2)  # = (1/aq)/sqrt(1+1/aq**2)
    rhonu = jnp.sum(w / v, axis=2)
    pnu = jnp.sum(w * v / 3.0, axis=2)
    ppnu = jnp.sum(w * v**3 / 3.0, axis=2)
    return rhonu, pnu, ppnu


def dtauda_(
    a,
    grhom,
    grhog,
    grhor,
    Omegam,
    OmegaDE,
    w_DE_0,
    w_DE_a,
    Omegak,
    Neff,
    Nmnu,
    logrhonu_spline,
):
    """Derivative of conformal time with respect to scale factor, ``dtau/da``.

    Scalar convenience wrapper: evaluates the massive-neutrino and dark-energy
    density ratios from the supplied spline / equation-of-state parameters, then
    returns ``dtau/da = sqrt(3 / grhoa4)`` via :func:`grhoa4`.
    """

    rhonu = jnp.exp(logrhonu_spline.evaluate(jnp.log(a)))
    rho_de = _dark_energy_density_ratio_jax(a, w_DE_0, w_DE_a)
    g4 = grhoa4(
        a,
        grhog,
        grhor * Neff,
        grhom * Omegam,
        0.0,
        grhom * OmegaDE,
        grhok=grhom * Omegak,
        grhomnu=grhor * Nmnu,
        rhonu=rhonu,
        rho_de=rho_de,
    )
    return (3.0 / g4) ** 0.5


def dadtau(a, param):
    """Derivative of scale factor with respect to conformal time, ``a' = da/dtau``.

    Batched over the cosmology dimension stored in ``param``; assembles the
    ``grho`` coefficients from ``param`` and evaluates ``sqrt(grhoa4 / 3)`` via
    :func:`grhoa4`.
    """

    rhonu = jnp.exp(param["logrhonu_of_loga_spline"].evaluate(jnp.log(a)))

    a = a[:, None]
    rho_de = _dark_energy_density_ratio_jax(
        a, param["w_DE_0"][None, :], param["w_DE_a"][None, :]
    )
    grhom = param["grhom"][None, :]
    g4 = grhoa4(
        a,
        param["grhog"][None, :],
        param["grhor"][None, :] * param["Neff"][None, :],
        grhom * param["Omegam"][None, :],
        0.0,
        grhom * param["OmegaDE"][None, :],
        grhok=grhom * param["Omegak"][None, :],
        grhomnu=param["grhor"][None, :] * param["Nmnu"][None, :],
        rhonu=rhonu,
        rho_de=rho_de,
    )
    return (g4 / 3.0) ** 0.5


def dtauda(a, param):
    """Derivative of conformal time with respect to scale factor, ``dtau/da``."""
    return 1 / dadtau(a, param)


def get_aprimeoa(*, param, aexp):
    """Compute the conformal Hubble function ``a'/a = a H(a)``.

    Assembles the ``grho`` coefficients from ``param`` (batched over the
    cosmology dimension) and evaluates :func:`conformal_hubble`.

    Args:
        param (dict): dictionary of cosmological parameters.
        aexp (float, jax.Array): scale factor.

    Returns:
        float: conformal ``H(a)``.
    """

    rhonu = jnp.exp(param["logrhonu_of_loga_spline"].evaluate(jnp.log(aexp)))
    rho_de = _dark_energy_density_ratio_jax(aexp, param["w_DE_0"], param["w_DE_a"])
    grhom = param["grhom"]

    return conformal_hubble(
        aexp,
        param["grhog"],
        param["grhor"] * param["Neff"],
        grhom * param["Omegam"],
        0.0,
        grhom * param["OmegaDE"],
        grhok=grhom * param["Omegak"],
        grhomnu=param["grhor"] * param["Nmnu"],
        rhonu=rhonu,
        rho_de=rho_de,
    )


def compute_angular_diameter_distance(*, aexp, param):
    """Compute the angular diameter distance.

    Args:
        aexp (float): scale factor.
        param (dict): dictionary of cosmological parameters.

    Returns:
        float: angular diameter distance.
    """
    aexpv = jnp.linspace(aexp, 1.0, 1000)
    aH = get_aprimeoa(param=param, aexp=aexpv) * aexpv
    Da = aexp * integrate_trapz(1 / aH, aexpv)
    return Da


def setup_background_evolution(*, amin, amax, param):
    num_neutrino = 512  # number of neutrino history arrays

    param["amin"] = amin
    param["amax"] = amax

    # mean densities (CAMB grho convention, 1/Mpc^2)
    param["grhom"] = critical_density_grho(param["H0"])  # 8 pi G rho_crit / c^2
    param["grhog"] = grhog(param["Tcmb"])  # photon density
    param["grhor"] = neutrino_density_grho(param["Tcmb"])  # neutrino density per flavour
    param["adotrad"] = radiation_hubble_rate(
        param["grhog"], param["grhor"], param["Neff"], param["Nmnu"]
    )

    # conversion factor for neutrino masses (m_nu c^2 / (k_B T_nu0))
    param["amnu"] = neutrino_mass_parameter(param["mnu"], param["Tcmb"])

    # Compute the scale factor linearly spaced in log(a)
    a = jnp.geomspace(amin * 0.9, amax * 1.1, num_neutrino)
    loga = jnp.log(a)
    param["a"] = a

    # Compute the neutrino density and pressure
    rhonu_, pnu_, ppnu_ = nu_background(a, param["amnu"])

    param["logrhonu_of_loga_spline"] = spline_interpolation(
        loga, jnp.log(rhonu_), uniform=True
    )
    param["logpnu_of_loga_spline"] = spline_interpolation(
        loga, jnp.log(pnu_), uniform=True
    )
    param["logppseudonu_of_loga_spline"] = spline_interpolation(
        loga, jnp.log(ppnu_), uniform=True
    )

    # compute the energy density today due to massive neutrinos
    rhonu = jnp.exp(param["logrhonu_of_loga_spline"].evaluate(0.0))
    Omegamnu = (param["grhor"] * rhonu) / param["grhom"]
    param["Omegamnu"] = Omegamnu

    # ensure curvature is correct
    Omegar = (
        (
            param["Neff"]
            + param["Nmnu"] * jnp.exp(param["logrhonu_of_loga_spline"].evaluate(0.0)[0])
        )
        * param["grhor"]
        / param["grhom"]
    )
    Omegag = param["grhog"] / param["grhom"]

    param["OmegaDE"] = 1.0 - param["Omegak"] - Omegar - Omegag - param["Omegam"]

    # Compute the conformal time interval
    param["taumin"] = amin / param["adotrad"]
    integrator = spline_interpolation(loga, dtauda(a, param) * a[:, None], uniform=True)
    param["tau"] = param["taumin"][None, :] + integrator.integral(loga)
    param["taumax"] = param["tau"][-1]

    return param


def batch_dimensions(param):
    for x in param:
        param[x] = jnp.atleast_1d(param[x])
    return param


@partial(
    jax.jit,
    static_argnames=(
        "thermo_module",
        "num_thermo",
        "rtol",
        "atol",
        "order",
        "class_thermo",
    ),
)
def evolve_background(
    *,
    param,
    thermo_module="RECFAST",
    num_thermo: int = 256,
    rtol: float = 1e-5,
    atol: float = 1e-7,
    order: int = 5,
    class_thermo=None,
):
    """Evolve the cosmological background and thermal history

    Parameters
    ----------
    param : dict
        Dictionary of cosmological parameters
    thermo_module : str, optional
        Thermal history module to use: 'RECFAST' (default, high accuracy),
        'MB95' (faster, approximate), or 'CLASS' (use external CLASS data)
    num_thermo : int, optional
        Number of sampling points for thermal history arrays. Default is 256.
        Uses adaptive sampling that concentrates 50% of points around recombination
        (600 < z < 1400) for optimal accuracy. Validated performance:
        - 256 (default): <0.13% error on P(k), 3.2x faster (adaptive sampling)
        - 512: <0.03% error on P(k), 1.7x faster (adaptive sampling)
        - 1024: reference accuracy, slower
    rtol : float, optional
        Relative tolerance for ODE solvers. Default is 1e-5.
    atol : float, optional
        Absolute tolerance for ODE solvers. Default is 1e-7.
    order : int, optional
        Order of spline interpolation. Default is 5.
    class_thermo : dict, optional
        CLASS thermodynamics data (only used when thermo_module='CLASS')

    Returns
    -------
    param : dict
        Updated parameter dictionary with background evolution results and
        spline interpolations for thermal quantities
    """
    amin = 1e-9
    amax = 1.01

    if thermo_module == "CLASS":
        amin = jnp.min(class_thermo["scale factor a"])
        amax = jnp.max(class_thermo["scale factor a"])

    param = batch_dimensions(param)

    param = setup_background_evolution(amin=amin, amax=amax, param=param)

    if thermo_module == "RECFAST":
        # Compute the thermal history
        aexp, cs2, Tm, mu, xe, dxeda = evaluate_thermo_recfast(
            param=param, num_thermo=num_thermo
        )

        param["aexp"] = aexp
        param["xe"] = xe
        param["cs2"] = cs2
        param["Tm"] = Tm

        tau = spline_interpolation(jnp.log(param["a"]), param["tau"]).evaluate(
            jnp.log(aexp)
        )
        param["tau_th"] = tau
        param["tau_of_a_spline"] = spline_interpolation(aexp, tau)
        param["a_of_tau_spline"] = spline_interpolation(tau, aexp)
        param["xe_of_tau_spline"] = spline_interpolation(tau, xe)
        param["cs2a_of_tau_spline"] = spline_interpolation(tau, aexp[:, None] * cs2)
        param["tempba_of_tau_spline"] = spline_interpolation(tau, aexp[:, None] * Tm)

        # Pre-composed splines for direct a-to-quantity lookups (performance optimization)
        param["xe_of_loga_spline"] = spline_interpolation(
            jnp.log(aexp), xe, uniform=False
        )
        param["cs2a_of_loga_spline"] = spline_interpolation(
            jnp.log(aexp), aexp[:, None] * cs2, uniform=False
        )

    elif thermo_module == "MB95":
        # Compute the thermal history
        th, param = compute_thermo_mb95(param=param, nthermo=num_thermo)

        xe = th["xe"]
        xeHI = th["xHII"]
        xeHeI = th["xHeII"]
        xeHeII = th["xHeIII"]
        aexp = th["a"]
        tau = th["tau"]
        cs2 = th["cs2"]
        Tm = th["tb"]

        param["xe"] = xe
        param["xeHI"] = xeHI
        param["xeHeI"] = xeHeI
        param["xeHeII"] = xeHeII
        param["aexp"] = aexp
        param["tau"] = tau

        param["xe_of_tau_spline"] = spline_interpolation(tau, xe)
        param["cs2a_of_tau_spline"] = spline_interpolation(tau, aexp * cs2)
        param["tempba_of_tau_spline"] = spline_interpolation(tau, aexp * Tm)
        param["tau_of_a_spline"] = spline_interpolation(aexp, tau)
        param["a_of_tau_spline"] = spline_interpolation(tau, aexp)

        # Pre-composed splines for direct a-to-quantity lookups (performance optimization)
        param["xe_of_loga_spline"] = spline_interpolation(
            jnp.log(aexp), xe, uniform=True
        )
        param["cs2a_of_loga_spline"] = spline_interpolation(
            jnp.log(aexp), aexp * cs2, uniform=True
        )

    elif thermo_module == "CLASS":
        # use input CLASS thermodynamics
        # interpolating splines for the thermal history
        param["cs2_of_tau_spline"] = spline_interpolation(
            class_thermo["conf. time [Mpc]"][::-1], class_thermo["c_b^2"][::-1]
        )
        param["tempb_of_tau_spline"] = spline_interpolation(
            class_thermo["conf. time [Mpc]"][::-1], class_thermo["Tb [K]"][::-1]
        )
        param["xe_of_tau_spline"] = spline_interpolation(
            class_thermo["conf. time [Mpc]"][::-1], class_thermo["x_e"][::-1]
        )
        param["a_of_tau_spline"] = spline_interpolation(
            class_thermo["conf. time [Mpc]"][::-1], class_thermo["scale factor a"][::-1]
        )
        param["tau_of_a_spline"] = spline_interpolation(
            class_thermo["scale factor a"][::-1], class_thermo["conf. time [Mpc]"][::-1]
        )

        param["aexp"] = class_thermo["scale factor a"][::-1]
        param["tau"] = class_thermo["conf. time [Mpc]"][::-1]

        # Pre-composed splines for direct a-to-quantity lookups (performance optimization)
        aexp_class = class_thermo["scale factor a"][::-1]
        param["xe_of_loga_spline"] = spline_interpolation(
            jnp.log(aexp_class), class_thermo["x_e"][::-1]
        )
        param["cs2a_of_loga_spline"] = spline_interpolation(
            jnp.log(aexp_class), aexp_class * class_thermo["c_b^2"][::-1]
        )

    # compute optical depth and visibility functions
    akthom = thomson_normalization(
        param["Omegab"] * (param["H0"] / 100.0) ** 2, param["YHe"]
    )

    tau_pre_recomb = param["tau_of_a_spline"].evaluate(1e-4)
    xe_full = 1 + param["YHe"] / (1 - param["YHe"])
    xe = jnp.where(
        tau <= tau_pre_recomb, xe_full, param["xe_of_tau_spline"].evaluate(tau)
    )
    xeprime = jnp.where(
        tau <= tau_pre_recomb, 0.0, param["xe_of_tau_spline"].derivative(tau)
    )

    opac = xe * akthom / aexp[:, None] ** 2

    opacspline = spline_interpolation(tau, opac, integrate_from_start=True)
    opacprime, opacpprime = opacspline.derivative12(tau)

    optical_depth_today = opacspline.integral(param["tau_of_a_spline"].evaluate(1.0))
    optical_depth = optical_depth_today - opacspline.integral(tau)

    expmmu = jnp.exp(-optical_depth)
    vis = opac * expmmu
    dvis = (opacprime + opac**2) * expmmu
    ddvis = (opacpprime + 3 * opac * opacprime + opac**3) * expmmu

    param["optical_depth"] = optical_depth
    param["opac"] = opac
    param["gvis"] = vis
    param["gvisprime"] = dvis
    param["gvispprime"] = ddvis

    param["tau0"] = param["tau_of_a_spline"].evaluate(1.0)
    param["tau_maxvis"] = param["tau"][jnp.argmax(param["gvis"])]

    param["xeprime"] = xeprime

    return param


@jax.jit
def compute_background_quantities(aexp, param):
    """Compute background density and equation of state at given scale factors.

    This function evaluates background cosmological quantities including neutrino
    density, dark energy density, and dark energy equation of state at specified
    scale factors. It uses pre-computed splines from evolve_background().

    Parameters
    ----------
    aexp : jnp.ndarray
        Scale factor values where background quantities are needed
    param : dict
        Parameter dictionary containing:
        - logrhonu_of_loga_spline: Spline for log neutrino density
        - w_DE_0: Dark energy equation of state today
        - w_DE_a: Dark energy equation of state time derivative
        - OmegaDE: Dark energy density parameter

    Returns
    -------
    dict
        Dictionary with keys:
        - 'rhonu': Neutrino density ratio rho_nu/rho_nu0 at given scale factors
        - 'rho_Q': Normalized dark energy density rho_Q(a)/rho_Q(a=1)
        - 'w_Q': Dark energy equation of state w(a) at given scale factors

    Examples
    --------
    >>> param = evolve_background(param=param, ...)
    >>> aexp_out = jnp.array([0.01, 0.1, 1.0])
    >>> bg_quantities = compute_background_quantities(aexp_out, param)
    >>> rhonu = bg_quantities['rhonu']
    >>> w_Q = bg_quantities['w_Q']

    Notes
    -----
    The dark energy density assumes a parametrization of the form:
        rho_Q(a) = rho_Q(1) * a^{-3(1 + w_0 + w_a)} * exp[3(a-1)*w_a]

    The equation of state is:
        w_Q(a) = w_0 + w_a * (1 - a)
    """
    a = jnp.atleast_1d(aexp)

    # Neutrino density ratio from spline
    rhonu = jnp.exp(param["logrhonu_of_loga_spline"].evaluate(jnp.log(a)))

    # Dark energy density (normalized to value at a=1) and equation of state
    rho_Q = _dark_energy_density_ratio_jax(a, param["w_DE_0"], param["w_DE_a"])
    w_Q = dark_energy_equation_of_state(a, param["w_DE_0"], param["w_DE_a"])

    return {
        "rhonu": rhonu,
        "rho_Q": rho_Q,
        "w_Q": w_Q,
    }
