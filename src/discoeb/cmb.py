"""CMB temperature source functions, transfer functions, and power spectra.

Once the perturbation hierarchy has been evolved (:mod:`discoeb.perturbations`,
on modax's Rodas5P kernel), the CMB anisotropy today is obtained by
*line-of-sight integration* (Seljak & Zaldarriaga 1996): rather than following
the photon multipole hierarchy up to ``l ~ 2500``, the temperature transfer
function is written as a single integral of a few local source terms against
spherical Bessel functions,

    ``Theta_l(k) = int_0^{tau_0} dtau [S j_l(x) + S_j1 j_l'(x) + S_j2 j_l''(x)]``

with ``x = k (tau_0 - tau)``. The sources are built from the perturbation state
at the save times and the visibility function ``g = kappa' exp(-kappa)``:

    * ``S``    -- the intrinsic monopole (Sachs-Wolfe) and polarization
      correction, weighted by ``g``, plus the integrated Sachs-Wolfe term
      ``2 Phi' exp(-kappa)`` which accumulates along the whole line of sight;
    * ``S_j1`` -- the Doppler term ``g (sigma + v_b)``;
    * ``S_j2`` -- the quadrupole/polarization term.

The angular power spectrum then follows from the primordial curvature spectrum,

    ``C_l = 4 pi int dln k  P_R(k) Theta_l(k)^2``,   ``P_R = A_s (k/k_p)^{n_s - 1}``

and is reported as ``D_l = l(l+1) C_l / 2pi`` in ``muK^2`` (:func:`dl_power_spectrum`).

**Where the sources are evaluated.** Inside the kernel launch: modax's Rodas5P
save hook calls :func:`build_source_device_function` at every save time with
the dense-output state, and the launch returns five numbers per save and
trajectory (``S_noisw, S_j1, S_j2, phi, exp(-kappa)``) instead of the
``nvar``-component history, so all cosmologies of a batch share one launch and
nothing is chunked. The ISW term's ``phi'`` is a centered difference of the
saved potential (:func:`assemble_sources`): the analytic derivative from the
right-hand side is exact algebra but amplifies the solver's state error tenfold
at ``l = 2``. The visibility ``g = kappa' exp(-kappa)`` comes from the
thermodynamics table through a cubic spline whose integral gives ``kappa``
(:func:`visibility_on_grid`); a trapezoid of ``kappa'`` on the save grid was
0.2 percent too large through recombination and cost 0.5 percent of ``C_l``.

**How the line of sight is projected.** As matrix products on grids shared by
the whole batch (:func:`projection_operator`, :func:`theta_ell_batch`). The
sources of every cosmology are shifted in ``tau`` onto one comoving-distance
grid ``chi = tau0_ref - tau`` (a 4-point Lagrange stencil; each cosmology's own
``tau0`` differs from the reference by up to ~100 Mpc), spline-interpolated in
``ln k`` from the solved modes to the fine quadrature grid by a precomputed
matrix, and contracted against one weighted table of ``j_l``, ``j_l'`` and
``j_l''`` on the ``(k, chi)`` grid -- a batched GEMM over cosmologies. Against a
per-cosmology projection with exact ``chi`` this costs 2e-4 in ``D_l``, and the
default ``float32`` arithmetic another 2.6e-4, both below the pipeline's other
numerical differences; it runs in 0.25 ms per cosmology (2.3 ms in ``float64``)
where the per-cosmology spline and Bessel evaluation took 13 ms. An FFTLog
projection was tried and rejected: it must resample the sources onto a
log-spaced ``chi`` grid that is coarsest at recombination, needs 32768 points to
match, and even then costs an order of magnitude more than the table
contraction, whose flop count is a few percent of the FFT's.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
from scipy import integrate, special

from .perturbations import (
    IX_CLXB,
    IX_CLXC,
    IX_ETAK,
    IX_G,
    IX_POL,
    IX_R,
    IX_VB,
    PERTURB_ATOL,
    PERTURB_FIRST_STEP,
    PERTURB_MAX_STEPS,
    PERTURB_RTOL,
    TAU_START,
    _as_cosmology,
    build_thermo_tables,
    density_coefficients,
)
from .spline_interpolation import spline_interpolation

DEFAULT_CMB_K = np.geomspace(1.0e-4, 1.0, 128, dtype=np.float64)
"""Default perturbation wave modes (Mpc^-1) for a CMB solve."""

DEFAULT_CMB_ELLS = np.unique(
    np.concatenate(
        [np.arange(2, 40, 2), np.arange(40, 200, 5), np.arange(200, 2501, 50)]
    )
).astype(np.int64)
"""Default multipoles at which ``C_l`` is evaluated."""

Z_REION_SEARCH_MAX = 30.0
"""Redshift below which a reionization visibility bump would be located."""

SOURCE_CHANNELS = 5
"""Numbers per save: ``S_noisw, S_j1, S_j2, phi, exp(-kappa)``."""

SHIFT_STENCIL = 4
"""Points of the Lagrange stencil that shifts a cosmology's sources in ``tau`` onto
the shared ``chi`` grid: 4 (cubic) leaves 2e-4 in ``D_l``, 2 (linear) 7e-3."""

PROJECTION_BUDGET_BYTES = 1.0e9
"""Device memory for one projection call's fine-``k`` source arrays, which sets how
many cosmologies are contracted together at once."""

_BESSEL_CACHE: dict[tuple, tuple] = {}
_CMB_GRID_CACHE: dict[tuple, dict[str, float]] = {}
_OPERATOR_CACHE: dict[tuple, tuple] = {}
_HOOK_CACHE: dict[tuple, tuple] = {}


# =============================================================================
# 1. The visibility function
# =============================================================================


def _grad_last(y, x):
    """Return ``dy/dx`` along the last axis by centered differences."""

    mid = (y[..., 2:] - y[..., :-2]) / (x[..., 2:] - x[..., :-2])
    first = (y[..., 1:2] - y[..., :1]) / (x[..., 1:2] - x[..., :1])
    last = (y[..., -1:] - y[..., -2:-1]) / (x[..., -1:] - x[..., -2:-1])
    return jnp.concatenate([first, mid, last], axis=-1)


VISIBILITY_REFINEMENT = 16
"""Sub-intervals per thermodynamics-table interval when integrating the optical depth."""


def visibility_on_grid(tau_grid, opacity, tau_save, tau0):
    """Return ``(kappa', g, exp(-kappa))`` on the save grid from a tabulated opacity.

    ``kappa'`` is interpolated by a cubic spline of ``ln kappa'`` in ``ln tau``
    (the thermodynamics table is log-spaced), and the optical depth
    ``kappa(tau) = int_tau^{tau0} kappa' dtau'`` is that spline integrated on a
    grid refined :data:`VISIBILITY_REFINEMENT`-fold. Integrating ``kappa'`` by
    the trapezoid rule on the save grid itself is not good enough: ``kappa'``
    falls by a factor ``e`` every ~15 Mpc through recombination, so on a 3 Mpc
    grid the trapezoid overestimates ``kappa`` by ``(h/L)^2/12 ~ 3e-3`` where
    ``kappa ~ 1``, which lowers ``g`` by 0.3 percent and ``C_l`` by 0.5 percent
    at every multipole. Past ``tau0`` both ``g`` and ``exp(-kappa)`` are zero,
    which zeroes every source there (the frozen state of a trajectory that has
    reached its own end is not physical source data).
    """

    from scipy.interpolate import CubicSpline

    tau_grid = np.asarray(tau_grid, dtype=np.float64)
    opacity = np.asarray(opacity, dtype=np.float64)
    log_spline = CubicSpline(np.log(tau_grid), np.log(opacity))

    n = len(tau_grid)
    fine = np.exp(
        np.linspace(
            np.log(tau_grid[0]), np.log(tau0), VISIBILITY_REFINEMENT * (n - 1) + 1
        )
    )
    kappa_prime_fine = np.exp(log_spline(np.log(fine)))
    kappa_fine = np.concatenate(
        [
            np.cumsum(
                (0.5 * (kappa_prime_fine[1:] + kappa_prime_fine[:-1]) * np.diff(fine))[
                    ::-1
                ]
            )[::-1],
            [0.0],
        ]
    )

    live = tau_save <= tau0
    tau_live = np.minimum(tau_save, tau0)
    kappa_prime = np.where(live, np.exp(log_spline(np.log(tau_live))), 0.0)
    kappa = np.interp(tau_live, fine, kappa_fine)
    emk = np.where(live, np.exp(-kappa), 0.0)
    return kappa_prime, kappa_prime * emk, emk


# =============================================================================
# 2. Source functions
# =============================================================================


def assemble_sources(channels, tau):
    """``(S, S_j1, S_j2)`` from the five channels, with the ISW term by centered differences.

    ``channels`` is ``(..., 5, n_k, n_tau)``; the returned array is ``(..., 3, n_k, n_tau)``.
    """

    s_noisw, s_j1, s_j2, phi, emk = (
        channels[..., i, :, :] for i in range(SOURCE_CHANNELS)
    )
    s_j0 = s_noisw + 2.0 * _grad_last(phi, jnp.asarray(tau)) * emk
    return jnp.stack([s_j0, s_j1, s_j2], axis=-3)


# =============================================================================
# 3. The projection operator: grids, Bessel tables and matrix products
# =============================================================================


def _bessel_row(ell: int, n_x: int, dx: float) -> np.ndarray:
    """Return ``j_ell`` sampled on the uniform grid ``x = 0, dx, ..., dx (n_x-1)``.

    ``j_ell(x)`` is exponentially small below its turning point ``x ~ ell``, so
    the row is left at zero until ``ell - 4 ell^{1/3}``.
    """

    x = np.linspace(0.0, dx * (n_x - 1), n_x)
    x_safe = np.where(x > 1.0e-30, x, 1.0)
    pref = np.where(x > 1.0e-30, np.sqrt(np.pi / (2.0 * x_safe)), 0.0)
    values = np.zeros_like(x)
    x_min = max(0.0, float(ell) - 4.0 * float(ell) ** (1.0 / 3.0))
    start = max(0, int(x_min / dx) - 1)
    values[start:] = pref[start:] * special.jv(float(ell) + 0.5, x[start:])
    if ell == 0:
        values[0] = 1.0
    return values


def _build_bessel_tables(ells, x_max: float, dx: float):
    """Return cached uniform ``j_ell`` and ``j_{ell+1}`` interpolation tables."""

    key = (
        tuple(np.asarray(ells, dtype=np.int64)),
        float(np.ceil(x_max / 250.0) * 250.0),
        float(dx),
    )
    if key in _BESSEL_CACHE:
        return _BESSEL_CACHE[key]

    ell_arr = np.asarray(ells, dtype=np.int64)
    n_x = int(np.ceil(key[1] / dx)) + 2
    unique = np.unique(np.concatenate([ell_arr, ell_arr + 1]))
    rows = {int(ell): _bessel_row(int(ell), n_x, dx) for ell in unique}
    result = (
        0.0,
        1.0 / dx,
        n_x,
        jnp.asarray(np.stack([rows[int(ell)] for ell in ell_arr])),
        jnp.asarray(np.stack([rows[int(ell) + 1] for ell in ell_arr])),
    )
    _BESSEL_CACHE[key] = result
    return result


def _interp_uniform_table(x, x0, inv_dx, n_x, vals):
    """Linearly interpolate a uniformly sampled table at arbitrary ``x``."""

    u = (x - x0) * inv_dx
    idx = jnp.clip(jnp.floor(u).astype(jnp.int32), 0, n_x - 2)
    frac = jnp.clip(u - idx, 0.0, 1.0)
    return (1.0 - frac) * vals[idx] + frac * vals[idx + 1]


def _k_band(ells, k, chi_max, chi_star):
    """Return the ``(n_ell, n_k)`` mask of wave modes that project onto each multipole.

    ``j_l(x)`` is exponentially small below its turning point, so modes with
    ``k chi_max < l - 4 l^{1/3}`` contribute nothing; and modes far above
    ``k ~ l / chi_star`` only contribute through rapidly oscillating tails that
    the finite ``tau`` and ``k`` grids cannot resolve, so integrating them
    injects aliasing noise rather than signal. Without the upper cut the damping
    tail is inflated by several percent (``+6%`` at ``l = 500``).
    """

    ells = ells[:, None]
    k_lo = jnp.maximum(0.0, ells - 4.0 * ells ** (1.0 / 3.0)) / chi_max
    k_hi = (ells + 2500.0) / chi_star
    return (k[None, :] >= k_lo) & (k[None, :] <= k_hi)


def _lagrange_shift_stencil(tau, delta, order=SHIFT_STENCIL):
    """Interpolate a ``tau``-sampled function at ``tau + delta``: stencil start indices and weights.

    Returns ``(i0, w)`` with ``i0`` of shape ``(n_tau,)`` and ``w`` of shape
    ``(n_tau, order)`` such that ``f(tau_j + delta) ~ sum_a w[j, a] f(tau[i0_j + a])``.
    Targets outside the grid are clamped to its ends.
    """

    tau = np.asarray(tau, dtype=np.float64)
    n = len(tau)
    target = np.clip(tau + delta, tau[0], tau[-1])
    i0 = np.clip(np.searchsorted(tau, target) - order // 2, 0, n - order)
    w = np.ones((n, order))
    for a in range(order):
        for b in range(order):
            if a != b:
                w[:, a] *= (target - tau[i0 + b]) / (tau[i0 + a] - tau[i0 + b])
    return i0, w


def projection_operator(ells, k_values, k_fine, tau_save, tau0_ref, dtype=jnp.float32):
    """Return the shared factors of the line-of-sight projection, cached per grids.

    ``interp`` is the ``(n_k_fine, n_k)`` natural-cubic-spline matrix in ``ln k``
    (the spline applied to the identity, so it reproduces
    :class:`~discoeb.spline_interpolation.spline_interpolation` to rounding);
    ``bessel`` is the ``(n_ell, 3, n_k_fine, n_chi)`` table of ``j_l``, ``j_l'``
    and ``j_l''`` at ``x = k chi`` on the shared grid ``chi = tau0_ref - tau_save``
    with the trapezoid weights of the ``tau`` integral folded in, so that

        ``Theta_l(k) = sum_s sum_j A[s, k, j] bessel[l, s, k, j]``

    for the three shifted, refined source channels ``A``; ``lnk_weights`` are the
    trapezoid weights of the ``C_l`` quadrature in ``ln k``.
    """

    ells = np.asarray(ells, dtype=np.int64)
    k_values = np.asarray(k_values, dtype=np.float64)
    k_fine = np.asarray(k_fine, dtype=np.float64)
    tau_save = np.asarray(tau_save, dtype=np.float64)
    key = (
        ells.tobytes(),
        k_values.tobytes(),
        k_fine.tobytes(),
        tau_save.tobytes(),
        float(tau0_ref),
        dtype,
    )
    if key in _OPERATOR_CACHE:
        return _OPERATOR_CACHE[key]

    interp = spline_interpolation(
        jnp.log(jnp.asarray(k_values)), jnp.eye(len(k_values))
    ).evaluate(jnp.log(jnp.asarray(k_fine)))

    chi = tau0_ref - tau_save
    x = jnp.asarray(k_fine)[:, None] * jnp.asarray(chi)[None, :]
    x0, inv_dx, n_x, jl_tab, jl1_tab = _build_bessel_tables(
        ells, float(np.max(k_fine) * np.max(chi)) + 2.0, 0.03
    )
    weights = np.zeros(len(tau_save))
    weights[1:] += 0.5 * np.diff(tau_save)
    weights[:-1] += 0.5 * np.diff(tau_save)

    def rows(i, ell):
        jl = _interp_uniform_table(x, x0, inv_dx, n_x, jl_tab[i])
        jl1 = _interp_uniform_table(x, x0, inv_dx, n_x, jl1_tab[i])
        inv_x = jnp.where(x > 1.0e-30, 1.0 / jnp.where(x > 1.0e-30, x, 1.0), 0.0)
        jld = ell * inv_x * jl - jl1
        jldd = -2.0 * inv_x * jld + (ell * (ell + 1.0) * inv_x * inv_x - 1.0) * jl
        return jnp.stack([jl, jld, jldd]) * jnp.asarray(weights)

    bessel = jnp.stack([rows(i, float(ell)) for i, ell in enumerate(ells)]).astype(
        dtype
    )

    lnk_weights = np.zeros(len(k_fine))
    dlnk = np.diff(np.log(k_fine))
    lnk_weights[1:] += 0.5 * dlnk
    lnk_weights[:-1] += 0.5 * dlnk

    op = (interp.astype(dtype), bessel, jnp.asarray(lnk_weights))
    _OPERATOR_CACHE[key] = op
    return op


@jax.jit
def theta_ell_batch(channels, tau, shift_idx, shift_w, band, interp, bessel):
    """``Theta_l(k_fine)`` of a group of cosmologies from their source channels, as matrix products.

    ``channels`` is ``(n_c, 5, n_k, n_tau)`` on the shared save grid; ``shift_idx``
    and ``shift_w`` are each cosmology's stencil onto the shared ``chi`` grid
    (:func:`_lagrange_shift_stencil`); ``band`` is the ``(n_c, n_ell, n_k_fine)``
    mask of :func:`_k_band`; ``interp`` and ``bessel`` come from
    :func:`projection_operator`. Returns ``(n_c, n_ell, n_k_fine)`` in float64.
    """

    dtype = bessel.dtype
    sources = assemble_sources(channels, tau[None, None, :])  # (c, 3, k, tau)
    shifted = sum(
        shift_w[:, None, None, :, a]
        * jnp.take_along_axis(sources, (shift_idx + a)[:, None, None, :], axis=3)
        for a in range(shift_w.shape[-1])
    ).astype(dtype)
    refined = jnp.einsum("fk,cskt->csft", interp, shifted)  # (c, 3, k_fine, chi)
    theta = jnp.einsum("csft,lsft->clf", refined, bessel)  # (c, l, k_fine)
    return jnp.where(band, theta, 0.0).astype(jnp.float64)


@jax.jit
def cl_from_theta(theta, lnk_weights, n_s, A_s, k_pivot, k_fine):
    """``C_l = 4 pi int dln k P_R(k) Theta_l(k)^2`` for ``theta`` of shape ``(n_c, n_ell, n_k_fine)``."""

    primordial = A_s[:, None] * (k_fine[None, :] / k_pivot) ** (n_s[:, None] - 1.0)
    return 4.0 * jnp.pi * jnp.einsum("clf,cf,f->cl", theta**2, primordial, lnk_weights)


def dl_power_spectrum(cl, ells, T_cmb):
    """Return ``D_l = l(l+1) C_l / 2pi`` in ``muK^2``."""

    ell = jnp.asarray(ells, dtype=jnp.float64)
    return ell * (ell + 1.0) * jnp.asarray(cl) / (2.0 * jnp.pi) * T_cmb**2 * 1.0e12


def _projection_group_size(n_cosmologies, n_k_fine, n_tau, dtype) -> int:
    per_cosmology = 3 * n_k_fine * n_tau * jnp.dtype(dtype).itemsize * 2
    return max(
        1, min(n_cosmologies, int(PROJECTION_BUDGET_BYTES // max(per_cosmology, 1)))
    )


# =============================================================================
# 4. Adaptive k and tau sampling
# =============================================================================


def _grhoa4(a, grhog, grhor, grhoc, grhob, grhov):
    """``8 pi G rho a^4``: the expansion rate is ``a' = sqrt(grhoa4 / 3)``."""

    return grhog + grhor + (grhoc + grhob) * a + grhov * a**4


def _dtau_da(a, *dens):
    """``dtau/da = 1 / a'``."""

    return np.sqrt(3.0 / _grhoa4(a, *dens))


def _cmb_grid_summary(cosmology) -> dict[str, float]:
    """Return the characteristic scales that set the adaptive ``k`` / ``tau`` grids.

    ``tau_star`` is the visibility peak and ``delta_tau_rec`` its Gaussian
    equivalent width; ``r_s`` is the sound horizon at recombination (which sets
    the acoustic oscillation period in ``k``) and ``k_D`` the Silk damping
    wavenumber (which sets their envelope). Together they say where the CMB
    sources have structure, and hence where the grids must be dense.
    """

    cosmology = _as_cosmology(cosmology)
    if cosmology in _CMB_GRID_CACHE:
        return _CMB_GRID_CACHE[cosmology]

    dens = density_coefficients(cosmology)
    tau_grid, values, _, tau0 = build_thermo_tables(cosmology)
    a_vals, opacity = values[0], values[1]

    _, vis, _ = visibility_on_grid(tau_grid, opacity, tau_grid, float(tau0))
    peak = int(np.argmax(vis))
    tau_star = float(tau_grid[peak])
    a_star = float(a_vals[peak])

    half = 0.5 * float(vis[peak])
    left = float(np.interp(half, vis[: peak + 1], tau_grid[: peak + 1]))
    right = float(np.interp(half, vis[peak:][::-1], tau_grid[peak:][::-1]))
    delta_tau_rec = max((right - left) / (2.0 * np.sqrt(2.0 * np.log(2.0))), 1.0)

    grhog_v, grhor_v, grhoc_v, grhob_v, _ = dens
    tau_eq = float(
        np.interp((grhog_v + grhor_v) / (grhoc_v + grhob_v), a_vals, tau_grid)
    )

    # Comoving sound horizon r_s = int c_s dtau = int_0^{a_star} da / (a' sqrt(3(1+R))),
    # with a' = sqrt(grhoa4/3) and R = 3 rho_b / (4 rho_g).
    def sound_integrand(ap):
        r = 0.75 * grhob_v * ap / grhog_v
        return 1.0 / np.sqrt(_grhoa4(ap, *dens) * (1.0 + r))

    r_s = float(integrate.quad(sound_integrand, 0.0, a_star, limit=200)[0])

    # Silk damping scale from the photon diffusion length.
    a_silk = np.linspace(max(float(a_vals[0]), 1.0e-8), a_star, 5000)
    opacity_silk = np.maximum(np.interp(a_silk, a_vals, opacity), 1.0e-30)
    r = 0.75 * grhob_v * a_silk / grhog_v
    integrand_d = (
        (r**2 + 16.0 * (1.0 + r) / 15.0)
        / (6.0 * (1.0 + r) ** 2 * opacity_silk)
        * _dtau_da(a_silk, *dens)
    )
    k_D = float(1.0 / np.sqrt(np.trapezoid(integrand_d, a_silk)))

    # A secondary visibility bump from reionization, if the thermal history has
    # one (it does not at present, so the fall-back below is what is used).
    post_rec = a_vals > 1.0 / (1.0 + Z_REION_SEARCH_MAX)
    if np.any(post_rec) and np.max(np.where(post_rec, vis, 0.0)) > 1.0e-3 * vis[peak]:
        reion_idx = int(np.argmax(np.where(post_rec, vis, 0.0)))
        tau_reion = float(tau_grid[reion_idx])
        delta_tau_reion = max(0.03 * tau0, 80.0)
    else:
        tau_reion = 0.6 * tau0
        delta_tau_reion = 0.05 * tau0

    out = {
        "tau0": float(tau0),
        "tau_eq": tau_eq,
        "tau_star": tau_star,
        "delta_tau_rec": delta_tau_rec,
        "tau_reion": tau_reion,
        "delta_tau_reion": delta_tau_reion,
        "r_s": r_s,
        "k_D": k_D,
    }
    _CMB_GRID_CACHE[cosmology] = out
    return out


def _resample_from_density(x, density, n):
    """Return ``n`` points drawn from ``x`` in proportion to ``density``."""

    cdf = np.cumsum(density) * (x[1] - x[0])
    cdf -= cdf[0]
    cdf /= cdf[-1]
    return np.interp(np.linspace(0.0, 1.0, n), cdf, x)


def cmb_k_grid(
    cosmology,
    *,
    n: int = 200,
    mode: str = "ode",
    k_min: float = 1.0e-5,
    k_max: float = 0.5,
    ell_min: int = 2,
    ell_max: int = 2500,
    n_ell_samples: int = 30,
    n_eval: int = 5000,
) -> np.ndarray:
    """Return a nonuniform wavenumber grid concentrated where the sources vary.

    ``mode="ode"`` resolves the acoustic oscillations of the source itself
    (period set by ``r_s``, envelope by Silk damping ``k_D``) and is the grid the
    perturbation solve runs on. ``mode="cl"`` additionally weights each ``k`` by
    how many multipoles project onto it, ``k ~ l / chi_star``, and is the finer
    grid the ``C_l`` quadrature uses after the sources are interpolated.
    """

    cosmology = _as_cosmology(cosmology)
    summary = _cmb_grid_summary(cosmology)
    x = np.linspace(np.log(k_min), np.log(k_max), n_eval)
    k = np.exp(x)
    primordial = k ** (cosmology.n_s + 2.0)
    acoustic_curv = (1.0 / summary["r_s"]) ** 2
    damped = primordial * np.exp(-((k / summary["k_D"]) ** 2))

    if mode == "cl":
        sigma_k = 1.0 / summary["delta_tau_rec"]
        chi_star = summary["tau0"] - summary["tau_star"]
        ell_samples = np.unique(
            np.geomspace(ell_min, ell_max, n_ell_samples).astype(int)
        )
        raw_weight = np.zeros_like(k)
        for ell in ell_samples:
            envelope = np.exp(-0.5 * ((k - ell / chi_star) / (3.0 * sigma_k)) ** 2)
            raw_weight += np.maximum(acoustic_curv, sigma_k**2) * envelope * damped
        floor = 1.0e-6 * np.max(raw_weight)
    else:
        smooth_curv = (k * summary["r_s"]) ** 2 / summary["tau_eq"] ** 2
        raw_weight = np.maximum(acoustic_curv, smooth_curv) * damped
        floor = 0.005 * np.max(raw_weight)

    grid = np.exp(_resample_from_density(x, (raw_weight + floor) ** (1.0 / 3.0), n))
    grid[0], grid[-1] = k_min, k_max
    return grid


def cmb_tau_grid(
    cosmology, k_values, n: int, *, tau_end: float | None = None, n_eval: int = 10000
) -> np.ndarray:
    """Return a nonuniform conformal-time grid on which to save the sources.

    The sources are sharply peaked at recombination (width ``delta_tau_rec``),
    modulated on the sound-horizon scale, and would have a secondary bump at
    reionization, so the sampling density is the cube root of the sum of those
    three weights. Endpoints are pinned to ``[TAU_START, tau_end]``, which the
    solver's save grid requires.

    ``tau_end`` defaults to this cosmology's ``tau0``. A batch of cosmologies
    passes the *largest* ``tau0`` of the batch so that no trajectory's save grid
    is cut short; the shaping weights still come from ``cosmology``.
    """

    k_arr = np.asarray(k_values, dtype=np.float64)
    s = _cmb_grid_summary(cosmology)
    tau0 = s["tau0"] if tau_end is None else float(tau_end)

    tau = np.linspace(TAU_START, tau0, n_eval, dtype=np.float64)
    weight = (
        np.exp(-0.5 * ((tau - s["tau_star"]) / s["delta_tau_rec"]) ** 2)
        / s["delta_tau_rec"] ** 2
    )
    weight += (
        np.exp(-0.5 * ((tau - s["tau_star"]) / s["r_s"]) ** 2)
        * (float(k_arr[-1]) / np.sqrt(3.0)) ** 2
    )
    weight += (
        0.3
        * np.exp(-0.5 * ((tau - s["tau_reion"]) / s["delta_tau_reion"]) ** 2)
        / s["delta_tau_reion"] ** 2
    )

    tau_save = _resample_from_density(
        tau, (weight + 0.005 * np.max(weight)) ** (1.0 / 3.0), n
    )
    tau_save[0], tau_save[-1] = TAU_START, tau0
    return tau_save


# =============================================================================
# 5. In-kernel sources: modax's save hook
# =============================================================================


def _kernel_settings(prepared, solve_kwargs) -> dict:
    """The modax settings the matter-power solve of :mod:`discoeb.perturbations` uses."""

    from .perturbations import CUDA_MAX_REGISTERS, IX_TAU_END, TRAJECTORIES_PER_BLOCK

    return dict(
        sparsity=prepared.sparsity,
        lu_precision="fp32",
        rtol=solve_kwargs.get("rtol", PERTURB_RTOL),
        atol=solve_kwargs.get("atol", PERTURB_ATOL),
        max_steps=solve_kwargs.get("max_steps", PERTURB_MAX_STEPS),
        pcoeff=0.3,
        icoeff=0.4,
        trajectories_per_block=TRAJECTORIES_PER_BLOCK,
        tf_index=IX_TAU_END,
        max_registers=CUDA_MAX_REGISTERS,
    )


# =============================================================================
# 6. In-kernel sources: modax's save hook
# =============================================================================
#
# modax's Rodas5P kernel can call a device function at every save time with
# the dense-output state. Two uses are built here. The *source hook* evaluates
# the three line-of-sight sources in the kernel and stores them per save, so
# the launch returns ``(n_traj, 3 n_save)`` numbers instead of the
# ``(n_traj, n_save, nvar)`` history and the projection proceeds as usual on
# the host. The *projection hook* goes all the way: it accumulates the
# trapezoid sum of the sources against the tabulated Bessel functions per save,
# so the launch returns ``Theta_l(k)`` for the wave modes it solved. Since
# ``Theta_l(k)`` oscillates far faster in ``k`` than the sources do, that mode
# has to solve on the fine ``k`` grid the host would otherwise only interpolate
# onto.


def _source_tables(prepared, tau_save):
    """Per-cosmology ``a``, ``g`` and ``exp(-kappa)`` on the save grid, as host arrays.

    Both visibility factors are zero past each cosmology's own ``tau0``, which
    zeroes every source there (the ISW term through ``exp(-kappa)``).
    """

    a_tab, g_tab, emk_tab = [], [], []
    for tau_grid, values, _, tau0 in prepared.tables:
        a_tab.append(np.interp(tau_save, tau_grid, values[0]))
        _, g, emk = visibility_on_grid(tau_grid, values[1], tau_save, float(tau0))
        g_tab.append(g)
        emk_tab.append(emk)
    return (
        np.ascontiguousarray(np.stack(a_tab)),
        np.ascontiguousarray(np.stack(g_tab)),
        np.ascontiguousarray(np.stack(emk_tab)),
    )


def build_source_device_function(a_tab, g_tab, emk_tab):
    """Return a device function ``sources(save_idx, y, tau, p)``.

    It returns ``(S_noisw, S_j1, S_j2, phi, exp(-kappa))``: the three CAMB-form
    CAMB-form sources of the module docstring *without* the ISW term, plus the
    Weyl potential and the transmission the ISW term ``2 phi' exp(-kappa)``
    is built from. ``phi'`` is deliberately not taken here: differentiating the
    right-hand side analytically at a save is exact algebra but amplifies the
    solver's state error, since the late-time ``phi'`` is a small residual of
    large cancelling terms (at ``rtol = 1e-4`` it moves ``D_l`` at ``l = 2`` by
    10 percent); a difference of saved potentials cancels that error, so the
    hooks difference ``phi`` across saves exactly as the host does. ``a``, ``g``
    and ``exp(-kappa)`` come from the tables of :func:`_source_tables`, indexed
    by the trajectory's cosmology tag and the save index. Flat LambdaCDM with
    massless neutrinos, as the host sources.
    """

    from numba_cuda_mlir import cuda

    from .perturbations import IX_COSMOLOGY, IX_K

    a_dev = cuda.to_device(a_tab)
    g_dev = cuda.to_device(g_tab)
    emk_dev = cuda.to_device(emk_tab)

    @cuda.jit(device=True)
    def sources(save_idx, y, tau, p):
        ci = int(p[IX_COSMOLOGY])
        k = p[IX_K]
        a = a_dev[ci, save_idx]
        g = g_dev[ci, save_idx]
        emk = emk_dev[ci, save_idx]

        etak = y[IX_ETAK]
        clxc, clxb, vb = y[IX_CLXC], y[IX_CLXB], y[IX_VB]
        clxg, qg, pig = y[IX_G], y[IX_G + 1], y[IX_G + 2]
        e2 = y[IX_POL]
        clxr, qr, pir = y[IX_R], y[IX_R + 1], y[IX_R + 2]

        a2 = a * a
        grhog_t = p[0] / a2
        grhor_t = p[1] / a2
        grhoc_t = p[2] / a
        grhob_t = p[3] / a
        grhov_t = p[4] * a2
        adotoa = math.sqrt((grhog_t + grhor_t + grhoc_t + grhob_t + grhov_t) / 3.0)

        dgrho = grhob_t * clxb + grhoc_t * clxc + grhog_t * clxg + grhor_t * clxr
        dgq = grhob_t * vb + grhog_t * qg + grhor_t * qr
        dgpi = grhog_t * pig + grhor_t * pir
        k2 = k * k
        phi = -(dgrho + 3.0 * dgq * adotoa / k + dgpi) / (2.0 * k2)  # Weyl potential
        z = (0.5 * dgrho / k + etak) / adotoa
        sigma = z + 1.5 * dgq / k2
        polter = pig / 10.0 + 9.0 / 15.0 * e2
        monopole = -etak / k + 2.0 * phi + 0.25 * clxg

        s_noisw = g * (monopole + 0.625 * polter)
        s_j1 = g * (sigma + vb)
        s_j2 = 1.875 * g * polter
        return s_noisw, s_j1, s_j2, phi, emk

    return sources


def build_source_hook(sources):
    """A save hook storing the five source channels of every save in the trajectory's row."""

    from numba_cuda_mlir import cuda

    @cuda.jit(device=True)
    def hook(save_idx, y, tau, p, acc):
        s_noisw, s_j1, s_j2, phi, emk = sources(save_idx, y, tau, p)
        base = SOURCE_CHANNELS * save_idx
        acc[base] = s_noisw
        acc[base + 1] = s_j1
        acc[base + 2] = s_j2
        acc[base + 3] = phi
        acc[base + 4] = emk

    return hook


def _solve_with_hook(prepared, k_values, tau_save, hook, hook_size, solve_kwargs):
    """One hooked launch of the whole chunk; returns ``(hook_out, k_sorted)``.

    ``hook_out`` is ``(n_k, n_cosmo, hook_size)``; no history is kept.
    """

    from solvers.rodas5P import solve as rodas5P_solve

    from .perturbations import _pack_batch

    params, y0, t_span, _, k_sorted = _pack_batch(prepared, k_values, tau_save)
    _, out = rodas5P_solve(
        prepared.ode_fn,
        y0,
        t_span,
        params,
        first_step=solve_kwargs.get("first_step", PERTURB_FIRST_STEP),
        save_hook=hook,
        hook_size=hook_size,
        save_history=False,
        **_kernel_settings(prepared, solve_kwargs),
    )
    n_cosmo = len(prepared.cosmologies)
    return jnp.asarray(out).reshape(len(k_sorted), n_cosmo, hook_size), k_sorted


# =============================================================================
# 6. End-to-end spectrum
# =============================================================================


def compute_cl_power_spectrum_batch(
    cosmologies,
    *,
    k_values: np.ndarray | None = None,
    ells: np.ndarray | None = None,
    n_save: int = 1000,
    n_k_fine: int = 1500,
    k_fine_values: np.ndarray | None = None,
    projection_dtype=jnp.float32,
    **solve_kwargs,
):
    """Return ``(ells, C_l, D_l)`` for a batch of cosmologies, shape ``(N, n_ell)``.

    One modax launch integrates the perturbation hierarchy of every wave mode of
    every cosmology, with the sources evaluated at the save times by the save
    hook; the launch returns ``5 n_save`` numbers per trajectory and no history,
    so nothing bounds the batch but the kernel's own memory. The save grid is
    shaped by the first cosmology and extended to the largest ``tau0`` of the
    batch (each cosmology's own ``tau0`` is respected through zeroed sources
    past it). The sources are then refined in ``ln k`` onto ``k_fine``
    (default: the ``C_l`` grid of :func:`cmb_k_grid`) and projected by
    :func:`theta_ell_batch` and :func:`cl_from_theta`.

    ``projection_dtype`` is the arithmetic of the projection's matrix products:
    ``float32`` (default, 2.6e-4 in ``D_l``) or ``float64`` (ten times slower
    on consumer GPUs). ``solve_kwargs`` reach the solver (``rtol``, ``atol``,
    ``first_step``, ``max_steps``).
    """

    from .perturbations import _prepare_solve

    cosmologies = tuple(_as_cosmology(c) for c in cosmologies)
    base = cosmologies[0]
    if k_values is None:
        k_values = cmb_k_grid(base, n=128, mode="ode")
    if ells is None:
        ells = DEFAULT_CMB_ELLS
    k_values = np.asarray(k_values, dtype=np.float64)
    ells = np.asarray(ells, dtype=np.int64)

    prepared = _prepare_solve(cosmologies)
    tau0_all = np.asarray(prepared.tau0, dtype=np.float64)
    tau0_ref = float(np.max(tau0_all))
    tau_save = cmb_tau_grid(base, k_values, n_save, tau_end=tau0_ref)
    if k_fine_values is None:
        k_fine = cmb_k_grid(
            base,
            n=n_k_fine,
            mode="cl",
            k_min=float(k_values[0]),
            k_max=float(k_values[-1]),
            ell_max=int(np.max(ells)),
        )
    else:
        k_fine = np.asarray(k_fine_values, dtype=np.float64)

    # The hook closes over device tables built for these cosmologies and this
    # save grid; it is the kernel-cache key, so keep it for repeat solves.
    key = (prepared.cosmologies, tau_save.tobytes())
    if key not in _HOOK_CACHE:
        _HOOK_CACHE[key] = build_source_hook(
            build_source_device_function(*_source_tables(prepared, tau_save))
        )
    out, k_sorted = _solve_with_hook(
        prepared,
        k_values,
        tau_save,
        _HOOK_CACHE[key],
        SOURCE_CHANNELS * len(tau_save),
        solve_kwargs,
    )
    # (n_k, n_cosmo, 5 n_tau) -> (n_cosmo, 5, n_k, n_tau)
    channels = jnp.transpose(
        out.reshape(len(k_sorted), len(cosmologies), len(tau_save), SOURCE_CHANNELS),
        (1, 3, 0, 2),
    )
    del out

    interp, bessel, lnk_weights = projection_operator(
        ells, k_sorted, k_fine, tau_save, tau0_ref, projection_dtype
    )
    ells_d = jnp.asarray(ells, dtype=jnp.float64)
    k_fine_d = jnp.asarray(k_fine)
    tau_d = jnp.asarray(tau_save)
    chi_max = float(tau0_ref - tau_save[0])
    stencils = [
        _lagrange_shift_stencil(tau_save, float(t0 - tau0_ref)) for t0 in tau0_all
    ]
    bands = jnp.stack(
        [
            _k_band(
                ells_d, k_fine_d, chi_max, float(t0 - _cmb_grid_summary(c)["tau_star"])
            )
            for t0, c in zip(tau0_all, cosmologies)
        ]
    )
    n_s = jnp.asarray([float(c.n_s) for c in cosmologies])
    A_s = jnp.asarray([float(c.A_s) for c in cosmologies])
    T_cmb = jnp.asarray([float(c.T_cmb) for c in cosmologies])
    k_pivot = float(base.k_pivot)

    # The projection of a group of cosmologies is one set of GEMMs; the group
    # is sized so its fine-k source arrays fit the memory budget.
    cls = []
    group = _projection_group_size(
        len(cosmologies), len(k_fine), len(tau_save), projection_dtype
    )
    for g0 in range(0, len(cosmologies), group):
        g1 = min(g0 + group, len(cosmologies))
        theta = theta_ell_batch(
            channels[g0:g1],
            tau_d,
            jnp.asarray(np.stack([st[0] for st in stencils[g0:g1]])),
            jnp.asarray(np.stack([st[1] for st in stencils[g0:g1]])),
            bands[g0:g1],
            interp,
            bessel,
        )
        cls.append(
            cl_from_theta(theta, lnk_weights, n_s[g0:g1], A_s[g0:g1], k_pivot, k_fine_d)
        )
    cl = jnp.concatenate(cls)
    return ells, cl, dl_power_spectrum(cl, ells_d, T_cmb[:, None])


def compute_cl_power_spectrum(cosmology, **kwargs):
    """Return ``(ells, C_l, D_l)`` for the unlensed scalar TT spectrum of one cosmology.

    See :func:`compute_cl_power_spectrum_batch` for the options.
    """

    ells_out, cl, dl = compute_cl_power_spectrum_batch([cosmology], **kwargs)
    return ells_out, cl[0], dl[0]
