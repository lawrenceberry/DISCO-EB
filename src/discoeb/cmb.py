"""CMB temperature and polarization sources, transfer functions, and power spectra.

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

**Polarization.** The E-mode transfer function needs no further source: in
CAMB's form it is the quadrupole/polarization term against ``j_l(x) / x^2``,

    ``E_l(k) = sqrt((l+2)!/(l-2)!) int_0^{tau_0} dtau S_j2 j_l(x) / x^2``

(Zaldarriaga & Seljak 1997; CAMB's ``sources(2)``), so the projection carries
a fourth Bessel row, ``j_l(x) / x^2`` with the ``l``-dependent prefactor folded
in, tabulated directly (not divided after interpolation, since ``j_l`` is
interpolated linearly and ``j_2(x) / x^2 -> 1/15`` at the end of the line of
sight, where a reionized universe still has visibility). Then
``C_l^{EE} = 4 pi int dln k P_R E_l^2`` and ``C_l^{TE} = 4 pi int dln k P_R
Theta_l E_l``; the three spectra are returned together as a :class:`Spectra`
(``tt, te, ee``). The lensed TE, EE and BB follow in :mod:`discoeb.lensing`.
The polarization source is a small state component at low ``k``, below the
scalar absolute tolerance of the solve, so the CMB solves weight the photon
quadrupole and ``E_2`` up in the solver's error norm in proportion to ``1/k``
(:func:`polarization_error_weights`); without that the low multipoles of EE
and TE come out 10 to 40 percent low, at no gain in speed.


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
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy import integrate, special

from .component import Component
from .eb_model import perturbation_model
from .perturbations import (
    BATCH_CACHE_SIZE,
    PARAM_COLUMNS,
    PHYSICAL_DENSITY_COLUMNS,
    THERMO_TABLE_CACHE_SIZE,
    BoundedCache,
    DeviceTableSlot,
    IX_G,
    IX_K,
    IX_POL,
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

SOURCE_QUANTITIES = ("s_noisw", "s_j1", "s_j2", "phi")
"""The first four channels, as the component model provides them; the fifth is a table."""


class LineOfSightSources(Component):
    """The CAMB-form temperature and polarization sources, as a component.

    Appended to the perturbation model, it provides the Weyl potential
    ``phi = -(dgrho + 3 aH dgq / k + dgpi) / (2 k^2)`` from the Einstein
    sources every species contributes to, and the three sources against
    ``j_l``, ``j_l'`` and ``j_l''`` of the module docstring, without the ISW
    term: ``g (monopole + 5/8 Pi)``, ``g (sigma + v_b)`` and ``15/8 g Pi``,
    with the visibility ``g`` given and the scattering source ``Pi``
    (``polter``) and the shear ``sigma`` from the Thomson interaction and the
    metric. One set of equations serves the device hook and the differentiable
    host path (:func:`source_channels_jax`), and any species the layout enables
    enters through the sums.
    """

    def phi(self, dgrho, dgq, dgpi, adotoa, k, inv_k2):
        return -(dgrho + 3.0 * dgq * adotoa / k + dgpi) * 0.5 * inv_k2

    def monopole(self, etak, k, phi, clxg):
        return -etak / k + 2.0 * phi + 0.25 * clxg

    def s_noisw(self, g, monopole, polter):
        return g * (monopole + 0.625 * polter)

    def s_j1(self, g, sigma, vb):
        return g * (sigma + vb)

    def s_j2(self, g, polter):
        return 1.875 * g * polter


def source_model(layout, thermal_history=None):
    """The perturbation model of ``layout`` with the :class:`LineOfSightSources` appended."""

    return perturbation_model(layout, thermal_history, extra=[LineOfSightSources()])


class Spectra(NamedTuple):
    """The scalar CMB spectra of one computation: temperature, cross and E-mode.

    Each field is an array over the multipoles (with a leading batch axis from
    :func:`compute_cl_power_spectrum_batch`); as a tuple it stacks and maps like
    one, ``jnp.stack(spectra)`` giving ``(3, ...)`` in the order ``tt, te, ee``.
    """

    tt: jax.Array
    te: jax.Array
    ee: jax.Array

SHIFT_STENCIL = 4
"""Points of the Lagrange stencil that shifts a cosmology's sources in ``tau`` onto
the shared ``chi`` grid: 4 (cubic) leaves 2e-4 in ``D_l``, 2 (linear) 7e-3."""

PROJECTION_BUDGET_BYTES = 1.0e9
"""Device memory for one projection call's fine-``k`` source arrays, which sets how
many cosmologies are contracted together at once."""

TAU_END_MARGIN = 0.01
"""How far past the fiducial cosmology's ``tau0`` the save grid of :func:`cmb_grids`
extends, as a fraction, so that the grids built once from a fiducial serve every
batch whose conformal ages fall within it. Measured against per-batch grids on
the 128 nearby cosmologies of the tests, the shift onto the fiducial's grid
moves ``D_l`` by at most 2.7e-4 at 1 percent, 1.0e-3 at 2 percent and 4.4e-3
at 5 percent (the low multipoles' late-time sources sit on the coarse end of
the grid)."""

POLARIZATION_WEIGHT_K = 0.1
"""Wave number (Mpc^-1) below which the photon quadrupole and ``E_2`` are weighted
up in the solver's error norm, by ``POLARIZATION_WEIGHT_K / k``."""

POLARIZATION_WEIGHT_MAX = 1000.0
"""Cap of that weight."""


def polarization_error_weights(k, nvar):
    """Per-trajectory solver error weights ``(n_traj, nvar)`` that resolve the polarization source at low ``k``.

    The E-mode source is ``g polter`` with ``polter = pig/10 + 3/5 E_2``. At
    ``k`` well below the acoustic scale both moments are tiny through
    recombination -- of the order of the solve's scalar ``atol = 1e-4`` at
    ``k ~ 1e-3 Mpc^-1`` -- so the step-size control, which measures each
    component's error against ``atol + rtol |y|``, does not resolve them, and
    the low multipoles of EE and TE, which those modes source, came out 10 to
    40 percent low (TT, whose sources are unit-size, was unaffected).
    Weighting the two components by ``clip(POLARIZATION_WEIGHT_K / k, 1,
    POLARIZATION_WEIGHT_MAX)`` divides their tolerance by that factor. The
    trajectories it slows are the low-``k`` ones, which take the fewest steps
    of a launch, so a batch costs the same within 3 percent at N=128, and EE
    agrees with CLASS to 1.2 percent at every multipole from 2 (TE to 0.4
    percent of ``sqrt(TT EE)``). A scalar ``atol`` of 1e-8 does the same at
    six times the cost.

    ``k`` is each trajectory's wave number in launch order; the modax solves
    take the result as ``error_weights``.
    """

    k = np.asarray(k, dtype=np.float64)
    weights = np.ones((len(k), int(nvar)))
    w = np.clip(POLARIZATION_WEIGHT_K / k, 1.0, POLARIZATION_WEIGHT_MAX)
    weights[:, IX_G + 2] = w
    weights[:, IX_POL] = w
    return jnp.asarray(weights)

_BESSEL_CACHE: BoundedCache = BoundedCache(BATCH_CACHE_SIZE)
_CMB_GRID_CACHE: BoundedCache = BoundedCache(THERMO_TABLE_CACHE_SIZE)
_VISIBILITY_PEAK_CACHE: BoundedCache = BoundedCache(THERMO_TABLE_CACHE_SIZE)
_OPERATOR_CACHE: BoundedCache = BoundedCache(BATCH_CACHE_SIZE)
_HOOK_SLOTS: dict[tuple, DeviceTableSlot] = {}


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


def source_channels_jax(model, history, k_sorted, a_tab, g_tab, emk_tab, row):
    """The five source channels of a saved history, as JAX arithmetic.

    The :func:`source_model`'s equations evaluated on the host over a history
    ``(n_k, n_tau, nvar)`` in ascending ``k`` so that they can be
    differentiated: the densities enter the sources through the metric
    combinations and the background expansion rate, and that path is part of
    the total derivative of ``C_l`` with respect to them. ``row`` is the
    parameter row of :data:`~discoeb.perturbations.PARAM_COLUMNS` by name,
    with the densities as the traced values; ``a_tab``, ``g_tab`` and
    ``emk_tab`` are one cosmology's rows of :func:`_source_tables`. Returns
    ``(5, n_k, n_tau)``.
    """

    base = dict(row)
    base.update(model.state_values(jnp.asarray(history), xp=jnp))
    base.update(
        k=jnp.asarray(k_sorted)[:, None],
        a=jnp.asarray(a_tab)[None, :],
        g=jnp.asarray(g_tab)[None, :],
    )
    s_noisw, s_j1, s_j2, phi = model.evaluate(*SOURCE_QUANTITIES, math=jnp, **base)
    emk = jnp.broadcast_to(jnp.asarray(emk_tab)[None, :], phi.shape)
    return jnp.stack([s_noisw, s_j1, s_j2, phi, emk], axis=0)


# =============================================================================
# 3. The projection operator: grids, Bessel tables and matrix products
# =============================================================================


def _bessel_row(ell: int, n_x: int, dx: float, over_x2: bool = False) -> np.ndarray:
    """Return ``j_ell`` sampled on the uniform grid ``x = 0, dx, ..., dx (n_x-1)``.

    ``j_ell(x)`` is exponentially small below its turning point ``x ~ ell``, so
    the row is left at zero until ``ell - 4 ell^{1/3}``. With ``over_x2`` the
    row is ``j_ell(x) / x^2``, the E-mode kernel, which is smooth through
    ``x = 0`` (``1/15`` there for ``ell = 2``, zero above) and so interpolates
    accurately where ``j_ell`` itself, divided afterwards, would not.
    """

    x = np.linspace(0.0, dx * (n_x - 1), n_x)
    x_safe = np.where(x > 1.0e-30, x, 1.0)
    pref = np.where(x > 1.0e-30, np.sqrt(np.pi / (2.0 * x_safe)), 0.0)
    values = np.zeros_like(x)
    x_min = max(0.0, float(ell) - 4.0 * float(ell) ** (1.0 / 3.0))
    start = max(0, int(x_min / dx) - 1)
    values[start:] = pref[start:] * special.jv(float(ell) + 0.5, x[start:])
    if over_x2:
        values = values / x_safe**2
        values[0] = 1.0 / 15.0 if ell == 2 else 0.0
    elif ell == 0:
        values[0] = 1.0
    return values


def _build_bessel_tables(ells, x_max: float, dx: float):
    """Return cached uniform ``j_ell``, ``j_{ell+1}`` and ``j_ell / x^2`` interpolation tables."""

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
        jnp.asarray(np.stack([_bessel_row(int(ell), n_x, dx, over_x2=True) for ell in ell_arr])),
    )
    _BESSEL_CACHE[key] = result
    return result


def _interp_uniform_table(x, x0, inv_dx, n_x, vals):
    """Linearly interpolate a uniformly sampled table at arbitrary ``x``."""

    u = (x - x0) * inv_dx
    idx = jnp.clip(jnp.floor(u).astype(jnp.int32), 0, n_x - 2)
    frac = jnp.clip(u - idx, 0.0, 1.0)
    return (1.0 - frac) * vals[idx] + frac * vals[idx + 1]


@jax.jit
def _k_band(ells, k, chi_max, chi_star):
    """Return the ``(n_ell, n_k)`` mask of wave modes that project onto each multipole.

    ``j_l(x)`` is exponentially small below its turning point, so modes with
    ``k chi_max < l - 4 l^{1/3}`` contribute nothing; and modes far above
    ``k ~ l / chi_star`` only contribute through rapidly oscillating tails that
    the finite ``tau`` and ``k`` grids cannot resolve, so integrating them
    injects aliasing noise rather than signal. Without the upper cut the damping
    tail is inflated by several percent (``+6%`` at ``l = 500``).

    ``chi_star`` may carry leading batch axes, ``(n_c,)`` for a batch of
    cosmologies, and the mask then has them too: ``(n_c, n_ell, n_k)``, from
    one jitted call rather than one eager call per cosmology.
    """

    ells = ells[:, None]
    chi_star = jnp.asarray(chi_star)[..., None, None]
    k_lo = jnp.maximum(0.0, ells - 4.0 * ells ** (1.0 / 3.0)) / chi_max
    k_hi = (ells + 2500.0) / chi_star
    return (k[None, :] >= k_lo) & (k[None, :] <= k_hi)


def _lagrange_shift_stencil(tau, delta, order=SHIFT_STENCIL):
    """Interpolate a ``tau``-sampled function at ``tau + delta``: stencil start indices and weights.

    Returns ``(i0, w)`` with ``i0`` of shape ``(n_tau,)`` and ``w`` of shape
    ``(n_tau, order)`` such that ``f(tau_j + delta) ~ sum_a w[j, a] f(tau[i0_j + a])``.
    Targets outside the grid are clamped to its ends. ``delta`` may carry
    leading axes, ``(n_c,)`` for one shift per cosmology of a batch, which
    then lead the shapes of ``i0`` and ``w``.
    """

    tau = np.asarray(tau, dtype=np.float64)
    n = len(tau)
    target = np.clip(np.asarray(delta, dtype=np.float64)[..., None] + tau, tau[0], tau[-1])
    i0 = np.clip(np.searchsorted(tau, target) - order // 2, 0, n - order)
    w = np.ones(target.shape + (order,))
    for a in range(order):
        for b in range(order):
            if a != b:
                w[..., a] *= (target - tau[i0 + b]) / (tau[i0 + a] - tau[i0 + b])
    return i0, w


class ProjectionOperator(NamedTuple):
    """The shared factors of the line-of-sight projection, from :func:`projection_operator`."""

    interp: jax.Array
    bessel: jax.Array
    bessel_e: jax.Array
    lnk_weights: jax.Array


def projection_operator(ells, k_values, k_fine, tau_save, tau0_ref, dtype=jnp.float32):
    """Return the shared factors of the line-of-sight projection, cached per grids.

    ``interp`` is the ``(n_k_fine, n_k)`` natural-cubic-spline matrix in ``ln k``
    (the spline applied to the identity, so it reproduces
    :class:`~discoeb.spline_interpolation.spline_interpolation` to rounding);
    ``bessel`` is the ``(n_k_fine, n_ell, 3, n_chi)`` table of ``j_l``, ``j_l'``
    and ``j_l''`` at ``x = k chi`` on the shared grid ``chi = tau0_ref - tau_save``
    with the trapezoid weights of the ``tau`` integral folded in, so that

        ``Theta_l(k) = sum_s sum_j A[s, k, j] bessel[k, l, s, j]``

    for the three shifted, refined source channels ``A``; ``bessel_e`` is the
    ``(n_k_fine, n_ell, n_chi)`` table of the E-mode kernel
    ``sqrt((l+2)!/(l-2)!) j_l(x) / x^2`` with the same weights, contracted
    against the third channel alone; ``lnk_weights`` are the trapezoid weights
    of the ``C_l`` quadrature in ``ln k``.
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
    x0, inv_dx, n_x, jl_tab, jl1_tab, jlx2_tab = _build_bessel_tables(
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

    def row_e(i, ell):
        prefactor = math.sqrt((ell + 2.0) * (ell + 1.0) * ell * (ell - 1.0))
        jlx2 = _interp_uniform_table(x, x0, inv_dx, n_x, jlx2_tab[i])
        return jlx2 * (prefactor * jnp.asarray(weights))
    # Stored as (n_k_fine, n_ell, 3, n_chi), the operand layout of the
    # contraction in :func:`theta_ell_batch`: with (n_ell, 3, n_k_fine, n_chi)
    # XLA transposes the table (gigabytes at inference grids) at every call.
    bessel = jnp.stack(
        [jnp.transpose(rows(i, float(ell)), (1, 0, 2)).astype(dtype) for i, ell in enumerate(ells)],
        axis=1,
    )
    bessel_e = jnp.stack([row_e(i, float(ell)).astype(dtype) for i, ell in enumerate(ells)], axis=1)

    lnk_weights = np.zeros(len(k_fine))
    dlnk = np.diff(np.log(k_fine))
    lnk_weights[1:] += 0.5 * dlnk
    lnk_weights[:-1] += 0.5 * dlnk

    op = ProjectionOperator(interp.astype(dtype), bessel, bessel_e, jnp.asarray(lnk_weights))
    _OPERATOR_CACHE[key] = op
    return op


@jax.jit
def theta_ell_batch(channels, tau, shift_idx, shift_w, band, interp, bessel, bessel_e):
    """``(Theta_l, E_l)`` at ``k_fine`` of a group of cosmologies from their source channels, as matrix products.

    ``channels`` is ``(n_c, 5, n_k, n_tau)`` on the shared save grid; ``shift_idx``
    and ``shift_w`` are each cosmology's stencil onto the shared ``chi`` grid
    (:func:`_lagrange_shift_stencil`); ``band`` is the ``(n_c, n_ell, n_k_fine)``
    mask of :func:`_k_band`; ``interp``, ``bessel`` and ``bessel_e`` come from
    :func:`projection_operator`. Returns the temperature and E-mode transfer
    functions, each ``(n_c, n_ell, n_k_fine)`` in float64.
    """

    dtype = bessel.dtype
    sources = assemble_sources(channels, tau[None, None, :])  # (c, 3, k, tau)
    shifted = sum(
        shift_w[:, None, None, :, a]
        * jnp.take_along_axis(sources, (shift_idx + a)[:, None, None, :], axis=3)
        for a in range(shift_w.shape[-1])
    ).astype(dtype)
    refined = jnp.einsum("fk,cskt->csft", interp, shifted)  # (c, 3, k_fine, chi)
    theta = jnp.einsum("csft,flst->clf", refined, bessel)  # (c, l, k_fine)
    theta_e = jnp.einsum("cft,flt->clf", refined[:, 2], bessel_e)
    return (
        jnp.where(band, theta, 0.0).astype(jnp.float64),
        jnp.where(band, theta_e, 0.0).astype(jnp.float64),
    )


@jax.jit
def cl_from_theta(theta, lnk_weights, n_s, A_s, k_pivot, k_fine, theta2=None):
    """``C_l = 4 pi int dln k P_R(k) Theta_l(k) Theta2_l(k)`` for transfer functions of shape ``(n_c, n_ell, n_k_fine)``.

    The auto spectrum of ``theta`` by default; the cross spectrum with ``theta2``.
    """

    primordial = A_s[:, None] * (k_fine[None, :] / k_pivot) ** (n_s[:, None] - 1.0)
    product = theta**2 if theta2 is None else theta * theta2
    return 4.0 * jnp.pi * jnp.einsum("clf,cf,f->cl", product, primordial, lnk_weights)


def spectra_from_theta(theta, theta_e, lnk_weights, n_s, A_s, k_pivot, k_fine) -> Spectra:
    """The ``(tt, te, ee)`` :class:`Spectra` of the two transfer functions of :func:`theta_ell_batch`."""

    args = (lnk_weights, n_s, A_s, k_pivot, k_fine)
    return Spectra(
        cl_from_theta(theta, *args),
        cl_from_theta(theta, *args, theta2=theta_e),
        cl_from_theta(theta_e, *args),
    )


def dl_power_spectrum(cl, ells, T_cmb):
    """Return ``D_l = l(l+1) C_l / 2pi`` in ``muK^2``."""

    ell = jnp.asarray(ells, dtype=jnp.float64)
    return ell * (ell + 1.0) * jnp.asarray(cl) / (2.0 * jnp.pi) * T_cmb**2 * 1.0e12


def _projection_group_size(n_cosmologies, n_k_fine, n_tau, dtype) -> int:
    per_cosmology = 4 * n_k_fine * n_tau * jnp.dtype(dtype).itemsize * 2
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

    # The secondary visibility bump from reionization when tau_reio > 0; without
    # one, the save grid's late-time weight falls back to a broad bump.
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


def _visibility_peak(cosmology) -> float:
    """The conformal time of the visibility maximum, ``tau_star``.

    The one number of :func:`_cmb_grid_summary` the batched spectrum needs per
    cosmology (for the ``k`` band of :func:`_k_band`), without the sound-horizon
    and Silk-damping quadratures the grids need only from their fiducial.
    """

    cosmology = _as_cosmology(cosmology)
    cached = _VISIBILITY_PEAK_CACHE.get(cosmology)
    if cached is None:
        tau_grid, values, _, tau0 = build_thermo_tables(cosmology)
        _, vis, _ = visibility_on_grid(tau_grid, values[1], tau_grid, float(tau0))
        cached = float(tau_grid[int(np.argmax(vis))])
        _VISIBILITY_PEAK_CACHE[cosmology] = cached
    return cached


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


class CMBGrids(NamedTuple):
    """The grids of one CMB spectrum computation, shaped by a fiducial cosmology.

    ``k_values`` are the wave modes the hierarchy is solved on, ``k_fine`` the
    quadrature grid in ``k`` the sources are refined onto, ``tau_save`` the
    conformal times the sources are recorded at, ``ells`` the multipoles.
    Built by :func:`cmb_grids`; passed to :func:`compute_cl_power_spectrum_batch`
    they fix the line-of-sight projection operator across batches.
    """

    k_values: np.ndarray
    ells: np.ndarray
    k_fine: np.ndarray
    tau_save: np.ndarray

    @property
    def tau_end(self) -> float:
        return float(self.tau_save[-1])


def cmb_grids(
    cosmology,
    *,
    k_values: np.ndarray | None = None,
    ells: np.ndarray | None = None,
    n_save: int = 1000,
    n_k_fine: int = 1500,
    k_fine_values: np.ndarray | None = None,
    tau_end: float | None = None,
    tau_end_margin: float = TAU_END_MARGIN,
) -> CMBGrids:
    """Return the :class:`CMBGrids` shaped by ``cosmology``, built once for many batches.

    The grids adapt to where the sources have structure -- the recombination
    epoch and the sound horizon in ``tau`` (:func:`cmb_tau_grid`), the acoustic
    scale and Silk damping in ``k`` (:func:`cmb_k_grid`) -- and nearby
    cosmologies share that structure, so one fiducial's grids serve them all;
    a batch has always been solved on its first cosmology's grids. Fixing them
    across batches is what lets the projection operator
    (:func:`projection_operator`: the Bessel table and the ``ln k`` refinement
    matrix) be built once rather than per batch, 0.3 s each time.

    The save grid ends at ``tau_end``, by default ``tau_end_margin`` past the
    fiducial's ``tau0``, and every cosmology projected on these grids must
    have ``tau0 <= tau_end`` (its sources are shifted onto the shared
    comoving-distance grid, which must reach ``tau0``). A cosmology's own
    trajectory still stops at its own ``tau0``; the saves past it carry zero
    sources. The shift grows with the margin and costs accuracy through the
    source interpolation (see :data:`TAU_END_MARGIN` for the measured cost),
    so keep the margin to what the batches need. A cosmology past ``tau_end``
    raises ``ValueError`` rather than projecting short.
    """

    cosmology = _as_cosmology(cosmology)
    if k_values is None:
        k_values = cmb_k_grid(cosmology, n=128, mode="ode")
    if ells is None:
        ells = DEFAULT_CMB_ELLS
    k_values = np.asarray(k_values, dtype=np.float64)
    ells = np.asarray(ells, dtype=np.int64)
    if tau_end is None:
        tau_end = _cmb_grid_summary(cosmology)["tau0"] * (1.0 + float(tau_end_margin))
    tau_save = cmb_tau_grid(cosmology, k_values, n_save, tau_end=float(tau_end))
    if k_fine_values is None:
        k_fine = cmb_k_grid(
            cosmology,
            n=n_k_fine,
            mode="cl",
            k_min=float(k_values[0]),
            k_max=float(k_values[-1]),
            ell_max=int(np.max(ells)),
        )
    else:
        k_fine = np.asarray(k_fine_values, dtype=np.float64)
    return CMBGrids(k_values, ells, k_fine, tau_save)


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
        error_weights=solve_kwargs.get("error_weights"),
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


def build_source_device_function(model, a_tab, g_tab, emk_tab):
    """Return a device function ``sources(save_idx, y, tau, p)``.

    It returns ``(S_noisw, S_j1, S_j2, phi, exp(-kappa))``: the three
    CAMB-form sources of the module docstring *without* the ISW term, plus the
    Weyl potential and the transmission the ISW term ``2 phi' exp(-kappa)``
    is built from. ``phi'`` is deliberately not taken here: differentiating the
    right-hand side analytically at a save is exact algebra but amplifies the
    solver's state error, since the late-time ``phi'`` is a small residual of
    large cancelling terms (at ``rtol = 1e-4`` it moves ``D_l`` at ``l = 2`` by
    10 percent); a difference of saved potentials cancels that error, so the
    hooks difference ``phi`` across saves exactly as the host does. The four
    quantities are the :func:`source_model`'s, compiled by
    :meth:`~discoeb.component.Model.build_function` from the same equations
    as the right-hand side, with ``a`` and ``g`` read from the tables of
    :func:`_source_tables`, indexed by the trajectory's cosmology tag and the
    save index; host arrays are uploaded, device arrays (a
    :class:`DeviceTableSlot`'s buffers, refilled per batch) are read in place.
    """

    from numba_cuda_mlir import cuda

    from .perturbations import IX_COSMOLOGY

    def on_device(table):
        if cuda.is_cuda_array(table):
            return table
        return cuda.to_device(np.ascontiguousarray(table, dtype=np.float64))

    a_dev = on_device(a_tab)
    g_dev = on_device(g_tab)
    emk_dev = on_device(emk_tab)
    quantities = model.build_function(
        SOURCE_QUANTITIES, inputs=("y", "tau", "a", "g"), params=PARAM_COLUMNS, name="sources"
    )

    @cuda.jit(device=True)
    def sources(save_idx, y, tau, p):
        ci = int(p[IX_COSMOLOGY])
        s_noisw, s_j1, s_j2, phi = quantities(
            y, tau, a_dev[ci, save_idx], g_dev[ci, save_idx], p
        )
        return s_noisw, s_j1, s_j2, phi, emk_dev[ci, save_idx]

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

    ``hook_out`` is ``(n_k, n_cosmo, hook_size)``; no history is kept. Unless
    ``solve_kwargs`` carries ``error_weights``, the launch uses
    :func:`polarization_error_weights`.
    """

    from modax.rodas5P import solve as rodas5P_solve

    from .perturbations import _pack_batch

    params, y0, t_span, _, k_sorted = _pack_batch(prepared, k_values, tau_save)
    prepared.bind()
    if solve_kwargs.get("error_weights") is None:
        solve_kwargs = {
            **solve_kwargs,
            "error_weights": polarization_error_weights(params[:, IX_K], y0.shape[-1]),
        }
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
    grids: CMBGrids | None = None,
    k_values: np.ndarray | None = None,
    ells: np.ndarray | None = None,
    n_save: int = 1000,
    n_k_fine: int = 1500,
    k_fine_values: np.ndarray | None = None,
    projection_dtype=jnp.float32,
    lensing_potential_ells: np.ndarray | None = None,
    **solve_kwargs,
):
    """Return ``(ells, C_l, D_l)`` for a batch of cosmologies, each a :class:`Spectra` of ``(N, n_ell)``.

    ``C_l`` holds the unlensed scalar ``tt``, ``te`` and ``ee`` spectra
    (dimensionless, relative to ``T_cmb``) and ``D_l = l(l+1) C_l / 2pi`` the
    same in ``muK^2``. With ``lensing_potential_ells`` the lensing potential
    spectrum ``C_l^{psi psi}`` at those multipoles (:mod:`discoeb.lensing`) is
    returned as a fourth element, ``(N, n_ell_psi)``, from the Weyl potential
    the same launch recorded; :func:`discoeb.lensing.lensed_spectra` turns the
    two into the lensed TT, TE, EE and BB.


    One modax launch integrates the perturbation hierarchy of every wave mode of
    every cosmology, with the sources evaluated at the save times by the save
    hook; the launch returns ``5 n_save`` numbers per trajectory and no history,
    so nothing bounds the batch but the kernel's own memory. The sources are
    then refined in ``ln k`` onto the fine grid and projected by
    :func:`theta_ell_batch` and :func:`spectra_from_theta`.

    ``grids`` fixes the wave modes, multipoles, fine ``k`` grid and save grid
    (:func:`cmb_grids`); built once from a fiducial cosmology, it makes the
    projection operator shared by every batch of an inference loop. Without
    it, the grids are shaped by the batch's first cosmology from the remaining
    keyword arguments, with the save grid ending at the largest ``tau0`` of the
    batch (each cosmology's own ``tau0`` is respected through zeroed sources
    past it), and the operator is rebuilt for each batch whose grids differ.

    ``projection_dtype`` is the arithmetic of the projection's matrix products:
    ``float32`` (default, 2.6e-4 in ``D_l``) or ``float64`` (ten times slower
    on consumer GPUs). ``solve_kwargs`` reach the solver (``rtol``, ``atol``,
    ``first_step``, ``max_steps``).
    """

    from .perturbations import _prepare_solve

    cosmologies = tuple(_as_cosmology(c) for c in cosmologies)
    prepared = _prepare_solve(cosmologies)
    tau0_all = np.asarray(prepared.tau0, dtype=np.float64)
    if grids is None:
        grids = cmb_grids(
            cosmologies[0],
            k_values=k_values,
            ells=ells,
            n_save=n_save,
            n_k_fine=n_k_fine,
            k_fine_values=k_fine_values,
            tau_end=float(np.max(tau0_all)),
        )
    k_values, ells, k_fine, tau_save = grids
    tau0_ref = grids.tau_end
    if np.max(tau0_all) > tau0_ref * (1.0 + 1.0e-12):
        raise ValueError(
            f"a cosmology's tau0 = {np.max(tau0_all):.1f} Mpc lies past the save grid's "
            f"end {tau0_ref:.1f} Mpc: build the grids with a larger tau_end_margin"
        )

    # The hook is part of modax's kernel-cache key, so like the right-hand
    # side it is compiled once per batch shape, against a slot of device
    # tables that each batch's visibility is copied into before its launch.
    n_cosmo, n_save_grid = len(cosmologies), len(tau_save)
    hook_slot = _HOOK_SLOTS.get((prepared.layout, n_cosmo, n_save_grid))
    if hook_slot is None:
        from .perturbations import ThermalHistory

        # The sources read the same thermodynamics buffers as the right-hand side.
        model = source_model(prepared.layout, ThermalHistory(*prepared.slot.buffers))
        hook_slot = DeviceTableSlot(
            (((n_cosmo, n_save_grid), np.float64),) * 3,
            lambda *buffers: build_source_hook(build_source_device_function(model, *buffers)),
        )
        _HOOK_SLOTS[(prepared.layout, n_cosmo, n_save_grid)] = hook_slot
    hook_key = (prepared.cosmologies, tau_save.tobytes())
    if hook_slot.loaded != hook_key:
        hook_slot.load(hook_key, *_source_tables(prepared, tau_save))
    out, k_sorted = _solve_with_hook(
        prepared,
        k_values,
        tau_save,
        hook_slot.device_function,
        SOURCE_CHANNELS * n_save_grid,
        solve_kwargs,
    )
    # (n_k, n_cosmo, 5 n_tau) -> (n_cosmo, 5, n_k, n_tau)
    channels = jnp.transpose(
        out.reshape(len(k_sorted), len(cosmologies), len(tau_save), SOURCE_CHANNELS),
        (1, 3, 0, 2),
    )
    del out

    interp, bessel, bessel_e, lnk_weights = projection_operator(
        ells, k_sorted, k_fine, tau_save, tau0_ref, projection_dtype
    )
    ells_d = jnp.asarray(ells, dtype=jnp.float64)
    k_fine_d = jnp.asarray(k_fine)
    tau_d = jnp.asarray(tau_save)
    # Everything per cosmology below is one host array and one device transfer
    # for the whole batch: the shifts onto the shared chi grid, the k bands, the
    # primordial amplitudes. A per-cosmology loop of eager ``jnp`` arithmetic
    # here cost 80 ms of host time a batch (and, while the kernel was in
    # flight, waited for it).
    chi_max = float(tau0_ref - tau_save[0])
    shift_idx, shift_w = _lagrange_shift_stencil(tau_save, tau0_all - tau0_ref)
    shift_idx, shift_w = jnp.asarray(shift_idx), jnp.asarray(shift_w)
    chi_star = np.asarray(
        [t0 - _visibility_peak(c) for t0, c in zip(tau0_all, cosmologies)]
    )
    bands = _k_band(ells_d, k_fine_d, chi_max, chi_star)
    n_s = jnp.asarray([float(c.n_s) for c in cosmologies])
    A_s = jnp.asarray([float(c.A_s) for c in cosmologies])
    T_cmb = jnp.asarray([float(c.T_cmb) for c in cosmologies])
    k_pivot = float(cosmologies[0].k_pivot)

    # The projection of a group of cosmologies is one set of GEMMs; the group
    # is sized so its fine-k source arrays fit the memory budget.
    # The projection of a group of cosmologies is one set of GEMMs; the group
    # is sized so its fine-k source arrays fit the memory budget. The groups
    # stay separate eager calls: under one ``jax.jit`` XLA merges the sliced
    # einsums back into a single whole-batch one (2.3 GB at N=128), which the
    # allocator cannot place beside the kernel's local memory and retries for
    # tens of seconds.
    groups = []
    group = _projection_group_size(
        len(cosmologies), len(k_fine), len(tau_save), projection_dtype
    )
    for g0 in range(0, len(cosmologies), group):
        g1 = min(g0 + group, len(cosmologies))
        theta, theta_e = theta_ell_batch(
            channels[g0:g1],
            tau_d,
            shift_idx[g0:g1],
            shift_w[g0:g1],
            bands[g0:g1],
            interp,
            bessel,
            bessel_e,
        )
        groups.append(
            spectra_from_theta(
                theta, theta_e, lnk_weights, n_s[g0:g1], A_s[g0:g1], k_pivot, k_fine_d
            )
        )
    cl = Spectra(*(jnp.concatenate(parts) for parts in zip(*groups)))
    dl = Spectra(*(dl_power_spectrum(c, ells_d, T_cmb[:, None]) for c in cl))
    if lensing_potential_ells is None:
        return ells, cl, dl
    from .lensing import lensing_potential_cl

    ells_pp = np.asarray(lensing_potential_ells, dtype=np.int64)
    operator, limber_weights, ells_limber = _lensing_operators(ells_pp, k_sorted, tau_save, tau0_ref, projection_dtype)
    cl_pp = lensing_potential_cl(
        channels[:, 3], shift_idx, shift_w, jnp.asarray(tau0_ref - tau_save), jnp.asarray(chi_star),
        operator, limber_weights, jnp.asarray(ells_limber, dtype=jnp.float64), jnp.asarray(k_sorted), n_s, A_s, k_pivot,
    )
    return ells, cl, dl, cl_pp


def _lensing_operators(ells, k_sorted, tau_save, tau0_ref, dtype):
    """The exact-projection table and the Limber weights of :mod:`discoeb.lensing`, cached per grids.

    Multipoles below :data:`discoeb.lensing.LIMBER_LMIN` are projected exactly,
    the rest in the Limber approximation; ``ells`` must be sorted.
    """

    from .lensing import LIMBER_LMIN, lensing_potential_operator, limber_interpolation

    key = ("lensing", ells.tobytes(), np.asarray(k_sorted).tobytes(), np.asarray(tau_save).tobytes(), float(tau0_ref), dtype)
    if key not in _OPERATOR_CACHE:
        exact, limber = ells[ells < LIMBER_LMIN], ells[ells >= LIMBER_LMIN]
        _OPERATOR_CACHE[key] = (
            lensing_potential_operator(exact, k_sorted, tau_save, tau0_ref, dtype),
            limber_interpolation(limber, k_sorted, tau0_ref - np.asarray(tau_save)),
            limber,
        )
    return _OPERATOR_CACHE[key]


def compute_cl_power_spectrum(cosmology, **kwargs):
    """Return ``(ells, C_l, D_l)`` of one cosmology, each spectrum a :class:`Spectra` of ``(n_ell,)``.

    See :func:`compute_cl_power_spectrum_batch` for the options.
    """

    ells_out, cl, dl = compute_cl_power_spectrum_batch([cosmology], **kwargs)
    return ells_out, Spectra(*(c[0] for c in cl)), Spectra(*(d[0] for d in dl))


def cl_power_spectrum_jax(
    cosmology,
    densities=None,
    *,
    grids: CMBGrids | None = None,
    k_values: np.ndarray | None = None,
    ells: np.ndarray | None = None,
    n_save: int = 1000,
    n_k_fine: int = 1500,
    k_fine_values: np.ndarray | None = None,
    projection_dtype=jnp.float64,
    one_column_per_trajectory: bool = True,
    **solve_kwargs,
):
    """``(ells, C_l, D_l)`` of one cosmology as JAX arrays, differentiable in the densities.

    The counterpart of :func:`discoeb.perturbations.matter_power_spectrum_jax`
    for the unlensed spectra (a :class:`Spectra` each, ``tt, te, ee`` of
    shape ``(n_ell,)``). ``densities`` is the five-vector of
    :data:`~discoeb.perturbations.PHYSICAL_DENSITY_COLUMNS`, defaulting to the
    cosmology's own; differentiating with respect to it runs modax's forward
    sensitivity system alongside the state, in either of the two thread layouts
    of :func:`~discoeb.perturbations.matter_power_spectrum_jax`
    (``one_column_per_trajectory``). The densities enter through the hierarchy
    and again through the source algebra (:func:`source_channels_jax`), and
    both paths are differentiated. As for ``P(k)``, the thermodynamics -- here
    the visibility and ``a(tau)`` on the save grid -- and the grids are held
    fixed, so this is the derivative at fixed ionisation history. The solve
    uses :func:`polarization_error_weights` unless ``error_weights`` is given.

    A hooked modax launch cannot be differentiated (the hook's accumulator is
    outside the sensitivity rule), so unlike :func:`compute_cl_power_spectrum`
    this path saves the state history on the ``tau`` grid and forms the sources
    on the host; for one cosmology that history is a few hundred megabytes. The
    projection defaults to ``float64``: the ``float32`` matrix products cost
    2.6e-4 in ``D_l``, which a derivative of the spectrum cannot afford.
    """

    from .perturbations import _history_jax, _pack_batch, _prepare_solve

    cosmology = _as_cosmology(cosmology)
    prepared = _prepare_solve((cosmology,))
    cosmology = prepared.cosmologies[0]
    tau0 = float(prepared.tau0[0])
    if grids is None:
        grids = cmb_grids(
            cosmology,
            k_values=k_values,
            ells=ells,
            n_save=n_save,
            n_k_fine=n_k_fine,
            k_fine_values=k_fine_values,
            tau_end=tau0,
        )
    k_values, ells, k_fine, tau_save = grids
    tau0_ref = grids.tau_end
    if tau0 > tau0_ref * (1.0 + 1.0e-12):
        raise ValueError(
            f"tau0 = {tau0:.1f} Mpc lies past the save grid's end {tau0_ref:.1f} Mpc: "
            "build the grids with a larger tau_end_margin"
        )

    history, densities, k_sorted, _ = _history_jax(
        prepared,
        k_values,
        densities,
        tau_save,
        rtol=solve_kwargs.get("rtol", PERTURB_RTOL),
        atol=solve_kwargs.get("atol", PERTURB_ATOL),
        first_step=solve_kwargs.get("first_step", PERTURB_FIRST_STEP),
        max_steps=solve_kwargs.get("max_steps", PERTURB_MAX_STEPS),
        one_column_per_trajectory=one_column_per_trajectory,
        sens_error_control=solve_kwargs.get("sens_error_control", False),
        error_weights=solve_kwargs.get("error_weights")
        if solve_kwargs.get("error_weights") is not None
        else polarization_error_weights(
            np.sort(np.asarray(k_values, dtype=np.float64)), prepared.layout.nvar
        ),
    )
    a_tab, g_tab, emk_tab = _source_tables(prepared, tau_save)
    # The parameter row by name, its densities the traced ones.
    params = _pack_batch(prepared, k_values, tau_save)[0]
    row = {name: float(params[0, column]) for name, column in PARAM_COLUMNS.items()}
    for name, column in PARAM_COLUMNS.items():
        if column in PHYSICAL_DENSITY_COLUMNS:
            row[name] = densities[column]
    channels = source_channels_jax(
        source_model(prepared.layout), history, k_sorted, a_tab[0], g_tab[0], emk_tab[0], row
    )

    interp, bessel, bessel_e, lnk_weights = projection_operator(
        ells, k_sorted, k_fine, tau_save, tau0_ref, projection_dtype
    )
    ells_d = jnp.asarray(ells, dtype=jnp.float64)
    k_fine_d = jnp.asarray(k_fine)
    idx, w = _lagrange_shift_stencil(tau_save, tau0 - tau0_ref)
    band = _k_band(
        ells_d,
        k_fine_d,
        float(tau0_ref - tau_save[0]),
        float(tau0 - _visibility_peak(cosmology)),
    )
    theta, theta_e = theta_ell_batch(
        channels[None],
        jnp.asarray(tau_save),
        jnp.asarray(idx)[None],
        jnp.asarray(w)[None],
        band[None],
        interp,
        bessel,
        bessel_e,
    )
    cl = spectra_from_theta(
        theta,
        theta_e,
        lnk_weights,
        jnp.asarray([float(cosmology.n_s)]),
        jnp.asarray([float(cosmology.A_s)]),
        float(cosmology.k_pivot),
        k_fine_d,
    )
    cl = Spectra(*(c[0] for c in cl))
    dl = Spectra(*(dl_power_spectrum(c, ells_d, float(cosmology.T_cmb)) for c in cl))
    return ells, cl, dl
