"""Weak lensing of the CMB temperature spectrum.

Two pieces, both batched over cosmologies:

1. **The lensing potential power spectrum** ``C_l^{psi psi}`` from the Weyl
   potential the perturbation kernel records at the save times (the fourth
   source channel of :mod:`discoeb.cmb`).  The lensing potential is the
   line-of-sight integral of the Weyl potential against the lensing kernel
   (Lewis & Challinor 2006, eq. 4.6; CAMB's ``sources(3)``):

   ``psi(n) = -2 int_0^{chi_*} dchi (chi_* - chi) / (chi chi_*) phi_W(chi n, tau_0 - chi)``,

   so ``Theta_l^psi(k) = -2 int dchi W(chi) phi_W(k, tau_0 - chi) j_l(k chi)``
   and ``C_l^{psi psi} = 4 pi int dln k P_R(k) |Theta_l^psi(k)|^2``, with the
   same primordial normalization as the temperature spectrum of
   :func:`discoeb.cmb.cl_from_theta`.  The kernel is broad in ``chi``, so the
   transfer function is smooth in ``k`` and the modes the hierarchy was solved
   on suffice for the quadrature (no refinement onto the fine grid), and the
   projection is a small matrix product per cosmology
   (:func:`lensing_potential_operator`, :func:`lensing_potential_cl`).

2. **The lensed temperature spectrum** by the full-sky correlation-function
   method of Challinor & Lewis (2005, Phys. Rev. D 71, 103010), the method of
   CAMB and CLASS: the lensed correlation function is

   ``xi~(beta) = sum_l (2l+1)/(4pi) C_l [X_000^2 d^l_00 + 8/(l(l+1)) C_gl,2 X'_000^2 d^l_{1-1} + C_gl,2^2 X_220^2 d^l_{2-2}]``

   to second order in ``C_gl,2``, with ``X_000 = exp(-l(l+1) sigma^2/4)``,
   ``X'_000 = -l(l+1)/4 X_000``, ``X_220 = sqrt((l+2)(l-1)l(l+1))/4 X_000``,
   ``sigma^2(beta) = C_gl(0) - C_gl(beta)``, ``C_gl = sum_l (2l+1)/(4pi) l(l+1)
   C_l^psi d^l_11`` and ``C_gl,2`` the same sum with ``d^l_{1-1}``; the lensed
   spectrum is the Legendre transform back, ``C~_l = 2 pi int xi~(beta)
   d^l_00(beta) dcos(beta)``, by Gauss-Legendre quadrature.  The Wigner
   d-functions on the quadrature nodes depend on nothing but ``l_max`` and the
   number of nodes and are tabulated once (:func:`lensing_tables`);
   :func:`lensed_tt` is then a handful of matrix products per cosmology and
   runs batched on the device.  It sums over every multipole;
   :func:`cl_on_every_ell` splines the sampled spectra onto them.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy import special

# ---------------------------------------------------------------------------
# 1. Lensing potential
# ---------------------------------------------------------------------------


def lensing_potential_operator(ells, k_values, tau_save, tau0_ref, dtype=jnp.float32):
    """``(n_ell, n_k, n_chi)`` table of ``j_l(k chi)`` with the ``chi`` trapezoid weights folded in.

    ``chi = tau0_ref - tau_save`` is the shared comoving-distance grid the
    sources are shifted onto; the per-cosmology lensing kernel is applied in
    :func:`lensing_potential_cl`.  Built on the host with SciPy's spherical
    Bessel functions (a second or so for 60 multipoles and 128 modes).
    """

    ells = np.asarray(ells, dtype=np.int64)
    k_values = np.asarray(k_values, dtype=np.float64)
    tau_save = np.asarray(tau_save, dtype=np.float64)
    chi = tau0_ref - tau_save
    weights = np.zeros(len(tau_save))
    weights[1:] += 0.5 * np.abs(np.diff(chi))
    weights[:-1] += 0.5 * np.abs(np.diff(chi))
    x = k_values[:, None] * chi[None, :]
    table = np.stack([special.spherical_jn(int(ell), x) * weights for ell in ells])
    return jnp.asarray(table, dtype=dtype)


LIMBER_LMIN = 200
"""Multipole from which the lensing potential is projected in the Limber approximation.

The exact line-of-sight integral needs the save grid to resolve ``j_l(k chi)``
along the whole line of sight, and the late-time part of the grid is coarse
(it is shaped for recombination); the Limber integrand is smooth in ``chi``.
Against CLASS the exact projection is good to 0.2 percent below ``l ~ 300``
and the Limber one to 0.3 percent above ``l ~ 200``.
"""


def limber_interpolation(ells, k_values, chi):
    """``(n_ell, n_chi, n_k)`` cubic-spline weights in ``ln k`` taking the modes to ``k = (l + 1/2) / chi``.

    Targets outside the solved modes get zero weight (no contribution).
    """

    from .spline_interpolation import spline_interpolation

    ells = np.asarray(ells, dtype=np.float64)
    k_values = np.asarray(k_values, dtype=np.float64)
    chi = np.asarray(chi, dtype=np.float64)
    chi_safe = np.where(chi > 0.0, chi, 1.0)
    target = (ells[:, None] + 0.5) / chi_safe[None, :]
    inside = (chi > 0.0)[None, :] & (target >= k_values[0]) & (target <= k_values[-1])
    lnk_t = jnp.log(jnp.asarray(np.where(inside, target, k_values[0])))
    weights = spline_interpolation(jnp.log(jnp.asarray(k_values)), jnp.eye(len(k_values))).evaluate(lnk_t.ravel())
    weights = weights.reshape(len(ells), len(chi), len(k_values)) * jnp.asarray(inside)[..., None]
    return weights.astype(jnp.float32)


def _shift(phi, shift_idx, shift_w):
    return sum(
        shift_w[:, None, :, a] * jnp.take_along_axis(phi, (shift_idx + a)[:, None, :], axis=2)
        for a in range(shift_w.shape[-1])
    )  # (c, k, chi)


def _lensing_kernel(chi, chi_star):
    chi_s = chi_star[:, None]
    return jnp.where((chi > 0.0) & (chi < chi_s), (chi_s - chi) / jnp.where(chi > 0.0, chi, 1.0) / chi_s, 0.0)


@jax.jit
def lensing_potential_cl(phi, shift_idx, shift_w, chi, chi_star, operator, limber_weights, ells_limber, k_values, n_s, A_s, k_pivot):
    """``C_l^{psi psi}`` for a batch of cosmologies, shape ``(n_c, n_ell_exact + n_ell_limber)``.

    ``phi`` is the Weyl potential channel ``(n_c, n_k, n_tau)`` on the save
    grid; ``shift_idx`` and ``shift_w`` are each cosmology's stencil onto the
    shared ``chi`` grid (:func:`discoeb.cmb._lagrange_shift_stencil`), ``chi``
    the shared grid ``(n_chi,)``, ``chi_star`` the comoving distance to each
    cosmology's visibility peak ``(n_c,)``, ``operator`` the table of
    :func:`lensing_potential_operator` for the multipoles projected exactly,
    ``limber_weights`` the table of :func:`limber_interpolation` and
    ``ells_limber`` the multipoles projected in the Limber approximation,
    ``k_values`` the solved modes in ascending order.
    """

    hi = jax.lax.Precision.HIGHEST
    shifted = _shift(phi, shift_idx, shift_w)
    kernel = _lensing_kernel(chi, chi_star)
    source = -2.0 * kernel[:, None, :] * shifted  # (c, k, chi)
    primordial = A_s[:, None] * (k_values[None, :] / k_pivot) ** (n_s[:, None] - 1.0)
    # Exact projection: C_l = 4 pi int dln k P_R(k) Theta_l(k)^2.
    theta = jnp.einsum("ckj,lkj->clk", source.astype(operator.dtype), operator, precision=hi).astype(jnp.float64)
    lnk = jnp.log(k_values)
    w = jnp.zeros_like(lnk).at[1:].add(0.5 * jnp.diff(lnk)).at[:-1].add(0.5 * jnp.diff(lnk))
    cl_exact = 4.0 * jnp.pi * jnp.einsum("clk,ck,k->cl", theta**2, primordial, w)
    # Limber: C_l = 2 pi^2 int dchi S(k_chi, chi)^2 P_R(k_chi) / (k_chi^3 chi^2), k_chi = (l + 1/2) / chi.
    source_at = jnp.einsum("ljk,ckj->clj", limber_weights, source.astype(jnp.float32), precision=hi).astype(jnp.float64)
    chi_safe = jnp.where(chi > 0.0, chi, 1.0)
    k_chi = (ells_limber[:, None] + 0.5) / chi_safe[None, :]  # (l, chi)
    p_r = A_s[:, None, None] * (k_chi[None] / k_pivot) ** (n_s[:, None, None] - 1.0)
    dchi = jnp.zeros_like(chi).at[1:].add(0.5 * jnp.abs(jnp.diff(chi))).at[:-1].add(0.5 * jnp.abs(jnp.diff(chi)))
    integrand = source_at**2 * p_r / (k_chi[None] ** 3 * chi_safe[None, None] ** 2) * dchi
    cl_limber = 2.0 * jnp.pi**2 * jnp.sum(jnp.where(chi > 0.0, integrand, 0.0), axis=-1)
    return jnp.concatenate([cl_exact, cl_limber], axis=1)


# ---------------------------------------------------------------------------
# 2. Lensing of the temperature spectrum
# ---------------------------------------------------------------------------


class LensingTables(NamedTuple):
    """Gauss-Legendre nodes and the Wigner d-functions on them, for :func:`lensed_tt`."""

    lmax: int
    x: jax.Array  # (n,) cos(beta)
    weights: jax.Array  # (n,)
    d00: jax.Array  # (lmax + 1, n)
    d11: jax.Array
    d1m1: jax.Array
    d2m2: jax.Array


def _wigner_d(m, mp, x, lmax):
    """``d^l_{m m'}(beta)`` for ``l = 0..lmax`` on ``x = cos(beta)``, by upward recursion."""

    x = np.asarray(x, dtype=np.float64)
    out = np.zeros((lmax + 1, len(x)))
    l0 = max(abs(m), abs(mp))
    start = {
        (0, 0): np.ones_like(x),
        (1, 1): 0.5 * (1.0 + x),
        (1, -1): 0.5 * (1.0 - x),
        (2, -2): (0.5 * (1.0 - x)) ** 2,
    }[(m, mp)]
    out[l0] = start
    prev = np.zeros_like(x)
    if l0 == 0:  # the recursion divides by l; seed it with d^1_00 = cos(beta)
        out[1], prev, l0 = x, out[0], 1
    for l in range(l0, lmax):
        num = (2 * l + 1) * (l * (l + 1) * x - m * mp) * out[l] - (l + 1) * np.sqrt((l * l - m * m) * (l * l - mp * mp)) * prev
        den = l * np.sqrt(((l + 1) ** 2 - m * m) * ((l + 1) ** 2 - mp * mp))
        prev = out[l]
        out[l + 1] = num / den
    return out


def lensing_tables(lmax: int, n_points: int | None = None, dtype=jnp.float32) -> LensingTables:
    """Tabulate the quadrature and the d-functions for :func:`lensed_tt`.

    ``n_points`` Gauss-Legendre nodes in ``cos(beta)`` (``lmax + 1024`` by
    default, which resolves the lensed correlation function to better than
    1e-4 of the spectrum).  Takes a few seconds of host time for ``lmax ~ 3000``
    and ~200 MB on the device in float32.
    """

    n = n_points or (lmax + 1024)
    x, w = np.polynomial.legendre.leggauss(n)
    return LensingTables(
        lmax=lmax,
        x=jnp.asarray(x, dtype=jnp.float64),
        weights=jnp.asarray(w, dtype=jnp.float64),
        d00=jnp.asarray(_wigner_d(0, 0, x, lmax), dtype=dtype),
        d11=jnp.asarray(_wigner_d(1, 1, x, lmax), dtype=dtype),
        d1m1=jnp.asarray(_wigner_d(1, -1, x, lmax), dtype=dtype),
        d2m2=jnp.asarray(_wigner_d(2, -2, x, lmax), dtype=dtype),
    )


def cl_on_every_ell(ells, cl, lmax: int, power: int = 1):
    """``C_l`` at every ``l = 0..lmax``, ``(n_c, lmax + 1)``, from samples ``(n_c, n_ell)`` at sorted ``ells``.

    :func:`lensed_tt` sums over every multipole, the spectra are computed at a
    few dozen; this is a natural cubic spline in ``ln l`` of
    ``[l(l+1)]^power C_l`` (``power = 1`` for the temperature, 2 for the
    lensing potential, so that the splined quantity is flat), zero below
    ``ells[0]``. ``ells[-1]`` must reach ``lmax``. Multipoles the spline
    must bridge where the spectrum curves sharply in ``ln l`` cost accuracy: a
    temperature sampled at ``l = 2, 4`` is off by 5e-3 at ``l = 3``, so sample
    every multipole there.
    """

    from .spline_interpolation import spline_interpolation

    ells = np.asarray(ells, dtype=np.float64)
    if ells[-1] < lmax:
        raise ValueError(f"samples end at l = {ells[-1]:.0f}, below lmax = {lmax}")
    ell = np.arange(lmax + 1, dtype=np.float64)
    inside = ell >= ells[0]
    weight = (ells * (ells + 1.0)) ** power
    spline = spline_interpolation(jnp.log(jnp.asarray(ells)), (jnp.asarray(cl, dtype=jnp.float64) * weight).T)
    values = spline.evaluate(jnp.log(jnp.asarray(np.where(inside, ell, ells[0]))))  # (lmax + 1, n_c)
    unweight = np.where(inside, ell * (ell + 1.0), 1.0) ** power
    return jnp.where(inside[None, :], (values / unweight[:, None]).T, 0.0)


def _lensed_tt_one(cl_tt, cl_pp, tables: LensingTables):
    """Lensed ``C_l`` for ``l = 0..lmax`` of one cosmology (float64 in, float64 out)."""

    hi = jax.lax.Precision.HIGHEST
    dtype = tables.d00.dtype
    ell = jnp.arange(tables.d00.shape[0], dtype=jnp.float64)  # 0..lmax (a static shape, unlike tables.lmax under jit)
    llp1 = ell * (ell + 1.0)
    norm = (2.0 * ell + 1.0) / (4.0 * jnp.pi)
    # Deflection correlation functions on the quadrature nodes.
    a = (norm * llp1 * cl_pp).astype(dtype)
    c_gl = jnp.einsum("l,ln->n", a, tables.d11, precision=hi)
    c_gl2 = jnp.einsum("l,ln->n", a, tables.d1m1, precision=hi)
    sigma2 = (jnp.sum(a.astype(jnp.float64)) - c_gl.astype(jnp.float64)).astype(dtype)
    # xi~(beta) = sum_l b_l X_000^2 [d00 + (l(l+1)/2) C_gl,2 d1m1 + C_gl,2^2 c_l^2 d2m2],
    # since X'_000 = -l(l+1)/4 X_000 and X_220 = c_l X_000, c_l^2 = (l+2)(l-1)l(l+1)/16.
    b = (norm * cl_tt).astype(dtype)
    half = (0.5 * llp1).astype(dtype)
    c2 = ((ell + 2.0) * (ell - 1.0) * llp1 / 16.0).astype(dtype)
    damping = jnp.exp(-0.5 * llp1.astype(dtype)[:, None] * sigma2[None, :])  # X_000^2, (l, n)
    xi = (
        jnp.einsum("l,ln,ln->n", b, damping, tables.d00, precision=hi)
        + c_gl2 * jnp.einsum("l,ln,ln->n", b * half, damping, tables.d1m1, precision=hi)
        + c_gl2**2 * jnp.einsum("l,ln,ln->n", b * c2, damping, tables.d2m2, precision=hi)
    )
    integrand = (tables.weights * xi.astype(jnp.float64)).astype(dtype)
    return 2.0 * jnp.pi * jnp.einsum("ln,n->l", tables.d00, integrand, precision=hi).astype(jnp.float64)


@partial(jax.jit, static_argnames=("chunk",))
def lensed_tt(cl_tt, cl_pp, tables: LensingTables, chunk: int = 4):
    """Lensed ``C_l^{TT}`` for a batch, shape ``(n_c, lmax + 1)``.

    ``cl_tt`` and ``cl_pp`` are the unlensed temperature and the lensing
    potential spectra, ``(n_c, lmax + 1)`` indexed by ``l`` from 0 (the
    monopole and dipole are ignored; :func:`cl_on_every_ell` fills them in from
    the sampled spectra of :func:`discoeb.cmb.compute_cl_power_spectrum_batch`).  The unlensed spectra must extend a few
    hundred multipoles past the highest lensed multipole wanted.  Cosmologies
    are processed ``chunk`` at a time: the per-cosmology work is an
    ``(lmax + 1, n_points)`` array of damping factors.
    """

    cl_tt = jnp.asarray(cl_tt, dtype=jnp.float64).at[:, :2].set(0.0)
    cl_pp = jnp.asarray(cl_pp, dtype=jnp.float64).at[:, :2].set(0.0)
    n_c = cl_tt.shape[0]
    pad = (-n_c) % chunk
    cl_tt = jnp.pad(cl_tt, ((0, pad), (0, 0)))
    cl_pp = jnp.pad(cl_pp, ((0, pad), (0, 0)))
    one = jax.vmap(_lensed_tt_one, in_axes=(0, 0, None))
    out = jax.lax.map(lambda args: one(args[0], args[1], tables), (cl_tt.reshape(-1, chunk, cl_tt.shape[1]), cl_pp.reshape(-1, chunk, cl_pp.shape[1])))
    return out.reshape(-1, cl_tt.shape[1])[:n_c]
