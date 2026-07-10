"""CMB temperature source functions, transfer functions, and power spectra.

Once the perturbation hierarchy has been evolved
(:mod:`discoeb.perturbations_system`), the CMB anisotropy today is obtained by
*line-of-sight integration* (Seljak & Zaldarriaga 1996): rather than following
the photon multipole hierarchy up to ``l ~ 2500``, the temperature transfer
function is written as a single integral of a few local source terms against
spherical Bessel functions,

    ``Theta_l(k) = int_0^{tau_0} dtau [S j_l(x) + S_j1 j_l'(x) + S_j2 j_l''(x)]``

with ``x = k (tau_0 - tau)``. The sources are built from the saved perturbation
history and the visibility function ``g = kappa' exp(-kappa)``:

    * ``S``    -- the intrinsic monopole (Sachs-Wolfe) and polarization
      correction, weighted by ``g``, plus the integrated Sachs-Wolfe term
      ``2 Phi' exp(-kappa)`` which accumulates along the whole line of sight;
    * ``S_j1`` -- the Doppler term ``g (sigma + v_b)``;
    * ``S_j2`` -- the quadrupole/polarization term.

The angular power spectrum then follows from the primordial curvature spectrum,

    ``C_l ~ A_s int dln k (k/k_p)^{n_s - 1} Theta_l(k)^2``,

and is reported as ``D_l = l(l+1) C_l`` in ``muK^2`` (the DISCO-EB
normalization; see :func:`dl_power_spectrum`).

The perturbation solve runs on the GPU (numba-CUDA Rodas5P); the source
construction, Bessel projection, and ``C_l`` quadrature run in JAX/NumPy on the
host. Ported from the DISCO2 prototype's ``cmb.py``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from scipy import integrate, interpolate, special

from .background import dtau_da, grhoa4
from .perturbations import (
    IX_CLXB,
    IX_CLXC,
    IX_ETAK,
    IX_G,
    IX_POL,
    IX_R,
    IX_VB,
    LMAX_G,
    LMAX_NR,
    LMAX_POL,
    anisotropic_stress,
    comoving_densities,
    density_perturbation,
    expansion_rate,
    momentum_perturbation,
)
from .perturbations_system import (
    TAU_START,
    build_thermo_tables,
    density_coefficients,
    solve_perturbation_history,
)

DEFAULT_CMB_K = np.geomspace(1.0e-4, 1.0, 128, dtype=np.float64)
"""Default perturbation wave modes (Mpc^-1) for a CMB solve."""

DEFAULT_CMB_ELLS = np.unique(
    np.concatenate([np.arange(2, 40, 2), np.arange(40, 200, 5), np.arange(200, 2501, 50)])
).astype(np.int64)
"""Default multipoles at which ``C_l`` is evaluated."""

Z_REION_SEARCH_MAX = 30.0
"""Redshift below which the reionization visibility bump is located."""

_BESSEL_CACHE: dict[tuple, tuple] = {}
_CMB_GRID_CACHE: dict[tuple, dict[str, float]] = {}


# =============================================================================
# 1. Finite differences and the visibility function
# =============================================================================


def _grad_last(y, x):
    """Return ``dy/dx`` along the last axis by centered differences."""

    mid = (y[..., 2:] - y[..., :-2]) / (x[..., 2:] - x[..., :-2])
    first = (y[..., 1:2] - y[..., :1]) / (x[..., 1:2] - x[..., :1])
    last = (y[..., -1:] - y[..., -2:-1]) / (x[..., -1:] - x[..., -2:-1])
    return jnp.concatenate([first, mid, last], axis=-1)


def _visibility_from_opacity(tau, opacity):
    """Return the optical depth and visibility function from the Thomson opacity.

    The optical depth is integrated *backwards* from today, ``kappa(tau_0) = 0``,
    so ``kappa(tau) = int_tau^{tau_0} kappa' dtau'``. The visibility function
    ``g = kappa' exp(-kappa)`` is the probability density that a photon seen
    today last scattered at ``tau``; it is sharply peaked at recombination, with
    a secondary bump at reionization.
    """

    dt = tau[..., 1:] - tau[..., :-1]
    integ = 0.5 * (opacity[..., 1:] + opacity[..., :-1]) * dt
    optical_depth = jnp.concatenate(
        [jnp.cumsum(integ[..., ::-1], axis=-1)[..., ::-1], jnp.zeros_like(tau[..., :1])],
        axis=-1,
    )
    gvis = opacity * jnp.exp(-optical_depth)
    gvisprime = _grad_last(gvis, tau)
    return {
        "opac": opacity,
        "opacprime": _grad_last(opacity, tau),
        "gvis": gvis,
        "gvisprime": gvisprime,
        "gvispprime": _grad_last(gvisprime, tau),
        "optical_depth": optical_depth,
    }


def visibility_functions(tau, opacity):
    """Return ``opacity``, ``optical_depth``, and the visibility ``g`` and derivatives."""

    out = _visibility_from_opacity(jnp.asarray(tau), jnp.asarray(opacity))
    return {
        "opacity": out["opac"],
        "optical_depth": out["optical_depth"],
        "g": out["gvis"],
        "gprime": out["gvisprime"],
        "gpprime": out["gvispprime"],
    }


# =============================================================================
# 2. Source functions
# =============================================================================


def extract_perturbations(yout, youtprime):
    """Name the perturbation state entries the source construction consumes."""

    zeros = jnp.zeros_like(yout[..., IX_G])
    return {
        "eta": yout[..., IX_ETAK],
        "deltac": yout[..., IX_CLXC],
        "deltab": yout[..., IX_CLXB],
        "thetab": yout[..., IX_VB],
        "deltag": yout[..., IX_G],
        "thetag": yout[..., IX_G + 1],
        "pig": yout[..., IX_G + 2],
        "shearg": yout[..., IX_G + 2] / 2.0,
        "theta3": yout[..., IX_G + 3] if LMAX_G >= 3 else zeros,
        "deltar": yout[..., IX_R],
        "thetar": yout[..., IX_R + 1],
        "pir": yout[..., IX_R + 2],
        "shearr": yout[..., IX_R + 2] / 2.0,
        "n3": yout[..., IX_R + 3] if LMAX_NR >= 3 else zeros,
        "e2": yout[..., IX_POL] if LMAX_POL >= 2 else zeros,
        "deltagprime": youtprime[..., IX_G],
        "thetagprime": youtprime[..., IX_G + 1],
    }


def compute_metric_perturbations(perturbations, dens, a, k_values):
    """Return the synchronous-gauge metric variables along each mode's history.

    ``z`` and ``sigma`` are the metric expansion and shear. ``phi`` is the *Weyl*
    potential ``(Phi + Psi) / 2`` -- the mean of the two Newtonian-gauge
    potentials, which differ by the anisotropic stress, ``Psi - Phi = -dgpi/k^2``.
    Twice its conformal-time derivative is the full ISW source ``Phi' + Psi'``.
    """

    grhog_v, grhor_v, grhoc_v, grhob_v, grhov_v = dens
    grhog_t, grhor_t, grhoc_t, grhob_t, grhov_t = comoving_densities(
        a, grhog_v, grhor_v, grhoc_v, grhob_v, grhov_v
    )
    aprimeoa = expansion_rate(grhog_t, grhor_t, grhoc_t, grhob_t, grhov_t)
    k = jnp.asarray(k_values, dtype=jnp.float64)[:, None]

    dgrho = density_perturbation(
        grhob_t, perturbations["deltab"],
        grhoc_t, perturbations["deltac"],
        grhog_t, perturbations["deltag"],
        grhor_t, perturbations["deltar"],
    )
    dgtheta = momentum_perturbation(
        grhob_t, perturbations["thetab"],
        grhog_t, perturbations["thetag"],
        grhor_t, perturbations["thetar"],
    )
    dgshear = anisotropic_stress(
        grhog_t, 2.0 * perturbations["shearg"], grhor_t, 2.0 * perturbations["shearr"]
    )

    z = (0.5 * dgrho / k + perturbations["eta"]) / aprimeoa
    sigma = z + 1.5 * dgtheta / (k * k)
    phi = -((dgrho + 3.0 * dgtheta * aprimeoa / k) + dgshear) / (2.0 * k * k)
    return {
        "aprimeoa": aprimeoa,
        "grhog_t": grhog_t,
        "grhor_t": grhor_t,
        "grhoc_t": grhoc_t,
        "grhob_t": grhob_t,
        "dgrho": dgrho,
        "dgtheta": dgtheta,
        "dgshear": dgshear,
        "z": z,
        "sigma": sigma,
        "phi": phi,
    }


def compute_source_function(perturbations, metric, vis, kmodes, tau):
    """Return the line-of-sight temperature sources ``S``, ``S_j1``, ``S_j2``.

    The decomposition follows CAMB: the ``j_l`` source carries the intrinsic
    monopole and the ISW term, the ``j_l'`` source carries the Doppler term, and
    the ``j_l''`` source carries the quadrupole/polarization term. ``polter`` is
    the combination of the photon quadrupole and the E-mode ``E_2`` that Thomson
    scattering couples to.

    Note that ``metric["phi"]`` is the *Weyl* potential ``(Phi + Psi) / 2``, not
    ``Phi``: it carries the ``dgpi`` term, and ``Psi - Phi = -dgpi / k^2``. That
    is what makes ``isw = 2 phi' exp(-kappa)`` the *complete* ISW source,
    ``exp(-kappa) (Phi' + Psi')``. CLASS instead splits the ISW across two
    channels, ``exp(-kappa) 2 Phi'`` against ``j_l`` and ``exp(-kappa) k (Psi -
    Phi)`` against ``j_l'``; integrating the latter by parts recovers the form
    used here together with an extra ``g (Psi - Phi)`` monopole contribution,
    and indeed ``monopole`` below reduces identically to ``delta_g/4 + alpha'``.
    So there is no separate anisotropic-stress term to add.
    """

    kmode = jnp.asarray(kmodes, dtype=jnp.float64)[:, None]
    polter = perturbations["pig"] / 10.0 + 9.0 / 15.0 * perturbations["e2"]
    phi = metric["phi"]

    phidot = _grad_last(phi, tau)
    isw = 2.0 * phidot * jnp.exp(-vis["optical_depth"])
    monopole = -perturbations["eta"] / kmode + 2.0 * phi + 0.25 * perturbations["deltag"]

    s_j0 = isw + vis["gvis"] * (monopole + 0.625 * polter)
    s_j1 = vis["gvis"] * (metric["sigma"] + perturbations["thetab"])
    s_j2 = 1.875 * vis["gvis"] * polter
    return {"S": s_j0, "S_j1": s_j1, "S_j2": s_j2, "isw": isw}


def source_functions(y, tau, a, k, opacity, dens):
    """Return the CMB source functions from a saved perturbation history.

    ``y`` has shape ``(n_k, n_tau, nvar)``; ``tau``, ``a`` and ``opacity`` are
    shared across modes and have shape ``(n_tau,)``.
    """

    tau = jnp.asarray(tau, dtype=jnp.float64)
    a = jnp.asarray(a, dtype=jnp.float64)
    y = jnp.asarray(y, dtype=jnp.float64)

    # d/dtau of every state variable, differentiating along the tau axis.
    yprime = jnp.moveaxis(_grad_last(jnp.moveaxis(y, 1, -1), tau), -1, 1)
    perturb = extract_perturbations(y, yprime)
    metric = compute_metric_perturbations(perturb, dens, a[None, :], k)
    vis = _visibility_from_opacity(tau[None, :], jnp.asarray(opacity)[None, :])
    return compute_source_function(perturb, metric, vis, k, tau[None, :])


# =============================================================================
# 3. Bessel projection and the C_l quadrature
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


def theta_ell_transfer_function(
    ells, k_values, tau, S, tau0, S_j1=None, S_j2=None, *, bessel_dx: float = 0.03
):
    """Return the temperature transfer function ``Theta_l(k)``, shape ``(n_k, n_ell)``.

    Derivatives of ``j_l`` come from the recurrences
    ``j_l' = (l/x) j_l - j_{l+1}`` and
    ``j_l'' = -(2/x) j_l' + (l(l+1)/x^2 - 1) j_l``,
    so only ``j_l`` and ``j_{l+1}`` need tabulating.

    Each multipole is restricted to the band of wavenumbers that actually
    projects onto it. ``j_l(x)`` is exponentially small below its turning point,
    so modes with ``k chi_max < l - 4 l^{1/3}`` contribute nothing; and modes far
    above ``k ~ l / chi_star`` only contribute through rapidly oscillating tails
    that the finite ``tau`` and ``k`` grids cannot resolve, so integrating them
    injects aliasing noise rather than signal. Without the upper cut the damping
    tail is inflated by several percent (``+6%`` at ``l = 500``).
    """

    ell_arr = np.atleast_1d(np.asarray(ells, dtype=np.int64))
    k = jnp.asarray(k_values, dtype=jnp.float64)
    tau = jnp.asarray(tau, dtype=jnp.float64)
    S = jnp.asarray(S, dtype=jnp.float64)
    S_j1 = jnp.zeros_like(S) if S_j1 is None else jnp.asarray(S_j1, dtype=jnp.float64)
    S_j2 = jnp.zeros_like(S) if S_j2 is None else jnp.asarray(S_j2, dtype=jnp.float64)

    chi = jnp.asarray(tau0, dtype=jnp.float64) - tau
    x = k[:, None] * chi
    x0, inv_dx, n_x, jl_tab, jl1_tab = _build_bessel_tables(
        ell_arr, float(jnp.max(x)) + 2.0, bessel_dx
    )

    # Comoving distance to the peak of the source (last scattering), which sets
    # the k -> l mapping k ~ l / chi_star.
    chi_max = jnp.maximum(jnp.max(chi), 1.0)
    chi_star = jnp.maximum(chi[jnp.argmax(jnp.max(jnp.abs(S), axis=0))], 1.0)

    def one_ell(i, ell_value):
        jl = _interp_uniform_table(x, x0, inv_dx, n_x, jl_tab[i])
        jl1 = _interp_uniform_table(x, x0, inv_dx, n_x, jl1_tab[i])
        inv_x = jnp.where(x > 1.0e-30, 1.0 / x, 0.0)
        jld = jnp.where(x > 1.0e-30, ell_value * inv_x * jl - jl1, 0.0)
        jldd = jnp.where(
            x > 1.0e-30,
            -2.0 * inv_x * jld + (ell_value * (ell_value + 1.0) * inv_x * inv_x - 1.0) * jl,
            0.0,
        )
        theta_ell = jnp.trapezoid(S * jl + S_j1 * jld + S_j2 * jldd, tau, axis=1)

        k_lo = jnp.maximum(0.0, ell_value - 4.0 * ell_value ** (1.0 / 3.0)) / chi_max
        k_hi = (ell_value + 2500.0) / chi_star
        return jnp.where((k >= k_lo) & (k <= k_hi), theta_ell, 0.0)

    theta = jax.vmap(one_ell)(
        jnp.arange(ell_arr.size), jnp.asarray(ell_arr, dtype=jnp.float64)
    )
    return theta.T


def compute_Cell(theta_ell, kmodes, n_s, k_p):
    """Return the un-normalized ``C_l = int dln k (k/k_p)^{n_s-1} Theta_l(k)^2``."""

    return jnp.trapezoid(
        (kmodes[:, None] / k_p) ** (n_s - 1.0) * theta_ell**2,
        x=jnp.log(kmodes),
        axis=0,
    )


def cl_power_spectrum(theta_ell, k_values, n_s, k_pivot, A_s):
    """Return the amplitude-normalized ``C_l``."""

    return A_s * compute_Cell(theta_ell, jnp.asarray(k_values), n_s, k_pivot)


def dl_power_spectrum(cl, ells, T_cmb):
    """Return ``D_l = l(l+1) C_l`` in ``muK^2`` (DISCO-EB normalization)."""

    ell = jnp.asarray(ells, dtype=jnp.float64)
    return ell * (ell + 1.0) * jnp.asarray(cl) * 2.0 * T_cmb**2 * 1.0e12


# =============================================================================
# 4. Adaptive k and tau sampling
# =============================================================================


def _cmb_grid_summary(cosmology) -> dict[str, float]:
    """Return the characteristic scales that set the adaptive ``k`` / ``tau`` grids.

    ``tau_star`` is the visibility peak and ``delta_tau_rec`` its Gaussian
    equivalent width; ``r_s`` is the sound horizon at recombination (which sets
    the acoustic oscillation period in ``k``) and ``k_D`` the Silk damping
    wavenumber (which sets their envelope). Together they say where the CMB
    sources have structure, and hence where the grids must be dense.
    """

    key = (
        cosmology.omega_b_h2, cosmology.omega_c_h2, cosmology.h, cosmology.T_cmb,
        cosmology.Y_He, cosmology.N_eff, cosmology.tau_reion, cosmology.n_s,
        cosmology.Omegak, cosmology.w_DE_0, cosmology.w_DE_a,
        cosmology.mnu, cosmology.num_massive_neutrinos,
    )
    if key in _CMB_GRID_CACHE:
        return _CMB_GRID_CACHE[key]

    dens = density_coefficients(cosmology)
    tau_grid, values, _, tau0 = build_thermo_tables(cosmology)
    a_vals, opacity = values[0], values[1]

    vis = np.asarray(
        _visibility_from_opacity(jnp.asarray(tau_grid), jnp.asarray(opacity))["gvis"]
    )
    peak = int(np.argmax(vis))
    tau_star = float(tau_grid[peak])
    a_star = float(a_vals[peak])

    half = 0.5 * float(vis[peak])
    left = float(np.interp(half, vis[: peak + 1], tau_grid[: peak + 1]))
    right = float(np.interp(half, vis[peak:][::-1], tau_grid[peak:][::-1]))
    delta_tau_rec = max((right - left) / (2.0 * np.sqrt(2.0 * np.log(2.0))), 1.0)

    grhog_v, grhor_v, grhoc_v, grhob_v, _ = dens
    tau_eq = float(np.interp((grhog_v + grhor_v) / (grhoc_v + grhob_v), a_vals, tau_grid))

    # Comoving sound horizon r_s = int c_s dtau = int_0^{a_star} da / (a' sqrt(3(1+R))),
    # with a' = a sqrt(grhoa2/3) and R = 3 rho_b / (4 rho_g). Since
    # grhoa4 = grhoa2 a^2, the a' and the 1/a combine into 1/sqrt(grhoa4).
    def sound_integrand(ap):
        r = 0.75 * grhob_v * ap / grhog_v
        return 1.0 / np.sqrt(grhoa4(ap, *dens) * (1.0 + r))

    r_s = float(integrate.quad(sound_integrand, 0.0, a_star, limit=200)[0])

    # Silk damping scale from the photon diffusion length.
    a_silk = np.linspace(max(float(a_vals[0]), 1.0e-8), a_star, 5000)
    opacity_silk = np.maximum(np.interp(a_silk, a_vals, opacity), 1.0e-30)
    r = 0.75 * grhob_v * a_silk / grhog_v
    integrand_d = (
        (r**2 + 16.0 * (1.0 + r) / 15.0)
        / (6.0 * (1.0 + r) ** 2 * opacity_silk)
        * np.array([dtau_da(float(ap), *dens) for ap in a_silk])
    )
    k_D = float(1.0 / np.sqrt(np.trapezoid(integrand_d, a_silk)))

    # Secondary visibility bump from reionization. Selecting on "after
    # recombination" is not enough -- the recombination tail still dominates
    # there -- so restrict to z < Z_REION_SEARCH_MAX, where only reionization
    # scatters.
    post_rec = a_vals > 1.0 / (1.0 + Z_REION_SEARCH_MAX)
    if np.any(post_rec) and np.max(np.where(post_rec, vis, 0.0)) > 0.0:
        reion_idx = int(np.argmax(np.where(post_rec, vis, 0.0)))
        tau_reion = float(tau_grid[reion_idx])
        delta_tau_reion = max(0.03 * tau0, 80.0)
    else:
        tau_reion = 0.6 * tau0
        delta_tau_reion = 0.05 * tau0

    out = {
        "tau0": float(tau0), "tau_eq": tau_eq, "tau_star": tau_star,
        "delta_tau_rec": delta_tau_rec, "tau_reion": tau_reion,
        "delta_tau_reion": delta_tau_reion, "r_s": r_s, "k_D": k_D,
    }
    _CMB_GRID_CACHE[key] = out
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

    summary = _cmb_grid_summary(cosmology)
    x = np.linspace(np.log(k_min), np.log(k_max), n_eval)
    k = np.exp(x)
    primordial = k ** (cosmology.n_s + 2.0)
    acoustic_curv = (1.0 / summary["r_s"]) ** 2
    damped = primordial * np.exp(-((k / summary["k_D"]) ** 2))

    if mode == "cl":
        sigma_k = 1.0 / summary["delta_tau_rec"]
        chi_star = summary["tau0"] - summary["tau_star"]
        ell_samples = np.unique(np.geomspace(ell_min, ell_max, n_ell_samples).astype(int))
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


def cmb_tau_grid(cosmology, k_values, n: int, *, n_eval: int = 10000) -> np.ndarray:
    """Return a nonuniform conformal-time grid on which to save the sources.

    The sources are sharply peaked at recombination (width ``delta_tau_rec``),
    modulated on the sound-horizon scale, and have a secondary bump at
    reionization, so the sampling density is the cube root of the sum of those
    three weights. Endpoints are pinned to ``[TAU_START, tau0]``, which the
    solver's save grid requires.
    """

    k_arr = np.asarray(k_values, dtype=np.float64)
    s = _cmb_grid_summary(cosmology)
    tau0 = s["tau0"]

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

    tau_save = _resample_from_density(tau, (weight + 0.005 * np.max(weight)) ** (1.0 / 3.0), n)
    tau_save[0], tau_save[-1] = TAU_START, tau0
    return tau_save


# =============================================================================
# 5. End-to-end spectrum
# =============================================================================


def _interpolate_sources_to_fine_k(k_values, channels, k_fine):
    """Akima-interpolate each source channel from the solve grid onto a finer ``k``."""

    log_k = np.log(np.asarray(k_values, dtype=np.float64))
    log_k_fine = np.log(np.asarray(k_fine, dtype=np.float64))
    out = []
    for channel in channels:
        channel = np.asarray(channel, dtype=np.float64)
        fine = np.empty((log_k_fine.size, channel.shape[1]), dtype=np.float64)
        for it in range(channel.shape[1]):
            fine[:, it] = interpolate.Akima1DInterpolator(log_k, channel[:, it])(log_k_fine)
        out.append(fine)
    return out


def compute_cl_power_spectrum(
    cosmology,
    *,
    k_values: np.ndarray | None = None,
    ells: np.ndarray | None = None,
    n_save: int = 1000,
    n_k_fine: int = 1500,
    k_fine_values: np.ndarray | None = None,
    **solve_kwargs,
):
    """Return ``(ells, C_l, D_l)`` for the unlensed scalar TT spectrum.

    Runs the numba-CUDA perturbation solve on ``k_values``, builds the
    line-of-sight sources on the saved ``tau`` grid, refines them in ``k``, and
    projects them onto spherical Bessel functions.
    """

    if k_values is None:
        k_values = cmb_k_grid(cosmology, n=128, mode="ode")
    if ells is None:
        ells = DEFAULT_CMB_ELLS
    k_values = np.asarray(k_values, dtype=np.float64)
    ells = np.asarray(ells, dtype=np.int64)

    tau_save = cmb_tau_grid(cosmology, k_values, n_save)
    hist, _, tables = solve_perturbation_history(
        k_values, cosmology, tau_save=tau_save, **solve_kwargs
    )
    tau_grid, values, _, tau0 = tables

    # a(tau) and kappa'(tau) on the save grid, from the thermodynamics tables.
    a_save = np.interp(tau_save, tau_grid, values[0])
    opacity_save = np.interp(tau_save, tau_grid, values[1])

    dens = density_coefficients(cosmology)
    src = source_functions(hist, tau_save, a_save, k_values, opacity_save, dens)

    if k_fine_values is None:
        k_fine = cmb_k_grid(
            cosmology, n=n_k_fine, mode="cl",
            k_min=float(k_values[0]), k_max=float(k_values[-1]),
            ell_max=int(np.max(ells)),
        )
    else:
        k_fine = np.asarray(k_fine_values, dtype=np.float64)

    s_j0, s_j1, s_j2 = _interpolate_sources_to_fine_k(
        k_values, (src["S"], src["S_j1"], src["S_j2"]), k_fine
    )

    theta = theta_ell_transfer_function(ells, k_fine, tau_save, s_j0, tau0, s_j1, s_j2)
    cl = cl_power_spectrum(theta, k_fine, cosmology.n_s, cosmology.k_pivot, cosmology.A_s)
    dl = dl_power_spectrum(cl, ells, cosmology.T_cmb)
    return ells, np.asarray(cl), np.asarray(dl)
