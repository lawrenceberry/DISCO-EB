"""Tanh reionization history (CAMB/CLASS ``reio_camb`` parameterization).

RECFAST follows the plasma through recombination and leaves it essentially
neutral (``x_e ~ 2e-4``). Reionization by the first sources re-ionizes hydrogen
(and singly-ionizes helium) around ``z ~ 7``, then doubly-ionizes helium around
``z ~ 3.5``. The resulting free electrons re-scatter CMB photons, damping the
temperature and polarization anisotropies by ``exp(-2 tau_reion)`` on scales
inside the horizon at reionization.

The ionization fraction is not solved from astrophysics but *parameterized* as a
smooth step, and the step's midpoint ``z_re`` is fixed by requiring the model to
reproduce the observed reionization optical depth ``tau_reion``:

    ``tau_reion = int_0^{z_start} x_e(z) akthom dtau/da dz``.

The step is taken in ``(1 + z)^{3/2}`` rather than ``z`` (a CAMB convention that
makes the width in redshift scale with the expansion), and helium's second
ionization is a separate, much narrower tanh at ``z = 3.5``.

Everything here is plain scalar/NumPy Python: reionization enters the solve only
through the precomputed ``x_e(tau)`` thermodynamics table.
"""

from __future__ import annotations

import numpy as np

from .background import dtau_da, helium_number_fraction

# CAMB's tanh-reionization shape parameters.
DELTA_Z = 0.5
"""Width of the hydrogen reionization step, in the ``(1+z)^{3/2}`` variable."""

HE_REION_Z = 3.5
"""Midpoint redshift of the second helium ionization."""

HE_REION_DZ = 0.4
"""Width (in ``z``) of the second helium ionization step."""

Z_RE_BRACKET = (2.0, 30.0)
"""Bracket searched for the reionization midpoint ``z_re``."""

Z_RE_FALLBACK = 7.5
"""Midpoint used when ``tau_reion`` lies outside the bracketed range."""


def _reion_start(z_re: float) -> float:
    """Return the redshift above which the tanh step is numerically zero."""

    return z_re + 8.0 * DELTA_Z


def reionization_xe(z, z_re: float, f_He: float, xe_before: float) -> np.ndarray:
    """Return the reionized free-electron fraction ``x_e(z)``.

    Parameters
    ----------
    z : array_like
        Redshifts at which to evaluate ``x_e``.
    z_re : float
        Midpoint of the hydrogen reionization step.
    f_He : float
        Helium-to-hydrogen number fraction, ``n_He / n_H``.
    xe_before : float
        Residual free-electron fraction left by recombination, used as the floor
        the step rises from.

    Notes
    -----
    Hydrogen reionization also singly-ionizes helium, so the fully reionized
    plateau is ``1 + f_He`` electrons per hydrogen. Helium's *second* ionization
    contributes another ``f_He`` at ``z ~ 3.5``.
    """

    z = np.asarray(z, dtype=np.float64)

    # Hydrogen (+ first helium ionization), stepped in (1+z)^{3/2}.
    window_mid = (1.0 + z_re) ** 1.5
    window_delta = 1.5 * np.sqrt(1.0 + z_re) * DELTA_Z
    xod = np.clip((window_mid - (1.0 + z) ** 1.5) / window_delta, -100.0, 100.0)
    x_h = (1.0 + f_He - xe_before) * 0.5 * (np.tanh(xod) + 1.0) + xe_before

    # Second helium ionization, a narrow step in z.
    xod_he = np.clip((HE_REION_Z - z) / HE_REION_DZ, -100.0, 100.0)
    x_he = np.where(
        z < HE_REION_Z + 5.0 * HE_REION_DZ,
        f_He * 0.5 * (np.tanh(xod_he) + 1.0),
        0.0,
    )
    return x_h + x_he


def reionization_optical_depth(
    z_re: float, f_He: float, xe_before: float, akthom: float, densities, *, n_z: int = 2000, **grho_kwargs
) -> float:
    """Return the Thomson optical depth ``tau`` accumulated by reionization.

    ``tau = int kappa' dtau = int x_e akthom a^-2 (dtau/da) da``, and with
    ``dz = -da / a^2`` this becomes an integral over redshift of
    ``x_e akthom (dtau/da)``.
    """

    z_grid = np.linspace(0.0, _reion_start(z_re), n_z)
    a_grid = 1.0 / (1.0 + z_grid)
    xe = reionization_xe(z_grid, z_re, f_He, xe_before)
    dtauda = np.array(
        [dtau_da(float(a), *densities, **grho_kwargs) for a in a_grid]
    )
    return float(np.trapezoid(xe * akthom * dtauda, z_grid))


def solve_reionization_redshift(
    tau_reion: float,
    f_He: float,
    xe_before: float,
    akthom: float,
    densities,
    *,
    tol: float = 1.0e-8,
    max_iter: int = 100,
    **grho_kwargs,
) -> float:
    """Return the midpoint ``z_re`` whose tanh step yields ``tau_reion``.

    Bisects :func:`reionization_optical_depth`, which is monotonic in ``z_re``.
    Returns ``-1.0`` when ``tau_reion <= 0`` (reionization disabled), and falls
    back to :data:`Z_RE_FALLBACK` when the target lies outside the bracket.
    """

    if tau_reion <= 0.0:
        return -1.0

    def residual(z_re: float) -> float:
        return (
            reionization_optical_depth(
                z_re, f_He, xe_before, akthom, densities, **grho_kwargs
            )
            - tau_reion
        )

    lo, hi = Z_RE_BRACKET
    f_lo, f_hi = residual(lo), residual(hi)
    if f_lo * f_hi > 0.0:
        return Z_RE_FALLBACK

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = residual(mid)
        if abs(f_mid) < tol or 0.5 * (hi - lo) < tol:
            return mid
        if f_lo * f_mid <= 0.0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return 0.5 * (lo + hi)


def apply_reionization(z_vals, xe_recomb, cosmology, akthom: float, densities, **grho_kwargs):
    """Return ``x_e(z)`` with reionization layered onto a recombination history.

    The two histories are combined with a maximum rather than a sum: the tanh
    step already rises from the residual recombination floor, so below the step
    it reproduces ``xe_recomb`` and above it the recombination value dominates.
    """

    if cosmology.tau_reion <= 0.0:
        return np.asarray(xe_recomb, dtype=np.float64)

    f_He = helium_number_fraction(cosmology.Y_He)
    z_vals = np.asarray(z_vals, dtype=np.float64)
    xe_recomb = np.asarray(xe_recomb, dtype=np.float64)

    z_re = solve_reionization_redshift(
        cosmology.tau_reion,
        f_He,
        _xe_at(z_vals, xe_recomb, _reion_start(Z_RE_BRACKET[1])),
        akthom,
        densities,
        **grho_kwargs,
    )
    if z_re < 0.0:
        return xe_recomb

    xe_before = _xe_at(z_vals, xe_recomb, _reion_start(z_re))
    return np.maximum(xe_recomb, reionization_xe(z_vals, z_re, f_He, xe_before))


def _xe_at(z_vals, xe_vals, z_query: float) -> float:
    """Interpolate the recombination ``x_e`` at a single redshift."""

    order = np.argsort(z_vals)
    return float(np.interp(z_query, np.asarray(z_vals)[order], np.asarray(xe_vals)[order]))
