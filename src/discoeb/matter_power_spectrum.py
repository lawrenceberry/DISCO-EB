"""Scalar linear matter power-spectrum mapping.

Ported from the DISCO2 prototype. Maps the evolved perturbation variables onto
the linear matter power spectrum. Like :mod:`discoeb.perturbations`, it provides
only scalar functions with no solver or grid machinery.

The total-matter contrast ``delta_m`` is built from the synchronous-gauge CDM and
baryon density contrasts evolved by :mod:`discoeb.perturbations`; with initial
conditions normalized to unit comoving curvature it acts as a transfer function
for the power spectrum. Wavenumbers ``k`` are in Mpc^-1 and ``P(k)`` in Mpc^3.
"""

import math


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
