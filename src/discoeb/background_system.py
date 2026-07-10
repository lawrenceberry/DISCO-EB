"""Fused background + RECFAST system solved with the rodas5Pjax integrator.

This module assembles the recombination thermodynamics (:mod:`discoeb.recfast`)
and the background conformal-time evolution into a single stiff ODE system and
integrates it on a requested redshift grid with the pure-JAX Rodas5P solver
(:func:`discoeb.integrators.rodas5Pjax_solve`).

State vector (as a function of redshift ``z``)::

    y = (x_H, x_He, T_mat, tau)

where ``x_H`` / ``x_He`` are the RECFAST hydrogen / helium ionization fractions,
``T_mat`` is the matter temperature in K, and ``tau`` is the conformal time in
Mpc measured from the starting redshift. The right-hand side is the
tau-augmented RECFAST RHS :func:`discoeb.recfast.recfast_rhs_with_tau`.

Because RECFAST is written with redshift decreasing, the system is integrated in
the monotonically increasing variable ``u = z_start - z`` (so the solver sees a
strictly increasing time axis), with the RHS negated accordingly.
"""

import jax
import jax.numpy as jnp

from .background import (
    _dark_energy_density_ratio_jax,
    critical_density_grho,
    grhob,
    grhoc,
    grhog,
    grhornomass,
    helium_number_fraction,
    neutrino_density_grho,
    neutrino_mass_parameter,
)
from .integrators import rodas5Pjax_solve
from .recfast import (
    equality_redshift,
    hubble_constant_si,
    initial_thermal_state,
    massive_neutrino_density_ratio,
    matter_density_fraction,
    present_hydrogen_number_density,
    recfast_rhs_with_tau,
    hubble_z_si,
)


# Packed-parameter layout consumed by the fused ODE. Index 0 carries the
# integration anchor ``z_start``; indices 1..16 are the RECFAST parameter row
# expected by :func:`discoeb.recfast.recfast_rhs_with_tau`.
#   p = (z_start, T_cmb, f_He, Nnow, H0_SI, omega_m, z_eq,
#        grhog, grhornomass, grhoc, grhob, grhov,
#        grhok, grhomnu, amnu, w_DE_0, w_DE_a)


def recfast_parameters(cosmology) -> dict:
    """Return the named scalar RECFAST inputs for a cosmology.

    Covers the full DISCO-EB background: spatial curvature (``grhok``), a
    degenerate massive-neutrino species (``grhomnu``, ``amnu``) and dynamical
    dark energy (``w_DE_0``, ``w_DE_a``), in addition to flat massless
    ``LambdaCDM``. Curvature and dark energy are dynamically negligible at
    recombination, but a massive neutrino is still relativistic there and shifts
    ``H(z)`` by several percent, so it must be carried as radiation.

    The dark-energy coefficient ``grhov`` closes the density budget at ``a = 1``
    against the critical density, reproducing the flat-massless
    :func:`discoeb.background.grhov` while accounting for curvature and the
    massive-neutrino density.
    """

    grhog_value = grhog(cosmology.T_cmb)
    grhornomass_value = grhornomass(grhog_value, cosmology.Neff_massless)
    grhoc_value = grhoc(cosmology.omega_c_h2)
    grhob_value = grhob(cosmology.omega_b_h2)

    grhom = critical_density_grho(cosmology.H0)
    grhok_value = grhom * cosmology.Omegak
    n_mnu = cosmology.num_massive_neutrinos
    if n_mnu > 0.0:
        grhomnu_value = neutrino_density_grho(cosmology.T_cmb) * n_mnu
        amnu_value = neutrino_mass_parameter(cosmology.mnu, cosmology.T_cmb)
        grho_mnu_today = grhomnu_value * float(
            massive_neutrino_density_ratio(1.0, amnu_value)
        )
    else:
        grhomnu_value = 0.0
        amnu_value = 0.0
        grho_mnu_today = 0.0

    grhov_value = grhom - (
        grhog_value
        + grhornomass_value
        + grho_mnu_today
        + grhoc_value
        + grhob_value
        + grhok_value
    )
    return {
        "T_cmb": cosmology.T_cmb,
        "f_He": helium_number_fraction(cosmology.Y_He),
        "Nnow": present_hydrogen_number_density(cosmology.omega_b_h2, cosmology.Y_He),
        "H0_SI": hubble_constant_si(cosmology.h),
        "omega_m": matter_density_fraction(
            cosmology.omega_b_h2, cosmology.omega_c_h2, cosmology.h
        ),
        "z_eq": equality_redshift(
            grhog_value, grhornomass_value, grhoc_value, grhob_value, grhomnu_value
        ),
        "grhog": grhog_value,
        "grhornomass": grhornomass_value,
        "grhoc": grhoc_value,
        "grhob": grhob_value,
        "grhov": grhov_value,
        "grhok": grhok_value,
        "grhomnu": grhomnu_value,
        "amnu": amnu_value,
        "w_DE_0": cosmology.w_DE_0,
        "w_DE_a": cosmology.w_DE_a,
    }


def _pack_params(z_start: float, args: dict):
    """Pack ``z_start`` + the RECFAST parameter row into one array."""

    return jnp.asarray(
        [
            z_start,
            args["T_cmb"],
            args["f_He"],
            args["Nnow"],
            args["H0_SI"],
            args["omega_m"],
            args["z_eq"],
            args["grhog"],
            args["grhornomass"],
            args["grhoc"],
            args["grhob"],
            args["grhov"],
            args["grhok"],
            args["grhomnu"],
            args["amnu"],
            args["w_DE_0"],
            args["w_DE_a"],
        ],
        dtype=jnp.float64,
    )


def _fused_ode(y, u, params):
    """Fused background+RECFAST RHS in the increasing variable ``u = z_start - z``.

    Returns ``dy/du = -dy/dz`` so the solver integrates from ``u = 0`` (at
    ``z = z_start``) toward larger ``u`` (lower redshift).
    """

    z_start = params[0]
    z = z_start - u
    return -recfast_rhs_with_tau(z, y, params[1:])


def solve_background_system(
    cosmology,
    z_grid,
    *,
    z_start: float = 3500.0,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    max_steps: int = 100000,
):
    """Integrate the fused background+RECFAST system on ``z_grid``.

    Parameters
    ----------
    cosmology : discoeb.cosmology.Cosmology
        Flat cosmology providing ``T_cmb``, ``omega_b_h2``, ``omega_c_h2``,
        ``h``, ``Y_He`` and ``Neff_massless``.
    z_grid : array-like
        Output redshifts, **descending** and starting at ``z_start`` (e.g.
        ``jnp.linspace(z_start, 50.0, n)``). The solver integrates from
        ``z_start`` down to ``z_grid[-1]``.
    z_start : float
        Starting redshift (fully ionized). RECFAST initial conditions are taken
        here via :func:`discoeb.recfast.initial_thermal_state`.
    rtol, atol : float
        Rodas5P step-size tolerances.
    max_steps : int
        Maximum adaptive steps.

    Returns
    -------
    dict
        ``{"z", "x_H", "x_He", "x_e", "T_mat", "tau", "Hz"}`` where ``x_e`` is the
        free-electron fraction ``x_H + f_He x_He``, ``tau`` is the conformal time
        in Mpc measured from ``z_start``, and ``Hz`` is ``H(z)`` in s^-1.
    """

    z_grid = jnp.asarray(z_grid, dtype=jnp.float64)
    args = recfast_parameters(cosmology)
    params = _pack_params(z_start, args)

    x_H0, x_He0, T_mat0 = initial_thermal_state(
        z_start, args["T_cmb"], args["Nnow"], args["f_He"]
    )
    y0 = jnp.asarray([x_H0, x_He0, T_mat0, 0.0], dtype=jnp.float64)

    # Increasing integration axis u = z_start - z (0 at z_start).
    t_span = z_start - z_grid

    sol = rodas5Pjax_solve(
        _fused_ode,
        y0,
        t_span,
        params,
        rtol=rtol,
        atol=atol,
        max_steps=max_steps,
    )
    ys = sol[0]  # single trajectory: (n_save, 4)

    x_H = ys[:, 0]
    x_He = ys[:, 1]
    T_mat = ys[:, 2]
    tau = ys[:, 3]
    x_e = x_H + args["f_He"] * x_He
    Hz = jax.vmap(
        lambda z: hubble_z_si(
            z,
            args["grhog"],
            args["grhornomass"],
            args["grhoc"],
            args["grhob"],
            args["grhov"],
            grhok=args["grhok"],
            grhomnu=args["grhomnu"],
            rhonu=massive_neutrino_density_ratio(1.0 / (1.0 + z), args["amnu"]),
            rho_de=_dark_energy_density_ratio_jax(
                1.0 / (1.0 + z), args["w_DE_0"], args["w_DE_a"]
            ),
        )
    )(z_grid)

    return {
        "z": z_grid,
        "x_H": x_H,
        "x_He": x_He,
        "x_e": x_e,
        "T_mat": T_mat,
        "tau": tau,
        "Hz": Hz,
    }
