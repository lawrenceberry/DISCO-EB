"""End-to-end test of the fused background + RECFAST system.

The fused ODE (:mod:`discoeb.background_system`) is integrated with the pure-JAX
Rodas5P solver (:func:`discoeb.integrators.rodas5Pjax_solve`) and its outputs are
compared against **CLASS** (configured to use RECFAST) via the ``classy`` Python
package: the free-electron fraction ``x_e``, the matter temperature ``T_mat``,
the conformal time ``tau``, and the Hubble rate ``H(z)``.
"""

import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from discoeb.background_system import solve_background_system
from discoeb.constants import C_SI, MPC_IN_M

from cosmologies import BENCHMARK_COSMOLOGIES, cosmology_to_class_params


DEFAULT_COSMOLOGY = BENCHMARK_COSMOLOGIES["planck_2018_flat_lcdm"]

COSMOLOGY_CASES = [
    pytest.param(DEFAULT_COSMOLOGY, id="planck_2018_flat_lcdm"),
]

_Z_START = 3500.0
_Z_GRID = np.linspace(_Z_START, 50.0, 160)

# Relative-error gates for (x_e, T_mat, tau, H(z)).
_GATES = {"x_e": 5.0e-3, "T_mat": 5.0e-3, "tau": 1.0e-2, "Hz": 1.0e-3}


def relative_error(actual, expected, floor):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    return np.abs(actual - expected) / np.maximum(np.abs(expected), floor)


def _class_reference(cosmology, z_grid, z_start):
    """Return CLASS (RECFAST) ``x_e``, ``T_b``, conformal-time offset and ``H(z)``.

    ``tau`` is returned as the conformal time elapsed since ``z_start`` (Mpc), to
    match the fused system's ``tau`` initial condition, and ``Hz`` is in s^-1.
    """

    Class = pytest.importorskip("classy").Class

    cosmo = Class()
    class_params = cosmology_to_class_params(cosmology, output="tCl")
    class_params["recombination"] = "recfast"
    cosmo.set(class_params)
    try:
        cosmo.compute()
        thermo = cosmo.get_thermodynamics()
        background = cosmo.get_background()
    finally:
        pass

    z_th = np.asarray(thermo["z"])
    th_order = np.argsort(z_th)

    def interp_thermo(key):
        return np.interp(z_grid, z_th[th_order], np.asarray(thermo[key])[th_order])

    z_bg = np.asarray(background["z"])
    bg_order = np.argsort(z_bg)
    conf_time = np.interp(
        z_grid, z_bg[bg_order], np.asarray(background["conf. time [Mpc]"])[bg_order]
    )
    conf_time_start = np.interp(
        z_start, z_bg[bg_order], np.asarray(background["conf. time [Mpc]"])[bg_order]
    )
    hubble_si = (
        np.interp(z_grid, z_bg[bg_order], np.asarray(background["H [1/Mpc]"])[bg_order])
        * C_SI
        / MPC_IN_M
    )

    cosmo.struct_cleanup()
    cosmo.empty()

    return {
        "x_e": interp_thermo("x_e"),
        "T_mat": interp_thermo("Tb [K]"),
        "tau": conf_time - conf_time_start,
        "Hz": hubble_si,
    }


@pytest.mark.parametrize("cosmology", COSMOLOGY_CASES)
def test_background_system_matches_class(cosmology):
    """Integrate the fused system with rodas5Pjax and compare to CLASS (RECFAST)."""

    out = solve_background_system(
        cosmology, _Z_GRID, z_start=_Z_START, rtol=1.0e-7, atol=1.0e-10
    )
    ref = _class_reference(cosmology, _Z_GRID, _Z_START)

    xe = np.asarray(out["x_e"])
    T_mat = np.asarray(out["T_mat"])
    tau = np.asarray(out["tau"])
    Hz = np.asarray(out["Hz"])

    # Sanity: fully ionized start, low residual today, monotone growing tau.
    assert xe[0] == pytest.approx(ref["x_e"][0], rel=1.0e-2)
    assert 1.0e-4 < xe[-1] < 1.0e-3
    assert np.all(np.diff(tau) > 0.0)

    assert float(np.max(relative_error(xe, ref["x_e"], floor=1.0e-8))) < _GATES["x_e"]
    assert float(np.max(relative_error(T_mat, ref["T_mat"], floor=1.0))) < _GATES["T_mat"]
    # tau[0] == 0 by construction; compare the elapsed conformal time elsewhere.
    assert (
        float(np.max(relative_error(tau[1:], ref["tau"][1:], floor=1.0e-6)))
        < _GATES["tau"]
    )
    assert float(np.max(relative_error(Hz, ref["Hz"], floor=1.0e-30))) < _GATES["Hz"]
