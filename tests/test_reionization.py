"""Reionization in the thermal history.

The thermodynamics tables take an optical depth ``tau_reio`` and add CAMB's
tanh reionization step to the RECFAST ionization fraction, with the
reionization redshift set by bisection on the optical depth.  Checked here:
the integrated Thomson opacity of the low-redshift step reproduces the
requested optical depth, the reionization redshift agrees with CLASS's for the
same ``tau_reio`` (both use the same parametrization), ``tau_reio = 0``
leaves the history untouched, and the unlensed TT spectrum of a batch of
cosmologies with different optical depths matches CLASS's ``reio_camb`` to the
TT gate of ``tests/test_cmb.py``, from the reionization bump at low ``l`` to
the ``exp(-2 tau_reio)`` suppression of the acoustic peaks.
"""

import numpy as np
import pytest

from test_perturbations import REFERENCE_COSMOLOGY, _perturbed_cosmologies

TAU_REIO = 0.0544

# Multipoles of the TT check: the reionization bump (l < 20) and the peaks.
TT_ELLS = np.asarray([2, 3, 4, 6, 8, 10, 15, 20, 30, 40, 80, 120, 180, 240, 320, 500, 800, 1200], dtype=np.int64)
DL_GATE = 5.0e-3


def _cuda_available() -> bool:
    try:
        from numba_cuda_mlir import cuda

        return bool(cuda.is_available())
    except Exception:
        return False


def _class(param, extra):
    """A computed CLASS instance for ``param`` with CAMB-style reionization at its ``tau_reio``."""

    Class = pytest.importorskip("classy").Class
    h = param["H0"] / 100.0
    c = Class()
    c.set(
        {
            "h": h,
            "omega_b": param["Omegab"] * h**2,
            "omega_cdm": (param["Omegam"] - param["Omegab"]) * h**2,
            "A_s": param["A_s"],
            "n_s": param["n_s"],
            "k_pivot": param["k_p"],
            "YHe": param["YHe"],
            "T_cmb": param["Tcmb"],
            "N_ur": param["Neff"],
            "N_ncdm": 0,
            "Omega_k": 0.0,
            "recombination": "recfast",
            "reio_parametrization": "reio_camb",
            "tau_reio": param["tau_reio"],
            **extra,
        }
    )
    c.compute()
    return c


def _tables(tau_reio):
    from discoeb.cmb import visibility_on_grid
    from discoeb.perturbations import _as_cosmology, build_thermo_tables

    cosmology = _as_cosmology({**REFERENCE_COSMOLOGY, "tau_reio": tau_reio})
    tau_grid, values, _, tau0 = build_thermo_tables(cosmology)
    _, g, emk = visibility_on_grid(tau_grid, values[1], tau_grid, float(tau0))
    return tau_grid, values, float(tau0), g, emk


def test_reionization_optical_depth():
    """exp(-kappa) today is exp(-tau_reio), and the bump sits where CLASS puts z_reio."""

    tau_grid, values, tau0, g, emk = _tables(TAU_REIO)
    late = tau_grid > 0.3 * tau0  # z < ~30: the reionization step; the recombination tail adds ~1e-4
    tau_late = np.trapezoid(values[1][late], tau_grid[late])
    assert abs(tau_late - TAU_REIO) < 1.0e-3, tau_late

    # the visibility has a secondary bump after recombination, at CLASS's z_reio
    c = _class({**REFERENCE_COSMOLOGY, "tau_reio": TAU_REIO}, {})
    try:
        z_reio_class = c.get_current_derived_parameters(["z_reio"])["z_reio"]
    finally:
        c.struct_cleanup()
        c.empty()
    a_bump = values[0][late][np.argmax(g[late])]
    z_bump = 1.0 / a_bump - 1.0
    # the visibility peaks a little below z_reio (the step is half way there); 1.5 in z
    assert abs(z_bump - z_reio_class) < 1.5, (z_bump, z_reio_class)


def test_no_reionization_by_default():
    """tau_reio = 0 (the default) leaves exp(-kappa) at unity today."""

    tau_grid, values, tau0, _, _ = _tables(0.0)
    late = tau_grid > 0.3 * tau0
    assert np.trapezoid(values[1][late], tau_grid[late]) < 5.0e-4


@pytest.mark.skipif(not _cuda_available(), reason="numba-CUDA requires a CUDA GPU")
def test_tt_power_spectrum_with_reionization_matches_class():
    """Unlensed TT D_l of a batch with tau_reio from 0.04 to 0.07 against CLASS reio_camb."""

    import jax

    from discoeb.cmb import cmb_grids, cmb_k_grid, compute_cl_power_spectrum_batch

    jax.config.update("jax_enable_x64", True)
    cosmologies = [
        {**c, "tau_reio": t} for c, t in zip(_perturbed_cosmologies(4), np.linspace(0.04, 0.07, 4))
    ]
    k_values = cmb_k_grid(cosmologies[0], n=128, mode="ode", k_min=1.0e-5, k_max=0.5)
    grids = cmb_grids(cosmologies[0], k_values=k_values, ells=TT_ELLS, n_save=1000, n_k_fine=1500)
    _, _, dl = compute_cl_power_spectrum_batch(cosmologies, grids=grids)
    dl = np.asarray(jax.block_until_ready(dl))

    for i, cosmology in enumerate(cosmologies):
        c = _class(cosmology, {"output": "tCl", "modes": "s", "lensing": "no", "l_max_scalars": 2500})
        try:
            raw = c.raw_cl(int(TT_ELLS.max()) + 1)
        finally:
            c.struct_cleanup()
            c.empty()
        ell = raw["ell"]
        dl_class = (ell * (ell + 1.0) * raw["tt"] * (cosmology["Tcmb"] * 1.0e6) ** 2 / (2.0 * np.pi))[TT_ELLS]
        rel = np.abs(dl[i] / dl_class - 1.0)
        assert rel.max() < DL_GATE, f"cosmology {i} (tau_reio {cosmology['tau_reio']:.3f}): D_l off by {rel.max():.3g} at l={TT_ELLS[np.argmax(rel)]}"
