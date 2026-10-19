"""Lensing of the temperature spectrum against CLASS.

Two checks of :mod:`discoeb.lensing`:

* the lensing convolution alone: CLASS's own unlensed TT and lensing potential
  spectra through :func:`lensed_tt` must reproduce CLASS's lensed TT (both use
  the full-sky correlation-function method), to 4e-4 for ``l <= 2000`` and
  1e-3 up to 2508 (the sums run to ``l = 3400``); this runs on any device;
* the lensing potential from the perturbation solve: ``C_l^{psi psi}`` of
  :func:`compute_cl_power_spectrum_batch` (exact projection below
  ``LIMBER_LMIN``, Limber above) against CLASS, to 1 percent up to ``l = 1000``
  and 2 percent at 2000, for the same cosmologies as ``test_tt_power_spectrum_matches_class``;
* end to end: the unlensed TT and ``C_l^{psi psi}`` of one solve, splined onto
  every multipole (:func:`cl_on_every_ell`) and lensed, against CLASS's lensed
  TT, to 5e-3 for ``l <= 1500`` (the TT gate); the lensing correction itself,
  lensed over unlensed, to 1e-3 there.  Above ``l ~ 1500`` the unlensed damping
  tail is 1 to 3 percent off CLASS and the lensed spectrum inherits that.
"""

import numpy as np
import pytest

from test_perturbations import _perturbed_cosmologies

LMAX_CLASS = 3500
LMAX = 3400


def _cuda_available() -> bool:
    try:
        from numba_cuda_mlir import cuda

        return bool(cuda.is_available())
    except Exception:
        return False


def _class_spectra(param, lmax=LMAX_CLASS):
    """CLASS unlensed TT, lensing potential and lensed TT (``C_l`` in muK^2 / dimensionless), ``l = 0..lmax``."""

    Class = pytest.importorskip("classy").Class
    h = param["H0"] / 100.0
    cosmo = Class()
    cosmo.set(
        {
            "output": "tCl,lCl,pCl",
            "lensing": "yes",
            "l_max_scalars": lmax,
            "h": h,
            "omega_b": param["Omegab"] * h**2,
            "omega_cdm": (param["Omegam"] - param["Omegab"]) * h**2,
            "A_s": param["A_s"],
            "n_s": param["n_s"],
            "k_pivot": param["k_p"],
            "T_cmb": param["Tcmb"],
            "YHe": param["YHe"],
            "N_ur": param["Neff"],
            "N_ncdm": 0,
            "Omega_k": 0.0,
            "recombination": "recfast",
            "reio_parametrization": "reio_none",
            # CLASS's default time sampling of the sources (0.1) puts its own
            # C_2^{pp} 1.2 percent low; 0.01 is converged to 1e-4
            "perturbations_sampling_stepsize": 0.01,
        }
    )
    try:
        cosmo.compute()
        raw = cosmo.raw_cl(lmax)
        lensed = cosmo.lensed_cl(lmax)
    finally:
        cosmo.struct_cleanup()
        cosmo.empty()
    norm = (param["Tcmb"] * 1.0e6) ** 2
    return raw["tt"] * norm, raw["pp"], lensed["tt"] * norm


def test_lensed_tt_matches_class_convolution():
    """CLASS's unlensed TT and C_l^pp through lensed_tt reproduce CLASS's lensed TT."""

    import jax
    import jax.numpy as jnp

    from discoeb.lensing import lensed_tt, lensing_tables

    jax.config.update("jax_enable_x64", True)
    tt_u, pp, tt_l = _class_spectra(_perturbed_cosmologies(1)[0])
    tables = lensing_tables(LMAX, dtype=jnp.float64)
    mine = np.asarray(lensed_tt(tt_u[None, : LMAX + 1], pp[None, : LMAX + 1], tables))[0]
    rel = mine[30:2509] / tt_l[30:2509] - 1.0
    ell = np.arange(30, 2509)
    assert np.abs(rel[ell <= 2000]).max() < 4.0e-4
    assert np.abs(rel).max() < 1.0e-3
    # lensing is a few percent effect at these multipoles, so the check is not vacuous
    assert np.abs(tt_u[1500:2509] / tt_l[1500:2509] - 1.0).max() > 0.02


@pytest.mark.skipif(not _cuda_available(), reason="numba-CUDA requires a CUDA GPU")
def test_lensing_potential_matches_class():
    """C_l^{psi psi} from the batched solve (exact projection and Limber) against CLASS."""

    import jax

    from discoeb.cmb import cmb_grids, cmb_k_grid, compute_cl_power_spectrum_batch

    jax.config.update("jax_enable_x64", True)
    cosmologies = _perturbed_cosmologies(4)
    ells_pp = np.unique(np.round(np.geomspace(2, 2500, 40)).astype(np.int64))
    k_values = cmb_k_grid(cosmologies[0], n=128, mode="ode", k_min=1.0e-5, k_max=0.5)
    grids = cmb_grids(cosmologies[0], k_values=k_values, ells=np.asarray([30, 300, 1000]), n_save=1200, n_k_fine=1500)
    # the Limber projection at l ~ 1000-2500 needs the Weyl potential up to k ~ 1 Mpc^-1
    grids = grids._replace(k_values=np.sort(np.concatenate([k_values, np.geomspace(0.55, 1.2, 16)])))
    _, _, _, cl_pp = compute_cl_power_spectrum_batch(cosmologies, grids=grids, lensing_potential_ells=ells_pp)
    cl_pp = np.asarray(jax.block_until_ready(cl_pp))
    for i, cosmology in enumerate(cosmologies):
        _, pp_class, _ = _class_spectra(cosmology, lmax=2600)
        rel = cl_pp[i] / pp_class[ells_pp] - 1.0
        assert np.abs(rel[ells_pp <= 1000]).max() < 1.0e-2, f"cosmology {i}: {rel}"
        assert np.abs(rel).max() < 2.0e-2, f"cosmology {i}: {rel}"


E2E_LMAX = 3000
E2E_LCHECK = 1500


@pytest.mark.skipif(not _cuda_available(), reason="numba-CUDA requires a CUDA GPU")
def test_lensed_tt_end_to_end_matches_class():
    """Unlensed TT and C_l^{psi psi} from one solve, lensed, against CLASS's lensed TT."""

    import jax

    from discoeb.cmb import cmb_grids, cmb_k_grid, compute_cl_power_spectrum_batch
    from discoeb.lensing import cl_on_every_ell, lensed_tt, lensing_tables

    jax.config.update("jax_enable_x64", True)
    cosmology = _perturbed_cosmologies(1)[0]
    # every multipole where D_l curves sharply in ln l, then the acoustic peaks
    # at CLASS's own spacing (a 25-step table at E2E_LMAX outgrows the device)
    ells = np.unique(np.concatenate([np.arange(2, 12), np.arange(12, 40, 2), np.arange(40, 200, 10), np.arange(200, E2E_LMAX + 1, 40)]))
    ells_pp = np.unique(np.round(np.geomspace(2, E2E_LMAX, 40)).astype(np.int64))
    k_values = cmb_k_grid(cosmology, n=128, mode="ode", k_min=1.0e-5, k_max=0.5)
    grids = cmb_grids(cosmology, k_values=k_values, ells=ells, n_save=1000, n_k_fine=1500)
    grids = grids._replace(k_values=np.sort(np.concatenate([k_values, np.geomspace(0.55, 1.2, 16)])))
    _, cl_tt, _, cl_pp = compute_cl_power_spectrum_batch([cosmology], grids=grids, lensing_potential_ells=ells_pp)

    norm = (cosmology["Tcmb"] * 1.0e6) ** 2
    tt_u = cl_on_every_ell(ells, cl_tt * norm, E2E_LMAX)
    tt_l = np.asarray(lensed_tt(tt_u, cl_on_every_ell(ells_pp, cl_pp, E2E_LMAX, power=2), lensing_tables(E2E_LMAX)))[0]
    tt_u = np.asarray(tt_u)[0]

    tt_u_class, _, tt_l_class = _class_spectra(cosmology)
    check = slice(2, E2E_LCHECK + 1)
    rel = tt_l[check] / tt_l_class[check] - 1.0
    assert np.abs(rel).max() < 5.0e-3, f"lensed TT off by {np.abs(rel).max():.3g} at l={2 + np.argmax(np.abs(rel))}"
    correction = (tt_l[check] / tt_u[check]) / (tt_l_class[check] / tt_u_class[check]) - 1.0
    assert np.abs(correction).max() < 1.0e-3, f"lensing correction off by {np.abs(correction).max():.3g}"
