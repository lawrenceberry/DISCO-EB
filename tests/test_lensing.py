"""Lensing of the temperature and polarization spectra against CLASS.

Three checks of :mod:`discoeb.lensing`:

* the lensing convolution alone: CLASS's own unlensed TT, TE, EE and lensing
  potential spectra through :func:`lensed_spectra` must reproduce CLASS's
  lensed TT, TE, EE and BB (both use the full-sky correlation-function
  method), to 5e-4 for ``l <= 2000`` and 1e-3 up to 2508 for TT and EE, 3e-4
  of ``sqrt(TT EE)`` for TE and 1 percent for the lensing BB up to 1500 (the
  sums run to ``l = 3400``); this runs on any device;
* the lensing potential from the perturbation solve: ``C_l^{psi psi}`` of
  :func:`compute_cl_power_spectrum_batch` (exact projection below
  ``LIMBER_LMIN``, Limber above) against CLASS, to 1 percent up to ``l = 1000``
  and 2 percent at 2000, for the same cosmologies as ``test_power_spectra_match_class``;
* end to end: the unlensed spectra and ``C_l^{psi psi}`` of one solve, splined
  onto every multipole (:func:`cl_on_every_ell`) and lensed, against CLASS's
  lensed spectra for ``l <= 1500``: TT to 5e-3 (the TT gate), EE to 2 percent
  and TE to 2 percent of ``sqrt(TT EE)`` (the polarization gates of
  ``tests/test_cmb.py``); the lensing corrections themselves, lensed over
  unlensed, to 1e-3 for TT and 5e-3 for EE.  Above ``l ~ 1500`` the unlensed
  damping tail is 1 to 3 percent off CLASS and the lensed spectra inherit that.
"""

import numpy as np
import pytest

from test_perturbations import _perturbed_cosmologies

LMAX_CLASS = 3500
LMAX = 3400

UNLENSED = ("tt", "te", "ee")
LENSED = ("tt", "te", "ee", "bb")


def _cuda_available() -> bool:
    try:
        from numba_cuda_mlir import cuda

        return bool(cuda.is_available())
    except Exception:
        return False


def _class_spectra(param, lmax=LMAX_CLASS):
    """CLASS unlensed ``(tt, te, ee, pp)`` and lensed ``(tt, te, ee, bb)`` spectra, ``l = 0..lmax``.

    Two dicts of ``C_l`` in muK^2 (``pp`` dimensionless).
    """

    Class = pytest.importorskip("classy").Class
    h = param["H0"] / 100.0
    cosmo = Class()
    cosmo.set(
        {
            "output": "tCl,pCl,lCl",
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
    unlensed = {name: raw[name] * norm for name in UNLENSED}
    unlensed["pp"] = raw["pp"]
    return unlensed, {name: lensed[name] * norm for name in LENSED}


def _te_error(te, te_ref, tt_ref, ee_ref):
    """The TE error against ``sqrt(TT EE)``, the scale TE's zero crossings do not spoil."""

    return np.abs(te - te_ref) / np.sqrt(tt_ref * ee_ref)


def test_lensed_spectra_match_class_convolution():
    """CLASS's unlensed spectra and C_l^pp through lensed_spectra reproduce CLASS's lensed spectra."""

    import jax
    import jax.numpy as jnp

    from discoeb.lensing import lensed_spectra, lensing_tables

    jax.config.update("jax_enable_x64", True)
    unlensed, lensed = _class_spectra(_perturbed_cosmologies(1)[0])
    tables = lensing_tables(LMAX, dtype=jnp.float64)
    cut = slice(None, LMAX + 1)
    mine = lensed_spectra(
        unlensed["tt"][None, cut], unlensed["te"][None, cut], unlensed["ee"][None, cut], unlensed["pp"][None, cut], tables
    )
    mine = {name: np.asarray(getattr(mine, name))[0] for name in LENSED}
    check = slice(30, 2509)
    ell = np.arange(30, 2509)
    for name in ("tt", "ee"):
        rel = mine[name][check] / lensed[name][check] - 1.0
        assert np.abs(rel[ell <= 2000]).max() < 5.0e-4, name
        assert np.abs(rel).max() < 1.0e-3, name
    te = _te_error(mine["te"][check], lensed["te"][check], lensed["tt"][check], lensed["ee"][check])
    assert te.max() < 3.0e-4
    bb = mine["bb"][check] / lensed["bb"][check] - 1.0
    assert np.abs(bb[ell <= 1500]).max() < 1.0e-2
    # lensing is a several percent effect at these multipoles, so the checks are not vacuous
    assert np.abs(unlensed["tt"][1500:2509] / lensed["tt"][1500:2509] - 1.0).max() > 0.02
    assert np.abs(unlensed["ee"][1000:2509] / lensed["ee"][1000:2509] - 1.0).max() > 0.05


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
        unlensed, _ = _class_spectra(cosmology, lmax=2600)
        rel = cl_pp[i] / unlensed["pp"][ells_pp] - 1.0
        assert np.abs(rel[ells_pp <= 1000]).max() < 1.0e-2, f"cosmology {i}: {rel}"
        assert np.abs(rel).max() < 2.0e-2, f"cosmology {i}: {rel}"


E2E_LMAX = 3000
E2E_LCHECK = 1500


@pytest.mark.skipif(not _cuda_available(), reason="numba-CUDA requires a CUDA GPU")
def test_lensed_spectra_end_to_end_match_class():
    """Unlensed TT, TE, EE and C_l^{psi psi} from one solve, lensed, against CLASS's lensed spectra."""

    import jax

    from discoeb.cmb import cmb_grids, cmb_k_grid, compute_cl_power_spectrum_batch
    from discoeb.lensing import cl_on_every_ell, lensed_spectra, lensing_tables
    from test_cmb import DL_GATE, POL_GATE

    jax.config.update("jax_enable_x64", True)
    cosmology = _perturbed_cosmologies(1)[0]
    # every multipole where D_l curves sharply in ln l, then the acoustic peaks
    # at CLASS's own spacing (a 25-step table at E2E_LMAX outgrows the device)
    ells = np.unique(np.concatenate([np.arange(2, 12), np.arange(12, 40, 2), np.arange(40, 200, 10), np.arange(200, E2E_LMAX + 1, 40)]))
    ells_pp = np.unique(np.round(np.geomspace(2, E2E_LMAX, 40)).astype(np.int64))
    k_values = cmb_k_grid(cosmology, n=128, mode="ode", k_min=1.0e-5, k_max=0.5)
    grids = cmb_grids(cosmology, k_values=k_values, ells=ells, n_save=1000, n_k_fine=1500)
    grids = grids._replace(k_values=np.sort(np.concatenate([k_values, np.geomspace(0.55, 1.2, 16)])))
    _, cl, _, cl_pp = compute_cl_power_spectrum_batch([cosmology], grids=grids, lensing_potential_ells=ells_pp)

    norm = (cosmology["Tcmb"] * 1.0e6) ** 2
    unlensed = {name: cl_on_every_ell(ells, getattr(cl, name) * norm, E2E_LMAX) for name in UNLENSED}
    lensed = lensed_spectra(
        unlensed["tt"], unlensed["te"], unlensed["ee"], cl_on_every_ell(ells_pp, cl_pp, E2E_LMAX, power=2), lensing_tables(E2E_LMAX)
    )
    lensed = {name: np.asarray(getattr(lensed, name))[0] for name in LENSED}
    unlensed = {name: np.asarray(unlensed[name])[0] for name in UNLENSED}

    unlensed_class, lensed_class = _class_spectra(cosmology)
    check = slice(2, E2E_LCHECK + 1)
    ell = np.arange(2, E2E_LCHECK + 1)
    for name, gate in (("tt", DL_GATE), ("ee", POL_GATE)):
        rel = lensed[name][check] / lensed_class[name][check] - 1.0
        assert np.abs(rel).max() < gate, f"lensed {name} off by {np.abs(rel).max():.3g} at l={ell[np.argmax(np.abs(rel))]}"
    te = _te_error(lensed["te"][check], lensed_class["te"][check], lensed_class["tt"][check], lensed_class["ee"][check])
    assert te.max() < POL_GATE, f"lensed te off by {te.max():.3g} of sqrt(TT EE) at l={ell[np.argmax(te)]}"
    for name, gate in (("tt", 1.0e-3), ("ee", 5.0e-3)):
        correction = (lensed[name][check] / unlensed[name][check]) / (lensed_class[name][check] / unlensed_class[name][check]) - 1.0
        assert np.abs(correction).max() < gate, f"{name} lensing correction off by {np.abs(correction).max():.3g} at l={ell[np.argmax(np.abs(correction))]}"
