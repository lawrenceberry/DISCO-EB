"""CMB source functions, transfer functions, and TT power spectrum vs CLASS.

The scalar helpers (visibility, Bessel projection, ``C_l`` quadrature) are
checked analytically; the end-to-end unlensed TT spectrum is compared against
**CLASS** via the ``classy`` Python package.

The end-to-end test needs a CUDA GPU (the perturbation solve is numba-CUDA) and
skips otherwise.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from cosmologies import BENCHMARK_COSMOLOGIES, cosmology_to_class_params

from discoeb.cmb import (
    cl_power_spectrum,
    cmb_k_grid,
    cmb_tau_grid,
    dl_power_spectrum,
    theta_ell_transfer_function,
    visibility_functions,
)


DEFAULT_COSMOLOGY = BENCHMARK_COSMOLOGIES["planck_2018_flat_lcdm"]

# Multipoles spanning the Sachs-Wolfe plateau, the first acoustic peaks, and the
# onset of Silk damping.
TT_ELLS = np.asarray([2, 10, 20, 40, 80, 120, 180, 240, 320, 500], dtype=np.int64)

# Wave modes the perturbation hierarchy is solved on before the sources are
# interpolated onto the finer line-of-sight grid.
CMB_N_K = 128

# Relative-error gate on D_ell vs CLASS (matches the DISCO2 CAMB gate).
DL_GATE = 1.0e-2

# CLASS truncates its own line-of-sight k-integration at k_max ~ 2.4 l_max / tau0,
# so l_max_scalars must sit well above the multipoles under test or CLASS itself
# is the inaccurate side of the comparison: at l_max = 550 its D_ell is 8-9% low
# at l = 400-500. This value is converged to <1e-4 for l <= 500.
CLASS_LMAX = 2500


def _cuda_available() -> bool:
    """Return whether a CUDA device is visible to numba."""

    try:
        from numba import cuda

        return cuda.is_available()
    except Exception:
        return False


def _class_unlensed_tt(cosmology, ells):
    """Return CLASS unlensed scalar TT ``D_ell`` in ``muK^2``."""

    Class = pytest.importorskip("classy").Class

    params = cosmology_to_class_params(cosmology, output="tCl")
    params.update({"l_max_scalars": CLASS_LMAX, "modes": "s"})

    cosmo = Class()
    cosmo.set(params)
    try:
        cosmo.compute()
        raw = cosmo.raw_cl(int(np.max(ells)) + 1)
        ell = raw["ell"]
        # CLASS returns dimensionless C_ell; convert to D_ell in muK^2.
        norm = (cosmology.T_cmb * 1.0e6) ** 2
        dl = ell * (ell + 1.0) * raw["tt"] * norm / (2.0 * np.pi)
        return dl[np.asarray(ells, dtype=np.int64)]
    finally:
        cosmo.struct_cleanup()
        cosmo.empty()


# =============================================================================
# Scalar helpers
# =============================================================================


def test_visibility_function_normalizes_and_peaks():
    """g = kappa' exp(-kappa) integrates to ~1 and the optical depth runs backwards."""

    tau = jnp.linspace(1.0, 400.0, 4096)
    opacity = 1.0e3 * jnp.exp(-((tau / 60.0) ** 2))

    vis = visibility_functions(tau, opacity)

    assert set(vis) == {"opacity", "optical_depth", "g", "gprime", "gpprime"}
    assert jnp.all(jnp.isfinite(vis["g"]))
    # Optical depth is accumulated back from today, so it decreases with tau.
    assert float(vis["optical_depth"][0]) > float(vis["optical_depth"][-1])
    assert float(vis["optical_depth"][-1]) == pytest.approx(0.0, abs=1e-12)
    # g is a probability density in tau: int g dtau = 1 - exp(-kappa(tau_min)) -> 1.
    assert float(jnp.trapezoid(vis["g"], tau)) == pytest.approx(1.0, rel=1e-4)
    # It peaks where the optical depth passes through unity.
    peak = int(jnp.argmax(vis["g"]))
    assert float(vis["optical_depth"][peak]) == pytest.approx(1.0, abs=0.15)


def test_theta_ell_transfer_function_reproduces_bessel_projection():
    """A delta-like source at tau_rec projects to Theta_l = j_l(k chi_rec)."""

    from scipy import special

    tau0, tau_rec, width = 300.0, 100.0, 0.05
    tau = jnp.linspace(tau_rec - 20.0 * width, tau_rec + 20.0 * width, 4001)
    # Unit-normalized narrow Gaussian, so int S j_l dtau -> j_l(k chi_rec).
    source = jnp.exp(-0.5 * ((tau - tau_rec) / width) ** 2) / (width * np.sqrt(2.0 * np.pi))
    k = jnp.asarray([0.01, 0.05, 0.1])
    ells = np.asarray([2, 8, 20])

    theta = theta_ell_transfer_function(
        ells, k, tau, source[None, :] * jnp.ones((3, 1)), tau0
    )

    chi_rec = tau0 - tau_rec
    expected = np.stack(
        [special.spherical_jn(int(ell), np.asarray(k) * chi_rec) for ell in ells], axis=-1
    )
    np.testing.assert_allclose(np.asarray(theta), expected, rtol=5e-3, atol=1e-6)


def test_cl_power_spectrum_scales_with_amplitude_and_transfer():
    """C_l is linear in A_s and quadratic in Theta_l."""

    k = jnp.geomspace(1.0e-4, 1.0, 64)
    ells = jnp.asarray([2, 10, 50])
    theta = jnp.ones((k.shape[0], ells.shape[0])) * 1.0e-4
    n_s, k_p = DEFAULT_COSMOLOGY.n_s, DEFAULT_COSMOLOGY.k_pivot

    cl = cl_power_spectrum(theta, k, n_s, k_p, 2.0e-9)
    cl_double_as = cl_power_spectrum(theta, k, n_s, k_p, 4.0e-9)
    cl_double_theta = cl_power_spectrum(2.0 * theta, k, n_s, k_p, 2.0e-9)

    assert cl.shape == ells.shape
    np.testing.assert_allclose(np.asarray(cl_double_as), 2.0 * np.asarray(cl), rtol=1e-12)
    np.testing.assert_allclose(np.asarray(cl_double_theta), 4.0 * np.asarray(cl), rtol=1e-12)
    assert jnp.all(dl_power_spectrum(cl, ells, DEFAULT_COSMOLOGY.T_cmb) > 0.0)


# =============================================================================
# Adaptive grids
# =============================================================================


def test_cmb_k_grid_is_sorted_and_spans_the_requested_range():
    grid = cmb_k_grid(DEFAULT_COSMOLOGY, n=64, k_min=1.0e-5, k_max=0.5)

    assert grid.shape == (64,)
    assert grid[0] == pytest.approx(1.0e-5)
    assert grid[-1] == pytest.approx(0.5)
    assert np.all(np.diff(grid) > 0.0)


def test_cmb_tau_grid_concentrates_around_recombination():
    """A large share of the save points must land inside the visibility peak."""

    from discoeb.cmb import _cmb_grid_summary

    k = np.geomspace(1.0e-4, 0.5, 8)
    tau = cmb_tau_grid(DEFAULT_COSMOLOGY, k, 512)
    s = _cmb_grid_summary(DEFAULT_COSMOLOGY)

    assert tau.shape == (512,)
    assert np.all(np.diff(tau) > 0.0)
    assert tau[-1] == pytest.approx(s["tau0"])

    # Recombination occupies a tiny fraction of [0, tau0], but the line-of-sight
    # integral is unresolved unless the grid oversamples it. Compare the local
    # sampling density there against the density a uniform grid would give.
    window = 10.0 * s["delta_tau_rec"]
    near_rec = np.abs(tau - s["tau_star"]) < window
    density_rec = near_rec.sum() / (2.0 * window)
    density_uniform = tau.size / (tau[-1] - tau[0])
    assert density_rec > 2.0 * density_uniform


# =============================================================================
# End-to-end vs CLASS
# =============================================================================


@pytest.mark.skipif(not _cuda_available(), reason="numba-CUDA requires a CUDA GPU")
def test_tt_power_spectrum_matches_class(benchmark):
    """Unlensed scalar TT D_ell vs CLASS for flat LambdaCDM, timed.

    The full spectrum (perturbation solve over ``CMB_N_K`` wave modes, source
    construction, and the line-of-sight Bessel projection) is timed with
    ``benchmark.pedantic``. The warmup round absorbs the one-time costs -- the
    numba-CUDA kernel compilation and the spherical-Bessel table build -- both of
    which are cached, so the single timed round measures the steady-state spectrum
    evaluation. The timed round reuses the same ``k_values`` and ``n_save``, which
    are part of the compiled-kernel cache key, so no recompilation occurs.
    """

    from discoeb.cmb import compute_cl_power_spectrum

    k_values = cmb_k_grid(
        DEFAULT_COSMOLOGY, n=CMB_N_K, mode="ode", k_min=1.0e-5, k_max=0.5
    )
    _, _, dl_ours = benchmark.pedantic(
        compute_cl_power_spectrum,
        args=(DEFAULT_COSMOLOGY,),
        kwargs=dict(k_values=k_values, ells=TT_ELLS, n_save=1000, n_k_fine=1500),
        rounds=1,
        warmup_rounds=1,
        iterations=1,
    )
    dl_class = _class_unlensed_tt(DEFAULT_COSMOLOGY, TT_ELLS)

    rel = np.abs(dl_ours / dl_class - 1.0)
    assert float(np.max(rel)) < DL_GATE
