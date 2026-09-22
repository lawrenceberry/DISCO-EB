"""The unlensed TT power spectrum against CLASS, and the time of its batched solve.

The end-to-end spectrum -- perturbation solve on modax's Rodas5P kernel over the
CMB wave modes with the sources evaluated in-kernel by the save hook, then the
batched line-of-sight projection and the ``C_l`` quadrature -- is run for the
same cosmologies as ``test_matter_power_spectrum_matches_class`` (one, and a
batch of 128 nearby flat LCDM models) and each is checked against its own
CLASS reference, run with CLASS's own recfast and no reionization, since the
DISCO-EB thermal history has none.

One internal check backs that up: the matrix-product projection must reproduce
the analytic transfer function of a delta-like source, ``Theta_l(k) = j_l(k
chi_rec)``, which pins the Bessel table, its derivative recurrences and the
``tau`` quadrature independently of any cosmology.

The benchmark times one warmed call of :func:`compute_cl_power_spectrum_batch`
at both batch sizes.

The derivative of the spectrum with respect to the five background densities
(:func:`cl_power_spectrum_jax`, the CMB counterpart of
``matter_power_spectrum_jax``) is checked against central differences and timed
in both layouts of the sensitivity system, as ``test_matter_power_spectrum_derivatives``
does for ``P(k)``.
"""

import numpy as np
import pytest

from test_perturbations import BATCH_SIZE_CASES, DENSITY_NAMES, _perturbed_cosmologies

# Multipoles spanning the Sachs-Wolfe plateau, the acoustic peaks and the onset
# of Silk damping.
TT_ELLS = np.asarray([2, 10, 20, 40, 80, 120, 180, 240, 320, 500, 800, 1200], dtype=np.int64)

# Wave modes the perturbation hierarchy is solved on before the sources are
# interpolated onto the finer line-of-sight grid, and the number of saved times.
CMB_N_K = 128
CMB_N_SAVE = 1000

DL_GATE = 5.0e-3

# Gate on dD_l/d(density) as a fraction of the derivative's largest value over
# the multipoles; see ``test_tt_power_spectrum_derivatives``.
CL_DERIV_GATE = 2.0e-3

# CLASS truncates its own line-of-sight k-integration at k_max ~ 2.4 l_max / tau0,
# so l_max_scalars must sit well above the multipoles under test or CLASS itself
# is the inaccurate side of the comparison.
CLASS_LMAX = 2500


def _cuda_available() -> bool:
    try:
        from numba_cuda_mlir import cuda

        return bool(cuda.is_available())
    except Exception:
        return False


def _class_unlensed_tt(param, ells):
    """Return CLASS unlensed scalar TT ``D_ell`` in ``muK^2`` at ``ells``."""

    Class = pytest.importorskip("classy").Class
    h = param["H0"] / 100.0
    cosmo = Class()
    cosmo.set(
        {
            "output": "tCl",
            "modes": "s",
            "lensing": "no",
            "l_max_scalars": CLASS_LMAX,
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
        }
    )
    try:
        cosmo.compute()
        raw = cosmo.raw_cl(int(np.max(ells)) + 1)
    finally:
        cosmo.struct_cleanup()
        cosmo.empty()
    ell = raw["ell"]
    norm = (param["Tcmb"] * 1.0e6) ** 2
    dl = ell * (ell + 1.0) * raw["tt"] * norm / (2.0 * np.pi)
    return dl[np.asarray(ells, dtype=np.int64)]


def _spectrum_kwargs(cosmologies):
    from discoeb.cmb import cmb_k_grid

    k_values = cmb_k_grid(cosmologies[0], n=CMB_N_K, mode="ode", k_min=1.0e-5, k_max=0.5)
    return dict(k_values=k_values, ells=TT_ELLS, n_save=CMB_N_SAVE, n_k_fine=1500)


def test_projection_reproduces_the_bessel_transfer_of_a_delta_source():
    """A unit-normalized narrow source at tau_rec projects to Theta_l(k) = j_l(k chi_rec)."""

    import jax.numpy as jnp
    from scipy import special

    from discoeb.cmb import (
        SOURCE_CHANNELS,
        _lagrange_shift_stencil,
        projection_operator,
        theta_ell_batch,
    )

    tau0, tau_rec, width = 300.0, 100.0, 0.05
    tau = np.linspace(tau_rec - 20.0 * width, tau_rec + 20.0 * width, 4001)
    k = np.asarray([0.01, 0.05, 0.1])
    ells = np.asarray([2, 8, 20])
    gaussian = np.exp(-0.5 * ((tau - tau_rec) / width) ** 2) / (width * np.sqrt(2.0 * np.pi))

    # Only the j_l channel is fed; phi = 0 leaves no ISW term, and the fine k
    # grid is the solve grid itself, so the interpolation matrix is the identity.
    channels = np.zeros((1, SOURCE_CHANNELS, len(k), len(tau)))
    channels[0, 0] = gaussian[None, :]
    interp, bessel, _ = projection_operator(ells, k, k, tau, tau0, dtype=jnp.float64)
    idx, w = _lagrange_shift_stencil(tau, 0.0)
    band = jnp.ones((1, len(ells), len(k)), dtype=bool)
    theta = theta_ell_batch(
        jnp.asarray(channels), jnp.asarray(tau), jnp.asarray(idx)[None], jnp.asarray(w)[None],
        band, interp, bessel,
    )

    expected = np.stack([special.spherical_jn(int(ell), k * (tau0 - tau_rec)) for ell in ells])
    np.testing.assert_allclose(np.asarray(theta[0]), expected, rtol=5e-3, atol=1e-6)


@pytest.mark.skipif(not _cuda_available(), reason="numba-CUDA requires a CUDA GPU")
@pytest.mark.parametrize("n_cosmologies", BATCH_SIZE_CASES)
def test_tt_power_spectrum_matches_class(n_cosmologies, benchmark):
    """Batched unlensed scalar TT ``D_ell`` vs CLASS, timed over the batch size."""

    from discoeb.cmb import compute_cl_power_spectrum_batch

    cosmologies = _perturbed_cosmologies(n_cosmologies)
    _, _, dl_ours = benchmark.pedantic(
        compute_cl_power_spectrum_batch,
        args=(cosmologies,),
        kwargs=_spectrum_kwargs(cosmologies),
        rounds=1,
        warmup_rounds=1,
        iterations=1,
    )
    dl_ours = np.asarray(dl_ours)
    assert dl_ours.shape == (n_cosmologies, len(TT_ELLS))
    assert np.isfinite(dl_ours).all()

    dl_class = np.stack([_class_unlensed_tt(c, TT_ELLS) for c in cosmologies])
    rel = np.abs(dl_ours / dl_class - 1.0)
    i, j = np.unravel_index(np.argmax(rel), rel.shape)
    assert float(rel[i, j]) < DL_GATE, (
        f"D_ell off by {rel[i, j]:.3g} for cosmology {i} at l={TT_ELLS[j]}"
    )


@pytest.mark.skipif(not _cuda_available(), reason="numba-CUDA requires a CUDA GPU")
@pytest.mark.parametrize(
    "one_column_per_trajectory",
    [
        pytest.param(True, id="one-column-per-thread"),
        pytest.param(False, id="five-columns-per-thread"),
    ],
)
def test_tt_power_spectrum_derivatives(one_column_per_trajectory, benchmark):
    """Time and check dD_l/d(background densities) by forward sensitivity.

    The CMB counterpart of ``test_matter_power_spectrum_derivatives``: the five
    densities enter the hierarchy and the source algebra, and both paths are in
    the total derivative. Timed as there, ``jax.value_and_grad`` of the summed
    spectrum; checked per multipole by one forward solve per density against
    central differences of the same function, and the gradient against the
    summed forward derivatives (the transposed sensitivity rule against the
    untransposed one).

    The error is measured against the largest ``|dD_l/d(density)|`` over the
    multipoles, not multipole by multipole: ``dD_l/dgrhob`` changes sign across
    the acoustic peaks (it passes through zero near ``l ~ 100``) and its sum
    over these multipoles cancels threefold, so neither a relative error at a
    single ``l`` nor the summed derivative of the ``P(k)`` test is
    well-conditioned here. At rtol = atol = 1e-4 the sensitivity and the
    central difference each carry ~5e-4 of that scale (both move to within
    ~1e-4 of each other at 1e-5), hence the gate. The projection runs in
    float64, since its float32 rounding (2.6e-4 in ``D_l``) would swamp any
    finite difference. The spectrum is the one of
    ``test_tt_power_spectrum_matches_class`` (same modes, saves and multipoles).
    """

    import jax
    import jax.numpy as jnp

    from discoeb.cmb import cl_power_spectrum_jax
    from discoeb.perturbations import (
        PHYSICAL_DENSITY_COLUMNS,
        _as_cosmology,
        density_coefficients,
    )

    cosmology = _perturbed_cosmologies(1)[0]
    kwargs = _spectrum_kwargs([cosmology])
    densities = jnp.asarray(
        np.asarray(density_coefficients(_as_cosmology(cosmology)), dtype=np.float64)
    )
    assert densities.shape == (len(PHYSICAL_DENSITY_COLUMNS),)

    def spectrum(d):
        _, _, dl = cl_power_spectrum_jax(
            cosmology, d, one_column_per_trajectory=one_column_per_trajectory, **kwargs
        )
        return dl

    value_and_grad = jax.value_and_grad(lambda d: jnp.sum(spectrum(d)))

    def run():
        value, grad = value_and_grad(densities)
        return float(value), np.asarray(jax.block_until_ready(grad))

    value, grad = benchmark.pedantic(run, rounds=1, warmup_rounds=1, iterations=1)
    assert np.isfinite(value) and np.isfinite(grad).all(), f"non-finite gradient {grad}"

    summed = np.empty_like(grad)
    for i, name in enumerate(DENSITY_NAMES):
        unit = jnp.zeros_like(densities).at[i].set(1.0)
        _, sens = jax.jvp(spectrum, (densities,), (unit,))
        sens = np.asarray(jax.block_until_ready(sens))
        summed[i] = sens.sum()

        step = 1.0e-5 * abs(float(densities[i]))
        plus = np.asarray(spectrum(densities.at[i].add(step)))
        minus = np.asarray(spectrum(densities.at[i].add(-step)))
        finite = (plus - minus) / (2.0 * step)
        scale = np.max(np.abs(finite))
        err = np.abs(sens - finite) / scale
        worst = int(np.argmax(err))
        assert err[worst] < CL_DERIV_GATE, (
            f"dD_l/d{name} at l={TT_ELLS[worst]}: sensitivity {sens[worst]:.8e} "
            f"against central difference {finite[worst]:.8e}, "
            f"{err[worst]:.3g} of the derivative's scale {scale:.3e}"
        )
    np.testing.assert_allclose(grad, summed, rtol=1.0e-6, err_msg="grad vs summed jvp")
