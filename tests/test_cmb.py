"""The unlensed TT, TE and EE power spectra against CLASS, and the time of their batched solve.

The end-to-end spectra -- perturbation solve on modax's Rodas5P kernel over the
CMB wave modes with the sources evaluated in-kernel by the save hook, then the
batched line-of-sight projection and the ``C_l`` quadrature -- are run for the
same cosmologies as ``test_matter_power_spectrum_matches_class`` (one, and a
batch of 128 nearby flat LCDM models) and each is checked against its own
CLASS reference, run with CLASS's own recfast and no reionization (the
cosmologies here have ``tau_reio = 0``; ``tests/test_reionization.py`` checks
the spectra with reionization). TT is gated at :data:`DL_GATE`, EE and TE at
:data:`POL_GATE`; TE crosses zero, so its error is measured against ``sqrt(TT
EE)`` everywhere and relative to TE itself where the two are well correlated.

One internal check backs that up: the matrix-product projection must reproduce
the analytic transfer functions of a delta-like source, ``Theta_l(k) = j_l(k
chi_rec) + j_l''(k chi_rec)`` from the temperature rows and ``sqrt((l+2)!/(l-2)!)
j_l / x^2`` from the E-mode row, which pins the Bessel tables, their derivative
recurrences and the ``tau`` quadrature independently of any cosmology.

The benchmark times one call of :func:`compute_cl_power_spectrum_batch` at
both batch sizes on a *fresh* batch: the warm-up round runs a distinct batch of
the same shape, so the timed round finds the kernels and JAX programs compiled
but nothing of its own cosmologies cached, and pays what an iteration of an
inference loop pays -- the RECFAST thermodynamics, the device tables, the
perturbation solve and the projection. It waits for the device: the call
returns JAX arrays as soon as the launches are queued, so the wall time of the
call alone would be the host's dispatch, not the solve.

A second timing (``test_power_spectra_successive_batches``) is the
inference loop's pattern: one compile, then batch after batch of 128 fresh
cosmologies, each paying its thermodynamics solve and table copy but never a
kernel compilation. It issues four such batches back to back, waits once at
the end, and holds them to four times one fresh batch, with modax's
kernel-build counter checked to be unchanged.

The derivatives of the three spectra with respect to the five background
densities (:func:`cl_power_spectrum_jax`, the CMB counterpart of
``matter_power_spectrum_jax``) are checked against central differences and timed
in both layouts of the sensitivity system, as ``test_matter_power_spectrum_derivatives``
does for ``P(k)``.
"""

import numpy as np
import pytest

from test_perturbations import (
    BATCH_SIZE_CASES,
    DENSITY_NAMES,
    _perturbed_cosmologies,
    _successive_batches,
    _wall_time,
    _warmup_cosmologies,
)

# Multipoles spanning the Sachs-Wolfe plateau, the acoustic peaks and the onset
# of Silk damping.
TT_ELLS = np.asarray([2, 10, 20, 40, 80, 120, 180, 240, 320, 500, 800, 1200], dtype=np.int64)

# Wave modes the perturbation hierarchy is solved on before the sources are
# interpolated onto the finer line-of-sight grid, and the number of saved times
# (the grids of ``_spectrum_kwargs``).
CMB_N_K = 128
CMB_N_SAVE = 1000

DL_GATE = 5.0e-3
"""Relative gate on the TT ``D_l``."""

POL_GATE = 2.0e-2
"""Gate on EE (relative) and on TE: against ``sqrt(TT EE)`` at every multipole, and
relative to TE itself where ``|TE| > TE_CORRELATION_MIN sqrt(TT EE)``."""

TE_CORRELATION_MIN = 0.3

SPECTRUM_NAMES = ("TT", "TE", "EE")

# Batches of ``CMB_N_K``-mode, 128-cosmology spectra issued in sequence by
# ``test_power_spectra_successive_batches``, the cosmologies of each batch
# checked against CLASS, and how far past ``SUCCESSIVE_BATCHES`` fresh single
# batches the sequence may run. A fresh batch is dominated by its
# thermodynamics solve and host table work, whose wall time is less steady
# than the kernel's; a recompile would add ~45 s, far outside the tolerance.
SUCCESSIVE_BATCHES = 4
SUCCESSIVE_BATCH_SIZE = 128
SUCCESSIVE_CLASS_CHECKS = 4
SUCCESSIVE_TOLERANCE = 0.25

# Gate on dD_l/d(density) as a fraction of the derivative's largest value over
# the multipoles; see ``test_power_spectra_derivatives``.
CL_DERIV_GATE = 2.0e-3

# A central difference is only checked where the change it measures,
# ``2 step |dD_l/d(density)|``, exceeds this fraction of ``D_l``: below it the
# difference of two float64 spectra is rounding (the dark-energy density moves
# EE and TE, which have no late-time source without reionization, by less).
FD_RESOLVABLE = 1.0e-9

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


def _class_unlensed(param, ells):
    """Return CLASS unlensed scalar ``D_ell`` in ``muK^2`` at ``ells``, a dict over ``tt, te, ee``."""

    Class = pytest.importorskip("classy").Class
    h = param["H0"] / 100.0
    cosmo = Class()
    cosmo.set(
        {
            "output": "tCl,pCl",
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
    factor = ell * (ell + 1.0) * norm / (2.0 * np.pi)
    ells = np.asarray(ells, dtype=np.int64)
    return {name: (raw[name] * factor)[ells] for name in ("tt", "te", "ee")}


def _assert_matches_class(dl_ours, dl_class, ells, who=lambda i: f"cosmology {i}"):
    """Gate a batch's ``D_ell`` (a ``Spectra`` of ``(n_c, n_ell)``) against its CLASS references.

    ``dl_class`` is one :func:`_class_unlensed` dict per cosmology; ``who``
    names cosmology ``i`` in the failure message.
    """

    tt_c, te_c, ee_c = (np.stack([d[name] for d in dl_class]) for name in ("tt", "te", "ee"))
    tt, te, ee = (np.asarray(s) for s in dl_ours)
    assert tt.shape == te.shape == ee.shape == tt_c.shape
    assert np.isfinite(tt).all() and np.isfinite(te).all() and np.isfinite(ee).all()

    def gate(err, limit, name):
        i, j = np.unravel_index(np.argmax(err), err.shape)
        assert float(err[i, j]) < limit, f"{name} off by {err[i, j]:.3g} for {who(i)} at l={ells[j]}"

    gate(np.abs(tt / tt_c - 1.0), DL_GATE, "TT D_ell")
    gate(np.abs(ee / ee_c - 1.0), POL_GATE, "EE D_ell")
    scale = np.sqrt(tt_c * ee_c)
    diff = np.abs(te - te_c)
    gate(diff / scale, POL_GATE, "TE D_ell against sqrt(TT EE)")
    correlated = np.abs(te_c) > TE_CORRELATION_MIN * scale
    assert correlated.any(axis=1).all()
    gate(np.where(correlated, diff / np.abs(te_c), 0.0), POL_GATE, "TE D_ell")


def _spectrum_kwargs(cosmologies):
    """The grids of every spectrum here, shaped by the first cosmology.

    Built once and passed as ``grids``, as an inference loop would: the
    projection operator is then shared by every batch, and the save grid ends a
    margin past the fiducial's ``tau0`` to cover the batches' spread.
    """

    from discoeb.cmb import cmb_grids, cmb_k_grid

    k_values = cmb_k_grid(cosmologies[0], n=CMB_N_K, mode="ode", k_min=1.0e-5, k_max=0.5)
    return dict(
        grids=cmb_grids(
            cosmologies[0], k_values=k_values, ells=TT_ELLS, n_save=CMB_N_SAVE, n_k_fine=1500
        )
    )


def test_projection_reproduces_the_bessel_transfer_of_a_delta_source():
    """A unit-normalized narrow source at tau_rec projects to the Bessel kernels at k chi_rec."""

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

    # The j_l channel and the quadrupole channel (j_l'' for the temperature,
    # j_l / x^2 for E) are fed; phi = 0 leaves no ISW term, and the fine k grid
    # is the solve grid itself, so the interpolation matrix is the identity.
    channels = np.zeros((1, SOURCE_CHANNELS, len(k), len(tau)))
    channels[0, 0] = gaussian[None, :]
    channels[0, 2] = gaussian[None, :]
    interp, bessel, bessel_e, _ = projection_operator(ells, k, k, tau, tau0, dtype=jnp.float64)
    idx, w = _lagrange_shift_stencil(tau, 0.0)
    band = jnp.ones((1, len(ells), len(k)), dtype=bool)
    theta, theta_e = theta_ell_batch(
        jnp.asarray(channels), jnp.asarray(tau), jnp.asarray(idx)[None], jnp.asarray(w)[None],
        band, interp, bessel, bessel_e,
    )

    x = k * (tau0 - tau_rec)
    expected_t, expected_e = [], []
    for ell in ells:
        jl = special.spherical_jn(int(ell), x)
        jld = special.spherical_jn(int(ell), x, derivative=True)
        jldd = -2.0 / x * jld + (ell * (ell + 1.0) / x**2 - 1.0) * jl
        expected_t.append(jl + jldd)
        expected_e.append(np.sqrt((ell + 2.0) * (ell + 1.0) * ell * (ell - 1.0)) * jl / x**2)
    np.testing.assert_allclose(np.asarray(theta[0]), np.stack(expected_t), rtol=5e-3, atol=1e-6)
    np.testing.assert_allclose(np.asarray(theta_e[0]), np.stack(expected_e), rtol=5e-3, atol=1e-6)


@pytest.mark.skipif(not _cuda_available(), reason="numba-CUDA requires a CUDA GPU")
@pytest.mark.parametrize("n_cosmologies", BATCH_SIZE_CASES)
def test_power_spectra_match_class(n_cosmologies, benchmark):
    """Batched unlensed scalar TT, TE and EE ``D_ell`` vs CLASS, timed fresh over the batch size."""

    import jax

    from discoeb.cmb import compute_cl_power_spectrum_batch

    cosmologies = _perturbed_cosmologies(n_cosmologies)
    kwargs = _spectrum_kwargs(cosmologies)
    # The warm-up round compiles on other cosmologies; the timed round is fresh.
    rounds = iter([_warmup_cosmologies(n_cosmologies), cosmologies])

    def spectrum():
        return jax.block_until_ready(compute_cl_power_spectrum_batch(next(rounds), **kwargs))

    _, _, dl_ours = benchmark.pedantic(spectrum, rounds=1, warmup_rounds=1, iterations=1)
    assert np.asarray(dl_ours.tt).shape == (n_cosmologies, len(TT_ELLS))
    _assert_matches_class(dl_ours, [_class_unlensed(c, TT_ELLS) for c in cosmologies], TT_ELLS)


def _kernel_builds() -> int:
    """How many Rodas5P kernels modax has built so far in this process."""

    from modax.rodas5P import _make_kernel

    return _make_kernel.cache_info().misses


@pytest.mark.skipif(not _cuda_available(), reason="numba-CUDA requires a CUDA GPU")
def test_power_spectra_successive_batches(benchmark):
    """One compile, then four fresh batches of 128 cosmologies at the cost of four single ones.

    The pattern of an inference loop: a sampler proposes a new batch of
    cosmologies every iteration and the model must not compile again for it.
    The right-hand side and the source hook are compiled once per batch
    *shape*, against device buffers that every batch's thermodynamics tables
    and visibility are copied into (``DeviceTableSlot``); modax's kernel cache
    is keyed on those device functions, so a fresh batch of the same size is a
    cache hit.

    A first batch warms up: it compiles the kernel and the projection, and
    primes the JAX programs of the batched thermodynamics solve. A second fresh
    batch, run alone and waited on, is the single-batch reference: its cost is
    the thermodynamics solve, the host table work, the table copy, the launch
    and the projection -- everything an iteration pays. Four more fresh batches
    are then issued back to back and waited on once; they must take no longer
    than four references (within :data:`SUCCESSIVE_TOLERANCE`), and modax must
    have built no kernel since the warm-up. The copy into the slot waits for
    the device, so the four run strictly in sequence, which is what the loop
    does anyway: the sampler needs one batch's spectra before it proposes the
    next.

    Re-running the four afterwards (the benchmark's round, with every host
    table cached) must reproduce the fresh run's spectra -- the slots are
    reloaded batch by batch -- and a few cosmologies of each batch are gated
    against CLASS as in the single-batch test.
    """

    import jax

    from discoeb.cmb import compute_cl_power_spectrum_batch

    warm, reference, *batches = _successive_batches(
        2 + SUCCESSIVE_BATCHES, SUCCESSIVE_BATCH_SIZE
    )
    kwargs = _spectrum_kwargs(warm)

    def alone(cosmologies):
        return jax.block_until_ready(compute_cl_power_spectrum_batch(cosmologies, **kwargs))

    def successive():
        # Every batch is dispatched before any is waited for.
        return jax.block_until_ready(
            [compute_cl_power_spectrum_batch(cosmologies, **kwargs) for cosmologies in batches]
        )

    alone(warm)
    builds = _kernel_builds()
    assert builds >= 1

    single, _ = _wall_time(lambda: alone(reference))
    four, fresh = _wall_time(successive)
    assert _kernel_builds() == builds, "a fresh batch of the same shape rebuilt the kernel"
    assert four <= (1.0 + SUCCESSIVE_TOLERANCE) * SUCCESSIVE_BATCHES * single, (
        f"{SUCCESSIVE_BATCHES} successive fresh batches took {four:.3f} s against "
        f"{single:.3f} s for one fresh batch alone ({four / single:.2f}x)"
    )
    dl_fresh = [np.stack([np.asarray(s) for s in dl]) for _, _, dl in fresh]

    results = benchmark.pedantic(successive, rounds=1, warmup_rounds=0, iterations=1)
    assert _kernel_builds() == builds

    checked = np.linspace(0, SUCCESSIVE_BATCH_SIZE - 1, SUCCESSIVE_CLASS_CHECKS).astype(int)
    for b, (cosmologies, (_, _, dl_ours)) in enumerate(zip(batches, results)):
        dl_ours = np.stack([np.asarray(s) for s in dl_ours])
        assert dl_ours.shape == (len(SPECTRUM_NAMES), SUCCESSIVE_BATCH_SIZE, len(TT_ELLS))
        assert np.isfinite(dl_ours).all()
        np.testing.assert_allclose(
            dl_ours, dl_fresh[b], rtol=1.0e-6, err_msg=f"batch {b} re-run vs fresh"
        )
        _assert_matches_class(
            dl_ours[:, checked],
            [_class_unlensed(cosmologies[i], TT_ELLS) for i in checked],
            TT_ELLS,
            who=lambda i, b=b: f"cosmology {checked[i]} of batch {b}",
        )


@pytest.mark.skipif(not _cuda_available(), reason="numba-CUDA requires a CUDA GPU")
@pytest.mark.parametrize(
    "one_column_per_trajectory",
    [
        pytest.param(True, id="one-column-per-thread"),
        pytest.param(False, id="five-columns-per-thread"),
    ],
)
def test_power_spectra_derivatives(one_column_per_trajectory, benchmark):
    """Time and check dD_l/d(background densities) of TT, TE and EE by forward sensitivity.

    The CMB counterpart of ``test_matter_power_spectrum_derivatives``: the five
    densities enter the hierarchy and the source algebra, and both paths are in
    the total derivative. Timed as there, ``jax.value_and_grad`` of the summed
    spectra; checked per multipole by one forward solve per density against
    central differences of the same function, and the gradient against the
    summed forward derivatives (the transposed sensitivity rule against the
    untransposed one).

    The error is measured against the largest ``|dD_l/d(density)|`` of each
    spectrum over the multipoles, not multipole by multipole: ``dD_l/dgrhob`` changes sign across
    the acoustic peaks (it passes through zero near ``l ~ 100``) and its sum
    over these multipoles cancels threefold, so neither a relative error at a
    single ``l`` nor the summed derivative of the ``P(k)`` test is
    well-conditioned here. At rtol = atol = 1e-4 the sensitivity and the
    central difference each carry ~5e-4 of that scale (both move to within
    ~1e-4 of each other at 1e-5), hence the gate. The projection runs in
    float64, since its float32 rounding (2.6e-4 in ``D_l``) would swamp any
    finite difference. The spectra are those of
    ``test_power_spectra_match_class`` (same modes, saves and multipoles).
    Entries whose central difference is below the rounding floor
    (:data:`FD_RESOLVABLE`) are not compared: the dark-energy density changes
    the polarization spectra, which have no late-time source here, by less
    than float64 resolves at the step; the sensitivity rule for those spectra
    is validated by the other four densities.
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
        return jnp.stack(dl)  # (3, n_ell): TT, TE, EE

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
        resolvable = 2.0 * step * np.abs(finite) > FD_RESOLVABLE * np.abs(plus)
        assert resolvable[0].any(), f"dD_l^TT/d{name} below the finite-difference resolution everywhere"
        scale = np.max(np.abs(np.where(resolvable, finite, 0.0)), axis=1, keepdims=True)
        err = np.where(resolvable, np.abs(sens - finite) / np.where(scale > 0.0, scale, 1.0), 0.0)
        s, worst = np.unravel_index(np.argmax(err), err.shape)
        assert err[s, worst] < CL_DERIV_GATE, (
            f"dD_l^{SPECTRUM_NAMES[s]}/d{name} at l={TT_ELLS[worst]}: sensitivity {sens[s, worst]:.8e} "
            f"against central difference {finite[s, worst]:.8e}, "
            f"{err[s, worst]:.3g} of the derivative's scale {scale[s, 0]:.3e}"
        )
    np.testing.assert_allclose(grad, summed, rtol=1.0e-6, err_msg="grad vs summed jvp")
