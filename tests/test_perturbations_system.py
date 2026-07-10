"""End-to-end Einstein-Boltzmann perturbation solve vs CLASS.

Solves the linear matter power spectrum with the numba-CUDA Rodas5P solver
(``rodas5Pnumba`` + the Schur-EB block-LU) and compares it against **CLASS**
linear ``P(k)`` via the ``classy`` Python package.

Two tests:

    * :func:`test_matter_power_spectrum_matches_class` -- a *batch* solve of
      ``N`` slightly-perturbed flat-LCDM cosmologies (``N in {1, 128}``) in one
      GPU launch (:func:`solve_matter_power_spectrum_batch`), timed with
      ``benchmark.pedantic``; the ``N = 128`` case shows the whole batch solving
      in essentially the same wall-time as a single cosmology.
    * :func:`test_matter_power_spectrum_extensions_match_class` -- one cosmology
      per solver code path (curvature, CPL dark energy, massive neutrinos, and
      all three at once), so a failure's id names the culprit.

Requires a CUDA GPU with numba-CUDA (the tests skip otherwise).
"""

import dataclasses

import numpy as np
import pytest

from cosmologies import (
    BENCHMARK_COSMOLOGIES,
    cosmology_to_class_params,
    perturbed_planck_cosmologies,
)


DEFAULT_COSMOLOGY = BENCHMARK_COSMOLOGIES["planck_2018_flat_lcdm"]

# A non-phantom-crossing quintessence cosmology: w(a) = -0.9 + 0.05(1-a) stays in
# [-0.9, -0.85] > -1, so the CPL fluid equations (ours and CLASS's non-PPF fluid)
# are regular. (The DESI benchmark crosses w = -1 and would require the PPF
# scheme, which is not implemented here.)
DARK_ENERGY_COSMOLOGY = dataclasses.replace(
    DEFAULT_COSMOLOGY, w_DE_0=-0.9, w_DE_a=0.05, cs2_DE=1.0
)

# Open universe. Omega_k is deliberately large (the Planck benchmark's 7e-4 would
# change P(k) by <0.1%, below the gate) so the curved-geometry corrections are
# actually exercised.
OPEN_COSMOLOGY = dataclasses.replace(DEFAULT_COSMOLOGY, Omegak=0.05)

# Planck's minimal-mass convention: one massive species of 0.06 eV.
MASSIVE_NU_COSMOLOGY = BENCHMARK_COSMOLOGIES["planck_2018_flat_lcdm_massive_nu"]

# All three extensions at once, to pin down how they compose: the DE fluid and
# the massive-neutrino hierarchies both extend the Schur-EB dense core, and their
# state-vector indices must interleave consistently.
EXTENDED_COSMOLOGY = dataclasses.replace(
    MASSIVE_NU_COSMOLOGY, Omegak=0.05, w_DE_0=-0.9, w_DE_a=0.05, cs2_DE=1.0
)

# The extension comparison cases, each exercising a distinct code path in the
# solver. These are purpose-built rather than the raw benchmark registry: the
# DESI benchmark's dark energy crosses w = -1 (phantom), which the CPL fluid
# cannot represent without the PPF scheme, and the Planck-curved benchmark's
# Omega_k = 7e-4 changes P(k) by less than the gate. Omega_k = 0.05 and a
# non-crossing w(a) are used instead so each extension is actually stressed.
EXTENSION_COSMOLOGY_CASES = [
    pytest.param(DEFAULT_COSMOLOGY, id="flat_lcdm"),
    pytest.param(OPEN_COSMOLOGY, id="curvature"),
    pytest.param(DARK_ENERGY_COSMOLOGY, id="dark_energy"),
    pytest.param(MASSIVE_NU_COSMOLOGY, id="massive_neutrinos"),
    pytest.param(EXTENDED_COSMOLOGY, id="all_extensions"),
]

# Batch sizes for the multi-cosmology solve: one cosmology, and a full 128-wide
# batch, both solved in a single GPU launch.
BATCH_SIZE_CASES = [pytest.param(1, id="N1"), pytest.param(128, id="N128")]

# Log-uniform matter-power benchmark wavenumbers in Mpc^-1: 128 modes over the
# standard 1e-4 -- 1 range (matching the DISCO2 benchmark), where the flat-LCDM
# solve still agrees with CLASS below the gate (~0.4% at k = 1).
MATTER_POWER_K = np.geomspace(1.0e-4, 1.0, 128, dtype=np.float64)

# The extensions run on a narrower range: at k ~ 1 the massive-neutrino
# free-streaming, with NQMAX = 3 momentum bins and the lmax truncations, drifts
# above the gate against CLASS, so the extension cases are validated over the
# 2e-3 -- 0.3 range where every code path agrees to < 0.5%.
EXTENSION_MATTER_POWER_K = np.geomspace(2.0e-3, 0.3, 32, dtype=np.float64)

# Relative-error gate on P(k) vs CLASS (matches the DISCO2 CAMB gate).
PK_GATE = 5.0e-3


def _cuda_available() -> bool:
    """Return whether a CUDA device is visible to numba."""

    try:
        from numba import cuda

        return cuda.is_available()
    except Exception:
        return False


def _class_linear_pk(cosmology, k_values):
    """Return CLASS linear ``P(k)`` [Mpc^3] at z=0 for ``cosmology``.

    Dynamical dark energy uses the CLASS fluid (non-PPF) scheme so it matches the
    perturbation solver's fluid treatment.
    """

    Class = pytest.importorskip("classy").Class

    params = cosmology_to_class_params(cosmology, output="mPk")
    params.update(
        {"P_k_max_1/Mpc": float(k_values[-1]) * 1.1, "z_max_pk": 0.0}
    )
    if cosmology.w_DE_0 != -1.0 or cosmology.w_DE_a != 0.0:
        params.update({"use_ppf": "no", "cs2_fld": cosmology.cs2_DE})

    cosmo = Class()
    cosmo.set(params)
    try:
        cosmo.compute()
        # Our delta_m is the CDM + baryon density contrast, so with massive
        # neutrinos the matching CLASS spectrum is pk_cb_lin, not pk_lin.
        pk = cosmo.pk_cb_lin if cosmology.num_massive_neutrinos > 0.0 else cosmo.pk_lin
        return np.array([pk(float(k), 0.0) for k in k_values])
    finally:
        cosmo.struct_cleanup()
        cosmo.empty()


@pytest.mark.skipif(not _cuda_available(), reason="numba-CUDA requires a CUDA GPU")
@pytest.mark.parametrize("n_cosmologies", BATCH_SIZE_CASES)
def test_matter_power_spectrum_matches_class(n_cosmologies, benchmark):
    """Batched linear matter P(k) vs CLASS, timed over the batch size.

    ``N`` slightly-perturbed flat-LCDM cosmologies are solved together in one
    ``rodas5Pnumba`` launch (:func:`solve_matter_power_spectrum_batch`), and each
    cosmology's P(k) is checked against its own CLASS reference -- so a mis-tagged
    thermodynamics table (wrong cosmology feeding a trajectory) would fail here.

    The batch solve is timed with ``benchmark.pedantic``: the warmup round absorbs
    the one-time numba-CUDA kernel compilation, so the single timed round measures
    the steady-state GPU solve. Because all ``N * n_k`` trajectories run in one
    launch, the ``N = 128`` timed round is nearly as fast as ``N = 1``.
    """

    from discoeb.perturbations_system import solve_matter_power_spectrum_batch

    cosmologies = perturbed_planck_cosmologies(n_cosmologies)

    pk_ours = benchmark.pedantic(
        solve_matter_power_spectrum_batch,
        args=(MATTER_POWER_K, cosmologies),
        rounds=1,
        warmup_rounds=1,
        iterations=1,
    )
    pk_class = np.stack([_class_linear_pk(c, MATTER_POWER_K) for c in cosmologies])

    rel = np.abs(pk_ours / pk_class - 1.0)
    assert float(np.max(rel)) < PK_GATE


@pytest.mark.skipif(not _cuda_available(), reason="numba-CUDA requires a CUDA GPU")
@pytest.mark.parametrize("cosmology", EXTENSION_COSMOLOGY_CASES)
def test_matter_power_spectrum_extensions_match_class(cosmology):
    """Linear matter P(k) vs CLASS across each solver code path.

    One case per extension, so a failure's parametrization id names the culprit:

        * ``flat_lcdm``          -- the baseline massless-neutrino solve;
        * ``curvature``          -- the curved-geometry metric corrections (the
          ``s2^2`` factors and the ``K h'`` term in ``eta'``), ~9% growth
          suppression;
        * ``dark_energy``        -- the CPL fluid perturbations, which grow the
          Schur-EB dense core by two variables;
        * ``massive_neutrinos``  -- one ``psi_l`` momentum-bin hierarchy per
          quadrature node (P_cb, since our delta_m is CDM + baryon);
        * ``all_extensions``     -- all three at once, checking their state-vector
          indices interleave consistently.
    """

    from discoeb.perturbations_system import solve_matter_power_spectrum

    pk_ours = solve_matter_power_spectrum(EXTENSION_MATTER_POWER_K, cosmology)
    pk_class = _class_linear_pk(cosmology, EXTENSION_MATTER_POWER_K)

    rel = np.abs(pk_ours / pk_class - 1.0)
    assert float(np.max(rel)) < PK_GATE
