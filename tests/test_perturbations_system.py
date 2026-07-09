"""End-to-end Einstein-Boltzmann perturbation solve vs CLASS.

Solves the linear matter power spectrum with the numba-CUDA Rodas5P solver
(:func:`discoeb.perturbations_system.solve_matter_power_spectrum`, which uses
``rodas5Pnumba`` + the Schur-EB block-LU) and compares it against **CLASS**
linear ``P(k)`` via the ``classy`` Python package, for:

    * flat LambdaCDM + massless neutrinos (Stage 1),
    * spatial curvature (``Omega_k != 0``), which modifies the metric relations,
    * dynamical dark energy (CPL fluid, ``w > -1`` throughout), which adds the
      DE fluid perturbations to the state and grows the Schur-EB dense core, and
    * massive neutrinos, which add one ``psi_l`` momentum-bin hierarchy per
      quadrature node.

Requires a CUDA GPU with numba-CUDA (the tests skip otherwise).
"""

import dataclasses

import numpy as np
import pytest

from cosmologies import BENCHMARK_COSMOLOGIES, cosmology_to_class_params


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

# Log-uniform matter-power benchmark wavenumbers in Mpc^-1.
MATTER_POWER_K = np.geomspace(2.0e-3, 0.3, 24, dtype=np.float64)

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
def test_matter_power_spectrum_matches_class():
    """Flat LambdaCDM + massless neutrinos: numba-CUDA P(k) vs CLASS."""

    from discoeb.perturbations_system import solve_matter_power_spectrum

    pk_ours = solve_matter_power_spectrum(MATTER_POWER_K, DEFAULT_COSMOLOGY)
    pk_class = _class_linear_pk(DEFAULT_COSMOLOGY, MATTER_POWER_K)

    rel = np.abs(pk_ours / pk_class - 1.0)
    assert float(np.max(rel)) < PK_GATE


@pytest.mark.skipif(not _cuda_available(), reason="numba-CUDA requires a CUDA GPU")
def test_matter_power_spectrum_matches_class_curvature():
    """Open universe (Omega_k = 0.05): numba-CUDA P(k) vs CLASS.

    Exercises the curved-geometry metric corrections (the ``s2^2`` factors and
    the ``K h'`` term in ``eta'``), which suppress growth by ~9% in P(k).
    """

    from discoeb.perturbations_system import solve_matter_power_spectrum

    pk_ours = solve_matter_power_spectrum(MATTER_POWER_K, OPEN_COSMOLOGY)
    pk_class = _class_linear_pk(OPEN_COSMOLOGY, MATTER_POWER_K)

    rel = np.abs(pk_ours / pk_class - 1.0)
    assert float(np.max(rel)) < PK_GATE


@pytest.mark.skipif(not _cuda_available(), reason="numba-CUDA requires a CUDA GPU")
def test_matter_power_spectrum_matches_class_dark_energy():
    """Dynamical dark energy (CPL fluid, w > -1): numba-CUDA P(k) vs CLASS fluid."""

    from discoeb.perturbations_system import solve_matter_power_spectrum

    pk_ours = solve_matter_power_spectrum(MATTER_POWER_K, DARK_ENERGY_COSMOLOGY)
    pk_class = _class_linear_pk(DARK_ENERGY_COSMOLOGY, MATTER_POWER_K)

    rel = np.abs(pk_ours / pk_class - 1.0)
    assert float(np.max(rel)) < PK_GATE


@pytest.mark.skipif(not _cuda_available(), reason="numba-CUDA requires a CUDA GPU")
def test_matter_power_spectrum_matches_class_massive_neutrinos():
    """One massive neutrino (0.06 eV): numba-CUDA P_cb(k) vs CLASS.

    Each of the ``NQMAX`` momentum bins contributes a ``psi_l`` hierarchy whose
    lowest three multipoles join the Schur-EB dense core, and whose free-streaming
    tail becomes an extra tridiagonal block.
    """

    from discoeb.perturbations_system import solve_matter_power_spectrum

    pk_ours = solve_matter_power_spectrum(MATTER_POWER_K, MASSIVE_NU_COSMOLOGY)
    pk_class = _class_linear_pk(MASSIVE_NU_COSMOLOGY, MATTER_POWER_K)

    rel = np.abs(pk_ours / pk_class - 1.0)
    assert float(np.max(rel)) < PK_GATE


@pytest.mark.skipif(not _cuda_available(), reason="numba-CUDA requires a CUDA GPU")
def test_matter_power_spectrum_matches_class_all_extensions():
    """Curvature + CPL dark energy + massive neutrinos together: P_cb(k) vs CLASS."""

    from discoeb.perturbations_system import solve_matter_power_spectrum

    pk_ours = solve_matter_power_spectrum(MATTER_POWER_K, EXTENDED_COSMOLOGY)
    pk_class = _class_linear_pk(EXTENDED_COSMOLOGY, MATTER_POWER_K)

    rel = np.abs(pk_ours / pk_class - 1.0)
    assert float(np.max(rel)) < PK_GATE
