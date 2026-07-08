"""End-to-end Einstein-Boltzmann perturbation solve vs CLASS.

Solves the linear matter power spectrum with the numba-CUDA Rodas5P solver
(:func:`discoeb.perturbations_system.solve_matter_power_spectrum`, which uses
``rodas5Pnumba`` + the Schur-EB block-LU) and compares it against **CLASS**
linear ``P(k)`` via the ``classy`` Python package.

This is Stage 1 of the ported perturbation solver: flat LambdaCDM + massless
neutrinos. Requires a CUDA GPU with numba-CUDA (the test skips otherwise).
"""

import numpy as np
import pytest

from cosmologies import BENCHMARK_COSMOLOGIES


DEFAULT_COSMOLOGY = BENCHMARK_COSMOLOGIES["planck_2018_flat_lcdm"]

# Log-uniform matter-power benchmark wavenumbers in Mpc^-1.
MATTER_POWER_K = np.geomspace(1.0e-3, 0.5, 24, dtype=np.float64)

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
    """Return CLASS linear ``P(k)`` [Mpc^3] at z=0 for a flat massless cosmology."""

    Class = pytest.importorskip("classy").Class

    cosmo = Class()
    cosmo.set(
        {
            "h": cosmology.h,
            "omega_b": cosmology.omega_b_h2,
            "omega_cdm": cosmology.omega_c_h2,
            "T_cmb": cosmology.T_cmb,
            "YHe": cosmology.Y_He,
            "N_ur": cosmology.Neff_massless,
            "N_ncdm": 0,
            "Omega_k": 0.0,
            "A_s": cosmology.A_s,
            "n_s": cosmology.n_s,
            "k_pivot": cosmology.k_pivot,
            "output": "mPk",
            "P_k_max_1/Mpc": float(k_values[-1]) * 1.1,
            "z_max_pk": 0.0,
        }
    )
    try:
        cosmo.compute()
        return np.array([cosmo.pk_lin(float(k), 0.0) for k in k_values])
    finally:
        cosmo.struct_cleanup()
        cosmo.empty()


@pytest.mark.skipif(not _cuda_available(), reason="numba-CUDA requires a CUDA GPU")
def test_matter_power_spectrum_matches_class():
    """Compare the numba-CUDA linear matter power spectrum against CLASS."""

    from discoeb.perturbations_system import solve_matter_power_spectrum

    pk_ours = solve_matter_power_spectrum(MATTER_POWER_K, DEFAULT_COSMOLOGY)
    pk_class = _class_linear_pk(DEFAULT_COSMOLOGY, MATTER_POWER_K)

    rel = np.abs(pk_ours / pk_class - 1.0)
    assert float(np.max(rel)) < PK_GATE
