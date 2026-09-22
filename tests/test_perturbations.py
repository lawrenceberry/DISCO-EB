"""GPU perturbation solves against CLASS: the matter power spectrum and its derivatives."""

import numpy as np
import pytest


REFERENCE_COSMOLOGY = {
    "Omegam": (0.02238280 + 0.1201075) / 0.6732117**2,
    "Omegab": 0.02238280 / 0.6732117**2,
    "w_DE_0": -1.0,
    "w_DE_a": 0.0,
    "cs2_DE": 1.0,
    "Omegak": 0.0,
    "A_s": 2.100549e-9,
    "n_s": 0.9660499,
    "H0": 67.32117,
    "Tcmb": 2.7255,
    "YHe": 0.2454006,
    "Neff": 3.046,
    "Nmnu": 0.0,
    "mnu": 0.0,
    "k_p": 0.05,
}

MATTER_POWER_K = np.geomspace(1.0e-4, 1.0, 128, dtype=np.float64)
PK_GATE = 5.0e-3
BATCH_SIZE_CASES = [pytest.param(1, id="N1"), pytest.param(128, id="N128")]


def _perturbed_cosmologies(n):
    """Return deterministic nearby flat LCDM parameter dictionaries."""

    if n == 1:
        return [REFERENCE_COSMOLOGY.copy()]
    phase = np.linspace(-1.0, 1.0, n, dtype=np.float64)
    cosmologies = []
    for ph in phase:
        param = REFERENCE_COSMOLOGY.copy()
        omega_b_h2 = 0.02238280 * (1.0 + 0.006 * ph)
        omega_c_h2 = 0.1201075 * (1.0 - 0.008 * ph)
        h = 0.6732117 * (1.0 + 0.004 * np.sin(np.pi * ph))
        param.update(
            {
                "Omegam": (omega_b_h2 + omega_c_h2) / h**2,
                "Omegab": omega_b_h2 / h**2,
                "H0": 100.0 * h,
                "YHe": REFERENCE_COSMOLOGY["YHe"]
                * (1.0 + 0.003 * np.cos(0.5 * np.pi * ph)),
                "Tcmb": REFERENCE_COSMOLOGY["Tcmb"] * (1.0 + 0.0015 * ph),
                "Neff": REFERENCE_COSMOLOGY["Neff"]
                * (1.0 + 0.002 * np.sin(2.0 * np.pi * ph)),
            }
        )
        cosmologies.append(param)
    return cosmologies


def _class_linear_pk(param, k_values):
    """Return the CLASS linear matter spectrum in Mpc cubed at redshift zero."""

    Class = pytest.importorskip("classy").Class
    h = param["H0"] / 100.0
    cosmo = Class()
    cosmo.set(
        {
            "h": h,
            "omega_b": param["Omegab"] * h**2,
            "omega_cdm": (param["Omegam"] - param["Omegab"]) * h**2,
            "T_cmb": param["Tcmb"],
            "YHe": param["YHe"],
            "N_ur": param["Neff"],
            "N_ncdm": 0,
            "Omega_k": 0.0,
            "A_s": param["A_s"],
            "n_s": param["n_s"],
            "k_pivot": 0.05,
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


@pytest.mark.parametrize("n_cosmologies", BATCH_SIZE_CASES)
def test_matter_power_spectrum_matches_class(n_cosmologies, benchmark):
    """Check and time one GPU launch over nearby flat LCDM cosmologies."""

    from discoeb.perturbations import solve_matter_power_spectrum_batch

    cosmologies = _perturbed_cosmologies(n_cosmologies)
    pk_ours = benchmark.pedantic(
        solve_matter_power_spectrum_batch,
        args=(MATTER_POWER_K, cosmologies),
        rounds=1,
        warmup_rounds=1,
        iterations=1,
    )
    pk_class = np.stack(
        [_class_linear_pk(param, MATTER_POWER_K) for param in cosmologies]
    )
    rel = np.abs(pk_ours / pk_class - 1.0)
    worst = np.unravel_index(np.argmax(rel), rel.shape)
    assert float(rel[worst]) < PK_GATE, (
        f"max relative error {rel[worst]:.6g} for cosmology {worst[0]} "
        f"at k={MATTER_POWER_K[worst[1]]:.6g} Mpc^-1"
    )


DENSITY_NAMES = ("grhog", "grhornomass", "grhoc", "grhob", "grhov")
DERIV_GATE = 1.0e-3


@pytest.mark.parametrize(
    "one_column_per_trajectory",
    [
        pytest.param(True, id="one-column-per-thread"),
        pytest.param(False, id="five-columns-per-thread"),
    ],
)
def test_matter_power_spectrum_derivatives(one_column_per_trajectory, benchmark):
    """Time and check dP(k)/d(background densities) by forward sensitivity.

    The five densities enter both the hierarchy and the matter weighting of
    ``delta_m``, so this is the total derivative. Central differences are the
    noisy side of the comparison: the solve runs at rtol = atol = 1e-4, which
    bounds how well any finite difference of it can agree.

    Both layouts of the sensitivity system are timed: the five columns stacked
    in each mode's thread, and one column per thread over five replicas of
    each mode (see :func:`matter_power_spectrum_jax`).
    """

    import jax
    import jax.numpy as jnp

    from discoeb.perturbations import (
        PHYSICAL_DENSITY_COLUMNS,
        _as_cosmology,
        density_coefficients,
        matter_power_spectrum_jax,
    )

    cosmology = _perturbed_cosmologies(1)[0]
    densities = jnp.asarray(
        np.asarray(density_coefficients(_as_cosmology(cosmology)), dtype=np.float64)
    )
    assert densities.shape == (len(PHYSICAL_DENSITY_COLUMNS),)

    def total_power(d):
        return jnp.sum(
            matter_power_spectrum_jax(
                MATTER_POWER_K,
                cosmology,
                d,
                one_column_per_trajectory=one_column_per_trajectory,
            )
        )

    value_and_grad = jax.value_and_grad(total_power)

    def run():
        value, grad = value_and_grad(densities)
        return float(value), np.asarray(jax.block_until_ready(grad))

    value, grad = benchmark.pedantic(run, rounds=1, warmup_rounds=1, iterations=1)
    assert np.isfinite(grad).all(), f"non-finite gradient {grad}"

    for i, name in enumerate(DENSITY_NAMES):
        step = 1.0e-5 * abs(float(densities[i]))
        plus = float(total_power(densities.at[i].add(step)))
        minus = float(total_power(densities.at[i].add(-step)))
        finite = (plus - minus) / (2.0 * step)
        rel = abs(grad[i] - finite) / max(abs(finite), 1e-300)
        assert rel < DERIV_GATE, (
            f"d(sum P)/d{name}: sensitivity {grad[i]:.8e} against central "
            f"difference {finite:.8e}, relative {rel:.3g}"
        )
