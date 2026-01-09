"""
Performance benchmarks for individual CMB functions.

This module provides detailed performance testing for each function in discoeb.cmb
to identify bottlenecks and track performance regressions.

Usage:
    # Run all performance benchmarks
    pytest tests/test_cmb_performance.py -v --benchmark-only

    # Run specific function benchmark
    pytest tests/test_cmb_performance.py::test_benchmark_compute_theta_ell -v

    # Compare against baseline
    pytest tests/test_cmb_performance.py --benchmark-compare --benchmark-compare-fail=mean:10%
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
import time
from functools import partial

from discoeb.background import evolve_background
from discoeb.perturbations import evolve_perturbations_batched, compute_time_derivatives
from discoeb.cmb import (
    extract_perturbations,
    compute_visibility_functions,
    compute_neutrino_perturbations,
    compute_metric_perturbations,
    compute_polarization_terms,
    compute_source_term_isw,
    compute_source_term_sachs_wolfe,
    compute_source_term_doppler,
    compute_source_term_polarization,
    compute_source_function,
    compute_theta_ell,
    compute_Cell,
    compute_Dell,
    compute_Cell_spectrum_from_cosmo_params,
)
from discoeb.background import compute_background_quantities


# ==============================================================================
# Test Fixture: Pre-computed Data
# ==============================================================================

@pytest.fixture(scope="module")
def cmb_test_data():
    """Generate test data for CMB benchmarks.

    This fixture runs once per module to generate all the intermediate
    data needed for benchmarking individual functions.
    """
    # Enable 64-bit precision
    jax.config.update("jax_enable_x64", True)
    jax.config.update('jax_platform_name', 'gpu')

    # Clear compilation cache
    jax.clear_caches()

    # Standard cosmology parameters
    param = {
        'Omegam': 0.3099,
        'Omegab': 0.0488911,
        'H0': 67.742,
        'n_s': 0.96822,
        'A_s': 2.1064e-09,
        'mnu': 0.06,
        'Tcmb': 2.7255,
        'YHe': 0.248,
        'Neff': 2.046,
        'Nmnu': 1,
        'w_DE_0': -0.99,
        'w_DE_a': 0.0,
        'cs2_DE': 1.0,
        'Omegak': 0.0,
        'k_p': 0.05,
    }

    print("\n" + "="*70)
    print("Setting up CMB test data (this may take a minute)...")
    print("="*70)

    # 1. Background evolution
    print("  [1/3] Computing background evolution...")
    param = evolve_background(param=param, thermo_module='RECFAST', num_thermo=1024)

    # 2. Perturbation evolution
    print("  [2/3] Computing perturbation evolution...")
    aexp_out = jnp.concatenate([
        jnp.geomspace(3e-4, 5e-3, 256, endpoint=False),
        jnp.geomspace(5e-3, 1.0, 64)
    ])

    yout, kmodes, param = evolve_perturbations_batched(
        param=param,
        kmin=1e-4,
        kmax=1.0,
        num_k=128,  # Use fewer k-modes for faster testing
        aexp_out=aexp_out,
        rtol=1e-4,
        atol=1e-4,
        return_full=True,
        dologk=True,
    )

    # 3. Time derivatives
    print("  [3/3] Computing time derivatives...")
    tau = param['tau_out']
    yprime = compute_time_derivatives(yout, tau, kmodes, param)

    # Extract parameters
    lmaxg = param['lmaxg']
    lmaxgp = param['lmaxgp']
    lmaxr = param['lmaxr']
    nqmax = param['nqmax']

    # Pre-compute all intermediate quantities
    perturbations = extract_perturbations(yout, yprime, lmaxg, lmaxgp, lmaxr)
    background_quantities = compute_background_quantities(aexp_out, param)
    tau = param['tau_of_a_spline'].evaluate(aexp_out)
    visibility_functions = compute_visibility_functions(tau, param)
    neutrinos = compute_neutrino_perturbations(
        yout, yprime, aexp_out, param, nqmax, perturbations['iq0']
    )
    metric = compute_metric_perturbations(
        perturbations, neutrinos, background_quantities, param, kmodes, aexp_out
    )
    polarization_terms = compute_polarization_terms(
        perturbations, metric, visibility_functions, yout, yprime, kmodes, lmaxg, lmaxgp
    )
    source_results = compute_source_function(
        perturbations, metric, visibility_functions, yout, yprime, kmodes, lmaxg, lmaxgp
    )

    print("✓ CMB test data ready!")
    print("="*70 + "\n")

    return {
        'param': param,
        'yout': yout,
        'yprime': yprime,
        'kmodes': kmodes,
        'aexp_out': aexp_out,
        'tau': tau,
        'lmaxg': lmaxg,
        'lmaxgp': lmaxgp,
        'lmaxr': lmaxr,
        'nqmax': nqmax,
        'perturbations': perturbations,
        'background_quantities': background_quantities,
        'visibility_functions': visibility_functions,
        'neutrinos': neutrinos,
        'metric': metric,
        'polarization_terms': polarization_terms,
        'source_results': source_results,
    }


# ==============================================================================
# Benchmark Tests
# ==============================================================================

def test_benchmark_evolve_background(benchmark):
    """Benchmark evolve_background function.

    This is the first step in the CMB pipeline that computes the background
    cosmology evolution and thermal history.
    """
    # Standard cosmology parameters
    param = {
        'Omegam': 0.3099,
        'Omegab': 0.0488911,
        'H0': 67.742,
        'n_s': 0.96822,
        'A_s': 2.1064e-09,
        'mnu': 0.06,
        'Tcmb': 2.7255,
        'YHe': 0.248,
        'Neff': 2.046,
        'Nmnu': 1,
        'w_DE_0': -0.99,
        'w_DE_a': 0.0,
        'cs2_DE': 1.0,
        'Omegak': 0.0,
        'k_p': 0.05,
    }

    # Warmup - first call includes JIT compilation
    evolve_background(param=param.copy(), thermo_module='RECFAST', num_thermo=1024)

    # Benchmark
    result = benchmark(
        evolve_background,
        param=param.copy(),
        thermo_module='RECFAST',
        num_thermo=1024
    )

    # Verify output contains expected splines
    assert 'tau_of_a_spline' in result
    assert 'xe_of_loga_spline' in result


def test_benchmark_compute_background_quantities(benchmark, cmb_test_data):
    """Benchmark compute_background_quantities function.

    This function computes background density and equation of state quantities
    including neutrino density, dark energy density, and dark energy EOS.
    """
    data = cmb_test_data

    # Warmup
    compute_background_quantities(data['aexp_out'], data['param'])

    # Benchmark
    result = benchmark(
        compute_background_quantities,
        data['aexp_out'],
        data['param']
    )

    # Verify output - function returns rhonu, rho_Q, w_Q
    assert 'rhonu' in result
    assert 'rho_Q' in result
    assert 'w_Q' in result
    assert result['rhonu'].shape == data['aexp_out'].shape


def test_benchmark_evolve_perturbations_batched(benchmark):
    """Benchmark evolve_perturbations_batched function.

    This is the most computationally intensive step in the CMB pipeline,
    solving the coupled ODE system for all k-modes.
    """
    jax.config.update("jax_enable_x64", True)
    jax.config.update('jax_platform_name', 'gpu')

    # Standard cosmology parameters with background evolution
    param = {
        'Omegam': 0.3099,
        'Omegab': 0.0488911,
        'H0': 67.742,
        'n_s': 0.96822,
        'A_s': 2.1064e-09,
        'mnu': 0.06,
        'Tcmb': 2.7255,
        'YHe': 0.248,
        'Neff': 2.046,
        'Nmnu': 1,
        'w_DE_0': -0.99,
        'w_DE_a': 0.0,
        'cs2_DE': 1.0,
        'Omegak': 0.0,
        'k_p': 0.05,
    }

    # Background evolution (not timed here)
    param = evolve_background(param=param, thermo_module='RECFAST', num_thermo=1024)

    aexp_out = jnp.concatenate([
        jnp.geomspace(3e-4, 5e-3, 256, endpoint=False),
        jnp.geomspace(5e-3, 1.0, 64)
    ])

    # Warmup - first call includes JIT compilation
    evolve_perturbations_batched(
        param=param,
        kmin=1e-4,
        kmax=1.0,
        num_k=64,  # Use fewer k-modes for faster benchmarking
        aexp_out=aexp_out,
        rtol=1e-4,
        atol=1e-4,
        return_full=True,
        dologk=True,
    )

    # Benchmark with 64 k-modes
    yout, kmodes, param_out = benchmark(
        evolve_perturbations_batched,
        param=param,
        kmin=1e-4,
        kmax=1.0,
        num_k=64,
        aexp_out=aexp_out,
        rtol=1e-4,
        atol=1e-4,
        return_full=True,
        dologk=True,
    )

    # Verify output shape
    assert yout.shape[0] == 64  # num_k
    assert yout.shape[1] == len(aexp_out)


def test_benchmark_compute_time_derivatives(benchmark, cmb_test_data):
    """Benchmark compute_time_derivatives function.

    This function computes time derivatives of perturbations, which is needed
    for CMB source function calculations.
    """
    data = cmb_test_data

    # Warmup
    compute_time_derivatives(data['yout'], data['tau'], data['kmodes'], data['param'])

    # Benchmark
    yprime = benchmark(
        compute_time_derivatives,
        data['yout'],
        data['tau'],
        data['kmodes'],
        data['param']
    )

    # Verify output shape matches input
    assert yprime.shape == data['yout'].shape


def test_benchmark_extract_perturbations(benchmark, cmb_test_data):
    """Benchmark extract_perturbations function."""
    data = cmb_test_data

    result = benchmark(
        extract_perturbations,
        data['yout'],
        data['yprime'],
        data['lmaxg'],
        data['lmaxgp'],
        data['lmaxr']
    )

    # Verify output is valid
    assert 'deltag' in result
    assert result['deltag'].shape == data['yout'][:, :, data['perturbations']['idxg']].shape


def test_benchmark_compute_visibility_functions(benchmark, cmb_test_data):
    """Benchmark compute_visibility_functions function."""
    data = cmb_test_data

    # JIT compile first
    compute_visibility_functions(data['tau'], data['param'])

    result = benchmark(
        compute_visibility_functions,
        data['tau'],
        data['param']
    )

    # Verify output
    assert 'gvis' in result
    assert 'optical_depth' in result
    assert result['gvis'].shape == data['tau'].shape


def test_benchmark_compute_neutrino_perturbations(benchmark, cmb_test_data):
    """Benchmark compute_neutrino_perturbations function."""
    data = cmb_test_data

    # Warmup JIT compilation
    compute_neutrino_perturbations(
        data['yout'],
        data['yprime'],
        data['aexp_out'],
        data['param'],
        data['nqmax'],
        data['perturbations']['iq0']
    )

    result = benchmark(
        compute_neutrino_perturbations,
        data['yout'],
        data['yprime'],
        data['aexp_out'],
        data['param'],
        data['nqmax'],
        data['perturbations']['iq0']
    )

    # Verify output
    assert 'drhonu' in result
    assert result['drhonu'].shape[0] == data['yout'].shape[0]


def test_benchmark_compute_metric_perturbations(benchmark, cmb_test_data):
    """Benchmark compute_metric_perturbations function."""
    data = cmb_test_data

    # Warmup
    compute_metric_perturbations(
        data['perturbations'],
        data['neutrinos'],
        data['background_quantities'],
        data['param'],
        data['kmodes'],
        data['aexp_out']
    )

    result = benchmark(
        compute_metric_perturbations,
        data['perturbations'],
        data['neutrinos'],
        data['background_quantities'],
        data['param'],
        data['kmodes'],
        data['aexp_out']
    )

    # Verify output
    assert 'alpha' in result
    assert 'alphaprime' in result


def test_benchmark_compute_polarization_terms(benchmark, cmb_test_data):
    """Benchmark compute_polarization_terms function."""
    data = cmb_test_data

    # Warmup
    compute_polarization_terms(
        data['perturbations'],
        data['metric'],
        data['visibility_functions'],
        data['yout'],
        data['yprime'],
        data['kmodes'],
        data['lmaxg'],
        data['lmaxgp']
    )

    result = benchmark(
        compute_polarization_terms,
        data['perturbations'],
        data['metric'],
        data['visibility_functions'],
        data['yout'],
        data['yprime'],
        data['kmodes'],
        data['lmaxg'],
        data['lmaxgp']
    )

    # Verify output
    assert 'polarization_term' in result


def test_benchmark_compute_source_term_isw(benchmark, cmb_test_data):
    """Benchmark compute_source_term_isw function."""
    data = cmb_test_data

    result = benchmark(
        compute_source_term_isw,
        data['metric'],
        data['visibility_functions']
    )

    # Verify output shape
    assert result.shape[0] == data['kmodes'].shape[0]


def test_benchmark_compute_source_term_sachs_wolfe(benchmark, cmb_test_data):
    """Benchmark compute_source_term_sachs_wolfe function."""
    data = cmb_test_data

    result = benchmark(
        compute_source_term_sachs_wolfe,
        data['perturbations'],
        data['metric'],
        data['visibility_functions'],
        data['polarization_terms'],
        data['kmodes']
    )

    # Verify output shape
    assert result.shape[0] == data['kmodes'].shape[0]


def test_benchmark_compute_source_term_doppler(benchmark, cmb_test_data):
    """Benchmark compute_source_term_doppler function."""
    data = cmb_test_data

    result = benchmark(
        compute_source_term_doppler,
        data['perturbations'],
        data['metric'],
        data['visibility_functions'],
        data['polarization_terms'],
        data['kmodes']
    )

    # Verify output shape
    assert result.shape[0] == data['kmodes'].shape[0]


def test_benchmark_compute_source_term_polarization(benchmark, cmb_test_data):
    """Benchmark compute_source_term_polarization function."""
    data = cmb_test_data

    result = benchmark(
        compute_source_term_polarization,
        data['visibility_functions'],
        data['polarization_terms'],
        data['kmodes']
    )

    # Verify output shape
    assert result.shape[0] == data['kmodes'].shape[0]


def test_benchmark_compute_source_function(benchmark, cmb_test_data):
    """Benchmark compute_source_function function."""
    data = cmb_test_data

    # Warmup
    compute_source_function(
        data['perturbations'],
        data['metric'],
        data['visibility_functions'],
        data['yout'],
        data['yprime'],
        data['kmodes'],
        data['lmaxg'],
        data['lmaxgp']
    )

    result = benchmark(
        compute_source_function,
        data['perturbations'],
        data['metric'],
        data['visibility_functions'],
        data['yout'],
        data['yprime'],
        data['kmodes'],
        data['lmaxg'],
        data['lmaxgp']
    )

    # Verify output
    assert 'S' in result
    assert 'S1' in result
    assert 'S2' in result
    assert 'S3' in result
    assert 'S4' in result


def test_benchmark_compute_theta_ell(benchmark, cmb_test_data):
    """Benchmark compute_theta_ell function - LINE-OF-SIGHT INTEGRATION.

    This is typically the most expensive operation in CMB spectrum computation.
    """
    data = cmb_test_data
    S = data['source_results']['S']
    tau0 = data['param']['tau_of_a_spline'].evaluate(1.0)

    # Use smaller ellmax and nk_fine for benchmarking
    ellmax = 100
    nk_fine = 256

    # Warmup compilation
    compute_theta_ell(
        ellmax=ellmax,
        kmodes=data['kmodes'],
        tau=data['tau'],
        S=S,
        tau0=tau0,
        nk_fine=nk_fine,
        chunk_size=32,
        k_chunk_size=32
    )

    # Benchmark
    theta_ell, kmodes_fine = benchmark(
        compute_theta_ell,
        ellmax=ellmax,
        kmodes=data['kmodes'],
        tau=data['tau'],
        S=S,
        tau0=tau0,
        nk_fine=nk_fine,
        chunk_size=32,
        k_chunk_size=32
    )

    # Verify output shape
    assert theta_ell.shape == (nk_fine, ellmax + 1)


def test_benchmark_compute_Cell(benchmark, cmb_test_data):
    """Benchmark compute_Cell function."""
    data = cmb_test_data
    S = data['source_results']['S']
    tau0 = data['param']['tau_of_a_spline'].evaluate(1.0)

    # Compute theta_ell first
    ellmax = 100
    nk_fine = 256
    theta_ell, kmodes_fine = compute_theta_ell(
        ellmax=ellmax,
        kmodes=data['kmodes'],
        tau=data['tau'],
        S=S,
        tau0=tau0,
        nk_fine=nk_fine,
        chunk_size=32,
        k_chunk_size=32
    )

    # Warmup
    compute_Cell(theta_ell, kmodes_fine, data['param']['n_s'], data['param']['k_p'])

    # Benchmark
    Cell = benchmark(
        compute_Cell,
        theta_ell,
        kmodes_fine,
        data['param']['n_s'],
        data['param']['k_p']
    )

    # Verify output shape
    assert Cell.shape == (ellmax + 1,)


def test_benchmark_compute_Dell(benchmark, cmb_test_data):
    """Benchmark compute_Dell function."""
    data = cmb_test_data

    # Create dummy Cell array
    ellmax = 2500
    Cell = jnp.ones(ellmax + 1)

    # Warmup
    compute_Dell(Cell, data['param']['A_s'], data['param']['Tcmb'], ellmax=ellmax)

    # Benchmark
    ell, Dell = benchmark(
        compute_Dell,
        Cell,
        data['param']['A_s'],
        data['param']['Tcmb'],
        ellmax=ellmax
    )

    # Verify output
    assert len(ell) == len(Dell)
    assert ell[0] == 2  # Start from ell=2


# ==============================================================================
# Comprehensive Pipeline Timing Analysis
# ==============================================================================

def test_detailed_timing_analysis():
    """Comprehensive timing analysis of the complete CMB spectrum pipeline.

    This test times each major step in compute_Cell_spectrum_from_cosmo_params,
    including background evolution, perturbation solving, and all CMB calculations.
    It provides detailed statistics to identify bottlenecks in the complete pipeline
    from cosmological parameters to CMB angular power spectrum.
    """
    jax.config.update("jax_enable_x64", True)
    jax.config.update('jax_platform_name', 'gpu')
    jax.clear_caches()

    # Standard cosmology parameters
    param_dict = {
        'Omegam': 0.3099,
        'Omegab': 0.0488911,
        'H0': 67.742,
        'n_s': 0.96822,
        'A_s': 2.1064e-09,
        'mnu': 0.06,
        'Tcmb': 2.7255,
        'YHe': 0.248,
        'Neff': 2.046,
        'Nmnu': 1,
        'w_DE_0': -0.99,
        'w_DE_a': 0.0,
        'cs2_DE': 1.0,
        'Omegak': 0.0,
        'k_p': 0.05,
    }

    print("\n" + "="*80)
    print("COMPLETE CMB PIPELINE TIMING ANALYSIS")
    print("="*80)
    print("\nTiming each step in compute_Cell_spectrum_from_cosmo_params")
    print("to identify bottlenecks in the complete pipeline.\n")

    def time_step(func, name, *args, **kwargs):
        """Time a single step with warmup."""
        # Warmup
        result = func(*args, **kwargs)
        if hasattr(result, 'block_until_ready'):
            result.block_until_ready()
        elif isinstance(result, (tuple, list)):
            for r in result:
                if hasattr(r, 'block_until_ready'):
                    r.block_until_ready()
        elif isinstance(result, dict):
            for v in result.values():
                if hasattr(v, 'block_until_ready'):
                    v.block_until_ready()

        # Timed run
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        if hasattr(result, 'block_until_ready'):
            result.block_until_ready()
        elif isinstance(result, (tuple, list)):
            for r in result:
                if hasattr(r, 'block_until_ready'):
                    r.block_until_ready()
        elif isinstance(result, dict):
            for v in result.values():
                if hasattr(v, 'block_until_ready'):
                    v.block_until_ready()
        t1 = time.perf_counter()

        elapsed = (t1 - t0) * 1000  # Convert to ms
        print(f"{name:50s}: {elapsed:10.2f} ms")
        return result, elapsed

    total_time = 0.0

    # 1. Background evolution
    print("\n[1/15] BACKGROUND EVOLUTION")
    print("-" * 80)
    param = param_dict.copy()
    param, t = time_step(
        evolve_background,
        "evolve_background",
        param=param,
        thermo_module='RECFAST',
        num_thermo=1024
    )
    total_time += t

    # 2. Perturbation evolution (MOST EXPENSIVE)
    print("\n[2/15] PERTURBATION EVOLUTION")
    print("-" * 80)
    aexp_out = jnp.concatenate([
        jnp.geomspace(3e-4, 5e-3, 256, endpoint=False),
        jnp.geomspace(5e-3, 1.0, 64)
    ])

    (yout, kmodes, param), t = time_step(
        evolve_perturbations_batched,
        "evolve_perturbations_batched (128 k-modes)",
        param=param,
        kmin=1e-4,
        kmax=1.0,
        num_k=128,
        aexp_out=aexp_out,
        rtol=1e-4,
        atol=1e-4,
        return_full=True,
        dologk=True,
    )
    total_time += t

    # 3. Time derivatives
    print("\n[3/15] TIME DERIVATIVES")
    print("-" * 80)
    tau = param['tau_out']
    yprime, t = time_step(
        compute_time_derivatives,
        "compute_time_derivatives",
        yout, tau, kmodes, param
    )
    total_time += t

    # Extract parameters
    lmaxg = param['lmaxg']
    lmaxgp = param['lmaxgp']
    lmaxr = param['lmaxr']
    nqmax = param['nqmax']

    # 4. Extract perturbations
    print("\n[4/15] EXTRACT PERTURBATIONS")
    print("-" * 80)
    perturbations, t = time_step(
        extract_perturbations,
        "extract_perturbations",
        yout, yprime, lmaxg, lmaxgp, lmaxr
    )
    total_time += t

    # 5. Compute background quantities
    print("\n[5/15] BACKGROUND QUANTITIES")
    print("-" * 80)
    background_quantities, t = time_step(
        compute_background_quantities,
        "compute_background_quantities",
        aexp_out, param
    )
    total_time += t

    # 6. Visibility functions
    print("\n[6/15] VISIBILITY FUNCTIONS")
    print("-" * 80)
    tau = param['tau_of_a_spline'].evaluate(aexp_out)
    visibility_functions, t = time_step(
        compute_visibility_functions,
        "compute_visibility_functions",
        tau, param
    )
    total_time += t

    # 7. Neutrino perturbations
    print("\n[7/15] NEUTRINO PERTURBATIONS")
    print("-" * 80)
    neutrinos, t = time_step(
        compute_neutrino_perturbations,
        "compute_neutrino_perturbations",
        yout, yprime, aexp_out, param, nqmax, perturbations['iq0']
    )
    total_time += t

    # 8. Metric perturbations
    print("\n[8/15] METRIC PERTURBATIONS")
    print("-" * 80)
    metric, t = time_step(
        compute_metric_perturbations,
        "compute_metric_perturbations",
        perturbations, neutrinos, background_quantities, param, kmodes, aexp_out
    )
    total_time += t

    # 9. Polarization terms
    print("\n[9/15] POLARIZATION TERMS")
    print("-" * 80)
    polarization_terms, t = time_step(
        compute_polarization_terms,
        "compute_polarization_terms",
        perturbations, metric, visibility_functions, yout, yprime, kmodes, lmaxg, lmaxgp
    )
    total_time += t

    # 10-13. Individual source terms
    print("\n[10/15] SOURCE TERM: ISW")
    print("-" * 80)
    source_isw, t = time_step(
        compute_source_term_isw,
        "compute_source_term_isw",
        metric, visibility_functions
    )
    total_time += t

    print("\n[11/15] SOURCE TERM: SACHS-WOLFE")
    print("-" * 80)
    source_sw, t = time_step(
        compute_source_term_sachs_wolfe,
        "compute_source_term_sachs_wolfe",
        perturbations, metric, visibility_functions, polarization_terms, kmodes
    )
    total_time += t

    print("\n[12/15] SOURCE TERM: DOPPLER")
    print("-" * 80)
    source_doppler, t = time_step(
        compute_source_term_doppler,
        "compute_source_term_doppler",
        perturbations, metric, visibility_functions, polarization_terms, kmodes
    )
    total_time += t

    print("\n[13/15] SOURCE TERM: POLARIZATION")
    print("-" * 80)
    source_pol, t = time_step(
        compute_source_term_polarization,
        "compute_source_term_polarization",
        visibility_functions, polarization_terms, kmodes
    )
    total_time += t

    # 14. Full source function (combines all source terms)
    print("\n[14/15] FULL SOURCE FUNCTION")
    print("-" * 80)
    source_results, t = time_step(
        compute_source_function,
        "compute_source_function",
        perturbations, metric, visibility_functions, yout, yprime, kmodes, lmaxg, lmaxgp
    )
    total_time += t
    S = source_results['S']

    # 15. Line-of-sight integration (SECOND MOST EXPENSIVE)
    print("\n[15/15] LINE-OF-SIGHT INTEGRATION")
    print("-" * 80)
    tau0 = param['tau_of_a_spline'].evaluate(1.0)

    # Test different configurations
    configs = [
        {"ellmax": 50, "nk_fine": 128, "chunk_size": 32, "k_chunk_size": 32},
        {"ellmax": 100, "nk_fine": 256, "chunk_size": 32, "k_chunk_size": 32},
        {"ellmax": 100, "nk_fine": 512, "chunk_size": 64, "k_chunk_size": 64},
    ]

    los_times = []
    for config in configs:
        def compute_theta():
            return compute_theta_ell(
                ellmax=config['ellmax'],
                kmodes=kmodes,
                tau=tau,
                S=S,
                tau0=tau0,
                nk_fine=config['nk_fine'],
                chunk_size=config['chunk_size'],
                k_chunk_size=config['k_chunk_size']
            )

        result, t = time_step(
            compute_theta,
            f"  compute_theta_ell (ellmax={config['ellmax']}, nk_fine={config['nk_fine']})",
        )
        los_times.append(t)

    # Use the middle config for subsequent steps
    theta_ell, kmodes_fine = compute_theta_ell(
        ellmax=100,
        kmodes=kmodes,
        tau=tau,
        S=S,
        tau0=tau0,
        nk_fine=256,
        chunk_size=32,
        k_chunk_size=32
    )

    # Angular power spectrum
    print("\n" + "-" * 80)
    Cell, t = time_step(
        compute_Cell,
        "compute_Cell",
        theta_ell, kmodes_fine, param['n_s'], param['k_p']
    )
    total_time += t

    # Temperature power spectrum
    (ell, Dell), t = time_step(
        compute_Dell,
        "compute_Dell",
        Cell, param['A_s'], param['Tcmb'], ellmax=100
    )
    total_time += t

    # Summary
    print("\n" + "="*80)
    print("PIPELINE TIMING SUMMARY")
    print("="*80)
    print(f"Total pipeline time (excl. LOI configs): {total_time:10.2f} ms ({total_time/1000:.2f} s)")
    print(f"Line-of-sight integration time range:     {min(los_times):10.2f} - {max(los_times):10.2f} ms")
    print("\nPRIMARY BOTTLENECKS:")
    print("  1. evolve_perturbations_batched - Solves coupled ODEs for all k-modes")
    print("  2. compute_theta_ell - Line-of-sight integration (scales with ellmax)")
    print("\nINTERPRETATION:")
    print("  - Functions taking < 10 ms are negligible")
    print("  - Functions taking 10-100 ms may benefit from optimization")
    print("  - Functions taking > 100 ms are primary bottlenecks")
    print("  - For production CMB spectra (ellmax=2500), compute_theta_ell")
    print("    will take significantly longer than shown here")
    print("="*80 + "\n")

    # Verify output
    assert len(ell) == 99  # ell starts from 2, ellmax=100
    assert len(Dell) == 99


def test_benchmark_full_pipeline_end_to_end(benchmark):
    """Benchmark the complete end-to-end CMB spectrum computation.

    This tests the full compute_Cell_spectrum_from_cosmo_params function
    as a black box, measuring total wall-clock time.
    """
    jax.config.update("jax_enable_x64", True)
    jax.config.update('jax_platform_name', 'gpu')

    param_dict = {
        'Omegam': 0.3099,
        'Omegab': 0.0488911,
        'H0': 67.742,
        'n_s': 0.96822,
        'A_s': 2.1064e-09,
        'mnu': 0.06,
        'Tcmb': 2.7255,
        'YHe': 0.248,
        'Neff': 2.046,
        'Nmnu': 1,
        'w_DE_0': -0.99,
        'w_DE_a': 0.0,
        'cs2_DE': 1.0,
        'Omegak': 0.0,
        'k_p': 0.05,
    }

    # Warmup - includes JIT compilation time
    print("\nWarming up (includes JIT compilation)...")
    compute_Cell_spectrum_from_cosmo_params(
        param_dict,
        ellmax=50,  # Small for fast warmup
        nmodes=64,
        kmin=1e-4,
        kmax=1.0
    )

    # Benchmark
    print("Benchmarking end-to-end pipeline...")
    Cell, param = benchmark(
        compute_Cell_spectrum_from_cosmo_params,
        param_dict,
        ellmax=50,
        nmodes=64,
        kmin=1e-4,
        kmax=1.0
    )

    # Verify output
    assert Cell.shape == (51,)  # ellmax + 1
    assert 'tau_of_a_spline' in param
