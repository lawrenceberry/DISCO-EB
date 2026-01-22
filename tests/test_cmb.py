"""
CMB Spectrum Testing Suite

This module provides comprehensive testing for DISCO-EB's CMB spectrum computation
by comparing against CAMB benchmarks across multiple cosmologies. It uses pytest-regression
to automatically track accuracy metrics and pytest-benchmark for timings. CAMB benchmarks are
automatically generated on first test run if they don't exist.

Usage:
    # Run all CMB tests
    pytest tests/test_cmb.py -v

    # Run all timing benchmarks
    pytest tests/test_cmb.py -v --benchmark-only

    # Run specific timing benchmark test
    pytest tests/test_cmb.py::test_benchmark_compute_theta_ell -v

    # Update timing baselines after performance improvements
    pytest tests/test_cmb.py --benchmark-autosave

    # Compare timings against currently stored baseline timings
    pytest tests/test_cmb.py --benchmark-compare --benchmark-compare-fail=mean:10%

    # Run accuracy tests (auto-generates CAMB benchmarks and baseline discrepancies on first run)
    pytest tests/test_cmb.py::test_cmb_vs_camb_accuracy -v

    # Update accuracy baseline after accuracy improvements
    pytest tests/test_cmb.py::test_cmb_vs_camb_accuracy --force-regen

    # Force regeneration of cached cmb_test_data (by deleting cache file)
    rm tests/resources/cmb_test_data_cache.pkl && pytest tests/test_cmb.py
"""
import os
import pickle
from datetime import datetime
from multiprocessing import Pool

import camb
import pytest
import jax
import jax.numpy as jnp
import numpy as np
import time
import pandas as pd

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

STANDARD_COSMOLOGIES = {
    "DISCO-Notebook": {
        'Omegam': 0.3099,            # Total matter density
        'Omegab': 0.0488911,         # Baryon density
        'H0': 67.742,            # Hubble constant (km/s/Mpc)
        'n_s': 0.96822,           # Spectral index
        'A_s': 2.1064e-09,        # Primordial amplitude
        'mnu': 0.06,              # Neutrino mass (eV)
        'Tcmb': 2.7255,            # CMB temperature (K)
        'YHe': 0.248,             # Helium mass fraction
        'Neff': 2.046,             # Effective relativistic species
        'Nmnu': 1     ,            # Massive neutrino species
        'w_DE_0': -0.99,             # Dark energy EOS today
        'w_DE_a': 0.0,               # Dark energy EOS derivative
        'cs2_DE': 1.0,               # Dark energy sound speed squared
        'Omegak': 0.0,               # Curvature
        'k_p': 0.05,              # Pivot scale (1/Mpc)
    },
}

@pytest.fixture(scope="module")
def cmb_test_data():
    """Generate test data for CMB benchmarks with file caching.

    This fixture runs once per module to generate all the intermediate
    data needed for benchmarking individual functions. The computed data
    is cached to disk for faster subsequent runs.

    Cache location: tests/resources/cmb_test_data_cache.pkl

    To force regeneration, delete the cache file:
        rm tests/resources/cmb_test_data_cache.pkl
    """
    # Enable 64-bit precision
    jax.config.update("jax_enable_x64", True)
    jax.config.update('jax_platform_name', 'gpu')

    # Clear compilation cache
    jax.clear_caches()

    # Cache file location
    cache_dir = "tests/resources"
    cache_file = os.path.join(cache_dir, "cmb_test_data_cache.pkl")

    # Try to load from cache if it exists
    if os.path.exists(cache_file):
        print("\n" + "="*70)
        print("Loading CMB test data from cache...")
        print(f"  Cache file: {cache_file}")
        print("="*70)

        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)

            print("✓ CMB test data loaded from cache!")
            print("  (Delete cache file to force regeneration)")
            print("="*70 + "\n")

            return cached_data

        except Exception as e:
            print(f"⚠ Failed to load cache ({e}), regenerating...")

    # Generate fresh data
    print("\n" + "="*70)
    print("Setting up CMB test data (this may take a minute)...")
    print("="*70)

    # Standard cosmology parameters
    param = STANDARD_COSMOLOGIES["DISCO-Notebook"]

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

    # Prepare data dictionary
    data = {
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

    # Save to cache
    print(f"  Saving to cache: {cache_file}")
    os.makedirs(cache_dir, exist_ok=True)

    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("✓ Cache saved successfully!")
    except Exception as e:
        print(f"⚠ Failed to save cache: {e}")

    print("="*70 + "\n")

    return data


# ==============================================================================
# Benchmark Tests
# ==============================================================================

def test_benchmark_evolve_background(benchmark, num_regression):
    """Benchmark evolve_background function.

    This is the first step in the CMB pipeline that computes the background
    cosmology evolution and thermal history.
    """
    # Standard cosmology parameters
    param = STANDARD_COSMOLOGIES["DISCO-Notebook"]

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

    # Numeric regression check - test full arrays and key scalar values
    test_points = jnp.array([0.001, 0.01, 0.1, 0.5, 1.0])
    num_regression.check({
        "H0": result["H0"],
        "Omegam": result["Omegam"],
        "grhom": result["grhom"],
        "tau_at_test_points": result['tau_of_a_spline'].evaluate(test_points),
        "xe_at_test_points": jnp.exp(result['xe_of_loga_spline'].evaluate(jnp.log(test_points))),
    }, default_tolerance=dict(atol=0, rtol=0.2))


def test_benchmark_compute_background_quantities(benchmark, cmb_test_data, num_regression):
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

    # Numeric regression check - check full arrays
    num_regression.check({
        "rhonu": result['rhonu'],
        "rho_Q": result['rho_Q'],
        "w_Q": result['w_Q'],
    }, default_tolerance=dict(atol=0, rtol=0.2))


def test_benchmark_evolve_perturbations_batched(benchmark, num_regression):
    """Benchmark evolve_perturbations_batched function.

    This is the most computationally intensive step in the CMB pipeline,
    solving the coupled ODE system for all k-modes.
    """
    jax.config.update("jax_enable_x64", True)
    jax.config.update('jax_platform_name', 'gpu')

    # Standard cosmology parameters with background evolution
    param = STANDARD_COSMOLOGIES["DISCO-Notebook"]

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

    # Numeric regression check - check representative k-modes (flattened for 1D requirement)
    # Select first, middle, and last k-modes to keep regression file manageable
    num_regression.check({
        "yout_k0_flat": yout[0, :, :].flatten(),
        "yout_k31_flat": yout[31, :, :].flatten(),
        "yout_k63_flat": yout[63, :, :].flatten(),
        "kmodes": kmodes,  # Full kmodes array
    }, default_tolerance=dict(atol=0, rtol=0.2))


def test_benchmark_compute_time_derivatives(benchmark, cmb_test_data, num_regression):
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

    # Numeric regression check - sample representative k-modes (flattened for 1D requirement)
    num_regression.check({
        "yprime_k0_flat": yprime[0, :, :].flatten(),
        "yprime_k63_flat": yprime[63, :, :].flatten(),
        "yprime_k127_flat": yprime[127, :, :].flatten(),
    }, default_tolerance=dict(atol=0, rtol=0.2))


def test_benchmark_extract_perturbations(benchmark, cmb_test_data, num_regression):
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

    # Numeric regression check - check full arrays for key quantities at sample k-modes
    # Store each k-mode separately since num_regression only supports 1D arrays
    num_regression.check({
        "deltag_k0": result['deltag'][0, :],
        "deltag_k63": result['deltag'][63, :],
        "deltag_k127": result['deltag'][127, :],
        "deltac_k0": result['deltac'][0, :],
        "deltac_k63": result['deltac'][63, :],
        "deltac_k127": result['deltac'][127, :],
        "thetag_k0": result['thetag'][0, :],
        "thetag_k63": result['thetag'][63, :],
        "deltab_k0": result['deltab'][0, :],
        "deltar_k0": result['deltar'][0, :],
    }, default_tolerance=dict(atol=0, rtol=0.2))


def test_benchmark_compute_visibility_functions(benchmark, cmb_test_data, num_regression):
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

    # Numeric regression check - check full arrays
    num_regression.check({
        "gvis": result['gvis'],
        "optical_depth": result['optical_depth'],
        "gvisprime": result['gvisprime'],
    }, default_tolerance=dict(atol=0, rtol=0.2))


def test_benchmark_compute_neutrino_perturbations(benchmark, cmb_test_data, num_regression):
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

    # Numeric regression check - sample k-modes (separate 1D arrays)
    num_regression.check({
        "drhonu_k0": result['drhonu'][0, :],
        "drhonu_k63": result['drhonu'][63, :],
        "drhonu_k127": result['drhonu'][127, :],
        "dpnu_k0": result['dpnu'][0, :],
    }, default_tolerance=dict(atol=0, rtol=0.2))


def test_benchmark_compute_metric_perturbations(benchmark, cmb_test_data, num_regression):
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

    # Numeric regression check - sample k-modes (separate 1D arrays)
    num_regression.check({
        "alpha_k0": result['alpha'][0, :],
        "alpha_k63": result['alpha'][63, :],
        "alpha_k127": result['alpha'][127, :],
        "alphaprime_k0": result['alphaprime'][0, :],
    }, default_tolerance=dict(atol=0, rtol=0.2))


def test_benchmark_compute_polarization_terms(benchmark, cmb_test_data, num_regression):
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

    # Numeric regression check - sample k-modes (separate 1D arrays)
    num_regression.check({
        "polarization_term_k0": result['polarization_term'][0, :],
        "polarization_term_k63": result['polarization_term'][63, :],
        "polarization_term_k127": result['polarization_term'][127, :],
    }, default_tolerance=dict(atol=0, rtol=0.2))


def test_benchmark_compute_source_term_isw(benchmark, cmb_test_data, num_regression):
    """Benchmark compute_source_term_isw function."""
    data = cmb_test_data

    result = benchmark(
        compute_source_term_isw,
        data['metric'],
        data['visibility_functions']
    )

    # Verify output shape
    assert result.shape[0] == data['kmodes'].shape[0]

    # Numeric regression check - sample k-modes (separate 1D arrays)
    num_regression.check({
        "source_isw_k0": result[0, :],
        "source_isw_k63": result[63, :],
        "source_isw_k127": result[127, :],
    }, default_tolerance=dict(atol=0, rtol=0.2))


def test_benchmark_compute_source_term_sachs_wolfe(benchmark, cmb_test_data, num_regression):
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

    # Numeric regression check - sample k-modes (separate 1D arrays)
    num_regression.check({
        "source_sw_k0": result[0, :],
        "source_sw_k63": result[63, :],
        "source_sw_k127": result[127, :],
    }, default_tolerance=dict(atol=0, rtol=0.2))


def test_benchmark_compute_source_term_doppler(benchmark, cmb_test_data, num_regression):
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

    # Numeric regression check - sample k-modes (separate 1D arrays)
    num_regression.check({
        "source_doppler_k0": result[0, :],
        "source_doppler_k63": result[63, :],
        "source_doppler_k127": result[127, :],
    }, default_tolerance=dict(atol=0, rtol=0.2))


def test_benchmark_compute_source_term_polarization(benchmark, cmb_test_data, num_regression):
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

    # Numeric regression check - sample k-modes (separate 1D arrays)
    num_regression.check({
        "source_polarization_k0": result[0, :],
        "source_polarization_k63": result[63, :],
        "source_polarization_k127": result[127, :],
    }, default_tolerance=dict(atol=0, rtol=0.2))


def test_benchmark_compute_source_function(benchmark, cmb_test_data, num_regression):
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

    # Numeric regression check - sample k-modes for all source terms (separate 1D arrays)
    num_regression.check({
        "S_k0": result['S'][0, :],
        "S_k63": result['S'][63, :],
        "S_k127": result['S'][127, :],
        "S1_k0": result['S1'][0, :],
        "S2_k0": result['S2'][0, :],
        "S3_k0": result['S3'][0, :],
        "S4_k0": result['S4'][0, :],
    }, default_tolerance=dict(atol=0, rtol=0.2))


def test_benchmark_compute_theta_ell(benchmark, cmb_test_data, num_regression):
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

    # Numeric regression check - sample k-modes with full ell range (separate 1D arrays)
    num_regression.check({
        "theta_ell_k_min": theta_ell[0, :],
        "theta_ell_k_max": theta_ell[255, :],
    }, default_tolerance=dict(atol=0, rtol=0.2))


def test_benchmark_compute_Cell(benchmark, cmb_test_data, num_regression):
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

    # Numeric regression check - check full Cell array
    num_regression.check({
        "Cell": Cell,
    }, default_tolerance=dict(atol=0, rtol=0.2))


def test_benchmark_compute_Dell(benchmark, cmb_test_data, num_regression):
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

    # Numeric regression check - check full Dell array
    num_regression.check({
        "ell": ell,
        "Dell": Dell,
    }, default_tolerance=dict(atol=0, rtol=0.2))


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
    param_dict = STANDARD_COSMOLOGIES["DISCO-Notebook"]

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
        yout, yprime, aexp_out, param, nqmax, perturbations['iq0'] # type: ignore
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
    S = source_results['S'] # type: ignore

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


@pytest.mark.parametrize("cosmology_name", list(STANDARD_COSMOLOGIES.keys()))
def test_benchmark_full_pipeline_end_to_end(benchmark, cosmology_name):
    """Benchmark the complete end-to-end CMB spectrum computation.

    This tests the full compute_Cell_spectrum_from_cosmo_params function
    as a black box, measuring total wall-clock time.
    """
    jax.config.update("jax_enable_x64", True)
    jax.config.update('jax_platform_name', 'gpu')

    # Check JAX backend
    print(f"JAX backend: {jax.default_backend()}")
    print(f"Devices: {jax.devices()}")

    # Clear JAX compilation cache
    jax.clear_caches()

    # Get cosmology parameters
    param_dict = STANDARD_COSMOLOGIES[cosmology_name]

    # Benchmark the computation, making sure the function is compiled first
    Cell, param = benchmark.pedantic(
        compute_Cell_spectrum_from_cosmo_params,
        kwargs=dict(param_dict=param_dict, ellmax=50, nmodes=64, kmin=1e-4, kmax=1.0),
        warmup_rounds=1,  # warmup call performs JIT compilation
        rounds=1,
    )

    # Verify output
    assert Cell.shape == (51,)  # ellmax + 1
    assert 'tau_of_a_spline' in param

    print(f"\n✓ Performance: {benchmark.stats['mean']:.3f}s for {cosmology_name}")


#==============================================================================
# Accuracy Tests Against CAMB with Regression Tracking
#==============================================================================

def disco_params_to_camb(param_dict):
    """Convert DISCO-EB parameters to CAMB format.

    Parameters
    ----------
    param_dict : dict
        DISCO-EB parameter dictionary

    Returns
    -------
    camb.CAMBparams
        CAMB parameters object
    """
    H0 = param_dict['H0']
    h = H0 / 100.0
    ombh2 = param_dict['Omegab'] * h**2
    omch2 = (param_dict['Omegam'] - param_dict['Omegab']) * h**2

    pars = camb.CAMBparams()
    pars.set_cosmology(
        H0=H0,
        ombh2=ombh2,
        omch2=omch2,
        omk=param_dict['Omegak'],
        tau=0.0,  # Reionization optical depth
        mnu=param_dict['mnu'],
    )
    pars.InitPower.set_params(
        As=param_dict['A_s'],
        ns=param_dict['n_s']
    )

    # Set accuracy parameters
    pars.set_accuracy(AccuracyBoost=2.0, lAccuracyBoost=2.0)

    return pars


def _compute_single_camb_spectrum(args):
    """Helper function to compute CAMB spectrum for a single cosmology.

    This function is designed to be called by multiprocessing.Pool.map()

    Parameters
    ----------
    args : tuple
        (cosmology_name, param_dict, ellmax)

    Returns
    -------
    tuple
        (cosmology_name, ell_array, D_ell_array)
    """
    cosmology_name, param_dict, ellmax = args

    # Convert to CAMB parameters
    pars = disco_params_to_camb(param_dict)
    pars.set_for_lmax(ellmax, lens_potential_accuracy=2)

    # Compute power spectrum
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='K')

    # Extract temperature power spectrum
    # D_ell = ell*(ell+1)*C_ell/(2*pi) in K^2 from CAMB
    ell_full = np.arange(powers['unlensed_total'].shape[0])
    D_ell_full = powers['unlensed_total'][:, 0]  # Temperature (TT) in K^2

    # Convert from K^2 to μK^2
    D_ell_full = D_ell_full * 1e12

    # Only keep ell >= 2
    mask = ell_full >= 2
    ell = ell_full[mask]
    D_ell = D_ell_full[mask]

    return (cosmology_name, ell, D_ell)


def generate_camb_benchmarks(
    cosmologies=None,
    ellmax=2500,
    output_dir="tests/resources/camb_benchmarks",
    force_regenerate=False,
    n_processes=None
):
    """Generate CAMB benchmark data for standard cosmologies in parallel.

    Parameters
    ----------
    cosmologies : dict, optional
        Dictionary of cosmology name -> parameter dict.
        If None, uses STANDARD_COSMOLOGIES.
    ellmax : int, optional
        Maximum multipole to compute. Default: 2500
    output_dir : str, optional
        Directory to save CSV files. Default: "tests/resources/camb_benchmarks"
    force_regenerate : bool, optional
        If True, regenerate even if files exist. Default: False
    n_processes : int, optional
        Number of parallel processes. If None, uses all available CPUs.

    Returns
    -------
    None
        Saves CSV files to output_dir
    """
    if cosmologies is None:
        cosmologies = STANDARD_COSMOLOGIES

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Check which cosmologies need computation
    tasks = []
    for cosmo_name, cosmo_params in cosmologies.items():
        csv_path = os.path.join(output_dir, f"{cosmo_name}.csv")

        if force_regenerate or not os.path.exists(csv_path):
            tasks.append((cosmo_name, cosmo_params, ellmax))
        else:
            print(f"Skipping {cosmo_name} (already exists)")

    if not tasks:
        print("All CAMB benchmarks already exist. Use force_regenerate=True to regenerate.")
        return

    # Compute CAMB spectra in parallel
    print(f"Generating CAMB benchmarks for {len(tasks)} cosmologies...")

    with Pool(processes=n_processes) as pool:
        results = pool.map(_compute_single_camb_spectrum, tasks)

    # Save results to CSV files
    for cosmo_name, ell, D_ell in results:
        csv_path = os.path.join(output_dir, f"{cosmo_name}.csv")
        df = pd.DataFrame({'ell': ell, 'D_ell': D_ell})
        df.to_csv(csv_path, index=False)
        print(f"✓ Saved {cosmo_name} → {csv_path}")

    print(f"\nGenerated {len(results)} CAMB benchmarks successfully!")


@pytest.fixture(scope="session")
def camb_benchmarks():
    """Load CAMB benchmark data for all standard cosmologies.

    This fixture loads pre-computed CAMB CMB spectra from CSV files in
    tests/resources/camb_benchmarks/. If any benchmarks are missing, they
    will be automatically generated.

    Returns
    -------
    dict
        Dictionary mapping cosmology names to benchmark data:
        {
            'cosmology_name': {
                'ell': array of multipole moments,
                'D_ell': array of D_ell values
            },
            ...
        }
    """
    benchmark_dir = "tests/resources/camb_benchmarks"

    # Create directory if it doesn't exist
    os.makedirs(benchmark_dir, exist_ok=True)

    # Check which benchmarks are missing
    missing_cosmologies = {}
    for cosmo_name in STANDARD_COSMOLOGIES:
        csv_path = os.path.join(benchmark_dir, f"{cosmo_name}.csv")
        if not os.path.exists(csv_path):
            missing_cosmologies[cosmo_name] = STANDARD_COSMOLOGIES[cosmo_name]

    # Generate missing benchmarks
    if missing_cosmologies:
        print(f"\n⚠ Generating {len(missing_cosmologies)} missing CAMB benchmarks...")
        print(f"   Cosmologies: {', '.join(missing_cosmologies.keys())}")
        generate_camb_benchmarks(
            cosmologies=missing_cosmologies,
            ellmax=2500,
            output_dir=benchmark_dir,
            force_regenerate=False,
        )
        print("✓ CAMB benchmarks generated successfully\n")

    # Load all CSV files in the benchmark directory
    benchmarks = {}
    for filename in os.listdir(benchmark_dir):
        if filename.endswith('.csv'):
            cosmology_name = filename.replace('.csv', '')
            csv_path = os.path.join(benchmark_dir, filename)

            try:
                df = pd.read_csv(csv_path)
                benchmarks[cosmology_name] = {
                    'ell': jnp.array(df['ell'].values),
                    'D_ell': jnp.array(df['D_ell'].values)
                }
            except Exception as e:
                print(f"Warning: Failed to load {csv_path}: {e}")

    return benchmarks


def compute_accuracy_metrics(disco_D_ell, camb_D_ell):
    """Compute accuracy metrics between DISCO-EB and CAMB.

    Parameters
    ----------
    disco_D_ell : array_like
        DISCO-EB D_ell values
    camb_D_ell : array_like
        CAMB D_ell values (same ell range)

    Returns
    -------
    dict
        Accuracy metrics with keys: mean_relative_error, max_relative_error, rmse, timestamp
    """
    # Convert to numpy for safer computation
    disco_D_ell = np.asarray(disco_D_ell)
    camb_D_ell = np.asarray(camb_D_ell)

    # Compute relative error
    rel_error = np.abs(disco_D_ell - camb_D_ell) / camb_D_ell

    return {
        "mean_relative_error": float(np.mean(rel_error)),
        "max_relative_error": float(np.max(rel_error)),
        "rmse": float(np.sqrt(np.mean((disco_D_ell - camb_D_ell)**2))),
        "timestamp": datetime.now().isoformat()
    }


@pytest.mark.parametrize("cosmology_name", list(STANDARD_COSMOLOGIES.keys()))
def test_cmb_vs_camb_accuracy(cosmology_name, camb_benchmarks, num_regression):
    """Test CMB spectrum accuracy against CAMB benchmarks with regression check.

    Uses pytest-regression to automatically track accuracy metrics. On first run,
    it establishes a baseline. On subsequent runs, it checks that accuracy hasn't
    worsened relative to the baseline.

    CAMB benchmarks are automatically generated by the fixture if missing.

    To update the baseline after intentional improvements:
        pytest tests/test_cmb.py::TestCMBSpectrum::test_cmb_vs_camb_accuracy --force-regen
    """
    import jax
    jax.config.update("jax_enable_x64", True)
    jax.config.update('jax_platform_name', 'gpu')

    # Clear JAX compilation cache
    jax.clear_caches()

    # Get cosmology parameters
    cosmo_params = STANDARD_COSMOLOGIES[cosmology_name]

    # Compute DISCO-EB spectrum
    print(f"\nComputing DISCO-EB spectrum for {cosmology_name}...")
    Cell_disco, param = compute_Cell_spectrum_from_cosmo_params(cosmo_params, ellmax=50, nmodes=64, kmin=1e-4, kmax=1.0)
    ell_disco, Dell_disco = compute_Dell(Cell_disco, A_s=param["A_s"], Tcmb=param["Tcmb"])

    # Get CAMB benchmark (guaranteed to exist by fixture)
    ell_camb = camb_benchmarks[cosmology_name]['ell']
    D_ell_camb = camb_benchmarks[cosmology_name]['D_ell']

    # Interpolate CAMB to DISCO-EB ell range for comparison
    D_ell_camb_interp = jnp.interp(ell_disco, ell_camb, D_ell_camb)

    # Compute accuracy metrics
    metrics = compute_accuracy_metrics(Dell_disco, D_ell_camb_interp)

    # Use pytest-regression to automatically compare against baseline
    # This will create a baseline on first run and check regression on subsequent runs
    num_regression.check({
        "mean_relative_error": metrics['mean_relative_error'],
        "max_relative_error": metrics['max_relative_error'],
        "rmse": metrics['rmse'],
    }, default_tolerance=dict(atol=0, rtol=0.1))  # Allow 10% relative tolerance

    # Print current metrics
    print(f"\n✓ Accuracy metrics for {cosmology_name}:")
    print(f"  Mean rel. error: {metrics['mean_relative_error']:.4f}")
    print(f"  Max rel. error: {metrics['max_relative_error']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.2f} μK²")

    # Standard accuracy assertion (should pass for all cosmologies)
    assert metrics['mean_relative_error'] < 0.5, \
        f"Mean relative error {metrics['mean_relative_error']:.4f} exceeds 50% threshold"
