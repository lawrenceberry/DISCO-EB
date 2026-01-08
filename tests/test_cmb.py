"""
CMB Spectrum Testing Suite with CAMB Benchmarks

This module provides comprehensive testing for DISCO-EB's CMB spectrum computation
by comparing against CAMB benchmarks across multiple cosmologies. It tracks both
accuracy and performance to detect regressions.

CAMB benchmarks are automatically generated on first test run if they don't exist.
The camb_benchmarks fixture handles this transparently.

Usage:
    # Run accuracy tests (auto-generates benchmarks if needed, establishes baseline on first run)
    pytest tests/test_cmb.py::TestCMBSpectrum::test_cmb_vs_camb_accuracy -v

    # Run performance tests
    pytest tests/test_cmb.py::TestCMBSpectrum::test_cmb_performance -v

    # Run all CMB tests
    pytest tests/test_cmb.py -v
"""

import pytest
import jax.numpy as jnp
import numpy as np
import os
import json
import pandas as pd
from datetime import datetime
from multiprocessing import Pool

import camb
from discoeb.cmb import compute_Cell_spectrum_from_cosmo_params, compute_Dell


# ==============================================================================
# Standard Cosmologies for Testing
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


# ==============================================================================
# Helper Functions for CAMB
# ==============================================================================

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


def load_accuracy_baseline(filepath="tests/resources/cmb_accuracy_baseline.json"):
    """Load baseline accuracy metrics from JSON file.

    Parameters
    ----------
    filepath : str
        Path to baseline JSON file

    Returns
    -------
    dict
        Baseline accuracy metrics by cosmology name
    """
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return {}


def save_accuracy_baseline(baseline, filepath="tests/resources/cmb_accuracy_baseline.json"):
    """Save accuracy metrics as new baseline.

    Parameters
    ----------
    baseline : dict
        Accuracy metrics by cosmology name
    filepath : str
        Path to save JSON file
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(baseline, f, indent=2)


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


def load_performance_baseline(filepath="tests/resources/cmb_performance_baseline.json"):
    """Load baseline performance metrics from JSON file.

    Parameters
    ----------
    filepath : str
        Path to baseline JSON file

    Returns
    -------
    dict
        Baseline performance metrics by cosmology name
    """
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return {}


def save_performance_baseline(baseline, filepath="tests/resources/cmb_performance_baseline.json"):
    """Save performance metrics as new baseline.

    Parameters
    ----------
    baseline : dict
        Performance metrics by cosmology name
    filepath : str
        Path to save JSON file
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(baseline, f, indent=2)


# ==============================================================================
# Test Classes
# ==============================================================================

class TestCMBSpectrum:
    """Tests for CMB spectrum computation and accuracy."""

    @pytest.mark.parametrize("cosmology_name", list(STANDARD_COSMOLOGIES.keys()))
    def test_cmb_vs_camb_accuracy(self, cosmology_name, camb_benchmarks):
        """Test CMB spectrum accuracy against CAMB benchmarks with regression check.

        On first run, this establishes a baseline. On subsequent runs, it checks
        that accuracy hasn't worsened relative to the baseline.

        CAMB benchmarks are automatically generated by the fixture if missing.
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
        Cell_disco, param = compute_Cell_spectrum_from_cosmo_params(cosmo_params, ellmax=2500)
        ell_disco, Dell_disco = compute_Dell(Cell_disco, A_s=param["A_s"], Tcmb=param["Tcmb"])

        # Get CAMB benchmark (guaranteed to exist by fixture)
        ell_camb = camb_benchmarks[cosmology_name]['ell']
        D_ell_camb = camb_benchmarks[cosmology_name]['D_ell']

        # Interpolate CAMB to DISCO-EB ell range for comparison
        D_ell_camb_interp = jnp.interp(ell_disco, ell_camb, D_ell_camb)

        # Compute accuracy metrics
        metrics = compute_accuracy_metrics(Dell_disco, D_ell_camb_interp)

        # Load baseline accuracy
        baseline_file = "tests/resources/cmb_accuracy_baseline.json"
        baseline = load_accuracy_baseline(baseline_file)

        # First run: establish baseline
        if cosmology_name not in baseline:
            baseline[cosmology_name] = metrics
            save_accuracy_baseline(baseline, baseline_file)
            print(f"\n✓ Baseline established for {cosmology_name}:")
            print(f"  Mean rel. error: {metrics['mean_relative_error']:.4f}")
            print(f"  Max rel. error: {metrics['max_relative_error']:.4f}")
            print(f"  RMSE: {metrics['rmse']:.2f} μK²")

        # Subsequent runs: check for regression
        else:
            baseline_metrics = baseline[cosmology_name]

            # Check that accuracy hasn't worsened (allow 10% tolerance)
            assert metrics['mean_relative_error'] <= baseline_metrics['mean_relative_error'] * 1.1, \
                f"Mean relative error worsened: {metrics['mean_relative_error']:.4f} > " \
                f"{baseline_metrics['mean_relative_error']:.4f} (baseline +10%)"

            assert metrics['max_relative_error'] <= baseline_metrics['max_relative_error'] * 1.1, \
                f"Max relative error worsened: {metrics['max_relative_error']:.4f} > " \
                f"{baseline_metrics['max_relative_error']:.4f} (baseline +10%)"

            print(f"\n✓ Accuracy check passed for {cosmology_name}")
            print(f"  Mean rel. error: {metrics['mean_relative_error']:.4f} (baseline: {baseline_metrics['mean_relative_error']:.4f})")
            print(f"  Max rel. error: {metrics['max_relative_error']:.4f} (baseline: {baseline_metrics['max_relative_error']:.4f})")

        # Standard accuracy assertion (should pass for all cosmologies)
        assert metrics['mean_relative_error'] < 0.2, \
            f"Mean relative error {metrics['mean_relative_error']:.4f} exceeds 20% threshold"


    @pytest.mark.parametrize("cosmology_name", list(STANDARD_COSMOLOGIES.keys()))
    def test_benchmark_cl_spectrum(self, cosmology_name, benchmark):
        """Tests CMB Cl spectrum computation performance.

        On first run, this establishes a performance baseline. On subsequent runs,
        it checks that performance hasn't degraded more than 20%.
        """
        import jax
        jax.config.update("jax_enable_x64", True)
        jax.config.update('jax_platform_name', 'gpu')

        # Check JAX backend
        print(f"JAX backend: {jax.default_backend()}")
        print(f"Devices: {jax.devices()}")

        # Clear JAX compilation cache
        jax.clear_caches()
        
        cosmo_params = STANDARD_COSMOLOGIES[cosmology_name]

        # Benchmark the computation, making sure the function is compiled first
        benchmark.pedantic(compute_Cell_spectrum_from_cosmo_params, kwargs=dict(param_dict=cosmo_params, ellmax=2500), rounds=1, warmup_rounds=1)

        # Load performance baseline
        baseline_file = "tests/resources/cmb_performance_baseline.json"
        baseline = load_performance_baseline(baseline_file)

        # Get timing from benchmark
        timing = benchmark.stats['mean']

        # First run: establish baseline
        if cosmology_name not in baseline:
            baseline[cosmology_name] = {
                "mean_time_seconds": timing,
                "timestamp": datetime.now().isoformat()
            }
            save_performance_baseline(baseline, baseline_file)
            print(f"\n✓ Performance baseline established for {cosmology_name}: {timing:.3f}s")

        # Subsequent runs: check for regression
        else:
            baseline_time = baseline[cosmology_name]['mean_time_seconds']

            # Allow 20% slowdown tolerance
            assert timing <= baseline_time * 1.2, \
                f"Performance regression for {cosmology_name}: {timing:.3f}s > " \
                f"{baseline_time:.3f}s (baseline +20%)"

            print(f"\n✓ Performance check passed for {cosmology_name}")
            print(f"  Current: {timing:.3f}s (baseline: {baseline_time:.3f}s)")
