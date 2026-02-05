#!/usr/bin/env python
"""
Find CAMB AccuracyBoost threshold for 0.7% mean absolute relative error.

This script runs CAMB at increasing AccuracyBoost values (0.5, 1.0, 1.5, ...)
until the mean absolute relative error compared to the reference benchmark
reaches 0.7% or below.

Note: AccuracyBoost values below ~0.5 can cause CAMB to crash/segfault
(especially with multiple threads), so this script starts from 0.5 by default.

Reference: tests/resources/camb_benchmarks/DISCO-Notebook.csv

Usage:
    python scripts/find_camb_accuracy_threshold.py
    python scripts/find_camb_accuracy_threshold.py --start 1.0 --step 0.5
"""

import argparse
import os
import sys

# Parse threads argument early, before importing CAMB
# OMP_NUM_THREADS must be set before CAMB is imported
def _get_threads_from_args():
    for i, arg in enumerate(sys.argv):
        if arg == '--threads' and i + 1 < len(sys.argv):
            return int(sys.argv[i + 1])
        if arg.startswith('--threads='):
            return int(arg.split('=')[1])
    return 1  # default

os.environ['OMP_NUM_THREADS'] = str(_get_threads_from_args())

import time
import numpy as np
import pandas as pd
import camb


# Cosmology parameters (matching DISCO-Notebook)
COSMOLOGY = {
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
    'Omegak': 0.0,
    'k_p': 0.05,
}


def load_reference_data(csv_path="tests/resources/camb_benchmarks/DISCO-Notebook.csv"):
    """Load reference CAMB benchmark data."""
    df = pd.read_csv(csv_path)
    return df['ell'].values, df['D_ell'].values


def create_camb_params(params, ellmax, accuracy_boost, lsample_boost=50.0):
    """Create CAMB parameters with specified accuracy settings.

    Parameters
    ----------
    params : dict
        Cosmological parameters
    ellmax : int
        Maximum multipole
    accuracy_boost : float
        CAMB AccuracyBoost parameter
    lsample_boost : float
        CAMB lSampleBoost parameter (>=50 computes all ells)

    Returns
    -------
    camb.CAMBparams
        Configured CAMB parameters
    """
    cpars = camb.CAMBparams()

    # Cosmological parameters
    h = params['H0'] / 100.0
    cpars.set_cosmology(
        H0=params['H0'],
        ombh2=params['Omegab'] * h**2,
        omch2=(params['Omegam'] - params['Omegab']) * h**2,
        omk=params['Omegak'],
        tau=0.0,
        mnu=params['mnu'],
        num_massive_neutrinos=params['Nmnu'],
        nnu=params['Neff'] + params['Nmnu'],
        YHe=params['YHe'],
        TCMB=params['Tcmb'],
    )

    # Dark energy
    cpars.DarkEnergy.w = params['w_DE_0']
    cpars.DarkEnergy.wa = params['w_DE_a']

    # Initial power spectrum
    cpars.InitPower.As = params['A_s']
    cpars.InitPower.ns = params['n_s']
    cpars.InitPower.pivot_scalar = params['k_p']

    # Set for lmax and accuracy
    cpars.set_for_lmax(ellmax)
    cpars.set_accuracy(AccuracyBoost=accuracy_boost, lSampleBoost=lsample_boost)

    return cpars


def run_camb(cpars):
    """Run CAMB and return D_ell spectrum.

    Returns
    -------
    tuple
        (ell, D_ell, elapsed_time)
    """
    t0 = time.perf_counter()
    results = camb.get_results(cpars)
    powers = results.get_cmb_power_spectra(cpars, CMB_unit='muK')
    elapsed = time.perf_counter() - t0

    # Extract TT spectrum (D_ell in μK²)
    ell_full = np.arange(powers['unlensed_total'].shape[0])
    D_ell_full = powers['unlensed_total'][:, 0]

    # Only keep ell >= 2
    mask = ell_full >= 2
    return ell_full[mask], D_ell_full[mask], elapsed


def compute_mean_abs_relative_error(D_ell_test, D_ell_ref, ell_test, ell_ref):
    """Compute mean absolute relative error.

    Interpolates test data to reference ell values if needed.
    """
    # Interpolate test data to reference ell grid
    D_ell_test_interp = np.interp(ell_ref, ell_test, D_ell_test)

    # Compute relative error
    rel_error = np.abs(D_ell_test_interp - D_ell_ref) / D_ell_ref

    return np.mean(rel_error)


def main():
    parser = argparse.ArgumentParser(
        description='Find CAMB AccuracyBoost threshold for target error'
    )
    parser.add_argument(
        '--start', type=float, default=0.5,
        help='Starting AccuracyBoost value (default: 0.5, lower values may crash especially with multiple threads)'
    )
    parser.add_argument(
        '--step', type=float, default=0.1,
        help='AccuracyBoost increment step (default: 0.1)'
    )
    parser.add_argument(
        '--max', type=float, default=10.0,
        help='Maximum AccuracyBoost to try (default: 10.0)'
    )
    parser.add_argument(
        '--target', type=float, default=0.7,
        help='Target mean absolute relative error in percent (default: 0.7)'
    )
    parser.add_argument(
        '--threads', type=int, default=1,
        help='Number of OMP threads (default: 1)'
    )
    args = parser.parse_args()

    # Load reference data
    print("=" * 80)
    print("CAMB ACCURACY THRESHOLD FINDER")
    print("=" * 80)
    print(f"\nFinding AccuracyBoost value that achieves ≤{args.target}% mean absolute relative error")
    print("Reference: tests/resources/camb_benchmarks/DISCO-Notebook.csv")
    print("Settings: lSampleBoost=50 (computes all ells)")
    print(f"Starting from AccuracyBoost={args.start}, step={args.step}")
    print()

    ell_ref, D_ell_ref = load_reference_data()
    ellmax = int(ell_ref.max())

    print(f"Reference data: ell range [{ell_ref.min():.0f}, {ell_ref.max():.0f}]")
    print(f"Target error: ≤{args.target}%")
    print()

    # Table header
    print("-" * 80)
    print(f"{'AccuracyBoost':>14} | {'Time (s)':>10} | {'Mean Abs Rel Error':>20} | {'Status':>10}")
    print("-" * 80)

    results = []
    accuracy_boost = args.start
    target_error = args.target / 100.0  # Convert from percent to fraction

    while accuracy_boost <= args.max:
        try:
            # Create CAMB parameters
            cpars = create_camb_params(
                COSMOLOGY,
                ellmax=ellmax,
                accuracy_boost=accuracy_boost,
                lsample_boost=50.0
            )

            # Run CAMB
            ell, D_ell, elapsed = run_camb(cpars)

            # Compute error
            error = compute_mean_abs_relative_error(D_ell, D_ell_ref, ell, ell_ref)
            error_pct = error * 100

            # Status
            if error <= target_error:
                status = "✓ TARGET"
            else:
                status = ""

            # Print row
            print(f"{accuracy_boost:>14.1f} | {elapsed:>10.3f} | {error_pct:>19.4f}% | {status:>10}")

            # Store result
            results.append({
                'accuracy_boost': accuracy_boost,
                'time_s': elapsed,
                'mean_abs_rel_error': error,
                'mean_abs_rel_error_pct': error_pct,
            })

            # Check if target reached
            if error <= target_error:
                print("-" * 80)
                print(f"\n✓ Target achieved at AccuracyBoost={accuracy_boost:.1f}")
                print(f"  Mean absolute relative error: {error_pct:.4f}%")
                print(f"  Computation time: {elapsed:.3f}s")
                break

        except Exception as e:
            print(f"{accuracy_boost:>14.1f} | {'ERROR':>10} | {str(e)[:20]:>20} |")
            results.append({
                'accuracy_boost': accuracy_boost,
                'time_s': float('nan'),
                'mean_abs_rel_error': float('nan'),
                'mean_abs_rel_error_pct': float('nan'),
                'error': str(e),
            })

        # Increment accuracy boost
        accuracy_boost += args.step
        accuracy_boost = round(accuracy_boost, 2)  # Avoid floating point issues

    else:
        print("-" * 80)
        print(f"\n✗ Target not reached within AccuracyBoost ≤ {args.max}")

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    df = pd.DataFrame(results)
    print("\nAll results:")
    print(df.to_string(index=False))

    return results


if __name__ == '__main__':
    main()
