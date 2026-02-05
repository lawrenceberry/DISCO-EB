#!/usr/bin/env python
"""
CAMB Benchmark Script with DISCO-EB-Matched Accuracy Settings

This script benchmarks CAMB with accuracy settings configured to match DISCO-EB's
default numerical precision. It tests performance across different CPU parallelism levels.

DISCO-EB Default Settings (to match):
======================================
- ODE tolerances: rtol=1e-4, atol=1e-4 (perturbations)
- k-sampling: 512 k-modes, CAMB-like hybrid spacing
- Multipole truncation: lmaxg=11, lmaxgp=11, lmaxr=11, lmaxnu=8
- Neutrino momentum bins: nqmax=3
- CMB ellmax: 2500
- Time sampling: 320 output scale factors concentrated around recombination
- FFTLog integration: 16384 coefficients

CAMB Accuracy Parameter Mapping:
================================
AccuracyBoost=1 corresponds to ~0.1% scalar C_l accuracy at ell>600.
DISCO-EB with default settings achieves ~10-25% agreement with CAMB,
so we need to adjust CAMB settings to match DISCO-EB's lower-accuracy defaults.

Key mappings:
- IntTolBoost: ODE integration tolerance (higher = tighter tolerances)
- TimeStepBoost: Time stepping density
- SourcekAccuracyBoost: k-sampling for source functions
- IntkAccuracyBoost: k-sampling for line-of-sight integration
- TransferkBoost: k-sampling for transfer functions
- lAccuracyBoost: Multipole hierarchy truncation
- neutrino_q_boost: Neutrino momentum sampling
- lSampleBoost: ell sampling density (>=50 computes all ells)

Usage:
    python scripts/benchmark_camb_accuracy_match.py
    python scripts/benchmark_camb_accuracy_match.py --max-threads 16
    python scripts/benchmark_camb_accuracy_match.py --ellmax 3000 --repeat 5
"""

import argparse
import os
import time
import numpy as np
import camb


# DISCO-EB standard cosmology (from test_cmb.py)
DISCO_COSMOLOGY = {
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


def create_camb_params_disco_matched(params, ellmax=2500, accuracy_level='disco_default'):
    """
    Create CAMB parameters matching DISCO-EB accuracy settings.

    Parameters
    ----------
    params : dict
        Cosmological parameters in DISCO-EB format
    ellmax : int
        Maximum multipole to compute
    accuracy_level : str
        One of:
        - 'disco_default': Match DISCO-EB default settings (lower accuracy, faster)
        - 'disco_high': Match DISCO-EB high-accuracy test settings
        - 'camb_default': CAMB default accuracy (~0.1% at ell>600)
        - 'camb_high': High accuracy CAMB settings

    Returns
    -------
    camb.CAMBparams
        Configured CAMB parameters object

    Notes
    -----
    DISCO-EB default accuracy settings:
    - rtol=1e-4, atol=1e-4 for perturbation ODE
    - lmaxg=11 (photon multipoles)
    - lmaxgp=11 (polarization multipoles)
    - lmaxr=11 (massless neutrino multipoles)
    - lmaxnu=8 (massive neutrino multipoles)
    - nqmax=3 (neutrino momentum bins)
    - 512 k-modes for perturbations
    - 320 time samples around recombination

    DISCO-EB high-accuracy test settings:
    - lmaxg=31, lmaxgp=31, lmaxr=31, lmaxnu=31
    - nqmax=5 (neutrino momentum bins)
    - rtol=1e-4, atol=1e-4
    """
    cpars = camb.CAMBparams()

    # Set cosmological parameters
    h = params['H0'] / 100.0
    cpars.set_cosmology(
        H0=params['H0'],
        ombh2=params['Omegab'] * h**2,
        omch2=(params['Omegam'] - params['Omegab']) * h**2,
        omk=params['Omegak'],
        tau=0.0,  # No reionization in DISCO-EB comparison
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

    # Set for lmax first (establishes baseline)
    cpars.set_for_lmax(ellmax)

    # Configure accuracy based on level
    # Note: set_accuracy() only accepts AccuracyBoost, lSampleBoost, lAccuracyBoost,
    # DoLateRadTruncation, and min_l_logl_sampling. Other parameters must be set
    # directly on cpars.Accuracy.
    if accuracy_level == 'disco_default':
        # Match DISCO-EB default accuracy (lower accuracy, faster)
        # DISCO-EB uses rtol=atol=1e-4 which is looser than CAMB default
        # DISCO-EB uses lmax=11 for photon hierarchy (CAMB default is ~25)
        # DISCO-EB uses nqmax=3 for neutrinos (CAMB default is higher)
        cpars.set_accuracy(
            AccuracyBoost=0.5,          # Lower overall accuracy to match DISCO-EB
            lAccuracyBoost=0.4,         # Fewer multipoles in hierarchy (DISCO: lmax=11)
            lSampleBoost=1.0,           # Standard ell sampling for output
        )
        # Set additional accuracy parameters directly
        cpars.Accuracy.IntTolBoost = 0.5            # Looser ODE tolerances (DISCO: rtol=atol=1e-4)
        cpars.Accuracy.TimeStepBoost = 0.5          # Coarser time stepping
        cpars.Accuracy.BackgroundTimeStepBoost = 0.5
        cpars.Accuracy.SourcekAccuracyBoost = 0.8   # k-sampling for sources
        cpars.Accuracy.IntkAccuracyBoost = 0.8      # k-sampling for integration
        cpars.Accuracy.TransferkBoost = 0.8         # k-sampling for transfers
        cpars.Accuracy.neutrino_q_boost = 0.5       # Neutrino momentum bins (DISCO: nqmax=3)

    elif accuracy_level == 'disco_high':
        # Match DISCO-EB high-accuracy test settings
        # Uses lmax=31 for all species, nqmax=5
        cpars.set_accuracy(
            AccuracyBoost=1.0,          # Standard CAMB accuracy
            lAccuracyBoost=1.2,         # More multipoles (DISCO: lmax=31)
            lSampleBoost=1.0,           # Standard ell sampling
        )
        cpars.Accuracy.IntTolBoost = 1.0
        cpars.Accuracy.TimeStepBoost = 1.0
        cpars.Accuracy.SourcekAccuracyBoost = 1.0
        cpars.Accuracy.IntkAccuracyBoost = 1.0
        cpars.Accuracy.TransferkBoost = 1.0
        cpars.Accuracy.neutrino_q_boost = 1.0

    elif accuracy_level == 'camb_default':
        # Standard CAMB accuracy (~0.1% at ell>600)
        # No modifications needed - this is the default
        pass

    elif accuracy_level == 'camb_high':
        # High accuracy CAMB settings
        cpars.set_accuracy(
            AccuracyBoost=2.0,
            lAccuracyBoost=2.0,
            lSampleBoost=2.0,
        )
        cpars.Accuracy.IntTolBoost = 2.0
        cpars.Accuracy.TimeStepBoost = 2.0
        cpars.Accuracy.SourcekAccuracyBoost = 2.0
        cpars.Accuracy.IntkAccuracyBoost = 2.0
        cpars.Accuracy.TransferkBoost = 2.0
        cpars.Accuracy.neutrino_q_boost = 2.0

    else:
        raise ValueError(f"Unknown accuracy_level: {accuracy_level}")

    return cpars


def benchmark_camb(cpars, n_threads, repeat=3, warmup=1):
    """
    Benchmark CAMB computation with specified thread count.

    Parameters
    ----------
    cpars : camb.CAMBparams
        CAMB parameters
    n_threads : int
        Number of OpenMP threads
    repeat : int
        Number of timed repetitions
    warmup : int
        Number of warmup runs (not timed)

    Returns
    -------
    dict
        Timing statistics
    """
    # Set thread count
    os.environ['OMP_NUM_THREADS'] = str(n_threads)

    # Warmup runs
    for _ in range(warmup):
        results = camb.get_results(cpars)
        _ = results.get_cmb_power_spectra(cpars, CMB_unit='muK')

    # Timed runs
    times = []
    for _ in range(repeat):

        t0 = time.perf_counter()
        results = camb.get_results(cpars)
        powers = results.get_cmb_power_spectra(cpars, CMB_unit='muK')
        t1 = time.perf_counter()

        times.append(t1 - t0)

    times = np.array(times)

    return {
        'n_threads': n_threads,
        'times': times,
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times),
        'powers': powers,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark CAMB with DISCO-EB matched accuracy settings'
    )
    parser.add_argument(
        '--max-threads', type=int, default=None,
        help='Maximum number of threads to test (default: number of CPUs)'
    )
    parser.add_argument(
        '--thread-counts', type=str, default=None,
        help='Comma-separated list of thread counts to test (e.g., "1,2,4,8")'
    )
    parser.add_argument(
        '--ellmax', type=int, default=2500,
        help='Maximum multipole (default: 2500)'
    )
    parser.add_argument(
        '--repeat', type=int, default=5,
        help='Number of timed repetitions per configuration (default: 5)'
    )
    parser.add_argument(
        '--warmup', type=int, default=1,
        help='Number of warmup runs (default: 1)'
    )
    parser.add_argument(
        '--accuracy', type=str, default='all',
        choices=['disco_default', 'disco_high', 'camb_default', 'camb_high', 'all'],
        help='Accuracy level to benchmark (default: all)'
    )
    args = parser.parse_args()

    # Determine thread counts to test
    if args.thread_counts:
        thread_counts = [int(x) for x in args.thread_counts.split(',')]
    else:
        import multiprocessing
        max_threads = args.max_threads or multiprocessing.cpu_count()
        # Test powers of 2 up to max_threads, plus max_threads itself
        thread_counts = []
        t = 1
        while t <= max_threads:
            thread_counts.append(t)
            t *= 2
        if max_threads not in thread_counts:
            thread_counts.append(max_threads)

    # Determine accuracy levels to test
    if args.accuracy == 'all':
        accuracy_levels = ['disco_default', 'disco_high', 'camb_default', 'camb_high']
    else:
        accuracy_levels = [args.accuracy]

    print("=" * 80)
    print("CAMB BENCHMARK WITH DISCO-EB MATCHED ACCURACY SETTINGS")
    print("=" * 80)
    print(f"\nCosmology: DISCO-Notebook standard")
    print(f"ellmax: {args.ellmax}")
    print(f"Repetitions: {args.repeat}")
    print(f"Thread counts: {thread_counts}")
    print(f"Accuracy levels: {accuracy_levels}")
    print()

    # Print accuracy parameter mapping
    print("-" * 80)
    print("ACCURACY SETTINGS EXPLANATION")
    print("-" * 80)
    print("""
DISCO-EB Default Settings → CAMB Parameter Mapping:
====================================================

1. ODE Tolerances (rtol=atol=1e-4):
   → IntTolBoost=0.5 (loosens CAMB's default tolerances)

2. Multipole Truncation (lmaxg=lmaxgp=lmaxr=11, lmaxnu=8):
   → lAccuracyBoost=0.4 (CAMB default uses ~25 multipoles)

3. Neutrino Momentum Bins (nqmax=3):
   → neutrino_q_boost=0.5 (fewer momentum quadrature points)

4. k-Sampling (512 modes, CAMB-like spacing):
   → SourcekAccuracyBoost=0.8, IntkAccuracyBoost=0.8, TransferkBoost=0.8

5. Time Stepping (320 points, concentrated at recombination):
   → TimeStepBoost=0.5

6. Overall Accuracy:
   → AccuracyBoost=0.5 (scales most other parameters)

Note: DISCO-EB achieves ~10-25% agreement with CAMB at default settings.
      The 'disco_default' preset aims to reproduce similar accuracy level.
""")

    all_results = {}

    for accuracy_level in accuracy_levels:
        print("\n" + "=" * 80)
        print(f"ACCURACY LEVEL: {accuracy_level.upper()}")
        print("=" * 80)

        # Create CAMB parameters
        cpars = create_camb_params_disco_matched(
            DISCO_COSMOLOGY,
            ellmax=args.ellmax,
            accuracy_level=accuracy_level
        )

        # Print actual accuracy settings
        acc = cpars.Accuracy
        print(f"\nCAMB Accuracy Settings:")
        print(f"  AccuracyBoost:        {acc.AccuracyBoost:.2f}")
        print(f"  lAccuracyBoost:       {acc.lAccuracyBoost:.2f}")
        print(f"  lSampleBoost:         {acc.lSampleBoost:.2f}")
        print(f"  IntTolBoost:          {acc.IntTolBoost:.2f}")
        print(f"  TimeStepBoost:        {acc.TimeStepBoost:.2f}")
        print(f"  SourcekAccuracyBoost: {acc.SourcekAccuracyBoost:.2f}")
        print(f"  IntkAccuracyBoost:    {acc.IntkAccuracyBoost:.2f}")
        print(f"  TransferkBoost:       {acc.TransferkBoost:.2f}")
        print(f"  neutrino_q_boost:     {acc.neutrino_q_boost:.2f}")

        results = {}

        print(f"\n{'Threads':>8} | {'Mean (s)':>10} | {'Std (s)':>10} | {'Min (s)':>10} | {'Max (s)':>10} | {'Speedup':>8}")
        print("-" * 72)

        baseline_time = None

        for n_threads in thread_counts:
            result = benchmark_camb(cpars, n_threads, repeat=args.repeat, warmup=args.warmup)
            results[n_threads] = result

            if baseline_time is None:
                baseline_time = result['mean']
                speedup = 1.0
            else:
                speedup = baseline_time / result['mean']

            print(f"{n_threads:>8} | {result['mean']:>10.4f} | {result['std']:>10.4f} | "
                  f"{result['min']:>10.4f} | {result['max']:>10.4f} | {speedup:>8.2f}x")

        all_results[accuracy_level] = results

    # Summary comparison across accuracy levels
    if len(accuracy_levels) > 1:
        print("\n" + "=" * 80)
        print("SUMMARY: TIMING COMPARISON ACROSS ACCURACY LEVELS")
        print("=" * 80)

        # Use maximum thread count for comparison
        max_t = max(thread_counts)
        print(f"\nUsing {max_t} threads:\n")
        print(f"{'Accuracy Level':>20} | {'Mean (s)':>10} | {'Relative':>10}")
        print("-" * 46)

        baseline = all_results['camb_default'][max_t]['mean'] if 'camb_default' in all_results else None

        for acc_level in accuracy_levels:
            mean_time = all_results[acc_level][max_t]['mean']
            if baseline:
                relative = mean_time / baseline
                print(f"{acc_level:>20} | {mean_time:>10.4f} | {relative:>10.2f}x")
            else:
                print(f"{acc_level:>20} | {mean_time:>10.4f} | {'N/A':>10}")

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)

    return all_results


if __name__ == '__main__':
    main()
