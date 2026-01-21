"""Test Jacobian reuse optimization in Rodas5Batched solver.

This test compares different values of jacobian_update_every to evaluate:
1. Speed improvement from reusing the Jacobian matrix
2. Accuracy impact from using a stale Jacobian

Usage:
    pytest tests/test_jacobian_reuse.py -v -s
"""
import pytest
import jax
import jax.numpy as jnp
import time

from discoeb.background import evolve_background
from discoeb.perturbations import evolve_perturbations_batched

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)

# Cosmological Parameters (same as test_perturbations.py)
Tcmb = 2.7255
YHe = 0.248
Omegam = 0.3099
Omegab = 0.0488911
w_DE_0 = -0.99
w_DE_a = 0.0
cs2_DE = 1.0
Tnu = (4/11)**(1/3)
Neff = 3.046
N_nu_mass = 1
N_nu_rel = Neff - N_nu_mass * (Tnu/((4/11)**(1/3)))**4
h = 0.67742
A_s = 2.1064e-09
n_s = 0.96822
k_p = 0.05
mnu = 0.06

# Modes to sample
nmodes = 256
kmin = 1e-5
kmax = 1e+1
aexp = 0.01
aexp_out = jnp.array([aexp])

# Initialize parameters
param = {}
param['Omegam'] = Omegam
param['Omegab'] = Omegab
param['w_DE_0'] = w_DE_0
param['w_DE_a'] = w_DE_a
param['cs2_DE'] = cs2_DE
param['Omegak'] = 0.0
param['A_s'] = A_s
param['n_s'] = n_s
param['H0'] = 100*h
param['Tcmb'] = Tcmb
param['YHe'] = YHe
param['Neff'] = N_nu_rel
param['Nmnu'] = N_nu_mass
param['mnu'] = mnu
param['k_p'] = k_p


@pytest.fixture(scope="module")
def background_param():
    """Compute background evolution once for all tests."""
    bg_param = evolve_background(param=param, thermo_module='RECFAST')
    return bg_param


def compute_perturbations_with_jacobian_update(background_param, jacobian_update_every):
    """Compute perturbations with specified Jacobian update frequency."""
    y, kmodes, _ = evolve_perturbations_batched(
        param=background_param,
        kmin=kmin,
        kmax=kmax,
        num_k=nmodes,
        aexp_out=aexp_out,
        lmaxg=11,
        lmaxgp=11,
        lmaxr=11,
        lmaxnu=8,
        nqmax=3,
        max_steps=2048,
        rtol=1e-4,
        atol=1e-4,
        batch_size=32,
        jacobian_update_every=jacobian_update_every,
    )
    return y, kmodes


class TestJacobianReuse:
    """Test suite for Jacobian reuse optimization."""

    def test_jacobian_reuse_accuracy(self, background_param):
        """Test accuracy impact of Jacobian reuse.

        Compares results from jacobian_update_every=1 (baseline) vs 2.
        """
        print("\n" + "="*70)
        print("Testing Jacobian Reuse Accuracy")
        print("="*70)

        # Baseline: update every step
        print("\nComputing baseline (jacobian_update_every=1)...")
        y_baseline, kmodes = compute_perturbations_with_jacobian_update(
            background_param, jacobian_update_every=1
        )

        # Test: update every 2 steps
        print("Computing with jacobian_update_every=2...")
        y_reuse2, _ = compute_perturbations_with_jacobian_update(
            background_param, jacobian_update_every=2
        )

        # Compute power spectrum for comparison
        fac = 2 * jnp.pi**2 * A_s
        iout = -1

        Pkbc_baseline = fac * (kmodes/k_p)**(n_s - 1) * kmodes**(-3) * y_baseline[:, iout, 6]**2
        Pkbc_reuse2 = fac * (kmodes/k_p)**(n_s - 1) * kmodes**(-3) * y_reuse2[:, iout, 6]**2

        # Compute relative error
        rel_error = jnp.abs(Pkbc_reuse2 - Pkbc_baseline) / Pkbc_baseline
        mean_rel_error = jnp.mean(rel_error)
        max_rel_error = jnp.max(rel_error)

        print(f"\nAccuracy comparison (jacobian_update_every=2 vs 1):")
        print(f"  Mean relative error: {mean_rel_error:.6e}")
        print(f"  Max relative error:  {max_rel_error:.6e}")

        # Store for reporting
        self._accuracy_results = {
            'mean_rel_error': float(mean_rel_error),
            'max_rel_error': float(max_rel_error),
        }

        # Assert reasonable accuracy (less than 1% mean error)
        assert mean_rel_error < 0.01, f"Mean relative error {mean_rel_error:.6e} exceeds 1%"

        print("\n✓ Accuracy test passed")

    def test_jacobian_reuse_timing(self, background_param):
        """Test speed improvement from Jacobian reuse.

        Compares timing of jacobian_update_every=1 vs 2.
        """
        print("\n" + "="*70)
        print("Testing Jacobian Reuse Timing")
        print("="*70)

        # Warmup runs to ensure JIT compilation
        print("\nWarmup run (jacobian_update_every=1)...")
        _ = compute_perturbations_with_jacobian_update(background_param, jacobian_update_every=1)

        print("Warmup run (jacobian_update_every=2)...")
        _ = compute_perturbations_with_jacobian_update(background_param, jacobian_update_every=2)

        # Timed runs
        n_runs = 3

        # Baseline timing
        print(f"\nTiming baseline (jacobian_update_every=1, {n_runs} runs)...")
        times_baseline = []
        for i in range(n_runs):
            t0 = time.perf_counter()
            y, _ = compute_perturbations_with_jacobian_update(background_param, jacobian_update_every=1)
            y.block_until_ready()
            t1 = time.perf_counter()
            times_baseline.append(t1 - t0)
            print(f"  Run {i+1}: {times_baseline[-1]:.3f}s")

        # Reuse timing
        print(f"\nTiming with jacobian_update_every=2 ({n_runs} runs)...")
        times_reuse2 = []
        for i in range(n_runs):
            t0 = time.perf_counter()
            y, _ = compute_perturbations_with_jacobian_update(background_param, jacobian_update_every=2)
            y.block_until_ready()
            t1 = time.perf_counter()
            times_reuse2.append(t1 - t0)
            print(f"  Run {i+1}: {times_reuse2[-1]:.3f}s")

        mean_baseline = sum(times_baseline) / len(times_baseline)
        mean_reuse2 = sum(times_reuse2) / len(times_reuse2)
        speedup = mean_baseline / mean_reuse2

        print(f"\nTiming summary:")
        print(f"  jacobian_update_every=1: {mean_baseline:.3f}s (mean)")
        print(f"  jacobian_update_every=2: {mean_reuse2:.3f}s (mean)")
        print(f"  Speedup: {speedup:.2f}x")

        # Store for reporting
        self._timing_results = {
            'mean_baseline': mean_baseline,
            'mean_reuse2': mean_reuse2,
            'speedup': speedup,
        }

        print("\n✓ Timing test complete")

    def test_jacobian_reuse_sweep(self, background_param):
        """Sweep over different jacobian_update_every values.

        Tests values 1, 2, 3, 4 to find optimal trade-off.
        """
        print("\n" + "="*70)
        print("Jacobian Reuse Parameter Sweep")
        print("="*70)

        update_values = [1, 2, 3, 4]
        results = []

        # First compute baseline
        print("\nComputing baseline (jacobian_update_every=1)...")
        y_baseline, kmodes = compute_perturbations_with_jacobian_update(
            background_param, jacobian_update_every=1
        )

        fac = 2 * jnp.pi**2 * A_s
        iout = -1
        Pkbc_baseline = fac * (kmodes/k_p)**(n_s - 1) * kmodes**(-3) * y_baseline[:, iout, 6]**2

        for update_every in update_values:
            print(f"\nTesting jacobian_update_every={update_every}...")

            # Warmup
            _ = compute_perturbations_with_jacobian_update(background_param, jacobian_update_every=update_every)

            # Timed run
            t0 = time.perf_counter()
            y, _ = compute_perturbations_with_jacobian_update(background_param, jacobian_update_every=update_every)
            y.block_until_ready()
            t1 = time.perf_counter()
            elapsed = t1 - t0

            # Accuracy
            Pkbc = fac * (kmodes/k_p)**(n_s - 1) * kmodes**(-3) * y[:, iout, 6]**2
            rel_error = jnp.abs(Pkbc - Pkbc_baseline) / Pkbc_baseline
            mean_rel_error = float(jnp.mean(rel_error))
            max_rel_error = float(jnp.max(rel_error))

            results.append({
                'update_every': update_every,
                'time': elapsed,
                'mean_rel_error': mean_rel_error,
                'max_rel_error': max_rel_error,
            })

            print(f"  Time: {elapsed:.3f}s, Mean rel. error: {mean_rel_error:.6e}, Max rel. error: {max_rel_error:.6e}")

        # Summary table
        print("\n" + "="*70)
        print("Summary Table")
        print("="*70)
        print(f"{'update_every':>12} {'Time (s)':>12} {'Speedup':>10} {'Mean Error':>14} {'Max Error':>14}")
        print("-"*70)

        baseline_time = results[0]['time']
        for r in results:
            speedup = baseline_time / r['time']
            print(f"{r['update_every']:>12} {r['time']:>12.3f} {speedup:>10.2f}x {r['mean_rel_error']:>14.2e} {r['max_rel_error']:>14.2e}")

        print("="*70)

        # Store results
        self._sweep_results = results


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v", "-s"])
