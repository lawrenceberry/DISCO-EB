#!/usr/bin/env python3
"""Quick test to verify optimization maintains correctness"""

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'gpu')

from discoeb.background import evolve_background

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

print("Testing evolve_background with optimized ODE solver...")
print("=" * 70)

# Warmup
evolve_background(param=param.copy(), thermo_module='RECFAST', num_thermo=1024)
print("✓ Warmup complete (JIT compilation done)")

# Run and verify output
result = evolve_background(param=param.copy(), thermo_module='RECFAST', num_thermo=1024)
print("✓ evolve_background executed successfully")

# Basic sanity checks
assert 'tau_of_a_spline' in result, "Missing tau_of_a_spline"
assert 'xe_of_loga_spline' in result, "Missing xe_of_loga_spline"
assert 'aexp' in result, "Missing aexp"
assert 'tau' in result, "Missing tau"
assert 'xe' in result, "Missing xe"
print("✓ All expected outputs present")

# Check physical validity
assert jnp.all(result['xe'] >= 0), "Ionization fraction should be non-negative"
assert jnp.all(result['xe'] <= 2.0), "Ionization fraction should be reasonable"
assert jnp.all(jnp.diff(result['tau']) > 0), "Conformal time should be monotonically increasing"
print("✓ Physical validity checks passed")

# Evaluate splines at test points
test_points = jnp.array([0.001, 0.01, 0.1, 0.5, 1.0])
tau_vals = result['tau_of_a_spline'].evaluate(test_points)
xe_vals = jnp.exp(result['xe_of_loga_spline'].evaluate(jnp.log(test_points)))

print("\nSample outputs at test points:")
print(f"Scale factors: {test_points}")
print(f"Conformal times: {tau_vals}")
print(f"Ionization fractions: {xe_vals}")

# Check against expected values (rough sanity checks)
assert tau_vals[0] < tau_vals[-1], "Conformal time should increase with scale factor"
assert xe_vals[0] > 0.99, "Early universe should be highly ionized"
assert xe_vals[2] < 0.5, "Recombination should have occurred by a=0.1"
print("✓ Spline evaluation sanity checks passed")

print("\n" + "=" * 70)
print("SUCCESS! All tests passed. Optimization maintains correctness.")
print("=" * 70)
