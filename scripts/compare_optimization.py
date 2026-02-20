#!/usr/bin/env python3
"""Compare optimized vs original implementation"""

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'gpu')

import time
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

print("Warming up...")
evolve_background(param=param.copy(), thermo_module='RECFAST', num_thermo=1024)

print("\nRunning benchmark...")
start = time.time()
result = evolve_background(param=param.copy(), thermo_module='RECFAST', num_thermo=1024)
jax.block_until_ready(result['xe'])
elapsed = time.time() - start

print(f"Time: {elapsed:.3f} seconds")

# Check key outputs
test_points = jnp.array([0.001, 0.01, 0.1, 0.5, 1.0])
tau_vals = result['tau_of_a_spline'].evaluate(test_points)
xe_vals = result['xe_of_loga_spline'].evaluate(jnp.log(test_points))

print(f"\nKey results:")
print(f"  H0: {result['H0']:.6f}")
print(f"  Omegam: {result['Omegam']:.6f}")
print(f"  tau_0: {result['tau0']:.6f}")
print(f"\nIonization history (log xe):")
for i, a in enumerate(test_points):
    z = 1/a - 1
    print(f"  a={a:.3f}, z={z:8.2f}: log(xe)={xe_vals[i]:.6f}")

print("\n✓ Test completed successfully")
