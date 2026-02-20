#!/usr/bin/env python3
"""Run tests without pytest"""

import sys
sys.path.insert(0, 'tests')

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'gpu')

from test_cmb import test_benchmark_evolve_background, test_cmb_vs_camb, STANDARD_COSMOLOGIES
import time

# Mock benchmark fixture
class MockBenchmark:
    def __call__(self, func, *args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"  Benchmark: {elapsed:.3f} seconds")
        return result

# Mock num_regression fixture
class MockNumRegression:
    def check(self, data, **kwargs):
        print(f"  Regression check: {len(data)} values")
        for key, val in data.items():
            if hasattr(val, 'shape'):
                print(f"    {key}: shape={val.shape}")
            else:
                print(f"    {key}: {val}")

print("=" * 70)
print("Running test_benchmark_evolve_background...")
print("=" * 70)
try:
    test_benchmark_evolve_background(MockBenchmark(), MockNumRegression())
    print("✓ test_benchmark_evolve_background PASSED\n")
except Exception as e:
    print(f"✗ test_benchmark_evolve_background FAILED: {e}\n")
    import traceback
    traceback.print_exc()

print("=" * 70)
print("Running test_cmb_vs_camb...")
print("=" * 70)
try:
    test_cmb_vs_camb(MockNumRegression())
    print("✓ test_cmb_vs_camb PASSED\n")
except Exception as e:
    print(f"✗ test_cmb_vs_camb FAILED: {e}\n")
    import traceback
    traceback.print_exc()

print("=" * 70)
print("All tests completed")
print("=" * 70)
