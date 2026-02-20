# Background Evolution Profiling Analysis

## Summary

Current performance: **~1.0 seconds** for `evolve_background()` with `num_thermo=1024` on GPU

## Key Findings from Profiling Data

### 1. GPU Underutilization (79% IDLE)
The profiling shows **79% GPU idle time**, indicating severe underutilization. This is the primary bottleneck.

**Root Cause**: The sequential nature of the thermal history ODE solver:
- `compute_thermal_history()` uses `jax.lax.fori_loop` for 1024 sequential iterations
- Each iteration solves a small stiff ODE problem using implicit GRKT4 solver
- Each ODE solve depends on the previous one, preventing parallelization
- This creates many small, sequential kernel launches with low GPU occupancy

### 2. Kernel Launch Overhead
**Most GPU kernels have very low occupancy**:
- Grid dimensions: often (1,1,1)
- Block dimensions: often (1,1,1) or (3,1,1)
- Theoretical occupancy: 0.78% - 1.5625% for many kernels

This is a fundamental limitation when solving many small sequential problems on GPU.

### 3. LU Decomposition Overhead
The implicit GRKT4 solver requires:
- **8,488 triangular solve operations** (total: 121.6ms, 4.6% of runtime)
- **12,732 LU factorizations** (total: 37.9ms, 1.4% of runtime)

These are necessary for the stiff recombination equations but contribute to overhead.

## Optimization Attempts

### ❌ Explicit Solvers (Tsit5, Dopri5)
- **Result**: Failed - broke recombination physics
- **Reason**: Recombination equations are too stiff for explicit methods

### ❌ Larger Initial Timesteps
- **Result**: No improvement or slightly slower
- **Reason**: Adaptive controller still needs many steps for accuracy

### ❌ Constant Step Size Controller
- **Result**: 32% slower (1.32s vs 1.0s)
- **Reason**: Removes beneficial adaptivity without enough compensation

## Recommendations

### ✅ Primary Recommendation: Use Fewer Sampling Points
According to the documentation in [`background.py:203-208`](src/discoeb/background.py#L203-L208):

- `num_thermo=512`: **1.7x faster**, <0.03% error on P(k)
- `num_thermo=1024`: Reference accuracy (current test setting)

**For most applications, use `num_thermo=512` for optimal speed/accuracy tradeoff.**

### ✅ Architecture-Level Optimizations (Future Work)
The 79% idle time is fundamentally due to sequential execution. Potential improvements:

1. **Batch Multiple Cosmologies**: If computing for multiple parameter sets, batch them together
2. **Restructure as Matrix ODE**: Reformulate to solve larger matrix systems instead of sequential scalar problems
3. **Hybrid CPU/GPU**: Keep ODE solving on CPU where sequential code performs better, use GPU only for vector operations

### ✅ Current Code is Already Well-Optimized
Given the constraints:
- Using appropriate implicit solver (GRKT4) for stiff equations
- Adaptive sampling concentrates points around recombination
- Forward-mode AD for efficiency
- Performance is already good (~1s for full background evolution)

## Profiling Breakdown (Total Time Basis)

| Operation | Time (μs) | % of Total | Count |
|-----------|-----------|------------|-------|
| IDLE | 2,095,223 | 79.1% | - |
| Triangular solve | 121,640 | 4.6% | 8,488 |
| Scatter | 53,682 | 2.0% | 1,021 |
| Gather | 52,251 | 2.0% | 2,122 |
| LU decomposition | 37,905 | 1.4% | 12,732 |
| Other operations | 287,786 | 10.9% | Various |

## Conclusion

The current implementation is **already well-optimized** for the given problem structure. The 79% GPU idle time is an inherent limitation of solving sequential stiff ODEs on GPU. For practical speedups:

1. Use `num_thermo=512` instead of 1024 when possible (1.7x faster)
2. Consider CPU execution for this specific bottleneck if doing single evaluations
3. Only use GPU acceleration when batching multiple cosmologies or doing full CMB calculations

The real performance gains will come from higher-level optimizations (batching, restructuring) rather than micro-optimizations to the ODE solver.
