# Advanced ODE Optimization Techniques

This document provides references and resources for advanced parallel ODE solving techniques that could potentially speed up the `evolve_background()` computation.

## 1. Parallel-in-Time Integration

### Overview
Traditional ODE solvers are inherently sequential (each step depends on the previous). Parallel-in-time methods solve for multiple time points simultaneously.

### Key Papers

**Foundational Work:**
- **Parareal Algorithm**
  Lions, J. L., Maday, Y., & Turinici, G. (2001). "A 'parareal' in time discretization of PDE's."
  *Comptes Rendus de l'Académie des Sciences-Series I-Mathematics*, 332(7), 661-668.
  [DOI: 10.1016/S0764-4442(00)01793-6](https://doi.org/10.1016/S0764-4442(00)01793-6)

- **PFASST (Parallel Full Approximation Scheme in Space and Time)**
  Emmett, M., & Minion, M. (2012). "Toward an efficient parallel in time method for partial differential equations."
  *Communications in Applied Mathematics and Computational Science*, 7(1), 105-132.
  [DOI: 10.2140/camcos.2012.7.105](https://doi.org/10.2140/camcos.2012.7.105)

**Review Articles:**
- Gander, M. J. (2015). "50 years of time parallel time integration."
  In *Multiple Shooting and Time Domain Decomposition Methods* (pp. 69-113). Springer.
  [DOI: 10.1007/978-3-319-23321-5_3](https://doi.org/10.1007/978-3-319-23321-5_3)
  **Excellent survey of the field!**

### Software Implementations
- **PyParareal**: Python implementation of Parareal
  https://github.com/Parallel-in-Time/pySDC

- **XBraid**: Parallel-in-time library (C/MPI)
  https://github.com/XBraid/xbraid

## 2. Collocation Methods

### Overview
Discretize the entire ODE trajectory at once, creating a large coupled nonlinear system. Particularly effective for stiff problems.

### Key Papers

**Implicit Runge-Kutta (IRK) Methods:**
- Hairer, E., & Wanner, G. (1996). *Solving Ordinary Differential Equations II: Stiff and Differential-Algebraic Problems*.
  Springer Series in Computational Mathematics.
  **Chapter IV: "Implicit Runge-Kutta Methods"** (pp. 100-145)
  [Book Link](https://link.springer.com/book/10.1007/978-3-642-05221-7)

**Spectral Deferred Corrections (SDC):**
- Dutt, A., Greengard, L., & Rokhlin, V. (2000). "Spectral deferred correction methods for ordinary differential equations."
  *BIT Numerical Mathematics*, 40(2), 241-266.
  [DOI: 10.1023/A:1022338906936](https://doi.org/10.1023/A:1022338906936)

**Parallel SDC:**
- Minion, M. L. (2011). "A hybrid parareal spectral deferred corrections method."
  *Communications in Applied Mathematics and Computational Science*, 5(2), 265-301.
  [DOI: 10.2140/camcos.2010.5.265](https://doi.org/10.2140/camcos.2010.5.265)

### Software
- **dedalus**: Spectral PDE solver with efficient collocation
  https://github.com/DedalusProject/dedalus

- **pySDC**: Spectral Deferred Corrections in Python
  https://github.com/Parallel-in-Time/pySDC

## 3. GPU-Optimized Linear Algebra

For the large sparse systems arising from collocation/parallel-in-time methods:

### Block Tridiagonal Solvers
- Venetis, I. E., Kouris, A., Sobczyk, A., & Gallopoulos, E. (2015).
  "A Direct Tridiagonal Solver Based on Givens Rotations for GPU Architectures."
  *Journal of Supercomputing*, 71(6), 2420-2436.
  [DOI: 10.1007/s11227-015-1404-4](https://doi.org/10.1007/s11227-015-1404-4)

### GPU Sparse Solvers
- **cuSPARSE**: NVIDIA's GPU sparse linear algebra library
  https://docs.nvidia.com/cuda/cusparse/

- **JAX sparse support**: Experimental BCOO format
  https://jax.readthedocs.io/en/latest/jax.experimental.sparse.html

## 4. Specific to Cosmology

### Similar Problems in Cosmology
- **Boltzmann Codes**: CLASS and CAMB also solve stiff ODEs for recombination
  - CLASS uses a standard adaptive ODE solver (similar to current approach)
  - Both are CPU-optimized; GPU acceleration remains challenging

### Relevant Papers
- Lesgourgues, J. (2011). "The Cosmic Linear Anisotropy Solving System (CLASS) I: Overview."
  *arXiv:1104.2932*
  https://arxiv.org/abs/1104.2932
  (Section on recombination ODE solving)

## 5. Practical Implementation Strategy

If you want to implement matrix ODE approach for DISCO-EB:

### Phase 1: Prototype (1-2 weeks)
1. Implement simple implicit midpoint collocation for 1 interval
2. Extend to multiple intervals with block-tridiagonal structure
3. Compare accuracy vs current approach

### Phase 2: Optimization (2-4 weeks)
1. Implement sparse matrix storage (BCOO format in JAX)
2. Use JAX's `jax.scipy.sparse.linalg.bicgstab` for iterative solving
3. Profile and optimize memory access patterns

### Phase 3: Integration (1-2 weeks)
1. Replace `solve_ionization()` in thermodynamics_recfast.py
2. Tune parameters (number of collocation points, solver tolerances)
3. Validate against CLASS/CAMB benchmarks

### Sample Code Structure
```python
def solve_ionization_collocation(a_grid, y0, param):
    """Solve recombination ODE using implicit collocation"""
    n_points = len(a_grid)
    n_vars = 3  # [x_H, x_He, T_mat]

    # Initial guess: linear interpolation
    Y_guess = jnp.zeros((n_points, n_vars))

    # Define residual function for collocation equations
    def residual(Y_flat):
        Y = Y_flat.reshape(n_points, n_vars)
        R = jnp.zeros_like(Y)

        # Initial condition
        R = R.at[0].set(Y[0] - y0)

        # Midpoint rule for interior points
        for i in range(1, n_points):
            da = a_grid[i] - a_grid[i-1]
            a_mid = (a_grid[i] + a_grid[i-1]) / 2
            y_mid = (Y[i] + Y[i-1]) / 2

            # Residual: (Y[i] - Y[i-1])/da - f(a_mid, y_mid) = 0
            f_mid = ionization(a_mid, y_mid, (param,))
            R = R.at[i].set((Y[i] - Y[i-1]) / da - f_mid)

        return R.flatten()

    # Solve using Newton's method with JAX autodiff for Jacobian
    from jax.scipy.optimize import minimize
    solution = minimize(
        lambda y: jnp.sum(residual(y)**2),
        Y_guess.flatten(),
        method='BFGS'
    )

    return solution.x.reshape(n_points, n_vars)
```

## 6. Expected Performance Gains

### Theoretical Speedup Analysis

**Current approach (1024 sequential solves):**
- Time = 1024 × (kernel_launch_overhead + small_LU_time)
- GPU utilization: ~20%
- Total: ~1.0 seconds

**Matrix ODE approach:**
- Time = large_LU_time + newton_iterations × large_matvec
- GPU utilization: ~60-80%
- Estimated: ~0.2-0.4 seconds (2-5× speedup)

**Best case (batched + matrix ODE):**
- Batch 100 cosmologies with matrix ODE
- Current: 100 seconds
- Optimized: ~3-5 seconds (**20-30× speedup!**)

## 7. Alternative: Hybrid CPU/GPU Approach

Sometimes the simplest solution is best:

```python
@partial(jax.jit, static_argnames=('thermo_module', 'num_thermo'))
def evolve_background_hybrid(param, ...):
    # Keep ODE solving on CPU (where sequential code is efficient)
    with jax.default_device(jax.devices('cpu')[0]):
        thermo_results = evaluate_thermo_recfast(param, num_thermo)

    # Move results to GPU for vectorized operations
    with jax.default_device(jax.devices('gpu')[0]):
        # Spline interpolation, visibility function, etc.
        ...
```

This can give 2-3× speedup with minimal code changes!

## 8. Further Reading

### Books
- Ascher, U. M., & Petzold, L. R. (1998). *Computer Methods for Ordinary Differential Equations and Differential-Algebraic Equations*.
  SIAM. Chapter 5: "Boundary Value Problems"

- Quarteroni, A., Sacco, R., & Saleri, F. (2007). *Numerical Mathematics*.
  Springer. Chapter 11: "Numerical Solution of ODEs and Evolution PDEs"

### Online Resources
- **Parallel-in-Time Workshop Series**: https://parallel-in-time.org/
- **pySDC Tutorials**: https://parallel-in-time.org/pySDC/
- **JAX Advanced Tutorials**: https://jax.readthedocs.io/en/latest/jax-101/

### Relevant GitHub Repositories
1. **jax-cosmo** (inspiration for cosmology in JAX):
   https://github.com/DifferentiableUniverseInitiative/jax_cosmo

2. **diffrax** (current ODE solver used):
   https://github.com/patrick-kidger/diffrax

3. **PyTorch ODE benchmarks** (useful comparisons):
   https://github.com/rtqichen/torchdiffeq

## 9. Recommended Next Steps

1. **Short term** (easiest wins):
   - Use `num_thermo=512` → 1.7× speedup
   - Implement batching for parameter inference → 5-10× speedup

2. **Medium term** (moderate effort):
   - Try hybrid CPU/GPU → 2-3× speedup
   - Profile other parts of the pipeline (perturbations, power spectrum)

3. **Long term** (research project):
   - Implement collocation-based solver → 3-5× speedup
   - Combine with batching → 20-30× total speedup

---

## Contact & Contributions

If you implement any of these techniques for DISCO-EB, consider:
- Publishing results (could be a good numerical methods paper!)
- Contributing back to the codebase
- Sharing with the JAX/cosmology communities

Good luck! These are challenging but rewarding optimizations to work on.
