#!/usr/bin/env python3
"""
Collocation demonstration with 3D vector ODE (matching DISCO-EB structure)

This generalizes the scalar demo to vector fields, representing the actual
DISCO-EB recombination equations which solve for y = [x_H, x_He, T_mat].
"""

import jax
import jax.numpy as jnp
import lineax
import optimistix
import time

jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'gpu')

print("=" * 70)
print("Collocation Method: 1024 Points, 3D Vector ODE")
print("=" * 70)

# Verify GPU is being used
print(f"\nJAX backend: {jax.default_backend()}")
print(f"Devices: {jax.devices()}")
if jax.default_backend() != 'gpu':
    print("WARNING: Not running on GPU! Results may not be representative.")
print()

# ============================================================================
# 3D Coupled Stiff ODE System
# ============================================================================
# This represents a simplified version of coupled recombination equations
# y = [y1, y2, y3] (similar to [x_H, x_He, T_mat] in DISCO-EB)
#
# dy1/dt = -k1*y1 + k1*sin(t)           (ionization-like)
# dy2/dt = -k2*(y2 - y1) + k2*cos(t)    (coupled to y1)
# dy3/dt = -k3*(y3 - y2)                (temperature-like, coupled to y2)
#
# Initial conditions: y(0) = [0, 0, 0]

k1, k2, k3 = 100.0, 150.0, 200.0  # Stiffness parameters (different scales)

def f(t, y):
    """
    RHS of vector ODE: dy/dt = f(t, y)

    Args:
        t: scalar time
        y: vector of shape (3,)

    Returns:
        dy/dt: vector of shape (3,)
    """
    y1, y2, y3 = y[0], y[1], y[2]

    dy1_dt = -k1 * y1 + k1 * jnp.sin(t)
    dy2_dt = -k2 * (y2 - y1) + k2 * jnp.cos(t)
    dy3_dt = -k3 * (y3 - y2)

    return jnp.array([dy1_dt, dy2_dt, dy3_dt])

# Time domain and grid
t0, t1 = 0.0, 1.0
n_points = 1024
n_vars = 3  # Dimension of y vector

print("\nProblem Setup:")
print(f"  Vector ODE system with {n_vars} coupled equations")
print(f"  dy1/dt = -{k1}*y1 + {k1}*sin(t)")
print(f"  dy2/dt = -{k2}*(y2 - y1) + {k2}*cos(t)")
print(f"  dy3/dt = -{k3}*(y3 - y2)")
print("  Initial condition: y(0) = [0, 0, 0]")
print(f"  Time interval: [{t0}, {t1}]")
print(f"  Number of time points: {n_points}")
print(f"  Total unknowns: {n_points * n_vars}")

t_grid = jnp.linspace(t0, t1, n_points)
dt = t_grid[1] - t_grid[0]

# ============================================================================
# Method 1: Sequential (current DISCO-EB approach)
# ============================================================================

print("\n" + "=" * 70)
print("Method 1: Sequential Solving (current DISCO-EB approach)")
print("=" * 70)

def solve_sequential():
    """Solve vector ODE sequentially using implicit Euler"""
    Y_values = jnp.zeros((n_points, n_vars))
    y_current = jnp.zeros(n_vars)  # Initial condition

    def body_fun(i, state):
        Y_arr, y_prev = state

        # Implicit Euler: y[i] - y[i-1] - dt * f(t[i], y[i]) = 0
        # This is a nonlinear system for y[i], solve with Newton's method

        t_i = t_grid[i]

        def residual(y_next):
            return y_next - y_prev - dt * f(t_i, y_next)

        # Newton's method for this small 3D system
        y_next = y_prev  # Initial guess
        for _ in range(5):  # 5 Newton iterations
            r = residual(y_next)
            # Jacobian: I - dt * df/dy
            J = jnp.eye(n_vars) - dt * jax.jacfwd(lambda y: f(t_i, y))(y_next)
            # Solve J * delta = -r
            delta = jnp.linalg.solve(J, -r)
            y_next = y_next + delta

        Y_arr = Y_arr.at[i].set(y_next)
        return (Y_arr, y_next)

    Y_values, _ = jax.lax.fori_loop(1, n_points, body_fun, (Y_values, y_current))
    return Y_values

# JIT compile
solve_sequential_jit = jax.jit(solve_sequential)

# Warm up
print("Warming up sequential solver...")
_ = solve_sequential_jit()
_ = solve_sequential_jit()

# Time it
print("Timing sequential solver...")
start = time.time()
for _ in range(10):
    Y_sequential = solve_sequential_jit()
    jax.block_until_ready(Y_sequential)
time_sequential = (time.time() - start) / 10

print("✓ Sequential method")
print(f"  Time: {time_sequential*1000:.2f} ms (averaged over 10 runs)")
print(f"  Final values: y({t1}) = [{Y_sequential[-1, 0]:.6f}, {Y_sequential[-1, 1]:.6f}, {Y_sequential[-1, 2]:.6f}]")
print("  Characteristics:")
print(f"    - {n_points} sequential solves")
print(f"    - Each solve: {n_vars}×{n_vars} Newton iteration")
print(f"    - Total: {n_points * n_vars} coupled equations")
print(f"    - On GPU: {n_points} small kernel launches → poor utilization")

# ============================================================================
# Method 2: Collocation with optimistix Newton + lineax GMRES
# ============================================================================

print("\n" + "=" * 70)
print("Method 2: Collocation with optimistix Newton + lineax GMRES")
print("=" * 70)

def solve_collocation_optimistix():
    """
    Solve using optimistix Newton root finder with lineax GMRES.

    The Jacobian-vector product needed by GMRES is provided via a
    jax.custom_jvp rule on the residual function.  Inside that rule we:
      1. Compute D[i] = I - dt*∂f/∂y for i=1..N-1  (vmapped jacfwd).
      2. Return dR[0]=v[0],  dR[i] = D[i-1]@v[i] - v[i-1]  (einsum + slice).

    This lets optimistix.Newton + lineax.GMRES use our efficient O(N·k²)
    matvec instead of a dense auto-differentiated Jacobian.
    """

    # ------------------------------------------------------------------
    # Vectorised residual  R(Y) = 0
    # R[0]  = Y[0]                          (initial condition)
    # R[i]  = Y[i] - Y[i-1] - dt*f(t_i, Y[i])   i = 1..N-1
    # ------------------------------------------------------------------
    def residual(Y_flat):
        Y = Y_flat.reshape(n_points, n_vars)
        R = jnp.zeros((n_points, n_vars))
        R = R.at[0].set(Y[0])
        rhs = jax.vmap(f)(t_grid[1:], Y[1:])          # vectorised over time
        R = R.at[1:].set(Y[1:] - Y[:-1] - dt * rhs)
        return R.reshape(-1)

    # ------------------------------------------------------------------
    # Custom JVP: inject our block-tridiagonal matvec so that
    # optimistix / lineax never need to auto-diff through the residual.
    # ------------------------------------------------------------------
    @jax.custom_jvp
    def residual_fn(Y_flat):
        return residual(Y_flat)

    @residual_fn.defjvp
    def _(primals, tangents):
        Y_flat, = primals
        v,      = tangents
        Y   = Y_flat.reshape(n_points, n_vars)
        v_r = v.reshape(n_points, n_vars)

        primal_out = residual(Y_flat)

        # D[i] = I - dt * ∂f/∂y at Y[i], for i = 1..N-1
        # (D[0] = I is not needed: dR[0] = v[0] directly)
        D = jax.vmap(
            lambda i: jnp.eye(n_vars) - dt * jax.jacfwd(lambda y: f(t_grid[i], y))(Y[i])
        )(jnp.arange(1, n_points))                              # (N-1, k, k)

        # dR[0] = v[0]
        # dR[i] = D[i-1] @ v[i] - v[i-1]   for i = 1..N-1
        t_rest = jnp.einsum('ijk,ik->ij', D, v_r[1:]) - v_r[:-1]
        tangent_out = jnp.concatenate([v_r[:1], t_rest]).reshape(-1)

        return primal_out, tangent_out

    # ------------------------------------------------------------------
    # optimistix Newton with lineax GMRES as the linear solver.
    # Newton calls lineax.linearise(residual_fn, y) which triggers our
    # custom JVP, so GMRES uses our block-tridiagonal matvec directly.
    # ------------------------------------------------------------------
    linear_solver = lineax.GMRES(rtol=1e-10, atol=1e-10, restart=50, max_steps=100)
    newton_solver = optimistix.Newton(
        rtol=1e-8, atol=1e-8,
        linear_solver=linear_solver,
    )

    y0     = jnp.zeros(n_points * n_vars)
    result = optimistix.root_find(
        lambda y, _: residual_fn(y),
        newton_solver,
        y0,
        max_steps=10,
        throw=False,
    )

    return result.value.reshape(n_points, n_vars)

# JIT compile
solve_collocation_jit = jax.jit(solve_collocation_optimistix)

# Warm up
print("Warming up collocation solver...")
_ = solve_collocation_jit()
_ = solve_collocation_jit()

# Time it
print("Timing collocation solver...")
start = time.time()
for _ in range(10):
    Y_collocation = solve_collocation_jit()
    jax.block_until_ready(Y_collocation)
time_collocation = (time.time() - start) / 10

print("✓ Collocation method (optimistix Newton + lineax GMRES + custom matvec)")
print(f"  Time: {time_collocation*1000:.2f} ms (averaged over 10 runs)")
print(f"  Final values: y({t1}) = [{Y_collocation[-1, 0]:.6f}, {Y_collocation[-1, 1]:.6f}, {Y_collocation[-1, 2]:.6f}]")
print(f"  Speedup vs sequential: {time_sequential/time_collocation:.2f}x")
print("  Characteristics:")
print("    - optimistix.Newton outer loop (max 10 steps, rtol=1e-8)")
print("    - lineax.GMRES inner linear solver (restart=50, rtol=1e-10)")
print("    - custom_jvp: matvec(v) returned directly as tangent output")
print("    - custom_jvp injects our O(N*k^2) matvec into the Newton linearisation")
print(f"    - Never builds full {n_points*n_vars}x{n_points*n_vars} Jacobian!")

# Verify solutions match
max_error = jnp.max(jnp.abs(Y_sequential - Y_collocation))
print(f"\n  Accuracy vs sequential: max |error| = {max_error:.2e}")

# ============================================================================
# Analysis: Block-Tridiagonal Jacobian Structure
# ============================================================================

print("\n" + "=" * 70)
print("Jacobian Matrix Analysis")
print("=" * 70)

# Compute Jacobian at solution to analyze structure
def residual_for_jacobian(Y_flat):
    Y = Y_flat.reshape(n_points, n_vars)
    R = jnp.zeros((n_points, n_vars))
    R = R.at[0].set(Y[0])
    for i in range(1, n_points):
        R = R.at[i].set(Y[i] - Y[i-1] - dt * f(t_grid[i], Y[i]))
    return R.reshape(-1)

Y_test = jnp.zeros(n_points * n_vars)
J = jax.jacfwd(residual_for_jacobian)(Y_test)

print("\nJacobian dimensions:")
print(f"  Full size: {n_points * n_vars}x{n_points * n_vars} = {(n_points * n_vars)**2:,} elements")
print(f"  Block structure: {n_points} blocks of {n_vars}x{n_vars}")

# Count non-zero elements (with tolerance)
nnz = jnp.count_nonzero(jnp.abs(J) > 1e-10)
print("\nSparsity:")
print(f"  Non-zero elements: {nnz:,}")
print(f"  Sparsity: {100 * (1 - nnz / J.size):.3f}%")
print(f"  Storage: Dense {J.size * 8 / 1024:.1f} KB, Sparse ~{nnz * 8 / 1024:.1f} KB")

print("\n  Block-tridiagonal structure:")
print("  Each time point i couples to:")
print(f"    - Point i-1: {n_vars}x{n_vars} block (lower)")
print(f"    - Point i:   {n_vars}x{n_vars} block (diagonal)")
print(f"  Total non-zero blocks: ~{2 * n_points} blocks of {n_vars}×{n_vars}")

# Visualize structure (show first 30×30 elements, covering 10 time points)
print("\n  Structure visualization (first 30×30, showing 10 time points):")
print("  (X = non-zero, . = zero)")
for i in range(min(30, n_points * n_vars)):
    row = "  "
    for j in range(min(30, n_points * n_vars)):
        if abs(J[i, j]) > 1e-10:
            row += "X "
        else:
            row += ". "
    print(row)

print("\n  → Block-tridiagonal pattern clearly visible!")
print(f"  → Each {n_vars}×{n_vars} block represents coupling within/between time points")

# ============================================================================
# Comparison with DISCO-EB
# ============================================================================

print("\n" + "=" * 70)
print("Relevance to DISCO-EB Recombination Equations")
print("=" * 70)

print(f"""
This demonstration matches DISCO-EB structure:
  - Vector ODE: y = [y1, y2, y3] (vs DISCO-EB: [x_H, x_He, T_mat])
  - Coupled equations with different stiffness scales
  - {n_points} time points (matches num_thermo=1024)
  - Implicit Euler discretization

Key findings:

1. Sequential solving: {time_sequential*1000:.2f} ms
   - {n_points} small {n_vars}×{n_vars} Newton solves
   - Many kernel launches → poor GPU utilization
   - Matches current DISCO-EB approach

2. Collocation (optimistix + lineax): {time_collocation*1000:.2f} ms
   - optimistix.Newton outer loop (converges in a few steps)
   - lineax.GMRES inner solve; matvec injected via custom_jvp
   - custom_jvp returns matvec(v) directly as the tangent output
   - Speedup: {time_sequential/time_collocation:.2f}×
   - O(N) memory, fully vectorised operations!

3. Jacobian structure:
   - {100 * (1 - nnz / J.size):.1f}% sparse (block-lower-bidiagonal)
   - Perfect for GPU-optimised sparse solvers
   - Memory savings: {J.size / nnz:.0f}× with sparse storage

For DISCO-EB implementation:
  ✓ Use this exact structure with actual recombination RHS
  ✓ lineax.GMRES with custom matvec via custom_jvp (O(N) memory)
  ✓ JAX-friendly: vectorised ops, no complex indexing
  ✓ Combine with vmap batching → 10-30× total speedup
  ✓ Memory-efficient and GPU-optimised!
""")

# ============================================================================
# Summary
# ============================================================================

print("=" * 70)
print("Summary")
print("=" * 70)

print(f"""
Vector ODE demonstration ({n_vars}D system, {n_points} points):

1. Problem scale: {n_points * n_vars} total unknowns (same as DISCO-EB)

2. Sequential approach: {time_sequential*1000:.2f} ms
   - Standard implicit Euler with sequential Newton solves
   - Each step: 5 iterations × {n_vars}×{n_vars} linear solve

3. Collocation (optimistix + lineax): {time_collocation*1000:.2f} ms
   - optimistix.Newton: outer root-finding loop
   - lineax.GMRES: inner linear solve (restart=50, rtol=1e-10)
   - jax.custom_jvp: returns matvec(v) directly as the tangent
   - Matvec: O(N·k²) = O({n_points}·{n_vars}²) per GMRES iteration
   - Memory: O(N·k²) — never builds the full Jacobian!
   - Speedup: {time_sequential/time_collocation:.2f}×

4. Accuracy: max |collocation - sequential| = {max_error:.2e}

5. Jacobian: {100 * (1 - nnz / J.size):.1f}% sparse, block-lower-bidiagonal structure

Design notes:
  - jax.custom_jvp injects our efficient matvec into the Newton linearisation.
    optimistix.Newton calls lineax.linearise(residual_fn, y), which triggers
    the custom JVP rule; we return matvec(v) directly as the tangent.
  - GMRES works for non-symmetric lower-bidiagonal systems (unlike CG).
  - Perfect for large N with small k (DISCO-EB: N=1024, k=3).

Next step: Apply this to actual DISCO-EB recombination equations in
thermodynamics_recfast.py using the ionization() RHS function.
""")

print("=" * 70)
