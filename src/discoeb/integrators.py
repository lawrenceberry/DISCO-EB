"""Adaptive Rodas5P ODE integrator (pure JAX).

:func:`rodas5Pjax_solve` is a pure-JAX ensemble solver (``ode_fn`` path) whose
Jacobian is recomputed each step via ``jax.jacfwd``. It is fully traceable,
``jit``-able and ``vmap``-able, and needs no CUDA toolchain, which makes it the
reference the GPU path is checked against.

It uses the Rodas5P W-transformed tableau (Steinebach 2023, BIT 63:27), matching
the Julia ``Rodas5P`` / ``GPURodas5P`` coefficients.

The production perturbation solve does not come through here: it runs modax's
numba-CUDA Rodas5P kernel (``solvers.rodas5P.solve``), handed the sparsity
pattern :mod:`discoeb.eb_sparsity` declares, from which the kernel compiles its
own sparse direct linear solve. That kernel used to be vendored into this
module; modax now carries it, so there is one implementation to maintain rather
than a fork.
"""

from __future__ import annotations

import functools
from typing import Callable, Literal

import jax
import jax.numpy as jnp
from jax.custom_batching import custom_vmap


# =============================================================================
# Shared pure-JAX scaffolding
# =============================================================================


def normalize_y0_params(y0, params):
    """Broadcast ``y0`` / ``params`` to a consistent ``(N, ...)`` ensemble layout.

    Accepts either 1-D (``(n_vars,)`` / ``(n_params,)``) or 2-D
    (``(N, n_vars)`` / ``(N, n_params)``) inputs and returns 2-D arrays with a
    common leading axis.
    """
    y0_in = jnp.asarray(y0, dtype=jnp.float64)
    params_arr = jnp.asarray(params)

    if y0_in.ndim == 1 and params_arr.ndim == 1:
        n = 1
        n_vars = y0_in.shape[0]
        y0_arr = jnp.broadcast_to(y0_in, (n, n_vars))
        params_arr = jnp.broadcast_to(params_arr, (n, params_arr.shape[0]))
    elif y0_in.ndim == 1:
        n = params_arr.shape[0]
        n_vars = y0_in.shape[0]
        y0_arr = jnp.broadcast_to(y0_in, (n, n_vars))
    else:
        n = y0_in.shape[0]
        n_vars = y0_in.shape[1]
        y0_arr = y0_in
        if params_arr.ndim == 1:
            params_arr = jnp.broadcast_to(params_arr, (n, params_arr.shape[0]))
        elif params_arr.shape[0] != n:
            raise ValueError(
                "params must have shape (n_params,) or (N, n_params) when y0 has "
                f"shape (N, n_vars); got y0.shape={y0_in.shape} and "
                f"params.shape={params_arr.shape}"
            )
    return y0_arr, params_arr, n, n_vars


def eval_ode_fn(fn: Callable, y, t, params):
    """Evaluate an ODE callback and normalize tuple/list outputs to an array."""
    return jnp.asarray(fn(y, t, params))


def normalize_inputs(y0, t_span, params, first_step, batch_size):
    y0_arr, params_arr, n, n_vars = normalize_y0_params(y0, params)
    times = jnp.asarray(t_span, dtype=jnp.float64)

    n_save = times.shape[0]
    dt0 = jnp.float64(
        first_step if first_step is not None else (times[-1] - times[0]) * 1e-6
    )
    bs = n if batch_size is None else batch_size
    n_chunks = (n + bs - 1) // bs
    return y0_arr, times, params_arr, n, n_vars, n_save, dt0, bs, n_chunks


def initial_history(y_init, n_save: int, n_vars: int):
    return jnp.zeros((n_save, n_vars), dtype=jnp.float64).at[0, :].set(y_init)


def error_norm(y, y_new, err_est, rtol, atol, error_weights):
    scale = atol + rtol * jnp.maximum(jnp.abs(y), jnp.abs(y_new))
    scaled_error = (err_est / scale) * error_weights
    denom = jnp.maximum(jnp.sum(error_weights != 0), 1)
    return jnp.sqrt(jnp.sum(scaled_error**2) / denom)


def error_norm_unweighted(y, y_new, err_est, rtol, atol):
    scale = atol + rtol * jnp.maximum(jnp.abs(y), jnp.abs(y_new))
    scaled_error = err_est / scale
    return jnp.sqrt(jnp.mean(scaled_error**2))


def build_error_weights(error_weights, n: int, n_vars: int):
    """Broadcast a user ``error_weights`` argument to ``(n, n_vars)``."""
    if error_weights is None:
        return jnp.ones((n, n_vars), dtype=jnp.float64)
    weights = jnp.asarray(error_weights, dtype=jnp.float64)
    if weights.ndim == 1:
        return jnp.broadcast_to(weights, (n, n_vars))
    return weights


def clamp_err_norm(err_norm, failed=False):
    """Clamp an error norm to a finite, strictly-positive range."""
    return jnp.where(
        failed | jnp.isnan(err_norm) | (err_norm > 1e18),
        1e18,
        jnp.where(err_norm == 0.0, 1e-18, err_norm),
    )


def step_size_factor(
    err_norm,
    err_prev=1.0,
    err_prev2=1.0,
    *,
    failed=False,
    exponent: float,
    pcoeff: float = 0.0,
    icoeff: float = 1.0,
    dcoeff: float = 0.0,
    safety: float,
    factor_min: float,
    factor_max: float,
):
    """PID step-size factor (Soderlind digital-filter form)."""
    safe_err = clamp_err_norm(err_norm, failed)
    if pcoeff == 0.0 and icoeff == 1.0 and dcoeff == 0.0:
        factor = safety * safe_err**exponent
        return jnp.clip(factor, factor_min, factor_max)
    e1 = exponent * (icoeff + pcoeff + dcoeff)
    e2 = -exponent * (pcoeff + 2.0 * dcoeff)
    e3 = exponent * dcoeff
    factor = safety * safe_err**e1 * err_prev**e2 * err_prev2**e3
    return jnp.clip(factor, factor_min, factor_max)


def build_batch_stats(trajectory_stats, *, n: int, n_chunks: int, batch_size: int):
    accepted_steps = trajectory_stats["accepted_steps"].reshape(n)
    rejected_steps = trajectory_stats["rejected_steps"].reshape(n)
    n_padded = n_chunks * batch_size
    pad_count = n_padded - n
    loop_steps_padded = jnp.pad(trajectory_stats["loop_steps"], (0, pad_count))
    loop_steps = loop_steps_padded.reshape(n_chunks, batch_size)
    valid_batches = (jnp.arange(n_padded) < n).reshape(n_chunks, batch_size)
    batch_loop_iterations = jnp.max(
        jnp.where(valid_batches, loop_steps, jnp.int32(0)), axis=1
    )
    valid_lanes = jnp.sum(valid_batches.astype(jnp.int32), axis=1)
    return {
        "accepted_steps": accepted_steps,
        "rejected_steps": rejected_steps,
        "batch_loop_iterations": batch_loop_iterations,
        "valid_lanes": valid_lanes,
    }


def _broadcast_for_vmap(arg, is_batched: bool, axis_size: int, name: str):
    arr = jnp.asarray(arg)
    if is_batched:
        if arr.ndim != 2:
            raise NotImplementedError(
                f"vmap over an already-ensembled {name} is not supported; "
                "call the solver with batched y0/params directly instead."
            )
        if arr.shape[0] != axis_size:
            raise ValueError(
                f"batched {name} has leading axis {arr.shape[0]}, expected {axis_size}"
            )
        return arr
    if arr.ndim != 1:
        raise NotImplementedError(
            f"vmap with unbatched ensemble-shaped {name} is not supported; "
            "call the solver with batched y0/params directly instead."
        )
    return jnp.broadcast_to(arr, (axis_size,) + arr.shape)


def _jax_stats_postprocess(stats, axis_size):
    """Default stats reshape for JAX-solver ``build_batch_stats`` output."""
    accepted = stats["accepted_steps"]
    rejected = stats["rejected_steps"]
    loop_steps = accepted + rejected
    stats_out = {
        "accepted_steps": accepted[:, None],
        "rejected_steps": rejected[:, None],
        "batch_loop_iterations": loop_steps[:, None],
        "valid_lanes": jnp.ones((axis_size, 1), dtype=stats["valid_lanes"].dtype),
    }
    stats_batched = jax.tree_util.tree_map(lambda _: True, stats_out)
    return stats_out, stats_batched


def per_trajectory_stats_postprocess(stats, axis_size):
    """Stats reshape when every key already has shape ``(axis_size,)``."""
    del axis_size
    stats_out = jax.tree_util.tree_map(lambda x: x[:, None], stats)
    stats_batched = jax.tree_util.tree_map(lambda _: True, stats_out)
    return stats_out, stats_batched


def make_custom_vmap_solver(
    solve_impl: Callable,
    *,
    return_stats: bool,
    stats_postprocess: Callable | None = None,
):
    """Wrap a solver implementation so outer ``jax.vmap`` becomes one ensemble call."""

    if stats_postprocess is None:
        stats_postprocess = _jax_stats_postprocess

    @custom_vmap
    def _solve(y0, t_span, params):
        return solve_impl(y0, t_span, params)

    @_solve.def_vmap
    def _solve_vmap(axis_size, in_batched, y0, t_span, params):
        y0_batched, t_span_batched, params_batched = in_batched
        if t_span_batched:
            t_span_arr = jnp.asarray(t_span)
            if t_span_arr.ndim != 2:
                raise NotImplementedError(
                    "vmap over nested t_span values is not supported; use a shared "
                    "t_span and vmap over y0 and/or params, or call the solver directly."
                )
            t_span = t_span_arr[0]

        y0_arr = _broadcast_for_vmap(y0, y0_batched, axis_size, "y0")
        params_arr = _broadcast_for_vmap(params, params_batched, axis_size, "params")
        result = solve_impl(y0_arr, t_span, params_arr)

        if not return_stats:
            return result[:, None, :, :], True

        sol, stats = result
        stats_out, stats_batched = stats_postprocess(stats, axis_size)
        return (sol[:, None, :, :], stats_out), (True, stats_batched)

    return _solve


def solve_adaptive_ensemble(
    *,
    params_arr,
    y0_arr,
    times,
    dt0,
    batch_size: int,
    n_chunks: int,
    rtol,
    atol,
    max_steps: int,
    return_stats: bool,
    step_factory: Callable,
    error_exponent: float,
    safety: float,
    factor_min: float,
    factor_max: float,
    pcoeff: float = 0.0,
    icoeff: float = 1.0,
    dcoeff: float = 0.0,
    error_weights_arr=None,
):
    n = y0_arr.shape[0]
    n_vars = y0_arr.shape[1]
    n_save = times.shape[0]
    tf = times[-1]

    def _solve_one(params_one, y0_one, error_weights_one=None):
        y_init = y0_one.copy()
        hist_init = initial_history(y_init, n_save, n_vars)
        step_info = step_factory(params_one)
        if len(step_info) == 3:
            step_one, extra_init, update_extra = step_info
            dense_eval = None
        else:
            step_one, extra_init, update_extra, dense_eval = step_info
        if callable(extra_init):
            extra_init = extra_init(y_init, times[0])

        def cond_fn(state):
            t, _, _, _, save_idx, n_steps, _, _, _, _, _ = state
            return (save_idx < n_save) & (t < tf) & (n_steps < max_steps)

        def body_fn(state):
            (
                t,
                y,
                dt,
                hist,
                save_idx,
                n_steps,
                accepted_steps,
                rejected_steps,
                extra,
                err_prev,
                err_prev2,
            ) = state
            if dense_eval is None:
                dt_limit = times[save_idx] - t
            else:
                dt_limit = tf - t
            dt_use = jnp.maximum(jnp.minimum(dt, dt_limit), 1e-30)

            step_out = step_one(y, t, dt_use, extra)
            if len(step_out) == 4:
                y_new, err_est, failed, extra_candidate = step_out
                dense_data = None
            else:
                y_new, err_est, failed, extra_candidate, dense_data = step_out
            if error_weights_one is None:
                err_norm = error_norm_unweighted(y, y_new, err_est, rtol, atol)
            else:
                err_norm = error_norm(y, y_new, err_est, rtol, atol, error_weights_one)

            accept = (err_norm <= 1.0) & ~jnp.isnan(err_norm) & ~failed
            t_new = jnp.where(accept, t + dt_use, t)
            y_out = jnp.where(accept, y_new, y)

            save_mask = (
                accept
                & (jnp.arange(n_save) >= save_idx)
                & (times <= t_new + 1e-12 * jnp.maximum(1.0, jnp.abs(times)))
            )

            if dense_eval is None:
                dense_values = jnp.broadcast_to(y_out, (n_save, n_vars))
            else:
                theta = (times - t) / dt_use
                dense_values = jax.lax.cond(
                    jnp.any(save_mask),
                    lambda _: dense_eval(theta, y, y_new, dense_data),
                    lambda _: hist,
                    operand=None,
                )
            hist_new = jnp.where(save_mask[:, None], dense_values, hist)
            save_count = jnp.sum(save_mask.astype(jnp.int32)).astype(jnp.int32)
            save_idx_new = save_idx + save_count

            factor = step_size_factor(
                err_norm,
                err_prev,
                err_prev2,
                failed=failed,
                exponent=error_exponent,
                pcoeff=pcoeff,
                icoeff=icoeff,
                dcoeff=dcoeff,
                safety=safety,
                factor_min=factor_min,
                factor_max=factor_max,
            )
            factor = jnp.where(
                accept,
                jnp.maximum(factor, 1.0),
                jnp.minimum(factor, safety),
            )
            dt_new = dt_use * factor
            rejected = ~accept
            extra_new = update_extra(extra, extra_candidate, accept)

            err_clamped = clamp_err_norm(err_norm, failed)
            err_prev_new = jnp.where(accept, err_clamped, err_prev)
            err_prev2_new = jnp.where(accept, err_prev, err_prev2)

            return (
                t_new,
                y_out,
                dt_new,
                hist_new,
                save_idx_new,
                n_steps + 1,
                accepted_steps + accept.astype(jnp.int32),
                rejected_steps + rejected.astype(jnp.int32),
                extra_new,
                err_prev_new,
                err_prev2_new,
            )

        init = (
            times[0],
            y_init,
            dt0,
            hist_init,
            jnp.int32(1),
            jnp.int32(0),
            jnp.int32(0),
            jnp.int32(0),
            extra_init,
            jnp.float64(1.0),
            jnp.float64(1.0),
        )
        (
            _,
            _,
            _,
            hist_final,
            _,
            loop_steps,
            accepted_steps,
            rejected_steps,
            _,
            _,
            _,
        ) = jax.lax.while_loop(cond_fn, body_fn, init)
        stats = {
            "accepted_steps": accepted_steps,
            "rejected_steps": rejected_steps,
            "loop_steps": loop_steps,
        }
        return hist_final, stats

    results, trajectory_stats = jax.lax.map(
        lambda xs: _solve_one(*xs),
        (params_arr, y0_arr, error_weights_arr),
        batch_size=batch_size,
    )
    if not return_stats:
        return results
    return results, build_batch_stats(
        trajectory_stats, n=n, n_chunks=n_chunks, batch_size=batch_size
    )


# =============================================================================
# Rodas5P W-transformed tableau (Steinebach 2023, BIT 63:27)
# =============================================================================

# fmt: off
_gamma = 0.21193756319429014

_a21 = 3.0
_a31 = 2.849394379747939;  _a32 = 0.45842242204463923
_a41 = -6.954028509809101; _a42 = 2.489845061869568;   _a43 = -10.358996098473584
_a51 = 2.8029986275628964; _a52 = 0.5072464736228206;  _a53 = -0.3988312541770524; _a54 = -0.04721187230404641
_a61 = -7.502846399306121; _a62 = 2.561846144803919;   _a63 = -11.627539656261098; _a64 = -0.18268767659942256; _a65 = 0.030198172008377946

_C21 = -14.155112264123755
_C31 = -17.97296035885952; _C32 = -2.859693295451294
_C41 = 147.12150275711716; _C42 = -1.41221402718213;    _C43 = 71.68940251302358
_C51 = 165.43517024871676; _C52 = -0.4592823456491126;  _C53 = 42.90938336958603;   _C54 = -5.961986721573306
_C61 = 24.854864614690072; _C62 = -3.0009227002832186;  _C63 = 47.4931110020768;     _C64 = 5.5814197821558125;  _C65 = -0.6610691825249471
_C71 = 30.91273214028599;  _C72 = -3.1208243349937974;  _C73 = 77.79954646070892;    _C74 = 34.28646028294783;   _C75 = -19.097331116725623; _C76 = -28.087943162872662
_C81 = 37.80277123390563;  _C82 = -3.2571969029072276;  _C83 = 112.26918849496327;   _C84 = 66.9347231244047;    _C85 = -40.06618937091002;  _C86 = -54.66780262877968;  _C87 = -9.48861652309627

_c2 = 0.6358126895828704
_c3 = 0.4095798393397535
_c4 = 0.9769306725060716
_c5 = 0.4288403609558664

_d1 = _gamma
_d2 = -0.42387512638858027
_d3 = -0.3384627126235924
_d4 =  1.8046452872882734
_d5 =  2.325825639765069

_H_DENSE = jnp.array(
    [
        [
            25.948786856663858, -2.5579724845846235, 10.433815404888879,
            -2.3679251022685204, 0.524948541321073, 1.1241088310450404,
            0.4272876194431874, -0.17202221070155493,
        ],
        [
            -9.91568850695171, -0.9689944594115154, 3.0438037242978453,
            -24.495224566215796, 20.176138334709044, 15.98066361424651,
            -6.789040303419874, -6.710236069923372,
        ],
        [
            11.419903575922262, 2.8879645146136994, 72.92137995996029,
            80.12511834622643, -52.072871366152654, -59.78993625266729,
            -0.15582684282751913, 4.883087185713722,
        ],
    ],
    dtype=jnp.float64,
)
# fmt: on


# =============================================================================
# rodas5Pjax: pure-JAX Rodas5P ensemble solver (ode_fn path)
# =============================================================================


def rodas5Pjax_solve(
    ode_fn,
    y0,
    t_span,
    params,
    *,
    custom_lu_solver=None,
    lu_precision: Literal["fp32", "fp64"] = "fp64",
    batch_size=None,
    rtol=1e-8,
    atol=1e-10,
    first_step=None,
    max_steps=100000,
    return_stats=False,
    error_weights=None,
    pcoeff=0.0,
    icoeff=1.0,
    dcoeff=0.0,
):
    """Rodas5P ensemble solver for nonlinear ODEs (pure JAX).

    Parameters
    ----------
    ode_fn : callable
        ODE right-hand side with signature ``dy/dt = ode_fn(y, t, params)``.
    y0 : array, shape (n_vars,) or (N, n_vars)
        Initial state. A 1-D array is broadcast to all trajectories.
    t_span : array-like, shape (n_save,)
        Strictly-increasing 1-D array of save times (including t0).
    params : array, shape (n_params,) or (N, n_params)
        Parameters passed to ``ode_fn``.

    Returns
    -------
    array, shape (N, n_save, n_vars)
        Solution at each save time for each trajectory (``(solution, stats)`` if
        ``return_stats``).
    """

    def solve_impl(y0_arr, t_span_arr, params_arr):
        return _rodas5Pjax_solve_impl(
            ode_fn,
            y0_arr,
            t_span_arr,
            params_arr,
            custom_lu_solver=custom_lu_solver,
            lu_precision=lu_precision,
            batch_size=batch_size,
            rtol=rtol,
            atol=atol,
            first_step=first_step,
            max_steps=max_steps,
            return_stats=return_stats,
            error_weights=error_weights,
            pcoeff=pcoeff,
            icoeff=icoeff,
            dcoeff=dcoeff,
        )

    return make_custom_vmap_solver(solve_impl, return_stats=return_stats)(
        y0, t_span, params
    )


@functools.partial(
    jax.jit,
    static_argnames=(
        "ode_fn",
        "custom_lu_solver",
        "lu_precision",
        "batch_size",
        "max_steps",
        "return_stats",
        "pcoeff",
        "icoeff",
        "dcoeff",
    ),
)
def _rodas5Pjax_solve_impl(
    ode_fn,
    y0,
    t_span,
    params,
    *,
    custom_lu_solver=None,
    lu_precision: Literal["fp32", "fp64"] = "fp64",
    batch_size=None,
    rtol=1e-8,
    atol=1e-10,
    first_step=None,
    max_steps=100000,
    return_stats=False,
    error_weights=None,
    pcoeff=0.0,
    icoeff=1.0,
    dcoeff=0.0,
):
    lu_dtype = jnp.float32 if lu_precision == "fp32" else jnp.float64
    ode_eval = functools.partial(eval_ode_fn, ode_fn)
    jac_fn = jax.jacfwd(ode_eval, argnums=0)
    dT_fn = jax.jacfwd(ode_eval, argnums=1)

    y0_arr, times, params_arr, n, n_vars, _, dt0, bs, n_chunks = normalize_inputs(
        y0, t_span, params, first_step, batch_size
    )
    error_weights_arr = build_error_weights(error_weights, n, n_vars)

    eye = jnp.eye(n_vars, dtype=lu_dtype)

    def step_factory(params_one):
        def _step_one(y, t, dt, extra):
            del extra
            jac = jac_fn(y, t, params_one).astype(lu_dtype)
            dT = dT_fn(y, t, params_one)
            dtgamma_inv = (1.0 / (dt * _gamma)).astype(lu_dtype)
            matrix = dtgamma_inv * eye - jac
            if custom_lu_solver is not None:
                lu = custom_lu_solver.factorize(matrix)
            else:
                lu = jax.scipy.linalg.lu_factor(matrix)
            inv_dt = 1.0 / dt

            def f_eval(u, t_stage):
                return ode_eval(u, t_stage, params_one)

            def lu_solve(rhs):
                if custom_lu_solver is not None:
                    sol = custom_lu_solver.solve(lu, rhs.astype(lu_dtype))
                else:
                    sol = jax.scipy.linalg.lu_solve(lu, rhs.astype(lu_dtype))
                return sol.astype(jnp.float64)

            dy = f_eval(y, t)
            k1 = lu_solve(dy + dt * _d1 * dT)

            u = y + _a21 * k1
            du = f_eval(u, t + _c2 * dt)
            k2 = lu_solve(du + dt * _d2 * dT + _C21 * k1 * inv_dt)

            u = y + _a31 * k1 + _a32 * k2
            du = f_eval(u, t + _c3 * dt)
            k3 = lu_solve(du + dt * _d3 * dT + (_C31 * k1 + _C32 * k2) * inv_dt)

            u = y + _a41 * k1 + _a42 * k2 + _a43 * k3
            du = f_eval(u, t + _c4 * dt)
            k4 = lu_solve(
                du + dt * _d4 * dT + (_C41 * k1 + _C42 * k2 + _C43 * k3) * inv_dt
            )

            u = y + _a51 * k1 + _a52 * k2 + _a53 * k3 + _a54 * k4
            du = f_eval(u, t + _c5 * dt)
            k5 = lu_solve(
                du
                + dt * _d5 * dT
                + (_C51 * k1 + _C52 * k2 + _C53 * k3 + _C54 * k4) * inv_dt
            )

            t_end = t + dt
            u = y + _a61 * k1 + _a62 * k2 + _a63 * k3 + _a64 * k4 + _a65 * k5
            du = f_eval(u, t_end)
            k6 = lu_solve(
                du
                + (_C61 * k1 + _C62 * k2 + _C63 * k3 + _C64 * k4 + _C65 * k5) * inv_dt
            )

            u = u + k6
            du = f_eval(u, t_end)
            k7 = lu_solve(
                du
                + (
                    _C71 * k1
                    + _C72 * k2
                    + _C73 * k3
                    + _C74 * k4
                    + _C75 * k5
                    + _C76 * k6
                )
                * inv_dt
            )

            u = u + k7
            du = f_eval(u, t_end)
            k8 = lu_solve(
                du
                + (
                    _C81 * k1
                    + _C82 * k2
                    + _C83 * k3
                    + _C84 * k4
                    + _C85 * k5
                    + _C86 * k6
                    + _C87 * k7
                )
                * inv_dt
            )

            y_new = u + k8
            dense_stages = (k1, k2, k3, k4, k5, k6, k7, k8)
            dense_coeffs = []
            for row in range(3):
                accum = jnp.zeros_like(y)
                for col, stage in enumerate(dense_stages):
                    accum = accum + _H_DENSE[row, col] * stage
                dense_coeffs.append(accum)
            return y_new, k8, jnp.bool_(False), (), tuple(dense_coeffs)

        def dense_eval(theta, y, y_new, dense_data):
            h1, h2, h3 = dense_data
            theta1 = 1.0 - theta
            return (
                theta1[:, None] * y
                + theta[:, None]
                * (
                    y_new
                    + theta1[:, None]
                    * (h1 + theta[:, None] * (h2 + theta[:, None] * h3))
                )
            )

        return _step_one, (), lambda extra, candidate, accept: extra, dense_eval

    return solve_adaptive_ensemble(
        params_arr=params_arr,
        y0_arr=y0_arr,
        times=times,
        dt0=dt0,
        batch_size=bs,
        n_chunks=n_chunks,
        rtol=rtol,
        atol=atol,
        max_steps=max_steps,
        return_stats=return_stats,
        step_factory=step_factory,
        error_exponent=-1.0 / 6.0,
        safety=0.9,
        factor_min=0.2,
        factor_max=6.0,
        pcoeff=pcoeff,
        icoeff=icoeff,
        dcoeff=dcoeff,
        error_weights_arr=error_weights_arr,
    )
