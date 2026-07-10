"""Adaptive Rodas5P ODE integrators.

Two distinct implementations of the Rodas5P stiff Rosenbrock-W method, ported
from the DISCO2 prototype:

* :func:`rodas5Pjax_solve` -- a pure-JAX ensemble solver (``ode_fn`` path) whose
  Jacobian is recomputed each step via ``jax.jacfwd``. It is fully traceable /
  ``jit``-able / ``vmap``-able and is used by :mod:`discoeb.background_system`.

* :func:`rodas5Pnumba_solve` -- the numba-CUDA batch solver. It requires a CUDA
  GPU and the ``numba`` package; its ``numba``/CUDA imports are performed lazily
  so this module imports cleanly on machines without ``numba``.

Both use the Rodas5P W-transformed tableau (Steinebach 2023, BIT 63:27), matching
the Julia ``Rodas5P`` / ``GPURodas5P`` coefficients.

The pure-JAX scaffolding (ensemble driver, PID step control, custom-vmap wrapper)
is inlined here from the DISCO2 ``_jax_common`` module so this file is a single
self-contained home for both integrators.
"""

from __future__ import annotations

import functools
from typing import Callable, Literal

import jax
import jax.numpy as jnp
from jax.custom_batching import custom_vmap


# =============================================================================
# Shared pure-JAX scaffolding (ported from DISCO2 ``src/_jax_common.py``)
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


# =============================================================================
# rodas5Pnumba: numba-CUDA Rodas5P batch solver
#
# Faithful port of the DISCO2 numba-CUDA Rodas5P solver (``src/rodas5P.py`` plus
# its ``src/_numba_common.py`` and ``src/_jax_numba_custom_call.py`` helpers).
#
# This path requires:
#   * the ``numba`` package with CUDA support and a visible CUDA GPU;
#   * a working ``g++`` (the XLA FFI launcher shim is compiled at first use);
#   * a thread-local ``custom_lu_solver`` (e.g. a Schur/EB LU solver) that
#     implements ``assemble_lu_local_factory`` / ``factorize_local`` /
#     ``solve_local`` / ``a_local_size`` / ``b_local_size`` / ``ipiv_size`` /
#     ``batches_per_block`` / ``block_dim`` (as in DISCO2).
#
# All ``numba`` imports are performed lazily (inside the functions that need
# them) so ``import discoeb.integrators`` — and therefore the pure-JAX
# ``rodas5Pjax_solve`` path — works on machines without ``numba`` installed.
# =============================================================================

import math as _math
import os as _os
from dataclasses import dataclass as _dataclass
from typing import Any as _Any

import numpy as _np


# --- Rodas5P tableau (uppercase names used by the numba kernel) --------------
# fmt: off
GAMMA = 0.21193756319429014
A21 = 3.0
A31 = 2.849394379747939;   A32 = 0.45842242204463923
A41 = -6.954028509809101;  A42 = 2.489845061869568;   A43 = -10.358996098473584
A51 = 2.8029986275628964;  A52 = 0.5072464736228206;  A53 = -0.3988312541770524; A54 = -0.04721187230404641
A61 = -7.502846399306121;  A62 = 2.561846144803919;   A63 = -11.627539656261098; A64 = -0.18268767659942256; A65 = 0.030198172008377946
C21 = -14.155112264123755
C31 = -17.97296035885952;  C32 = -2.859693295451294
C41 = 147.12150275711716;  C42 = -1.41221402718213;   C43 = 71.68940251302358
C51 = 165.43517024871676;  C52 = -0.4592823456491126; C53 = 42.90938336958603;   C54 = -5.961986721573306
C61 = 24.854864614690072;  C62 = -3.0009227002832186; C63 = 47.4931110020768;    C64 = 5.5814197821558125;  C65 = -0.6610691825249471
C71 = 30.91273214028599;   C72 = -3.1208243349937974; C73 = 77.79954646070892;   C74 = 34.28646028294783;   C75 = -19.097331116725623; C76 = -28.087943162872662
C81 = 37.80277123390563;   C82 = -3.2571969029072276; C83 = 112.26918849496327;  C84 = 66.9347231244047;    C85 = -40.06618937091002;  C86 = -54.66780262877968; C87 = -9.48861652309627
C2 = 0.6358126895828704
C3 = 0.4095798393397535
C4 = 0.9769306725060716
C5 = 0.4288403609558664
D1 = GAMMA
D2 = -0.42387512638858027
D3 = -0.3384627126235924
D4 =  1.8046452872882734
D5 =  2.325825639765069
# fmt: on

SAFETY = 0.9
FACTOR_MIN = 0.2
FACTOR_MAX = 6.0
EXPONENT = -1.0 / 6.0
CUDA_MAX_REGISTERS = int(_os.environ.get("DISCO_CUDA_MAX_REGISTERS", "168"))

_WORKSPACE_CACHE: dict = {}

# XLA-FFI numba-CUDA ABI argument kinds.
ABI_ARRAY = 0
ABI_SCALAR_F64 = 1
ABI_SCALAR_I32 = 2
ABI_RAW_PTR = 3

_CAPSULE_NAME = b"xla._CUSTOM_CALL_TARGET"
_TARGET_NAME = "discoeb_numba_cuda_launch"
_CUSTOM_CALL_API_VERSION = 4
_REGISTERED = False
_LOADED_LIB = None


# --- numba-CUDA host helpers (ported from DISCO2 ``_numba_common.py``) -------


@_dataclass
class NumbaWorkspace:
    y0_dev: _Any
    times_dev: _Any
    params_dev: _Any
    hist_dev: _Any
    accepted_dev: _Any
    rejected_dev: _Any
    loop_dev: _Any


@_dataclass(frozen=True)
class PreparedNumbaSolve:
    kernel: _Any
    workspace: _Any
    dt0: _Any
    rtol: _Any
    atol: _Any
    max_steps: _Any
    blocks: int
    threads: _Any


def _numba_normalize_inputs(y0, t_span, params, first_step, *, solver_name: str):
    y0_in = _np.asarray(y0, dtype=_np.float64)
    params_arr = _np.asarray(params, dtype=_np.float64)
    times = _np.asarray(t_span, dtype=_np.float64)

    if y0_in.ndim == 1 and params_arr.ndim == 1:
        n = 1
        y0_arr = _np.broadcast_to(y0_in, (n, y0_in.shape[0])).copy()
        params_arr = _np.broadcast_to(params_arr, (n, params_arr.shape[0])).copy()
    elif y0_in.ndim == 1:
        n = params_arr.shape[0]
        y0_arr = _np.broadcast_to(y0_in, (n, y0_in.shape[0])).copy()
    else:
        n = y0_in.shape[0]
        y0_arr = y0_in
        if params_arr.ndim == 1:
            params_arr = _np.broadcast_to(params_arr, (n, params_arr.shape[0])).copy()
        elif params_arr.shape[0] != n:
            raise ValueError(
                "params must have shape (n_params,) or (N, n_params) when y0 has "
                f"shape (N, n_vars); got y0.shape={y0_in.shape} and "
                f"params.shape={params_arr.shape}"
            )

    if y0_arr.ndim != 2:
        raise ValueError(f"custom-kernel {solver_name} expects y0 shape (N, n_vars)")
    if params_arr.ndim != 2:
        raise ValueError(
            f"custom-kernel {solver_name} expects params shape (N, n_params)"
        )
    if times.ndim != 1 or times.shape[0] < 2:
        raise ValueError("t_span must be a 1-D array with at least two save times")
    if _np.any(_np.diff(times) <= 0.0):
        raise ValueError("t_span must be strictly increasing")

    return y0_arr, times, params_arr, initial_step(times, first_step)


def _numba_build_error_weights(error_weights, n: int, n_vars: int):
    """NumPy per-component error weights for the numba kernel (see JAX variant)."""
    if error_weights is None:
        return _np.ones((n, n_vars), dtype=_np.float64)
    weights = _np.asarray(error_weights, dtype=_np.float64)
    if weights.ndim == 1:
        weights = _np.broadcast_to(weights, (n, n_vars))
    return _np.ascontiguousarray(weights, dtype=_np.float64)


def initial_step(times, first_step):
    return (
        _np.float64(first_step)
        if first_step is not None
        else _np.float64((times[-1] - times[0]) * 1e-6)
    )


def copy_workspace_inputs(workspace, y0_arr, times, params_arr):
    workspace.y0_dev.copy_to_device(y0_arr)
    workspace.times_dev.copy_to_device(times)
    workspace.params_dev.copy_to_device(params_arr)


def numpy_stats(accepted_steps, rejected_steps, loop_steps):
    return {
        "accepted_steps": accepted_steps,
        "rejected_steps": rejected_steps,
        "loop_steps": loop_steps,
        "batch_loop_iterations": loop_steps,
        "valid_lanes": _np.ones_like(loop_steps, dtype=_np.int32),
    }


def jax_stats(accepted, rejected, loop_steps):
    return {
        "accepted_steps": accepted,
        "rejected_steps": rejected,
        "loop_steps": loop_steps,
        "batch_loop_iterations": loop_steps,
        "valid_lanes": jnp.ones_like(loop_steps, dtype=jnp.int32),
    }


def as_launch_block_dim(block_dim):
    if isinstance(block_dim, int):
        return block_dim
    if isinstance(block_dim, (tuple, list)):
        return tuple(int(x) for x in block_dim)
    x = getattr(block_dim, "x", None)
    y = getattr(block_dim, "y", 1)
    z = getattr(block_dim, "z", 1)
    if x is not None:
        return (int(x), int(y), int(z))
    return 128


@functools.cache
def as_cuda_device(fn):
    from numba import cuda
    from numba.cuda.dispatcher import CUDADispatcher

    if isinstance(fn, CUDADispatcher):
        return fn
    return cuda.jit(device=True)(fn)


# --- XLA-FFI numba-CUDA launch bridge (ported from ``_jax_numba_custom_call``) --


@_dataclass(frozen=True)
class CudaLaunch:
    """Compiled CUDA kernel launch metadata for an XLA FFI call."""

    function: int
    grid: tuple
    block: tuple
    shared_mem: int = 0


def _as_3d(value):
    if isinstance(value, int):
        return (int(value), 1, 1)
    parts = tuple(int(v) for v in value)
    if len(parts) == 1:
        return (parts[0], 1, 1)
    if len(parts) == 2:
        return (parts[0], parts[1], 1)
    if len(parts) == 3:
        return parts
    raise ValueError(f"launch dimensions must have rank 1, 2, or 3; got {value!r}")


def _pycapsule_new(ptr_value, name=_CAPSULE_NAME):
    import ctypes

    ctypes.pythonapi.PyCapsule_New.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_void_p,
    ]
    ctypes.pythonapi.PyCapsule_New.restype = ctypes.py_object
    return ctypes.pythonapi.PyCapsule_New(ctypes.c_void_p(ptr_value), name, None)


def _ffi_bridge_source() -> str:
    return r"""
#include <cstdint>
#include <dlfcn.h>
#include <mutex>
#include <string>
#include <vector>

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

using CuLaunchKernel = int (*)(void*, unsigned int, unsigned int, unsigned int,
                               unsigned int, unsigned int, unsigned int,
                               unsigned int, void*, void**, void**);

static CuLaunchKernel LoadCuLaunchKernel() {
  static std::once_flag once;
  static CuLaunchKernel fn = nullptr;
  std::call_once(once, []() {
    void* lib = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    if (lib == nullptr) return;
    fn = reinterpret_cast<CuLaunchKernel>(dlsym(lib, "cuLaunchKernel"));
  });
  return fn;
}

static ffi::Error LaunchNumbaCuda(
    void* stream, int64_t function, int64_t grid_x, int64_t grid_y,
    int64_t grid_z, int64_t block_x, int64_t block_y, int64_t block_z,
    int64_t shared_mem, ffi::RemainingArgs args, ffi::RemainingRets rets) {
  CuLaunchKernel cuLaunchKernel = LoadCuLaunchKernel();
  if (cuLaunchKernel == nullptr) {
    return ffi::Error(ffi::ErrorCode::kInternal,
                      "could not load cuLaunchKernel from libcuda.so.1");
  }

  std::vector<void*> arg_values;
  arg_values.reserve(args.size() + rets.size());
  for (size_t i = 0; i < args.size(); ++i) {
    auto arg = args.get<ffi::AnyBuffer>(i);
    if (!arg.has_value()) return arg.error();
    arg_values.push_back(arg.value().untyped_data());
  }
  for (size_t i = 0; i < rets.size(); ++i) {
    auto ret = rets.get<ffi::AnyBuffer>(i);
    if (!ret.has_value()) return ret.error();
    arg_values.push_back(ret.value()->untyped_data());
  }

  std::vector<void*> params;
  params.reserve(arg_values.size());
  for (void*& value : arg_values) {
    params.push_back(&value);
  }

  int err = cuLaunchKernel(reinterpret_cast<void*>(function),
                           static_cast<unsigned int>(grid_x),
                           static_cast<unsigned int>(grid_y),
                           static_cast<unsigned int>(grid_z),
                           static_cast<unsigned int>(block_x),
                           static_cast<unsigned int>(block_y),
                           static_cast<unsigned int>(block_z),
                           static_cast<unsigned int>(shared_mem),
                           stream, params.data(), nullptr);
  if (err != 0) {
    return ffi::Error(ffi::ErrorCode::kInternal,
                      "cuLaunchKernel failed with CUDA driver error " +
                          std::to_string(err));
  }
  return ffi::Error::Success();
}

struct ArrayArg {
  void* meminfo = nullptr;
  void* parent = nullptr;
  int64_t nitems = 0;
  int64_t itemsize = 0;
  void* data = nullptr;
  std::vector<int64_t> dims;
  std::vector<int64_t> strides;
};

struct KernelArgStorage {
  ArrayArg array;
  double f64 = 0.0;
  int32_t i32 = 0;
  void* ptr = nullptr;
};

static void AddArrayParams(ffi::AnyBuffer buf, KernelArgStorage& storage,
                           std::vector<void*>& params) {
  storage.array.nitems = static_cast<int64_t>(buf.element_count());
  storage.array.itemsize = static_cast<int64_t>(ffi::ByteWidth(buf.element_type()));
  storage.array.data = buf.untyped_data();
  auto dims = buf.dimensions();
  storage.array.dims.assign(dims.begin(), dims.end());
  storage.array.strides.resize(storage.array.dims.size());
  int64_t stride = storage.array.itemsize;
  for (int64_t i = static_cast<int64_t>(storage.array.dims.size()) - 1; i >= 0; --i) {
    storage.array.strides[static_cast<size_t>(i)] = stride;
    stride *= storage.array.dims[static_cast<size_t>(i)];
  }

  params.push_back(&storage.array.meminfo);
  params.push_back(&storage.array.parent);
  params.push_back(&storage.array.nitems);
  params.push_back(&storage.array.itemsize);
  params.push_back(&storage.array.data);
  for (int64_t& dim : storage.array.dims) params.push_back(&dim);
  for (int64_t& stride_value : storage.array.strides) params.push_back(&stride_value);
}

static ffi::Error AddBufferParam(ffi::AnyBuffer buf, int64_t kind,
                                 KernelArgStorage& storage,
                                 std::vector<void*>& params) {
  switch (kind) {
    case 0:
      AddArrayParams(buf, storage, params);
      return ffi::Error::Success();
    case 3:
      storage.ptr = buf.untyped_data();
      params.push_back(&storage.ptr);
      return ffi::Error::Success();
    default:
      return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                        "unknown Numba CUDA ABI argument kind");
  }
}

static ffi::Error LaunchNumbaCudaAbi(
    void* stream, int64_t function, int64_t grid_x, int64_t grid_y,
    int64_t grid_z, int64_t block_x, int64_t block_y, int64_t block_z,
    int64_t shared_mem, ffi::Span<const int64_t> arg_kinds,
    ffi::Span<const double> scalar_f64_values,
    ffi::Span<const int32_t> scalar_i32_values,
    ffi::RemainingArgs args, ffi::RemainingRets rets) {
  CuLaunchKernel cuLaunchKernel = LoadCuLaunchKernel();
  if (cuLaunchKernel == nullptr) {
    return ffi::Error(ffi::ErrorCode::kInternal,
                      "could not load cuLaunchKernel from libcuda.so.1");
  }
  std::vector<KernelArgStorage> storage(arg_kinds.size());
  std::vector<void*> params;
  params.reserve(arg_kinds.size() * 12);
  size_t arg_idx = 0;
  size_t ret_idx = 0;
  size_t f64_idx = 0;
  size_t i32_idx = 0;

  for (size_t i = 0; i < arg_kinds.size(); ++i) {
    const int64_t kind = arg_kinds[i];
    if (kind == 1) {
      if (f64_idx >= scalar_f64_values.size()) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "not enough f64 scalar values");
      }
      storage[i].f64 = scalar_f64_values[f64_idx++];
      params.push_back(&storage[i].f64);
    } else if (kind == 2) {
      if (i32_idx >= scalar_i32_values.size()) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "not enough i32 scalar values");
      }
      storage[i].i32 = scalar_i32_values[i32_idx++];
      params.push_back(&storage[i].i32);
    } else if (arg_idx < args.size()) {
      auto arg = args.get<ffi::AnyBuffer>(arg_idx++);
      if (!arg.has_value()) return arg.error();
      ffi::Error err = AddBufferParam(arg.value(), kind, storage[i], params);
      if (!err.success()) return err;
    } else {
      auto ret = rets.get<ffi::AnyBuffer>(ret_idx++);
      if (!ret.has_value()) return ret.error();
      ffi::Error err = AddBufferParam(*ret.value(), kind, storage[i], params);
      if (!err.success()) return err;
    }
  }
  if (arg_idx != args.size() || ret_idx != rets.size()) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "kernel ABI kinds did not consume all buffers");
  }

  int err = cuLaunchKernel(reinterpret_cast<void*>(function),
                           static_cast<unsigned int>(grid_x),
                           static_cast<unsigned int>(grid_y),
                           static_cast<unsigned int>(grid_z),
                           static_cast<unsigned int>(block_x),
                           static_cast<unsigned int>(block_y),
                           static_cast<unsigned int>(block_z),
                           static_cast<unsigned int>(shared_mem),
                           stream, params.data(), nullptr);
  if (err != 0) {
    return ffi::Error(ffi::ErrorCode::kInternal,
                      "cuLaunchKernel failed with CUDA driver error " +
                          std::to_string(err));
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    discoeb_numba_cuda_launch, LaunchNumbaCuda,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<void*>>()
        .Attr<int64_t>("function")
        .Attr<int64_t>("grid_x")
        .Attr<int64_t>("grid_y")
        .Attr<int64_t>("grid_z")
        .Attr<int64_t>("block_x")
        .Attr<int64_t>("block_y")
        .Attr<int64_t>("block_z")
        .Attr<int64_t>("shared_mem")
        .RemainingArgs()
        .RemainingRets());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    discoeb_numba_cuda_abi_launch, LaunchNumbaCudaAbi,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<void*>>()
        .Attr<int64_t>("function")
        .Attr<int64_t>("grid_x")
        .Attr<int64_t>("grid_y")
        .Attr<int64_t>("grid_z")
        .Attr<int64_t>("block_x")
        .Attr<int64_t>("block_y")
        .Attr<int64_t>("block_z")
        .Attr<int64_t>("shared_mem")
        .Attr<ffi::Span<const int64_t>>("arg_kinds")
        .Attr<ffi::Span<const double>>("scalar_f64_values")
        .Attr<ffi::Span<const int32_t>>("scalar_i32_values")
        .RemainingArgs()
        .RemainingRets());
"""


def _build_ffi_bridge():
    import hashlib
    import subprocess
    import sysconfig
    import tempfile
    from pathlib import Path

    include_dir = Path(jax.ffi.include_dir())
    build_dir = Path(tempfile.gettempdir()) / "discoeb_jax_numba_cuda_bridge"
    build_dir.mkdir(parents=True, exist_ok=True)
    source = _ffi_bridge_source()
    digest = hashlib.sha256(source.encode()).hexdigest()[:16]
    src_path = build_dir / f"bridge-{digest}.cc"
    so_path = build_dir / f"bridge-{digest}{sysconfig.get_config_var('EXT_SUFFIX')}"
    if so_path.exists():
        return so_path
    src_path.write_text(source)
    cmd = [
        "g++",
        "-std=c++17",
        "-shared",
        "-fPIC",
        "-O2",
        f"-I{include_dir}",
        str(src_path),
        "-ldl",
        "-o",
        str(so_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return so_path


def register_target() -> None:
    """Register the generic CUDA launcher with JAX once per process."""

    import ctypes

    global _LOADED_LIB, _REGISTERED
    if _REGISTERED:
        return
    so_path = _build_ffi_bridge()
    _LOADED_LIB = ctypes.CDLL(str(so_path))
    symbol = getattr(_LOADED_LIB, _TARGET_NAME)
    capsule = _pycapsule_new(ctypes.cast(symbol, ctypes.c_void_p).value)
    jax.ffi.register_ffi_target(_TARGET_NAME, capsule, platform="CUDA", api_version=1)
    symbol = getattr(_LOADED_LIB, "discoeb_numba_cuda_abi_launch")
    capsule = _pycapsule_new(ctypes.cast(symbol, ctypes.c_void_p).value)
    jax.ffi.register_ffi_target(
        "discoeb_numba_cuda_abi_launch", capsule, platform="CUDA", api_version=1
    )
    _REGISTERED = True


def compile_raw_pointer_kernel(kernel, argtypes) -> int:
    """Compile a ``cuda.jit`` kernel and return its legacy ``CUfunction`` pointer."""

    import ctypes

    compiled = kernel.compile(tuple(argtypes))
    cufunc = compiled.library.get_cufunc()
    kernel_handle = int(cufunc.handle)
    libcuda = ctypes.CDLL("libcuda.so.1")
    cu_kernel_get_function = libcuda.cuKernelGetFunction
    cu_kernel_get_function.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
    cu_kernel_get_function.restype = ctypes.c_int
    function = ctypes.c_void_p()
    err = cu_kernel_get_function(ctypes.byref(function), ctypes.c_void_p(kernel_handle))
    if err != 0:
        raise RuntimeError(f"cuKernelGetFunction failed with CUDA driver error {err}")
    if function.value is None:
        raise RuntimeError("cuKernelGetFunction returned a null function pointer")
    return int(function.value)


def make_launch(kernel, argtypes, *, grid, block, shared_mem: int = 0) -> CudaLaunch:
    return CudaLaunch(
        function=compile_raw_pointer_kernel(kernel, argtypes),
        grid=_as_3d(grid),
        block=_as_3d(block),
        shared_mem=int(shared_mem),
    )


def ffi_abi_call(
    launch,
    inputs,
    output_specs,
    *,
    input_kinds,
    output_kinds,
    scalar_f64_values=(),
    scalar_i32_values=(),
):
    """Launch a Numba CUDA kernel using Numba's normal array/scalar ABI."""

    register_target()
    attrs = {
        "function": _np.int64(launch.function),
        "grid_x": _np.int64(launch.grid[0]),
        "grid_y": _np.int64(launch.grid[1]),
        "grid_z": _np.int64(launch.grid[2]),
        "block_x": _np.int64(launch.block[0]),
        "block_y": _np.int64(launch.block[1]),
        "block_z": _np.int64(launch.block[2]),
        "shared_mem": _np.int64(launch.shared_mem),
        "arg_kinds": _np.asarray(
            tuple(input_kinds) + tuple(output_kinds), dtype=_np.int64
        ),
        "scalar_f64_values": _np.asarray(tuple(scalar_f64_values), dtype=_np.float64),
        "scalar_i32_values": _np.asarray(tuple(scalar_i32_values), dtype=_np.int32),
    }
    result = jax.ffi.ffi_call(
        "discoeb_numba_cuda_abi_launch",
        tuple(output_specs),
        has_side_effect=False,
        custom_call_api_version=_CUSTOM_CALL_API_VERSION,
    )(*inputs, **attrs)
    if not isinstance(result, tuple):
        return (result,)
    return result


# --- Rodas5P numba kernel and solver (ported from DISCO2 ``src/rodas5P.py``) --


@_dataclass
class Workspace(NumbaWorkspace):
    weights_dev: _Any


@_dataclass(frozen=True)
class PreparedSolve(PreparedNumbaSolve):
    kernel: _Any
    lu_solver: _Any
    workspace: Workspace


def get_workspace(cache, n, n_vars, n_save, n_params):
    from numba import cuda

    key = (n, n_vars, n_save, n_params)
    workspace = cache.get(key)
    if workspace is not None:
        return workspace

    workspace = Workspace(
        y0_dev=cuda.device_array((n, n_vars), dtype=_np.float64),
        times_dev=cuda.device_array(n_save, dtype=_np.float64),
        params_dev=cuda.device_array((n, n_params), dtype=_np.float64),
        hist_dev=cuda.device_array((n, n_save, n_vars), dtype=_np.float64),
        accepted_dev=cuda.device_array(n, dtype=_np.int32),
        rejected_dev=cuda.device_array(n, dtype=_np.int32),
        loop_dev=cuda.device_array(n, dtype=_np.int32),
        weights_dev=cuda.device_array((n, n_vars), dtype=_np.float64),
    )
    cache[key] = workspace
    return workspace


def _make_numba_kernel(
    ode_fn,
    jac_fn,
    time_jac_fn,
    n_vars: int,
    pcoeff: float = 0.0,
    icoeff: float = 1.0,
    dcoeff: float = 0.0,
    lu_precision: str = "fp32",
    batches_per_block="suggested",
    custom_lu_solver=None,
    tf_local_idx: int = -1,
):
    from numba import cuda

    e1 = EXPONENT * (icoeff + pcoeff + dcoeff)
    e2 = -EXPONENT * (pcoeff + 2.0 * dcoeff)
    e3 = EXPONENT * dcoeff
    lu_dtype = _np.float32 if lu_precision == "fp32" else _np.float64
    jac_device = as_cuda_device(jac_fn)

    if custom_lu_solver is None:
        raise NotImplementedError(
            "The thread-local Rodas5P kernel requires a thread-local LU solver "
            "(e.g. a Schur/EB LU solver) passed as custom_lu_solver."
        )
    lu_solver = custom_lu_solver

    assemble_lu_fn = lu_solver.assemble_lu_local_factory(jac_device, n_vars, lu_dtype)
    factorize_fn = lu_solver.factorize_local
    solve_fn = lu_solver.solve_local

    ode_device = as_cuda_device(ode_fn)

    @cuda.jit(device=True)
    def ode_write(y_local, t, p_row, out_local):
        ode_device(y_local, t, p_row, out_local)

    if time_jac_fn is None:

        @cuda.jit(device=True)
        def time_jac_write(y_local, t, p_row, out_local):
            for j in range(n_vars):
                out_local[j] = 0.0
    else:
        time_jac_device = as_cuda_device(time_jac_fn)

        @cuda.jit(device=True)
        def time_jac_write(y_local, t, p_row, out_local):
            time_jac_device(y_local, t, p_row, out_local)

    a_local = int(lu_solver.a_local_size())
    b_local = int(lu_solver.b_local_size())
    ipiv_local = int(lu_solver.ipiv_size)

    @cuda.jit(max_registers=CUDA_MAX_REGISTERS)
    def kernel(
        y0,
        times,
        params,
        dt0,
        rtol,
        atol,
        max_steps,
        weights,
        hist,
        accepted_out,
        rejected_out,
        loop_out,
    ):
        i = cuda.grid(1)
        if i >= y0.shape[0]:
            return

        n_save = times.shape[0]
        tf = times[n_save - 1]
        p_row = params[i]

        y = cuda.local.array(n_vars, _np.float64)
        u = cuda.local.array(n_vars, _np.float64)
        work = cuda.local.array(n_vars, _np.float64)
        dT = cuda.local.array(n_vars, _np.float64)
        k_stages = cuda.local.array((8, n_vars), lu_dtype)
        lu = cuda.local.array(a_local, lu_dtype)
        rhs = cuda.local.array(b_local, lu_dtype)
        ipiv = cuda.local.array(ipiv_local, _np.int32)

        for j in range(n_vars):
            val = y0[i, j]
            y[j] = val
            hist[i, 0, j] = val

        t = times[0]
        save_idx = 1
        dt = dt0
        n_steps = 0
        accepted = 0
        rejected = 0
        err_prev = 1.0
        err_prev2 = 1.0

        tf_local = tf
        if tf_local_idx >= 0:
            tf_local = p_row[tf_local_idx]

        while save_idx < n_save and t < tf_local and n_steps < max_steps:
            dt_use = dt
            if dt_use > tf_local - t:
                dt_use = tf_local - t
            if dt_use < 1e-30:
                dt_use = 1e-30
            inv_dt = 1.0 / dt_use
            t_end = t + dt_use

            dtgamma_inv = 1.0 / (dt_use * GAMMA)
            assemble_lu_fn(y, t, p_row, lu, dtgamma_inv)
            time_jac_write(y, t, p_row, dT)
            factorize_fn(lu, ipiv)

            # Stage 1
            ode_write(y, t, p_row, work)
            for j in range(n_vars):
                rhs[j] = lu_dtype(work[j] + dt_use * D1 * dT[j])
            solve_fn(lu, ipiv, rhs)
            for j in range(n_vars):
                k_stages[0, j] = rhs[j]
            for j in range(n_vars):
                u[j] = y[j] + A21 * k_stages[0, j]

            # Stage 2
            ode_write(u, t + C2 * dt_use, p_row, work)
            for j in range(n_vars):
                rhs[j] = lu_dtype(
                    work[j] + dt_use * D2 * dT[j] + C21 * k_stages[0, j] * inv_dt
                )
            solve_fn(lu, ipiv, rhs)
            for j in range(n_vars):
                k_stages[1, j] = rhs[j]
            for j in range(n_vars):
                u[j] = y[j] + (A31 * k_stages[0, j] + A32 * k_stages[1, j])

            # Stage 3
            ode_write(u, t + C3 * dt_use, p_row, work)
            for j in range(n_vars):
                rhs[j] = lu_dtype(
                    work[j]
                    + dt_use * D3 * dT[j]
                    + (C31 * k_stages[0, j] + C32 * k_stages[1, j]) * inv_dt
                )
            solve_fn(lu, ipiv, rhs)
            for j in range(n_vars):
                k_stages[2, j] = rhs[j]
            for j in range(n_vars):
                u[j] = y[j] + (
                    A41 * k_stages[0, j] + A42 * k_stages[1, j] + A43 * k_stages[2, j]
                )

            # Stage 4
            ode_write(u, t + C4 * dt_use, p_row, work)
            for j in range(n_vars):
                rhs[j] = lu_dtype(
                    work[j]
                    + dt_use * D4 * dT[j]
                    + (
                        C41 * k_stages[0, j]
                        + C42 * k_stages[1, j]
                        + C43 * k_stages[2, j]
                    )
                    * inv_dt
                )
            solve_fn(lu, ipiv, rhs)
            for j in range(n_vars):
                k_stages[3, j] = rhs[j]
            for j in range(n_vars):
                u[j] = y[j] + (
                    A51 * k_stages[0, j]
                    + A52 * k_stages[1, j]
                    + A53 * k_stages[2, j]
                    + A54 * k_stages[3, j]
                )

            # Stage 5
            ode_write(u, t + C5 * dt_use, p_row, work)
            for j in range(n_vars):
                rhs[j] = lu_dtype(
                    work[j]
                    + dt_use * D5 * dT[j]
                    + (
                        C51 * k_stages[0, j]
                        + C52 * k_stages[1, j]
                        + C53 * k_stages[2, j]
                        + C54 * k_stages[3, j]
                    )
                    * inv_dt
                )
            solve_fn(lu, ipiv, rhs)
            for j in range(n_vars):
                k_stages[4, j] = rhs[j]
            for j in range(n_vars):
                u[j] = y[j] + (
                    A61 * k_stages[0, j]
                    + A62 * k_stages[1, j]
                    + A63 * k_stages[2, j]
                    + A64 * k_stages[3, j]
                    + A65 * k_stages[4, j]
                )

            # Stage 6
            ode_write(u, t_end, p_row, work)
            for j in range(n_vars):
                rhs[j] = lu_dtype(
                    work[j]
                    + (
                        C61 * k_stages[0, j]
                        + C62 * k_stages[1, j]
                        + C63 * k_stages[2, j]
                        + C64 * k_stages[3, j]
                        + C65 * k_stages[4, j]
                    )
                    * inv_dt
                )
            solve_fn(lu, ipiv, rhs)
            for j in range(n_vars):
                k_stages[5, j] = rhs[j]
                u[j] += k_stages[5, j]

            # Stage 7
            ode_write(u, t_end, p_row, work)
            for j in range(n_vars):
                rhs[j] = lu_dtype(
                    work[j]
                    + (
                        C71 * k_stages[0, j]
                        + C72 * k_stages[1, j]
                        + C73 * k_stages[2, j]
                        + C74 * k_stages[3, j]
                        + C75 * k_stages[4, j]
                        + C76 * k_stages[5, j]
                    )
                    * inv_dt
                )
            solve_fn(lu, ipiv, rhs)
            for j in range(n_vars):
                k_stages[6, j] = rhs[j]
                u[j] += k_stages[6, j]

            # Stage 8
            ode_write(u, t_end, p_row, work)
            for j in range(n_vars):
                rhs[j] = lu_dtype(
                    work[j]
                    + (
                        C81 * k_stages[0, j]
                        + C82 * k_stages[1, j]
                        + C83 * k_stages[2, j]
                        + C84 * k_stages[3, j]
                        + C85 * k_stages[4, j]
                        + C86 * k_stages[5, j]
                        + C87 * k_stages[6, j]
                    )
                    * inv_dt
                )
            solve_fn(lu, ipiv, rhs)
            for j in range(n_vars):
                k_stages[7, j] = rhs[j]

            # Weighted RMS error estimate.
            err_local = 0.0
            for j in range(n_vars):
                y_new_j = u[j] + k_stages[7, j]
                scale = atol + rtol * max(_math.fabs(y[j]), _math.fabs(y_new_j))
                r = weights[i, j] * k_stages[7, j] / scale
                err_local += r * r
            err_norm = _math.sqrt(err_local / n_vars)
            accept = err_norm <= 1.0 and not _math.isnan(err_norm)

            if _math.isnan(err_norm) or err_norm > 1e18:
                safe_err = 1e18
            elif err_norm == 0.0:
                safe_err = 1e-18
            else:
                safe_err = err_norm
            factor = SAFETY * safe_err**e1 * err_prev**e2 * err_prev2**e3
            if accept:
                err_prev2 = err_prev
                err_prev = safe_err
            if factor < FACTOR_MIN:
                factor = FACTOR_MIN
            elif factor > FACTOR_MAX:
                factor = FACTOR_MAX
            dt = dt_use * factor

            if accept:
                t_old = t
                t_new = t_old + dt_use
                while save_idx < n_save and times[save_idx] <= t_new + 1e-12 * max(
                    1.0, _math.fabs(times[save_idx])
                ):
                    theta = (times[save_idx] - t_old) / dt_use
                    theta1 = 1.0 - theta
                    for j in range(n_vars):
                        h1 = (
                            25.948786856663858 * k_stages[0, j]
                            - 2.5579724845846235 * k_stages[1, j]
                            + 10.433815404888879 * k_stages[2, j]
                            - 2.3679251022685204 * k_stages[3, j]
                            + 0.524948541321073 * k_stages[4, j]
                            + 1.1241088310450404 * k_stages[5, j]
                            + 0.4272876194431874 * k_stages[6, j]
                            - 0.17202221070155493 * k_stages[7, j]
                        )
                        h2 = (
                            -9.91568850695171 * k_stages[0, j]
                            - 0.9689944594115154 * k_stages[1, j]
                            + 3.0438037242978453 * k_stages[2, j]
                            - 24.495224566215796 * k_stages[3, j]
                            + 20.176138334709044 * k_stages[4, j]
                            + 15.98066361424651 * k_stages[5, j]
                            - 6.789040303419874 * k_stages[6, j]
                            - 6.710236069923372 * k_stages[7, j]
                        )
                        h3 = (
                            11.419903575922262 * k_stages[0, j]
                            + 2.8879645146136994 * k_stages[1, j]
                            + 72.92137995996029 * k_stages[2, j]
                            + 80.12511834622643 * k_stages[3, j]
                            - 52.072871366152654 * k_stages[4, j]
                            - 59.78993625266729 * k_stages[5, j]
                            - 0.15582684282751913 * k_stages[6, j]
                            + 4.883087185713722 * k_stages[7, j]
                        )
                        y_new_j = u[j] + k_stages[7, j]
                        # Rosenbrock continuous extension between y_old and y_new.
                        # y[j] is still y_old here; it is advanced after this loop.
                        hist[i, save_idx, j] = theta1 * y[j] + theta * (
                            y_new_j + theta1 * (h1 + theta * (h2 + theta * h3))
                        )
                    save_idx += 1
                for j in range(n_vars):
                    y[j] = u[j] + k_stages[7, j]
                t += dt_use
                accepted += 1
            else:
                rejected += 1
            n_steps += 1

        while save_idx < n_save:
            for j in range(n_vars):
                hist[i, save_idx, j] = y[j]
            save_idx += 1

        accepted_out[i] = accepted
        rejected_out[i] = rejected
        loop_out[i] = n_steps

    return kernel, lu_solver


@functools.cache
def _make_numba_jax_launch(
    ode_fn,
    jac_fn,
    time_jac_fn,
    n: int,
    n_vars: int,
    n_save: int,
    n_params: int,
    pcoeff: float = 0.0,
    icoeff: float = 1.0,
    dcoeff: float = 0.0,
    lu_precision: str = "fp32",
    batches_per_block="suggested",
    custom_lu_solver=None,
    tf_local_idx: int = -1,
):
    from numba import types

    kernel, lu_solver = _make_numba_kernel(
        ode_fn,
        jac_fn,
        time_jac_fn,
        n_vars,
        pcoeff,
        icoeff,
        dcoeff,
        lu_precision,
        batches_per_block,
        custom_lu_solver,
        tf_local_idx,
    )
    f64_2d = types.float64[:, ::1]
    f64_1d = types.float64[::1]
    i32_1d = types.int32[::1]
    argtypes = (
        f64_2d,
        f64_1d,
        f64_2d,
        types.float64,
        types.float64,
        types.float64,
        types.int32,
        f64_2d,
        types.float64[:, :, ::1],
        i32_1d,
        i32_1d,
        i32_1d,
    )
    batches_per_block = int(lu_solver.batches_per_block)
    threads = as_launch_block_dim(lu_solver.block_dim)
    blocks = (n + batches_per_block - 1) // batches_per_block
    return make_launch(kernel, argtypes, grid=blocks, block=threads)


def rodas5Pnumba_solve(
    ode_fn,
    jac_fn,
    y0,
    t_span,
    params,
    *,
    time_jac_fn=None,
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
    lu_precision: str = "fp32",
    batches_per_block="suggested",
    custom_lu_solver=None,
    tf_local_idx: int = -1,
):
    """JAX-callable numba-CUDA Rodas5P batch solve.

    Requires numba-cuda, a CUDA GPU, and a thread-local ``custom_lu_solver``
    (see the module section header). ``ode_fn``/``jac_fn``/``time_jac_fn`` are
    CUDA-device callbacks with the packed-row signatures documented in DISCO2.
    """

    # The numba kernel's ABI is float64 throughout. If JAX hands us float32
    # buffers the kernel reinterprets their memory as float64 and silently
    # returns garbage rather than failing, so refuse them up front.
    for name, arr in (("y0", y0), ("t_span", t_span), ("params", params)):
        dtype = jnp.result_type(arr)
        if dtype != jnp.float64:
            raise TypeError(
                f"rodas5Pnumba_solve requires float64 arrays, but {name!r} has "
                f"dtype {dtype}. Enable 64-bit JAX with "
                'jax.config.update("jax_enable_x64", True) before building inputs.'
            )

    def solve_impl(y0_arr, t_span_arr, params_arr):
        return _rodas5Pnumba_solve_impl(
            ode_fn,
            jac_fn,
            y0_arr,
            t_span_arr,
            params_arr,
            time_jac_fn=time_jac_fn,
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
            lu_precision=lu_precision,
            batches_per_block=batches_per_block,
            custom_lu_solver=custom_lu_solver,
            tf_local_idx=tf_local_idx,
        )

    return make_custom_vmap_solver(
        solve_impl,
        return_stats=return_stats,
        stats_postprocess=per_trajectory_stats_postprocess,
    )(y0, t_span, params)


def _rodas5Pnumba_solve_impl(
    ode_fn,
    jac_fn,
    y0,
    t_span,
    params,
    *,
    time_jac_fn,
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
    lu_precision: str = "fp32",
    batches_per_block="suggested",
    custom_lu_solver=None,
    tf_local_idx: int = -1,
):
    del batch_size
    y0_arr, params_arr, n, n_vars = normalize_y0_params(y0, params)
    times = jnp.asarray(t_span, dtype=jnp.float64)
    n_save = times.shape[0]
    n_params = params_arr.shape[1]
    dt0 = initial_step(times, first_step)
    weights_arr = jnp.asarray(_numba_build_error_weights(error_weights, n, n_vars))

    launch = _make_numba_jax_launch(
        ode_fn,
        jac_fn,
        time_jac_fn,
        n,
        n_vars,
        n_save,
        n_params,
        pcoeff,
        icoeff,
        dcoeff,
        lu_precision,
        batches_per_block,
        custom_lu_solver,
        tf_local_idx,
    )
    hist_spec = jax.ShapeDtypeStruct((n, n_save, n_vars), jnp.float64)
    int_spec = jax.ShapeDtypeStruct((n,), jnp.int32)
    output_specs = (hist_spec, int_spec, int_spec, int_spec)
    result = ffi_abi_call(
        launch,
        (y0_arr, times, params_arr, weights_arr),
        output_specs,
        input_kinds=(
            ABI_ARRAY,
            ABI_ARRAY,
            ABI_ARRAY,
            ABI_SCALAR_F64,
            ABI_SCALAR_F64,
            ABI_SCALAR_F64,
            ABI_SCALAR_I32,
            ABI_ARRAY,
        ),
        output_kinds=(ABI_ARRAY,) * len(output_specs),
        scalar_f64_values=(dt0, rtol, atol),
        scalar_i32_values=(max_steps,),
    )
    hist, accepted, rejected, loop_steps = result[:4]
    if not return_stats:
        return hist
    return hist, jax_stats(accepted, rejected, loop_steps)
