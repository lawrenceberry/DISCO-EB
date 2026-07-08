"""numba-cuda natural-cubic spline interpolators.

Ported from the DISCO2 ``src/cubic_splines.py`` numba-cuda kernels, with small
host-side helpers to build the second-derivative tables and evaluate batched
query points. The kernels operate on a batch of spline tables that share a
common abscissa ``x`` (shape ``(n,)``) with per-table ordinates ``y`` (shape
``(n_splines, n)``); ``compute_second_derivatives`` fills the natural-cubic
second derivatives and ``eval`` interpolates at arbitrary query points, each
tagged with the spline table it belongs to.

The ``@cuda.jit`` device/kernel functions require ``numba`` with CUDA support and
a visible GPU. This module imports ``numba`` at import time (unlike
:mod:`discoeb.integrators`, whose numba path is lazy), so it is only importable
where numba-cuda is installed.
"""

import numpy as np
from numba import cuda


@cuda.jit
def compute_second_derivatives_kernel(x, y, second, cp, dp):
    """Compute natural-cubic second derivatives for each spline table.

    One CUDA block handles one spline table (``blockIdx.x``); ``cp``/``dp`` are
    per-table scratch buffers of length ``n_splines * (n - 2)`` for the Thomas
    sweep.
    """

    spline = cuda.blockIdx.x
    if cuda.threadIdx.x != 0 or spline >= y.shape[0]:
        return
    n = x.shape[0]
    m = n - 2
    if m <= 0:
        return
    base = spline * m
    h_prev = x[1] - x[0]
    h_next = x[2] - x[1]
    diag = 2.0 * (h_prev + h_next)
    rhs = 6.0 * (
        (y[spline, 2] - y[spline, 1]) / h_next - (y[spline, 1] - y[spline, 0]) / h_prev
    )
    off_next = x[2] - x[1] if m > 1 else 0.0
    cp[base] = off_next / diag if m > 1 else 0.0
    dp[base] = rhs / diag

    for k in range(1, m):
        h_prev = x[k + 1] - x[k]
        h_next = x[k + 2] - x[k + 1]
        diag = 2.0 * (h_prev + h_next)
        off_prev = x[k + 1] - x[k]
        denom = diag - off_prev * cp[base + k - 1]
        rhs = 6.0 * (
            (y[spline, k + 2] - y[spline, k + 1]) / h_next
            - (y[spline, k + 1] - y[spline, k]) / h_prev
        )
        cp[base + k] = (x[k + 2] - x[k + 1]) / denom if k < m - 1 else 0.0
        dp[base + k] = (rhs - off_prev * dp[base + k - 1]) / denom

    second[spline, 0] = 0.0
    second[spline, n - 1] = 0.0
    second[spline, n - 2] = dp[base + m - 1]
    for kk in range(m - 2, -1, -1):
        second[spline, kk + 1] = dp[base + kk] - cp[base + kk] * second[spline, kk + 2]


@cuda.jit
def eval_cubic_spline_kernel(x, y, second, points, spline_index, out):
    """Evaluate natural-cubic splines at batched query points (binary search)."""

    i = cuda.grid(1)
    if i >= points.shape[0]:
        return
    p = points[i]
    spline = spline_index[i]
    lo = 0
    hi = x.shape[0] - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if x[mid] <= p:
            lo = mid
        else:
            hi = mid
    idx = lo
    if idx < 0:
        idx = 0
    if idx > x.shape[0] - 2:
        idx = x.shape[0] - 2
    x0 = x[idx]
    x1 = x[idx + 1]
    h = x1 - x0
    left = x1 - p
    right = p - x0
    m0 = second[spline, idx]
    m1 = second[spline, idx + 1]
    out[i] = (
        m0 * left**3 / (6.0 * h)
        + m1 * right**3 / (6.0 * h)
        + (y[spline, idx] - m0 * h**2 / 6.0) * left / h
        + (y[spline, idx + 1] - m1 * h**2 / 6.0) * right / h
    )


@cuda.jit
def eval_cubic_spline_branchless_binary_kernel(x, y, second, points, spline_index, out):
    """Evaluate natural-cubic splines with fixed-round binary interval selection."""

    i = cuda.grid(1)
    if i >= points.shape[0]:
        return
    p = points[i]
    spline = spline_index[i]
    lo = 0
    hi = x.shape[0] - 1
    step = 1
    while step < x.shape[0]:
        step *= 2
    step //= 2
    while step > 0:
        mid = lo + step
        choose = mid < hi and x[mid] <= p
        lo = mid if choose else lo
        hi = hi if choose else mid
        step //= 2
    idx = lo
    if idx < 0:
        idx = 0
    if idx > x.shape[0] - 2:
        idx = x.shape[0] - 2
    x0 = x[idx]
    x1 = x[idx + 1]
    h = x1 - x0
    left = x1 - p
    right = p - x0
    m0 = second[spline, idx]
    m1 = second[spline, idx + 1]
    out[i] = (
        m0 * left**3 / (6.0 * h)
        + m1 * right**3 / (6.0 * h)
        + (y[spline, idx] - m0 * h**2 / 6.0) * left / h
        + (y[spline, idx + 1] - m1 * h**2 / 6.0) * right / h
    )


# =============================================================================
# Host-side convenience wrappers
# =============================================================================

_THREADS_PER_BLOCK = 128


def compute_second_derivatives(x, y):
    """Return the natural-cubic second-derivative table for a batch of splines.

    Args:
        x: shared abscissa, shape ``(n,)`` (strictly increasing).
        y: ordinates, shape ``(n_splines, n)``.

    Returns:
        np.ndarray: second derivatives, shape ``(n_splines, n)``.
    """

    x = np.ascontiguousarray(x, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    n_splines, n = y.shape
    m = max(n - 2, 0)
    x_dev = cuda.to_device(x)
    y_dev = cuda.to_device(y)
    second_dev = cuda.device_array((n_splines, n), dtype=np.float64)
    cp_dev = cuda.device_array(max(n_splines * m, 1), dtype=np.float64)
    dp_dev = cuda.device_array(max(n_splines * m, 1), dtype=np.float64)
    compute_second_derivatives_kernel[n_splines, 1](
        x_dev, y_dev, second_dev, cp_dev, dp_dev
    )
    return second_dev.copy_to_host()


def evaluate(x, y, second, points, spline_index, *, branchless=False):
    """Evaluate batched natural-cubic splines at ``points``.

    Args:
        x: shared abscissa, shape ``(n,)``.
        y: ordinates, shape ``(n_splines, n)``.
        second: second-derivative table from :func:`compute_second_derivatives`.
        points: query abscissae, shape ``(m,)``.
        spline_index: per-point spline table index, shape ``(m,)`` (int).
        branchless: use the fixed-round binary-search kernel variant.

    Returns:
        np.ndarray: interpolated values, shape ``(m,)``.
    """

    x = np.ascontiguousarray(x, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    second = np.ascontiguousarray(second, dtype=np.float64)
    points = np.ascontiguousarray(points, dtype=np.float64)
    spline_index = np.ascontiguousarray(spline_index, dtype=np.int64)

    x_dev = cuda.to_device(x)
    y_dev = cuda.to_device(y)
    second_dev = cuda.to_device(second)
    points_dev = cuda.to_device(points)
    index_dev = cuda.to_device(spline_index)
    out_dev = cuda.device_array(points.shape[0], dtype=np.float64)

    blocks = (points.shape[0] + _THREADS_PER_BLOCK - 1) // _THREADS_PER_BLOCK
    kernel = (
        eval_cubic_spline_branchless_binary_kernel
        if branchless
        else eval_cubic_spline_kernel
    )
    kernel[blocks, _THREADS_PER_BLOCK](
        x_dev, y_dev, second_dev, points_dev, index_dev, out_dev
    )
    return out_dev.copy_to_host()


def make_uniform_log_grid_spline_eval(x_min, inv_dx, n_grid):
    """Return a ``@cuda.jit(device=True)`` evaluator for a uniform log-spaced grid.

    For the perturbation solve the thermodynamics tables are sampled on a uniform
    grid in ``log(tau)``, so the interval index is ``int((x - x_min) * inv_dx)``
    (no search). The returned device function has signature
    ``spline_eval(x, values, seconds, cosmology_idx, channel) -> float`` and is
    intended to be called from inside another CUDA device function.

    ``values`` / ``seconds`` are device arrays of shape
    ``(n_cosmologies, n_channels, n_grid)``; ``x_min`` / ``inv_dx`` are device
    arrays of shape ``(n_cosmologies,)``.
    """

    @cuda.jit(device=True)
    def spline_eval(x, values, seconds, cosmology_idx, channel):
        xm = x_min[cosmology_idx]
        inv = inv_dx[cosmology_idx]
        idx = int((x - xm) * inv)
        if idx < 0:
            idx = 0
        if idx > n_grid - 2:
            idx = n_grid - 2
        h = 1.0 / inv
        x0 = xm + idx * h
        x1 = x0 + h
        left = x1 - x
        right = x - x0
        y0 = values[cosmology_idx, channel, idx]
        y1 = values[cosmology_idx, channel, idx + 1]
        m0 = seconds[cosmology_idx, channel, idx]
        m1 = seconds[cosmology_idx, channel, idx + 1]
        return (
            m0 * left**3 / (6.0 * h)
            + m1 * right**3 / (6.0 * h)
            + (y0 - m0 * h**2 / 6.0) * left / h
            + (y1 - m1 * h**2 / 6.0) * right / h
        )

    return spline_eval
