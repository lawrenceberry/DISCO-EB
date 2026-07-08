"""Einstein-Boltzmann perturbation ODE solve (numba-CUDA rodas5Pnumba).

Assembles the scalar perturbation equations of :mod:`discoeb.perturbations` into a
batched stiff ODE solve over wave modes ``k`` using the numba-CUDA Rodas5P solver
(:func:`discoeb.integrators.rodas5Pnumba_solve`) with the thread-local Schur-EB
block-LU solver (:class:`discoeb.schur_eb.SchurEBSolver`).

The background + thermodynamics splines (``a(tau)``, Thomson opacity
``kappa'(tau)``, baryon sound speed ``c_s,b^2(tau)``) are precomputed on the host
with DISCO-EB's RECFAST pipeline (:func:`discoeb.background.evolve_background`),
resampled onto a uniform ``log(tau)`` grid, and uploaded to the device as cubic
spline tables. The perturbation right-hand side and its (linear, sparse)
Jacobian are compiled to numba-CUDA device functions.

This is Stage 1 of the ported solver: **flat LambdaCDM + massless neutrinos**
(``NVAR == 50``), matching the hand-tuned Schur-EB layout. Dynamical dark energy,
massive-neutrino hierarchies, and spatial curvature are follow-on stages that
grow the state layout and require a generalized Schur-EB block-LU.

Requires numba-CUDA + a GPU (via ``rodas5Pnumba_solve``), a working ``g++`` for
the one-time XLA-FFI launcher shim, and ``jax`` with CUDA support.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import interpolate

from .background import (
    dark_energy_density_ratio,
    dtau_da,
    grhob,
    grhoc,
    grhog,
    grhornomass,
    grhov,
    helium_number_fraction,
    thomson_normalization,
)
from .background_system import solve_background_system
from .constants import C_SI, K_B_SI, M_H
from .matter_power_spectrum import matter_power_spectrum, total_matter_density_contrast
from .perturbations import (
    IX_CLXB,
    IX_CLXC,
    IX_ETAK,
    IX_G,
    IX_POL,
    IX_R,
    IX_VB,
    LMAX_G,
    LMAX_NR,
    LMAX_POL,
    NVAR,
    adiabatic_initial_conditions,
)

# Solver / grid presets (mirror the DISCO2 matter-power defaults).
N_THERMO_GRID = 2048
PERTURB_RTOL = 1.0e-4
PERTURB_ATOL = 1.0e-4
PERTURB_FIRST_STEP = 1.0e-2
PERTURB_MAX_STEPS = 20000
TAU_START = 0.1

# Packed-parameter row layout consumed by the device rhs/jac.
#   p = (grhog, grhornomass, grhoc, grhob, grhov,
#        tau_start, tau_end, k, cosmology_idx, save_index,
#        w_DE_0, w_DE_a, cs2_DE)
IX_TAU_START = 5
IX_TAU_END = 6
IX_K = 7
IX_COSMOLOGY = 8
IX_SAVE_INDEX = 9
IX_W_DE_0 = 10
IX_W_DE_A = 11
IX_CS2_DE = 12
N_PARAM = 13

THERMO_TABLE_DTYPE = np.float32
BATCHES_PER_BLOCK = 32


def density_coefficients(cosmology) -> tuple[float, float, float, float, float]:
    """Return the background ``grho`` density coefficients for ``cosmology``.

    Uses the massless-neutrino count (this stage is flat LambdaCDM + massless nu).
    """

    grhog_value = grhog(cosmology.T_cmb)
    return (
        grhog_value,
        grhornomass(grhog_value, cosmology.Neff_massless),
        grhoc(cosmology.omega_c_h2),
        grhob(cosmology.omega_b_h2),
        grhov(
            cosmology.omega_b_h2,
            cosmology.omega_c_h2,
            cosmology.h,
            cosmology.Neff_massless,
            cosmology.T_cmb,
        ),
    )


Z_RECOMB_START = 3500.0
"""Redshift at which the RECFAST recombination history begins (fully ionized)."""

N_BACKGROUND_A = 4096
"""Number of scale-factor samples for the absolute conformal-time quadrature."""

N_RECOMB_Z = 2048
"""Number of redshift samples for the background_system recombination solve."""


def build_thermo_tables(cosmology, n_grid: int = N_THERMO_GRID):
    """Return ``(tau_grid, values, seconds, tau0)`` thermodynamics spline tables.

    ``values`` has rows ``(a, kappa', c_s,b^2)`` sampled on a uniform ``log(tau)``
    grid over ``[TAU_START, tau0]``; ``seconds`` are the matching cubic second
    derivatives (in ``log(tau)``). The recombination history ``x_e(z)`` / ``T_b(z)``
    comes from :func:`discoeb.background_system.solve_background_system` (the
    RECFAST + rodas5Pjax pipeline); the absolute conformal time ``tau(a)`` comes
    from the background quadrature ``dtau/da`` in :mod:`discoeb.background`. Before
    the RECFAST start (``z > Z_RECOMB_START``) the plasma is treated as fully
    ionized with ``T_b = T_cmb (1+z)``.
    """

    from scipy.integrate import cumulative_trapezoid

    grhog_v, grhornomass_v, grhoc_v, grhob_v, grhov_v = density_coefficients(cosmology)

    # Absolute conformal time tau(a) from the background: dtau/da = 1/(a^2 H),
    # including dynamical dark energy (and, later, curvature) so the a(tau) map
    # is consistent with the expansion the perturbation RHS integrates on.
    w0 = cosmology.w_DE_0
    wa = cosmology.w_DE_a
    amin = 1.0e-9
    a_grid = np.geomspace(amin, 1.0, N_BACKGROUND_A)
    dtauda_vals = np.array(
        [
            dtau_da(
                float(a),
                grhog_v,
                grhornomass_v,
                grhoc_v,
                grhob_v,
                grhov_v,
                rho_de=float(dark_energy_density_ratio(float(a), w0, wa)),
            )
            for a in a_grid
        ]
    )
    # Radiation-era conformal time already elapsed by amin: a ~ adotrad * tau.
    adotrad = ((grhog_v + grhornomass_v) / 3.0) ** 0.5
    tau_of_a = cumulative_trapezoid(dtauda_vals, a_grid, initial=0.0) + amin / adotrad
    tau0 = float(tau_of_a[-1])

    # Recombination history from the RECFAST + rodas5Pjax background system.
    z_bg = np.linspace(Z_RECOMB_START, 0.0, N_RECOMB_Z)
    bg = solve_background_system(cosmology, z_bg, z_start=Z_RECOMB_START)
    z_bs = np.asarray(bg["z"])
    order = np.argsort(z_bs)
    z_sorted = z_bs[order]
    xe_sorted = np.asarray(bg["x_e"])[order]
    tb_sorted = np.asarray(bg["T_mat"])[order]
    xe_recomb_start = float(np.interp(Z_RECOMB_START, z_sorted, xe_sorted))

    # Perturbation thermodynamics grid, uniform in log(tau).
    log_tau_grid = np.linspace(np.log(TAU_START), np.log(tau0), n_grid)
    tau_grid = np.exp(log_tau_grid)
    a_vals = np.interp(tau_grid, tau_of_a, a_grid)
    z_vals = 1.0 / a_vals - 1.0

    pre_recomb = z_vals > Z_RECOMB_START
    xe_vals = np.where(pre_recomb, xe_recomb_start, np.interp(z_vals, z_sorted, xe_sorted))
    tb_vals = np.where(
        pre_recomb, cosmology.T_cmb * (1.0 + z_vals), np.interp(z_vals, z_sorted, tb_sorted)
    )

    akthom = float(thomson_normalization(cosmology.omega_b_h2, cosmology.Y_He))
    opacity_vals = xe_vals * akthom / a_vals**2

    # Baryon sound speed cs2 = (k_B/(m_H c^2)) T_b (1 - (1/3) dln T_b/dln a).
    # Floor T_b to a small positive value: the background_system matter
    # temperature can overshoot slightly negative in the last few z ~ 0 samples
    # (where T_b, and hence cs2, is dynamically negligible for the matter power).
    barssc = K_B_SI / (M_H * C_SI**2)
    tb_safe = np.maximum(tb_vals, 1.0e-2)
    dlnT_dlna = np.gradient(np.log(tb_safe), np.log(a_vals))
    cs2_vals = np.maximum(barssc * tb_safe * (1.0 - dlnT_dlna / 3.0), 0.0)

    values = np.stack((a_vals, opacity_vals, cs2_vals))
    seconds = np.stack(
        [interpolate.CubicSpline(log_tau_grid, row)(log_tau_grid, 2) for row in values]
    )
    return tau_grid, values, seconds, tau0


def make_params(cosmology, k_values: np.ndarray, tau_end: float) -> np.ndarray:
    """Return per-mode packed perturbation parameters for the device rhs."""

    densities = density_coefficients(cosmology)
    n = len(k_values)
    rows = np.repeat(np.asarray(densities)[None, :], n, axis=0)
    tau_start = np.full(n, TAU_START, dtype=np.float64)
    tau_end_col = np.full(n, tau_end, dtype=np.float64)
    cosmology_idx = np.zeros(n, dtype=np.float64)
    save_index = np.ones(n, dtype=np.float64)
    w_de_0 = np.full(n, cosmology.w_DE_0, dtype=np.float64)
    w_de_a = np.full(n, cosmology.w_DE_a, dtype=np.float64)
    cs2_de = np.full(n, cosmology.cs2_DE, dtype=np.float64)
    return np.column_stack(
        (rows, tau_start, tau_end_col, np.asarray(k_values, dtype=np.float64),
         cosmology_idx, save_index, w_de_0, w_de_a, cs2_de)
    )


def make_initial_states(cosmology, k_values: np.ndarray, layout) -> np.ndarray:
    """Return adiabatic initial states for every wave mode, shape ``(n_k, nvar)``.

    The always-present core (metric + fluid + photon/polarization/massless-nu
    hierarchies) is set by :func:`adiabatic_initial_conditions`; when dark energy
    is enabled, the DE fluid is given its adiabatic value
    ``clxq = (1 + w_0) clxc``, ``thetaq = 0``.
    """

    densities = density_coefficients(cosmology)
    y0 = np.zeros((len(k_values), layout.nvar), dtype=np.float64)
    for i, k in enumerate(k_values):
        core = adiabatic_initial_conditions(float(k), TAU_START, *densities[:4])
        y0[i, : len(core)] = core
        if layout.enable_dark_energy:
            y0[i, layout.ix_clxq] = (1.0 + cosmology.w_DE_0) * core[IX_CLXC]
            y0[i, layout.ix_thetaq] = 0.0
    return y0


def build_numba_callbacks(layout, tau_grid, values, seconds):
    """Return numba-CUDA ``(rhs, jac, time_jac)`` device callbacks for a layout.

    The thermodynamics tables are uploaded once and read through a uniform
    ``log(tau)`` cubic-spline device evaluator; the perturbation RHS and its
    (linear, sparse) Jacobian are compiled as ``cuda.jit(device=True)`` functions.
    The dynamical-dark-energy fluid equations are included when
    ``layout.enable_dark_energy`` is set; because the flag is a compile-time
    constant, numba prunes the DE branches entirely when it is disabled (so the
    flat-LambdaCDM kernel is unchanged).
    """

    from numba import cuda, float64

    NVAR = layout.nvar
    ENABLE_DE = layout.enable_dark_energy
    IX_CLXQ = layout.ix_clxq if ENABLE_DE else 0
    IX_THETAQ = layout.ix_thetaq if ENABLE_DE else 0

    n_grid = tau_grid.shape[0]
    tau_min = np.asarray([np.log(tau_grid[0])], dtype=np.float64)
    inv_dtau = np.asarray(
        [(n_grid - 1) / (np.log(tau_grid[-1]) - np.log(tau_grid[0]))], dtype=np.float64
    )
    values_t = np.ascontiguousarray(values[None, :, :], dtype=THERMO_TABLE_DTYPE)
    seconds_t = np.ascontiguousarray(seconds[None, :, :], dtype=THERMO_TABLE_DTYPE)

    tau_min_dev = cuda.to_device(tau_min)
    inv_dtau_dev = cuda.to_device(inv_dtau)
    values_dev = cuda.to_device(values_t)
    seconds_dev = cuda.to_device(seconds_t)

    @cuda.jit(device=True)
    def spline_eval(x, p, channel):
        cosmology_idx = int(p[IX_COSMOLOGY])
        x_min = tau_min_dev[cosmology_idx]
        inv_dx = inv_dtau_dev[cosmology_idx]
        idx = int((x - x_min) * inv_dx)
        if idx < 0:
            idx = 0
        if idx > n_grid - 2:
            idx = n_grid - 2
        h = 1.0 / inv_dx
        x0 = x_min + idx * h
        x1 = x0 + h
        left = x1 - x
        right = x - x0
        y0 = values_dev[cosmology_idx, channel, idx]
        y1 = values_dev[cosmology_idx, channel, idx + 1]
        m0 = seconds_dev[cosmology_idx, channel, idx]
        m1 = seconds_dev[cosmology_idx, channel, idx + 1]
        return (
            m0 * left**3 / (6.0 * h)
            + m1 * right**3 / (6.0 * h)
            + (y0 - m0 * h**2 / 6.0) * left / h
            + (y1 - m1 * h**2 / 6.0) * right / h
        )

    @cuda.jit(device=True)
    def rhs(y, tau, p, out):
        k = p[IX_K]
        log_tau = math.log(tau)
        a = spline_eval(log_tau, p, 0)
        opacity = spline_eval(log_tau, p, 1)
        cs2 = spline_eval(log_tau, p, 2)
        if opacity < 1.0e-30:
            opacity = 1.0e-30
        if cs2 < 0.0:
            cs2 = 0.0

        a2 = a * a
        grhog_t = p[0] / a2
        grhor_t = p[1] / a2
        grhoc_t = p[2] / a
        grhob_t = p[3] / a
        if ENABLE_DE:
            w0 = p[IX_W_DE_0]
            wa = p[IX_W_DE_A]
            w_Q = w0 + wa * (1.0 - a)
            rho_Q = a ** (-3.0 * (1.0 + w0 + wa)) * math.exp(3.0 * (a - 1.0) * wa)
            grhov_t = p[4] * rho_Q * a2
        else:
            grhov_t = p[4] * a2
        adotoa = math.sqrt((grhog_t + grhor_t + grhoc_t + grhob_t + grhov_t) / 3.0)

        etak = y[IX_ETAK]
        clxc = y[IX_CLXC]
        clxb = y[IX_CLXB]
        vb = y[IX_VB]
        clxg = y[IX_G]
        qg = y[IX_G + 1]
        pig = y[IX_G + 2]
        e2 = y[IX_POL]
        clxr = y[IX_R]
        qr = y[IX_R + 1]
        pir = y[IX_R + 2]
        dgrho = grhob_t * clxb + grhoc_t * clxc + grhog_t * clxg + grhor_t * clxr
        dgq = grhob_t * vb + grhog_t * qg + grhor_t * qr
        if ENABLE_DE:
            clxq = y[IX_CLXQ]
            thetaq = y[IX_THETAQ]
            dgrho = dgrho + grhov_t * clxq
            dgq = dgq + grhov_t * (1.0 + w_Q) * thetaq / k
        z = (0.5 * dgrho / k + etak) / adotoa
        sigma = z + 1.5 * dgq / (k * k)
        photbar = grhog_t / grhob_t
        pb43 = 4.0 / 3.0 * photbar
        delta_p_b = cs2 * clxb
        polter = pig / 10.0 + 9.0 * e2 / 15.0
        vbdot = -adotoa * vb + k * delta_p_b - photbar * opacity * (4.0 * vb / 3.0 - qg)

        for i in range(NVAR):
            out[i] = 0.0
        out[IX_ETAK] = 0.5 * dgq
        out[IX_CLXC] = -k * z
        out[IX_CLXB] = -k * (z + vb)
        out[IX_VB] = vbdot
        out[IX_G] = -k * (4.0 * z / 3.0 + qg)
        out[IX_G + 1] = (
            4.0 * (-vbdot - adotoa * vb + k * delta_p_b) / (3.0 * pb43)
            + k * clxg / 3.0
            - 2.0 * k * pig / 3.0
        )
        out[IX_G + 2] = (
            2.0 * k * qg / 5.0
            - 3.0 * k * y[IX_G + 3] / 5.0
            - opacity * (pig - polter)
            + 8.0 * k * sigma / 15.0
        )
        for ell in range(3, LMAX_G):
            out[IX_G + ell] = (
                k * ell * y[IX_G + ell - 1] / (2 * ell + 1)
                - k * (ell + 1) * y[IX_G + ell + 1] / (2 * ell + 1)
                - opacity * y[IX_G + ell]
            )
        out[IX_G + LMAX_G] = (
            k * y[IX_G + LMAX_G - 1]
            - (LMAX_G + 1) * y[IX_G + LMAX_G] / tau
            - opacity * y[IX_G + LMAX_G]
        )

        out[IX_POL] = -opacity * (e2 - polter) - k * y[IX_POL + 1] / 3.0
        for ell in range(3, LMAX_POL):
            idx = IX_POL + ell - 2
            polfac = (ell + 3) * (ell - 1) / (ell + 1)
            out[idx] = (
                -opacity * y[idx]
                + k * ell * y[idx - 1] / (2 * ell + 1)
                - polfac * k * y[idx + 1] / (2 * ell + 1)
            )
        idx_last = IX_POL + LMAX_POL - 2
        out[idx_last] = (
            -opacity * y[idx_last]
            + k * LMAX_POL * y[idx_last - 1] / (2 * LMAX_POL + 1)
            - (LMAX_POL + 3) * y[idx_last] / tau
        )

        out[IX_R] = -k * (4.0 * z / 3.0 + qr)
        out[IX_R + 1] = k * (clxr - 2.0 * pir) / 3.0
        out[IX_R + 2] = (
            2.0 * k * qr / 5.0 - 3.0 * k * y[IX_R + 3] / 5.0 + 8.0 * k * sigma / 15.0
        )
        for ell in range(3, LMAX_NR):
            out[IX_R + ell] = k * ell * y[IX_R + ell - 1] / (2 * ell + 1) - k * (
                ell + 1
            ) * y[IX_R + ell + 1] / (2 * ell + 1)
        out[IX_R + LMAX_NR] = (
            k * y[IX_R + LMAX_NR - 1] - (LMAX_NR + 1) * y[IX_R + LMAX_NR] / tau
        )

        if ENABLE_DE:
            cs2_Q = p[IX_CS2_DE]
            w_Q_prime = -wa * adotoa * a
            ca2_Q = w_Q - w_Q_prime / (3.0 * (1.0 + w_Q) * adotoa)
            out[IX_CLXQ] = (
                -(1.0 + w_Q) * (thetaq + k * z)
                - 3.0 * (cs2_Q - w_Q) * adotoa * clxq
                - 9.0 * (1.0 + w_Q) * (cs2_Q - ca2_Q) * adotoa**2 / k**2 * thetaq
            )
            out[IX_THETAQ] = (
                -(1.0 - 3.0 * cs2_Q) * adotoa * thetaq
                + cs2_Q / (1.0 + w_Q) * k**2 * clxq
            )

    def make_jac():
        # The Einstein-Boltzmann RHS is linear in y, so the Jacobian is the
        # (background-dependent) coefficient matrix. It is assembled here as a
        # sparse set of nonzero entries and returned as a dense 50x50 nested
        # tuple; numba's dead-code elimination prunes the zero entries.
        zc = {
            IX_ETAK: "zc_etak",
            IX_CLXB: "zc_clxb",
            IX_CLXC: "zc_clxc",
            IX_G: "zc_g",
            IX_R: "zc_r",
        }
        # z depends on clxq through dgrho, so clxq joins the metric coupling.
        if ENABLE_DE:
            zc[IX_CLXQ] = "zc_clxq"
        sigc = dict(zc)
        sigc[IX_VB] = "sigc_vb"
        sigc[IX_G + 1] = "sigc_qg"
        sigc[IX_R + 1] = "sigc_qr"
        # sigma depends on thetaq through dgq.
        if ENABLE_DE:
            sigc[IX_THETAQ] = "sigc_thetaq"
        mat: dict[tuple[int, int], list[str]] = {}

        def setc(r, c, e):
            mat[(r, c)] = [e]

        def addc(r, c, e):
            mat.setdefault((r, c), []).append(e)

        setc(IX_ETAK, IX_VB, "0.5 * grhob_t")
        setc(IX_ETAK, IX_G + 1, "0.5 * grhog_t")
        setc(IX_ETAK, IX_R + 1, "0.5 * grhor_t")
        if ENABLE_DE:
            setc(IX_ETAK, IX_THETAQ, "0.5 * grhov_t * (1.0 + w_Q) / k")
        for c, t in zc.items():
            addc(IX_CLXC, c, f"-k * {t}")
            addc(IX_CLXB, c, f"-k * {t}")
            addc(IX_G, c, f"-(4.0 * k / 3.0) * {t}")
            addc(IX_R, c, f"-(4.0 * k / 3.0) * {t}")
        addc(IX_CLXB, IX_VB, "-k")
        addc(IX_G, IX_G + 1, "-k")
        addc(IX_R, IX_R + 1, "-k")

        vbf = {
            IX_VB: "(-adotoa - 4.0 * photbar * opacity / 3.0)",
            IX_CLXB: "k * cs2",
            IX_G + 1: "photbar * opacity",
        }
        for c, e in vbf.items():
            addc(IX_VB, c, e)
            addc(IX_G + 1, c, f"(-4.0 / (3.0 * pb43)) * ({e})")
        addc(IX_G + 1, IX_VB, "(-4.0 * adotoa) / (3.0 * pb43)")
        addc(IX_G + 1, IX_CLXB, "(4.0 * k * cs2) / (3.0 * pb43)")
        addc(IX_G + 1, IX_G, "k / 3.0")
        addc(IX_G + 1, IX_G + 2, "-2.0 * k / 3.0")
        addc(IX_G + 2, IX_G + 1, "2.0 * k / 5.0")
        addc(IX_G + 2, IX_G + 3, "-3.0 * k / 5.0")
        addc(IX_G + 2, IX_G + 2, "-0.9 * opacity")
        addc(IX_G + 2, IX_POL, "0.6 * opacity")
        for c, t in sigc.items():
            addc(IX_G + 2, c, f"(8.0 * k / 15.0) * {t}")
        for ell in range(3, LMAX_G):
            row = IX_G + ell
            setc(row, row - 1, f"k * {ell} / {2 * ell + 1}")
            setc(row, row, "-opacity")
            setc(row, row + 1, f"-k * {ell + 1} / {2 * ell + 1}")
        setc(IX_G + LMAX_G, IX_G + LMAX_G - 1, "k")
        setc(IX_G + LMAX_G, IX_G + LMAX_G, f"-{LMAX_G + 1} / tau - opacity")

        addc(IX_POL, IX_G + 2, "0.1 * opacity")
        addc(IX_POL, IX_POL, "-0.4 * opacity")
        addc(IX_POL, IX_POL + 1, "-k / 3.0")
        for ell in range(3, LMAX_POL):
            row = IX_POL + ell - 2
            polfac = (ell + 3) * (ell - 1) / (ell + 1)
            setc(row, row - 1, f"k * {ell} / {2 * ell + 1}")
            setc(row, row, "-opacity")
            setc(row, row + 1, f"-{polfac!r} * k / {2 * ell + 1}")
        row = IX_POL + LMAX_POL - 2
        setc(row, row - 1, f"k * {LMAX_POL} / {2 * LMAX_POL + 1}")
        setc(row, row, f"-opacity - {LMAX_POL + 3} / tau")

        setc(IX_R + 1, IX_R, "k / 3.0")
        setc(IX_R + 1, IX_R + 2, "-2.0 * k / 3.0")
        setc(IX_R + 2, IX_R + 1, "2.0 * k / 5.0")
        setc(IX_R + 2, IX_R + 3, "-3.0 * k / 5.0")
        for c, t in sigc.items():
            addc(IX_R + 2, c, f"(8.0 * k / 15.0) * {t}")
        for ell in range(3, LMAX_NR):
            row = IX_R + ell
            setc(row, row - 1, f"k * {ell} / {2 * ell + 1}")
            setc(row, row + 1, f"-k * {ell + 1} / {2 * ell + 1}")
        setc(IX_R + LMAX_NR, IX_R + LMAX_NR - 1, "k")
        setc(IX_R + LMAX_NR, IX_R + LMAX_NR, f"-{LMAX_NR + 1} / tau")

        if ENABLE_DE:
            # DE density row: clxq' = -(1+w)(thetaq + k z) - 3(cs2-w) aH clxq
            #                        - 9(1+w)(cs2-ca2) aH^2/k^2 thetaq.
            for c, t in zc.items():
                addc(IX_CLXQ, c, f"-(1.0 + w_Q) * k * {t}")
            addc(IX_CLXQ, IX_CLXQ, "-3.0 * (cs2_Q - w_Q) * adotoa")
            addc(
                IX_CLXQ,
                IX_THETAQ,
                "-(1.0 + w_Q) - 9.0 * (1.0 + w_Q) * (cs2_Q - ca2_Q) * adotoa**2 / k**2",
            )
            # DE velocity row: thetaq' = -(1-3cs2) aH thetaq + cs2 k^2 clxq/(1+w).
            setc(IX_THETAQ, IX_CLXQ, "cs2_Q / (1.0 + w_Q) * k**2")
            setc(IX_THETAQ, IX_THETAQ, "-(1.0 - 3.0 * cs2_Q) * adotoa")

        assignments = []
        for (row, col), terms in sorted(mat.items()):
            assignments.append(f"    j_{row}_{col} = " + " + ".join(terms))
        rows = []
        for row in range(NVAR):
            rows.append(
                "("
                + ", ".join(
                    f"j_{row}_{col}" if (row, col) in mat else "0.0"
                    for col in range(NVAR)
                )
                + ",)"
            )
        if ENABLE_DE:
            grhov_block = (
                "    w0 = p[IX_W_DE_0]\n"
                "    wa = p[IX_W_DE_A]\n"
                "    cs2_Q = p[IX_CS2_DE]\n"
                "    w_Q = w0 + wa * (1.0 - a)\n"
                "    rho_Q = a ** (-3.0 * (1.0 + w0 + wa)) * math.exp(3.0 * (a - 1.0) * wa)\n"
                "    grhov_t = p[4] * rho_Q * a2"
            )
            de_defs = (
                "    w_Q_prime = -wa * adotoa * a\n"
                "    ca2_Q = w_Q - w_Q_prime / (3.0 * (1.0 + w_Q) * adotoa)\n"
                "    zc_clxq = 0.5 * grhov_t / (k * adotoa)\n"
                "    sigc_thetaq = 1.5 * grhov_t * (1.0 + w_Q) / k**3"
            )
        else:
            grhov_block = "    grhov_t = p[4] * a2"
            de_defs = ""

        source = f"""
def jac(y, tau, p):
    k = p[IX_K]
    log_tau = math.log(tau)
    a = spline_eval(log_tau, p, 0)
    opacity = spline_eval(log_tau, p, 1)
    cs2 = spline_eval(log_tau, p, 2)
    if opacity < 1.0e-30:
        opacity = 1.0e-30
    if cs2 < 0.0:
        cs2 = 0.0
    a2 = a * a
    grhog_t = p[0] / a2
    grhor_t = p[1] / a2
    grhoc_t = p[2] / a
    grhob_t = p[3] / a
{grhov_block}
    adotoa = math.sqrt((grhog_t + grhor_t + grhoc_t + grhob_t + grhov_t) / 3.0)
    photbar = grhog_t / grhob_t
    pb43 = 4.0 / 3.0 * photbar
    zc_etak = 1.0 / adotoa
    zc_clxb = 0.5 * grhob_t / (k * adotoa)
    zc_clxc = 0.5 * grhoc_t / (k * adotoa)
    zc_g = 0.5 * grhog_t / (k * adotoa)
    zc_r = 0.5 * grhor_t / (k * adotoa)
    sigc_vb = 1.5 * grhob_t / (k * k)
    sigc_qg = 1.5 * grhog_t / (k * k)
    sigc_qr = 1.5 * grhor_t / (k * k)
{de_defs}
{chr(10).join(assignments)}
    return ({", ".join(rows)})
"""
        ns = {
            "cuda": cuda,
            "math": math,
            "spline_eval": spline_eval,
            "IX_K": IX_K,
            "IX_W_DE_0": IX_W_DE_0,
            "IX_W_DE_A": IX_W_DE_A,
            "IX_CS2_DE": IX_CS2_DE,
        }
        exec(source, ns)
        return cuda.jit(device=True)(ns["jac"])

    @cuda.jit(device=True)
    def time_jac(y, tau, p, out):
        eps = 1.0e-6 * max(abs(tau), 1.0)
        fp = cuda.local.array(NVAR, float64)
        fm = cuda.local.array(NVAR, float64)
        rhs(y, tau + eps, p, fp)
        rhs(y, tau - eps, p, fm)
        inv = 1.0 / (2.0 * eps)
        for i in range(NVAR):
            out[i] = (fp[i] - fm[i]) * inv

    return rhs, make_jac(), time_jac


def solve_perturbations(
    k_values,
    cosmology,
    *,
    rtol: float = PERTURB_RTOL,
    atol: float = PERTURB_ATOL,
    first_step: float = PERTURB_FIRST_STEP,
    max_steps: int = PERTURB_MAX_STEPS,
):
    """Solve the perturbation hierarchy for each ``k`` and return final states.

    Returns
    -------
    np.ndarray
        Final perturbation state for each wave mode, shape ``(n_k, NVAR)``,
        evaluated at ``tau0`` (today).
    """

    from .integrators import rodas5Pnumba_solve
    from .perturbation_layout import PerturbationLayout
    from .schur_eb import DENSE_IDX, K_BASES, DENSE_C0, A_SIZE, SchurEBSolver

    layout = PerturbationLayout.from_cosmology(cosmology)
    if layout.has_massive_neutrinos:
        raise NotImplementedError(
            "massive-neutrino perturbations are not yet wired into the numba solve"
        )

    k_arr = np.ascontiguousarray(np.asarray(k_values, dtype=np.float64))
    order = np.argsort(k_arr, kind="stable")
    k_sorted = k_arr[order]

    tau_grid, values, seconds, tau0 = build_thermo_tables(cosmology)
    params = np.ascontiguousarray(make_params(cosmology, k_sorted, tau0))
    y0 = np.ascontiguousarray(make_initial_states(cosmology, k_sorted, layout))
    t_span = np.asarray((TAU_START, tau0), dtype=np.float64)

    rhs, jac, time_jac = build_numba_callbacks(layout, tau_grid, values, seconds)

    if layout.enable_dark_energy:
        # The DE fluid variables join the densely-coupled metric core.
        lu_solver = SchurEBSolver(
            batches_per_block=BATCHES_PER_BLOCK,
            block_dim=(BATCHES_PER_BLOCK, 1, 1),
            dense_idx=DENSE_IDX + (layout.ix_clxq, layout.ix_thetaq),
            k_bases=K_BASES,
            dense_c0=DENSE_C0,
            tridiag_len=A_SIZE,
            nvar=layout.nvar,
        )
    else:
        lu_solver = SchurEBSolver(
            batches_per_block=BATCHES_PER_BLOCK,
            block_dim=(BATCHES_PER_BLOCK, 1, 1),
        )

    sol = rodas5Pnumba_solve(
        rhs,
        jac,
        y0,
        t_span,
        params,
        time_jac_fn=time_jac,
        lu_precision="fp32",
        custom_lu_solver=lu_solver,
        rtol=rtol,
        atol=atol,
        first_step=first_step,
        max_steps=max_steps,
        pcoeff=0.3,
        icoeff=0.4,
        batches_per_block=BATCHES_PER_BLOCK,
        tf_local_idx=IX_TAU_END,
    )
    hist = np.asarray(sol)  # (n_k, n_save, nvar)
    y_final_sorted = hist[:, 1, :]

    y_final = np.empty_like(y_final_sorted)
    y_final[order] = y_final_sorted
    return y_final


def solve_matter_power_spectrum(
    k_values,
    cosmology,
    **solve_kwargs,
) -> np.ndarray:
    """Return the linear matter power spectrum ``P(k)`` in Mpc^3 for ``cosmology``."""

    k_arr = np.ascontiguousarray(np.asarray(k_values, dtype=np.float64))
    y_final = solve_perturbations(k_arr, cosmology, **solve_kwargs)
    densities = density_coefficients(cosmology)
    grhoc_val, grhob_val = densities[2], densities[3]

    pk = np.empty(len(k_arr), dtype=np.float64)
    for i, k in enumerate(k_arr):
        delta_m = total_matter_density_contrast(
            grhoc_val, y_final[i, IX_CLXC], grhob_val, y_final[i, IX_CLXB]
        )
        pk[i] = matter_power_spectrum(
            float(k), delta_m, cosmology.A_s, cosmology.n_s, cosmology.k_pivot
        )
    return pk
