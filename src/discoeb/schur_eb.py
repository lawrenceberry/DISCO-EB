"""Thread-local Schur / Einstein-Boltzmann block-LU solver (numba-cuda).

This is the ``custom_lu_solver`` consumed by
:func:`discoeb.integrators.rodas5Pnumba_solve`: it factorizes and solves the
per-step Rosenbrock linear system ``(1/(dt*gamma)) I - J`` entirely in
thread-local memory, exploiting the bordered-block sparsity of the linear
Einstein-Boltzmann Jacobian.

The state splits into a densely-coupled "core" (metric + fluid + the lowest
photon / polarization / neutrino multipoles, and -- when enabled -- the
dark-energy fluid perturbations) bordered by free-streaming tridiagonal
hierarchies (photon temperature, E-mode polarization, massless neutrinos). Each
tridiagonal block couples back into the dense core only through its lowest
multipole, so the system is solved by a Schur complement onto the dense core
plus one Thomas sweep per block.

The dense-core membership (``dense_idx``), tridiagonal block bases
(``k_bases``), block-to-core couplings (``dense_c0``), block length
(``tridiag_len``) and state size (``nvar``) are constructor parameters, so the
same solver serves the flat-LambdaCDM + massless layout (11-var core, default)
and its dark-energy extension (13-var core). Requires ``numba`` with CUDA
support and a visible GPU.
"""

import numba
import numpy as np
from numba import cuda

# Flat-LambdaCDM + massless-neutrino defaults (NVAR == 50).
D_SIZE = 11
A_SIZE = 13
NUM_BLOCKS = 3
B_PAYLOAD = 50

DENSE_IDX = (0, 1, 2, 3, 4, 5, 6, 20, 34, 35, 36)
K_BASES = (7, 21, 37)
DENSE_C0 = (6, 7, 10)


class SchurEBSolver:
    def __init__(
        self,
        batches_per_block=128,
        block_dim=(128, 1, 1),
        precision=np.float32,
        *,
        dense_idx=DENSE_IDX,
        k_bases=K_BASES,
        dense_c0=DENSE_C0,
        tridiag_len=A_SIZE,
        nvar=B_PAYLOAD,
    ):
        self.batches_per_block = batches_per_block
        self.block_dim = block_dim
        self.precision = precision

        self.dense_idx = tuple(int(i) for i in dense_idx)
        self.k_bases = tuple(int(i) for i in k_bases)
        self.dense_c0 = tuple(int(i) for i in dense_c0)
        self.tridiag_len = int(tridiag_len)
        self.nvar = int(nvar)
        self.d_size = len(self.dense_idx)
        self.n_blocks = len(self.k_bases)
        self.ipiv_size = self.d_size

        # Buffer layout offsets within a single system's LU buffer.
        tl = self.tridiag_len
        self._s_off = 0
        self._p_off = self.d_size * self.d_size
        self._invd_off = self._p_off + self.n_blocks * tl
        self._u_off = self._invd_off + self.n_blocks * tl
        self._c0_off = self._u_off + self.n_blocks * (tl - 1)
        self._a_payload = self._c0_off + self.n_blocks

        # Compact Jacobian buffer consumed by ``assemble_lu``. It contains only
        # the entries used by the bordered-block factorization:
        #
        #   dense core | block diagonals | upper bands | lower/coupling bands |
        #   dense-to-block couplings
        #
        # ``lower[..., 0]`` is the block row's coupling to its dense-core
        # variable; subsequent entries are the ordinary subdiagonal.
        self.jac_dense_off = 0
        self.jac_diag_off = self.d_size * self.d_size
        self.jac_upper_off = self.jac_diag_off + self.n_blocks * tl
        self.jac_lower_off = self.jac_upper_off + self.n_blocks * (tl - 1)
        self.jac_c0_off = self.jac_lower_off + self.n_blocks * tl
        self.jac_size = self.jac_c0_off + self.n_blocks

        self.factorize_local = self._make_factorize_local()
        self.solve_local = self._make_solve_local()

    def a_local_size(self):
        return self._a_payload

    def b_local_size(self):
        return self.nvar

    def jacobian_slots(self):
        """Map consumed ``(row, column)`` entries to the compact buffer."""

        slots = {}
        n = self.d_size
        tl = self.tridiag_len
        for r, r_orig in enumerate(self.dense_idx):
            for c, c_orig in enumerate(self.dense_idx):
                slots[(r_orig, c_orig)] = self.jac_dense_off + r * n + c

        for b, kb in enumerate(self.k_bases):
            for i in range(tl):
                slots[(kb + i, kb + i)] = self.jac_diag_off + b * tl + i
            for i in range(tl - 1):
                slots[(kb + i, kb + i + 1)] = (
                    self.jac_upper_off + b * (tl - 1) + i
                )

            dense_orig = self.dense_idx[self.dense_c0[b]]
            slots[(kb, dense_orig)] = self.jac_lower_off + b * tl
            for i in range(1, tl):
                slots[(kb + i, kb + i - 1)] = self.jac_lower_off + b * tl + i
            slots[(dense_orig, kb)] = self.jac_c0_off + b

        return slots

    def _make_factorize_local(self):
        n = self.d_size
        s_off = self._s_off

        @cuda.jit(device=True)
        def factorize_device(lu, ipiv):
            # Dense LU with partial pivoting on the d_size x d_size Schur block.
            for i in range(n):
                max_val = 0.0
                pivot_row = i
                for k in range(i, n):
                    val = abs(lu[s_off + k * n + i])
                    if val > max_val:
                        max_val = val
                        pivot_row = k

                ipiv[i] = pivot_row

                if max_val == 0.0:
                    continue

                if pivot_row != i:
                    for j in range(n):
                        tmp = lu[s_off + i * n + j]
                        lu[s_off + i * n + j] = lu[s_off + pivot_row * n + j]
                        lu[s_off + pivot_row * n + j] = tmp

                pivot_val = lu[s_off + i * n + i]
                inv_pivot = 1.0 / pivot_val
                for k in range(i + 1, n):
                    lu[s_off + k * n + i] *= inv_pivot

                for k in range(i + 1, n):
                    factor = lu[s_off + k * n + i]
                    for j in range(i + 1, n):
                        lu[s_off + k * n + j] -= factor * lu[s_off + i * n + j]

        return factorize_device

    def _make_solve_local(self):
        n = self.d_size
        n_blocks = self.n_blocks
        tl = self.tridiag_len
        last = tl - 1
        dense_c0 = self.dense_c0
        dense_idx = self.dense_idx
        k_bases = self.k_bases
        s_off = self._s_off
        p_off = self._p_off
        invd_off = self._invd_off
        u_off = self._u_off
        c0_off = self._c0_off
        q_size = n_blocks * tl

        @cuda.jit(device=True)
        def solve_device(lu, ipiv, rhs):
            r_dense = cuda.local.array(n, numba.float64)
            for d in range(n):
                r_dense[d] = rhs[dense_idx[d]]

            Q_arr = cuda.local.array(q_size, numba.float64)

            # 1. Forward substitution (bottom-to-top) on each tridiagonal RHS.
            for b in range(n_blocks):
                kb = k_bases[b]
                Q_arr[b * tl + last] = rhs[kb + last] * lu[invd_off + b * tl + last]
                for i in range(last - 1, -1, -1):
                    r_i = rhs[kb + i]
                    u_i = lu[u_off + b * (tl - 1) + i]
                    Q_arr[b * tl + i] = (
                        r_i - u_i * Q_arr[b * tl + i + 1]
                    ) * lu[invd_off + b * tl + i]

                # 2. Update dense RHS through the block-to-core coupling.
                dc0 = dense_c0[b]
                c0 = lu[c0_off + b]
                r_dense[dc0] -= c0 * Q_arr[b * tl + 0]

            # 3. Dense LU solve for r_dense.
            for i in range(n):
                pivot = ipiv[i]
                if pivot != i:
                    tmp = r_dense[i]
                    r_dense[i] = r_dense[pivot]
                    r_dense[pivot] = tmp
                for j in range(i):
                    r_dense[i] -= lu[s_off + i * n + j] * r_dense[j]

            for i in range(n - 1, -1, -1):
                for j in range(i + 1, n):
                    r_dense[i] -= lu[s_off + i * n + j] * r_dense[j]
                r_dense[i] /= lu[s_off + i * n + i]

            for d in range(n):
                rhs[dense_idx[d]] = r_dense[d]

            # 4. Back substitution (top-to-bottom) for the tridiagonal variables.
            for b in range(n_blocks):
                kb = k_bases[b]
                dc0 = dense_c0[b]
                x_prev = lu[p_off + b * tl + 0] * r_dense[dc0] + Q_arr[b * tl + 0]
                rhs[kb + 0] = x_prev
                for i in range(1, tl):
                    x_i = lu[p_off + b * tl + i] * x_prev + Q_arr[b * tl + i]
                    rhs[kb + i] = x_i
                    x_prev = x_i

        return solve_device

    def assemble_lu_local_factory(self, jac_device, n_vars, lu_dtype):
        n = self.d_size
        n_blocks = self.n_blocks
        tl = self.tridiag_len
        last = tl - 1
        dense_c0 = self.dense_c0
        s_off = self._s_off
        p_off = self._p_off
        invd_off = self._invd_off
        u_off = self._u_off
        c0_off = self._c0_off
        jac_dense_off = self.jac_dense_off
        jac_diag_off = self.jac_diag_off
        jac_upper_off = self.jac_upper_off
        jac_lower_off = self.jac_lower_off
        jac_c0_off = self.jac_c0_off
        jac_size = self.jac_size

        @cuda.jit(device=True)
        def assemble_lu(y_local, t, p_row, lu_buf, dtgamma_inv):
            values = cuda.local.array(jac_size, numba.float64)
            jac_device(y_local, t, p_row, values)

            # 1. Initialize the dense Schur block S (n x n).
            for r in range(n):
                for c in range(n):
                    v = values[jac_dense_off + r * n + c]
                    if r == c:
                        lu_buf[s_off + r * n + c] = lu_dtype(dtgamma_inv - v)
                    else:
                        lu_buf[s_off + r * n + c] = lu_dtype(-v)

            # 2. Thomas factorization of each tridiagonal block.
            for b in range(n_blocks):
                v_d = values[jac_diag_off + b * tl + last]
                m_d = dtgamma_inv - v_d
                inv_d = 1.0 / m_d
                lu_buf[invd_off + b * tl + last] = lu_dtype(inv_d)

                v_l = values[jac_lower_off + b * tl + last]
                m_l = -v_l
                lu_buf[p_off + b * tl + last] = lu_dtype(-m_l * inv_d)

                for idx in range(last - 1, -1, -1):
                    v_d = values[jac_diag_off + b * tl + idx]
                    m_d = dtgamma_inv - v_d

                    v_u = values[jac_upper_off + b * (tl - 1) + idx]
                    m_u = -v_u
                    lu_buf[u_off + b * (tl - 1) + idx] = lu_dtype(m_u)

                    denom = m_d + m_u * lu_buf[p_off + b * tl + idx + 1]
                    inv_d = 1.0 / denom
                    lu_buf[invd_off + b * tl + idx] = lu_dtype(inv_d)

                    # At idx == 0 this is the block's coupling to the dense core;
                    # subsequent entries are the ordinary subdiagonal.
                    v_l = values[jac_lower_off + b * tl + idx]
                    m_l = -v_l

                    lu_buf[p_off + b * tl + idx] = lu_dtype(-m_l * inv_d)

                # 3. Schur-complement update: S_{dc0,dc0} += C_0 * P_0.
                v_c0 = values[jac_c0_off + b]
                m_c0 = -v_c0
                lu_buf[c0_off + b] = lu_dtype(m_c0)
                dc0 = dense_c0[b]
                lu_buf[s_off + dc0 * n + dc0] += lu_dtype(
                    m_c0 * lu_buf[p_off + b * tl + 0]
                )

        return assemble_lu
