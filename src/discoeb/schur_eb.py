"""Thread-local Schur / Einstein-Boltzmann block-LU solver (numba-cuda).

Ported from the DISCO2 prototype. This is the ``custom_lu_solver`` consumed by
:func:`discoeb.integrators.rodas5Pnumba_solve`: it factorizes and solves the
per-step Rosenbrock linear system ``(1/(dt*gamma)) I - J`` entirely in
thread-local memory, exploiting the bordered-block sparsity of the linear
Einstein-Boltzmann Jacobian.

The layout is hand-derived for the flat-LambdaCDM + massless-neutrino state of
:mod:`discoeb.perturbations` (``NVAR == 50``): an 11x11 dense "core" of the
densely-coupled variables (metric + fluid + the lowest photon / polarization /
neutrino multipoles) bordered by three 13-entry tridiagonal free-streaming
hierarchies (photon temperature, E-mode polarization, massless neutrinos). Each
tridiagonal block couples back into the dense core only through its lowest
multipole, so the whole system is solved by a Schur complement onto the dense
core plus three Thomas sweeps.

Requires ``numba`` with CUDA support and a visible GPU.
"""

import numba
import numpy as np
from numba import cuda

D_SIZE = 11
A_SIZE = 13
NUM_BLOCKS = 3

# Per-system payload sizes in the thread-local LU / RHS buffers.
#   LU payload  = S (121) + P (39) + InvD (39) + U (36) + C0 (3) = 238 words
#   RHS payload = n_vars                                          = 50 words
A_PAYLOAD = 238
B_PAYLOAD = 50

# Layout offsets within a single system's LU buffer.
S_OFF = 0          # dense Schur complement, 11x11
P_OFF = 121        # tridiagonal P factors, 3*13
INVD_OFF = 160     # tridiagonal inverse diagonals, 3*13
U_OFF = 199        # tridiagonal upper factors, 3*12
C0_OFF = 235       # dense<-tridiagonal couplings, 3

DENSE_IDX = (0, 1, 2, 3, 4, 5, 6, 20, 34, 35, 36)
K_BASES = (7, 21, 37)
DENSE_C0 = (6, 7, 10)


class SchurEBSolver:
    def __init__(
        self, batches_per_block=128, block_dim=(128, 1, 1), precision=np.float32
    ):
        self.batches_per_block = batches_per_block
        self.block_dim = block_dim
        self.precision = precision
        self.ipiv_size = D_SIZE
        self.factorize_local = self._make_factorize_local()
        self.solve_local = self._make_solve_local()

    def a_local_size(self):
        return A_PAYLOAD

    def b_local_size(self):
        return B_PAYLOAD

    def _make_factorize_local(self):
        @cuda.jit(device=True)
        def factorize_device(lu, ipiv):
            # Dense LU with partial pivoting on the 11x11 Schur block (S_OFF).
            n = D_SIZE
            for i in range(n):
                max_val = 0.0
                pivot_row = i
                for k in range(i, n):
                    val = abs(lu[S_OFF + k * n + i])
                    if val > max_val:
                        max_val = val
                        pivot_row = k

                ipiv[i] = pivot_row

                if max_val == 0.0:
                    continue

                if pivot_row != i:
                    for j in range(n):
                        tmp = lu[S_OFF + i * n + j]
                        lu[S_OFF + i * n + j] = lu[S_OFF + pivot_row * n + j]
                        lu[S_OFF + pivot_row * n + j] = tmp

                pivot_val = lu[S_OFF + i * n + i]
                inv_pivot = 1.0 / pivot_val
                for k in range(i + 1, n):
                    lu[S_OFF + k * n + i] *= inv_pivot

                for k in range(i + 1, n):
                    factor = lu[S_OFF + k * n + i]
                    for j in range(i + 1, n):
                        lu[S_OFF + k * n + j] -= factor * lu[S_OFF + i * n + j]

        return factorize_device

    def _make_solve_local(self):
        @cuda.jit(device=True)
        def solve_device(lu, ipiv, rhs):
            r_dense = cuda.local.array(11, numba.float64)
            for d in range(11):
                r_dense[d] = rhs[DENSE_IDX[d]]

            Q_arr = cuda.local.array(39, numba.float64)

            # 1. Forward substitution (bottom-to-top) on tridiagonal RHS
            for b in range(3):
                kb = K_BASES[b]

                # i = 12
                r_12 = rhs[kb + 12]
                Q_arr[b * 13 + 12] = r_12 * lu[INVD_OFF + b * 13 + 12]

                # i = 11 down to 0
                for i in range(11, -1, -1):
                    r_i = rhs[kb + i]
                    u_i = lu[U_OFF + b * 12 + i]
                    Q_arr[b * 13 + i] = (
                        r_i - u_i * Q_arr[b * 13 + i + 1]
                    ) * lu[INVD_OFF + b * 13 + i]

                # 2. Update dense RHS
                dc0 = DENSE_C0[b]
                c0 = lu[C0_OFF + b]
                r_dense[dc0] -= c0 * Q_arr[b * 13 + 0]

            # 3. Dense LU solve for r_dense
            n = D_SIZE
            # Forward substitution (L)
            for i in range(n):
                pivot = ipiv[i]
                if pivot != i:
                    tmp = r_dense[i]
                    r_dense[i] = r_dense[pivot]
                    r_dense[pivot] = tmp
                for j in range(i):
                    r_dense[i] -= lu[S_OFF + i * n + j] * r_dense[j]

            # Backward substitution (U)
            for i in range(n - 1, -1, -1):
                for j in range(i + 1, n):
                    r_dense[i] -= lu[S_OFF + i * n + j] * r_dense[j]
                r_dense[i] /= lu[S_OFF + i * n + i]

            # Put dense solution back into rhs
            for d in range(11):
                rhs[DENSE_IDX[d]] = r_dense[d]

            # 4. Backward substitution (top-to-bottom) for tridiagonal variables
            for b in range(3):
                kb = K_BASES[b]
                dc0 = DENSE_C0[b]

                # x_0 = P_0 * x_dense + Q_0
                x_prev = lu[P_OFF + b * 13 + 0] * r_dense[dc0] + Q_arr[b * 13 + 0]
                rhs[kb + 0] = x_prev

                for i in range(1, 13):
                    x_i = lu[P_OFF + b * 13 + i] * x_prev + Q_arr[b * 13 + i]
                    rhs[kb + i] = x_i
                    x_prev = x_i

        return solve_device

    def assemble_lu_local_factory(self, jac_device, n_vars, lu_dtype):
        @cuda.jit(device=True)
        def assemble_lu(y_local, t, p_row, lu_buf, dtgamma_inv):
            values = jac_device(y_local, t, p_row)

            # 1. Initialize S (11x11)
            for r in range(11):
                r_orig = DENSE_IDX[r]
                for c in range(11):
                    c_orig = DENSE_IDX[c]
                    v = values[r_orig][c_orig]
                    if r == c:
                        lu_buf[S_OFF + r * 11 + c] = lu_dtype(dtgamma_inv - v)
                    else:
                        lu_buf[S_OFF + r * 11 + c] = lu_dtype(-v)

            # 2. Process tridiagonal blocks (Thomas factorization inline)
            for b in range(3):
                kb = K_BASES[b]
                dc0 = DENSE_C0[b]

                # Bottom-to-top Thomas factorization
                # i = 12
                v_d = values[kb + 12][kb + 12]
                m_d = dtgamma_inv - v_d
                inv_d = 1.0 / m_d
                lu_buf[INVD_OFF + b * 13 + 12] = lu_dtype(inv_d)

                v_l = values[kb + 12][kb + 11]
                m_l = -v_l
                lu_buf[P_OFF + b * 13 + 12] = lu_dtype(-m_l * inv_d)

                # i = 11 down to 0
                for idx in range(11, -1, -1):
                    v_d = values[kb + idx][kb + idx]
                    m_d = dtgamma_inv - v_d

                    v_u = values[kb + idx][kb + idx + 1]
                    m_u = -v_u
                    lu_buf[U_OFF + b * 12 + idx] = lu_dtype(m_u)

                    denom = m_d + m_u * lu_buf[P_OFF + b * 13 + idx + 1]
                    inv_d = 1.0 / denom
                    lu_buf[INVD_OFF + b * 13 + idx] = lu_dtype(inv_d)

                    if idx > 0:
                        v_l = values[kb + idx][kb + idx - 1]
                        m_l = -v_l
                    else:
                        # Coupling to dense
                        c0_orig = DENSE_IDX[dc0]
                        v_l = values[kb + 0][c0_orig]
                        m_l = -v_l

                    lu_buf[P_OFF + b * 13 + idx] = lu_dtype(-m_l * inv_d)

                # 3. Update S (Schur complement)
                c0_orig = DENSE_IDX[dc0]
                v_c0 = values[c0_orig][kb + 0]
                m_c0 = -v_c0
                lu_buf[C0_OFF + b] = lu_dtype(m_c0)

                # S_dense,dense += C_0 * P_0
                lu_buf[S_OFF + dc0 * 11 + dc0] += lu_dtype(
                    m_c0 * lu_buf[P_OFF + b * 13 + 0]
                )

        return assemble_lu
