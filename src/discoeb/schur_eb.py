"""Thread-local Schur / Einstein-Boltzmann block-LU solver (numba-cuda).

This is the ``linear_solver`` consumed by modax's Rodas5P kernel
(``solvers.rodas5P.solve``): it factorizes and solves the per-step Rosenbrock
linear system ``(1/(dt*gamma)) I - J`` entirely in thread-local memory,
exploiting the bordered-block sparsity of the linear Einstein-Boltzmann
Jacobian.

The state splits into a densely-coupled "core" (metric + fluid + the lowest
photon / polarization / neutrino multipoles, and -- when enabled -- the
dark-energy fluid perturbations) bordered by free-streaming tridiagonal
hierarchies (photon temperature, E-mode polarization, massless neutrinos). Each
tridiagonal block couples back into the dense core only through its lowest
multipole, so the system is solved by a Schur complement onto the dense core
plus one Thomas sweep per block.

The buffer is **not** this class's own layout any more. modax colours the
sparsity pattern and stores the Jacobian column-compressed, entry ``(r, c)`` at
``r * n_colours + colour[c]``; :meth:`bind` takes that layout and builds the
slot tables the device functions index through. The pattern handed to modax is
:meth:`sparsity`: everything the factorization reads *or writes*, the dense
core's LU fill-in included, which makes it a superset of the true nonzeros and
leaves every factor a slot of its own.

The factorization is in place, and the factors reuse the slots of the entries
they are computed from: ``P`` lands on the block diagonals and ``invD`` on the
lower band, while ``U`` and the block-to-core couplings are already where they
belong and are never moved.

The dense-core membership (``dense_idx``), tridiagonal block bases
(``k_bases``), block-to-core couplings (``dense_c0``), block length
(``tridiag_len``) and state size (``nvar``) are constructor parameters, so the
same solver serves the flat-LambdaCDM + massless layout (11-var core, default)
and its dark-energy extension (13-var core). Requires ``numba-cuda-mlir`` and a
visible GPU.
"""

import numpy as np
from numba_cuda_mlir import cuda, types

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
        *,
        dense_idx=DENSE_IDX,
        k_bases=K_BASES,
        dense_c0=DENSE_C0,
        tridiag_len=A_SIZE,
        nvar=B_PAYLOAD,
    ):
        self.dense_idx = tuple(int(i) for i in dense_idx)
        self.k_bases = tuple(int(i) for i in k_bases)
        self.dense_c0 = tuple(int(i) for i in dense_c0)
        self.tridiag_len = int(tridiag_len)
        self.nvar = int(nvar)
        self.d_size = len(self.dense_idx)
        self.n_blocks = len(self.k_bases)
        # Only the dense core is pivoted; the tridiagonal blocks are swept.
        self.ipiv_size = self.d_size
        self.factorize_local = None
        self.solve_local = None

    # --- what the factorization touches --------------------------------------
    def jacobian_slots(self):
        """The ``(row, column)`` entries the factorization reads or writes."""
        entries = set()
        tl = self.tridiag_len
        for r in self.dense_idx:
            for c in self.dense_idx:
                entries.add((r, c))  # the core, whole: its LU fills in
        for b, kb in enumerate(self.k_bases):
            dense_orig = self.dense_idx[self.dense_c0[b]]
            for i in range(tl):
                entries.add((kb + i, kb + i))
            for i in range(tl - 1):
                entries.add((kb + i, kb + i + 1))
            entries.add((kb, dense_orig))
            for i in range(1, tl):
                entries.add((kb + i, kb + i - 1))
            entries.add((dense_orig, kb))
        return entries

    def sparsity(self):
        """The pattern to hand modax, as an ``(nnz, 2)`` index array."""
        return np.array(sorted(self.jacobian_slots()), dtype=np.int64)

    # --- binding to a compressed layout --------------------------------------
    def bind(self, compressed):
        """Build the device functions against ``compressed``'s slot map."""
        n, nb, tl = self.d_size, self.n_blocks, self.tridiag_len
        slot = compressed.slot

        core = np.array(
            [
                [slot(self.dense_idx[r], self.dense_idx[c]) for c in range(n)]
                for r in range(n)
            ],
            dtype=np.int32,
        ).ravel()
        diag = np.zeros((nb, tl), dtype=np.int32)  # P lands here
        lower = np.zeros((nb, tl), dtype=np.int32)  # invD lands here
        upper = np.zeros((nb, tl - 1), dtype=np.int32)  # U is already here
        c0 = np.zeros(nb, dtype=np.int32)
        for b, kb in enumerate(self.k_bases):
            dense_orig = self.dense_idx[self.dense_c0[b]]
            for i in range(tl):
                diag[b, i] = slot(kb + i, kb + i)
                lower[b, i] = (
                    slot(kb, dense_orig) if i == 0 else slot(kb + i, kb + i - 1)
                )
            for i in range(tl - 1):
                upper[b, i] = slot(kb + i, kb + i + 1)
            c0[b] = slot(dense_orig, kb)

        tables = (core, diag.ravel(), lower.ravel(), upper.ravel(), c0)
        self.factorize_local = self._make_factorize(*tables)
        self.solve_local = self._make_solve(*tables)
        return self

    def _make_factorize(self, core, diag, lower, upper, c0):
        n = self.d_size
        n_blocks = self.n_blocks
        tl = self.tridiag_len
        last = tl - 1
        dense_c0 = self.dense_c0

        @cuda.jit(device=True)
        def factorize_device(lu, ipiv):
            s_slot = cuda.const.array_like(core)
            d_slot = cuda.const.array_like(diag)
            l_slot = cuda.const.array_like(lower)
            u_slot = cuda.const.array_like(upper)
            c_slot = cuda.const.array_like(c0)

            # In place: each entry is read into a register before the factor
            # that replaces it is written. U and the core couplings already sit
            # where they belong, so only P (over the block diagonals) and invD
            # (over the lower band) are written here.
            for b in range(n_blocks):
                m_d = lu[d_slot[b * tl + last]]
                m_l = lu[l_slot[b * tl + last]]
                inv_d = 1.0 / m_d
                lu[l_slot[b * tl + last]] = inv_d
                lu[d_slot[b * tl + last]] = -m_l * inv_d

                for idx in range(last - 1, -1, -1):
                    m_d = lu[d_slot[b * tl + idx]]
                    m_u = lu[u_slot[b * (tl - 1) + idx]]
                    # At idx == 0 this is the block's coupling to the dense
                    # core; subsequent entries are the ordinary subdiagonal.
                    m_l = lu[l_slot[b * tl + idx]]

                    denom = m_d + m_u * lu[d_slot[b * tl + idx + 1]]
                    inv_d = 1.0 / denom
                    lu[l_slot[b * tl + idx]] = inv_d
                    lu[d_slot[b * tl + idx]] = -m_l * inv_d

                # Schur-complement update: S_{dc0,dc0} += C_0 * P_0.
                m_c0 = lu[c_slot[b]]
                dc0 = dense_c0[b]
                lu[s_slot[dc0 * n + dc0]] += m_c0 * lu[d_slot[b * tl + 0]]

            # Dense LU with partial pivoting on the d_size x d_size Schur block.
            for i in range(n):
                max_val = 0.0
                pivot_row = i
                for kk in range(i, n):
                    val = abs(lu[s_slot[kk * n + i]])
                    if val > max_val:
                        max_val = val
                        pivot_row = kk

                ipiv[i] = pivot_row

                if max_val == 0.0:
                    continue

                if pivot_row != i:
                    for j in range(n):
                        tmp = lu[s_slot[i * n + j]]
                        lu[s_slot[i * n + j]] = lu[s_slot[pivot_row * n + j]]
                        lu[s_slot[pivot_row * n + j]] = tmp

                inv_pivot = 1.0 / lu[s_slot[i * n + i]]
                for kk in range(i + 1, n):
                    lu[s_slot[kk * n + i]] *= inv_pivot

                for kk in range(i + 1, n):
                    factor = lu[s_slot[kk * n + i]]
                    for j in range(i + 1, n):
                        lu[s_slot[kk * n + j]] -= factor * lu[s_slot[i * n + j]]

        return factorize_device

    def _make_solve(self, core, diag, lower, upper, c0):
        n = self.d_size
        n_blocks = self.n_blocks
        tl = self.tridiag_len
        last = tl - 1
        dense_c0 = self.dense_c0
        dense_idx = self.dense_idx
        k_bases = self.k_bases
        q_size = self.n_blocks * self.tridiag_len

        @cuda.jit(device=True)
        def solve_device(lu, ipiv, rhs):
            s_slot = cuda.const.array_like(core)
            d_slot = cuda.const.array_like(diag)
            l_slot = cuda.const.array_like(lower)
            u_slot = cuda.const.array_like(upper)
            c_slot = cuda.const.array_like(c0)

            r_dense = cuda.local.array(n, types.float64)
            for d in range(n):
                r_dense[d] = rhs[dense_idx[d]]

            Q_arr = cuda.local.array(q_size, types.float64)

            # 1. Forward substitution (bottom-to-top) on each tridiagonal RHS.
            for b in range(n_blocks):
                kb = k_bases[b]
                Q_arr[b * tl + last] = rhs[kb + last] * lu[l_slot[b * tl + last]]
                for i in range(last - 1, -1, -1):
                    r_i = rhs[kb + i]
                    u_i = lu[u_slot[b * (tl - 1) + i]]
                    Q_arr[b * tl + i] = (r_i - u_i * Q_arr[b * tl + i + 1]) * lu[
                        l_slot[b * tl + i]
                    ]

                # 2. Update dense RHS through the block-to-core coupling.
                dc0 = dense_c0[b]
                r_dense[dc0] -= lu[c_slot[b]] * Q_arr[b * tl + 0]

            # 3. Dense LU solve for r_dense.
            for i in range(n):
                pivot = ipiv[i]
                if pivot != i:
                    tmp = r_dense[i]
                    r_dense[i] = r_dense[pivot]
                    r_dense[pivot] = tmp
                for j in range(i):
                    r_dense[i] -= lu[s_slot[i * n + j]] * r_dense[j]

            for i in range(n - 1, -1, -1):
                for j in range(i + 1, n):
                    r_dense[i] -= lu[s_slot[i * n + j]] * r_dense[j]
                r_dense[i] /= lu[s_slot[i * n + i]]

            for d in range(n):
                rhs[dense_idx[d]] = r_dense[d]

            # 4. Back substitution (top-to-bottom) for the tridiagonal variables.
            for b in range(n_blocks):
                kb = k_bases[b]
                dc0 = dense_c0[b]
                x_prev = lu[d_slot[b * tl + 0]] * r_dense[dc0] + Q_arr[b * tl + 0]
                rhs[kb + 0] = x_prev
                for i in range(1, tl):
                    x_i = lu[d_slot[b * tl + i]] * x_prev + Q_arr[b * tl + i]
                    rhs[kb + i] = x_i
                    x_prev = x_i

        return solve_device
