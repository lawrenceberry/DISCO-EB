"""Where the linear Einstein-Boltzmann Jacobian has its nonzeros.

modax's Rodas5P kernel takes a sparsity pattern and does two things with it: it
colours the column intersection graph, so the Jacobian costs ``n_colours + 1``
Enzyme sweeps rather than ``nvar + 1``; and it hands the pattern to
``solvers._sparse_direct``, which orders it, factorises it symbolically and
compiles a sparse LU and a pair of sparse triangular solves for that exact
structure. Neither needs to know any physics -- only this module does.

The structure is a densely coupled **core** bordered by free-streaming
**hierarchies**:

* The core holds the metric perturbation, the CDM and baryon fluid, the lowest
  three photon and massless-neutrino multipoles, the lowest polarization
  multipole, the dark-energy fluid when it is evolved, and the lowest three
  phase-space multipoles of every massive-neutrino momentum bin. Everything in
  it reaches everything else through the Einstein constraints (``dgrho``,
  ``dgq``, ``z``, ``sigma``), so it is declared dense.
* Each hierarchy -- photon temperature ``Theta_3..lmaxg``, E-mode polarization
  ``E_3..lmaxpol``, massless neutrinos ``N_3..lmaxr``, and one tail per massive
  neutrino bin -- is tridiagonal: multipole ``l`` couples to ``l - 1`` and
  ``l + 1`` and to nothing else. Each tail meets the core only at its lowest
  multipole, through the quadrupole it streams out of, which is one entry in
  each direction.

Declaring the core dense makes this a **superset** of the true Jacobian, which
is what the pattern must be: colouring a superset only costs sweeps, colouring a
subset silently corrupts the entries where two columns of a colour group turn out
to overlap.

The pattern need not include the fill-in the factorisation produces -- that is
what the symbolic factorisation is for. On this structure it produces none:
minimum degree peels each hierarchy from its truncated end inwards, where every
variable has degree two, and reaches the core with nothing left to fill.
"""

import numpy as np


def perturbation_sparsity(layout) -> np.ndarray:
    """The ``(nnz, 2)`` index array of the Jacobian's nonzeros for ``layout``."""

    core, hierarchies = _blocks(layout)
    entries = set()
    for r in core:
        for c in core:
            entries.add((r, c))
    for parent, tail in hierarchies:
        for i, idx in enumerate(tail):
            entries.add((idx, idx))
            if i:
                entries.add((idx, tail[i - 1]))
            if i + 1 < len(tail):
                entries.add((idx, tail[i + 1]))
        entries.add((tail[0], parent))
        entries.add((parent, tail[0]))
    return np.array(sorted(entries), dtype=np.int64)


def _blocks(layout):
    """``(core_indices, [(parent, tail_indices), ...])`` for a layout."""

    core = [layout.ix_etak, layout.ix_clxc, layout.ix_clxb, layout.ix_vb]
    hierarchies = []

    # Photon temperature: Theta_0..2 in the core, Theta_3.. streaming off the
    # quadrupole Theta_2.
    core += [layout.ix_g + l for l in (0, 1, 2)]
    hierarchies.append(
        (layout.ix_g + 2, [layout.ix_g + l for l in range(3, layout.lmaxg + 1)])
    )

    # E-mode polarization starts at l = 2, so E_2 is the core member and the
    # tail is E_3.. -- stored from ix_pol + (l - 2).
    core.append(layout.ix_pol)
    hierarchies.append(
        (layout.ix_pol, [layout.ix_pol + l - 2 for l in range(3, layout.lmaxpol + 1)])
    )

    core += [layout.ix_r + l for l in (0, 1, 2)]
    hierarchies.append(
        (layout.ix_r + 2, [layout.ix_r + l for l in range(3, layout.lmaxr + 1)])
    )

    if layout.enable_dark_energy:
        core += [layout.ix_clxq, layout.ix_thetaq]

    if layout.has_massive_neutrinos:
        for q in range(layout.nqmax):
            core += [layout.ix_psi(l, q) for l in (0, 1, 2)]
            hierarchies.append(
                (
                    layout.ix_psi(2, q),
                    [layout.ix_psi(l, q) for l in range(3, layout.lmaxnu + 1)],
                )
            )

    return core, hierarchies
