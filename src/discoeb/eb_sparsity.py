"""Where the linear Einstein-Boltzmann Jacobian has its nonzeros.

modax's Rodas5P kernel takes a sparsity pattern and does two things with it: it
colours the column intersection graph, so the Jacobian costs ``n_colours + 1``
Enzyme sweeps rather than ``nvar + 1``; and it hands the pattern to
``modax._sparse_direct``, which orders it, factorises it symbolically and
compiles a sparse LU and a pair of sparse triangular solves for that exact
structure. Neither needs to know any physics -- only the components do.

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

Each state declares which of these it is (``State(..., tail_from=3)`` in
:mod:`discoeb.eb_model`), and :meth:`discoeb.component.Model.sparsity` assembles
the pattern; this module keeps the layout-level entry point.

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

from .eb_model import perturbation_model


def perturbation_sparsity(layout) -> np.ndarray:
    """The ``(nnz, 2)`` index array of the Jacobian's nonzeros for ``layout``."""

    return perturbation_model(layout).sparsity()
