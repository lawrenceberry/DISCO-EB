"""The Jacobian sparsity pattern, and what modax's analysis makes of it.

All host-side: the pattern is a property of the layout, and the ordering and
symbolic factorisation that turn it into a compiled solver run on the host when
the kernel is built. Nothing here needs a GPU.
"""

import numpy as np
import pytest

from discoeb.eb_sparsity import perturbation_sparsity
from discoeb.perturbations import PerturbationLayout

sparse_direct = pytest.importorskip("solvers._sparse_direct")
colour_sparsity = pytest.importorskip("solvers._sparsity").colour_sparsity
normalize_sparsity = pytest.importorskip("solvers._sparsity").normalize_sparsity


LAYOUTS = {
    "flat-lcdm": PerturbationLayout(),
    "dark-energy": PerturbationLayout(enable_dark_energy=True),
    "massive-nu": PerturbationLayout(nqmax=3, lmaxnu=8),
    # The hand-written Schur solver required every tridiagonal block to have the
    # same length, so this layout used to raise NotImplementedError.
    "ragged-blocks": PerturbationLayout(
        lmaxg=11, lmaxpol=13, lmaxr=17, nqmax=2, lmaxnu=9, enable_dark_energy=True
    ),
}


@pytest.mark.parametrize("name", list(LAYOUTS))
def test_the_pattern_covers_the_whole_state(name):
    layout = LAYOUTS[name]
    pattern = perturbation_sparsity(layout)

    assert pattern.ndim == 2 and pattern.shape[1] == 2
    assert pattern.min() >= 0 and pattern.max() < layout.nvar
    # Every variable appears on the diagonal, and every variable couples to
    # something: a state index the pattern never mentions is one the sparse
    # solver would silently drop.
    entries = {(int(r), int(c)) for r, c in pattern}
    assert {(i, i) for i in range(layout.nvar)} <= entries


def test_the_flat_lcdm_pattern_is_the_expected_size():
    """50 variables: an 11-variable dense core and three 13-variable tails."""
    layout = LAYOUTS["flat-lcdm"]
    assert layout.nvar == 50
    # 11 * 11 core + 3 * (13 diagonal + 12 super + 12 sub + 2 couplings)
    assert len(perturbation_sparsity(layout)) == 11 * 11 + 3 * 39 == 238


@pytest.mark.parametrize("name", list(LAYOUTS))
def test_amd_finds_a_perfect_elimination_order(name):
    """This structure fills in not at all, so ``L + U`` is the pattern itself.

    Minimum degree peels each free-streaming hierarchy from its truncated end,
    where every variable has degree two, and reaches the densely coupled core
    with nothing left to fill -- which is the elimination the hand-written Schur
    solver performed by hand, found here without being told the structure.
    """
    layout = LAYOUTS[name]
    pattern = normalize_sparsity(perturbation_sparsity(layout), layout.nvar)
    layout_lu = sparse_direct.analyse(pattern, "amd")

    assert layout_lu.nnz == sum(len(cols) for cols in pattern)
    core_size = layout_lu.n_vars - sum(
        len(tail) for _, tail in _hierarchies(layout)
    )
    assert set(layout_lu.order[-core_size:]) == set(_core(layout))


def test_the_solver_and_the_kernel_share_one_layout():
    """The slots the sweeps write are the slots the factorisation reads."""
    layout = LAYOUTS["flat-lcdm"]
    solver = sparse_direct.sparse_direct_solver(
        perturbation_sparsity(layout), layout.nvar
    )
    pattern = normalize_sparsity(perturbation_sparsity(layout), layout.nvar)

    assert solver.compressed.n_colours == 12  # 13 Enzyme sweeps, not 51
    assert solver.compressed.size == solver.nnz == 238
    slots = [solver.compressed.slot(r, c) for r, cols in enumerate(pattern) for c in cols]
    assert sorted(slots) == list(range(238)), "every slot claimed exactly once"


def test_the_colouring_keeps_every_entry_apart():
    """Two columns of one colour sharing a row would share a slot.

    modax checks this when it colours; asserting it here is what says the
    *pattern* is the reason it holds, rather than luck in the colouring.
    """
    for name, layout in LAYOUTS.items():
        pattern = normalize_sparsity(perturbation_sparsity(layout), layout.nvar)
        compressed = colour_sparsity(pattern)
        for row, cols in enumerate(pattern):
            colours = {compressed.colour[c] for c in cols}
            assert len(colours) == len(cols), f"{name}: row {row} collides"


def _core(layout):
    from discoeb.eb_sparsity import _blocks

    return _blocks(layout)[0]


def _hierarchies(layout):
    from discoeb.eb_sparsity import _blocks

    return _blocks(layout)[1]


def test_the_core_and_the_tails_partition_the_state():
    """No variable is in two blocks, and none is in neither."""
    for name, layout in LAYOUTS.items():
        core = _core(layout)
        tails = [i for _, tail in _hierarchies(layout) for i in tail]
        assert len(set(core)) == len(core), f"{name}: duplicate in the core"
        assert sorted(core + tails) == list(range(layout.nvar)), name
        assert np.all(np.diff(sorted(tails)) > 0)
