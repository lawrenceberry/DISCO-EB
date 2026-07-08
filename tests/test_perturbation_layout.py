"""Tests for the compile-time perturbation state-vector layout."""

import pytest

import discoeb.perturbations as P
from discoeb.perturbation_layout import FLAT_MASSLESS_LAYOUT, PerturbationLayout

from cosmologies import BENCHMARK_COSMOLOGIES


def test_flat_massless_layout_matches_perturbations_module():
    """The default trimmed layout reproduces the fixed flat-LCDM indices (NVAR=50)."""

    L = FLAT_MASSLESS_LAYOUT
    assert L.nvar == P.NVAR == 50
    assert L.ix_etak == P.IX_ETAK
    assert L.ix_clxc == P.IX_CLXC
    assert L.ix_clxb == P.IX_CLXB
    assert L.ix_vb == P.IX_VB
    assert L.ix_g == P.IX_G
    assert L.ix_pol == P.IX_POL
    assert L.ix_r == P.IX_R
    assert not L.has_dark_energy
    assert not L.has_massive_neutrinos


def test_dark_energy_adds_two_state_variables():
    """Enabling the DE fluid appends exactly clxq, thetaq after the core."""

    base = PerturbationLayout()
    de = PerturbationLayout(enable_dark_energy=True)
    assert de.nvar == base.nvar + 2
    assert de.has_dark_energy
    assert de.ix_clxq == base.nvar
    assert de.ix_thetaq == base.nvar + 1


def test_massive_neutrinos_add_momentum_hierarchies():
    """Massive neutrinos append nqmax * (lmaxnu + 1) variables, multipole-major."""

    nqmax, lmaxnu = 5, 12
    L = PerturbationLayout(nqmax=nqmax, lmaxnu=lmaxnu)
    base = PerturbationLayout()
    assert L.has_massive_neutrinos
    assert L.nvar == base.nvar + nqmax * (lmaxnu + 1)
    assert L.ix_massive_nu == base.nvar
    # Multipole-major, bin-minor packing.
    assert L.ix_psi(0, 0) == base.nvar
    assert L.ix_psi(0, nqmax - 1) == base.nvar + nqmax - 1
    assert L.ix_psi(1, 0) == base.nvar + nqmax
    assert L.ix_psi(lmaxnu, nqmax - 1) == L.nvar - 1


def test_dark_energy_and_massive_neutrinos_stack():
    """DE variables precede the massive-neutrino block."""

    nqmax, lmaxnu = 4, 10
    L = PerturbationLayout(nqmax=nqmax, lmaxnu=lmaxnu, enable_dark_energy=True)
    base = PerturbationLayout()
    assert L.ix_clxq == base.nvar
    assert L.ix_thetaq == base.nvar + 1
    assert L.ix_massive_nu == base.nvar + 2
    assert L.nvar == base.nvar + 2 + nqmax * (lmaxnu + 1)


def test_disabled_components_raise_on_index_access():
    """Indices of disabled components are not addressable."""

    L = PerturbationLayout()
    with pytest.raises(AttributeError):
        _ = L.ix_clxq
    with pytest.raises(AttributeError):
        _ = L.ix_massive_nu
    with pytest.raises(AttributeError):
        _ = L.ix_psi(0, 0)


@pytest.mark.parametrize(
    "name,expect_de,expect_massive",
    [
        ("planck_2018_flat_lcdm", False, False),
        ("planck_2018_curved_lcdm", False, False),
        ("desi_2024_dynamical_dark_energy", True, False),
        ("planck_2018_flat_lcdm_massive_nu", False, True),
    ],
)
def test_from_cosmology_trims_to_enabled_components(name, expect_de, expect_massive):
    """from_cosmology enables only the components a cosmology needs."""

    cosmology = BENCHMARK_COSMOLOGIES[name]
    L = PerturbationLayout.from_cosmology(cosmology, nqmax=5)
    assert L.has_dark_energy == expect_de
    assert L.has_massive_neutrinos == expect_massive
    if not expect_de and not expect_massive:
        assert L.nvar == 50  # curvature does not change the layout
    if expect_massive:
        assert L.nvar == 50 + 5 * (L.lmaxnu + 1)
    if expect_de:
        assert L.nvar == 52
