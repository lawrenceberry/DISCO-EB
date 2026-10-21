"""The Einstein-Boltzmann component model, on the host.

The equations themselves are gated on the GPU against CLASS
(``test_matter_power_spectrum_matches_class``); here the model's assembly is
checked: that its state order is the one every index in the code base assumes,
that each layout builds and runs as plain Python, and that the sums the
Einstein constraints read are the sums over the species the layout enables.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from discoeb.component import Component
from discoeb.eb_model import MassiveNeutrinos, Photons, layout_state_names, perturbation_model
from discoeb.perturbations import PARAM_COLUMNS, N_PARAM, PerturbationLayout, make_initial_states

HOST = dict(
    jit=lambda f: f,
    cuda=SimpleNamespace(local=SimpleNamespace(array=lambda shape, dtype: np.zeros(shape))),
    float64=np.float64,
)

LAYOUTS = {
    "flat-lcdm": PerturbationLayout(),
    "dark-energy": PerturbationLayout(enable_dark_energy=True),
    "massive-nu": PerturbationLayout(nqmax=3, lmaxnu=8),
    "ragged-blocks": PerturbationLayout(
        lmaxg=11, lmaxpol=13, lmaxr=17, nqmax=2, lmaxnu=9, enable_dark_energy=True
    ),
}


class ConstantThermalHistory(Component):
    """A thermal history frozen at one epoch, standing in for the spline tables."""

    def __init__(self, a=1e-3, opacity=50.0, cs2=1e-8, rhonu=3.0):
        self.a = lambda: a
        self.opacity = lambda: opacity
        self.cs2 = lambda: cs2
        self.rhonu = lambda: rhonu


def build(layout):
    model = perturbation_model(layout, ConstantThermalHistory())
    return model, model.build(PARAM_COLUMNS, **HOST)


@pytest.mark.parametrize("name", list(LAYOUTS))
def test_the_state_order_is_the_layout(name):
    layout = LAYOUTS[name]
    model, _ = build(layout)
    assert tuple(model.state) == layout_state_names(layout)
    assert model.nvar == layout.nvar
    assert model.index["etak"] == layout.ix_etak
    assert model.index["vb"] == layout.ix_vb
    assert model.index["theta_0"] == layout.ix_g
    assert model.index[f"theta_{layout.lmaxg}"] == layout.ix_g + layout.lmaxg
    assert model.index["e_0"] == layout.ix_pol  # E_2
    assert model.index[f"e_{layout.lmaxpol - 2}"] == layout.ix_pol + layout.lmaxpol - 2
    assert model.index["n_0"] == layout.ix_r
    if layout.enable_dark_energy:
        assert model.index["clxq"] == layout.ix_clxq
    if layout.has_massive_neutrinos:
        for q in range(layout.nqmax):
            for l in range(layout.lmaxnu + 1):
                assert model.index[f"psi_{q}_{l}"] == layout.ix_psi(l, q)


@pytest.mark.parametrize("name", list(LAYOUTS))
def test_the_rhs_builds_and_is_finite(name):
    layout = LAYOUTS[name]
    model, rhs = build(layout)
    rng = np.random.default_rng(0)
    p = np.zeros(N_PARAM)
    p[:5] = [1.5e-7, 1.0e-7, 1.1e-4, 1.9e-5, 2.0e-7]
    p[PARAM_COLUMNS["k"]] = 0.05
    p[PARAM_COLUMNS["w_DE_0"]], p[PARAM_COLUMNS["cs2_DE"]] = -0.9, 1.0
    p[PARAM_COLUMNS["grhor_nu"]], p[PARAM_COLUMNS["n_mnu"]], p[PARAM_COLUMNS["amnu"]] = 1e-7, 1.0, 300.0
    y = rng.standard_normal(layout.nvar)
    dy = np.asarray(rhs(tuple(y), 100.0, tuple(p)))
    assert dy.shape == (layout.nvar,) and np.isfinite(dy).all()


def test_the_einstein_sources_sum_over_the_enabled_species():
    model, _ = build(LAYOUTS["flat-lcdm"])
    lines = [l.strip() for l in model.source.splitlines()]
    assert (
        "grho = Curvature__grho(grhok) + ColdDarkMatter__grho(grhoc_t) + Baryons__grho(grhob_t)"
        " + Photons__grho(grhog_t) + MasslessNeutrinos__grho(grhor_t) + CosmologicalConstant__grho(grhov_t)"
    ) in lines
    assert "dgq = Baryons__dgq(grhob_t, vb) + Photons__dgq(grhog_t, qg) + MasslessNeutrinos__dgq(grhor_t, n)" in lines
    assert "rhonu" not in model.source  # the massive-neutrino table is not read

    model, _ = build(LAYOUTS["ragged-blocks"])
    assert "DarkEnergy__grho(grhov_t) + MassiveNeutrinos__grho(nu_norm, rhonu)" in model.source
    assert "DarkEnergy__dgq(grhov_t, w_Q, thetaq, k) + MassiveNeutrinos__dgq(nu_norm, psi)" in model.source
    assert "psi = cuda.local.array((2, 10), float64)" in model.source
    assert "MassiveNeutrinos__d_psi(k, z, sigma, a, amnu, psi, inv_tau, c_MassiveNeutrinos__d_psi)" in model.source
    assert "d_psi[i, j] += c_MassiveNeutrinos__d_psi[i, j]" in model.source


def test_thomson_scattering_writes_the_photon_and_baryon_derivatives():
    model, _ = build(LAYOUTS["flat-lcdm"])
    lines = [l.strip() for l in model.source.splitlines()]
    assert "d_vb = Baryons__d_vb(adotoa, vb, k, cs2, clxb) + ThomsonScattering__d_vb(photbar, opacity, slip)" in lines
    assert lines.index("Photons__d_theta(k, z, sigma, theta, inv_tau, c_Photons__d_theta)") < lines.index(
        "ThomsonScattering__d_theta(opacity, slip, polter, theta, c_ThomsonScattering__d_theta)"
    )
    assert "ThomsonScattering__d_e(opacity, polter, e, c_ThomsonScattering__d_e)" in lines
    # Neutrinos are collisionless: one writer.
    assert sum("__d_n(" in l for l in lines) == 1


def test_a_hierarchy_needs_an_octupole():
    with pytest.raises(ValueError, match="octupole"):
        build(PerturbationLayout(lmaxg=2))


def test_the_truncations_are_the_components_settings():
    """The layout's lmax values reach the components, which size their hierarchies."""
    layout = LAYOUTS["ragged-blocks"]
    model, _ = build(layout)
    photons = next(c for c in model.components if isinstance(c, Photons))
    assert (photons.LMAX_G, photons.LMAX_POL) == (11, 13)
    assert (model.states["theta"].shape, model.states["e"].shape) == ((12,), (12,))
    neutrinos = next(c for c in model.components if isinstance(c, MassiveNeutrinos))
    assert (neutrinos.NQMAX, neutrinos.LMAX_NU) == (2, 9)
    assert model.states["psi"].shape == (2, 10) and len(neutrinos.weight) == 2
    # The defaults are the DISCO_LMAX_* environment settings, 15 unless set.
    assert Photons.LMAX_G == PerturbationLayout().lmaxg


def test_the_initial_state_and_sparsity_come_from_the_components():
    """The adiabatic mode: etak ~ -k, matter at 3/4 of the photon density, zero above the octupole."""
    layout = LAYOUTS["ragged-blocks"]
    model, _ = build(layout)
    cosmology = SimpleNamespace(w_DE_0=-0.9, w_DE_a=0.1)
    densities = (1.5e-7, 1.0e-7, 1.1e-4, 1.9e-5, 2.0e-7)
    k = np.array([1e-3, 1e-1])
    y0 = model.initial_state(
        k, tau=0.1, grhog=densities[0], grhornomass=densities[1], grhoc=densities[2],
        grhob=densities[3], grhov=densities[4], w_DE_0=-0.9, w_DE_a=0.1,
    )
    assert y0.shape == (2, layout.nvar) and np.isfinite(y0).all()
    np.testing.assert_allclose(y0[:, model.index["etak"]], -k, rtol=1e-3)
    np.testing.assert_allclose(y0[:, model.index["clxc"]], 0.75 * y0[:, model.index["theta_0"]])
    np.testing.assert_allclose(y0[:, model.index["clxq"]], (1.0 - 0.9) * y0[:, model.index["clxc"]])
    assert (y0[:, model.index["theta_4"]:model.index["e_0"]] == 0.0).all()
    assert (y0[:, model.index["n_4"]:model.index["n_17"] + 1] == 0.0).all()
    np.testing.assert_allclose(y0[:, model.index["psi_1_0"]], -0.25 * y0[:, model.index["n_0"]] * model.components[-2].dlnf0[1])
    # The sparsity pattern is read off the same states.
    pattern = model.sparsity()
    assert pattern.shape[1] == 2 and pattern.max() < layout.nvar
    assert len(perturbation_model(PerturbationLayout()).sparsity()) == 11 * 11 + 3 * 39


def test_make_initial_states_uses_the_model():
    from discoeb.perturbations import _as_cosmology

    cosmology = _as_cosmology(
        dict(
            Omegam=0.31, Omegab=0.049, w_DE_0=-1.0, w_DE_a=0.0, cs2_DE=1.0, Omegak=0.0, A_s=2.1e-9,
            n_s=0.966, H0=67.0, Tcmb=2.7255, YHe=0.245, Neff=3.046, Nmnu=0.0, mnu=0.0, k_p=0.05,
        )
    )
    y0 = make_initial_states(cosmology, np.array([1e-2, 1e-1]), PerturbationLayout())
    assert y0.shape == (2, 50) and np.isfinite(y0).all()
    np.testing.assert_allclose(y0[:, 0], [-1e-2, -1e-1], rtol=1e-3)  # etak ~ -k
