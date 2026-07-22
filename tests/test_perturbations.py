import pytest
import jax
import jax.numpy as jnp
import numpy as np


from discoeb.background import evolve_background
from discoeb.perturbations import evolve_perturbations, evolve_perturbations_batched

from conftest import k_CLASS, Pkbc_CLASS

## Cosmological Parameters
Tcmb    = 2.7255
YHe     = 0.248
Omegam  = 0.3099
Omegab  = 0.0488911
Omegac  = Omegam - Omegab
w_DE_0  = -0.99
w_DE_a  = 0.0
cs2_DE  = 1.0

# Initialize neutrinos.
num_massive_neutrinos = 1
mnu     = 0.06  #eV
Tnu     = (4/11)**(1/3) #0.71611 # Tncdm of CLASS
Neff    = 3.046 # -1 if massive neutrino present
N_nu_mass = 1
N_nu_rel = Neff - N_nu_mass * (Tnu/((4/11)**(1/3)))**4
h       = 0.67742
A_s     = 2.1064e-09
n_s     = 0.96822
k_p     = 0.05

# modes to sample
nmodes = 512
kmin = 1e-5
kmax = 1e+2
kmax_small=1e+1
aexp = 0.01
aexp_out = jnp.array([aexp]) 


## Compute Background evolution
param = {}
param['Omegam']  = Omegam
param['Omegab']  = Omegab
param['w_DE_0']  = w_DE_0
param['w_DE_a']  = w_DE_a
param['cs2_DE']  = cs2_DE
param['Omegak']  = 0.0
param['A_s']     = A_s
param['n_s']     = n_s
param['H0']      = 100*h
param['Tcmb']    = Tcmb
param['YHe']     = YHe
param['Neff']    = N_nu_rel
param['Nmnu']    = N_nu_mass
param['mnu']     = mnu

iout = -1
fac = 2 * jnp.pi**2 * A_s

relevant_indices = (k_CLASS >= kmin) & (k_CLASS <= kmax)
test_points = k_CLASS[relevant_indices]
test_values = Pkbc_CLASS[relevant_indices]

small_kmax_indices = (k_CLASS >= kmin) & (k_CLASS <= kmax_small)
small_kmax_test_points = k_CLASS[relevant_indices]
small_kmax_test_values = Pkbc_CLASS[relevant_indices]

@jax.jit
def perturbations_jit(param): 
    return evolve_perturbations( 
        param=param, aexp_out=aexp_out, kmin=kmin, kmax=kmax, num_k=256,
        lmaxg = 11, lmaxgp = 11, lmaxr = 11, lmaxnu  = 8,
        nqmax = 3, rtol = 1e-3, atol = 1e-3,
        pcoeff = 0.25, icoeff = 0.80, dcoeff = 0.0,
        factormax  = 20.0, factormin  = 0.3, max_steps  = 2048)

@jax.jit
def perturbations_batched_jit(param):
    return evolve_perturbations_batched(param=param,
        aexp_out=aexp_out, kmin=kmin, kmax=kmax, num_k=256,
        lmaxg = 11, lmaxgp = 11, lmaxr = 11, lmaxnu  = 8,
        nqmax = 3, rtol = 1e-3, atol = 1e-3,
        pcoeff = 0.25, icoeff = 0.80, dcoeff = 0.0,
        factormax  = 20.0, factormin  = 0.3, max_steps  = 2048, batch_size=32)

@pytest.fixture(scope="session")
def background_param():
    bg_param = evolve_background(param=param, thermo_module='RECFAST')
    yield bg_param

class TestEvolvePerturbations:

    @pytest.mark.parametrize("batch_size", [4, 8, 16, 32, 64])
    def test_power_spectrum_varying_batchsize_small_kmax(self, batch_size, background_param):

        y, kmodes, _ = evolve_perturbations_batched(
            param=background_param, kmin=kmin, kmax=kmax_small, num_k=nmodes, aexp_out=aexp_out,
            lmaxg = 31, lmaxgp = 31, lmaxr = 31, lmaxnu = 31, nqmax = 5,
            max_steps=2048, rtol=1e-4, atol=1e-4, batch_size=batch_size
        )
        
        Pkbc = fac *(kmodes/k_p)**(n_s - 1) * kmodes**(-3) * y[:,iout,6]**2
        disco_eb_values = jnp.interp(small_kmax_test_points, kmodes, Pkbc)

        assert jnp.allclose(disco_eb_values, small_kmax_test_values, rtol=0.1), "relative error exceeded 0.1"
        assert jnp.allclose(disco_eb_values, small_kmax_test_values, rtol=0.01), "relative error exceeded 0.01"
        assert jnp.allclose(disco_eb_values, small_kmax_test_values, rtol=0.005), "relative error exceeded 0.005"


    def test_power_spectrum_vs_CLASS_batched(self, background_param):

        y, kmodes, _ = evolve_perturbations_batched(
            param=background_param, kmin=kmin, kmax=kmax, num_k=nmodes, aexp_out=aexp_out,
            lmaxg = 31, lmaxgp = 31, lmaxr = 31, lmaxnu = 31, nqmax = 5,
            max_steps=2048, rtol=1e-4, atol=1e-4, batch_size=32
        )
        
        Pkbc = fac *(kmodes/k_p)**(n_s - 1) * kmodes**(-3) * y[:,iout,6]**2
        disco_eb_values = jnp.interp(test_points, kmodes, Pkbc)
        
        assert jnp.allclose(disco_eb_values, test_values, rtol=0.1), "relative error exceeded 0.1"
        assert jnp.allclose(disco_eb_values, test_values, rtol=0.01), "relative error exceeded 0.01"
        assert jnp.allclose(disco_eb_values, test_values, rtol=0.005), "relative error exceeded 0.005"


    def test_power_spectrum_vs_CLASS(self, background_param):

        y, kmodes, _ = evolve_perturbations(
            param=background_param, kmin=kmin, kmax=kmax, num_k=nmodes, aexp_out=aexp_out,
            lmaxg = 31, lmaxgp = 31, lmaxr = 31, lmaxnu = 31, nqmax = 5,
            max_steps=2048, rtol=1e-5, atol=1e-5
        )
        
        
        Pkbc = fac *(kmodes/k_p)**(n_s - 1) * kmodes**(-3) * y[:,iout,6]**2
        disco_eb_values = jnp.interp(test_points, kmodes, Pkbc)

        assert jnp.allclose(disco_eb_values, test_values, rtol=0.1), "relative error exceeded 0.1"
        assert jnp.allclose(disco_eb_values, test_values, rtol=0.01), "relative error exceeded 0.01"
        assert jnp.allclose(disco_eb_values, test_values, rtol=0.005), "relative error exceeded 0.005"

    def test_benchmark_evolve_perturbations(self, benchmark, background_param):
        _ = perturbations_jit(param=background_param)

        _ = benchmark(perturbations_jit, param=background_param)


    def test_benchmark_evolve_perturbations_batched(self, benchmark, background_param):
        _ = perturbations_batched_jit(param=background_param)

        _ = benchmark(perturbations_batched_jit, param=background_param)
REFERENCE_COSMOLOGY = {
    "Omegam": (0.02238280 + 0.1201075) / 0.6732117**2,
    "Omegab": 0.02238280 / 0.6732117**2,
    "w_DE_0": -1.0,
    "w_DE_a": 0.0,
    "cs2_DE": 1.0,
    "Omegak": 0.0,
    "A_s": 2.100549e-9,
    "n_s": 0.9660499,
    "H0": 67.32117,
    "Tcmb": 2.7255,
    "YHe": 0.2454006,
    "Neff": 3.046,
    "Nmnu": 0.0,
    "mnu": 0.0,
    "k_p": 0.05,
}

MATTER_POWER_K = np.geomspace(1.0e-4, 1.0, 128, dtype=np.float64)
PK_GATE = 5.0e-3
BATCH_SIZE_CASES = [pytest.param(1, id="N1"), pytest.param(128, id="N128")]


def _perturbed_cosmologies(n):
    """Return deterministic nearby flat LCDM parameter dictionaries."""

    if n == 1:
        return [REFERENCE_COSMOLOGY.copy()]
    phase = np.linspace(-1.0, 1.0, n, dtype=np.float64)
    cosmologies = []
    for ph in phase:
        param = REFERENCE_COSMOLOGY.copy()
        omega_b_h2 = 0.02238280 * (1.0 + 0.006 * ph)
        omega_c_h2 = 0.1201075 * (1.0 - 0.008 * ph)
        h = 0.6732117 * (1.0 + 0.004 * np.sin(np.pi * ph))
        param.update(
            {
                "Omegam": (omega_b_h2 + omega_c_h2) / h**2,
                "Omegab": omega_b_h2 / h**2,
                "H0": 100.0 * h,
                "YHe": REFERENCE_COSMOLOGY["YHe"]
                * (1.0 + 0.003 * np.cos(0.5 * np.pi * ph)),
                "Tcmb": REFERENCE_COSMOLOGY["Tcmb"] * (1.0 + 0.0015 * ph),
                "Neff": REFERENCE_COSMOLOGY["Neff"]
                * (1.0 + 0.002 * np.sin(2.0 * np.pi * ph)),
            }
        )
        cosmologies.append(param)
    return cosmologies


def _class_linear_pk(param, k_values):
    """Return the CLASS linear matter spectrum in Mpc cubed at redshift zero."""

    Class = pytest.importorskip("classy").Class
    h = param["H0"] / 100.0
    cosmo = Class()
    cosmo.set(
        {
            "h": h,
            "omega_b": param["Omegab"] * h**2,
            "omega_cdm": (param["Omegam"] - param["Omegab"]) * h**2,
            "T_cmb": param["Tcmb"],
            "YHe": param["YHe"],
            "N_ur": param["Neff"],
            "N_ncdm": 0,
            "Omega_k": 0.0,
            "A_s": param["A_s"],
            "n_s": param["n_s"],
            "k_pivot": 0.05,
            "output": "mPk",
            "P_k_max_1/Mpc": float(k_values[-1]) * 1.1,
            "z_max_pk": 0.0,
        }
    )
    try:
        cosmo.compute()
        return np.array([cosmo.pk_lin(float(k), 0.0) for k in k_values])
    finally:
        cosmo.struct_cleanup()
        cosmo.empty()


@pytest.mark.parametrize("n_cosmologies", BATCH_SIZE_CASES)
def test_matter_power_spectrum_matches_class(n_cosmologies, benchmark):
    """Check and time one GPU launch over nearby flat LCDM cosmologies."""

    from discoeb.perturbations import solve_matter_power_spectrum_batch

    cosmologies = _perturbed_cosmologies(n_cosmologies)
    pk_ours = benchmark.pedantic(
        solve_matter_power_spectrum_batch,
        args=(MATTER_POWER_K, cosmologies),
        rounds=1,
        warmup_rounds=1,
        iterations=1,
    )
    pk_class = np.stack([_class_linear_pk(param, MATTER_POWER_K) for param in cosmologies])
    rel = np.abs(pk_ours / pk_class - 1.0)
    worst = np.unravel_index(np.argmax(rel), rel.shape)
    assert float(rel[worst]) < PK_GATE, (
        f"max relative error {rel[worst]:.6g} for cosmology {worst[0]} "
        f"at k={MATTER_POWER_K[worst[1]]:.6g} Mpc^-1"
    )
