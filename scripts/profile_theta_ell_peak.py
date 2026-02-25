"""Isolate peak GPU memory of compute_theta_ell vs the rest of the pipeline."""
import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "gpu")
import jax.numpy as jnp

# Import the sub-functions directly
from discoeb.cmb import (
    compute_theta_ell, compute_Cell,
    compute_time_derivatives, extract_perturbations,
    compute_background_quantities, compute_visibility_functions,
    compute_neutrino_perturbations, compute_metric_perturbations,
    compute_source_function,
)
from discoeb.background import evolve_background
from discoeb.perturbations import evolve_perturbations_batched

cosmo_params = {
    'Omegam': 0.3099, 'Omegab': 0.0488911, 'H0': 67.742,
    'n_s': 0.96822, 'A_s': 2.1064e-09, 'mnu': 0.06,
    'Tcmb': 2.7255, 'YHe': 0.248, 'Neff': 2.046, 'Nmnu': 1,
    'w_DE_0': -0.99, 'w_DE_a': 0.0, 'cs2_DE': 1.0,
    'Omegak': 0.0, 'k_p': 0.05,
}

device = jax.devices()[0]

def get_peak():
    m = device.memory_stats()
    return m['peak_bytes_in_use'] / 1e9 if m else 0

def get_current():
    m = device.memory_stats()
    return m['bytes_in_use'] / 1e9 if m else 0

# ============================================================
# WARMUP: run full pipeline once to JIT-compile everything
# ============================================================
from discoeb.cmb import compute_Cell_spectrum_from_cosmo_params
jax.clear_caches()
result = compute_Cell_spectrum_from_cosmo_params(cosmo_params, ellmax=2500, nmodes=128)
jax.block_until_ready(result)
del result
print(f"After warmup: peak={get_peak():.3f} GB, current={get_current():.3f} GB")

# ============================================================
# SECOND RUN: step by step to isolate memory per stage
# ============================================================
# We can't easily reset peak_bytes_in_use, so instead track bytes_in_use
# at each stage to understand live memory, and note when peak changes.

lmaxg, lmaxgp, lmaxr, lmaxnu, nqmax = 11, 11, 11, 8, 3
ellmax = 2500
n_k_dense = 8192
n_fftlog = 16384

# Step 1: Background
param = cosmo_params.copy()
param = evolve_background(param=param, thermo_module='RECFAST', num_thermo=1024)
jax.block_until_ready(jax.tree.leaves(param))
print(f"After background: peak={get_peak():.3f} GB, current={get_current():.3f} GB")

# Step 2: Perturbations
aexp_out = jnp.concatenate([
    jnp.geomspace(1e-4, 1/(1+3000), 32, endpoint=False),
    jnp.geomspace(1/(1+3000), 1/(1+1400), 64, endpoint=False),
    jnp.geomspace(1/(1+1400), 1/(1+600), 128, endpoint=False),
    jnp.geomspace(1/(1+600), 1.0, 96),
])
yout, kmodes, param = evolve_perturbations_batched(
    param=param, kmin=1e-4, kmax=1.0, num_k=128, aexp_out=aexp_out,
    lmaxg=lmaxg, lmaxgp=lmaxgp, lmaxr=lmaxr, lmaxnu=lmaxnu, nqmax=nqmax,
    rtol=1e-4, atol=1e-4, return_full=True, k_sampling_method='camb',
)
jax.block_until_ready(jax.tree.leaves(yout))
print(f"After perturbations: peak={get_peak():.3f} GB, current={get_current():.3f} GB")

# Steps 3-10: Source function
tau = param['tau_out']
yprime = compute_time_derivatives(yout, tau, kmodes, param, lmaxg, lmaxgp, lmaxr, lmaxnu, nqmax)
jax.block_until_ready(yprime)

iq0 = 10 + lmaxg + lmaxgp + lmaxr
perturbations = extract_perturbations(yout, yprime, lmaxg, lmaxgp, lmaxr)
background_quantities = compute_background_quantities(aexp_out, param)
tau = param['tau_of_a_spline'].evaluate(aexp_out)
visibility_functions = compute_visibility_functions(tau, param)
neutrinos = compute_neutrino_perturbations(yout, yprime, aexp_out, param, nqmax, iq0)
metric = compute_metric_perturbations(perturbations, neutrinos, background_quantities, param, kmodes, aexp_out)
source_results = compute_source_function(perturbations, metric, visibility_functions, yout, yprime, kmodes, lmaxg, lmaxgp, tau)
S = source_results['S']
jax.block_until_ready(S)
print(f"After source function: peak={get_peak():.3f} GB, current={get_current():.3f} GB")
print(f"  S shape: {S.shape}, S size: {S.nbytes / 1e9:.3f} GB")

# Step 11: compute_theta_ell (this is the one we care about)
tau0 = param['tau_of_a_spline'].evaluate(1.0)

# Delete everything we don't need to free GPU memory before theta_ell
del yout, yprime, perturbations, background_quantities, visibility_functions
del neutrinos, metric, source_results
import gc; gc.collect()
print(f"Before theta_ell: peak={get_peak():.3f} GB, current={get_current():.3f} GB")

theta_ell, kmodes_out = compute_theta_ell(
    ellmax=ellmax, kmodes=kmodes, tau=tau, S=S, tau0=tau0,
    n_k_dense=n_k_dense, n_fftlog=n_fftlog,
)
jax.block_until_ready(theta_ell)
print(f"After theta_ell: peak={get_peak():.3f} GB, current={get_current():.3f} GB")
print(f"  theta_ell shape: {theta_ell.shape}, size: {theta_ell.nbytes / 1e9:.3f} GB")
