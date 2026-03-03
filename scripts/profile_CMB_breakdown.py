"""Profile breakdown of compute_Cell_spectrum_from_cosmo_params pipeline.

Runs each pipeline step twice: first to JIT-compile, second to time.
"""
import time
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'gpu')

from discoeb.background import evolve_background
from discoeb.perturbations import evolve_perturbations_batched, compute_time_derivatives
from discoeb.cmb import (
    extract_perturbations, compute_background_quantities,
    compute_visibility_functions, compute_neutrino_perturbations,
    compute_metric_perturbations, compute_source_function,
    compute_theta_ell, compute_Cell
)

cosmo_params = {
    'Omegam': 0.3099, 'Omegab': 0.0488911, 'H0': 67.742,
    'n_s': 0.96822, 'A_s': 2.1064e-09, 'mnu': 0.06, 'Tcmb': 2.7255,
    'YHe': 0.248, 'Neff': 2.046, 'Nmnu': 1, 'w_DE_0': -0.99,
    'w_DE_a': 0.0, 'cs2_DE': 1.0, 'Omegak': 0.0, 'k_p': 0.05,
}

lmaxg=11; lmaxgp=11; lmaxr=11; lmaxnu=8; nqmax=3
ellmax=2500; nmodes=128; kmin=1e-4; kmax=1.0
n_k_dense=8192; n_fftlog=16384

def _get_aexp_out():
    z_break1, z_break2, z_break3 = 3000., 1400., 600.
    a_start, a_end = 1e-4, 1.0
    a1 = jnp.geomspace(a_start, 1/(1+z_break1), 32, endpoint=False)
    a2 = jnp.geomspace(1/(1+z_break1), 1/(1+z_break2), 64, endpoint=False)
    a3 = jnp.geomspace(1/(1+z_break2), 1/(1+z_break3), 128, endpoint=False)
    a4 = jnp.geomspace(1/(1+z_break3), a_end, 96)
    return jnp.concatenate([a1, a2, a3, a4])

def run_pipeline():
    """Run the full pipeline, returning intermediate timings."""
    timings = {}

    t0 = time.time()
    param = cosmo_params.copy()
    param = evolve_background(param=param, thermo_module='RECFAST', num_thermo=1024)
    jax.block_until_ready(jax.tree_util.tree_leaves(param))
    timings['background'] = time.time() - t0

    aexp_out = _get_aexp_out()
    t0 = time.time()
    yout, kmodes, param = evolve_perturbations_batched(
        param=param, kmin=kmin, kmax=kmax, num_k=nmodes, aexp_out=aexp_out,
        lmaxg=lmaxg, lmaxgp=lmaxgp, lmaxr=lmaxr, lmaxnu=lmaxnu, nqmax=nqmax,
        rtol=1e-4, atol=1e-4, return_full=True, k_sampling_method='camb',
    )
    jax.block_until_ready(yout)
    timings['perturbations'] = time.time() - t0

    tau = param['tau_out']
    t0 = time.time()
    yprime = compute_time_derivatives(yout, tau, kmodes, param, lmaxg, lmaxgp, lmaxr, lmaxnu, nqmax)
    jax.block_until_ready(yprime)
    timings['time_derivatives'] = time.time() - t0

    iq0 = 10 + lmaxg + lmaxgp + lmaxr
    t0 = time.time()
    perturbations = extract_perturbations(yout, yprime, lmaxg, lmaxgp, lmaxr)
    background_quantities = compute_background_quantities(aexp_out, param)
    tau = param['tau_of_a_spline'].evaluate(aexp_out)
    visibility_functions = compute_visibility_functions(tau, param)
    neutrinos = compute_neutrino_perturbations(yout, yprime, aexp_out, param, nqmax, iq0)
    metric = compute_metric_perturbations(perturbations, neutrinos, background_quantities, param, kmodes, aexp_out)
    source_results = compute_source_function(perturbations, metric, visibility_functions, yout, yprime, kmodes, lmaxg, lmaxgp, tau)
    S = source_results['S']
    jax.block_until_ready(S)
    timings['source_function'] = time.time() - t0

    tau0 = param['tau_of_a_spline'].evaluate(1.0)
    t0 = time.time()
    theta_ell, kmodes_out = compute_theta_ell(
        ellmax=ellmax, kmodes=kmodes, tau=tau, S=S, tau0=tau0,
        n_k_dense=n_k_dense, n_fftlog=n_fftlog,
    )
    jax.block_until_ready(theta_ell)
    timings['theta_ell'] = time.time() - t0

    t0 = time.time()
    Cell = compute_Cell(theta_ell, kmodes_out, cosmo_params['n_s'], cosmo_params['k_p'])
    jax.block_until_ready(Cell)
    timings['Cell'] = time.time() - t0

    return timings

# Warmup (compile everything)
print("Warming up (JIT compilation)...")
_ = run_pipeline()

# Timed run
print("\n--- Timing breakdown (post-JIT) ---\n")
timings = run_pipeline()

total = sum(timings.values())
for name, t in timings.items():
    pct = 100 * t / total
    print(f"  {name:25s} {t:.3f}s  ({pct:5.1f}%)")
print(f"  {'TOTAL':25s} {total:.3f}s")
