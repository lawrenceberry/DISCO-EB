"""Profile breakdown of compute_theta_ell components."""
import time
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'gpu')

from discoeb.cmb import (
    compute_Cell_spectrum_from_cosmo_params,
    _fftlog_bessel_integral_coefficients,
)
from discoeb.spline_interpolation import spline_interpolation
from discoeb.background import evolve_background
from discoeb.perturbations import evolve_perturbations_batched, compute_time_derivatives
from discoeb.cmb import (
    extract_perturbations, compute_background_quantities,
    compute_visibility_functions, compute_neutrino_perturbations,
    compute_metric_perturbations, compute_source_function,
)

cosmo_params = {
    'Omegam': 0.3099, 'Omegab': 0.0488911, 'H0': 67.742,
    'n_s': 0.96822, 'A_s': 2.1064e-09, 'mnu': 0.06, 'Tcmb': 2.7255,
    'YHe': 0.248, 'Neff': 2.046, 'Nmnu': 1, 'w_DE_0': -0.99,
    'w_DE_a': 0.0, 'cs2_DE': 1.0, 'Omegak': 0.0, 'k_p': 0.05,
}

lmaxg=11; lmaxgp=11; lmaxr=11; lmaxnu=8; nqmax=3
ellmax=2500; nmodes=128; n_k_dense=8192; n_fftlog=16384

# Warmup full pipeline
print("Warming up...")
compute_Cell_spectrum_from_cosmo_params(cosmo_params, ellmax=ellmax)

# Get S, kmodes, tau, tau0 for theta_ell
param = cosmo_params.copy()
param = evolve_background(param=param, thermo_module='RECFAST', num_thermo=1024)
def _get_aexp_out():
    a1 = jnp.geomspace(1e-4, 1/(1+3000.), 32, endpoint=False)
    a2 = jnp.geomspace(1/(1+3000.), 1/(1+1400.), 64, endpoint=False)
    a3 = jnp.geomspace(1/(1+1400.), 1/(1+600.), 128, endpoint=False)
    a4 = jnp.geomspace(1/(1+600.), 1.0, 96)
    return jnp.concatenate([a1, a2, a3, a4])
aexp_out = _get_aexp_out()
yout, kmodes, param = evolve_perturbations_batched(
    param=param, kmin=1e-4, kmax=1.0, num_k=nmodes, aexp_out=aexp_out,
    lmaxg=lmaxg, lmaxgp=lmaxgp, lmaxr=lmaxr, lmaxnu=lmaxnu, nqmax=nqmax,
    rtol=1e-4, atol=1e-4, return_full=True, k_sampling_method='camb',
)
tau_arr = param['tau_out']
yprime = compute_time_derivatives(yout, tau_arr, kmodes, param, lmaxg, lmaxgp, lmaxr, lmaxnu, nqmax)
iq0 = 10 + lmaxg + lmaxgp + lmaxr
perturbations = extract_perturbations(yout, yprime, lmaxg, lmaxgp, lmaxr)
background_quantities = compute_background_quantities(aexp_out, param)
tau = param['tau_of_a_spline'].evaluate(aexp_out)
visibility_functions = compute_visibility_functions(tau, param)
neutrinos = compute_neutrino_perturbations(yout, yprime, aexp_out, param, nqmax, iq0)
metric = compute_metric_perturbations(perturbations, neutrinos, background_quantities, param, kmodes, aexp_out)
source_results = compute_source_function(perturbations, metric, visibility_functions, yout, yprime, kmodes, lmaxg, lmaxgp, tau)
S = source_results['S']
tau0 = param['tau_of_a_spline'].evaluate(1.0)
jax.block_until_ready(S)

# Now profile theta_ell components individually
print("\n--- theta_ell component breakdown ---\n")

# 1. k-interpolation
log_k_orig = jnp.log(kmodes)
log_k_dense = jnp.linspace(log_k_orig[0], log_k_orig[-1], n_k_dense)
kmodes_dense = jnp.exp(log_k_dense)

@jax.jit
def do_k_interp(S, log_k_orig, log_k_dense):
    def _interp_one_slice(y_slice):
        spl = spline_interpolation(log_k_orig, y_slice)
        return spl.evaluate(log_k_dense)
    return jax.vmap(_interp_one_slice)(S.T).T

# warmup
S_dense = do_k_interp(S, log_k_orig, log_k_dense)
jax.block_until_ready(S_dense)

t0 = time.time()
S_dense = do_k_interp(S, log_k_orig, log_k_dense)
jax.block_until_ready(S_dense)
print(f"  k-interpolation (128→8192):  {time.time()-t0:.3f}s")

# 2. chi-interpolation
chi = tau0 - tau
chi = chi[::-1]
S_chi = S_dense[:, ::-1]
log_chi_min = jnp.log(jnp.maximum(chi[0], 1.0))
log_chi_max = jnp.log(chi[-1])
pad = 2.0
dlog_chi = (log_chi_max + pad - (log_chi_min - pad)) / n_fftlog
chi_fftlog = jnp.exp((log_chi_min - pad) + jnp.arange(n_fftlog) * dlog_chi)
log_chi_orig = jnp.log(jnp.maximum(chi, 1e-10))
log_chi_fft = jnp.log(chi_fftlog)

@jax.jit
def do_chi_interp(S_chi, log_chi_orig, log_chi_fft):
    def interp_S_to_fftlog(S_k):
        spline = spline_interpolation(log_chi_orig, S_k)
        return spline.evaluate(log_chi_fft)
    return jax.vmap(interp_S_to_fftlog)(S_chi)

S_fftlog = do_chi_interp(S_chi, log_chi_orig, log_chi_fft)
jax.block_until_ready(S_fftlog)

t0 = time.time()
S_fftlog = do_chi_interp(S_chi, log_chi_orig, log_chi_fft)
jax.block_until_ready(S_fftlog)
print(f"  chi-interpolation (320→16384): {time.time()-t0:.3f}s")

# 3. FFT + coefficients
bias = -1.1
@jax.jit
def do_fft(S_fftlog, chi_fftlog, log_chi_min_pad):
    S_biased = S_fftlog * chi_fftlog[None, :] ** (-bias)
    c_n_fft = jnp.fft.rfft(S_biased, axis=-1)
    n_freqs = n_fftlog // 2 + 1
    eta_n = 2.0 * jnp.pi * jnp.arange(n_freqs) / (n_fftlog * dlog_chi)
    phase_shift = jnp.exp(-1j * eta_n * log_chi_min_pad)
    return c_n_fft * phase_shift / n_fftlog

c_n = do_fft(S_fftlog, chi_fftlog, log_chi_min - pad)
jax.block_until_ready(c_n)

t0 = time.time()
c_n = do_fft(S_fftlog, chi_fftlog, log_chi_min - pad)
jax.block_until_ready(c_n)
print(f"  FFT + phase shift:           {time.time()-t0:.3f}s")

# 4. Bessel coefficients
n_freqs = n_fftlog // 2 + 1
eta_n = 2.0 * jnp.pi * jnp.arange(n_freqs) / (n_fftlog * dlog_chi)
p_n = bias + 1j * eta_n
ells = jnp.arange(ellmax + 1, dtype=jnp.float64)

@jax.jit
def do_bessel(ells, p_n):
    return _fftlog_bessel_integral_coefficients(ells, p_n)

M_ell_p = do_bessel(ells, p_n)
jax.block_until_ready(M_ell_p)

t0 = time.time()
M_ell_p = do_bessel(ells, p_n)
jax.block_until_ready(M_ell_p)
print(f"  Bessel coefficients:         {time.time()-t0:.3f}s")

# 5. k-power + GEMMs
@jax.jit
def do_gemm(c_n, p_n, kmodes_dense, M_ell_p):
    log_k = jnp.log(kmodes_dense)[:, None]
    k_power = jnp.exp(-(p_n[None, :] + 1.0) * log_k)
    ck = c_n * k_power
    theta_ell_0 = jnp.einsum('k,l->kl', ck[:, 0].real, M_ell_p[:, 0].real)
    ck_pos = ck[:, 1:-1]
    M_pos = M_ell_p[:, 1:-1]
    ck_nyq = ck[:, -1]
    M_nyq = M_ell_p[:, -1]
    theta_ell_pos = 2.0 * (
        jnp.einsum('kn,ln->kl', ck_pos.real, M_pos.real)
        - jnp.einsum('kn,ln->kl', ck_pos.imag, M_pos.imag)
    )
    theta_ell_nyq = jnp.real(jnp.einsum('k,l->kl', ck_nyq, M_nyq))
    return theta_ell_0 + theta_ell_pos + theta_ell_nyq

theta_ell = do_gemm(c_n, p_n, kmodes_dense, M_ell_p)
jax.block_until_ready(theta_ell)

t0 = time.time()
theta_ell = do_gemm(c_n, p_n, kmodes_dense, M_ell_p)
jax.block_until_ready(theta_ell)
print(f"  k-power + GEMMs:             {time.time()-t0:.3f}s")

print()
