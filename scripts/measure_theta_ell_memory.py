"""Measure peak GPU memory of compute_theta_ell via XLA's device_memory_profile."""
import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "gpu")

from discoeb.cmb import compute_Cell_spectrum_from_cosmo_params

cosmo_params = {
    'Omegam': 0.3099, 'Omegab': 0.0488911, 'H0': 67.742,
    'n_s': 0.96822, 'A_s': 2.1064e-09, 'mnu': 0.06,
    'Tcmb': 2.7255, 'YHe': 0.248, 'Neff': 2.046, 'Nmnu': 1,
    'w_DE_0': -0.99, 'w_DE_a': 0.0, 'cs2_DE': 1.0,
    'Omegak': 0.0, 'k_p': 0.05,
}

# Warm up (JIT compile)
jax.clear_caches()
result = compute_Cell_spectrum_from_cosmo_params(cosmo_params, ellmax=2500, nmodes=128)
jax.block_until_ready(result)

# Measure peak memory on the real run
device = jax.devices()[0]
mem_before = device.memory_stats()
if mem_before is not None:
    peak_before = mem_before.get('peak_bytes_in_use', 0)
else:
    peak_before = 0

# Reset peak
_ = compute_Cell_spectrum_from_cosmo_params(cosmo_params, ellmax=2500, nmodes=128)
jax.block_until_ready(_)

mem_after = device.memory_stats()
if mem_after is not None:
    peak_after = mem_after.get('peak_bytes_in_use', 0)
    bytes_in_use = mem_after.get('bytes_in_use', 0)
    print(f"Peak bytes in use: {peak_after / 1e9:.3f} GB")
    print(f"Current bytes in use: {bytes_in_use / 1e9:.3f} GB")
else:
    print("memory_stats() not available, using nvidia-smi fallback")
    import subprocess, os
    result_smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
         f"--id={os.environ.get('CUDA_VISIBLE_DEVICES', '0')}"],
        capture_output=True, text=True)
    print(f"GPU memory used: {result_smi.stdout.strip()} MiB")
