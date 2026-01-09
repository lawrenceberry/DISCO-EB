import time

import jax
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'gpu')
from jax.profiler import ProfileOptions

from discoeb.cmb import compute_Cell_spectrum_from_cosmo_params
from tests.test_cmb import STANDARD_COSMOLOGIES

# Check JAX backend
print(f"JAX backend: {jax.default_backend()}")
print(f"Devices: {jax.devices()}")

# Define cosmological parameters
cosmo_params = STANDARD_COSMOLOGIES["DISCO-Notebook"]

# # Clear JAX compilation cache
jax.clear_caches()

print("STARTING COMPUTATION...")

start = time.time()

options = ProfileOptions()
options.advanced_configuration = {
    'gpu_max_callback_api_events': 8*1024*1024,
    'gpu_max_activity_api_events': 8*1024*1024,
    'gpu_pm_sample_buffer_size_per_gpu_mb': 128,
    'gpu_pm_sample_interval_us': 20000,
}
with jax.profiler.trace("./profile_CMB_trace", profiler_options=options):
    Cell, param = compute_Cell_spectrum_from_cosmo_params(cosmo_params, ellmax=1)
    Cell.block_until_ready()

end = time.time()

print(f"\nSUCCESS!")
print(f"Total time: {end-start:.2f} seconds (including compilation)")
print(f"Cell shape: {Cell.shape}")
