import time

import jax
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'gpu')

from discoeb.cmb import compute_Cell_spectrum_from_cosmo_params


# Check JAX backend
print(f"JAX backend: {jax.default_backend()}")
print(f"Devices: {jax.devices()}")

# Define cosmological parameters
cosmo_params = {
    'Omegam': 0.3099,            # Total matter density
    'Omegab': 0.0488911,         # Baryon density
    'H0': 67.742,            # Hubble constant (km/s/Mpc)
    'n_s': 0.96822,           # Spectral index
    'A_s': 2.1064e-09,        # Primordial amplitude
    'mnu': 0.06,              # Neutrino mass (eV)
    'Tcmb': 2.7255,            # CMB temperature (K)
    'YHe': 0.248,             # Helium mass fraction
    'Neff': 2.046,             # Effective relativistic species
    'Nmnu': 1     ,            # Massive neutrino species
    'w_DE_0': -0.99,             # Dark energy EOS today
    'w_DE_a': 0.0,               # Dark energy EOS derivative
    'cs2_DE': 1.0,               # Dark energy sound speed squared
    'Omegak': 0.0,               # Curvature
    'k_p': 0.05,              # Pivot scale (1/Mpc)
}

# # Clear JAX compilation cache
jax.clear_caches()

# Warm up the function, triggering JIT compilation
print("WARMING UP... (this will trigger JIT compilation)")
compute_Cell_spectrum_from_cosmo_params(cosmo_params, ellmax=2500)
                                            
print("STARTING COMPUTATION...")
start = time.time()
with jax.profiler.trace("./profile_CMB_trace"):
    result = compute_Cell_spectrum_from_cosmo_params(cosmo_params, ellmax=2500)
jax.block_until_ready(result)
end = time.time()

print(f"\nSUCCESS!")
print(f"Total time: {end-start:.2f} seconds (including compilation)")
