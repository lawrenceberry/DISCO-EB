#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import time

# 1. Disable pre-allocation so JAX doesn't lock the whole VRAM
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# 2. Use the platform allocator for better crash recovery 
# (This makes it play nicer with Xorg/Chrome)
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

from jax import config
config.update("jax_debug_nans", True)
config.update("jax_enable_x64", True)

# In[2]:


import jax
import jax.numpy as jnp
print(jax.devices())


# In[3]:

from discoeb.cmb import compute_Cell_spectrum_from_cosmo_params

# In[4]:

Npar = 5

## Set the Cosmological Parameters
# insert parameters into a dictionary
param = {}
# OmegaDE is inferred since flatness is assumed currently
param['Omegam']  = jnp.linspace(0.3, 0.3099, num=Npar)            # Total matter density parameter
param['Omegab']  = jnp.linspace(0.04, 0.0488911, num=Npar)         # Baryon density parameter
param['w_DE_0']  = -0.99             # Dark energy equation of state parameter today
param['w_DE_a']  = 0.0               # Dark energy equation of state parameter time derivative
param['cs2_DE']  = 1.0               # Dark energy sound speed squared
param['Omegak']  = 0.0
param['A_s']     = 2.1064e-09        # Scalar amplitude of the primordial power spectrum
param['n_s']     = 0.96822           # Scalar spectral index
param['H0']      = 67.742            # Hubble constant today in units of 100 km/s/Mpc
param['Tcmb']    = 2.7255            # CMB temperature today in K
param['YHe']     = 0.248             # Helium mass fraction
param['Neff']    = 2.046             # Effective number of ultrarelativistic neutrinos
                                      # -1 if massive neutrino present
param['Nmnu']    = 1                 # Number of massive neutrinos (must be 1 currently)
param['mnu']     = 0.06              # Sum of neutrino masses in eV 
param['k_p']     = 0.05              # Pivot scale in 1/Mpc

# modes to sample
#nmodes = 512                         # number of modes to sample
nmodes=20
kmin   = 1e-4                        # minimum k in 1/Mpc
kmax   = 1.0                          # maximum k in 1/Mpc

# In[5]:


param['use_tca'] = False
param['use_rsa'] = False
param['use_ur_fluid'] = False

start = time.time()
result = compute_Cell_spectrum_from_cosmo_params(param, ellmax=2500, 
    n_k_dense=128,
    n_fftlog=64,
    nmodes=20)
jax.block_until_ready(result)
end = time.time()

print(result[0].shape)

print(f"\nSUCCESS!")
print(f"Total time: {end-start:.2f} seconds (including compilation)")
# In[6]:








