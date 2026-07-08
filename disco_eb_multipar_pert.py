#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

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


from discoeb.perturbations import evolve_perturbations

from discoeb.background import evolve_background

from discoeb.perturbations import model_synchronous


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


param = evolve_background(param=param, thermo_module='RECFAST', num_thermo=1024)


# In[6]:


aexp_out = jnp.geomspace(3e-4, 5e-3, 256, endpoint=False)
aexp_out = jnp.append(aexp_out, jnp.geomspace(5e-3, 1.0, 64))

from discoeb.perturbations import evolve_perturbations
print(f"Computing perturbations for {nmodes} k-modes at {len(aexp_out)} output times...")
#with jax.disable_jit():
yout, kmodes, param = evolve_perturbations(
    param=param, 
    kmin=kmin, 
    kmax=kmax, 
    num_k=nmodes, 
    aexp_out=aexp_out,
    rtol=1e-4, 
    atol=1e-4,
    return_full=True,
    k_sampling_method='log', # use log spacing
    lmaxnu=12,
    max_steps = 8196
    #max_steps = 1024
)


quit()
# # Full mode debugging  

# In[25]:


def get_perturbations( param, kmin, kmax, nmodes ):

  aexp_out = jnp.geomspace(3e-4, 5e-3, 256, endpoint=False)
  aexp_out = jnp.append( aexp_out, jnp.geomspace(5e-3, 1.0, 64))
  ## Compute Background+thermal evolution
  # Use more points for thermal history to avoid issues with adaptive sampling
  print("Computing background evolution with RECFAST...")
  param = evolve_background(param=param, thermo_module='RECFAST', num_thermo=1024)
  print(f"Background evolution complete. Thermal history has {len(param['aexp'])} points")

  # compute perturbation evolution
  aexp_out = jnp.geomspace(3e-4, 5e-3, 256, endpoint=False)
  aexp_out = jnp.append( aexp_out, jnp.geomspace(5e-3, 1.0, 64))

  # Info: you can also use the non-batched solver 'evolve_perturbations' here as a drop-in replacement, but it's slower
  # batched solver will use the smallest timestep of the batch
  y, kmodes, param = evolve_perturbations( param=param, kmin=kmin, kmax=kmax, num_k=nmodes, aexp_out=aexp_out, 
                                  # lmaxg = 31, lmaxgp = 31, lmaxr = 31, lmaxnu = 31, nqmax = 5, # from compare_class
                                  rtol=1e-4, atol=1e-4 , return_full=True,
                                  k_sampling_method='log', # use log spacing
                                  max_steps = 4096
                                  )
  y = y
  tau = param['tau_out']
  lmaxg = param['lmaxg']
  lmaxgp = param['lmaxgp']
  lmaxr = param['lmaxr']
  lmaxnu = param['lmaxnu']
  nqmax = param['nqmax']
  idxtau = jnp.arange(len(tau))
  idxk = jnp.arange(len(kmodes))
  yprime = jax.vmap( lambda ik : jax.vmap( lambda itau : model_synchronous( tau=tau[itau], y=y[ik,itau,:], param=param, kmode=kmodes[ik], lmaxg=lmaxg, lmaxgp=lmaxgp, lmaxr=lmaxr, lmaxnu=lmaxnu, nqmax=nqmax ) )(idxtau) )(idxk)

  return y, yprime, kmodes, param


# In[26]:

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
nmodes = 20#512                         # number of modes to sample
kmin   = 1e-4                        # minimum k in 1/Mpc
kmax   = 1.0                          # maximum k in 1/Mpc


# In[27]:

#get_perturbations(param, kmin, kmax, nmodes)
m_jit = jax.jit(get_perturbations, static_argnames=["kmin","kmax","nmodes"])
m_jit(param, kmin, kmax, nmodes)





