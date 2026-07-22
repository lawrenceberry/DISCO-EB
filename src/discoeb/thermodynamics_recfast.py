
####################################################################################################################
# This file is not licensed under the GNU GPL license!
# Large portions of this code, which is part of the DISCO-EB module have been adapted from the recfast code base.
# The original recfast code can be found at: https://www.astro.ubc.ca/people/scott/recfast.html
# The recfast code base is distributed under the following license:
####################################################################################################################
 # Integrator for Cosmic Recombination of Hydrogen and Helium,
 # developed by Douglas Scott (dscott@astro.ubc.ca)
 # based on calculations in the papers Seager, Sasselov & Scott
 # (ApJ, 523, L1, 1999; ApJS, 128, 407, 2000)
 # and "fudge" updates in Wong, Moss & Scott (2008).
 #
 # Permission to use, copy, modify and distribute without fee or royalty at
 # any tier, this software and its documentation, for any purpose and without
 # fee or royalty is hereby granted, provided that you agree to comply with
 # the following copyright notice and statements, including the disclaimer,
 # and that the same appear on ALL copies of the software and documentation,
 # including modifications that you make for internal use or for distribution:
 #
 # Copyright 1999-2010 by University of British Columbia.  All rights reserved.
####################################################################################################################
# This file contains a JAX-compatible version of the recfast code base, which has been adapted for use in the 
# DISCO-EB module by Oliver Hahn. The DISCO-EB module is distributed under the GNU GPL license.
####################################################################################################################
 
import diffrax as drx
import jax
#jax.config.update('jax_disable_jit', True)
import jax.numpy as jnp
from jax_cosmo.scipy.integrate import romb
from functools import partial
from typing import Tuple
from .spline_interpolation import spline_interpolation

from .ode_integrators_stiff import GRKT4
# from diffrax import Tsit5

# Pre-compute constants
const_G = 6.67430e-11               # Gravitational constant [m^3/kg/s^2], PDG 2023
const_mH      = 1.67353284e-27      # H atom mass [kg], PDG 2023
const_me      = 9.1093837015e-31    # Electron mass [kg], PDG 2023
const_mHe_mH  = 3.97146570884       # Helium / Hydrogen mass ratio
const_c       = 2.99792458e+08      # Speed of light [m/s]     
const_h       = 6.62607015e-34      # Planck's constant [Js], PDG 2023
const_kB      = 1.380649e-23        # Boltzman's constant [J/K], PDG 2023
const_sigma   = 6.6524587321e-29    # Thomson scattering cross section [m^2], PDG 2023
const_arad = 4 * 5.670374419e-8 / const_c # Radiation constant [Ws/m^3/K^4], PDG 2023

# bigH = 100.0e3/(1.0e6*3.0856775807e16)  # Ho in s-1
bigH = 3.2407792902755102e-18       # Ho in s-1 
const_dens_fac = 11.223810928601939 # 3 * bigH**2 / (8 * jnp.pi * const_G * const_mH)

const_c2ok    = 1.62581581e4 # K / eV
const_c_Mpc_s = 9.71561189e-15 # Mpc/s

## COMMENTS FOR ALL CONSTANTS HAVE BEEN PRESERVED AS MUCH AS POSSIBLE AS THEY ARE IN THE ORIGINAL CODE

const_EionHe12s = 6.314878282674e5  # HeII 1s ionization energy in [K] 

# 2 photon rates and atomic levels
Lambda = 8.2245809                  # H 2s-1s two photon rate
Lambda_He = 51.3                    # HeI 2s-1s two photon rate
L_H_ion = 1.096787737e7             # level for H ionization in m^-1
L_H_alpha = 8.225916453e6           # level for H Ly alpha in m^-1
L_He1_ion = 1.98310772e7            # level for HeI ionization in m^-1
L_He2_ion = 4.389088863e7           # level for HeII ionization in m^-1
L_He_2s = 1.66277434e7              # level for HeI 2s in m^-1
L_He_2p = 1.71134891e7              # level for He 2p (21P1-11S0) in m^-1    

# Atomic data for HeI
A2P_s = 1.798287e9                  # Einstein A coefficient for He 21P1-11S0
A2P_t = 177.58                      # Einstein A coefficient for He 23P1-11S0  
L_He_2Pt = 1.690871466e7            # level for 23P012-11S0 in m^-1
L_He_2St = 1.5985597526e7           # level for 23S1-11S0 in m^-1
L_He2St_ion = 3.8454693845e6        # level for 23S1-continuum in m^-1
sigma_He_2Ps = 1.436289e-22         # H ionization x-section at HeI 21P1-11S0 freq. in m^2
sigma_He_2Pt = 1.484872e-22         # H ionization x-section at HeI 23P1-11S0 freq. in m^2

# Gaussian fits
AGauss1 = -0.14                     # amplitude of the 1st Gaussian for the H fudging
AGauss2 = 0.05                      # amplitude of the 2nd Gaussian for the H fudging
zGauss1 = 7.28                      # ln(1+z) central value of the 1st Gaussian
zGauss2 = 6.75                      # ln(1+z) central value of the 2nd Gaussian
wGauss1 = 0.18                      # width of the 1st Gaussian
wGauss2 = 0.33                      # width of the 2nd Gaussian

H_frac = 1.e-3 

Lalpha = 1.0/L_H_alpha                        # Ly alpha wavelength in SI units
Lalpha_He = 1.0/L_He_2p                       # Helium I 2p-1s wavelength in SI units
DeltaB = const_h*const_c*(L_H_ion-L_H_alpha)  # energy of first excited state from continuum = 3.4eV
CDB = DeltaB/const_kB                         # CDB=DeltaB/k_B			Constants derived from B1,B2,R
DeltaB_He = const_h*const_c*(L_He1_ion-L_He_2s)  # energy of first excited state from cont. for He = 3.4eV
CDB_He = DeltaB_He/const_kB                   # CDB_He=DeltaB_He/k_B n=2-infinity for He in Kelvin
CB1 = const_h*const_c*L_H_ion/const_kB        # CDB*4.			Lalpha and sigma_Th, calculated
CB1_He1 = const_h*const_c*L_He1_ion/const_kB  # CB1 for HeI ionization potential
CB1_He2 = const_h*const_c*L_He2_ion/const_kB  # CB1 for HeII ionization potential

CR = 2.0*jnp.pi*(const_me/const_h)*(const_kB/const_h) 
CK = Lalpha**3/(8.0*jnp.pi)         
CK_He = Lalpha_He**3/(8.0*jnp.pi) 
CL = const_c*const_h/(const_kB*Lalpha) 
CL_He = const_c*const_h/(const_kB/L_He_2s)
CT = (8.0/3.0)*(const_sigma/(const_me*const_c))*const_arad  
Bfact = const_h*const_c*(L_He_2p-L_He_2s)/const_kB  
CL_PSt = const_h*const_c*(L_He_2Pt - L_He_2St)/const_kB
const_h_c_L_He_2St_kB = const_h*const_c*L_He_2St/(const_kB)

# Pequignot, Petitjean & Boisson fitting parameter for Hydrogen
a_PPB = 4.309
b_PPB = -0.6166
c_PPB = 0.6703
d_PPB = 0.5300

# Verner and Ferland type fitting parameter for Helium
a_VF = 10.**(-16.744)
b_VF = 0.711
T_0 = 10**(0.477121)
T_1 = 10**(5.114)

# HeI triplet recombination rate fitting parameters
a_trip = 10.**(-16.306)
b_trip = 0.761
  

#def ionization(a, y, params, compute_xH_fn, compute_xHe_fn):
def ionization(loga, y, params, compute_xH_fn, compute_xHe_fn):
  #jax.lax.cond(a > 1e-8,lambda: jax.debug.print("x={y}, a={a}", y=y, a=a),lambda: None)
  #a = jnp.where(a==0, -1, a)
  a = jnp.exp(loga)
  #jax.debug.print("a={a},y={y}",a=a,y=y)
  param = params[0]
  #saha_inputs = params[1]
  #jax.debug.print("a={a},saha_inputs={b}",a=a,b=saha_inputs)

  z = 1/a - 1
  # Pre-compute derived constants
  H = param['H0']/100.0
  HO = H*bigH
  fu = 1.105
  mu_H = 1.0/(1.0-param['YHe'])
  # mu_T = const_mHe_mH/(const_mHe_mH-(const_mHe_mH-1.0)*param['YHe'])
  fHe = param['YHe']/(const_mHe_mH*(1.0-param['YHe']))
  Nnow = const_dens_fac * H * H * param['Omegab'] / mu_H

  x_H = y[:, 0]
  x_H = jnp.where(x_H<0, 0, x_H)
  x_He = y[:, 1]
  x_He = jnp.where(x_He<0, 0, x_He)
  x = x_H + fHe * x_He
  Tmat = jnp.abs(y[:, 2])

  # Calculate common terms once
  n = Nnow * (1 + z)**3
  n_He = fHe * n
  Trad = param['Tcmb'] * (1 + z)
  z_term = (1 + z)

  # Hubble parameter calculation
  # Hprime = a'/a, dtau = dt/a -> da/dtau/a = da/dt = Ha
  from .background import get_aprimeoa
  Hz = (1e-5*get_aprimeoa(param=param, aexp=a)[0]) / a * const_c * bigH
  
  # Temperature and rate calculations
  Tmat_1e4 = Tmat / 1e4
  CR_Tmat_15 = (CR * Tmat)**1.5
  
  # Get the radiative rates
  Rdown = 1e-19 * a_PPB * Tmat_1e4**b_PPB / (1 + c_PPB * Tmat_1e4**d_PPB)
  Rup = Rdown * CR_Tmat_15 * jnp.exp(-CDB / Tmat)

  # Calculate He rates using a fit
  sq_0 = jnp.sqrt(Tmat / T_0)
  sq_1 = jnp.sqrt(Tmat / T_1)
  
  sq_0_term = 1 + sq_0
  sq_1_term = 1 + sq_1
  
  Rdown_He_common = a_VF / (sq_0 * sq_0_term**(1 - b_VF) * sq_1_term**(1 + b_VF))
  Rdown_He = Rdown_He_common
  Rup_He = 4 * Rdown_He * CR_Tmat_15 * jnp.exp(-CDB_He / Tmat)

  # Calculate Boltzmann factor with numerical stability
  He_Boltz = jnp.exp(jnp.minimum(680.0, Bfact / Tmat))

  # HeI calculations
  Rdown_trip = a_trip / (sq_0 * sq_0_term**(1 - b_trip) * sq_1_term**(1 + b_trip))
  Rup_trip = Rdown_trip * jnp.exp(-const_h*const_c*L_He2St_ion/(const_kB*Tmat)) * CR_Tmat_15 * (4/3)
  
  # Peebles coefficient calculation
  K_gaussian = 1 + AGauss1*jnp.exp(-((jnp.log(z_term)-zGauss1)/wGauss1)**2) + AGauss2*jnp.exp(-((jnp.log(z_term)-zGauss2)/wGauss2)**2)
  K = CK / Hz * K_gaussian
  
  # He calculations
  one_minus_x_He = 1 - x_He
  one_minus_x_H = 1 - x_H
  n_He_1_minus_x_He = n_He * one_minus_x_He
  
  tauHe_s = A2P_s * CK_He * 3 * n_He_1_minus_x_He / Hz
  pHe_s = (1 - jnp.exp(-tauHe_s)) / tauHe_s
  
  # Doppler calculation
  Doppler_term = jnp.sqrt(2 * const_kB * Tmat / (const_mH * const_mHe_mH * const_c**2))
  Doppler_2p = const_c * L_He_2p * Doppler_term
  Doppler_2Pt = const_c * L_He_2Pt * Doppler_term
  
  # Gamma calculations
  gamma_2Ps_term = 3 * A2P_s * fHe * one_minus_x_He * const_c**2
  gamma_2Ps_denom = jnp.sqrt(jnp.pi) * sigma_He_2Ps * 8 * jnp.pi * Doppler_2p * one_minus_x_H * (const_c * L_He_2p)**2
  gamma_2Ps = gamma_2Ps_term / gamma_2Ps_denom
  
  # AHcon calculation for H continuum opacity
  pb = 0.36  # value from KIV (2007)
  qb = 0.86  # He fudge factor
  AHcon = A2P_s / (1 + pb * (gamma_2Ps**qb))
  
  # K_He calculation with numerical stability
  K_He_cond1 = jnp.logical_or(x_He < 5e-9, x_He > 0.98)
  K_He_cond2 = x_H < 0.9999999
  
  K_He_default = CK_He / Hz
  K_He_case1 = 1.0 / ((A2P_s * pHe_s + AHcon) * 3.0 * n_He_1_minus_x_He)
  K_He_case2 = 1.0 / (A2P_s * pHe_s * 3.0 * n_He_1_minus_x_He)
  
  K_He = jnp.where(K_He_cond1, K_He_default, 
           jnp.where(K_He_cond2, K_He_case1, K_He_case2))
  
  # Triplet calculations
  tauHe_t = A2P_t * n_He_1_minus_x_He * 3 / (8 * jnp.pi * Hz * L_He_2Pt**3)
  pHe_t = (1 - jnp.exp(-tauHe_t)) / tauHe_t
  
  gamma_2Pt_term = 3 * A2P_t * fHe * one_minus_x_He * const_c**2
  gamma_2Pt_denom = jnp.sqrt(jnp.pi) * sigma_He_2Pt * 8 * jnp.pi * Doppler_2Pt * one_minus_x_H * (const_c * L_He_2Pt)**2
  gamma_2Pt = gamma_2Pt_term / gamma_2Pt_denom
  
  # KIV (2007) parameters
  pb_t = 0.66
  qb_t = 0.9
  AHcon_t = A2P_t / (1 + pb_t * gamma_2Pt**qb_t) / 3
  
  # CfHe_t calculation
  exp_CL_PSt = jnp.exp(-CL_PSt / Tmat)
  CfHe_t_case1 = A2P_t * pHe_t * exp_CL_PSt
  CfHe_t_case2 = (A2P_t * pHe_t + AHcon_t) * exp_CL_PSt
  CfHe_t = jnp.where(x_H > 0.99999, CfHe_t_case1, CfHe_t_case2)
  CfHe_t /= (Rup_trip + CfHe_t)  # "C" factor for triplets
  
  # Time calculations
  timeTh = (1 / (CT * Trad**4)) * (1 + x + fHe) / x
  timeH = 2 / (3 * HO * z_term**1.5)
  
  # f0 calculation
  Hz_z_term = Hz * z_term
  x_x_H_n_Rdown = x * x_H * n * Rdown
  Rup_1_minus_x_H_exp_CL = Rup * one_minus_x_H * jnp.exp(-CL / Tmat)
  rate_diff = x_x_H_n_Rdown - Rup_1_minus_x_H_exp_CL
  
  K_Lambda_n_1_minus_x_H = K * Lambda * n * one_minus_x_H
  denom_term = 1 / fu + K_Lambda_n_1_minus_x_H / fu + K * Rup * n * one_minus_x_H
  
  f0_case1 = compute_xH_fn(1/a-1)#jnp.exp(saha_inputs[1].evaluate(jnp.log(a))[0].T)-1e-9
  f0_case2 = rate_diff / Hz_z_term
  f0_case3 = (rate_diff * (1 + K_Lambda_n_1_minus_x_H)) / (Hz_z_term * denom_term)
  
  f0 = jnp.where(x_H > 0.99, f0_case1,
          jnp.where(x_H > 0.985, f0_case2, f0_case3))
  
  # f1 calculation
  x_x_He_n_Rdown_He = x * x_He * n * Rdown_He
  Rup_He_1_minus_x_He_exp_CL_He = Rup_He * one_minus_x_He * jnp.exp(-CL_He / Tmat)
  rate_diff_He = x_x_He_n_Rdown_He - Rup_He_1_minus_x_He_exp_CL_He
  
  K_He_Lambda_He_n_He_1_minus_x_He_He_Boltz = K_He * Lambda_He * n_He_1_minus_x_He * He_Boltz
  He_denom_term = 1 + K_He * (Lambda_He + Rup_He) * n_He_1_minus_x_He * He_Boltz
  
  f1_main = (rate_diff_He * (1 + K_He_Lambda_He_n_He_1_minus_x_He_He_Boltz)) / (Hz_z_term * He_denom_term)
  
  # Triplet contribution
  trip_rate_diff = x * x_He * n * Rdown_trip - one_minus_x_He * 3 * Rup_trip * jnp.exp(-const_h_c_L_He_2St_kB / Tmat)
  trip_contrib = trip_rate_diff * CfHe_t / Hz_z_term
  trip_cond = jnp.logical_or(x_He < 5e-9, x_He > 0.98)
  
  f1_case1= compute_xHe_fn(1/a-1)#jnp.exp(saha_inputs[0].evaluate(jnp.log(a))[0].T)-1e-9
  f1 = jnp.where(jnp.logical_or(x_He>0.9, a<3e-4),f1_case1,
                    jnp.where(x_He < 1e-8, 0.0, 
                      f1_main + jnp.where(trip_cond, 0.0, trip_contrib))
                )

  #jax.debug.print("at a={a}, saha = {y}, {z}", a=a, y=f0_case1, z=f1_case1)

  # f2 calculation
  epsilon = Hz * (1 + x + fHe) / (CT * Trad**3 * x)
  f2_case1 = Tmat / z_term
  f2_case2 = CT * (Trad**4) * x / (1 + x + fHe) * (Tmat - Trad) / Hz_z_term + 2 * Tmat / z_term
  f2 = jnp.where(timeTh < H_frac * timeH, f2_case1, f2_case2)
  
  #dzda = -1/a**2
  dzdlna = -1/a
  #jax.lax.cond(a > 1e-9,lambda: jax.debug.print("f0={f0}, f1 = {f1}, f2={f2}, dzda = {dzda}",f0=f0, f1=f1,f2=f2 , dzda=dzda),lambda: None)
  
  #print(f0.shape, f1.shape, f2.shape)
  dy = jnp.stack(jnp.broadcast_arrays(f0, f1, f2), axis=1)
  #print("returning ",f0.shape, f1.shape, f2.shape, dy.shape)
  return dy * dzdlna#dzda

def solve_ionization( *, astart : float, aend : float, ystart : jnp.ndarray, rtol : float = 1e-6, atol : float = 1e-8, max_steps : int = 128, param : dict ) -> jnp.ndarray:
  sol =drx.diffeqsolve(
        terms=drx.ODETerm(ionization),
        solver=GRKT4(),
        t0=astart,
        t1=aend,
        dt0=jnp.abs(astart*1e-3),
        y0=ystart[:3],
        stepsize_controller = drx.PIDController(rtol=rtol,atol=atol,dtmin=1e-16), 
        max_steps=max_steps,
        args=(param,),
        # adjoint=drx.RecursiveCheckpointAdjoint(),
        #adjoint=drx.ForwardMode(),
        adjoint=drx.ImplicitAdjoint(),
        throw=False,
    )
  if sol.result != dfx.RESULTS.successful:
    jax.debug.print("Solver failed with code: {r}",r=sol.result)
  y_end = sol.ys[-1, :]
  dyda = ionization( aend, y_end, (param,) )
  return jnp.array([y_end[0], y_end[1], y_end[2], dyda[0], dyda[1], dyda[2]])

def Saha_HeII( a, param ):
    a = a[:, None]
    Tcmb = param['Tcmb'][None, :]
    fHe = param['fHe'][None, :]
    """
    Saha equation for HeII recombination
    """
    def NHnow( a, YHe, H0, Omegab ):
      const_Hfac = 1/(1.0e+06*3.0856775807e+13) # 1 km/s/Mpc in 1/s
      mu_H = 1/(1-YHe) 
      rho_c = 3 * (H0*const_Hfac)**2 / (8 * jnp.pi * const_G) 
      return rho_c * Omegab / (const_mH * mu_H) / a**3
    T = Tcmb / a
    betaE = const_EionHe12s / T
    A = 1 + fHe
    B = 1 + 2*fHe
    NHnow_val = NHnow( a, param['YHe'][None, :], param['H0'][None, :], param['Omegab'][None, :] )
    R = (2*jnp.pi* const_me * const_kB / const_h**2 * T )**1.5 / NHnow_val * jnp.exp( - betaE )

    xe = (R * B)/(jnp.sqrt((R-A)**2 * 0.25 + R * B) + (R - A)*0.5) - A

    return xe

def Saha(T, z, N_now, E_ion, offset, g_factor, f_factor):

    # Thermal de Broglie factor
    phi = (CR * T)**1.5 / N_now / (1+z)**3
    R = g_factor * phi * jnp.exp(-E_ion / T)

    fully_ionized = T > E_ion * 0.1#R>1e8
    # Quadratic coefficients
    alpha = jnp.where(fully_ionized, 1.0, R + offset)
    beta = jnp.where(fully_ionized, f_factor * (1+f_factor), R * f_factor)

    discr = jnp.maximum(alpha**2 + 4.0*beta, 1e-35)
    
    # Quadratic formula (stabilized for large alpha/beta
    return jnp.where(R>1e-20, (2.0 * beta) / (jnp.sqrt(discr) + alpha), 0.0)

def _get_adaptive_sampling(a0: float, a1: float, N: int) -> jax.Array:
  """
  Generate adaptive sampling in scale factor with concentration around recombination.

  Distributes points to concentrate sampling where ionization fraction changes rapidly:
  - 5% very early times (z > 3000)
  - 10% pre-recombination (1400 < z < 3000)
  - 50% recombination era (600 < z < 1400) - where xe changes most
  - 35% post-recombination (z < 600)

  This improves accuracy of spline interpolation without increasing total points.
  """
  # Allocate points adaptively
  n_very_early = max(8, int(N * 0.05))    # At least 8 points, ~5%
  n_pre_recomb = max(8, int(N * 0.10))    # ~10%
  n_recomb = int(N * 0.50)                 # 50% - concentrate here!
  n_post = N - n_very_early - n_pre_recomb - n_recomb  # ~35%

  # Redshift breakpoints
  z_break1 = 10000  # Very early
  z_break2 = 1400  # Start of recombination
  z_break3 = 600   # End of recombination

  a_break1 = 1.0 / (1.0 + z_break1)
  a_break2 = 1.0 / (1.0 + z_break2)
  a_break3 = 1.0 / (1.0 + z_break3)

  # Ensure breakpoints are within bounds
  a_break1 = jnp.maximum(a_break1, a0)

  # Create segments with geometric spacing
  a1_seg = jnp.geomspace(a0, a_break1, n_very_early, endpoint=False)
  a2_seg = jnp.geomspace(a_break1, a_break2, n_pre_recomb, endpoint=False)
  a3_seg = jnp.geomspace(a_break2, a_break3, n_recomb, endpoint=False)
  a4_seg = jnp.geomspace(a_break3, a1, n_post)

  return jnp.concatenate([a1_seg, a2_seg, a3_seg, a4_seg])

def compute_thermal_history( *, a0 : float, a1 : float, N : int, rtol : float = 1e-3, atol : float = 1e-6, param : dict ) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """
  Compute the thermal history of the Universe from a0 to a1 in N steps

  Args:
      a0 (float): initial scale factor
      a1 (float): final scale factor
      N (int): number of steps
      rtol (float, optional): relative tolerance. Defaults to 1e-3.
      atol (float, optional): absolute tolerance. Defaults to 1e-6.
      param (dict): dictionary of cosmological parameters

  Returns:
      Tuple[jnp.ndarray, jnp.ndarray]: [xeHI, xeHeII, Tm, dxHIda, dxHeIda, dTmda], scale factor
  """

  # Use adaptive sampling that concentrates points around recombination
  a = _get_adaptive_sampling(a0, a1, N+1)
  B = max(param[k].shape[0] for k in ['H0','YHe','Omegab']) # Never batch in T_cmb
  y_init = jnp.zeros((6, N, B))

  H = param['H0']/100.0
  HO = H*bigH
  mu_H = 1.0/(1.0-param['YHe'])
  Nnow = 3.0 * HO * HO * param['Omegab'] / (8.0 * jnp.pi * const_G * mu_H * const_mH)
  fHe = param['YHe']/(const_mHe_mH*(1.0-param['YHe']))
  Tcmb = param['Tcmb']
  Tcmb2 = Tcmb**2
  Tcmb3 = Tcmb2*Tcmb

  def loop_body(i, y_arr):
    astart = a[i]
    zstart = 1.0/astart - 1.0
    aend   = a[i+1]
    zend   = 1.0/aend - 1.0
    dzda   = -1.0/aend**2
    tcmb = param['Tcmb'].squeeze()
    yinit = jnp.array([1.0, 1.0, tcmb*(1.0 + zstart), 0.0, 0.0, -tcmb*(1.0 + zstart)])

    y_prev = jnp.where(i > 0, y_arr[:, i-1], yinit[:, None])

    cond1 = (zend > 3500.0)
    cond2 = jnp.logical_and(i > 0, y_prev[1] > 0.99)
    cond3 = jnp.logical_and(i > 0, y_prev[0] > 0.99)

    def f_case1(): # if zend > 3500.0:
      return jnp.array([1.0, 1.0, tcmb*(1.0 + zend), 0.0, 0.0, -tcmb*(1.0 + zend)])
    
    def f_case2(): # elif i>0 and x_He0 > 0.99:
      x_H0 = 1.0
      rhs  = (jnp.exp(1.5 * jnp.log(CR * tcmb/(1.0+zend))
          - CB1_He1/(tcmb*(1.0+zend))) / Nnow) * 4.0
      x_He0 = 0.5*(jnp.sqrt((rhs-1.0)**2 + 4.0*(1.0+fHe)*rhs) - (rhs-1.0))
      dxHeIdz =((-3*(-(CB1_He1/Tcmb) + CR*Tcmb)**1.5*(Nnow + 2*fHe*Nnow + 
              4*(-((CB1_He1 - CR*Tcmb**2)/(Tcmb*(1+zend))))**1.5 - 
              jnp.sqrt(Nnow**2 + (16*(-CB1_He1 + CR*Tcmb**2)**3)/(Tcmb3*(1 + zend)**3) + 
                8*(1 + 2*fHe)*Nnow*(-((CB1_He1 - CR*Tcmb**2)/(Tcmb*(1+zend))))**1.5)))
           /(Nnow*(1 + zend)**2.5*jnp.sqrt(Nnow**2 + (16*(-CB1_He1 + CR*Tcmb**2)**3)/
               (Tcmb**3*(1 + zend)**3) + 8*(1 + 2*fHe)*Nnow*(-((CB1_He1 - CR*Tcmb**2)/(Tcmb + Tcmb*zend)))**1.5)))
      return jnp.array([x_H0, (x_He0 - 1.0)/fHe, tcmb*(1.0+zend), 0.0, dxHeIdz*dzda, -tcmb*(1.0+zend)])
    
    def f_case3(): # elif i>0 and x_H > 0.99:
      rhs   = jnp.exp(1.5*jnp.log(CR*tcmb/(1.0+zend))
          - CB1/(tcmb*(1.0+zend))) / Nnow
      x_H0  = 0.5*(jnp.sqrt(rhs**2 + 4.0*rhs) - rhs)
      dxHdz = ((3*((2*(-((CB1 - CR*Tcmb**2)/(Tcmb*(1+zend))))**1.5)/(1 + zend) + 
                   ((CB1 - CR*Tcmb**2)*(2*CB1**2 - 4*CB1*CR*Tcmb**2 + Tcmb**2*
                 (2*CR**2*Tcmb**2 + Nnow*(1 + zend)**2*jnp.sqrt(-((CB1 - CR*Tcmb**2)/(Tcmb*(1+zend)))))))/
            (Tcmb**1.5*(1 + zend)**2.5*jnp.sqrt((-CB1 + CR*Tcmb**2)*
                (CB1**2 - 2*CB1*CR*Tcmb**2 + Tcmb**2*(CR**2*Tcmb**2 + Nnow*(1 + zend)**2*
                                                      jnp.sqrt(-((CB1 - CR*Tcmb**2)/(Tcmb*(1+zend))))))))))/(2.*Nnow))


      y_sol = solve_ionization(astart=astart, aend=aend, ystart=y_prev, rtol=rtol, atol=atol, max_steps=128, param=param)
      y_sol = y_sol.at[0].set(x_H0)
      y_sol = y_sol.at[3].set(dxHdz*dzda)
      return y_sol
    
    def f_case4(): # else:
      return solve_ionization(astart=astart, aend=aend, ystart=y_prev, rtol=rtol, atol=atol, max_steps=128, param=param)

    new_val = jax.lax.cond(
      cond1,
      f_case1,
      lambda: jax.lax.cond(
        cond2,
        f_case2,
        lambda: jax.lax.cond(cond3, f_case3, f_case4)
      )
    )

    return y_arr.at[:, i].set(new_val)

  #y_final = jax.lax.fori_loop(0, N, loop_body, y_init)


  # FIRST POINT OF ORDER : DETERMINE SAHA INPUTS
  Nnow_2d = jnp.atleast_1d(Nnow)[:, jnp.newaxis]
  fHe_2d = jnp.atleast_1d(fHe)[:, jnp.newaxis]
  z_2d = (1/a-1)[jnp.newaxis, :]
  #Saha_inputs = {'HeII':[const_EionHe12s, 1.0 + fHe_2d, 1.0, fHe_2d], 'HeI':[CB1_He1, 1.0, 4.0, fHe_2d], 'HI':[CB1, 0.0, 1.0, 1.0]}
  Saha_inputs = {'HeII':[const_EionHe12s, 1.0 + fHe, 1.0, fHe], 'HeI':[CB1_He1, 1.0, 4.0, fHe], 'HI':[CB1, 0.0, 1.0, 1.0]}


  # --- Hydrogen Logic ---
  #def compute_xH(z):
  #    T_z = Tcmb * (1.0 + z)
  #    return Saha(T_z, z, Nnow_2d, *Saha_inputs['HI'])
  def compute_xH(z):
      T_z = Tcmb * (1.0 + z)
      return Saha(T_z, z, Nnow, *Saha_inputs['HI'])

  #x_H, dxH_dz = jax.jvp(compute_xH, (z_2d,), (jnp.ones_like(z_2d),))
  dxH_func = jax.jacobian(compute_xH)
  xH_ini = compute_xH(1/a[0]-1)

  # --- Helium Total Logic ---
  #def compute_xHe(z):
  #  T_z = Tcmb * (1.0 + z)
  #  val_HeI  = Saha(T_z, z, Nnow_2d, *Saha_inputs['HeI'])
  #  val_HeII = Saha(T_z, z, Nnow_2d, *Saha_inputs['HeII'])
  #  return (val_HeI + val_HeII) / fHe
  def compute_xHe(z):
     T_z = Tcmb * (1.0 + z)
     val_HeI  = Saha(T_z, z, Nnow, *Saha_inputs['HeI'])
     val_HeII = Saha(T_z, z, Nnow, *Saha_inputs['HeII'])
     return (val_HeI + val_HeII) / fHe

  #x_He, dxHe_dz = jax.jvp(compute_xHe, (z_2d,),(jnp.ones_like(z_2d),))
  dxHe_func = jax.jacobian(compute_xHe)
  xHe_ini = compute_xHe(1/a[0]-1)
  
  #threshold = 1e-9
  #dxHe_dz = jnp.where(dxHe_dz < threshold, 0.0, dxHe_dz)
  #dxH_dz = jnp.where(dxH_dz < threshold, 0.0, dxH_dz)
  jnp.set_printoptions(threshold=jnp.inf)
  #import numpy as np
  #print("z = {}".format(np.array(1/a-1)))
  #print("x_H = {}".format(np.array(x_H)))
  #print("x_He = {}".format(np.array(x_He)))
  #print("dx_H = {}".format(np.array(dxH_dz)))
  #print("dx_He = {}".format(np.array(dxHe_dz)))
  
  #def get_f_curve(z, f_prime, f0=0):
  #  # Calculate the spacing between points
  #  dz = jnp.diff(z)
  #  # Trapezoid rule: average height * width
  #  avg_heights = (f_prime[:-1] + f_prime[1:]) / 2.0
  #  integrals = avg_heights * dz
  #  
  #  # Prefix with 0 and sum to get the curve
  #  f_values = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(integrals)])
  #  return f_values + f0
    
  #print(get_f_curve(1/a-1, dxH_dz[0]))
  #print(get_f_curve(1/a-1, dxHe_dz[0]))
  #quit()

  #saha_inputs = (x_H, dxH_dz, x_He, dxHe_dz)
  #saha_inputs = [spline_interpolation(jnp.log(a), jnp.log(dxHe_dz+threshold).T), spline_interpolation(jnp.log(a), jnp.log(dxH_dz+threshold).T)]
  saha_inputs = [dxH_func, dxHe_func]
  
  """jax.debug.print("Comparison {a}={b}", a=jnp.log(dxHe_dz+threshold), b=saha_inputs[0]._y_)
  
  aidx = jnp.arange(len(a))
  jax.debug.print("a = {a}, aidx={aidx}, dxHe_dz = {dxHe_dz}",a=a, aidx=aidx, dxHe_dz=dxHe_dz[0])
  jax.debug.print("stack = {s}",s=jnp.stack([a, aidx,dxHe_dz[0]],axis=1))
  jax.debug.print("\n\n")
  def debug_eval(aval):

    x_new = jnp.atleast_1d(jnp.log(aval))
    x_new = x_new.reshape(x_new.shape[0], -1) # Extend last dimension if not already done
    idx = saha_inputs[0]._find_index(x_new)
    xidx = jnp.take_along_axis(saha_inputs[0]._x_, idx, axis=0)
    yidx = jnp.take_along_axis(saha_inputs[0]._y_, idx, axis=0)
    dx = jnp.exp(saha_inputs[0].evaluate(jnp.log(aval)))-threshold
    jax.debug.print("At a={a}, idx={idx}, dx={dx},  xclose={xidx},yclose={yidx}, yclose_traf={ytraf}", a=aval, idx=idx, dx=dx, xidx=xidx, yidx=yidx, ytraf=jnp.exp(yidx)-threshold)
    return dx

  avals = jnp.geomspace(1e-10, 1, num=1000)
  results = jax.vmap(debug_eval)(avals)
  
  jax.debug.print("\n\n")
  #a_spec = 1.71506123e-04
  #jax.debug.print("Final eval at a={a}, dx={dx}",a=a_spec, dx=jnp.exp(saha_inputs[0].evaluate(jnp.log(a_spec)))-threshold)
  """

  # SECOND POINT OF ORDER : DETERMINE SHAPE!!
  astart = a0
  aend = a1
  zstart = 1.0/astart - 1.0

  #xH_ini = x_H[:, 0]
  #xHe_ini = x_He[:, 0]
  T_ini = Tcmb*(1.0 + zstart)
  #print(xH_ini, xHe_ini)
  y0 = jnp.stack(jnp.broadcast_arrays(xH_ini, xHe_ini, T_ini), axis=1)
  #print(y0.shape, xH_ini.shape, xHe_ini.shape, T_ini.shape)
  
  #rint(param, saha_inputs)
  ionization_partial = partial(ionization, compute_xH_fn=dxH_func, compute_xHe_fn=dxHe_func)
  dy0 = jax.eval_shape(ionization_partial, astart, y0, (param,))
  y0 = jnp.broadcast_to(y0, dy0.shape)

  #return a.reshape(len(a),1,1), a.reshape(len(a),1,1), a
  sol =drx.diffeqsolve(
      terms=drx.ODETerm(ionization_partial),
      solver=drx.Dopri5(),#GRKT4(),
      t0=jnp.log(astart),
      t1=jnp.log(aend),
      dt0=jnp.log(aend/astart)*1e-3,#jnp.abs(astart*1e-3),
      y0=y0,
      #stepsize_controller = drx.PIDController(rtol=rtol,atol=atol, dtmax=0.01), 
      stepsize_controller = drx.PIDController(rtol=rtol,atol=atol, dtmax=0.1), 
      #max_steps=max_steps,
      args=(param,saha_inputs),
      # adjoint=drx.RecursiveCheckpointAdjoint(),
      #adjoint=drx.ForwardMode(),
      #adjoint=drx.ImplicitAdjoint(),
      adjoint=drx.RecursiveCheckpointAdjoint(),
      throw=False,
      saveat=drx.SaveAt(dense=True)
  )


  #jax.debug.print("ys = {v} \n , dys = {dv}",v=jax.vmap(sol.evaluate)(jnp.log(a)), dv=jax.vmap(sol.derivative)(jnp.log(a)))
  return jax.vmap(sol.evaluate)(jnp.log(a)), jax.vmap(sol.derivative)(jnp.log(a)), a


@partial(jax.jit, static_argnames=("num_thermo", "rtol", "atol"))
def evaluate_thermo(
    *, param: dict, num_thermo=2048, rtol=1e-3, atol=1e-6
) -> jax.Array:
    
    param['fHe'] = param['YHe']/(const_mHe_mH*(1.0-param['YHe']))
    
    y, dy, a = compute_thermal_history(
        a0=param['amin'],
        a1=param['amax'],
        N=num_thermo,
        rtol=rtol,
        atol=atol,
        param=param,
    )
    
    print(y.shape, dy.shape)

    # extract the relevant quantities from the solution
    xeHI      = y[:, :, 0]
    xHe       = y[:, :, 1]
    #xeHeI     = y[:, :, 1]
    #xeHeII    = Saha_HeII(a, param)
    # val_HeII = Saha(T_z, z, Nnow, *Saha_inputs['HeII'])
    #xe        = xeHI + param['fHe'] * xeHeI + xeHeII
    xe        = xeHI + param['fHe'] * xHe
    mu        = 1/(1 + (1/const_mHe_mH-1) * param['YHe'] + (1-param['YHe']) * xe)
    Tm        = y[:, :, 2]

    # extract the derivatives that were also computed, which allows to compute cs2 and dxedtau
    dxeHIda  = dy[:, :, 0]
    #dxeHeIda = y[:, :, 1]
    dxeHeda = dy[:, :, 1]

    #a_broadcasted = jnp.broadcast_to(a[:, None], (1025, 5))
    #val, vjp_fun = jax.vjp(lambda x: Saha_HeII(x, param), a_broadcasted)
    #dxHeIIda = vjp_fun(jnp.ones_like(val))[0]

    dTmda    = dy[:, :, 2]

    daTmda   = Tm + a[:, None] * dTmda
    cs2      = const_kB/ const_mH / const_c**2 / mu * Tm * (4 - daTmda / (Tm)) /3
    #dxeda = (dxeHIda + param['fHe'] * dxeHeIda + dxHeIIda)
    dxeda = (dxeHIda + param['fHe'] * dxeHeda)
    #from .background import dadtau, dtauda_
    #dxedtau  = (dxeHIda + param['fHe'] * dxeHeIda + dxHeIIda) * dadtau(a=a, param=param)

    ## compute conformal times tau for all entries in a
    ## vmap romb over all intervals in parallel instead of sequentially via scan
    #_dtauda = lambda a_: dtauda_(a_, param['grhom'], param['grhog'], param['grhor'],
    #                param['Omegam'], param['OmegaDE'],
    #                param['w_DE_0'], param['w_DE_a'],
    #                param['Omegak'], param['Neff'], param['Nmnu'],
    #                param['logrhonu_of_loga_spline'])
    #tau_increments = jax.vmap(lambda lo, hi: romb(_dtauda, lo, hi))(a[:-1], a[1:])
    #tau0 = param['taumin']
    #tau = tau0 + jnp.concatenate([jnp.array([0.0]), jnp.cumsum(tau_increments)])

    #return param, tau, a, cs2, Tm, mu, xe, xeHI, xeHeI, xeHeII, dxedtau
    return a, cs2, Tm, mu, xe, dxeda
