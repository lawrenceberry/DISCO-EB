
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
 
import math
from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np

from .spline_interpolation import spline_interpolation

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

CR = 2.0*math.pi*(const_me/const_h)*(const_kB/const_h) 
CK = Lalpha**3/(8.0*math.pi)         
CK_He = Lalpha_He**3/(8.0*math.pi) 
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

# ---------------------------------------------------------------------------
# Accelerator right-hand side
# ---------------------------------------------------------------------------
# The recombination history is integrated by modax's numba-CUDA kernels, one
# cosmology per thread, so the right-hand side below is written for
# ``cuda.jit(device=True)``: scalar ``math`` on a tuple state ``(x_H, x_He, T_m)``
# and a per-cosmology parameter row ``p`` indexed at the constants below. It is
# the RECFAST ``ionization`` routine statement for statement, with the
# ``jnp.where`` case selection written as branches and the two Saha slopes
# (which the JAX version took by ``jax.jacobian``) differentiated by hand.

P_H0 = 0
P_YHE = 1
P_OMEGAB = 2
P_TCMB = 3
P_GRHOM = 4
P_GRHOG = 5
P_GRHOR = 6
P_OMEGAM = 7
P_OMEGADE = 8
P_W_DE_0 = 9
P_W_DE_A = 10
P_OMEGAK = 11
P_NEFF = 12
P_NMNU = 13
P_AMNU = 14
N_THERMO_PARAMS = 15
"""Columns of the per-cosmology parameter row the device right-hand side reads."""

NQ_NU_BACKGROUND = 8
"""Momentum nodes of the massive-neutrino background density, as :func:`background.nu_background`."""

THERMO_SOLVER = "rodas5P"
"""Which modax kernel integrates recombination: ``"rodas5P"`` or ``"tsit5"``."""

_RECFAST_RHS = None


def recfast_rhs():
    """Return (and cache) the device right-hand side ``f(y, ln a, p) -> dy/dln a``.

    ``y = (x_H, x_He, T_m)`` with ``x_H = n_HII / n_H``, ``x_He = (n_HeII + 2 n_HeIII) / n_He``
    and the matter temperature in K; ``p`` is a :data:`N_THERMO_PARAMS` row.
    The function object is the identity key of modax's compiled-kernel cache, so
    it is built once per process.
    """

    global _RECFAST_RHS
    if _RECFAST_RHS is not None:
        return _RECFAST_RHS

    from numba_cuda_mlir import cuda

    from .background import get_neutrino_momentum_bins

    # Host constants even when first built inside a jit trace.
    with jax.ensure_compile_time_eval():
        q_nodes, w_nodes = (
            np.asarray(v, dtype=np.float64) for v in get_neutrino_momentum_bins(NQ_NU_BACKGROUND)
        )

    def neutrino_density(nodes):
        """rho_nu / rho_nu(massless), the quadrature of ``nu_background`` folded node by node."""
        (q, w), rest = nodes[-1], nodes[:-1]
        q, w = float(q), float(w)
        if not rest:

            @cuda.jit(device=True)
            def rhonu(a, amnu):
                return w * math.sqrt(1.0 + (a * amnu / q) ** 2)

            return rhonu

        below = neutrino_density(rest)

        @cuda.jit(device=True)
        def rhonu(a, amnu):
            return below(a, amnu) + w * math.sqrt(1.0 + (a * amnu / q) ** 2)

        return rhonu

    rhonu_of = neutrino_density(list(zip(q_nodes, w_nodes)))

    @cuda.jit(device=True)
    def hubble(a, p):
        """H(a) in 1/s, from the same densities as :func:`background.get_aprimeoa`."""
        rhonu = rhonu_of(a, p[P_AMNU])
        w0 = p[P_W_DE_0]
        wa = p[P_W_DE_A]
        rho_Q = a ** (-3.0 * (1.0 + w0 + wa)) * math.exp(3.0 * (a - 1.0) * wa)
        grhom = p[P_GRHOM]
        grho = (
            grhom * p[P_OMEGAM] / a
            + (p[P_GRHOG] + p[P_GRHOR] * (p[P_NEFF] + p[P_NMNU] * rhonu)) / (a * a)
            + grhom * p[P_OMEGADE] * rho_Q * a * a
            + grhom * p[P_OMEGAK]
        )
        aprimeoa = math.sqrt(grho / 3.0)  # conformal H in 1/Mpc, times 1e5 below
        return 1.0e-5 * aprimeoa / a * const_c * bigH

    @cuda.jit(device=True)
    def saha_slope(z, Tcmb, Nnow, E_ion, offset, g_factor, f_factor):
        """``d x / d z`` of the Saha ionization fraction :func:`Saha` gives at ``z``."""
        T = Tcmb * (1.0 + z)
        phi = (CR * T) ** 1.5 / Nnow / (1.0 + z) ** 3
        R = g_factor * phi * math.exp(-E_ion / T)
        if R <= 1.0e-20:
            return 0.0
        if T > E_ion * 0.1:  # fully ionized: the quadratic has constant coefficients
            return 0.0
        alpha = R + offset
        beta = R * f_factor
        dR = R * (-1.5 / (1.0 + z) + E_ion / (Tcmb * (1.0 + z) ** 2))
        discr = alpha * alpha + 4.0 * beta
        if discr < 1.0e-35:
            root = math.sqrt(1.0e-35)
            droot = 0.0
        else:
            root = math.sqrt(discr)
            droot = (alpha + 2.0 * f_factor) * dR / root
        denom = root + alpha
        return (2.0 * f_factor * dR * denom - 2.0 * beta * (droot + dR)) / (denom * denom)

    def rhs(y, lna, p):
        a = math.exp(lna)
        z = 1.0 / a - 1.0
        YHe = p[P_YHE]
        Tcmb = p[P_TCMB]

        # Pre-compute derived constants
        H = p[P_H0] / 100.0
        HO = H * bigH
        fu = 1.105
        mu_H = 1.0 / (1.0 - YHe)
        fHe = YHe / (const_mHe_mH * (1.0 - YHe))
        Nnow = const_dens_fac * H * H * p[P_OMEGAB] / mu_H

        x_H = y[0]
        if x_H < 0.0:
            x_H = 0.0
        x_He = y[1]
        if x_He < 0.0:
            x_He = 0.0
        x = x_H + fHe * x_He
        Tmat = abs(y[2])

        # Calculate common terms once
        z_term = 1.0 + z
        n = Nnow * z_term**3
        n_He = fHe * n
        Trad = Tcmb * z_term
        Hz = hubble(a, p)
        Hz_z_term = Hz * z_term
        one_minus_x_He = 1.0 - x_He
        one_minus_x_H = 1.0 - x_H
        n_He_1_minus_x_He = n_He * one_minus_x_He

        # Temperature and rate calculations
        Tmat_1e4 = Tmat / 1.0e4
        CR_Tmat_15 = (CR * Tmat) ** 1.5

        # Get the radiative rates
        Rdown = 1.0e-19 * a_PPB * Tmat_1e4**b_PPB / (1.0 + c_PPB * Tmat_1e4**d_PPB)
        Rup = Rdown * CR_Tmat_15 * math.exp(-CDB / Tmat)

        # Calculate He rates using a fit
        sq_0 = math.sqrt(Tmat / T_0)
        sq_1 = math.sqrt(Tmat / T_1)
        Rdown_He = a_VF / (sq_0 * (1.0 + sq_0) ** (1.0 - b_VF) * (1.0 + sq_1) ** (1.0 + b_VF))
        Rup_He = 4.0 * Rdown_He * CR_Tmat_15 * math.exp(-CDB_He / Tmat)

        # Boltzmann factor with numerical stability
        He_Boltz = math.exp(min(680.0, Bfact / Tmat))

        # HeI triplets
        Rdown_trip = a_trip / (
            sq_0 * (1.0 + sq_0) ** (1.0 - b_trip) * (1.0 + sq_1) ** (1.0 + b_trip)
        )
        Rup_trip = (
            Rdown_trip
            * math.exp(-const_h * const_c * L_He2St_ion / (const_kB * Tmat))
            * CR_Tmat_15
            * (4.0 / 3.0)
        )

        # Peebles coefficient with the Gaussian fudges of Rubino-Martin et al. (2010)
        log_z_term = math.log(z_term)
        K = (
            CK
            / Hz
            * (
                1.0
                + AGauss1 * math.exp(-(((log_z_term - zGauss1) / wGauss1) ** 2))
                + AGauss2 * math.exp(-(((log_z_term - zGauss2) / wGauss2) ** 2))
            )
        )

        # --- Hydrogen ------------------------------------------------------------
        if x_H > 0.99:
            # Saha, up to the redshift where the case-B rate equations take over
            f0 = saha_slope(z, Tcmb, Nnow, CB1, 0.0, 1.0, 1.0)
        else:
            rate_diff = x * x_H * n * Rdown - Rup * one_minus_x_H * math.exp(-CL / Tmat)
            if x_H > 0.985:
                f0 = rate_diff / Hz_z_term
            else:
                K_Lambda_n_1_minus_x_H = K * Lambda * n * one_minus_x_H
                denom_term = 1.0 / fu + K_Lambda_n_1_minus_x_H / fu + K * Rup * n * one_minus_x_H
                f0 = rate_diff * (1.0 + K_Lambda_n_1_minus_x_H) / (Hz_z_term * denom_term)

        # --- Helium --------------------------------------------------------------
        if x_He > 0.9 or a < 3.0e-4:
            # Saha for HeIII -> HeII and HeII -> HeI, summed over both stages
            f1 = (
                saha_slope(z, Tcmb, Nnow, CB1_He1, 1.0, 4.0, fHe)
                + saha_slope(z, Tcmb, Nnow, const_EionHe12s, 1.0 + fHe, 1.0, fHe)
            ) / fHe
        elif x_He < 1.0e-8:
            f1 = 0.0
        else:
            # Sobolev escape probability of the HeI 2^1P - 1^1S line
            tauHe_s = A2P_s * CK_He * 3.0 * n_He_1_minus_x_He / Hz
            pHe_s = (1.0 - math.exp(-tauHe_s)) / tauHe_s
            Doppler_term = math.sqrt(2.0 * const_kB * Tmat / (const_mH * const_mHe_mH * const_c**2))
            # K_He: continuum opacity of neutral hydrogen (KIV 2007) while any
            # hydrogen is neutral. (Here 1e-8 <= x_He <= 0.9, so the RECFAST
            # fall-back to CK_He / Hz outside that range cannot be reached.)
            if x_H < 0.9999999:
                Doppler_2p = const_c * L_He_2p * Doppler_term
                gamma_2Ps = (
                    3.0 * A2P_s * fHe * one_minus_x_He * const_c**2
                    / (
                        math.sqrt(math.pi)
                        * sigma_He_2Ps
                        * 8.0
                        * math.pi
                        * Doppler_2p
                        * one_minus_x_H
                        * (const_c * L_He_2p) ** 2
                    )
                )
                pb = 0.36  # value from KIV (2007)
                qb = 0.86  # He fudge factor
                AHcon = A2P_s / (1.0 + pb * gamma_2Ps**qb)
                K_He = 1.0 / ((A2P_s * pHe_s + AHcon) * 3.0 * n_He_1_minus_x_He)
            else:
                K_He = 1.0 / (A2P_s * pHe_s * 3.0 * n_He_1_minus_x_He)

            rate_diff_He = x * x_He * n * Rdown_He - Rup_He * one_minus_x_He * math.exp(-CL_He / Tmat)
            K_He_Lambda = K_He * Lambda_He * n_He_1_minus_x_He * He_Boltz
            He_denom_term = 1.0 + K_He * (Lambda_He + Rup_He) * n_He_1_minus_x_He * He_Boltz
            f1 = rate_diff_He * (1.0 + K_He_Lambda) / (Hz_z_term * He_denom_term)

            # Triplet contribution (2^3P - 1^1S), likewise always reached here
            tauHe_t = A2P_t * n_He_1_minus_x_He * 3.0 / (8.0 * math.pi * Hz * L_He_2Pt**3)
            pHe_t = (1.0 - math.exp(-tauHe_t)) / tauHe_t
            exp_CL_PSt = math.exp(-CL_PSt / Tmat)
            if x_H > 0.99999:
                CfHe_t = A2P_t * pHe_t * exp_CL_PSt
            else:
                Doppler_2Pt = const_c * L_He_2Pt * Doppler_term
                gamma_2Pt = (
                    3.0 * A2P_t * fHe * one_minus_x_He * const_c**2
                    / (
                        math.sqrt(math.pi)
                        * sigma_He_2Pt
                        * 8.0
                        * math.pi
                        * Doppler_2Pt
                        * one_minus_x_H
                        * (const_c * L_He_2Pt) ** 2
                    )
                )
                pb_t = 0.66  # KIV (2007) parameters
                qb_t = 0.9
                AHcon_t = A2P_t / (1.0 + pb_t * gamma_2Pt**qb_t) / 3.0
                CfHe_t = (A2P_t * pHe_t + AHcon_t) * exp_CL_PSt
            CfHe_t = CfHe_t / (Rup_trip + CfHe_t)  # "C" factor for triplets
            trip_rate_diff = (
                x * x_He * n * Rdown_trip
                - one_minus_x_He * 3.0 * Rup_trip * math.exp(-const_h_c_L_He_2St_kB / Tmat)
            )
            f1 = f1 + trip_rate_diff * CfHe_t / Hz_z_term

        # --- Matter temperature ---------------------------------------------------
        timeTh = (1.0 / (CT * Trad**4)) * (1.0 + x + fHe) / x
        timeH = 2.0 / (3.0 * HO * z_term**1.5)
        if timeTh < H_frac * timeH:
            # Tight Compton coupling: T_m follows T_rad, to first order
            f2 = Tmat / z_term
        else:
            f2 = (
                CT * Trad**4 * x / (1.0 + x + fHe) * (Tmat - Trad) / Hz_z_term
                + 2.0 * Tmat / z_term
            )

        # The rates above are d/dz; the solver integrates in ln a.
        dzdlna = -1.0 / a
        return (f0 * dzdlna, f1 * dzdlna, f2 * dzdlna)

    _RECFAST_RHS = rhs
    return rhs


def Saha(T, z, N_now, E_ion, offset, g_factor, f_factor):
    """Saha ionization fraction (JAX). Its ``z`` slope is :func:`recfast_rhs`'s ``saha_slope``."""

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


def _get_adaptive_sampling(a0: float, a1: float, N: int) -> np.ndarray:
  """
  Generate adaptive sampling in scale factor with concentration around recombination.

  Distributes points to concentrate sampling where ionization fraction changes rapidly:
  - 5% very early times (z > 3000)
  - 10% pre-recombination (1400 < z < 3000)
  - 50% recombination era (600 < z < 1400) - where xe changes most
  - 35% post-recombination (z < 600)

  This improves accuracy of spline interpolation without increasing total points.
  The grid is a host constant: it is the kernel's save-time table.
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
  a_break1 = max(a_break1, a0)

  # Create segments with geometric spacing
  a1_seg = np.geomspace(a0, a_break1, n_very_early, endpoint=False)
  a2_seg = np.geomspace(a_break1, a_break2, n_pre_recomb, endpoint=False)
  a3_seg = np.geomspace(a_break2, a_break3, n_recomb, endpoint=False)
  a4_seg = np.geomspace(a_break3, a1, n_post)

  return np.concatenate([a1_seg, a2_seg, a3_seg, a4_seg])


def thermo_parameter_rows(param: dict) -> jax.Array:
  """Stack the background parameters into ``(B, N_THERMO_PARAMS)`` rows for the kernel."""

  keys = ('H0', 'YHe', 'Omegab', 'Tcmb', 'grhom', 'grhog', 'grhor', 'Omegam',
          'OmegaDE', 'w_DE_0', 'w_DE_a', 'Omegak', 'Neff', 'Nmnu', 'amnu')
  columns = jnp.broadcast_arrays(*[jnp.reshape(jnp.asarray(param[k], dtype=jnp.float64), (-1,)) for k in keys])
  return jnp.stack(columns, axis=-1)


def compute_thermal_history( *, a0 : float, a1 : float, N : int, rtol : float = 1e-3, atol : float = 1e-6, param : dict ) -> Tuple[jnp.ndarray, jnp.ndarray, np.ndarray]:
  """
  Compute the thermal history of the Universe from a0 to a1 in N steps

  The RECFAST system is integrated on the GPU by modax (:data:`THERMO_SOLVER`),
  one cosmology per CUDA thread. Called with a batch of cosmologies it is one
  kernel launch; called for one cosmology inside an outer ``jax.vmap`` it is
  still one launch, since modax's batching rule turns the vmap into an
  ensemble solve.

  Args:
      a0 (float): initial scale factor
      a1 (float): final scale factor
      N (int): number of steps
      rtol (float, optional): relative tolerance. Defaults to 1e-3.
      atol (float, optional): absolute tolerance. Defaults to 1e-6.
      param (dict): dictionary of cosmological parameters

  Returns:
      ``y`` of shape ``(N+1, B, 3)`` holding ``[x_H, x_He, T_m]``, its derivative
      ``dy/da`` of the same shape, and the scale factors ``a`` of shape ``(N+1,)``.
  """

  # Use adaptive sampling that concentrates points around recombination
  a = _get_adaptive_sampling(a0, a1, N+1)
  lna = jnp.asarray(np.log(a))

  rows = thermo_parameter_rows(param)
  B = rows.shape[0]

  # Deep in the fully ionized era both Saha stages sit at their limits:
  # x_H = 1 and x_He = 2 (HeIII), and T_m = T_rad.
  T_ini = rows[:, P_TCMB] * (1.0 / a0)
  y0 = jnp.stack([jnp.ones(B), 2.0 * jnp.ones(B), T_ini], axis=-1)

  if THERMO_SOLVER == "rodas5P":
    from modax.rodas5P import solve as modax_solve
    settings = dict(lu_precision="fp64")
  elif THERMO_SOLVER == "tsit5":
    from modax.tsit5 import solve as modax_solve
    settings = {}
  else:
    raise ValueError(f"unknown THERMO_SOLVER {THERMO_SOLVER!r}")

  if B == 1:
    # A scalar solve, so that an enclosing jax.vmap over cosmologies lowers to
    # one ensemble launch instead of one per lane.
    sol = modax_solve(recfast_rhs(), y0[0], lna, rows[0], rtol=rtol, atol=atol, **settings)
  else:
    sol = modax_solve(recfast_rhs(), y0, lna, rows, rtol=rtol, atol=atol, **settings)
  y = jnp.moveaxis(sol, 0, 1)  # (N+1, B, 3)

  # The slopes cs2 needs, from a spline of the saved history on its own grid.
  spline = spline_interpolation(lna, y.reshape(N+1, 3*B))
  dy_dlna = spline.derivative(lna).reshape(N+1, B, 3)
  dy = dy_dlna / jnp.asarray(a)[:, None, None]

  return y, dy, a


@partial(jax.jit, static_argnames=("num_thermo", "rtol", "atol", "amin", "amax"))
def evaluate_thermo(
    *, param: dict, num_thermo=2048, rtol=1e-3, atol=1e-6, amin: float = 1e-9, amax: float = 1.01
) -> jax.Array:
    """Thermal history on the adaptive grid: ``(a, cs2, T_m, mu, x_e, dx_e/da)``.

    ``amin`` and ``amax`` are static: the save-time grid they span is a host
    constant of the kernel launch.
    """

    param['fHe'] = param['YHe']/(const_mHe_mH*(1.0-param['YHe']))
    
    y, dy, a = compute_thermal_history(
        a0=amin,
        a1=amax,
        N=num_thermo,
        rtol=rtol,
        atol=atol,
        param=param,
    )
    a = jnp.asarray(a)

    # extract the relevant quantities from the solution
    xeHI      = y[:, :, 0]
    xHe       = y[:, :, 1]
    xe        = xeHI + param['fHe'] * xHe
    mu        = 1/(1 + (1/const_mHe_mH-1) * param['YHe'] + (1-param['YHe']) * xe)
    Tm        = y[:, :, 2]

    # extract the derivatives that were also computed, which allows to compute cs2 and dxedtau
    dxeHIda  = dy[:, :, 0]
    dxeHeda = dy[:, :, 1]
    dTmda    = dy[:, :, 2]

    daTmda   = Tm + a[:, None] * dTmda
    cs2      = const_kB/ const_mH / const_c**2 / mu * Tm * (4 - daTmda / (Tm)) /3
    dxeda = (dxeHIda + param['fHe'] * dxeHeda)

    return a, cs2, Tm, mu, xe, dxeda
