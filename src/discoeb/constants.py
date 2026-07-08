"""Physical constants and unit conversions.

All quantities are given in SI units unless the name states otherwise. The
derived cosmological normalizations (``GRHO_CRITICAL_H2``, ``RHO_CRIT_100_SI``,
...) are the building blocks used by :mod:`discoeb.background` to convert the
physical density parameters supplied by the user (``Omega_i``, ``Omega_i h^2``,
``T_cmb``, ...) into the CAMB-style ``grho`` coefficients that enter the
Friedmann equation.
"""

import math

C_KM_S = 2.99792458e5
"""Speed of light in km/s."""

C_SI = C_KM_S * 1.0e3
"""Speed of light in m/s."""

SIGMA_T = 6.6524587321e-29
"""Thomson scattering cross section in m^2."""

G_NEWTON = 6.67430e-11
"""Newton's gravitational constant in m^3 kg^-1 s^-2."""

MPC_IN_M = 3.0856775814913673e22
"""One megaparsec in metres."""

SIGMA_SB = 5.670374419e-8
"""Stefan-Boltzmann constant in W m^-2 K^-4."""

K_B_SI = 1.380649e-23
"""Boltzmann constant in J K^-1."""

EV_IN_J = 1.602176634e-19
"""One electron volt in joules."""

M_H = 1.673575e-27
"""Hydrogen atom mass in kg."""

M_HE4 = 6.646479073e-27
"""Helium-4 atom mass in kg."""

M_HE4_OVER_M_H = M_HE4 / M_H
"""Helium/hydrogen mass ratio; close to, but not exactly, 4."""

GRHO_CRITICAL_H2 = 3.0 * (100.0 / C_KM_S) ** 2
"""Critical-density normalization in CAMB ``grho`` units.

The physical critical density today is defined with the actual present-day
Hubble rate:

``rho_crit,0 = 3*H0^2 / (8*pi*G)``.

Cosmological density parameters are often supplied as ``Omega_i h^2``, with
``H0 = 100*h km/s/Mpc``. It is therefore convenient to define the reference
rate ``H_100 = 100 km/s/Mpc`` and use

``rho_crit,0 = rho_crit,100 * h^2``.

Then a physical density can be recovered directly from the common input
parameter:

``rho_i,0 = Omega_i*rho_crit,0 = (Omega_i h^2)*rho_crit,100``.

In the CAMB ``grho`` convention, this ``H_100`` critical density contributes

``8*pi*G*rho_crit,100 = 3*(H_100/c)^2``.

This value converts physical densities such as ``Omega_b h^2`` into ``grho``
coefficients. Numerically it reproduces the historical DISCO-EB literal
``grhom = 3.33795017e-11 * H0^2`` (with ``H0 = 100 h``) to ~1e-9 relative.
"""

H100_SI = 100.0 * 1.0e3 / MPC_IN_M
"""Reference Hubble rate ``H_100 = 100 km/s/Mpc`` in s^-1."""

RHO_CRIT_100_SI = 3.0 * H100_SI**2 / (8.0 * math.pi * G_NEWTON)
"""SI critical density for ``H_100 = 100 km/s/Mpc``.

The expression is

``rho_crit,100 = 3*H_100^2 / (8*pi*G)``.

It is used for quantities that must pass through SI units, such as photon
energy density and the present-day hydrogen number density.
"""

NEUTRINO_TEMPERATURE_FACTOR = (4.0 / 11.0) ** (1.0 / 3.0)
"""Neutrino-to-photon temperature ratio ``T_nu0 / T_cmb = (4/11)^(1/3)``.

After electron-positron annihilation heats the photon bath but not the
already-decoupled neutrinos, the neutrino temperature today is reduced by this
factor relative to the CMB temperature.
"""

NEUTRINO_MASS_KELVIN_PER_EV = 1.62581581e4
"""Conversion from neutrino mass in eV to the dimensionless mass parameter.

The dimensionless massive-neutrino mass parameter is

``amnu = m_nu c^2 / (k_B T_nu0)``,

with ``T_nu0 = (4/11)^(1/3) T_cmb`` the neutrino temperature today. Factoring
out the CMB temperature gives ``amnu = m_nu[eV] * NEUTRINO_MASS_KELVIN_PER_EV
/ T_cmb[K]``, so this constant is

``(eV / k_B) / (4/11)^(1/3) ~ 1.62581581e4 K/eV``.

The historical DISCO-EB literal is kept here to preserve bit-level behaviour of
the background pipeline.
"""

FERMI_DIRAC_CONST = 5.682196976983475
"""Fermi-Dirac normalization ``7*pi^4/120``.

This is the dimensionless momentum integral of the relativistic Fermi-Dirac
distribution and normalizes the neutrino momentum-bin weights so that a single
massless flavour has unit density.
"""

H_PLANCK = 6.62607015e-34
"""Planck constant in J s (PDG 2023)."""

M_E = 9.1093837015e-31
"""Electron mass in kg (PDG 2023)."""


# =============================================================================
# RECFAST recombination constants
#
# Atomic-physics inputs and the derived RECFAST prefactors used by
# :mod:`discoeb.recfast`. The names mirror the historical RECFAST code base
# (Seager, Sasselov & Scott 1999/2000; Wong, Moss & Scott 2008). Raw wavenumbers
# are in m^-1, decay rates in s^-1, cross sections in m^2; the derived ``C*``
# quantities are the temperature-like / geometric prefactors that appear in the
# RECFAST right-hand side.
# =============================================================================

L_H_ION = 1.096787737e7
"""Hydrogen ionization wavenumber in m^-1."""

L_H_ALPHA = 8.225916453e6
"""Hydrogen Ly-alpha transition wavenumber in m^-1."""

L_HE1_ION = 1.98310772e7
"""Neutral-helium ionization wavenumber in m^-1."""

L_HE2_ION = 4.389088863e7
"""Singly ionized helium ionization wavenumber in m^-1."""

L_HE_2S = 1.66277434e7
"""Neutral-helium 2S singlet wavenumber in m^-1."""

L_HE_2P = 1.71134891e7
"""Neutral-helium 2P singlet wavenumber in m^-1."""

L_HE_2PT = 1.690871466e7
"""Neutral-helium 2P triplet wavenumber in m^-1."""

L_HE_2ST = 1.5985597526e7
"""Neutral-helium 2S triplet wavenumber in m^-1."""

L_HE2ST_ION = 3.8454693845e6
"""Neutral-helium triplet ionization threshold wavenumber in m^-1."""

LAMBDA_2S1S = 8.2245809
"""Hydrogen two-photon decay rate in s^-1."""

LAMBDA_HE = 51.3
"""Helium two-photon decay rate in s^-1."""

A2P_S = 1.798287e9
"""Helium singlet 2P spontaneous decay rate in s^-1."""

A2P_T = 177.58
"""Helium triplet 2P spontaneous decay rate in s^-1."""

SIGMA_HE_2PS = 1.436289e-22
"""Helium singlet continuum opacity cross section in m^2."""

SIGMA_HE_2PT = 1.484872e-22
"""Helium triplet continuum opacity cross section in m^2."""

RECFAST_FUDGE = 1.125
"""Hydrogen recombination fudge factor used by RECFAST."""

AGAUSS1 = -0.14
AGAUSS2 = 0.079
"""Gaussian correction amplitudes for the hydrogen escape correction."""

ZGAUSS1 = 7.28
ZGAUSS2 = 6.73
"""Gaussian correction centers in log(1+z)."""

WGAUSS1 = 0.18
WGAUSS2 = 0.33
"""Gaussian correction widths in log(1+z)."""

CR = 2.0 * math.pi * M_E * K_B_SI / H_PLANCK**2
"""Saha prefactor coefficient ``2*pi*m_e*k_B / h_P^2``."""

CB1_HE1 = H_PLANCK * C_SI * L_HE1_ION / K_B_SI
"""Neutral-helium ionization temperature in K."""

CB1_HE2 = H_PLANCK * C_SI * L_HE2_ION / K_B_SI
"""Singly ionized helium ionization temperature in K."""

CDB = H_PLANCK * C_SI * (L_H_ION - L_H_ALPHA) / K_B_SI
"""Hydrogen Balmer-continuum energy gap expressed as a temperature."""

CDB_HE = H_PLANCK * C_SI * (L_HE1_ION - L_HE_2S) / K_B_SI
"""Helium singlet continuum energy gap expressed as a temperature."""

CK = (1.0 / L_H_ALPHA) ** 3 / (8.0 * math.pi)
"""Hydrogen Sobolev wavelength factor ``lambda_alpha^3/(8*pi)``."""

CK_HE = (1.0 / L_HE_2P) ** 3 / (8.0 * math.pi)
"""Helium singlet Sobolev wavelength factor ``lambda^3/(8*pi)``."""

CL = H_PLANCK * C_SI * L_H_ALPHA / K_B_SI
"""Hydrogen Ly-alpha transition temperature in K."""

CL_HE = H_PLANCK * C_SI * L_HE_2S / K_B_SI
"""Helium singlet transition temperature in K."""

BFACT = H_PLANCK * C_SI * (L_HE_2P - L_HE_2S) / K_B_SI
"""Helium singlet 2P-2S splitting expressed as a temperature."""

CL_PST = H_PLANCK * C_SI * (L_HE_2PT - L_HE_2ST) / K_B_SI
"""Helium triplet 2P-2S splitting expressed as a temperature."""

CB1_HE2ST = H_PLANCK * C_SI * L_HE2ST_ION / K_B_SI
"""Helium triplet ionization threshold expressed as a temperature."""

CL_HE_2ST = H_PLANCK * C_SI * L_HE_2ST / K_B_SI
"""Helium triplet 2S transition temperature in K."""

A_RAD = 4.0 * SIGMA_SB / C_SI
"""Radiation constant in SI units, ``4*sigma_SB/c``."""

CT = (8.0 / 3.0) * (SIGMA_T / (M_E * C_SI)) * A_RAD
"""Compton-cooling coefficient appearing in the matter-temperature equation."""

PI = math.pi
"""Circle constant as a plain float, so backend-agnostic helpers avoid ``math.pi``."""

SQRT_PI = math.pi**0.5
"""``sqrt(pi)`` for the Doppler/Sobolev escape primitives."""
