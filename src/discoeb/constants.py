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
