"""Named cosmologies used by DISCO-EB tests.

The package module :mod:`discoeb.cosmology` intentionally defines only the
parameter container. Literature-specific reference cosmologies live here so
tests can choose representative models without making the library endorse a
single default data set.
"""

import dataclasses

import numpy as np

from discoeb.constants import NEUTRINO_TEMPERATURE_FACTOR
from discoeb.cosmology import Cosmology


PLANCK_2018_FLAT_LCDM = Cosmology(
    omega_b_h2=0.02238280,
    omega_c_h2=0.1201075,
    h=0.6732117,
    tau_reion=0.05430842,
    A_s=2.100549e-9,
    n_s=0.9660499,
    T_cmb=2.7255,
    Y_He=0.2454006,
    N_eff=3.046,
    k_pivot=0.05,
    Omegak=0.0,
    w_DE_0=-1.0,
    w_DE_a=0.0,
    cs2_DE=1.0,
    num_massive_neutrinos=0.0,
    mnu=0.0,
)
"""Planck 2018 baseline flat Lambda-CDM best-fit parameters.

Numbers are the high-precision best fit for the Planck Legacy Archive
``base_plikHM_TTTEEE_lowl_lowE_lensing`` chain. This test instance keeps the
massive-neutrino extension disabled so scalar massless-neutrino background
tests remain algebraic; the Planck minimal-mass neutrino convention is covered
separately by :data:`PLANCK_2018_FLAT_LCDM_MASSIVE_NU`.

Reference: Planck Collaboration, "Planck 2018 results. VI. Cosmological
parameters", A&A 641, A6 (2020); Planck Legacy Archive cosmological parameter
tables.
"""


PLANCK_2018_CURVED_LCDM = Cosmology(
    omega_b_h2=0.02242,
    omega_c_h2=0.11933,
    h=0.6766,
    tau_reion=0.0561,
    A_s=2.105209331337507e-9,
    n_s=0.9665,
    T_cmb=2.7255,
    Y_He=0.2454,
    N_eff=3.046,
    k_pivot=0.05,
    Omegak=0.0007,
    w_DE_0=-1.0,
    w_DE_a=0.0,
    cs2_DE=1.0,
    num_massive_neutrinos=0.0,
    mnu=0.0,
)
"""Planck 2018 non-flat Lambda-CDM representative.

The base densities use the Planck 2018 TT,TE,EE+lowE+lensing+BAO column, while
``Omegak = 0.0007`` is the same data combination's curvature constraint
central value for the one-parameter non-flat extension.

Reference: Planck Collaboration, "Planck 2018 results. VI. Cosmological
parameters", A&A 641, A6 (2020), non-flat Lambda-CDM constraints with BAO.
"""


DESI_2024_DYNAMICAL_DARK_ENERGY = Cosmology(
    omega_b_h2=0.02242,
    omega_c_h2=0.11933,
    h=0.6797,
    tau_reion=0.0561,
    A_s=2.105209331337507e-9,
    n_s=0.9665,
    T_cmb=2.7255,
    Y_He=0.2454,
    N_eff=3.046,
    k_pivot=0.05,
    Omegak=0.0,
    w_DE_0=-0.850,
    w_DE_a=-0.59,
    cs2_DE=1.0,
    num_massive_neutrinos=0.0,
    mnu=0.0,
)
"""Flat CPL dynamical-dark-energy representative.

The CPL values are from the Planck 2018 plus non-CMB data compilation of Park,
de Cruz Perez, and Ratra, which was designed to check the DESI 2024
``w0waCDM`` hint without using DESI BAO data. Other baseline parameters follow
the Planck 2018 TT,TE,EE+lowE+lensing+BAO column used above.

Reference: Park, de Cruz Perez, and Ratra, arXiv:2405.00502 (2024), reporting
``w0 = -0.850`` and ``wa = -0.59`` for flat CPL dark energy.
"""


PLANCK_2018_FLAT_LCDM_MASSIVE_NU = Cosmology(
    omega_b_h2=0.02238280,
    omega_c_h2=0.1201075,
    h=0.6732117,
    tau_reion=0.05430842,
    A_s=2.100549e-9,
    n_s=0.9660499,
    T_cmb=2.7255,
    Y_He=0.2454006,
    N_eff=3.046,
    k_pivot=0.05,
    Omegak=0.0,
    w_DE_0=-1.0,
    w_DE_a=0.0,
    cs2_DE=1.0,
    num_massive_neutrinos=1.0,
    mnu=0.06,
)
"""Planck 2018 flat Lambda-CDM with the minimal massive-neutrino convention.

Planck 2018 baseline chains assume one massive neutrino with
``sum_mnu = 0.06 eV`` and the remaining effective species treated as massless.
DISCO-EB's background model represents this as one degenerate massive species
of mass ``0.06 eV`` plus ``N_eff - 1`` massless species.

Reference: Planck Collaboration, "Planck 2018 results. VI. Cosmological
parameters", A&A 641, A6 (2020), baseline neutrino-mass convention.
"""


BENCHMARK_COSMOLOGIES = {
    "planck_2018_flat_lcdm": PLANCK_2018_FLAT_LCDM,
    "planck_2018_curved_lcdm": PLANCK_2018_CURVED_LCDM,
    "desi_2024_dynamical_dark_energy": DESI_2024_DYNAMICAL_DARK_ENERGY,
    "planck_2018_flat_lcdm_massive_nu": PLANCK_2018_FLAT_LCDM_MASSIVE_NU,
}
"""Registry of representative literature cosmologies."""


def perturbed_planck_cosmologies(n: int) -> list[Cosmology]:
    """Return ``n`` deterministically perturbed Planck-2018 flat-LCDM cosmologies.

    A batch of near-identical cosmologies for exercising the multi-cosmology
    solve: for ``n == 1`` the unperturbed base is returned, otherwise the six
    parameters that shape the linear matter power spectrum are swept along a
    single ``phase in [-1, 1]`` line by a few tenths of a percent. Mirrors the
    perturbation pattern of the DISCO2 ``perturbed_planck_cosmologies`` helper.
    """

    base = PLANCK_2018_FLAT_LCDM
    if n == 1:
        return [base]
    phase = np.linspace(-1.0, 1.0, n, dtype=np.float64)
    return [
        dataclasses.replace(
            base,
            omega_b_h2=base.omega_b_h2 * (1.0 + 0.006 * ph),
            omega_c_h2=base.omega_c_h2 * (1.0 - 0.008 * ph),
            h=base.h * (1.0 + 0.004 * np.sin(np.pi * ph)),
            Y_He=base.Y_He * (1.0 + 0.003 * np.cos(0.5 * np.pi * ph)),
            T_cmb=base.T_cmb * (1.0 + 0.0015 * ph),
            N_eff=base.N_eff * (1.0 + 0.002 * np.sin(2.0 * np.pi * ph)),
        )
        for ph in phase
    ]


def cosmology_to_class_params(cosmology: Cosmology, *, output: str = "tCl") -> dict:
    """Return CLASS parameters for a benchmark :class:`discoeb.cosmology.Cosmology`.

    The DISCO-EB dataclass stores baryon and cold-dark-matter densities as
    ``Omega_i h^2`` already, while CLASS expects those same physical densities
    under ``omega_b`` and ``omega_cdm``. DISCO-EB stores massive neutrinos as a
    degenerate-mass approximation, so a nonzero massive-neutrino count is encoded
    as one CLASS ``ncdm`` species with degeneracy ``deg_ncdm``.
    """

    class_params = {
        "h": cosmology.h,
        "omega_b": cosmology.omega_b_h2,
        "omega_cdm": cosmology.omega_c_h2,
        "A_s": cosmology.A_s,
        "n_s": cosmology.n_s,
        "k_pivot": cosmology.k_pivot,
        "T_cmb": cosmology.T_cmb,
        "YHe": cosmology.Y_He,
        "N_ur": cosmology.Neff_massless,
        "N_ncdm": 0,
        "Omega_k": cosmology.Omegak,
        "output": output,
        "background_verbose": 0,
        "thermodynamics_verbose": 0,
    }

    if cosmology.num_massive_neutrinos > 0.0:
        class_params |= {
            "N_ncdm": 1,
            "m_ncdm": cosmology.mnu,
            "deg_ncdm": cosmology.num_massive_neutrinos,
            # DISCO-EB gives every neutrino the instantaneous-decoupling
            # temperature T_nu = (4/11)^(1/3) T_cmb, absorbing the non-instantaneous
            # correction into N_eff = 3.046. CLASS instead defaults its ncdm species
            # to T_ncdm = 0.71611, which carries that correction in the temperature
            # and inflates rho_ncdm by (0.71611/0.713766)^4 = 1.0132. Pin CLASS to
            # our convention or the massive-neutrino density differs by 1.3%.
            "T_ncdm": NEUTRINO_TEMPERATURE_FACTOR,
        }

    if cosmology.w_DE_0 != -1.0 or cosmology.w_DE_a != 0.0:
        class_params |= {
            "Omega_Lambda": 0,
            "w0_fld": cosmology.w_DE_0,
            "wa_fld": cosmology.w_DE_a,
        }

    return class_params
