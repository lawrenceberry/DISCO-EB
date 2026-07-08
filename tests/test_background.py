from dataclasses import dataclass

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from discoeb.background import (
    dtau_da,
    dtau_dz,
    evolve_background,
    grhoa4,
    grhob,
    grhoc,
    grhog,
    grhornomass,
    grhov,
    helium_number_fraction,
    hubble_a,
    hubble_z,
    radiation_neutrino_factor,
    thomson_normalization,
)
from discoeb.constants import GRHO_CRITICAL_H2

from conftest import a_RECFAST, xe_RECFAST

## Cosmological Parameters
Tcmb    = 2.7255
YHe     = 0.248
Omegam  = 0.3099
Omegab  = 0.0488911
w_DE_0  = -0.99
w_DE_a  = 0.0
cs2_DE  = 1.0

# Initialize neutrinos.
mnu     = 0.06  #eV
Tnu     = (4/11)**(1/3) #0.71611 # Tncdm of CLASS
Neff    = 3.046 # -1 if massive neutrino present
N_nu_mass = 1
N_nu_rel = Neff - N_nu_mass * (Tnu/((4/11)**(1/3)))**4
h       = 0.67742
A_s     = 2.1064e-09
n_s     = 0.96822

# Initialize neutrinos.
num_massive_neutrinos = 1
mnu     = 0.06  #eV
Tnu     = (4/11)**(1/3) #0.71611 # Tncdm of CLASS
Neff    = 3.046 # -1 if massive neutrino present
N_nu_mass = 1
N_nu_rel = Neff - N_nu_mass * (Tnu/((4/11)**(1/3)))**4
h       = 0.67742
A_s     = 2.1064e-09
n_s     = 0.96822
k_p     = 0.05



## Compute Background evolution
param = {}
param['Omegam']  = Omegam
param['Omegab']  = Omegab
param['w_DE_0']  = w_DE_0
param['w_DE_a']  = w_DE_a
param['cs2_DE']  = cs2_DE
param['Omegak']  = 0.0
param['A_s']     = A_s
param['n_s']     = n_s
param['H0']      = 100*h
param['Tcmb']    = Tcmb
param['YHe']     = YHe
param['Neff']    = N_nu_rel
param['Nmnu']    = N_nu_mass
param['mnu']     = mnu

background_recfast_jit = jax.jit(lambda param: evolve_background(param=param, thermo_module='RECFAST'))

@pytest.fixture(scope="session", autouse=True)
def jit_functions():
    _ = background_recfast_jit(param=param)

class TestEvolveBackground:
    
    def test_evolve_background_recfast_vs_DISCOEB_v0_1_0_baseline(self, benchmark):
        solution_param = benchmark(background_recfast_jit, param=param)

        xe = solution_param['xe_of_tau_spline'].evaluate(solution_param['tau_of_a_spline'].evaluate(a_RECFAST))
        # The thermo splines carry a trailing batch (cosmology) axis of size 1;
        # squeeze it to compare against the 1-D stored baseline.
        xe = jnp.squeeze(xe, axis=-1)

        assert xe.shape == xe_RECFAST.shape
        assert jnp.allclose(xe, xe_RECFAST, rtol=0.01), "relative error exceeded 0.01"
        assert jnp.allclose(xe, xe_RECFAST, rtol=0.005), "relative error exceeded 0.005"


# =============================================================================
# Scalar background-primitive tests (ported from DISCO2 test_background.py).
#
# The DISCO2 originals validated the flat-LambdaCDM scalar helpers against CAMB.
# Here we validate the same helpers against CLASS via its Python wrapper
# ``classy``. CLASS exposes background densities as ``(.)rho_i`` in Mpc^-2
# (i.e. ``8*pi*G*rho_i/3``), so the CAMB ``grho`` coefficients map onto CLASS as
#     grhog       = 3 * rho_g   * a^4      (constant, radiation)
#     grhornomass = 3 * rho_ur  * a^4      (constant, massless neutrinos)
#     grhoc       = 3 * rho_cdm * a^3      (matter, rho ~ a^-3)
#     grhob       = 3 * rho_b   * a^3      (matter, rho ~ a^-3)
#     grhov       = 3 * rho_lambda         (constant, cosmological constant)
#     grhoa4      = 3 * rho_tot * a^4
# and the physical Hubble rate ``H(a)`` in Mpc^-1 equals CLASS ``Hubble(z)``.
# =============================================================================


@dataclass(frozen=True)
class Cosmology:
    """Flat-LambdaCDM cosmology described by physical densities ``Omega_i h^2``."""

    omega_b_h2: float
    omega_c_h2: float
    h: float
    A_s: float
    n_s: float
    T_cmb: float
    Y_He: float
    N_eff: float
    k_pivot: float


PLANCK_2018_FLAT_LCDM = Cosmology(
    omega_b_h2=0.02238280,
    omega_c_h2=0.1201075,
    h=0.6732117,
    A_s=2.100549e-9,
    n_s=0.9660499,
    T_cmb=2.7255,
    Y_He=0.2454006,
    N_eff=3.046,
    k_pivot=0.05,
)

STANDARD_COSMOLOGIES = {"planck_2018_flat_lcdm": PLANCK_2018_FLAT_LCDM}

COSMOLOGY_CASES = [
    pytest.param(cosmology, id=name) for name, cosmology in STANDARD_COSMOLOGIES.items()
]


def density_args(cosmology: Cosmology):
    """Return the flat-LambdaCDM ``grho`` coefficients for a cosmology."""

    grhog_value = grhog(cosmology.T_cmb)
    return (
        grhog_value,
        grhornomass(grhog_value, cosmology.N_eff),
        grhoc(cosmology.omega_c_h2),
        grhob(cosmology.omega_b_h2),
        grhov(
            cosmology.omega_b_h2,
            cosmology.omega_c_h2,
            cosmology.h,
            cosmology.N_eff,
            cosmology.T_cmb,
        ),
    )


class ClassBackground:
    """Thin accessor for a computed CLASS background + thermodynamics table.

    Provides power-law (log-log) interpolation of the CLASS background densities
    so that the DISCO-EB ``grho`` coefficients can be reconstructed at arbitrary
    scale factors, plus helpers derived from the thermodynamics table.
    """

    def __init__(self, cosmo):
        self._cosmo = cosmo
        bg = cosmo.get_background()
        a = 1.0 / (1.0 + np.asarray(bg["z"]))
        order = np.argsort(a)
        self._log_a = np.log(a[order])
        self._bg = {key: np.asarray(value)[order] for key, value in bg.items()}

        th = cosmo.get_thermodynamics()
        self._xe = np.asarray(th["x_e"])
        self._a_th = np.asarray(th["scale factor a"])
        self._kappa_prime = np.asarray(th["kappa' [Mpc^-1]"])

    def rho(self, key, a):
        """Log-log interpolate a CLASS ``(.)rho_i`` column [Mpc^-2] at ``a``."""
        return np.exp(np.interp(np.log(a), self._log_a, np.log(self._bg[key])))

    def grho_total(self, a):
        """CLASS ``8*pi*G*rho_total a^4`` matching :func:`grhoa4`."""
        return 3.0 * self.rho("(.)rho_tot", a) * a**4

    def hubble(self, z):
        """CLASS physical Hubble rate ``H(z)`` in Mpc^-1."""
        return self._cosmo.Hubble(z)

    def akthom(self):
        """Thomson normalization ``kappa' a^2 / x_e`` from the CLASS thermo table.

        CLASS stores the conformal opacity rate ``kappa' = x_e akthom / a^2``, so
        ``akthom = kappa' a^2 / x_e`` is (up to recombination physics) constant.
        Evaluated deep in the fully-ionized era to avoid recombination.
        """
        mask = self._a_th < 1.0e-3
        return np.mean(self._kappa_prime[mask] * self._a_th[mask] ** 2 / self._xe[mask])

    def helium_number_fraction(self):
        """Helium/hydrogen number ratio from the fully-ionized ``x_e`` plateau.

        With hydrogen and helium fully ionized (He -> He++), ``x_e -> 1 + 2 f_He``,
        so ``f_He = (max(x_e) - 1) / 2``.
        """
        return (np.max(self._xe) - 1.0) / 2.0


@pytest.fixture(scope="module", params=COSMOLOGY_CASES)
def class_background(request):
    cosmology = request.param
    Class = pytest.importorskip("classy").Class

    cosmo = Class()
    cosmo.set(
        {
            "h": cosmology.h,
            "omega_b": cosmology.omega_b_h2,
            "omega_cdm": cosmology.omega_c_h2,
            "T_cmb": cosmology.T_cmb,
            "YHe": cosmology.Y_He,
            "N_ur": cosmology.N_eff,
            "N_ncdm": 0,
            "Omega_k": 0.0,
            "output": "tCl",
            "background_verbose": 0,
            "thermodynamics_verbose": 0,
        }
    )
    cosmo.compute()
    try:
        yield cosmology, ClassBackground(cosmo)
    finally:
        cosmo.struct_cleanup()
        cosmo.empty()


def test_radiation_only_scaling():
    """Check the Friedmann helpers in a pure-radiation toy universe."""

    g_rad = 2.0e-8
    args = (g_rad, 0.0, 0.0, 0.0, 0.0)
    a_values = np.array([1.0e-6, 1.0e-4, 1.0e-2, 1.0])
    dtau_values = np.array([dtau_da(float(a), *args) for a in a_values])
    ha2_values = np.array([hubble_a(float(a), *args) * a**2 for a in a_values])

    assert np.array([grhoa4(float(a), *args) for a in a_values]) == pytest.approx(
        np.full_like(a_values, g_rad)
    )
    assert dtau_values == pytest.approx(np.full_like(a_values, dtau_values[0]))
    assert ha2_values == pytest.approx(np.full_like(a_values, ha2_values[0]))


def test_matter_only_scaling():
    """Check the Friedmann helpers in a pure-matter toy universe."""

    g_matter = 3.0e-7
    args = (0.0, 0.0, g_matter, 0.0, 0.0)
    a_values = np.array([1.0e-6, 1.0e-4, 1.0e-2, 1.0])
    dtau_scaled = np.array([dtau_da(float(a), *args) * np.sqrt(a) for a in a_values])
    h_scaled = np.array([hubble_a(float(a), *args) * a**1.5 for a in a_values])

    assert np.array([grhoa4(float(a), *args) for a in a_values]) == pytest.approx(
        g_matter * a_values
    )
    assert dtau_scaled == pytest.approx(np.full_like(a_values, dtau_scaled[0]))
    assert h_scaled == pytest.approx(np.full_like(a_values, h_scaled[0]))


def test_lambda_only_hubble_is_constant():
    """Check that a pure-Lambda universe has a scale-factor-independent Hubble rate."""

    args = (0.0, 0.0, 0.0, 0.0, 4.0e-8)
    a_values = np.array([1.0e-4, 1.0e-2, 0.1, 1.0])
    h_values = np.array([hubble_a(float(a), *args) for a in a_values])

    assert h_values == pytest.approx(np.full_like(a_values, h_values[0]))


@pytest.mark.parametrize("cosmology", COSMOLOGY_CASES)
def test_algebraic_background_identities(cosmology):
    """Check exact scalar identities that should hold for every named cosmology."""

    grhog_value = grhog(cosmology.T_cmb)
    args = density_args(cosmology)
    z = 11.0
    a = 1.0 / (1.0 + z)

    assert grhoc(cosmology.omega_c_h2) == pytest.approx(
        GRHO_CRITICAL_H2 * cosmology.omega_c_h2, rel=1.0e-15
    )
    assert grhob(cosmology.omega_b_h2) == pytest.approx(
        GRHO_CRITICAL_H2 * cosmology.omega_b_h2, rel=1.0e-15
    )
    assert grhornomass(grhog_value, cosmology.N_eff) == pytest.approx(
        grhog_value * radiation_neutrino_factor(cosmology.N_eff), rel=1.0e-15
    )
    assert hubble_z(z, *args) == pytest.approx(hubble_a(a, *args), rel=1.0e-15)
    assert dtau_dz(z, *args) == pytest.approx(
        -dtau_da(a, *args) / (1.0 + z) ** 2,
        rel=1.0e-15,
    )


def test_class_background_coefficients_match_scalar_functions(class_background):
    """Compare scalar density and opacity normalizations against CLASS."""

    cosmology, bg = class_background
    grhog_value = grhog(cosmology.T_cmb)
    a_values = np.array([1.0e-6, 1.0e-4, 1.0e-2, 1.0])

    assert grhog_value == pytest.approx(
        np.mean(3.0 * bg.rho("(.)rho_g", a_values) * a_values**4), rel=1.0e-4
    )
    assert grhornomass(grhog_value, cosmology.N_eff) == pytest.approx(
        np.mean(3.0 * bg.rho("(.)rho_ur", a_values) * a_values**4), rel=1.0e-4
    )
    assert grhoc(cosmology.omega_c_h2) == pytest.approx(
        np.mean(3.0 * bg.rho("(.)rho_cdm", a_values) * a_values**3), rel=1.0e-4
    )
    assert grhob(cosmology.omega_b_h2) == pytest.approx(
        np.mean(3.0 * bg.rho("(.)rho_b", a_values) * a_values**3), rel=1.0e-4
    )
    assert grhov(
        cosmology.omega_b_h2,
        cosmology.omega_c_h2,
        cosmology.h,
        cosmology.N_eff,
        cosmology.T_cmb,
    ) == pytest.approx(np.mean(3.0 * bg.rho("(.)rho_lambda", a_values)), rel=1.0e-4)
    assert thomson_normalization(cosmology.omega_b_h2, cosmology.Y_He) == pytest.approx(
        bg.akthom(), rel=1.0e-3
    )
    assert helium_number_fraction(cosmology.Y_He) == pytest.approx(
        bg.helium_number_fraction(), rel=5.0e-3
    )


def test_class_background_density_sum_matches_grhoa4(class_background):
    """Compare the scalar ``grhoa4`` polynomial against the CLASS density total."""

    cosmology, bg = class_background
    args = density_args(cosmology)
    a_values = np.array([1.0e-6, 1.0e-4, 1.0e-2, 1.0])
    ours = np.array([grhoa4(float(a), *args) for a in a_values])
    class_total = np.array([bg.grho_total(float(a)) for a in a_values])

    assert ours == pytest.approx(class_total, rel=1.0e-4)


def test_class_hubble_z_matches_scalar_function(class_background):
    """Compare the scalar redshift Hubble function against CLASS ``H(z)``."""

    cosmology, bg = class_background
    args = density_args(cosmology)
    z_values = np.array([0.0, 1.0, 10.0, 100.0, 1100.0])
    ours = np.array([hubble_z(float(z), *args) for z in z_values])
    class_hubble = np.array([bg.hubble(float(z)) for z in z_values])

    assert ours == pytest.approx(class_hubble, rel=1.0e-5)


def test_dtau_da_matches_hubble_identity_with_class_hubble(class_background):
    """Check ``dtau/da`` against the equivalent expression built from CLASS ``H(z)``."""

    cosmology, bg = class_background
    args = density_args(cosmology)
    a_values = np.array([1.0e-6, 1.0e-4, 1.0e-2, 1.0])
    z_values = 1.0 / a_values - 1.0
    class_dtau_da = np.array(
        [1.0 / (a**2 * bg.hubble(float(z))) for a, z in zip(a_values, z_values)]
    )
    ours = np.array([dtau_da(float(a), *args) for a in a_values])

    assert ours == pytest.approx(class_dtau_da, rel=1.0e-4)

