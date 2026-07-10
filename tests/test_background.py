import numpy as np
import pytest

from discoeb.background import (
    critical_density_grho,
    dark_energy_density_ratio,
    dtau_da,
    dtau_dz,
    get_neutrino_momentum_bins,
    grhoa4,
    grhob,
    grhoc,
    grhog,
    grhornomass,
    grhov,
    helium_number_fraction,
    hubble_a,
    hubble_z,
    massive_neutrino_density,
    neutrino_density_grho,
    neutrino_mass_parameter,
    radiation_neutrino_factor,
    thomson_normalization,
)
from discoeb.constants import GRHO_CRITICAL_H2

from cosmologies import BENCHMARK_COSMOLOGIES, cosmology_to_class_params


# =============================================================================
# Scalar background-primitive tests (ported from DISCO2 test_background.py).
#
# The DISCO2 originals validated the flat-LambdaCDM scalar helpers against CAMB.
# Here we validate the same helpers against CLASS via its Python wrapper
# ``classy``. CLASS exposes background densities as ``(.)rho_i`` in Mpc^-2
# (i.e. ``8*pi*G*rho_i/3``), so the CAMB ``grho`` coefficients map onto CLASS as
#     grhog       = 3 * rho_g    * a^4      (constant, radiation)
#     grhornomass = 3 * rho_ur   * a^4      (constant, massless neutrinos)
#     grhoc       = 3 * rho_cdm  * a^3      (matter, rho ~ a^-3)
#     grhob       = 3 * rho_b    * a^3      (matter, rho ~ a^-3)
#     grhov       = 3 * rho_lambda          (cosmological constant), or
#                   3 * rho_fld / rho_de(a) (CPL dark-energy fluid)
#     grhomnu * rhonu(a) = 3 * rho_ncdm[0] * a^4  (massive neutrinos)
#     grhok       = 3 * (rho_crit - rho_tot) * a^2  (curvature)
#     grhoa4      = 3 * rho_crit * a^4
# and the physical Hubble rate ``H(a)`` in Mpc^-1 equals CLASS ``Hubble(z)``.
#
# Note that ``rho_crit``, not ``rho_tot``, is the reference for ``grhoa4``:
# CLASS excludes curvature from ``rho_tot`` and defines ``rho_crit = 3 H^2 / (8 pi G)``,
# which is exactly what ``grhoa4 / a^4`` equals in every model.
# =============================================================================


COSMOLOGY_CASES = [
    pytest.param(BENCHMARK_COSMOLOGIES[name], id=name)
    for name in (
        "planck_2018_flat_lcdm",
        "planck_2018_curved_lcdm",
        "desi_2024_dynamical_dark_energy",
        "planck_2018_flat_lcdm_massive_nu",
    )
]

NQMAX = 15
"""Momentum bins for the massive-neutrino background quadrature in these tests."""


def massive_neutrino_args(cosmology):
    """Return ``(grhomnu, amnu)`` for the degenerate massive-neutrino species.

    ``grhomnu`` is the coefficient the species would have if it stayed
    relativistic; its actual density is ``grhomnu * rhonu(a)``. Both are zero
    when the cosmology has no massive neutrinos.
    """

    n_mnu = cosmology.num_massive_neutrinos
    if n_mnu <= 0.0:
        return 0.0, 0.0
    grhomnu = neutrino_density_grho(cosmology.T_cmb) * n_mnu
    return grhomnu, neutrino_mass_parameter(cosmology.mnu, cosmology.T_cmb)


def rhonu_of_a(a, amnu):
    """Return ``rho_nu(a) / rho_nu,massless``, which tends to 1 while relativistic."""

    if amnu == 0.0:
        return 1.0
    q, w = get_neutrino_momentum_bins(NQMAX)
    return massive_neutrino_density(float(a), amnu, np.asarray(q), np.asarray(w))[0]


def density_args(cosmology):
    """Return the general ``grho`` coefficients ``(g, rnomass, c, b, v)``.

    The dark-energy coefficient is fixed by closing the density budget at
    ``a = 1`` against the critical density, which reproduces the flat-LambdaCDM
    :func:`grhov` and additionally accounts for spatial curvature and the
    massive-neutrino density.
    """

    grhog_value = grhog(cosmology.T_cmb)
    grhornomass_value = grhornomass(grhog_value, cosmology.Neff_massless)
    grhoc_value = grhoc(cosmology.omega_c_h2)
    grhob_value = grhob(cosmology.omega_b_h2)

    grhom = critical_density_grho(cosmology.H0)
    grhomnu, amnu = massive_neutrino_args(cosmology)
    grhov_value = grhom - (
        grhog_value
        + grhornomass_value
        + grhomnu * rhonu_of_a(1.0, amnu)
        + grhoc_value
        + grhob_value
        + grhom * cosmology.Omegak
    )
    return (grhog_value, grhornomass_value, grhoc_value, grhob_value, grhov_value)


def grho_kwargs(cosmology, a):
    """Return the scale-factor-dependent keyword arguments of the general model."""

    grhomnu, amnu = massive_neutrino_args(cosmology)
    return {
        "grhok": critical_density_grho(cosmology.H0) * cosmology.Omegak,
        "grhomnu": grhomnu,
        "rhonu": rhonu_of_a(a, amnu),
        "rho_de": dark_energy_density_ratio(
            float(a), cosmology.w_DE_0, cosmology.w_DE_a
        ),
    }


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

    def has(self, key):
        """Return whether CLASS produced a given background column."""
        return key in self._bg

    def rho(self, key, a):
        """Log-log interpolate a CLASS ``(.)rho_i`` column [Mpc^-2] at ``a``."""
        return np.exp(np.interp(np.log(a), self._log_a, np.log(self._bg[key])))

    def grho_total(self, a):
        """CLASS ``3 H^2 a^4``, the quantity :func:`grhoa4` computes.

        Uses ``rho_crit = 3 H^2 / (8 pi G)`` rather than ``rho_tot``, because
        CLASS omits the curvature term from ``rho_tot``.
        """
        return 3.0 * self.rho("(.)rho_crit", a) * a**4

    def grhok(self, a):
        """CLASS curvature coefficient ``3 (rho_crit - rho_tot) a^2``.

        CLASS writes the Friedmann equation as ``H^2 = rho_tot - K/a^2`` (in
        ``8 pi G / 3 = 1`` units) with ``rho_crit = H^2``, so the curvature
        contribution is the difference of the two.
        """
        rho_k = self.rho("(.)rho_crit", a) - self.rho("(.)rho_tot", a)
        return 3.0 * rho_k * a**2

    def dark_energy_key(self):
        """Return the CLASS column holding the dark-energy density."""
        return "(.)rho_fld" if self.has("(.)rho_fld") else "(.)rho_lambda"

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
    cosmo.set(cosmology_to_class_params(cosmology, output="tCl"))
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
    kwargs = grho_kwargs(cosmology, a)

    assert grhoc(cosmology.omega_c_h2) == pytest.approx(
        GRHO_CRITICAL_H2 * cosmology.omega_c_h2, rel=1.0e-15
    )
    assert grhob(cosmology.omega_b_h2) == pytest.approx(
        GRHO_CRITICAL_H2 * cosmology.omega_b_h2, rel=1.0e-15
    )
    assert grhornomass(grhog_value, cosmology.Neff_massless) == pytest.approx(
        grhog_value * radiation_neutrino_factor(cosmology.Neff_massless), rel=1.0e-15
    )
    assert hubble_z(z, *args, **kwargs) == pytest.approx(
        hubble_a(a, *args, **kwargs), rel=1.0e-15
    )
    assert dtau_dz(z, *args, **kwargs) == pytest.approx(
        -dtau_da(a, *args, **kwargs) / (1.0 + z) ** 2,
        rel=1.0e-15,
    )


@pytest.mark.parametrize("cosmology", COSMOLOGY_CASES)
def test_density_budget_closes_at_today(cosmology):
    """``grhoa4(1)`` reproduces the critical density ``3 H_0^2`` by construction."""

    args = density_args(cosmology)
    total = grhoa4(1.0, *args, **grho_kwargs(cosmology, 1.0))

    assert total == pytest.approx(critical_density_grho(cosmology.H0), rel=1.0e-12)


def test_flat_lambda_cdm_grhov_matches_general_budget_closure():
    """For flat massless LambdaCDM the general closure reduces to :func:`grhov`."""

    cosmology = BENCHMARK_COSMOLOGIES["planck_2018_flat_lcdm"]

    assert density_args(cosmology)[4] == pytest.approx(
        grhov(
            cosmology.omega_b_h2,
            cosmology.omega_c_h2,
            cosmology.h,
            cosmology.Neff_massless,
            cosmology.T_cmb,
        ),
        rel=1.0e-12,
    )


def test_class_background_coefficients_match_scalar_functions(class_background):
    """Compare scalar density and opacity normalizations against CLASS."""

    cosmology, bg = class_background
    grhog_value = grhog(cosmology.T_cmb)
    a_values = np.array([1.0e-6, 1.0e-4, 1.0e-2, 1.0])

    assert grhog_value == pytest.approx(
        np.mean(3.0 * bg.rho("(.)rho_g", a_values) * a_values**4), rel=1.0e-4
    )
    assert grhornomass(grhog_value, cosmology.Neff_massless) == pytest.approx(
        np.mean(3.0 * bg.rho("(.)rho_ur", a_values) * a_values**4), rel=1.0e-4
    )
    assert grhoc(cosmology.omega_c_h2) == pytest.approx(
        np.mean(3.0 * bg.rho("(.)rho_cdm", a_values) * a_values**3), rel=1.0e-4
    )
    assert grhob(cosmology.omega_b_h2) == pytest.approx(
        np.mean(3.0 * bg.rho("(.)rho_b", a_values) * a_values**3), rel=1.0e-4
    )
    # Dark energy: CLASS stores rho_lambda (constant) or rho_fld (CPL fluid). Our
    # single coefficient grhov is the a = 1 value, scaled by dark_energy_density_ratio.
    grhov_value = density_args(cosmology)[4]
    de_ours = np.array(
        [
            grhov_value * dark_energy_density_ratio(float(a), cosmology.w_DE_0, cosmology.w_DE_a)
            for a in a_values
        ]
    )
    assert de_ours == pytest.approx(
        3.0 * bg.rho(bg.dark_energy_key(), a_values), rel=1.0e-4
    )

    # Massive neutrinos: relativistic at early times, rho ~ a^-3 once a*amnu >> q.
    grhomnu, amnu = massive_neutrino_args(cosmology)
    if grhomnu > 0.0:
        mnu_ours = np.array([grhomnu * rhonu_of_a(a, amnu) for a in a_values])
        assert mnu_ours == pytest.approx(
            3.0 * bg.rho("(.)rho_ncdm[0]", a_values) * a_values**4, rel=1.0e-4
        )

    # Curvature enters grhoa4 as grhok * a^2 and is absent from CLASS's rho_tot.
    if cosmology.Omegak != 0.0:
        grhok_ours = critical_density_grho(cosmology.H0) * cosmology.Omegak
        assert np.array([bg.grhok(float(a)) for a in a_values]) == pytest.approx(
            np.full_like(a_values, grhok_ours), rel=1.0e-3
        )

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
    ours = np.array(
        [grhoa4(float(a), *args, **grho_kwargs(cosmology, a)) for a in a_values]
    )
    class_total = np.array([bg.grho_total(float(a)) for a in a_values])

    assert ours == pytest.approx(class_total, rel=1.0e-4)


def test_class_hubble_z_matches_scalar_function(class_background):
    """Compare the scalar redshift Hubble function against CLASS ``H(z)``."""

    cosmology, bg = class_background
    args = density_args(cosmology)
    z_values = np.array([0.0, 1.0, 10.0, 100.0, 1100.0])
    ours = np.array(
        [
            hubble_z(float(z), *args, **grho_kwargs(cosmology, 1.0 / (1.0 + z)))
            for z in z_values
        ]
    )
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
    ours = np.array(
        [dtau_da(float(a), *args, **grho_kwargs(cosmology, a)) for a in a_values]
    )

    assert ours == pytest.approx(class_dtau_da, rel=1.0e-4)
