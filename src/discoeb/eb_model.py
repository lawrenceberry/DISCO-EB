"""The linear Einstein-Boltzmann system in synchronous gauge, as components.

Each species of :mod:`discoeb.component` owns its perturbation variables and
states its own equations; the Einstein constraints, the Friedmann equation and
the Thomson coupling are components of their own. What a function needs it
names in its signature -- ``d_vb(adotoa, vb, k, cs2, clxb)`` -- and
:class:`~discoeb.component.Model` injects it, so a species never sees another
species' internals except through the quantities they publish.

Conventions (CAMB's): ``k`` in 1/Mpc, conformal time ``tau`` in Mpc, ``etak =
k eta``, velocities are the ``q_i = (1 + w_i) theta_i / k`` of CAMB, and the
density coefficients are ``grho_i_t = 8 pi G rho_i a^2``. The metric enters the
fluid equations through ``0.5 h' = k z`` and the shear ``sigma``; see
:class:`Metric`. The curved-geometry relations follow CLASS's ``perturbations.c``
with ``K = -grhok / 3`` and ``s2^2 = 1 - 3K/k^2``.

The Boltzmann hierarchies are vector states -- ``theta`` holds ``Theta_0 ..
Theta_lmax``, ``e`` holds ``E_2 .. E_lmax``, ``n`` the neutrino multipoles and
``psi`` one row per massive-neutrino momentum bin -- and their equations are
loops over multipoles, with the free-streaming tail written once in
:func:`stream`. Their truncations are the components' settings: ``LMAX_G``
and ``LMAX_POL`` on :class:`Photons`, ``LMAX_NR`` on
:class:`MasslessNeutrinos`, ``LMAX_NU`` and ``NQMAX`` on
:class:`MassiveNeutrinos`, each overridable by ``DISCO_<NAME>`` in the
environment or by a constructor argument, and read by the equations as
``self.NQMAX`` like any attribute. Each state also carries its
adiabatic initial condition (CAMB's ``initial()`` series at ``k tau << 1``,
with the growing-mode quantities of :class:`AdiabaticMode`) and its place in
the Jacobian's sparsity pattern (``tail_from``), so the layout, the initial
state and the sparse solver are all read off the components. The photon
monopole, dipole and quadrupole are ``theta[0]``, ``theta[1]``, ``theta[2]``;
the scalar names ``clxg``, ``qg``, ``pig`` and ``E2`` are published for the
equations that read them.

The background lives here too, on the same species: the Friedmann sum ``grho``
the perturbation kernel reads is, evaluated by :meth:`Model.evaluate` on JAX
arrays at ``a``, the expansion rate the conformal-time quadrature and the
thermal history integrate (:func:`expansion_rate`, :func:`dtauda`,
:func:`setup_background_evolution`, :func:`evolve_background`), and
:func:`density_parameters` closes the density budget at ``a = 1`` with that
same sum. The density constants and the massive-neutrino quadrature are defined
once, below.
"""

from __future__ import annotations

import math
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from .component import Component, Interaction, Model, Setting, Species, State
from .spline_interpolation import FastUniformCubicSpline1D, spline_interpolation
from .util import generalized_gauss_laguerre_weights, integrate_trapz

SUMS = ("grho", "dgrho", "dgq", "dgpi")
"""The quantities summed over species: the Friedmann density, the sources of
the Einstein constraints ``dgrho = Sum grho_i_t delta_i`` and
``dgq = Sum grho_i_t q_i``, and the anisotropic stress ``dgpi = Sum grho_i_t pi_i``
the Weyl potential of the line-of-sight sources needs."""


def _needs_octupole(**lmax):
    """The quadrupole equations couple to ``l = 3``, so a hierarchy truncates at 3 or above."""

    low = {name: l for name, l in lmax.items() if l < 3}
    if low:
        raise ValueError(f"a hierarchy must reach at least the octupole; got {low}")


# --- scales, the Friedmann equation and the metric ----------------------------------


class Background(Component):
    """Powers of the scale factor, the Friedmann equation, ``1/tau`` and ``1/k^2``."""

    def a2(self, a):
        return a * a

    def inv_a(self, a):
        return 1.0 / a

    def inv_a2(self, inv_a):
        return inv_a * inv_a

    def inv_tau(self, tau):
        return 1.0 / tau

    def inv_k2(self, k):
        return 1.0 / (k * k)

    def adotoa(self, grho):
        """The conformal Hubble rate ``a'/a = sqrt(grho / 3)``."""
        return math.sqrt(grho * (1.0 / 3.0))


class AdiabaticMode(Component):
    """The adiabatic growing mode deep in radiation domination, for the initial conditions.

    CAMB's ``initial()`` series at ``x = k tau << 1``, normalised to unit
    comoving curvature: the neutrino fraction of the radiation ``R_nu``, the
    matter-radiation parameter ``om``, and their combinations. These are
    evaluated on the host from the background densities when the initial
    state is assembled; the states' ``initial`` functions read them by name.
    """

    def x(self, k, tau):
        return k * tau

    def Rv(self, grhog, grhornomass):
        return grhornomass / (grhog + grhornomass)

    def Rp15(self, Rv):
        return 4.0 * Rv + 15.0

    def om(self, grhob, grhoc, grhog, grhornomass):
        return (grhob + grhoc) / math.sqrt(3.0 * (grhog + grhornomass))

    def omtau(self, om, tau):
        return om * tau


class Curvature(Species):
    """Spatial curvature: a density ``grhok = 8 pi G rho_K a^2`` and the metric factors."""

    def grho(self, grhok):
        return grhok

    def s2(self, grhok, inv_k2):
        """``s2^2 = 1 - 3K/k^2 = 1 + grhok/k^2`` (1 when flat)."""
        return 1.0 + grhok * inv_k2

    def inv_s2(self, s2):
        return 1.0 / s2

    def Kcurv(self, grhok):
        return -grhok * (1.0 / 3.0)


class Metric(Component):
    """The synchronous-gauge metric: ``etak = k eta`` and the Einstein constraints.

    ``h' = (k^2 s2^2 eta + 1.5 a^2 delta_rho) / (0.5 aH)`` with ``z = h'/(2k)``,
    ``eta' = (1.5 a^2 (rho+p) theta + 0.5 K h') / (k^2 s2^2)``, and the shear
    ``sigma = k alpha = (z + 1.5 dgq/k^2) / s2^2``.
    """

    etak = State(initial=lambda k, x, Rp15: -k * (1.0 - x * x / 12.0 * (-10.0 / Rp15 + 1.0)))

    def z(self, dgrho, k, s2, etak, adotoa):
        return (0.5 * dgrho / k + s2 * etak) / adotoa

    def sigma(self, z, dgq, inv_k2, inv_s2):
        return (z + 1.5 * dgq * inv_k2) * inv_s2

    def d_etak(self, dgq, Kcurv, z, inv_s2):
        return (0.5 * dgq + Kcurv * z) * inv_s2


# --- matter -------------------------------------------------------------------------
# The density parameters are read through the trajectory's sensitivity
# direction, ``grho_i = grho_i_nominal + eps * dir_grho_i``: ``eps`` is zero in
# every primal solve and d/d(eps) is the derivative along ``dir``.


class ColdDarkMatter(Species):
    clxc = State(initial=lambda clxg: 0.75 * clxg)  # the adiabatic mode: 3/4 of the photon density

    def grhoc(self, grhoc_nominal, eps, dir_grhoc):
        return grhoc_nominal + eps * dir_grhoc

    def grhoc_t(self, grhoc, inv_a):
        return grhoc * inv_a

    def grho(self, grhoc_t):
        return grhoc_t

    def dgrho(self, grhoc_t, clxc):
        return grhoc_t * clxc

    def d_clxc(self, z, k):
        """CDM is pressureless and at rest in synchronous gauge: ``clxc' = -k z``."""
        return -k * z


class Baryons(Species):
    """Baryon continuity and Euler equations; the Thomson drag is :class:`ThomsonScattering`."""

    clxb = State(initial=lambda clxg: 0.75 * clxg)
    vb = State(initial=lambda qg: 0.75 * qg)

    def grhob(self, grhob_nominal, eps, dir_grhob):
        return grhob_nominal + eps * dir_grhob

    def grhob_t(self, grhob, inv_a):
        return grhob * inv_a

    def grho(self, grhob_t):
        return grhob_t

    def dgrho(self, grhob_t, clxb):
        return grhob_t * clxb

    def dgq(self, grhob_t, vb):
        return grhob_t * vb

    def d_clxb(self, z, vb, k):
        return -k * (z + vb)

    def d_vb(self, adotoa, vb, k, cs2, clxb):
        return -adotoa * vb + k * cs2 * clxb


# --- radiation ----------------------------------------------------------------------


def stream(kv, x, inv_tau, output):
    """Free streaming of the multipoles ``3 .. L`` of a hierarchy ``x``, into ``output``.

    ``x_l' = kv (l x_{l-1} - (l+1) x_{l+1}) / (2l+1)`` and, closing the
    hierarchy at ``L = len(x) - 1``, ``x_L' = kv x_{L-1} - (L+1) x_L / tau``.
    ``kv`` is ``k`` for massless particles and ``k v`` for a massive-neutrino
    momentum bin. Collisional damping is not here: it is the Thomson
    interaction's contribution to the same derivatives.
    """

    lmax = len(x) - 1
    for l in range(3, lmax):
        output[l] = kv * (l * x[l - 1] - (l + 1) * x[l + 1]) * (1.0 / (2 * l + 1))
    output[lmax] = kv * x[lmax - 1] - (lmax + 1) * x[lmax] * inv_tau


class Photons(Species):
    """Photon temperature multipoles ``theta`` (``Theta_0 .. Theta_LMAX_G``) and E-modes ``e``.

    ``e[l - 2]`` is ``E_l`` for ``l = 2 .. LMAX_POL``. The collisionless part
    of the Boltzmann hierarchy; Thomson scattering, which damps the multipoles
    and sources the polarization, is :class:`ThomsonScattering`. The E-mode
    hierarchy couples upwards with ``(l+3)(l-1)/(l+1)`` in place of ``l+1``
    and closes with ``k L E_{L-1} / (2L+1) - (L+3) E_L / tau``. Both
    hierarchies stream freely above their lowest three (``theta``) or one
    (``e``) multipoles, which is their sparsity role; the E-modes start at
    zero.
    """

    LMAX_G = Setting(15)
    LMAX_POL = Setting(15)

    def _theta_initial(self, x, omtau, output):
        """The monopole ``delta_gamma`` and dipole ``q_gamma`` of the growing mode."""
        output[0] = x * x / 3.0 * (1.0 - omtau / 5.0)
        output[1] = x**3 / 27.0 * (1.0 - omtau / 5.0)

    theta = State(lambda self: self.LMAX_G + 1, tail_from=3, initial=_theta_initial)
    e = State(lambda self: self.LMAX_POL - 1, tail_from=1)

    def __init__(self, **settings):
        super().__init__(**settings)
        _needs_octupole(LMAX_G=self.LMAX_G, LMAX_POL=self.LMAX_POL)

    def grhog(self, grhog_nominal, eps, dir_grhog):
        return grhog_nominal + eps * dir_grhog

    def grhog_t(self, grhog, inv_a2):
        return grhog * inv_a2

    def grho(self, grhog_t):
        return grhog_t

    def clxg(self, theta):
        return theta[0]

    def qg(self, theta):
        return theta[1]

    def pig(self, theta):
        return theta[2]

    def E2(self, e):
        """The E-mode quadrupole, ``e[0]``."""
        return e[0]

    def dgrho(self, grhog_t, clxg):
        return grhog_t * clxg

    def dgq(self, grhog_t, qg):
        return grhog_t * qg

    def dgpi(self, grhog_t, pig):
        return grhog_t * pig

    def d_theta(self, k, z, sigma, theta, inv_tau, output):
        output[0] = -k * ((4.0 / 3.0) * z + theta[1])
        output[1] = k * theta[0] * (1.0 / 3.0) - (2.0 / 3.0) * k * theta[2]
        output[2] = (2.0 / 5.0) * k * theta[1] - (3.0 / 5.0) * k * theta[3] + (8.0 / 15.0) * k * sigma
        stream(k, theta, inv_tau, output)

    def d_e(self, k, e, inv_tau, output):
        lmax = len(e) + 1
        output[0] = -k * e[1] * (1.0 / 3.0)
        for l in range(3, lmax):
            upfac = (l + 3.0) * (l - 1.0) / (l + 1.0)
            output[l - 2] = k * (l * e[l - 3] - upfac * e[l - 1]) * (1.0 / (2 * l + 1))
        output[lmax - 2] = (
            k * lmax * e[lmax - 3] * (1.0 / (2 * lmax + 1)) - (lmax + 3) * e[lmax - 2] * inv_tau
        )


class ThomsonScattering(Interaction):
    """Thomson scattering between the photons and the baryons.

    The momentum exchange (drag on ``vb``, its reaction on ``qg``), the
    scattering source ``Pi = pig/10 + (9/15) E_2`` of the quadrupoles, and the
    damping ``-kappa' X_l`` of every higher temperature and polarization
    multipole. ``opacity`` is ``kappa' = a n_e sigma_T``. The buffers it fills
    start at zero, so the monopole it does not scatter is left alone.
    """

    def photbar(self, grhog_t, grhob_t):
        """The photon-to-baryon density ratio ``R^-1 = rho_gamma / rho_b``."""
        return grhog_t / grhob_t

    def slip(self, vb, qg):
        return (4.0 / 3.0) * vb - qg

    def polter(self, pig, E2):
        return pig * (1.0 / 10.0) + 9.0 * E2 * (1.0 / 15.0)

    def d_vb(self, photbar, opacity, slip):
        return -photbar * opacity * slip

    def d_theta(self, opacity, slip, polter, theta, output):
        output[1] = opacity * slip
        output[2] = -opacity * (theta[2] - polter)
        for l in range(3, len(theta)):
            output[l] = -opacity * theta[l]

    def d_e(self, opacity, polter, e, output):
        output[0] = -opacity * (e[0] - polter)
        for i in range(1, len(e)):
            output[i] = -opacity * e[i]


class MasslessNeutrinos(Species):
    """Massless neutrinos ``n`` (``N_0 .. N_LMAX_NR``): collisionless free streaming."""

    LMAX_NR = Setting(15)

    def _n_initial(self, x, Rv, Rp15, omtau, clxg, output):
        """The growing mode's neutrino monopole, dipole, quadrupole and octupole."""
        output[0] = clxg
        output[1] = (4.0 * Rv + 23.0) / Rp15 * x**3 / 27.0
        output[2] = -4.0 / 3.0 * x * x / Rp15 * (1.0 + omtau / 4.0 * (4.0 * Rv - 5.0) / (2.0 * Rv + 15.0))
        output[3] = -4.0 / 21.0 / Rp15 * x**3

    n = State(lambda self: self.LMAX_NR + 1, tail_from=3, initial=_n_initial)

    def __init__(self, **settings):
        super().__init__(**settings)
        _needs_octupole(LMAX_NR=self.LMAX_NR)

    def grhornomass(self, grhornomass_nominal, eps, dir_grhornomass):
        return grhornomass_nominal + eps * dir_grhornomass

    def grhor_t(self, grhornomass, inv_a2):
        return grhornomass * inv_a2

    def grho(self, grhor_t):
        return grhor_t

    def dgrho(self, grhor_t, n):
        return grhor_t * n[0]

    def dgq(self, grhor_t, n):
        return grhor_t * n[1]

    def dgpi(self, grhor_t, n):
        return grhor_t * n[2]

    def d_n(self, k, z, sigma, n, inv_tau, output):
        output[0] = -k * ((4.0 / 3.0) * z + n[1])
        output[1] = k * (n[0] - 2.0 * n[2]) / 3.0
        output[2] = (2.0 / 5.0) * k * n[1] - (3.0 / 5.0) * k * n[3] + (8.0 / 15.0) * k * sigma
        stream(k, n, inv_tau, output)


class MassiveNeutrinos(Species):
    """Massive neutrinos on ``NQMAX`` momentum bins: ``psi[i, l]`` (Ma & Bertschinger).

    Per bin ``i`` with node ``q_i``, weight ``w_i`` and ``dlnf0/dlnq``, with
    ``v_i = q / sqrt(q^2 + a^2 amnu^2)`` and ``kv = k v_i`` (CLASS
    ``perturbations.c``, synchronous gauge, ``h' = 2 k z``, ``metric_shear =
    k sigma``)::

        psi0' = -kv psi1 + (k z / 3) dlnf0
        psi1' = (kv / 3) (psi0 - 2 psi2)
        psi2' = (kv / 5) (2 psi1 - 3 psi3) - (2/15) k sigma dlnf0

    and the free-streaming tail above. The bins' moments ``Sum w psi0 / v``,
    ``Sum w psi1`` and ``Sum w v psi2`` enter the Einstein sources; the background density
    ratio ``rhonu`` comes from the tables. Deep in radiation domination the
    neutrinos are still relativistic, so each bin starts as the massless
    hierarchy, ``psi_l = -(1/4) N_l dlnf0/dlnq``. The quadrature nodes and
    weights are constants of the instance, which the providers read through
    ``self``. Each bin adds ``LMAX_NU + 1`` state variables; 3 bins give the
    default layout ``nvar = 98``, and 5 bins compile ~4x slower for no gain
    in ``P(k)`` accuracy against CLASS.
    """

    LMAX_NU = Setting(15)
    NQMAX = Setting(3)

    def _psi_initial(self, n, output):
        """Still relativistic: each bin starts as the massless hierarchy scaled by dlnf0/dlnq."""
        for i in range(self.NQMAX):
            for l in range(4):
                output[i, l] = -0.25 * n[l] * self.dlnf0[i]

    psi = State(lambda self: (self.NQMAX, self.LMAX_NU + 1), tail_from=3, initial=_psi_initial)

    def __init__(self, **settings):
        super().__init__(**settings)
        _needs_octupole(LMAX_NU=self.LMAX_NU)
        q, w, dlnf0 = neutrino_momentum_bins(self.NQMAX)
        self.inv_q = tuple(1.0 / float(x) for x in q)
        self.weight = tuple(float(x) for x in w)
        self.dlnf0 = tuple(float(x) for x in dlnf0)

    def nu_norm(self, grhor_nu, n_mnu, inv_a2):
        return grhor_nu * n_mnu * inv_a2

    def grho(self, nu_norm, rhonu):
        return nu_norm * rhonu

    def dgrho(self, nu_norm, a, amnu, psi):
        total = 0.0
        for i in range(self.NQMAX):
            inv_v = math.sqrt(1.0 + (a * amnu * self.inv_q[i]) ** 2)
            total += self.weight[i] * psi[i, 0] * inv_v
        return nu_norm * total

    def dgq(self, nu_norm, psi):
        total = 0.0
        for i in range(self.NQMAX):
            total += self.weight[i] * psi[i, 1]
        return nu_norm * total

    def dgpi(self, nu_norm, a, amnu, psi):
        """The shear ``Sum w v psi2``: the pressure weighting ``q/epsilon`` of a massive bin."""
        total = 0.0
        for i in range(self.NQMAX):
            inv_v = math.sqrt(1.0 + (a * amnu * self.inv_q[i]) ** 2)
            total += self.weight[i] * psi[i, 2] / inv_v
        return nu_norm * total

    def d_psi(self, k, z, sigma, a, amnu, psi, inv_tau, output):
        for i in range(self.NQMAX):
            inv_v = math.sqrt(1.0 + (a * amnu * self.inv_q[i]) ** 2)
            kv = k * (1.0 / inv_v)
            dl = self.dlnf0[i]
            output[i, 0] = -kv * psi[i, 1] + k * z * (1.0 / 3.0) * dl
            output[i, 1] = kv * (1.0 / 3.0) * (psi[i, 0] - 2.0 * psi[i, 2])
            output[i, 2] = kv * (1.0 / 5.0) * (2.0 * psi[i, 1] - 3.0 * psi[i, 3]) - (2.0 / 15.0) * k * sigma * dl
            stream(kv, psi[i], inv_tau, output[i])


# --- dark energy --------------------------------------------------------------------


class CosmologicalConstant(Species):
    """A cosmological constant: density only, no perturbations."""

    def grhov(self, grhov_nominal, eps, dir_grhov):
        return grhov_nominal + eps * dir_grhov

    def grhov_t(self, grhov, a2):
        return grhov * a2

    def grho(self, grhov_t):
        return grhov_t


class DarkEnergy(Species):
    """A CPL fluid ``w = w_0 + w_a (1 - a)`` with rest-frame sound speed ``cs2_DE``.

    Synchronous-gauge fluid equations after Ballesteros & Lesgourgues 2010, with
    the adiabatic sound speed ``c_a^2 = w - w' / (3 (1+w) aH)``, ``w' = -w_a aH a``:

    ``clxq' = -(1+w)(thetaq + k z) - 3 (cs2 - w) aH clxq
              - 9 (1+w)(cs2 - c_a^2) (aH)^2 thetaq / k^2``,
    ``thetaq' = -(1 - 3 cs2) aH thetaq + cs2 k^2 clxq / (1+w)``.

    Valid for ``w != -1``; a cosmological constant is :class:`CosmologicalConstant`.
    The fluid starts in the adiabatic mode, ``clxq = (1 + w_0) clxc`` at rest.
    """

    clxq = State(initial=lambda w_DE_0, clxc: (1.0 + w_DE_0) * clxc)  # adiabatic, at rest
    thetaq = State()

    def grhov(self, grhov_nominal, eps, dir_grhov):
        return grhov_nominal + eps * dir_grhov

    def w_Q(self, a, w_DE_0, w_DE_a):
        return w_DE_0 + w_DE_a * (1.0 - a)

    def rho_Q(self, a, w_DE_0, w_DE_a):
        """``rho_Q(a) / rho_Q(1)`` for the CPL equation of state."""
        return a ** (-3.0 * (1.0 + w_DE_0 + w_DE_a)) * math.exp(3.0 * (a - 1.0) * w_DE_a)

    def grhov_t(self, grhov, rho_Q, a2):
        return grhov * rho_Q * a2

    def grho(self, grhov_t):
        return grhov_t

    def dgrho(self, grhov_t, clxq):
        return grhov_t * clxq

    def dgq(self, grhov_t, w_Q, thetaq, k):
        return grhov_t * (1.0 + w_Q) * thetaq / k

    def w_Q_prime(self, w_DE_a, adotoa, a):
        return -w_DE_a * adotoa * a

    def ca2_Q(self, w_Q, w_Q_prime, adotoa):
        return w_Q - w_Q_prime / (3.0 * (1.0 + w_Q) * adotoa)

    def d_clxq(self, clxq, thetaq, z, k, adotoa, w_Q, cs2_DE, ca2_Q, inv_k2):
        return (
            -(1.0 + w_Q) * (thetaq + k * z)
            - 3.0 * (cs2_DE - w_Q) * adotoa * clxq
            - 9.0 * (1.0 + w_Q) * (cs2_DE - ca2_Q) * adotoa**2 * inv_k2 * thetaq
        )

    def d_thetaq(self, clxq, thetaq, k, adotoa, w_Q, cs2_DE):
        return -(1.0 - 3.0 * cs2_DE) * adotoa * thetaq + cs2_DE / (1.0 + w_Q) * k**2 * clxq


# --- the model ----------------------------------------------------------------------


def perturbation_model(layout, thermal_history: Component | None = None, extra=()) -> Model:
    """The components a :class:`~discoeb.perturbations.PerturbationLayout` enables.

    ``thermal_history`` provides ``a``, ``opacity``, ``cs2`` and ``rhonu`` as
    functions of ``tau`` (on the device, spline tables; in a test, anything);
    without it the model still gives the state layout, the initial conditions
    and the sparsity pattern. ``extra`` components (the line-of-sight sources
    of :mod:`discoeb.cmb`) are appended. The layout's truncations are passed to the
    components as their settings, so a layout built from the components' own
    (environment-loaded) settings reproduces them exactly. The state order
    reproduces the layout's indices: metric, CDM, baryons, photon temperature
    and polarization, massless neutrinos, then the dark-energy fluid and the
    massive-neutrino bins when enabled.
    """

    components = [
        Background(),
        AdiabaticMode(),
        Curvature(),
        Metric(),
        ColdDarkMatter(),
        Baryons(),
        Photons(LMAX_G=layout.lmaxg, LMAX_POL=layout.lmaxpol),
        MasslessNeutrinos(LMAX_NR=layout.lmaxr),
    ]
    if thermal_history is not None:
        components.insert(0, thermal_history)
    if layout.enable_dark_energy:
        components.append(DarkEnergy())
    else:
        components.append(CosmologicalConstant())
    if layout.has_massive_neutrinos:
        components.append(MassiveNeutrinos(LMAX_NU=layout.lmaxnu, NQMAX=layout.nqmax))
    components.append(ThomsonScattering())
    components.extend(extra)
    return Model(components, sums=SUMS)


def perturbation_sparsity(layout) -> np.ndarray:
    """The ``(nnz, 2)`` index array of the Jacobian's nonzeros for ``layout``.

    modax's Rodas5P kernel takes a sparsity pattern and does two things with
    it: it colours the column intersection graph, so the Jacobian costs
    ``n_colours + 1`` Enzyme sweeps rather than ``nvar + 1``; and it hands the
    pattern to ``modax._sparse_direct``, which orders it, factorises it
    symbolically and compiles a sparse LU and a pair of sparse triangular
    solves for that exact structure. Neither needs to know any physics -- only
    the states do, through ``State(..., tail_from=l)``: everything below ``l``
    is densely coupled core (the metric, the fluids, the lowest multipoles,
    reaching each other through the Einstein constraints), and from ``l`` on
    each hierarchy is a tridiagonal free-streaming tail meeting the core only
    through the multipole it streams out of. Declaring the core dense makes
    this a superset of the true Jacobian, which is what a colouring needs: a
    subset would silently corrupt the entries where two columns of a colour
    group overlap. The pattern need not include fill-in; on this structure
    minimum degree peels each tail from its truncated end and produces none.
    """

    return perturbation_model(layout).sparsity()


def layout_state_names(layout) -> tuple[str, ...]:
    """The state names of :func:`perturbation_model` in the layout's index order."""

    names = ["etak", "clxc", "clxb", "vb"]
    names += [f"theta_{l}" for l in range(layout.lmaxg + 1)]
    names += [f"e_{i}" for i in range(layout.lmaxpol - 1)]
    names += [f"n_{l}" for l in range(layout.lmaxr + 1)]
    if layout.enable_dark_energy:
        names += ["clxq", "thetaq"]
    if layout.has_massive_neutrinos:
        names += [f"psi_{i}_{l}" for i in range(layout.nqmax) for l in range(layout.lmaxnu + 1)]
    return tuple(names)


# --- the expansion history ----------------------------------------------------------
# Density coefficients in the convention of the rest of the code (and of CAMB):
# ``8 pi G rho / c^2`` in 1/Mpc^2, so that ``H^2 = sum_i rho_i / 3`` is the Hubble
# rate in 1/Mpc and ``a'/a = a H`` the conformal one. The perturbation kernel's
# ``grho_i_t = 8 pi G rho_i a^2`` are these times ``a^2``.

GRHOM_PER_H0SQ = 3.33795017e-11  # 8 pi G rho_crit / c^2 per (H0 in km/s/Mpc)^2
GRHOG_PER_TCMB4 = 1.49594245e-13  # photons, per K^4
GRHOR_PER_TCMB4 = 3.39739477e-14  # one massless neutrino flavour, per K^4
C2OK = 1.62581581e4  # K / eV: amnu = m_nu c^2 / (k_B T_nu0) = mnu * C2OK / Tcmb

NQ_NEUTRINO_TABLE = 100
"""Momentum nodes of the massive-neutrino background density: the ``rho_nu`` tables
and the budget closing both integrate the Fermi-Dirac distribution on these."""


def get_neutrino_momentum_bins(nqmax: int) -> tuple[jax.Array, jax.Array]:
    """The momentum nodes ``q`` (in units of ``k_B T_nu0 / c``) and weights ``w``.

    ``w`` is normalised so that one massless flavour integrates to unit density.
    ``nqmax = 3, 4, 5`` are CAMB's high-accuracy formulas; higher counts use a
    generalised Gauss-Laguerre rule.
    """

    fermi_dirac_const = 7 * jnp.pi**4 / 120
    if nqmax == 3:
        q = jnp.array([0.913201, 3.37517, 7.79184])
        dlfdlq = -q / (1 + jnp.exp(-q))
        w = jnp.array([0.0687359, 3.31435, 2.29911]) / (-0.25 * dlfdlq)
    elif nqmax == 4:
        q = jnp.array([0.7, 2.62814, 5.90428, 12.0])
        dlfdlq = -q / (1 + jnp.exp(-q))
        w = jnp.array([0.0200251, 1.84539, 3.52736, 0.289427]) / (-0.25 * dlfdlq)
    elif nqmax == 5:
        q = jnp.array([0.583165, 2.0, 4.0, 7.26582, 13.0])
        dlfdlq = -q / (1 + jnp.exp(-q))
        w = jnp.array([0.0081201, 0.689407, 2.8063, 2.05156, 0.12681]) / (-0.25 * dlfdlq)
    else:
        alpha = 1
        q, w = generalized_gauss_laguerre_weights(nqmax, alpha)
        w *= q**3 / (1 + jnp.exp(-q)) * q**-alpha
    return q, w / fermi_dirac_const


def neutrino_momentum_bins(nqmax: int):
    """Return ``(q, w, dlnf0_dlnq)`` momentum bins as NumPy arrays.

    ``dlnf0/dlnq = -q/(1 + exp(-q))`` is the log-derivative of the Fermi-Dirac
    background distribution that sources the ``psi`` hierarchy.
    """

    with jax.ensure_compile_time_eval():
        q, w = get_neutrino_momentum_bins(nqmax)
    q = np.asarray(q, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    dlfdlq = -q / (1.0 + np.exp(-q))
    return q, w, dlfdlq


# The massive-neutrino density relative to the massless flavour depends on
# ``y = amnu * a`` alone, so it is tabulated once over ``ln y`` and shared by
# every cosmology. Below the table the neutrinos are relativistic and the ratio
# is 1; that limit also serves ``amnu = 0``.
_LN_Y_MIN, _LN_Y_MAX = math.log(1.0e-6), math.log(1.0e4)
_NEUTRINO_TABLE: dict = {}
_NEUTRINO_NODES: dict = {}


def _neutrino_table(n_y_grid: int = 1000):
    """The spline of ``ln(rho_nu / rho_nu,massless)`` over ``ln y``.

    Built on first use rather than at import, so it takes the ``jax_enable_x64``
    setting in force then, and concretely even when first used inside a ``jit``.
    """

    key = (n_y_grid, bool(jax.config.jax_enable_x64))
    table = _NEUTRINO_TABLE.get(key)
    if table is None:
        with jax.ensure_compile_time_eval():
            log_y = jnp.linspace(_LN_Y_MIN, _LN_Y_MAX, n_y_grid)
            q, w = get_neutrino_momentum_bins(NQ_NEUTRINO_TABLE)
            ratio = jnp.sum(w[None, :] * jnp.sqrt(1.0 + (jnp.exp(log_y)[:, None] / q[None, :]) ** 2), axis=1)
            table = FastUniformCubicSpline1D(log_y, jnp.log(ratio))
        _NEUTRINO_TABLE[key] = table
    return table


def neutrino_density_ratio(a, amnu, xp=jnp):
    """``rho_nu(a) / rho_nu,massless`` of one flavour of mass ``amnu`` (1 when massless).

    With ``xp=jnp`` (the default) the shared table, so it traces under ``jit``;
    with ``xp=np`` the same quadrature summed directly, for the host-side
    budget closing of a cosmology given as floats.
    """

    if xp is np:
        nodes = _NEUTRINO_NODES.get(NQ_NEUTRINO_TABLE)
        if nodes is None:
            q, w, _ = neutrino_momentum_bins(NQ_NEUTRINO_TABLE)
            nodes = _NEUTRINO_NODES[NQ_NEUTRINO_TABLE] = (q, w)
        q, w = nodes
        y = np.asarray(a, dtype=np.float64) * float(amnu)
        return np.sum(w * np.sqrt(1.0 + (y[..., None] / q) ** 2), axis=-1)
    massive = amnu > 0.0
    ln_y = jnp.log(jnp.where(massive, amnu, 1.0) * a)
    ln_y = jnp.clip(ln_y, _LN_Y_MIN, _LN_Y_MAX)
    ratio = jnp.exp(jnp.vectorize(_neutrino_table().evaluate)(ln_y))
    return jnp.where(massive, ratio, 1.0)


_BACKGROUND_MODEL: list = []


def background_model() -> Model:
    """The species whose ``grho`` is the Friedmann sum, with every component enabled.

    The dark-energy fluid at ``w = -1`` is the cosmological constant and a
    massive-neutrino density with ``n_mnu = 0`` vanishes, so one model serves
    every cosmology; the hierarchy truncations are irrelevant to the background.
    """

    if not _BACKGROUND_MODEL:
        _BACKGROUND_MODEL.append(
            Model(
                [Background(), Curvature(), ColdDarkMatter(), Baryons(), Photons(),
                 MasslessNeutrinos(), DarkEnergy(), MassiveNeutrinos()],
                sums=SUMS,
            )
        )
    return _BACKGROUND_MODEL[0]


def density_parameters(param, xp=jnp) -> dict:
    """The background densities of a parameter dictionary, as the model names them.

    ``grhom`` is the critical density, ``grhog`` the photons', ``grhor`` one
    massless neutrino flavour's, ``grhornomass`` the ``Neff`` massless flavours',
    ``grhoc``, ``grhob`` and ``grhok`` the cold dark matter, baryons and
    curvature, ``grhor_nu``, ``n_mnu`` and ``amnu`` the massive neutrinos', and
    ``grhov`` the dark energy's, fixed by closing the budget at ``a = 1`` with
    the model's own Friedmann sum: ``grhov = grhom - grho(a=1 | grhov=0)``.
    ``xp`` is ``jnp`` for a traced cosmology, ``np`` for one given as floats.
    """

    H0, Tcmb = param["H0"], param["Tcmb"]
    grhom = GRHOM_PER_H0SQ * H0**2
    grhor = GRHOR_PER_TCMB4 * Tcmb**4
    d = {
        "grhom": grhom,
        "grhog": GRHOG_PER_TCMB4 * Tcmb**4,
        "grhor": grhor,
        "grhornomass": param["Neff"] * grhor,
        "grhoc": grhom * (param["Omegam"] - param["Omegab"]),
        "grhob": grhom * param["Omegab"],
        "grhok": grhom * param["Omegak"],
        "grhor_nu": grhor,
        "n_mnu": param["Nmnu"],
        "amnu": param["mnu"] * C2OK / Tcmb,
        "w_DE_0": param["w_DE_0"],
        "w_DE_a": param["w_DE_a"],
        "cs2_DE": param.get("cs2_DE", 1.0),
    }
    rhonu_today = neutrino_density_ratio(1.0, d["amnu"], xp)
    without_dark_energy = background_model().evaluate(
        "grho", math=(math if xp is np else jnp), a=1.0, rhonu=rhonu_today, grhov=0.0, **d
    )
    d["grhov"] = grhom - without_dark_energy
    d["OmegaDE"] = d["grhov"] / grhom
    d["grho_mnu_today"] = grhor * d["n_mnu"] * rhonu_today
    return d


def expansion_rate(a, densities):
    """The conformal Hubble rate ``a'/a = a H`` at scalar ``a``, from the model's ``adotoa``."""

    rhonu = neutrino_density_ratio(a, densities["amnu"])
    return background_model().evaluate("adotoa", math=jnp, a=a, rhonu=rhonu, **densities)


def dtauda(a, densities):
    """``d tau / d a = 1 / (a^2 H)`` at scalar ``a`` (``jax.vmap`` over an array)."""

    return 1.0 / (a * expansion_rate(a, densities))


def get_aprimeoa(*, densities, a):
    """The conformal Hubble rate ``a'/a = a H`` at scalar ``a``."""

    return expansion_rate(a, densities)


def compute_angular_diameter_distance(*, aexp, param):
    """The angular diameter distance to ``aexp``, from ``param['densities']``."""

    aexpv = jnp.linspace(aexp, 1.0, 1000)
    aH = jax.vmap(lambda a: expansion_rate(a, param["densities"]))(aexpv) * aexpv
    return aexp * integrate_trapz(1 / aH, aexpv)


def setup_background_evolution(*, amin, amax, param):
    """Mean densities, the massive-neutrino spline and ``tau(a)`` of ``param``.

    Sets the ``grhom`` / ``grhog`` / ``grhor`` / ``amnu`` coefficients and
    ``OmegaDE`` the thermodynamics kernel reads, ``param['densities']`` (what
    :func:`dtauda` takes), the ``log rho_nu`` spline, and the conformal time on
    the ``param['a']`` grid from the model's expansion rate. One cosmology per
    call: batch with ``jax.vmap``.
    """

    num_neutrino = 512  # points of the a grid

    param["amin"] = amin
    param["amax"] = amax
    densities = density_parameters(param)
    param["grhom"] = densities["grhom"]
    param["grhog"] = densities["grhog"]
    param["grhor"] = densities["grhor"]
    param["amnu"] = densities["amnu"]
    param["OmegaDE"] = densities["OmegaDE"]
    param["Omegamnu"] = densities["grho_mnu_today"] / densities["grhom"]
    param["densities"] = densities

    a = jnp.geomspace(amin * 0.9, amax * 1.1, num_neutrino)
    loga = jnp.log(a)
    param["a"] = a
    rhonu = neutrino_density_ratio(a, densities["amnu"])
    param["logrhonu_of_loga_spline"] = spline_interpolation(loga, jnp.log(rhonu)[:, None], uniform=True)

    radiation = (
        densities["grhog"] + densities["grhornomass"]
        + densities["grhor_nu"] * densities["n_mnu"] * neutrino_density_ratio(amin, densities["amnu"])
    )
    param["adotrad"] = jnp.sqrt(radiation / 3.0)
    param["taumin"] = amin / param["adotrad"]
    dtauda_values = jnp.reshape(jax.vmap(dtauda, in_axes=(0, None))(a, densities), a.shape)
    integrator = FastUniformCubicSpline1D(loga, dtauda_values * a)
    param["tau"] = param["taumin"] + jax.vmap(integrator.integral)(loga)
    param["taumax"] = param["tau"][-1]
    return param


def batch_dimensions(param):
    for x in param:
        param[x] = jnp.atleast_1d(param[x])
    return param


@partial(jax.jit, static_argnames=("thermo_module", "num_thermo", "rtol", "atol", "order", "class_thermo"))
def evolve_background(*, param, thermo_module="RECFAST", num_thermo: int = 256, rtol: float = 1e-5,
                      atol: float = 1e-7, order: int = 5, class_thermo=None):
    """Evolve the cosmological background and thermal history.

    ``thermo_module`` is ``'RECFAST'`` (default) or ``'CLASS'`` (external CLASS
    data in ``class_thermo``); ``num_thermo`` the number of thermal-history
    samples (adaptive, half of them through recombination: 256 gives < 0.13
    percent on P(k), 512 < 0.03 percent); ``rtol`` and ``atol`` the RECFAST
    solve's tolerances. Returns the parameter dictionary with the background,
    the thermal history and its splines, the optical depth and the visibility
    function.
    """

    from .thermodynamics_recfast import evaluate_thermo

    amin = 1e-9
    amax = 1.01
    if thermo_module == "CLASS":
        amin = jnp.min(class_thermo["scale factor a"])
        amax = jnp.max(class_thermo["scale factor a"])

    param = setup_background_evolution(amin=amin, amax=amax, param=param)

    if thermo_module == "RECFAST":
        aexp, cs2, Tm, _mu, xe, _dxeda = evaluate_thermo(
            param=param, num_thermo=num_thermo, rtol=rtol, atol=atol, amin=amin, amax=amax
        )
        param["aexp"] = aexp
        param["xe"] = xe
        param["cs2"] = cs2
        param["Tm"] = Tm

        tau = spline_interpolation(jnp.log(param["a"]), param["tau"]).evaluate(jnp.log(aexp))
        param["tau_th"] = tau
        param["tau_of_a_spline"] = spline_interpolation(aexp, tau)
        param["a_of_tau_spline"] = spline_interpolation(tau, aexp)
        param["xe_of_tau_spline"] = spline_interpolation(tau, xe)
        param["cs2a_of_tau_spline"] = spline_interpolation(tau, aexp[:, None] * cs2)
        param["tempba_of_tau_spline"] = spline_interpolation(tau, aexp[:, None] * Tm)
        # Pre-composed splines for direct a-to-quantity lookups.
        param["xe_of_loga_spline"] = spline_interpolation(jnp.log(aexp), xe, uniform=False)
        param["cs2a_of_loga_spline"] = spline_interpolation(jnp.log(aexp), aexp[:, None] * cs2, uniform=False)
    elif thermo_module == "CLASS":
        tau_class = class_thermo["conf. time [Mpc]"][::-1]
        aexp_class = class_thermo["scale factor a"][::-1]
        param["cs2_of_tau_spline"] = spline_interpolation(tau_class, class_thermo["c_b^2"][::-1])
        param["tempb_of_tau_spline"] = spline_interpolation(tau_class, class_thermo["Tb [K]"][::-1])
        param["xe_of_tau_spline"] = spline_interpolation(tau_class, class_thermo["x_e"][::-1])
        param["a_of_tau_spline"] = spline_interpolation(tau_class, aexp_class)
        param["tau_of_a_spline"] = spline_interpolation(aexp_class, tau_class)
        param["aexp"] = aexp_class
        param["tau"] = tau_class
        param["xe_of_loga_spline"] = spline_interpolation(jnp.log(aexp_class), class_thermo["x_e"][::-1])
        param["cs2a_of_loga_spline"] = spline_interpolation(
            jnp.log(aexp_class), aexp_class * class_thermo["c_b^2"][::-1]
        )
        aexp, tau = aexp_class, tau_class

    # Optical depth and visibility functions.
    akthom = 2.3038921003709498e-9 * (1.0 - param["YHe"]) * param["Omegab"] * param["H0"] ** 2
    tau_pre_recomb = param["tau_of_a_spline"].evaluate(1e-4)
    xe_full = 1 + param["YHe"] / (1 - param["YHe"])
    xe = jnp.where(tau <= tau_pre_recomb, xe_full, param["xe_of_tau_spline"].evaluate(tau))
    xeprime = jnp.where(tau <= tau_pre_recomb, 0.0, param["xe_of_tau_spline"].derivative(tau))
    opac = xe * akthom / aexp[:, None] ** 2

    opacspline = spline_interpolation(tau, opac, integrate_from_start=True)
    opacprime, opacpprime = opacspline.derivative12(tau)
    optical_depth_today = opacspline.integral(param["tau_of_a_spline"].evaluate(1.0))
    optical_depth = optical_depth_today - opacspline.integral(tau)
    expmmu = jnp.exp(-optical_depth)

    param["optical_depth"] = optical_depth
    param["opac"] = opac
    param["gvis"] = opac * expmmu
    param["gvisprime"] = (opacprime + opac**2) * expmmu
    param["gvispprime"] = (opacpprime + 3 * opac * opacprime + opac**3) * expmmu
    param["tau0"] = param["tau_of_a_spline"].evaluate(1.0)
    param["tau_maxvis"] = param["tau"][jnp.argmax(param["gvis"])]
    param["xeprime"] = xeprime
    return param
