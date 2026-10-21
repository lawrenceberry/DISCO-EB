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
environment or by a constructor argument. Each state also carries its
adiabatic initial condition (CAMB's ``initial()`` series at ``k tau << 1``,
with the growing-mode quantities of :class:`AdiabaticMode`) and its place in
the Jacobian's sparsity pattern (``tail_from``), so the layout, the initial
state and the sparse solver are all read off the components. The photon
monopole, dipole and quadrupole are ``theta[0]``, ``theta[1]``, ``theta[2]``;
the scalar names ``clxg``, ``qg``, ``pig`` and ``E2`` are published for the
equations that read them.
"""

from __future__ import annotations

import math

import numpy as np

from .background import get_neutrino_momentum_bins
from .component import Component, Interaction, Model, Setting, Species, State

SUMS = ("grho", "dgrho", "dgq")
"""The quantities summed over species: the Friedmann density, and the sources
of the Einstein constraints ``dgrho = Sum grho_i_t delta_i`` and
``dgq = Sum grho_i_t q_i``."""


def _needs_octupole(**lmax):
    """The quadrupole equations couple to ``l = 3``, so a hierarchy truncates at 3 or above."""

    low = {name: l for name, l in lmax.items() if l < 3}
    if low:
        raise ValueError(f"a hierarchy must reach at least the octupole; got {low}")


# --- background ---------------------------------------------------------------------


class Background(Component):
    """Powers of the scale factor, the Friedmann equation, ``1/tau`` and ``1/k^2``."""

    def a2(a):
        return a * a

    def inv_a(a):
        return 1.0 / a

    def inv_a2(inv_a):
        return inv_a * inv_a

    def inv_tau(tau):
        return 1.0 / tau

    def inv_k2(k):
        return 1.0 / (k * k)

    def adotoa(grho):
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

    def x(k, tau):
        return k * tau

    def Rv(grhog, grhornomass):
        return grhornomass / (grhog + grhornomass)

    def Rp15(Rv):
        return 4.0 * Rv + 15.0

    def om(grhob, grhoc, grhog, grhornomass):
        return (grhob + grhoc) / math.sqrt(3.0 * (grhog + grhornomass))

    def omtau(om, tau):
        return om * tau


class Curvature(Species):
    """Spatial curvature: a density ``grhok = 8 pi G rho_K a^2`` and the metric factors."""

    def grho(grhok):
        return grhok

    def s2(grhok, inv_k2):
        """``s2^2 = 1 - 3K/k^2 = 1 + grhok/k^2`` (1 when flat)."""
        return 1.0 + grhok * inv_k2

    def inv_s2(s2):
        return 1.0 / s2

    def Kcurv(grhok):
        return -grhok * (1.0 / 3.0)


class Metric(Component):
    """The synchronous-gauge metric: ``etak = k eta`` and the Einstein constraints.

    ``h' = (k^2 s2^2 eta + 1.5 a^2 delta_rho) / (0.5 aH)`` with ``z = h'/(2k)``,
    ``eta' = (1.5 a^2 (rho+p) theta + 0.5 K h') / (k^2 s2^2)``, and the shear
    ``sigma = k alpha = (z + 1.5 dgq/k^2) / s2^2``.
    """

    etak = State(initial=lambda k, x, Rp15: -k * (1.0 - x * x / 12.0 * (-10.0 / Rp15 + 1.0)))

    def z(dgrho, k, s2, etak, adotoa):
        return (0.5 * dgrho / k + s2 * etak) / adotoa

    def sigma(z, dgq, inv_k2, inv_s2):
        return (z + 1.5 * dgq * inv_k2) * inv_s2

    def d_etak(dgq, Kcurv, z, inv_s2):
        return (0.5 * dgq + Kcurv * z) * inv_s2


# --- matter -------------------------------------------------------------------------
# The density parameters are read through the trajectory's sensitivity
# direction, ``grho_i = grho_i_nominal + eps * dir_grho_i``: ``eps`` is zero in
# every primal solve and d/d(eps) is the derivative along ``dir``.


class ColdDarkMatter(Species):
    clxc = State(initial=lambda clxg: 0.75 * clxg)  # the adiabatic mode: 3/4 of the photon density

    def grhoc(grhoc_nominal, eps, dir_grhoc):
        return grhoc_nominal + eps * dir_grhoc

    def grhoc_t(grhoc, inv_a):
        return grhoc * inv_a

    def grho(grhoc_t):
        return grhoc_t

    def dgrho(grhoc_t, clxc):
        return grhoc_t * clxc

    def d_clxc(z, k):
        """CDM is pressureless and at rest in synchronous gauge: ``clxc' = -k z``."""
        return -k * z


class Baryons(Species):
    """Baryon continuity and Euler equations; the Thomson drag is :class:`ThomsonScattering`."""

    clxb = State(initial=lambda clxg: 0.75 * clxg)
    vb = State(initial=lambda qg: 0.75 * qg)

    def grhob(grhob_nominal, eps, dir_grhob):
        return grhob_nominal + eps * dir_grhob

    def grhob_t(grhob, inv_a):
        return grhob * inv_a

    def grho(grhob_t):
        return grhob_t

    def dgrho(grhob_t, clxb):
        return grhob_t * clxb

    def dgq(grhob_t, vb):
        return grhob_t * vb

    def d_clxb(z, vb, k):
        return -k * (z + vb)

    def d_vb(adotoa, vb, k, cs2, clxb):
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

    def _theta_initial(x, omtau, output):
        """The monopole ``delta_gamma`` and dipole ``q_gamma`` of the growing mode."""
        output[0] = x * x / 3.0 * (1.0 - omtau / 5.0)
        output[1] = x**3 / 27.0 * (1.0 - omtau / 5.0)

    theta = State(lambda self: self.LMAX_G + 1, tail_from=3, initial=_theta_initial)
    e = State(lambda self: self.LMAX_POL - 1, tail_from=1)

    def __init__(self, **settings):
        super().__init__(**settings)
        _needs_octupole(LMAX_G=self.LMAX_G, LMAX_POL=self.LMAX_POL)

    def grhog(grhog_nominal, eps, dir_grhog):
        return grhog_nominal + eps * dir_grhog

    def grhog_t(grhog, inv_a2):
        return grhog * inv_a2

    def grho(grhog_t):
        return grhog_t

    def clxg(theta):
        return theta[0]

    def qg(theta):
        return theta[1]

    def pig(theta):
        return theta[2]

    def E2(e):
        """The E-mode quadrupole, ``e[0]``."""
        return e[0]

    def dgrho(grhog_t, clxg):
        return grhog_t * clxg

    def dgq(grhog_t, qg):
        return grhog_t * qg

    def d_theta(k, z, sigma, theta, inv_tau, output):
        output[0] = -k * ((4.0 / 3.0) * z + theta[1])
        output[1] = k * theta[0] * (1.0 / 3.0) - (2.0 / 3.0) * k * theta[2]
        output[2] = (2.0 / 5.0) * k * theta[1] - (3.0 / 5.0) * k * theta[3] + (8.0 / 15.0) * k * sigma
        stream(k, theta, inv_tau, output)

    def d_e(k, e, inv_tau, output):
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

    def photbar(grhog_t, grhob_t):
        """The photon-to-baryon density ratio ``R^-1 = rho_gamma / rho_b``."""
        return grhog_t / grhob_t

    def slip(vb, qg):
        return (4.0 / 3.0) * vb - qg

    def polter(pig, E2):
        return pig * (1.0 / 10.0) + 9.0 * E2 * (1.0 / 15.0)

    def d_vb(photbar, opacity, slip):
        return -photbar * opacity * slip

    def d_theta(opacity, slip, polter, theta, output):
        output[1] = opacity * slip
        output[2] = -opacity * (theta[2] - polter)
        for l in range(3, len(theta)):
            output[l] = -opacity * theta[l]

    def d_e(opacity, polter, e, output):
        output[0] = -opacity * (e[0] - polter)
        for i in range(1, len(e)):
            output[i] = -opacity * e[i]


class MasslessNeutrinos(Species):
    """Massless neutrinos ``n`` (``N_0 .. N_LMAX_NR``): collisionless free streaming."""

    LMAX_NR = Setting(15)

    def _n_initial(x, Rv, Rp15, omtau, clxg, output):
        """The growing mode's neutrino monopole, dipole, quadrupole and octupole."""
        output[0] = clxg
        output[1] = (4.0 * Rv + 23.0) / Rp15 * x**3 / 27.0
        output[2] = -4.0 / 3.0 * x * x / Rp15 * (1.0 + omtau / 4.0 * (4.0 * Rv - 5.0) / (2.0 * Rv + 15.0))
        output[3] = -4.0 / 21.0 / Rp15 * x**3

    n = State(lambda self: self.LMAX_NR + 1, tail_from=3, initial=_n_initial)

    def __init__(self, **settings):
        super().__init__(**settings)
        _needs_octupole(LMAX_NR=self.LMAX_NR)

    def grhornomass(grhornomass_nominal, eps, dir_grhornomass):
        return grhornomass_nominal + eps * dir_grhornomass

    def grhor_t(grhornomass, inv_a2):
        return grhornomass * inv_a2

    def grho(grhor_t):
        return grhor_t

    def dgrho(grhor_t, n):
        return grhor_t * n[0]

    def dgq(grhor_t, n):
        return grhor_t * n[1]

    def d_n(k, z, sigma, n, inv_tau, output):
        output[0] = -k * ((4.0 / 3.0) * z + n[1])
        output[1] = k * (n[0] - 2.0 * n[2]) / 3.0
        output[2] = (2.0 / 5.0) * k * n[1] - (3.0 / 5.0) * k * n[3] + (8.0 / 15.0) * k * sigma
        stream(k, n, inv_tau, output)


def neutrino_momentum_bins(nqmax: int):
    """Return ``(q, w, dlnf0_dlnq)`` momentum bins as NumPy arrays.

    ``w`` are normalized so a single massless flavour integrates to unit density,
    and ``dlnf0/dlnq = -q/(1 + exp(-q))`` is the log-derivative of the
    Fermi-Dirac background distribution that sources the ``psi`` hierarchy.
    """

    q, w = get_neutrino_momentum_bins(nqmax)
    q = np.asarray(q, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    dlfdlq = -q / (1.0 + np.exp(-q))
    return q, w, dlfdlq


class MassiveNeutrinos(Species):
    """Massive neutrinos on ``NQMAX`` momentum bins: ``psi[i, l]`` (Ma & Bertschinger).

    Per bin ``i`` with node ``q_i``, weight ``w_i`` and ``dlnf0/dlnq``, with
    ``v_i = q / sqrt(q^2 + a^2 amnu^2)`` and ``kv = k v_i`` (CLASS
    ``perturbations.c``, synchronous gauge, ``h' = 2 k z``, ``metric_shear =
    k sigma``)::

        psi0' = -kv psi1 + (k z / 3) dlnf0
        psi1' = (kv / 3) (psi0 - 2 psi2)
        psi2' = (kv / 5) (2 psi1 - 3 psi3) - (2/15) k sigma dlnf0

    and the free-streaming tail above. The bins' moments ``Sum w psi0 / v``
    and ``Sum w psi1`` enter the Einstein sources; the background density
    ratio ``rhonu`` comes from the tables. Deep in radiation domination the
    neutrinos are still relativistic, so each bin starts as the massless
    hierarchy, ``psi_l = -(1/4) N_l dlnf0/dlnq``. The quadrature nodes and
    weights are constants of the instance, available to the providers by
    name. Each bin adds ``LMAX_NU + 1`` state variables; 3 bins give the
    default layout ``nvar = 98``, and 5 bins compile ~4x slower for no gain
    in ``P(k)`` accuracy against CLASS.
    """

    LMAX_NU = Setting(15)
    NQMAX = Setting(3)

    def _psi_initial(n, output):
        """Still relativistic: each bin starts as the massless hierarchy scaled by dlnf0/dlnq."""
        for i in range(NQMAX):
            for l in range(4):
                output[i, l] = -0.25 * n[l] * dlnf0[i]

    psi = State(lambda self: (self.NQMAX, self.LMAX_NU + 1), tail_from=3, initial=_psi_initial)

    def __init__(self, **settings):
        super().__init__(**settings)
        _needs_octupole(LMAX_NU=self.LMAX_NU)
        q, w, dlnf0 = neutrino_momentum_bins(self.NQMAX)
        self.inv_q = tuple(1.0 / float(x) for x in q)
        self.weight = tuple(float(x) for x in w)
        self.dlnf0 = tuple(float(x) for x in dlnf0)

    def nu_norm(grhor_nu, n_mnu, inv_a2):
        return grhor_nu * n_mnu * inv_a2

    def grho(nu_norm, rhonu):
        return nu_norm * rhonu

    def dgrho(nu_norm, a, amnu, psi):
        total = 0.0
        for i in range(NQMAX):
            inv_v = math.sqrt(1.0 + (a * amnu * inv_q[i]) ** 2)
            total += weight[i] * psi[i, 0] * inv_v
        return nu_norm * total

    def dgq(nu_norm, psi):
        total = 0.0
        for i in range(NQMAX):
            total += weight[i] * psi[i, 1]
        return nu_norm * total

    def d_psi(k, z, sigma, a, amnu, psi, inv_tau, output):
        for i in range(NQMAX):
            inv_v = math.sqrt(1.0 + (a * amnu * inv_q[i]) ** 2)
            kv = k * (1.0 / inv_v)
            dl = dlnf0[i]
            output[i, 0] = -kv * psi[i, 1] + k * z * (1.0 / 3.0) * dl
            output[i, 1] = kv * (1.0 / 3.0) * (psi[i, 0] - 2.0 * psi[i, 2])
            output[i, 2] = kv * (1.0 / 5.0) * (2.0 * psi[i, 1] - 3.0 * psi[i, 3]) - (2.0 / 15.0) * k * sigma * dl
            stream(kv, psi[i], inv_tau, output[i])


# --- dark energy --------------------------------------------------------------------


class CosmologicalConstant(Species):
    """A cosmological constant: density only, no perturbations."""

    def grhov(grhov_nominal, eps, dir_grhov):
        return grhov_nominal + eps * dir_grhov

    def grhov_t(grhov, a2):
        return grhov * a2

    def grho(grhov_t):
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

    def grhov(grhov_nominal, eps, dir_grhov):
        return grhov_nominal + eps * dir_grhov

    def w_Q(a, w_DE_0, w_DE_a):
        return w_DE_0 + w_DE_a * (1.0 - a)

    def rho_Q(a, w_DE_0, w_DE_a):
        """``rho_Q(a) / rho_Q(1)`` for the CPL equation of state."""
        return a ** (-3.0 * (1.0 + w_DE_0 + w_DE_a)) * math.exp(3.0 * (a - 1.0) * w_DE_a)

    def grhov_t(grhov, rho_Q, a2):
        return grhov * rho_Q * a2

    def grho(grhov_t):
        return grhov_t

    def dgrho(grhov_t, clxq):
        return grhov_t * clxq

    def dgq(grhov_t, w_Q, thetaq, k):
        return grhov_t * (1.0 + w_Q) * thetaq / k

    def w_Q_prime(w_DE_a, adotoa, a):
        return -w_DE_a * adotoa * a

    def ca2_Q(w_Q, w_Q_prime, adotoa):
        return w_Q - w_Q_prime / (3.0 * (1.0 + w_Q) * adotoa)

    def d_clxq(clxq, thetaq, z, k, adotoa, w_Q, cs2_DE, ca2_Q, inv_k2):
        return (
            -(1.0 + w_Q) * (thetaq + k * z)
            - 3.0 * (cs2_DE - w_Q) * adotoa * clxq
            - 9.0 * (1.0 + w_Q) * (cs2_DE - ca2_Q) * adotoa**2 * inv_k2 * thetaq
        )

    def d_thetaq(clxq, thetaq, k, adotoa, w_Q, cs2_DE):
        return -(1.0 - 3.0 * cs2_DE) * adotoa * thetaq + cs2_DE / (1.0 + w_Q) * k**2 * clxq


# --- the model ----------------------------------------------------------------------


def perturbation_model(layout, thermal_history: Component | None = None) -> Model:
    """The components a :class:`~discoeb.perturbations.PerturbationLayout` enables.

    ``thermal_history`` provides ``a``, ``opacity``, ``cs2`` and ``rhonu`` as
    functions of ``tau`` (on the device, spline tables; in a test, anything);
    without it the model still gives the state layout, the initial conditions
    and the sparsity pattern. The layout's truncations are passed to the
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
    return Model(components, sums=SUMS)


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
