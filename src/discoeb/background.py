import math
from functools import partial
import jax
import jax.numpy as jnp
import equinox as eqx

from .thermodynamics_recfast import evaluate_thermo as evaluate_thermo_recfast

from .spline_interpolation import spline_interpolation, FastUniformCubicSpline1D
from .util import generalized_gauss_laguerre_weights, integrate_trapz

from abc import abstractmethod


def get_neutrino_momentum_bins(  nqmax : int ) -> tuple[jax.Array, jax.Array]:
    """Get the momentum bins and integral kernel weights for neutrinos

    Args:
        nqmax (int): Number of momentum bins.

    Returns:
        jax.Array: q, w
    """
    fermi_dirac_const = 7 * jnp.pi**4 / 120 # Should be equivalent to the below due to compiler optimizations
    #fermi_dirac_const = 5.682196976983475

    # nqmax = 3,4,5 are from high accuracy formulas from CAMB, higher values resort to modified Gauss-Laguerre,
    # which is not pre-computed however
    if nqmax == 3:
        q = jnp.array([0.913201, 3.37517, 7.79184])
        dlfdlq = -q/(1+jnp.exp(-q))
        w = jnp.array([0.0687359, 3.31435, 2.29911]) / (-0.25*dlfdlq)
    elif nqmax == 4:
        q = jnp.array([0.7, 2.62814, 5.90428, 12.0])
        dlfdlq = -q/(1+jnp.exp(-q))
        w = jnp.array([0.0200251, 1.84539, 3.52736, 0.289427]) / (-0.25*dlfdlq)
    elif nqmax == 5:
        q = jnp.array([0.583165, 2.0, 4.0, 7.26582, 13.0])
        dlfdlq = -q/(1+jnp.exp(-q))
        w = jnp.array([0.0081201, 0.689407, 2.8063, 2.05156, 0.12681]) / (-0.25*dlfdlq)
    else:
        alpha = 1
        q, w = generalized_gauss_laguerre_weights( nqmax, alpha )
        w *= q**3 / (1 + jnp.exp(-q)) * q**-alpha

    return q, w / fermi_dirac_const


# Density coefficients in the convention of the rest of the code (and of CAMB):
# ``8 pi G rho / c^2`` in 1/Mpc^2, so that ``H^2 = sum_i rho_i / 3`` is the Hubble
# rate in 1/Mpc and ``a'/a = a H`` the conformal one. The device kernels
# (``thermo_parameter_rows``, ``density_coefficients``) use the same numbers.
GRHOM_PER_H0SQ = 3.33795017e-11  # 8 pi G rho_crit / c^2 per (H0 in km/s/Mpc)^2
GRHOG_PER_TCMB4 = 1.49594245e-13  # photons, per K^4
GRHOR_PER_TCMB4 = 3.39739477e-14  # one massless neutrino flavour, per K^4
C2OK = 1.62581581e4  # K / eV: amnu = m_nu c^2 / (k_B T_nu0) = mnu * C2OK / Tcmb

# The massive-neutrino density, pressure and pseudo-pressure relative to the
# massless flavour depend on ``y = amnu * a`` alone, so they are tabulated once
# over ``ln y`` and shared by every cosmology. Below the table the neutrinos are
# relativistic and the ratios are 1; that limit also serves ``amnu = 0``.
_LN_Y_MIN, _LN_Y_MAX = math.log(1.0e-6), math.log(1.0e4)
_NEUTRINO_TABLES: dict = {}


def _init_static_splines(n_y_grid=1000, nq=100):
    y_grid = jnp.exp(jnp.linspace(_LN_Y_MIN, _LN_Y_MAX, n_y_grid))
    log_y_grid = jnp.log(y_grid)

    q, w = get_neutrino_momentum_bins(nq)
    y_2d = y_grid[:, None]
    q_2d = q[None, :]
    w_2d = w[None, :]
    v_2d = 1.0 / jnp.sqrt(1.0 + (y_2d / q_2d) ** 2)

    I_rho = jnp.sum(w_2d / v_2d, axis=1)
    I_P = jnp.sum(w_2d * v_2d / 3.0, axis=1)
    I_PP = jnp.sum(w_2d * (v_2d ** 3) / 3.0, axis=1)

    return (
        FastUniformCubicSpline1D(log_y_grid, jnp.log(I_rho)),
        FastUniformCubicSpline1D(log_y_grid, jnp.log(I_P)),
        FastUniformCubicSpline1D(log_y_grid, jnp.log(I_PP)),
    )


def neutrino_tables(n_y_grid: int = 1000, nq: int = 100):
    """``(ln rho, ln P, ln PP)`` splines over ``ln y`` from :func:`_init_static_splines`.

    Built on first use rather than at import, so they take the ``jax_enable_x64``
    setting in force then, and concretely even when first used inside a ``jit``.
    """

    key = (n_y_grid, nq, bool(jax.config.jax_enable_x64))
    tables = _NEUTRINO_TABLES.get(key)
    if tables is None:
        with jax.ensure_compile_time_eval():
            tables = _init_static_splines(n_y_grid, nq)
        _NEUTRINO_TABLES[key] = tables
    return tables


class Species(eqx.Module):
    """A background component; ``rho(a)`` is ``8 pi G rho / c^2`` in 1/Mpc^2 at scalar ``a``."""

    @abstractmethod
    def rho(self, a: jnp.ndarray) -> jnp.ndarray:
        pass


class Photons(Species):
    rho_g: jnp.ndarray

    def __init__(self, Tcmb: jnp.ndarray):
        self.rho_g = GRHOG_PER_TCMB4 * Tcmb**4

    def rho(self, a: jnp.ndarray) -> jnp.ndarray:
        return self.rho_g * (a ** -4)


class MasslessNeutrinos(Species):
    rho_ur: jnp.ndarray

    def __init__(self, Tcmb: jnp.ndarray, Neff: jnp.ndarray):
        self.rho_ur = Neff * GRHOR_PER_TCMB4 * Tcmb**4

    def rho(self, a: jnp.ndarray) -> jnp.ndarray:
        return self.rho_ur * (a ** -4)


class MassiveNeutrinos(Species):
    """``Nmnu`` flavours of mass ``mnu`` (eV); density from the shared ``y`` table."""

    amnu: jnp.ndarray
    prefactor: jnp.ndarray

    def __init__(self, mnu: jnp.ndarray, Tcmb: jnp.ndarray, Nmnu: jnp.ndarray):
        self.amnu = mnu * C2OK / Tcmb
        self.prefactor = Nmnu * GRHOR_PER_TCMB4 * Tcmb**4

    def _ratio(self, table, a):
        """``exp(table(ln y))`` clipped to the table, and 1 for a massless flavour."""
        massive = self.amnu > 0.0
        ln_y = jnp.log(jnp.where(massive, self.amnu, 1.0) * a)
        ln_y = jnp.clip(ln_y, _LN_Y_MIN, _LN_Y_MAX)
        return jnp.where(massive, jnp.exp(table.evaluate(ln_y)), 1.0)

    def rho(self, a: jnp.ndarray) -> jnp.ndarray:
        return self._ratio(neutrino_tables()[0], a) * self.prefactor * (a ** -4)

    def P(self, a: jnp.ndarray) -> jnp.ndarray:
        return self._ratio(neutrino_tables()[1], a) * self.prefactor * (a ** -4)

    def PP(self, a: jnp.ndarray) -> jnp.ndarray:
        return self._ratio(neutrino_tables()[2], a) * self.prefactor * (a ** -4)


class Baryons(Species):
    rho_b: jnp.ndarray

    def __init__(self, Omega_b: jnp.ndarray, H0: jnp.ndarray):
        self.rho_b = Omega_b * GRHOM_PER_H0SQ * H0**2

    def rho(self, a: jnp.ndarray) -> jnp.ndarray:
        return self.rho_b * (a ** -3)


class ColdDarkMatter(Species):
    rho_c: jnp.ndarray

    def __init__(self, Omega_c: jnp.ndarray, H0: jnp.ndarray):
        self.rho_c = Omega_c * GRHOM_PER_H0SQ * H0**2

    def rho(self, a: jnp.ndarray) -> jnp.ndarray:
        return self.rho_c * (a ** -3)


class Curvature(Species):
    """The curvature term of the Friedmann equation, ``grhom Omega_k a^-2``."""

    rho_k: jnp.ndarray

    def __init__(self, Omega_k: jnp.ndarray, H0: jnp.ndarray):
        self.rho_k = Omega_k * GRHOM_PER_H0SQ * H0**2

    def rho(self, a: jnp.ndarray) -> jnp.ndarray:
        return self.rho_k * (a ** -2)


class CPLDarkEnergy(Species):
    rho_de: jnp.ndarray
    w0: jnp.ndarray
    wa: jnp.ndarray

    def __init__(self, Omega_de: jnp.ndarray, w0: jnp.ndarray, wa: jnp.ndarray, H0: jnp.ndarray):
        self.rho_de = Omega_de * GRHOM_PER_H0SQ * H0**2
        self.w0 = w0
        self.wa = wa

    def rho(self, a: jnp.ndarray) -> jnp.ndarray:
        return self.rho_de * a**(-3*(1+self.w0+self.wa)) * jnp.exp(3*(a-1)*self.wa)


class AllSpecies(eqx.Module):
    radiation: tuple[Species, ...] = ()
    matter: tuple[Species, ...] = ()
    other: tuple[Species, ...] = ()

    def radiation_rho(self, a: jnp.ndarray) -> jnp.ndarray:
        return sum(s.rho(a) for s in self.radiation) if self.radiation else jnp.zeros_like(a)

    def matter_rho(self, a: jnp.ndarray) -> jnp.ndarray:
        return sum(s.rho(a) for s in self.matter) if self.matter else jnp.zeros_like(a)

    def other_rho(self, a: jnp.ndarray) -> jnp.ndarray:
        return sum(s.rho(a) for s in self.other) if self.other else jnp.zeros_like(a)

    def total_rho(self, a: jnp.ndarray) -> jnp.ndarray:
        return self.radiation_rho(a) + self.matter_rho(a) + self.other_rho(a)


def build_species(param: dict) -> AllSpecies:
    """The :class:`AllSpecies` of a parameter dictionary.

    Photons, massless and massive neutrinos, baryons, cold dark matter (the rest
    of ``Omegam``), curvature and CPL dark energy. The dark-energy density is
    ``param['OmegaDE']`` when :func:`setup_background_evolution` has closed the
    budget already (the same value the thermodynamics kernel reads), else it is
    closed here at ``a = 1`` from the other species.
    """

    H0, Tcmb = param['H0'], param['Tcmb']
    radiation = (
        Photons(Tcmb=Tcmb),
        MasslessNeutrinos(Tcmb=Tcmb, Neff=param['Neff']),
        MassiveNeutrinos(mnu=param['mnu'], Tcmb=Tcmb, Nmnu=param['Nmnu']),
    )
    matter = (
        Baryons(Omega_b=param['Omegab'], H0=H0),
        ColdDarkMatter(Omega_c=param['Omegam'] - param['Omegab'], H0=H0),
    )
    curvature = Curvature(Omega_k=param['Omegak'], H0=H0)
    Omega_de = param.get('OmegaDE')
    if Omega_de is None:
        grhom = GRHOM_PER_H0SQ * H0**2
        today = sum(s.rho(1.0) for s in radiation + matter) + curvature.rho(1.0)
        Omega_de = 1.0 - today / grhom
    dark_energy = CPLDarkEnergy(Omega_de=Omega_de, w0=param['w_DE_0'], wa=param['w_DE_a'], H0=H0)
    return AllSpecies(radiation=radiation, matter=matter, other=(curvature, dark_energy))


def nu_background( a, amnu, nq : int = 8 ):
    """ computes the neutrino density and pressure of one flavour of massive neutrinos
        in units of the mean density of one flavour of massless neutrinos

    Args:
        a (N,): scale factor
        amnu (B,): neutrino mass in units of neutrino temperature (m_nu*c**2/(k_B*T_nu0).
        nq (int, optional): number of integration points. Defaults to 8.

    Returns:
        tuple[float, float, float]: rho_nu/rho_nu0, p_nu/p_nu0, pp_nu/pp_nu0
    """
    a = a[:, None, None]
    amnu = amnu[None, :, None]

    # q is the comoving momentum in units of k_B*T_nu0/c.
    v    = lambda q: 1 / jnp.sqrt(1 + (a * amnu / q)**2)   # = (1/aq) / sqrt(1+1/aq**2)

    q, w = get_neutrino_momentum_bins( nq )
    q = q[None, None, :]
    w = w[None, None, :]
    rhonu = jnp.sum( w * 1. / v(q) , axis=2)
    pnu = jnp.sum( w * v(q) / 3 , axis=2)
    ppnu = jnp.sum( w * v(q)**3 / 3 , axis=2)

    return rhonu, pnu, ppnu


def dtauda(a, species):
    """``d tau / d a = 1 / (a^2 H)`` at scalar ``a`` (``jax.vmap`` over an array)."""
    Hubble = jnp.sqrt(species.total_rho(a)/3.)
    return 1./(a*a*Hubble)


def get_aprimeoa(*, species, a):
    """The conformal Hubble rate ``a'/a = a H`` at scalar ``a``."""
    return a*jnp.sqrt(species.total_rho(a)/3.)


def compute_angular_diameter_distance( *, aexp, param ):
    """Compute the angular diameter distance

    Args:
        aexp (float): scale factor
        param (dict): dictionary of cosmological parameters

    Returns:
        float: angular diameter distance
    """
    aexpv = jnp.linspace( aexp, 1.0, 1000 )
    aH = jax.vmap(lambda a: get_aprimeoa(species=param['species'], a=a))(aexpv) * aexpv
    Da = aexp * integrate_trapz( 1/aH, aexpv)
    return Da

def setup_background_evolution( *, amin, amax, param, species=None ):
    """Mean densities, the massive-neutrino splines and ``tau(a)`` of ``param``.

    Sets the ``grhom`` / ``grhog`` / ``grhor`` / ``amnu`` coefficients and
    ``OmegaDE`` the device kernels read, the ``log rho_nu`` splines, and the
    conformal time on the ``param['a']`` grid from the expansion rate of the
    :class:`AllSpecies` (``param['species']``; built by :func:`build_species`
    when not given). One cosmology per call: batch with ``jax.vmap``.
    """
    num_neutrino = 512  # number of neutrino history arrays

    param['amin'] = amin
    param['amax'] = amax

    # mean densities
    param['grhom'] = GRHOM_PER_H0SQ * param['H0']**2    # 8πG rho_c / c^2 * in 1/Mpc^2
    param['grhog'] = GRHOG_PER_TCMB4 * param['Tcmb']**4  # photon density in 1/Mpc^2
    param['grhor'] = GRHOR_PER_TCMB4 * param['Tcmb']**4  # neutrino density per flavour in 1/Mpc^2
    param['amnu'] = param['mnu'] * C2OK / param['Tcmb'] # conversion factor for Neutrinos masses (m_nu*c**2/(k_B*T_nu0)

    # Compute the scale factor linearly spaced in log(a)
    a = jnp.geomspace(amin*0.9, amax*1.1, num_neutrino)
    loga = jnp.log(a)
    param['a'] = a

    # Compute the neutrino density and pressure
    rhonu_, pnu_, ppnu_ = nu_background( a, jnp.atleast_1d(param['amnu']) )

    param['logrhonu_of_loga_spline']     = spline_interpolation( loga, jnp.log(rhonu_), uniform=True )
    param['logpnu_of_loga_spline']       = spline_interpolation( loga, jnp.log(pnu_), uniform=True )
    param['logppseudonu_of_loga_spline'] = spline_interpolation( loga, jnp.log(ppnu_), uniform=True )

    # compute the energy density today due to massive neutrinos
    rhonu_today = jnp.reshape(jnp.exp(param['logrhonu_of_loga_spline'].evaluate(0.0)), (-1,))[0]
    param['Omegamnu'] = (param['grhor'] * rhonu_today) / param['grhom']

    # ensure curvature is correct
    Omegar = (param['Neff']+param['Nmnu']*rhonu_today) * param['grhor'] / param['grhom']
    Omegag = param['grhog'] / param['grhom']

    param['OmegaDE'] = 1.0 - param['Omegak'] - Omegar - Omegag - param['Omegam']

    if species is None:
        species = build_species(param)
    param['species'] = species
    param['adotrad'] = jnp.sqrt(species.radiation_rho(amin) * amin**4 / 3.0)

    # Compute the conformal time interval
    param['taumin'] = amin / param['adotrad']
    dtauda_values = jnp.reshape(jax.vmap(dtauda, in_axes=(0, None))(a, species), a.shape)
    integrator = FastUniformCubicSpline1D(loga, dtauda_values * a)
    param['tau'] =  param['taumin'] + jax.vmap(integrator.integral)(loga)
    param['taumax'] = param['tau'][-1]

    return param

def batch_dimensions(param):
  for x in param:
    param[x] = jnp.atleast_1d(param[x])
  return param

@partial(jax.jit, static_argnames=('thermo_module', 'num_thermo', 'rtol', 'atol', 'order', 'class_thermo'))
def evolve_background( *, param, thermo_module = 'RECFAST', num_thermo: int = 256, rtol: float = 1e-5, atol: float = 1e-7, order: int = 5, class_thermo = None ):
    """Evolve the cosmological background and thermal history

    Parameters
    ----------
    param : dict
        Dictionary of cosmological parameters
    thermo_module : str, optional
        Thermal history module to use: 'RECFAST' (default), or 'CLASS' (use
        external CLASS data)
    num_thermo : int, optional
        Number of sampling points for thermal history arrays. Default is 256.
        Uses adaptive sampling that concentrates 50% of points around recombination
        (600 < z < 1400) for optimal accuracy. Validated performance:
        - 256 (default): <0.13% error on P(k), 3.2x faster (adaptive sampling)
        - 512: <0.03% error on P(k), 1.7x faster (adaptive sampling)
        - 1024: reference accuracy, slower
    rtol : float, optional
        Relative tolerance for ODE solvers. Default is 1e-5.
    atol : float, optional
        Absolute tolerance for ODE solvers. Default is 1e-7.
    order : int, optional
        Order of spline interpolation. Default is 5.
    class_thermo : dict, optional
        CLASS thermodynamics data (only used when thermo_module='CLASS')

    Returns
    -------
    param : dict
        Updated parameter dictionary with background evolution results and
        spline interpolations for thermal quantities
    """
    c2ok = 1.62581581e4 # K / eV

    amin = 1e-9
    amax = 1.01

    if thermo_module == 'CLASS':
        amin = jnp.min( class_thermo['scale factor a'] )
        amax = jnp.max( class_thermo['scale factor a'] )
    

    param = setup_background_evolution( amin=amin, amax=amax, param=param )

    if thermo_module == 'RECFAST':
        # Compute the thermal history
        #param, tau, aexp, cs2, Tm, mu, xe, xeHI, xeHeI, xeHeII, xeprime_recfast = evaluate_thermo_recfast( param=param, num_thermo=num_thermo )
        aexp, cs2, Tm, mu, xe, dxeda = evaluate_thermo_recfast(
            param=param,
            num_thermo=num_thermo,
            rtol=rtol,
            atol=atol,
            amin=amin,
            amax=amax,
        )

        param['aexp'] = aexp
        #param['tau'] = tau
        param['xe'] = xe
        #param['xeHI'] = xeHI
        #param['xeHeI'] = xeHeI
        #param['xeHeII'] = xeHeII
        param['cs2'] = cs2
        param['Tm'] = Tm

        # TODO ::  Can be possibly optimized
        tau = spline_interpolation(jnp.log(param['a']), param['tau']).evaluate(jnp.log(aexp))
        param['tau_th'] = tau
        param['tau_of_a_spline']      = spline_interpolation( aexp, tau )
        param['a_of_tau_spline']      = spline_interpolation( tau, aexp )
        param['xe_of_tau_spline']     = spline_interpolation( tau, xe )
        param['cs2a_of_tau_spline']   = spline_interpolation( tau, aexp[:,None]*cs2 )
        param['tempba_of_tau_spline'] = spline_interpolation( tau, aexp[:,None]*Tm )

        # Pre-composed splines for direct a-to-quantity lookups (performance optimization)
        param['xe_of_loga_spline']    = spline_interpolation( jnp.log(aexp), xe , uniform = False)
        param['cs2a_of_loga_spline']  = spline_interpolation( jnp.log(aexp), aexp[:,None]*cs2 , uniform = False)

    elif thermo_module == 'CLASS':
        # use input CLASS thermodynamics
        # interpolating splines for the thermal history        
        param['cs2_of_tau_spline']   = spline_interpolation( class_thermo['conf. time [Mpc]'][::-1], class_thermo['c_b^2'][::-1] )
        param['tempb_of_tau_spline'] = spline_interpolation( class_thermo['conf. time [Mpc]'][::-1], class_thermo['Tb [K]'][::-1] )
        param['xe_of_tau_spline']    = spline_interpolation( class_thermo['conf. time [Mpc]'][::-1], class_thermo['x_e'][::-1] )
        param['a_of_tau_spline']     = spline_interpolation( class_thermo['conf. time [Mpc]'][::-1], class_thermo['scale factor a'][::-1] )
        param['tau_of_a_spline']     = spline_interpolation( class_thermo['scale factor a'][::-1],   class_thermo['conf. time [Mpc]'][::-1] )

        param['aexp'] = class_thermo['scale factor a'][::-1]
        param['tau'] = class_thermo['conf. time [Mpc]'][::-1]

        # Pre-composed splines for direct a-to-quantity lookups (performance optimization)
        aexp_class = class_thermo['scale factor a'][::-1]
        param['xe_of_loga_spline']   = spline_interpolation( jnp.log(aexp_class), class_thermo['x_e'][::-1] )
        param['cs2a_of_loga_spline'] = spline_interpolation( jnp.log(aexp_class), aexp_class * class_thermo['c_b^2'][::-1] )
    


    # compute optical depth and visibility functions
    akthom = 2.3038921003709498e-9 * (1.0 - param['YHe']) * param['Omegab'] * param['H0']**2

    tau_pre_recomb = param['tau_of_a_spline'].evaluate( 1e-4 )
    xe_full   = 1 + param['YHe'] / (1 - param['YHe'])
    xe        = jnp.where( tau <= tau_pre_recomb, xe_full, param['xe_of_tau_spline'].evaluate( tau ) )
    xeprime   = jnp.where( tau <= tau_pre_recomb, 0.0, param['xe_of_tau_spline'].derivative( tau ) )

    # xe = param['xe_of_tau_spline'].evaluate( tau )
    # xeprime = param['xe_of_tau_spline'].derivative( tau )
    # xepprime  = param['xe_of_tau_spline'].derivative2( tau )
    opac       = xe * akthom / aexp[:,None]**2

    opacspline = spline_interpolation( tau, opac, integrate_from_start=True)
    opacprime, opacpprime = opacspline.derivative12( tau )

    optical_depth_today = opacspline.integral( param['tau_of_a_spline'].evaluate( 1.0 ) )
    optical_depth = optical_depth_today - opacspline.integral( tau )

    expmmu    = jnp.exp(-optical_depth)
    vis       = opac * expmmu
    dvis      = (opacprime + opac**2) * expmmu
    ddvis     = (opacpprime + 3 * opac * opacprime + opac**3) * expmmu

    param['optical_depth'] = optical_depth
    param['opac'] = opac
    param['gvis'] = vis
    param['gvisprime'] = dvis
    param['gvispprime'] = ddvis

    param['tau0'] = param['tau_of_a_spline'].evaluate(1.0)
    param['tau_maxvis'] = param['tau'][jnp.argmax(param['gvis'])]

    #if thermo_module == 'RECFAST':
    #    param['xeprime_recf'] = xeprime_recfast

    param['xeprime'] = xeprime

    return param


@jax.jit
def compute_background_quantities(aexp, param):
    """Compute background density and equation of state at given scale factors.

    This function evaluates background cosmological quantities including neutrino
    density, dark energy density, and dark energy equation of state at specified
    scale factors. It uses pre-computed splines from evolve_background().

    Parameters
    ----------
    aexp : jnp.ndarray
        Scale factor values where background quantities are needed
    param : dict
        Parameter dictionary containing:
        - logrhonu_of_loga_spline: Spline for log neutrino density
        - w_DE_0: Dark energy equation of state today
        - w_DE_a: Dark energy equation of state time derivative
        - OmegaDE: Dark energy density parameter

    Returns
    -------
    dict
        Dictionary with keys:
        - 'rhonu': Neutrino density ratio ρ_ν/ρ_ν0 at given scale factors
        - 'rho_Q': Normalized dark energy density ρ_Q(a)/ρ_Q(a=1)
        - 'w_Q': Dark energy equation of state w(a) at given scale factors

    Examples
    --------
    >>> param = evolve_background(param=param, ...)
    >>> aexp_out = jnp.array([0.01, 0.1, 1.0])
    >>> bg_quantities = compute_background_quantities(aexp_out, param)
    >>> rhonu = bg_quantities['rhonu']
    >>> w_Q = bg_quantities['w_Q']

    Notes
    -----
    The dark energy density assumes a parametrization of the form:
        ρ_Q(a) = ρ_Q(1) * a^{-3(1 + w_0 + w_a)} * exp[3(a-1)*w_a]

    The equation of state is:
        w_Q(a) = w_0 + w_a * (1 - a)
    """
    a = jnp.atleast_1d(aexp)

    # Neutrino density ratio from spline
    rhonu = jnp.exp(param['logrhonu_of_loga_spline'].evaluate(jnp.log(a)))

    # Dark energy density (normalized to value at a=1)
    rho_Q = a**(-3 * (1 + param['w_DE_0'] + param['w_DE_a'])) * jnp.exp(3 * (a - 1) * param['w_DE_a'])

    # Dark energy equation of state
    w_Q = param['w_DE_0'] + param['w_DE_a'] * (1.0 - a)

    return {
        'rhonu': rhonu,
        'rho_Q': rho_Q,
        'w_Q': w_Q,
    }
