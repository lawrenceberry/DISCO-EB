from functools import partial
import jax
import jax.numpy as jnp
import diffrax as drx
import equinox as eqx

from jax_cosmo.scipy.integrate import romb

from .thermodynamics_recfast import evaluate_thermo as evaluate_thermo_recfast
from .thermodynamics_mb95 import compute_thermo as compute_thermo_mb95

from .spline_interpolation import spline_interpolation
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


# TODO :: update with values from thermodynamics_recfast.py by putting those into some constants.py
CONST_k_B = 1.3806504e-23
CONST_eV = 1.602176487e-19
CONST_C = 299792458
CONST_G = 6.67428e-11
CONST_neutrino_inst_dec_ratio = (4.0/11.0) ** (1.0/3.0)
CONST_Mpc_to_m = 3.085677581282e22 #CAMB uses 3.085678e22

class Species(eqx.Module):
  @abstractmethod
  def rho(self, a: jnp.ndarray) -> jnp.ndarray:
      pass



class Photons(Species):
  rho_g: jnp.ndarray 
  def __init__(self, Tcmb: jnp.ndarray):
      self.rho_g = jnp.pi**2/15. * (CONST_k_B/CONST_eV * Tcmb)**4
  def rho(self, a: jnp.ndarray) -> jnp.ndarray:
      return self.rho_g[:, None] * (a[None, :] ** -4)
      
class MasslessNeutrinos(Species):
  rho_ur: jnp.ndarray 
  def __init__(self, Tcmb: jnp.ndarray, Neff: jnp.ndarray):
      self.rho_ur = Neff * jnp.pi**2/15. * (CONST_k_B/CONST_eV * Tcmb * CONST_neutrino_inst_dec_ratio)**4
  def rho(self, a: jnp.ndarray) -> jnp.ndarray:
      return self.rho_ur[:, None] * (a[None, :] ** -4)

class MassiveNeutrinos(Species):
    # Physical properties
    amnu: jnp.ndarray
    prefactor: jnp.ndarray

    # Pre-computed static 1D reference tables over dimensionless variable y = amnu * a
    # Important: These are computed ONCE over all cosmologies, since we smartly factorize the integral
    # y = a_mnu * a
    spline_ln_rho: jnp.ndarray = eqx.field(static=True)
    spline_ln_P: jnp.ndarray = eqx.field(static=True)
    spline_ln_PP: jnp.ndarray = eqx.field(static=True)

    def __init__(self, mnu: jnp.ndarray, Tcmb: jnp.ndarray, Nmnu : jnp.ndarray, nq: int = 8, n_y_grid: int = 1000):
        # conversion factor for neutrinos masses to scale factor, a_mnu = (m_nu*c**2/(k_B*T_nu0)
        self.amnu = mnu * CONST_eV / (Tcmb * CONST_neutrino_inst_dec_ratio * CONST_k_B)
        # conversion factor for neutrino relative to absolute densities
        rho_gamma0 = jnp.pi**2/15. * (CONST_k_B/CONST_eV * Tcmb)**4
        self.prefactor = rho_gamma0 * CONST_neutrino_inst_dec_ratio**4 * Nmnu

        # 1. Build log-spaced grid in y = a * amnu 
        # covering ultra-relativistic to non-relativistic regimes (-6 to 4)
        y_grid = jnp.logspace(-6, 4, n_y_grid)
        log_y_grid = jnp.log(y_grid)

        q, w = get_neutrino_momentum_bins(nq)
        y_2d = y_grid[:, None]
        q_2d = q[None, :]
        w_2d = w[None, :]
        v_2d = 1.0 / jnp.sqrt(1.0 + (y_2d / q_2d) ** 2)

        I_rho = jnp.sum(w_2d / v_2d, axis=1)
        I_P = jnp.sum(w_2d * v_2d / 3.0, axis=1)
        I_PP = jnp.sum(w_2d * (v_2d ** 3) / 3.0, axis=1)

        # 4. Store as static spline field
        self.spline_ln_rho = spline_interpolation(log_y_grid, jnp.log(I_rho), uniform=True)
        self.spline_ln_P = spline_interpolation(log_y_grid, jnp.log(I_P), uniform=True)
        self.spline_ln_PP = spline_interpolation(log_y_grid, jnp.log(I_PP), uniform=True)

    def rho(self, a: jnp.ndarray) -> jnp.ndarray:
        """Returns normalized energy density rho_nu / rho_nu0 of shape (N_cosmo, N_a)."""
        ln_y_eval = jnp.log(self.amnu[:, None] * a[None, :])
        return jnp.exp(self.spline_ln_rho.evaluate(ln_y_eval)) * self.prefactor

    def P(self, a: jnp.ndarray) -> jnp.ndarray:
        """Returns normalized energy density rho_nu / rho_nu0 of shape (N_cosmo, N_a)."""
        ln_y_eval = jnp.log(self.amnu[:, None] * a[None, :])
        return jnp.exp(self.spline_ln_P.evaluate(ln_y_eval)) * self.prefactor
        
    def PP(self, a: jnp.ndarray) -> jnp.ndarray:
        """Returns normalized energy density rho_nu / rho_nu0 of shape (N_cosmo, N_a)."""
        ln_y_eval = jnp.log(self.amnu[:, None] * a[None, :])
        return jnp.exp(self.spline_ln_PP.evaluate(ln_y_eval)) * self.prefactor
class Baryons(Species):
    rho_b: jnp.ndarray
    def __init__(self, Omega_b: jnp.ndarray, H0 : jnp.ndarray):
      rho_crit = (3.*(H0 * 1000./CONST_Mpc_to_m)**2)/(8.0 * jnp.pi * CONST_G)
      self.rho_b = Omega_b * rho_crit
    def rho(self, a: jnp.ndarray) -> jnp.ndarray:
        return self.rho_b[:, None] * (a[None, :]  ** -3)
class ColdDarkMatter(Species):
    rho_c: jnp.ndarray
    def __init__(self, Omega_c: jnp.ndarray, H0 : jnp.ndarray):
      rho_crit = (3.*(H0 * 1000./CONST_Mpc_to_m)**2)/(8.0 * jnp.pi * CONST_G)
      self.rho_c = Omega_c * rho_crit
    def rho(self, a: jnp.ndarray) -> jnp.ndarray:
        return self.rho_c[:, None] * (a[None, :]  ** -3)

class CPLDarkEnergy(Species):
    rho_de : jnp.ndarray
    w0 : jnp.ndarray
    wa : jnp.ndarray
    def __init__(self, Omega_de: jnp.ndarray, w0:jnp.ndarray, wa:jnp.ndarray, H0: jnp.ndarray):
      rho_crit = (3.*(H0 * 1000./CONST_Mpc_to_m)**2)/(8.0 * jnp.pi * CONST_G)
      self.rho_de = rho_crit * Omega_de
      self.w0 = w0
      self.wa = wa
    def rho(self, a: jnp.ndarray) -> jnp.ndarray:
        return self.rho_de[:, None] * a[None, :]**(-3*(1+self.w0+self.wa)) * jnp.exp(3*(a[None, :]-1)*self.wa)
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


def dtauda_(a, grhom, grhog, grhor, Omegam, OmegaDE, w_DE_0, w_DE_a, Omegak, Neff, Nmnu, logrhonu_spline):
    """Derivative of conformal time with respect to scale factor"""
    # rhonu = jax.vmap( lambda aa: nu_background(aa,amnu)[0] )( jnp.atleast_1d(a) )
    rhonu = jnp.exp(logrhonu_spline.evaluate(jnp.log(a)))
    rho_DE = a**(-3*(1+w_DE_0+w_DE_a)) * jnp.exp(3*(a-1)*w_DE_a)
    grho2 = grhom * Omegam * a \
        + (grhog + grhor*(Neff+Nmnu*rhonu)) \
        + grhom * OmegaDE * rho_DE * a**4 \
        + grhom * Omegak * a**2
    return jnp.sqrt(3.0 / grho2)

def dadtau(a, param ):
    """Derivative of scale factor with respect to conformal time"""
    rhonu = jnp.exp(param['logrhonu_of_loga_spline'].evaluate(jnp.log(a)))

    a = a[:, None]
    w0 = param['w_DE_0'][None, :]
    wa = param['w_DE_a'][None, :]
    grhom = param['grhom'][None, :]
    grhog = param['grhog'][None, :]
    grhor = param['grhor'][None, :]
    Om = param['Omegam'][None, :]
    Neff = param['Neff'][None, :]
    Nmnu = param['Nmnu'][None, :]
    Ode = param['OmegaDE'][None, :]
    Ok = param['Omegak'][None, :]

    # rhonu = jax.vmap( lambda aa: nu_background(aa,param['amnu'])[0] )( jnp.atleast_1d(a) )
    rho_DE = a**(-3*(1+w0+wa)) * jnp.exp(3*(a-1)*wa)

    grho2 = grhom * Om * a \
        + (grhog + grhor*(Neff+Nmnu*rhonu)) \
        + grhom * Ode * rho_DE * a**4 \
        + grhom * Ok * a**2
    return jnp.sqrt(grho2 / 3.0)


def dtauda(a, param ):
    """Derivative of conformal time with respect to scale factor"""
    return 1/dadtau(a, param)

def dtauda(a, species):
    Hubble = jnp.sqrt(species.total_rho(a)/3.)
    return 1./Hubble


def get_aprimeoa( *, param, aexp ):
    """Compute the conformal Hubble function

    Args:
        param (dict): dictionary of cosmological parameters
        aexp (float, jax.Array): scale factor

    Returns:
        float: conformal H(a)
    """
    rhonu = jnp.exp(param['logrhonu_of_loga_spline'].evaluate(jnp.log(aexp)))
    rho_Q = aexp**(-3*(1+param['w_DE_0']+param['w_DE_a'])) * jnp.exp(3*(aexp-1)*param['w_DE_a'])

    # ... background energy density
    grho = (
        param['grhom'] * param['Omegam'] / aexp
        + (param['grhog'] + param['grhor'] * (param['Neff'] + param['Nmnu'] * rhonu)) / aexp**2
        + param['grhom'] * param['OmegaDE'] * rho_Q * aexp**2
        + param['grhom'] * param['Omegak']
    )

    aprimeoa = jnp.sqrt(grho / 3.0)
    return aprimeoa
def get_aprimeoa(*, species, a):
    return jnp.sqrt(species.total_rho(a)/3.)

def compute_angular_diameter_distance( *, aexp, param ):
    """Compute the angular diameter distance

    Args:
        aexp (float): scale factor
        param (dict): dictionary of cosmological parameters

    Returns:
        float: angular diameter distance
    """
    aexpv = jnp.linspace( aexp, 1.0, 1000 )
    aH = get_aprimeoa( param=param, aexp=aexpv ) * aexpv
    Da = aexp * integrate_trapz( 1/aH, aexpv)
    return Da

def setup_background_evolution( *, amin, amax, species, param ):
    c2ok = 1.62581581e4 # K / eV
    num_neutrino = 512  # number of neutrino history arrays

    param['amin'] = amin
    param['amax'] = amax
    
    
    param['species'] = species

    # mean densities
    #Omegak = 0.0 #1.0 - Omegam - OmegaL
    #param['grhom'] = 3.33795017e-11 * param['H0']**2    # 8πG rho_c / c^2 * in 1/Mpc^2
    #param['grhog'] = 1.49594245e-13 * param['Tcmb']**4  # photon density in 1/Mpc^2
    #param['grhor'] = 3.39739477e-14 * param['Tcmb']**4  # neutrino density per flavour in 1/Mpc^2
    # param['adotrad'] = jnp.sqrt((param['grhog']+param['grhor']*(param['Neff']+param['Nmnu'])) / 3.0)
    # param['adotrad'] = 2.8948e-7 * param['Tcmb']**2 # Hubble during radiation domination
    atest = jnp.array([amin])
    param['adotrad'] = jnp.sqrt(species.radiation_rho(atest)/atest**(-4)/3.)

    #param['amnu'] = param['mnu'] * c2ok / param['Tcmb'] # conversion factor for Neutrinos masses (m_nu*c**2/(k_B*T_nu0)

    # Compute the scale factor linearly spaced in log(a)
    a = jnp.geomspace(amin*0.9, amax*1.1, num_neutrino)
    loga = jnp.log(a)
    param['a'] = a

    # Compute the neutrino density and pressure
    #rhonu_, pnu_, ppnu_ = nu_background( a, param['amnu'] )

    #param['logrhonu_of_loga_spline']     = spline_interpolation( loga, jnp.log(rhonu_), uniform=True )
    #param['logpnu_of_loga_spline']       = spline_interpolation( loga, jnp.log(pnu_), uniform=True )
    #param['logppseudonu_of_loga_spline'] = spline_interpolation( loga, jnp.log(ppnu_), uniform=True )

    # compute the energy density today due to massive neutrinos
    #rhonu = jnp.exp(param['logrhonu_of_loga_spline'].evaluate(0.0))
    #Omegamnu = (param['grhor'] * rhonu) / param['grhom']
    #param['Omegamnu'] = Omegamnu

    # ensure curvature is correct
    #Omegar = (param['Neff']+param['Nmnu']*jnp.exp(param['logrhonu_of_loga_spline'].evaluate(0.0)[0])) * param['grhor'] / param['grhom']
    #Omegag = param['grhog'] / param['grhom']

    #param['OmegaDE'] = 1.0 - param['Omegak'] - Omegar - Omegag - param['Omegam']


    # Compute the conformal time interval
    param['taumin'] = amin / param['adotrad']
    #integrator = spline_interpolation(loga, dtauda(a,param) * a[:, None], uniform=True)
    integrator = spline_interpolation(loga, (dtauda(a,species) * a[None, :]).T, uniform=True)
    param['tau'] =  param['taumin'] + integrator.integral(loga)
    param['taumax'] = param['tau'][-1]
    #param['taumax'] = param['taumin'] + integrator.integral(amax)[0] - integrator.integral(amin)[0]

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
        Thermal history module to use: 'RECFAST' (default, high accuracy),
        'MB95' (faster, approximate), or 'CLASS' (use external CLASS data)
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
    
    param = batch_dimensions(param)
    
    photons = Photons(Tcmb=param['Tcmb'])
    massless_neutrinos = MasslessNeutrinos(Neff=param['Neff'],Tcmb=param['Tcmb']) # TODO :: rename, Neff is typically the total, not just the massless contribution
    massive_neutrinos = MassiveNeutrinos(mnu=param['mnu'],Tcmb=param['Tcmb'],Nmnu=param['Nmnu'])
    baryons = Baryons(Omega_b=param['Omegab'], H0=param['H0'])
    cdm = ColdDarkMatter(Omega_c=param['Omegam']-param['Omegab'], H0=param['H0']) # TODO :: extend to general matter, not just CDM + baryons, which is anyways wrong, due to the preference of massive neutrinos
    w0wa_de = CPLDarkEnergy(
        Omega_de=1-param['Omegam'], # TODO :: do properly with budget equation
        w0=param['w_DE_0'],wa=param['w_DE_a'],H0=param['H0']
    )

    species = AllSpecies(
        radiation=(photons, massless_neutrinos, massive_neutrinos),
        matter=(baryons, cdm),
        other=(w0wa_de,)
    )

    param = setup_background_evolution( amin=amin, amax=amax, species=species, param=param )

    if thermo_module == 'RECFAST':
        # Compute the thermal history
        #param, tau, aexp, cs2, Tm, mu, xe, xeHI, xeHeI, xeHeII, xeprime_recfast = evaluate_thermo_recfast( param=param, num_thermo=num_thermo )
        aexp, cs2, Tm, mu, xe, dxeda = evaluate_thermo_recfast(
            param=param,
            num_thermo=num_thermo,
            rtol=rtol,
            atol=atol,
        )

        param['aexp'] = aexp
        #param['tau'] = tau
        param['xe'] = xe
        #param['xeHI'] = xeHI
        #param['xeHeI'] = xeHeI
        #param['xeHeII'] = xeHeII
        param['cs2'] = cs2
        param['Tm'] = Tm

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

    elif thermo_module == 'MB95':

        # Compute the thermal history
        th, param = compute_thermo_mb95( param=param, nthermo=num_thermo )

        xe = th['xe']
        xeHI = th['xHII']
        xeHeI = th['xHeII']
        xeHeII = th['xHeIII']
        aexp = th['a']
        tau = th['tau']
        cs2 = th['cs2']
        Tm = th['tb']

        param['xe'] = xe
        param['xeHI'] = xeHI
        param['xeHeI'] = xeHeI
        param['xeHeII'] = xeHeII
        param['aexp'] = aexp
        param['tau'] = tau

        param['xe_of_tau_spline']     = spline_interpolation( tau, xe )
        param['cs2a_of_tau_spline']   = spline_interpolation( tau, aexp*cs2 )
        param['tempba_of_tau_spline'] = spline_interpolation( tau, aexp*Tm )
        param['tau_of_a_spline'] = spline_interpolation( aexp, tau )
        param['a_of_tau_spline'] = spline_interpolation( tau, aexp )

        # Pre-composed splines for direct a-to-quantity lookups (performance optimization)
        param['xe_of_loga_spline']    = spline_interpolation( jnp.log(aexp), xe , uniform = True)
        param['cs2a_of_loga_spline']  = spline_interpolation( jnp.log(aexp), aexp*cs2 , uniform = True)

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
