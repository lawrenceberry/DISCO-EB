import math
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy import integrate, interpolate

from .util import lngamma_complex_e, root_find_bisect, root_find_bisect_nocond, savgol_filter

import diffrax as drx
from jaxtyping import Array, PyTree, Scalar
import equinox as eqx

from functools import partial
import jax.flatten_util as fu

from .ode_integrators_stiff import Rodas5, Rodas5Transformed, Rodas5Batched
from diffrax import Kvaerno5
from .approximations import (
    compute_fields_rsa,
    in_tca_regime,
    in_rsa_regime,
    in_ur_fluid_regime,
    compute_shearprime_ufa,
    apply_rsa_state_projection,
    get_approximation_settings,
)

# Import background functions
from .background import dtauda, evolve_background, get_aprimeoa, get_neutrino_momentum_bins



def nu_perturb( a : float, amnu: float, psi0: jax.Array, psi1 : jax.Array, psi2 : jax.Array, nqmax : int ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """ Compute the perturbations of density, energy flux, pressure, and
        shear stress of one flavor of massive neutrinos, in units of the mean
        density of one flavor of massless neutrinos, by integrating over 
        momentum.

    Args:
        a (float): scale factor
        amnu (float): neutrino mass in units of neutrino temperature (m_nu*c**2/(k_B*T_nu0).
        psi0 (jax.Array): l=0 neutrino perturbations for all momentum bins
        psi1 (jax.Array): l=1 neutrino perturbations for all momentum bins
        psi2 (jax.Array): l=2 neutrino perturbations for all momentum bins
        nq (int, optional): _description_. Defaults to 1000.
        nqmax (int): number of momentum bins

    Returns:
        _type_: drhonu, dpnu, fnu, shearnu
    """
    
    q, w = get_neutrino_momentum_bins( nqmax )
    #q = q[:, None]
    #w = w[:, None]
    aq   = a * amnu / q
    v    = 1 / jnp.sqrt(1 + aq**2)

    drhonu  = jnp.sum(w * psi0 / v)
    dpnu    = jnp.sum(w * psi0 * v) / 3
    fnu     = jnp.sum(w * psi1) 
    shearnu = jnp.sum(w * psi2 * v) * 2 / 3

    return drhonu, dpnu, fnu, shearnu

def nu_perturb_prime( *, a : float, amnu : float, aprimeoa : float, psi0: jax.Array, psi2: jax.Array, psi0prime : jax.Array, psi2prime : jax.Array, nqmax : int ) -> tuple[jax.Array, jax.Array]:
    """ Compute the time derivative of the mean density in massive neutrinos 
          and the shear perturbation.

    Args:
        a (float): scale factor
        aprimeoa (float): conformal Hubble rate
        amnu (float): neutrino mass in units of neutrino temperature (m_nu*c**2/(k_B*T_nu0).
        psi0 (jax.Array): l=0 neutrino perturbations for all momentum bins
        psi2 (jax.Array): l=2 neutrino perturbations for all momentum bins
        psi0prime (jax.Array): time derivative of l=0 neutrino perturbations for all momentum bins
        psi2prime (jax.Array): time derivative of l=2 neutrino perturbations for all momentum bins
        nqmax (int): number of momentum bins

    Returns:
        _type_: rho_nu_prime, shear_nu_prime
    """
    
    q, w   = get_neutrino_momentum_bins( nqmax )
    #q = q[:, None]
    #w = w[:, None]
    aq     = a * amnu / q
    v      = 1 / jnp.sqrt(1 + aq**2)
    vprime = -0.5*aq*aprimeoa * v**3

    rho_nu_prime   = jnp.sum(w * (psi0prime / v - psi0 / v**2 * vprime))
    shear_nu_prime = jnp.sum(w * (psi2prime*v + psi2*vprime)) * 2 / 3

    return rho_nu_prime, shear_nu_prime


def model_synchronous(*, tau, y, param, kmode, lmaxg, lmaxgp, lmaxr, lmaxnu, nqmax, idx):     
    """Solve the synchronous gauge perturbation equations for a single mode.

    Parameters
    ----------
    tau : float
        conformal time
    yin : array_like
        input vector of perturbations
    param : array_like
        dictionary of parameters and interpolated background functions
    kmode : float
        wavenumber of modef
    lmaxg : int
        maximum photon temperature hierarchy multipole
    lmaxgp : int
        maximum photon polarization hierarchy multipole
    lmaxr : int
        maximum massless neutrino hierarchy multipole
    lmaxnu : int
        maximum neutrino hierarchy multipole
    nqmax : int
        maximum number of momentum bins for massive neutrinos

    Returns
    -------
    f : array_like
        RHS of perturbation equations
    """
    #print("ODE input", y.shape, idx)
    def to_scalar(x):
        return jnp.ravel(x)[0]
    def take_idx(x):
      return jnp.take(x, idx, mode="wrap")

    rsa_settings = get_approximation_settings(param)
    use_tca = to_scalar(rsa_settings['use_tca'])
    tca_tau_c_over_tau_h_trigger = to_scalar(rsa_settings['tca_tau_c_over_tau_h_trigger'])
    tca_tau_c_over_tau_k_trigger = to_scalar(rsa_settings['tca_tau_c_over_tau_k_trigger'])
    use_rsa = to_scalar(rsa_settings['use_rsa'])
    tau_c_over_tau_trigger = to_scalar(rsa_settings['tau_c_over_tau_trigger'])
    tau_over_tau_k_trigger = to_scalar(rsa_settings['tau_over_tau_k_trigger'])
    use_ur_fluid = to_scalar(rsa_settings['use_ur_fluid'])
    ur_fluid_tau_over_tau_k_trigger = to_scalar(rsa_settings['ur_fluid_tau_over_tau_k_trigger'])
    #jax.debug.print("TCA = {tca}. RSA = {rsa}, UR = {ur}", tca=use_tca, rsa=use_rsa, ur=use_ur_fluid)

    y = jnp.ravel(y)
    tau = to_scalar(tau)
    kmode = to_scalar(kmode)
    Omegac = take_idx(param['Omegam'] - param['Omegab'])

    iq0 = 10 + lmaxg + lmaxgp + lmaxr
    iq1 = iq0 + nqmax
    iq2 = iq1 + nqmax
    iq3 = iq2 + nqmax
    iq4 = iq3 + nqmax

    # y = jnp.copy(yin)
    f = jnp.zeros_like( y )

    #TODO: add curvature
    # ... curvature
    K = 0
    tau = tau + 1e-50
    
    # def cotKgen_zero_curv():
    #     return 1.0/(kmode*tau)
    # def cotKgen_pos_curv():
    #     return jnp.sqrt(K)/kmode/jnp.tan(jnp.sqrt(K)*tau)
    # def cotKgen_neg_curv():
    #     return jnp.sqrt(-K)/kmode/jnp.tanh(jnp.sqrt(-K)*tau)
    
    # cotKgen = jax.lax.switch(int(1+jax.lax.sign(K)), [cotKgen_neg_curv, cotKgen_zero_curv, cotKgen_pos_curv])
    s2_squared = 1.-3.*K/kmode**2
    s_l2 = 1.0
    s_l3 = 1.0

    # ... metric
    a = y[0]
    a = jnp.where(a<1e-8,1e-8,a)
    a = jnp.where(a>1.1,1.1,a)
    loga = jnp.log(a)

    #ahprime = y[1]
    eta = y[2]

    # ... cdm
    deltac = y[3]
    thetac = y[4]

    # ... baryons
    deltab = y[5]
    thetab = y[6]

    # ... photons
    deltag = y[7]
    thetag = y[8]
    shearg = y[9] / 2.0

    # ... massless neutrinos
    deltar = y[ 9 + lmaxg + lmaxgp]
    thetar = y[10 + lmaxg + lmaxgp]
    shearr = y[11 + lmaxg + lmaxgp] / 2.0

    # ... quintessence field
    deltaq = y[-2]
    thetaq = y[-1]

    # ... evaluate thermodynamics
    # tempb   = param['tempba_of_tau_spline'].evaluate( tau ) / a
    # xeprime = param['xe_of_tau_spline'].derivative( tau )
    # cs2     = param['cs2a_of_tau_spline'].evaluate( tau ) / a
    # xe      = param['xe_of_tau_spline'].evaluate( tau )

    # Use pre-composed splines for direct log(a) lookup (performance optimization)
    cs2     = take_idx(param['cs2a_of_tau_spline'].evaluate( tau )) / a
    #xe      = take_idx(param['xe_of_loga_spline'].evaluate( loga ))
    xe      = take_idx(param['xe_of_tau_spline'].evaluate( tau ))
    
    #jax.debug.print("XE = {xe} | {xe2} (tau={tau},a={a})",xe=param['xe_of_loga_spline'].evaluate( loga ), xe2=param['xe_of_tau_spline'].evaluate( tau ), tau=tau, a=a)
    
    # ... Photon mass density over baryon mass density
    photbar = take_idx( param['grhog'] / (param['grhom'] * param['Omegab'] * a))
    pb43 = 4.0 / 3.0 * photbar

    # massive neutrinos
    rhonu = jnp.exp(take_idx(param['logrhonu_of_loga_spline'].evaluate(loga)))
    pnu = jnp.exp(take_idx(param['logpnu_of_loga_spline'].evaluate(loga)))

    # ... quintessence
    cs2_Q              = take_idx(param['cs2_DE'])
    w_Q                = take_idx(param['w_DE_0'] + param['w_DE_a'] * (1.0 - a) )
    rho_Q              = take_idx(a**(-3*(1+param['w_DE_0']+param['w_DE_a'])) * jnp.exp(3*(a-1)*param['w_DE_a']))
    rho_plus_p_theta_Q = take_idx((1+w_Q) * rho_Q * param['grhom'] * param['OmegaDE'] * thetaq * a**2)
    
    # ... homogeneous background
    # grho = (
    #     param['grhom'] * param['Omegam'] / a
    #     + (param['grhog'] + param['grhor'] * (param['Neff'] + param['Nmnu'] * rhonu)) / a**2
    #     + param['grhom'] * param['OmegaDE'] * rho_Q * a**2
    #     + param['grhom'] * param['Omegak']
    # )

    # gpres = (
    #     (param['grhog'] + param['grhor'] * param['Neff']) / 3.0 + param['grhor'] * param['Nmnu'] * pnu
    # ) / a**2 + w_Q * param['grhom'] * param['OmegaDE'] * rho_Q * a**2

    # ... compute expansion rate
    aprimeoa = take_idx(get_aprimeoa( param=param, aexp=a ))
    xeprime = take_idx(param['xe_of_loga_spline'].derivative( loga ) * aprimeoa)
    gpres = take_idx((
        (param['grhog'] + param['grhor'] * param['Neff']) / 3.0 + param['grhor'] * param['Nmnu'] * pnu
    ) / a**2 + w_Q * param['grhom'] * param['OmegaDE'] * rho_Q * a**2)
    aprimeprimeoa = take_idx(0.5 * (aprimeoa**2 - gpres))
    # aprimeoa = jnp.sqrt(grho / 3.0)                # Friedmann I
    # aprimeprimeoa = 0.5 * (aprimeoa**2 - gpres)    # Friedmann II

    # quintessence EOS time derivatives
    w_Q_prime = take_idx( -param['w_DE_a'] * aprimeoa * a)
    ca2_Q     = w_Q - w_Q_prime / 3 / ((1+w_Q)+1e-6) / aprimeoa

    # ... Thomson opacity coefficient
    akthom = take_idx(2.3038921003709498e-9 * (1.0 - param['YHe']) * param['Omegab'] * param['H0']**2)

    # ... Thomson opacity
    opac    = take_idx(xe * akthom / a**2)
    opac = jnp.clip(opac, 0, 1e3)
    #jax.lax.cond(kmode>0.001548,
    #  lambda :jax.debug.print("OPAC = {opac} (a={a}, xe={xe}, akthom={akthom})", a=a, xe=take_idx(xe), akthom=akthom, opac=opac),
    #  lambda:None
    #  )
    tauc    = take_idx(1. / jnp.maximum(opac, 1e-30))
    taucprime = take_idx(tauc * (2 * aprimeoa - xeprime / jnp.maximum(xe, 1e-30)))
    tauh = take_idx(1.0 / jnp.maximum(aprimeoa, 1e-30))
    tauk = take_idx(1.0 / jnp.maximum(kmode, 1e-30))

    do_relativistic_sa = jnp.logical_and(
        jnp.asarray(use_rsa),
        in_rsa_regime(
            tau=tau,
            kmode=kmode,
            tau_c=tauc,
            tau_c_over_tau_trigger=tau_c_over_tau_trigger,
            tau_over_tau_k_trigger=tau_over_tau_k_trigger,
        ),
    )
    do_ur_fluid = jnp.logical_and(
        jnp.logical_not(do_relativistic_sa),
        jnp.logical_and(
            jnp.asarray(use_ur_fluid),
            in_ur_fluid_regime(
                tau=tau,
                kmode=kmode,
                tau_over_tau_k_trigger=ur_fluid_tau_over_tau_k_trigger,
            ),
        ),
    )
    do_tca = jnp.logical_and(
        jnp.logical_not(do_relativistic_sa),
        jnp.logical_and(
            jnp.asarray(use_tca),
            in_tca_regime(
                tau_h=tauh,
                tau_k=tauk,
                tau_c=tauc,
                tau_c_over_tau_h_trigger=tca_tau_c_over_tau_h_trigger,
                tau_c_over_tau_k_trigger=tca_tau_c_over_tau_k_trigger,
            ),
        ),
    )

    #tauc    = 1. / opac
    #taucprime = tauc * (2*aprimeoa - xeprime/xe)
    #F       = tauc / (1+pb43) #CLASS perturbations.c:10072
    #Fprime  = taucprime/(1+pb43) + tauc*pb43*aprimeoa/(1+pb43)**2 #CLASS perturbations.c:10074

    
    # ... background scale factor evolution
    #print(f.shape, (aprimeoa*a).shape)
    f = f.at[0].set( aprimeoa * a )
    
    # ... evaluate metric perturbations
    drhonu, dpnu, fnu, shearnu = nu_perturb( a, take_idx(param['amnu']), y[iq0:iq1], y[iq1:iq2], y[iq2:iq3], nqmax=nqmax )

    dgrho = (
        take_idx(param['grhom']) * (Omegac * deltac + take_idx(param['Omegab']) * deltab) / a
        + (take_idx(param['grhog']) * deltag + take_idx(param['grhor']) * (take_idx(param['Neff']) * deltar + take_idx(param['Nmnu']) * drhonu)) / a**2
        + take_idx(param['grhom'] * param['OmegaDE']) * deltaq * rho_Q * a**2
    )

    # RSA metric estimate without relativistic species (BLT11-inspired closure)
    dgrho_wo_rel = take_idx(
        param['grhom'] * (Omegac * deltac + param['Omegab'] * deltab) / a
        + param['grhor'] * param['Nmnu'] * drhonu / a**2
        + param['grhom'] * param['OmegaDE'] * deltaq * rho_Q * a**2
    )
    hprime_rsa = to_scalar((2.0 * kmode**2 * eta + dgrho_wo_rel) / jnp.maximum(aprimeoa, 1e-30))
    deltag_rsa, thetag_rsa, shearg_rsa, deltar_rsa, thetar_rsa, shearr_rsa = compute_fields_rsa(
        kmode=kmode,
        aprimeoa=aprimeoa,
        hprime=hprime_rsa,
        eta=eta,
        deltab=deltab,
        thetab=thetab,
        cs2_b=cs2,
        tau_c=tauc,
        tau_c_prime=taucprime,
    )

    deltag_eff = jnp.where(do_relativistic_sa, deltag_rsa, deltag)
    thetag_eff = jnp.where(do_relativistic_sa, thetag_rsa, thetag)
    shearg_eff = jnp.where(do_relativistic_sa, shearg_rsa, shearg)
    deltar_eff = jnp.where(do_relativistic_sa, deltar_rsa, deltar)
    thetar_eff = jnp.where(do_relativistic_sa, thetar_rsa, thetar)
    shearr_eff = jnp.where(do_relativistic_sa, shearr_rsa, shearr)

    dgpres = take_idx(
        (param['grhog'] * deltag_eff + param['grhor'] * param['Neff'] * deltar_eff) / a**2 / 3.0 
        + param['grhor'] * param['Nmnu'] * dpnu / a**2 
        + (cs2_Q * param['grhom'] * param['OmegaDE'] * deltaq * rho_Q * a**2 + (cs2_Q-ca2_Q)*(3*aprimeoa * rho_plus_p_theta_Q / kmode**2))
    )
    dgtheta = take_idx(
        param['grhom'] * (Omegac * thetac + param['Omegab'] * thetab) / a
        + 4.0 / 3.0 * (param['grhog'] * thetag_eff + param['Neff'] * param['grhor'] * thetar_eff) / a**2
        + param['Nmnu'] * param['grhor'] * kmode * fnu / a**2
        + rho_plus_p_theta_Q
    )
    dgshear = take_idx(
        4.0 / 3.0 * (param['grhog'] * shearg_eff + param['Neff'] * param['grhor'] * shearr_eff) / a**2
        + param['Nmnu'] * param['grhor'] * shearnu / a**2
    )

    dahprimedtau = -(dgrho + 3.0 * dgpres) * a
    

    f = f.at[1].set( dahprimedtau )
    

    # ... force energy conservation
    hprime_full = to_scalar((2.0 * kmode**2 * eta + dgrho) / aprimeoa)
    hprime = to_scalar(jnp.where(do_relativistic_sa, hprime_rsa, hprime_full))

    etaprime = to_scalar(0.5 * dgtheta / kmode**2)
    alpha  = to_scalar((hprime + 6.*etaprime)/2./kmode**2)
    f = f.at[2].set( etaprime )
    
    # alphaprime = -3*dgshear/(2*kmode**2) + eta - 2*aprimeoa*alpha
    # alphaprime -=  9/2 * a**2/kmode**2 * 4/3 * 16/45/opac * (thetag+kmode**2*alpha) * param['grhog']

    # ... cdm equations of motion, MB95 eq. (42)
    deltacprime = -thetac - 0.5 * hprime
    f = f.at[3].set( deltacprime )
    thetacprime = -aprimeoa * thetac  # thetac = 0 in synchronous gauge!
    f = f.at[4].set( thetacprime )

    idxb = 5
    # --- baryon equations of motion, MB95 eqs. (66) ---------------------------------------------
    # ... baryon density, BLT11 eq. (2.1a)
    deltabprime = -thetab - 0.5 * hprime
    f = f.at[idxb+0].set( deltabprime )
    # ... baryon velocity, BLT11 eq. (2.1b)

    thetabprime = -aprimeoa * thetab + kmode**2 * cs2 * deltab \
                + pb43 * opac * (thetag_eff - thetab)
    thetabprime = jnp.where(thetabprime > 1e10, 1e10, thetabprime)
    f = f.at[idxb+1].set( thetabprime )

    # --- photon equations of motion, MB95 eqs. (63) ---------------------------------------------
    idxg  = 7
    idxgp = 7 + (lmaxg+1)


    def _photon_hierarchy(f_in):
        # ... polarization term
        polter = y[idxg+2] + y[idxgp+0] + y[idxgp+2]
        # ... photon density, BLT11 eq. (2.4a)
        deltagprime = 4.0 / 3.0 * (-thetag_eff - 0.5 * hprime)
        f_in = f_in.at[idxg+0].set( deltagprime )
        # ... photon velocity, BLT11 eq. (2.4b)
        thetagprime = kmode**2 * (0.25 * deltag_eff - s2_squared * shearg_eff) \
                    - opac * (thetag_eff - thetab)
        f_in = f_in.at[idxg+1].set( thetagprime )
        # ... photon shear, BLT11 eq. (2.4c)
        sheargprime = 8./15. * (thetag_eff+kmode**2*alpha) -3/5*kmode*s_l3/s_l2*y[idxg+3] \
                    - opac*(y[idxg+2]-0.1*s_l2*polter)
        sheargprime = jnp.where(jnp.abs(sheargprime)>1e10, 1e10, sheargprime)
        f_in = f_in.at[idxg+2].set( sheargprime )

        #... photon temperature l>=3, BLT11 eq. (2.4d)
        ell  = jnp.arange(3, lmaxg )
        res = kmode  / (2 * ell + 1) * (ell * y[idxg+ell-1] - (ell + 1) * y[idxg+ell+1]) - opac * y[idxg+ell]
        res = jnp.where(jnp.abs(res) > 1e10, 1e10, res)
        f_in = f_in.at[idxg+ell].set( res )
        # photon temperature hierarchy truncation, BLT11 eq. (2.5)
        f_in = f_in.at[idxg+lmaxg].set( kmode * y[idxg+lmaxg-1] - (lmaxg + 1) / tau * y[idxg+lmaxg] - opac * y[idxg+lmaxg] )

        #... polarization equations, BLT11 eq. (2.4e)
        ell  = jnp.arange(0, lmaxgp) # l=0...lmaxgp-1
        f_in = f_in.at[idxgp+ell].set( kmode  / (2 * ell + 1) * (ell * y[idxgp+ell-1] - (ell + 1) * y[idxgp+ell+1]) - opac * y[idxgp+ell] )
        f_in = f_in.at[idxgp+0].add( opac * polter / 2 )  # photon polarization l=0
        f_in = f_in.at[idxgp+2].add( opac * polter / 10 ) # photon polarization l=2

        # photon polarization hierarchy truncation
        f_in = f_in.at[idxgp+lmaxgp].set( kmode * y[idxgp+lmaxgp-1] - (lmaxgp + 1) / tau * y[idxgp+lmaxgp] - opac * y[idxgp+lmaxgp] )
        return f_in

    def _photon_tca(f_in):
        deltabprime_tca = -thetab - 0.5 * hprime
        thetabprime_tca = (
            -aprimeoa * thetab + kmode**2 * cs2 * deltab + kmode**2 * pb43 * (0.25 * deltag - s2_squared * shearg)
        ) / (1.0 + pb43)

        deltagprime_tca = 4.0 / 3.0 * (-thetag - 0.5 * hprime)
        slip = (
            2.0 * pb43 / (1.0 + pb43) * aprimeoa * (thetab - thetag)
            + tauc
            * (
                -aprimeprimeoa * thetab
                - 0.5 * aprimeoa * kmode**2 * deltag
                + kmode**2 * (cs2 * deltabprime_tca - 0.25 * deltagprime_tca)
            )
            / (1.0 + pb43)
        )
        thetabprime_tca = thetabprime_tca + pb43 / (1.0 + pb43) * slip
        thetagprime_tca = (-thetabprime_tca - aprimeoa * thetab + kmode**2 * cs2 * deltab) / jnp.maximum(pb43, 1e-30) + kmode**2 * (
            0.25 * deltag - s2_squared * shearg
        )

        f_in = f_in.at[idxb + 0].set(deltabprime_tca)
        f_in = f_in.at[idxb + 1].set(thetabprime_tca)
        f_in = f_in.at[idxg + 0].set(deltagprime_tca)
        f_in = f_in.at[idxg + 1].set(thetagprime_tca)
        f_in = f_in.at[idxg + 2:idxg + lmaxg + 1].set(0.0)
        f_in = f_in.at[idxgp:idxgp + lmaxgp + 1].set(0.0)
        return f_in

    def _photon_rsa(f_in):
        return f_in.at[idxg:idxg + lmaxg + 1].set(0.0).at[idxgp:idxgp + lmaxgp + 1].set(0.0)

    f = jax.lax.cond(
        do_relativistic_sa,
        _photon_rsa,
        lambda f_in: jax.lax.cond(do_tca, _photon_tca, _photon_hierarchy, f_in),
        f,
    )

    # --- Massless neutrino equations of motion -------------------------------------------------------
    idxr = 9 + lmaxg + lmaxgp
    deltarprime = 4.0 / 3.0 * (-thetar_eff - 0.5 * hprime)
    f = f.at[idxr+0].set( deltarprime )
    thetarprime = kmode**2 * (0.25 * deltar_eff - shearr_eff)
    f = f.at[idxr+1].set( thetarprime )

    def _massless_nu_hierarchy(f_in):
        shearrprime = 8./15. * (thetar_eff + kmode**2 * alpha) - 0.6 * kmode * y[idxr+3]
        f_in = f_in.at[idxr+2].set( shearrprime )
        ell = jnp.arange(3, lmaxr)
        f_in = f_in.at[idxr+ell].set( kmode / (2 * ell + 1) * (ell * y[idxr+ell-1] - (ell + 1) * y[idxr+ell+1]) )

        # ... truncate moment expansion
        f_in = f_in.at[idxr+lmaxr].set( kmode * y[idxr+lmaxr-1] - (lmaxr + 1) / tau * y[idxr+lmaxr] )
        return f_in

    def _massless_nu_ufa(f_in):
        shearrprime = compute_shearprime_ufa(
            tau=tau,
            shearr=shearr_eff,
            thetar=thetar_eff,
            hprime=hprime,
        )
        f_in = f_in.at[idxr+2].set( shearrprime )
        return f_in.at[idxr+3:idxr + lmaxr + 1].set(0.0)

    def _massless_nu_rsa(f_in):
        return f_in.at[idxr:idxr + lmaxr + 1].set(0.0)

    f = jax.lax.cond(
        do_relativistic_sa,
        _massless_nu_rsa,
        lambda f_in: jax.lax.cond(do_ur_fluid, _massless_nu_ufa, _massless_nu_hierarchy, f_in),
        f,
    )

    # --- Massive neutrino equations of motion --------------------------------------------------------
    q, _ = get_neutrino_momentum_bins( nqmax )
    #q = q[:, None]
    aq = a * take_idx(param['amnu']) / q

    v = 1 / jnp.sqrt(1 + aq**2)
    dlfdlq = -q / (1.0 + jnp.exp(-q))  # derivative of the Fermi-Dirac distribution


    f = f.at[iq0 : iq1].set(
        -kmode * v * y[iq1 : iq2] + hprime* dlfdlq / 6.0 
    )

    f = f.at[iq1 : iq2].set(
        kmode * v * (y[iq0 : iq1] - 2.0 * y[iq2 : iq3]) / 3.0
    )

    f = f.at[iq2 : iq3].set(
        kmode * v * (2 * y[iq1 : iq2] - 3 * y[iq3 : iq4]) / 5.0 - (hprime / 15 + 2 / 5 * etaprime) * dlfdlq
    )

    ell = jnp.arange(3, lmaxnu)
    vv = jnp.tile(v, lmaxnu - 3)
    denl = jnp.repeat( 2*ell+1, nqmax)
    

    f = f.at[iq0 + 3 * nqmax : iq0 + lmaxnu * nqmax].set(
        kmode * vv / denl * (
            jnp.repeat( ell, nqmax) * y[iq0 + 2*nqmax : iq0 + (lmaxnu-1)*nqmax] 
            - jnp.repeat( ell+1, nqmax) * y[iq0 + 4*nqmax : iq0 + (lmaxnu+1)*nqmax]
        ) 
    )


    # Truncate moment expansion.
    f = f.at[-nqmax-2 :-2].set(
        kmode * v * y[-2 * nqmax-2 : -nqmax-2] - (lmaxnu + 1) / tau * y[-nqmax-2 :-2]
    )

    # ---- Quintessence equations of motion -----------------------------------------------------------
    # ... Ballesteros & Lesgourgues (2010, BL10), arXiv:1004.5509
    f = f.at[-2].set( # BL10, eq. (3.5)
        -(1+w_Q) *(thetaq + 0.5 * hprime) - 3*(cs2_Q - w_Q) * aprimeoa * deltaq - 9*(1+w_Q)*(cs2_Q-ca2_Q)*aprimeoa**2/kmode**2 * thetaq
    )
    f = f.at[-1].set( # BL10, eq. (3.6)
        -(1-3*cs2_Q)*aprimeoa*thetaq + cs2_Q/(1+w_Q) * kmode**2 * deltaq
    )

    #jax.lax.cond(
    #   f_norm > 1e5,
    #   lambda: jax.debug.callback(lambda t,norm, y:print(f"⚠️ Crisis at t={t} | dy norm: {norm} | y: {y}"), a, f_norm, y),
    #   lambda: None
    #)
    #jax.debug.print("a={a} -> in = {y}, out = {f}", a=a, y=y, f=f)
    #print("Inside ODE:",y.shape, f.shape)
    #worst_index = jnp.argmax(jnp.abs(f))
    #worst_value = f[worst_index]
    #jax.debug.print("a={a:.2e} , tau = {tau:.2e}, -> norm={m}  < worst = {i}, w={w}", tau=tau,a=a,m=jnp.linalg.norm(f), i=worst_index, w=worst_value)
    return f.flatten()
    #return f



def model_synchronous_ALL(*, tau, y, param, kmode, lmaxg, lmaxgp, lmaxr, lmaxnu, nqmax ):     
    """Solve the synchronous gauge perturbation equations for a single mode.

    Parameters
    ----------
    tau : float
        conformal time
    yin : array_like
        input vector of perturbations
    param : array_like
        dictionary of parameters and interpolated background functions
    kmode : float
        wavenumber of modef
    lmaxg : int
        maximum photon temperature hierarchy multipole
    lmaxgp : int
        maximum photon polarization hierarchy multipole
    lmaxr : int
        maximum massless neutrino hierarchy multipole
    lmaxnu : int
        maximum neutrino hierarchy multipole
    nqmax : int
        maximum number of momentum bins for massive neutrinos

    Returns
    -------
    f : array_like
        RHS of perturbation equations
    """
    print("Inside ODE:",y.shape)
    def to_scalar(x):
        return jnp.ravel(x)[0]
    rsa_settings = get_approximation_settings(param)
    use_tca = rsa_settings['use_tca']
    tca_tau_c_over_tau_h_trigger = rsa_settings['tca_tau_c_over_tau_h_trigger']
    tca_tau_c_over_tau_k_trigger = rsa_settings['tca_tau_c_over_tau_k_trigger']
    use_rsa = rsa_settings['use_rsa']
    tau_c_over_tau_trigger = rsa_settings['tau_c_over_tau_trigger']
    tau_over_tau_k_trigger = rsa_settings['tau_over_tau_k_trigger']
    use_ur_fluid = rsa_settings['use_ur_fluid']
    ur_fluid_tau_over_tau_k_trigger = rsa_settings['ur_fluid_tau_over_tau_k_trigger']

    #y = jnp.ravel(y)
    tau = to_scalar(tau)
    kmode = to_scalar(kmode)
    Omegac = param['Omegam'] - param['Omegab']

    iq0 = 10 + lmaxg + lmaxgp + lmaxr
    iq1 = iq0 + nqmax
    iq2 = iq1 + nqmax
    iq3 = iq2 + nqmax
    iq4 = iq3 + nqmax

    # y = jnp.copy(yin)
    f = jnp.zeros_like( y )

    #TODO: add curvature
    # ... curvature
    K = 0
    tau = tau + 1e-50
    
    # def cotKgen_zero_curv():
    #     return 1.0/(kmode*tau)
    # def cotKgen_pos_curv():
    #     return jnp.sqrt(K)/kmode/jnp.tan(jnp.sqrt(K)*tau)
    # def cotKgen_neg_curv():
    #     return jnp.sqrt(-K)/kmode/jnp.tanh(jnp.sqrt(-K)*tau)
    
    # cotKgen = jax.lax.switch(int(1+jax.lax.sign(K)), [cotKgen_neg_curv, cotKgen_zero_curv, cotKgen_pos_curv])
    s2_squared = 1.-3.*K/kmode**2
    s_l2 = 1.0
    s_l3 = 1.0

    # ... metric
    a = y[0]
    loga = jnp.log(a)

    #ahprime = y[1]
    eta = y[2]

    # ... cdm
    deltac = y[3]
    thetac = y[4]

    # ... baryons
    deltab = y[5]
    thetab = y[6]

    # ... photons
    deltag = y[7]
    thetag = y[8]
    shearg = y[9] / 2.0

    # ... massless neutrinos
    deltar = y[ 9 + lmaxg + lmaxgp]
    thetar = y[10 + lmaxg + lmaxgp]
    shearr = y[11 + lmaxg + lmaxgp] / 2.0

    # ... quintessence field
    deltaq = y[-2]
    thetaq = y[-1]

    # ... evaluate thermodynamics
    # tempb   = param['tempba_of_tau_spline'].evaluate( tau ) / a
    # xeprime = param['xe_of_tau_spline'].derivative( tau )
    # cs2     = param['cs2a_of_tau_spline'].evaluate( tau ) / a
    # xe      = param['xe_of_tau_spline'].evaluate( tau )

    # Use pre-composed splines for direct log(a) lookup (performance optimization)
    cs2     = to_scalar(param['cs2a_of_loga_spline'].evaluate( loga ))/a
    xe      = to_scalar(param['xe_of_loga_spline'].evaluate( loga ))
    
    # ... Photon mass density over baryon mass density
    photbar = param['grhog'] / (param['grhom'] * param['Omegab'] * a)
    pb43 = 4.0 / 3.0 * photbar

    # massive neutrinos
    rhonu = jnp.exp(param['logrhonu_of_loga_spline'].evaluate(loga))
    pnu = jnp.exp(param['logpnu_of_loga_spline'].evaluate(loga))

    # ... quintessence
    cs2_Q              = param['cs2_DE'] 
    w_Q                = param['w_DE_0'] + param['w_DE_a'] * (1.0 - a) 
    rho_Q              = a**(-3*(1+param['w_DE_0']+param['w_DE_a'])) * jnp.exp(3*(a-1)*param['w_DE_a'])
    rho_plus_p_theta_Q = (1+w_Q) * rho_Q * param['grhom'] * param['OmegaDE'] * thetaq * a**2
    
    # ... homogeneous background
    # grho = (
    #     param['grhom'] * param['Omegam'] / a
    #     + (param['grhog'] + param['grhor'] * (param['Neff'] + param['Nmnu'] * rhonu)) / a**2
    #     + param['grhom'] * param['OmegaDE'] * rho_Q * a**2
    #     + param['grhom'] * param['Omegak']
    # )

    # gpres = (
    #     (param['grhog'] + param['grhor'] * param['Neff']) / 3.0 + param['grhor'] * param['Nmnu'] * pnu
    # ) / a**2 + w_Q * param['grhom'] * param['OmegaDE'] * rho_Q * a**2

    # ... compute expansion rate
    aprimeoa = to_scalar(get_aprimeoa( param=param, aexp=a ))
    xeprime = to_scalar(param['xe_of_loga_spline'].derivative( loga ) * aprimeoa)
    gpres = (
        (param['grhog'] + param['grhor'] * param['Neff']) / 3.0 + param['grhor'] * param['Nmnu'] * pnu
    ) / a**2 + w_Q * param['grhom'] * param['OmegaDE'] * rho_Q * a**2
    aprimeprimeoa = to_scalar(0.5 * (aprimeoa**2 - gpres))
    # aprimeoa = jnp.sqrt(grho / 3.0)                # Friedmann I
    # aprimeprimeoa = 0.5 * (aprimeoa**2 - gpres)    # Friedmann II

    # quintessence EOS time derivatives
    w_Q_prime = -param['w_DE_a'] * aprimeoa * a
    ca2_Q     = w_Q - w_Q_prime / 3 / ((1+w_Q)+1e-6) / aprimeoa

    # ... Thomson opacity coefficient
    akthom = 2.3038921003709498e-9 * (1.0 - param['YHe']) * param['Omegab'] * param['H0']**2

    # ... Thomson opacity
    opac    = to_scalar(xe * akthom / a**2)
    tauc    = to_scalar(1. / jnp.maximum(opac, 1e-30))
    taucprime = to_scalar(tauc * (2 * aprimeoa - xeprime / jnp.maximum(xe, 1e-30)))
    tauh = to_scalar(1.0 / jnp.maximum(aprimeoa, 1e-30))
    tauk = to_scalar(1.0 / jnp.maximum(kmode, 1e-30))

    do_relativistic_sa = jnp.logical_and(
        jnp.asarray(use_rsa),
        in_rsa_regime(
            tau=tau,
            kmode=kmode,
            tau_c=tauc,
            tau_c_over_tau_trigger=tau_c_over_tau_trigger,
            tau_over_tau_k_trigger=tau_over_tau_k_trigger,
        ),
    )
    do_ur_fluid = jnp.logical_and(
        jnp.logical_not(do_relativistic_sa),
        jnp.logical_and(
            jnp.asarray(use_ur_fluid),
            in_ur_fluid_regime(
                tau=tau,
                kmode=kmode,
                tau_over_tau_k_trigger=ur_fluid_tau_over_tau_k_trigger,
            ),
        ),
    )
    do_tca = jnp.logical_and(
        jnp.logical_not(do_relativistic_sa),
        jnp.logical_and(
            jnp.asarray(use_tca),
            in_tca_regime(
                tau_h=tauh,
                tau_k=tauk,
                tau_c=tauc,
                tau_c_over_tau_h_trigger=tca_tau_c_over_tau_h_trigger,
                tau_c_over_tau_k_trigger=tca_tau_c_over_tau_k_trigger,
            ),
        ),
    )
    #tauc    = 1. / opac
    #taucprime = tauc * (2*aprimeoa - xeprime/xe)
    #F       = tauc / (1+pb43) #CLASS perturbations.c:10072
    #Fprime  = taucprime/(1+pb43) + tauc*pb43*aprimeoa/(1+pb43)**2 #CLASS perturbations.c:10074

    
    # ... background scale factor evolution
    print(f.shape, (aprimeoa*a).shape)
    f = f.at[0].set( aprimeoa * a )
    
    # ... evaluate metric perturbations
    drhonu, dpnu, fnu, shearnu = nu_perturb( a, param['amnu'], y[iq0:iq1], y[iq1:iq2], y[iq2:iq3], nqmax=nqmax )

    dgrho = (
        param['grhom'] * (Omegac * deltac + param['Omegab'] * deltab) / a
        + (param['grhog'] * deltag + param['grhor'] * (param['Neff'] * deltar + param['Nmnu'] * drhonu)) / a**2
        + param['grhom'] * param['OmegaDE'] * deltaq * rho_Q * a**2
    )

    # RSA metric estimate without relativistic species (BLT11-inspired closure)
    dgrho_wo_rel = (
        param['grhom'] * (Omegac * deltac + param['Omegab'] * deltab) / a
        + param['grhor'] * param['Nmnu'] * drhonu / a**2
        + param['grhom'] * param['OmegaDE'] * deltaq * rho_Q * a**2
    )
    hprime_rsa = to_scalar((2.0 * kmode**2 * eta + dgrho_wo_rel) / jnp.maximum(aprimeoa, 1e-30))
    deltag_rsa, thetag_rsa, shearg_rsa, deltar_rsa, thetar_rsa, shearr_rsa = compute_fields_rsa(
        kmode=kmode,
        aprimeoa=aprimeoa,
        hprime=hprime_rsa,
        eta=eta,
        deltab=deltab,
        thetab=thetab,
        cs2_b=cs2,
        tau_c=tauc,
        tau_c_prime=taucprime,
    )

    deltag_eff = jnp.where(do_relativistic_sa, deltag_rsa, deltag)
    thetag_eff = jnp.where(do_relativistic_sa, thetag_rsa, thetag)
    shearg_eff = jnp.where(do_relativistic_sa, shearg_rsa, shearg)
    deltar_eff = jnp.where(do_relativistic_sa, deltar_rsa, deltar)
    thetar_eff = jnp.where(do_relativistic_sa, thetar_rsa, thetar)
    shearr_eff = jnp.where(do_relativistic_sa, shearr_rsa, shearr)
    dgpres = (
        (param['grhog'] * deltag_eff + param['grhor'] * param['Neff'] * deltar_eff) / a**2 / 3.0 
        + param['grhor'] * param['Nmnu'] * dpnu / a**2 
        + (cs2_Q * param['grhom'] * param['OmegaDE'] * deltaq * rho_Q * a**2 + (cs2_Q-ca2_Q)*(3*aprimeoa * rho_plus_p_theta_Q / kmode**2))
    )
    dgtheta = (
        param['grhom'] * (Omegac * thetac + param['Omegab'] * thetab) / a
        + 4.0 / 3.0 * (param['grhog'] * thetag_eff + param['Neff'] * param['grhor'] * thetar_eff) / a**2
        + param['Nmnu'] * param['grhor'] * kmode * fnu / a**2
        + rho_plus_p_theta_Q
    )
    dgshear = (
        4.0 / 3.0 * (param['grhog'] * shearg_eff + param['Neff'] * param['grhor'] * shearr_eff) / a**2
        + param['Nmnu'] * param['grhor'] * shearnu / a**2
    )

    dahprimedtau = -(dgrho + 3.0 * dgpres) * a
    
    print(f.shape, (dahprimedtau).shape)
    f = f.at[1].set( dahprimedtau )

    # ... force energy conservation
    hprime_full = to_scalar((2.0 * kmode**2 * eta + dgrho) / aprimeoa)
    hprime = to_scalar(jnp.where(do_relativistic_sa, hprime_rsa, hprime_full))

    etaprime = to_scalar(0.5 * dgtheta / kmode**2)
    alpha  = to_scalar((hprime + 6.*etaprime)/2./kmode**2)
    f = f.at[2].set( etaprime )
    
    # alphaprime = -3*dgshear/(2*kmode**2) + eta - 2*aprimeoa*alpha
    # alphaprime -=  9/2 * a**2/kmode**2 * 4/3 * 16/45/opac * (thetag+kmode**2*alpha) * param['grhog']

    # ... cdm equations of motion, MB95 eq. (42)
    deltacprime = -thetac - 0.5 * hprime
    f = f.at[3].set( deltacprime )
    thetacprime = -aprimeoa * thetac  # thetac = 0 in synchronous gauge!
    f = f.at[4].set( thetacprime )

    idxb = 5
    # --- baryon equations of motion, MB95 eqs. (66) ---------------------------------------------
    # ... baryon density, BLT11 eq. (2.1a)
    deltabprime = -thetab - 0.5 * hprime
    f = f.at[idxb+0].set( deltabprime )
    # ... baryon velocity, BLT11 eq. (2.1b)
    thetabprime = -aprimeoa * thetab + kmode**2 * cs2 * deltab \
                + pb43 * opac * (thetag_eff - thetab)
    f = f.at[idxb+1].set( thetabprime )

    # --- photon equations of motion, MB95 eqs. (63) ---------------------------------------------
    idxg  = 7
    idxgp = 7 + (lmaxg+1)

    def _photon_hierarchy(f_in):
        # ... polarization term
        polter = y[idxg+2] + y[idxgp+0] + y[idxgp+2]
        # ... photon density, BLT11 eq. (2.4a)
        deltagprime = 4.0 / 3.0 * (-thetag_eff - 0.5 * hprime)
        f_in = f_in.at[idxg+0].set( deltagprime )
        # ... photon velocity, BLT11 eq. (2.4b)
        thetagprime = kmode**2 * (0.25 * deltag_eff - s2_squared * shearg_eff) \
                    - opac * (thetag_eff - thetab)
        f_in = f_in.at[idxg+1].set( thetagprime )
        # ... photon shear, BLT11 eq. (2.4c)
        sheargprime = 8./15. * (thetag_eff+kmode**2*alpha) -3/5*kmode*s_l3/s_l2*y[idxg+3] \
                    - opac*(y[idxg+2]-0.1*s_l2*polter)
        f_in = f_in.at[idxg+2].set( sheargprime )

        #... photon temperature l>=3, BLT11 eq. (2.4d)
        ell  = jnp.arange(3, lmaxg )
        ell_2d = ell[:,None]
        f_in = f_in.at[idxg+ell].set( kmode  / (2 * ell_2d + 1) * (ell_2d * y[idxg+ell-1] - (ell_2d + 1) * y[idxg+ell+1]) - opac * y[idxg+ell] )
        # photon temperature hierarchy truncation, BLT11 eq. (2.5)
        f_in = f_in.at[idxg+lmaxg].set( kmode * y[idxg+lmaxg-1] - (lmaxg + 1) / tau * y[idxg+lmaxg] - opac * y[idxg+lmaxg] )

        #... polarization equations, BLT11 eq. (2.4e)
        ell  = jnp.arange(0, lmaxgp) # l=0...lmaxgp-1
        ell_2d = ell[:, None]
        f_in = f_in.at[idxgp+ell].set( kmode  / (2 * ell_2d + 1) * (ell_2d * y[idxgp+ell-1] - (ell_2d + 1) * y[idxgp+ell+1]) - opac * y[idxgp+ell] )
        f_in = f_in.at[idxgp+0].add( opac * polter / 2 )  # photon polarization l=0
        f_in = f_in.at[idxgp+2].add( opac * polter / 10 ) # photon polarization l=2

        # photon polarization hierarchy truncation
        f_in = f_in.at[idxgp+lmaxgp].set( kmode * y[idxgp+lmaxgp-1] - (lmaxgp + 1) / tau * y[idxgp+lmaxgp] - opac * y[idxgp+lmaxgp] )
        return f_in

    def _photon_tca(f_in):
        deltabprime_tca = -thetab - 0.5 * hprime
        thetabprime_tca = (
            -aprimeoa * thetab + kmode**2 * cs2 * deltab + kmode**2 * pb43 * (0.25 * deltag - s2_squared * shearg)
        ) / (1.0 + pb43)

        deltagprime_tca = 4.0 / 3.0 * (-thetag - 0.5 * hprime)
        slip = (
            2.0 * pb43 / (1.0 + pb43) * aprimeoa * (thetab - thetag)
            + tauc
            * (
                -aprimeprimeoa * thetab
                - 0.5 * aprimeoa * kmode**2 * deltag
                + kmode**2 * (cs2 * deltabprime_tca - 0.25 * deltagprime_tca)
            )
            / (1.0 + pb43)
        )
        thetabprime_tca = thetabprime_tca + pb43 / (1.0 + pb43) * slip
        thetagprime_tca = (-thetabprime_tca - aprimeoa * thetab + kmode**2 * cs2 * deltab) / jnp.maximum(pb43, 1e-30) + kmode**2 * (
            0.25 * deltag - s2_squared * shearg
        )

        f_in = f_in.at[idxb + 0].set(deltabprime_tca)
        f_in = f_in.at[idxb + 1].set(thetabprime_tca)
        f_in = f_in.at[idxg + 0].set(deltagprime_tca)
        f_in = f_in.at[idxg + 1].set(thetagprime_tca)
        f_in = f_in.at[idxg + 2:idxg + lmaxg + 1].set(0.0)
        f_in = f_in.at[idxgp:idxgp + lmaxgp + 1].set(0.0)
        return f_in

    def _photon_rsa(f_in):
        return f_in.at[idxg:idxg + lmaxg + 1].set(0.0).at[idxgp:idxgp + lmaxgp + 1].set(0.0)

    f = jax.lax.cond(
        do_relativistic_sa,
        _photon_rsa,
        lambda f_in: jax.lax.cond(do_tca, _photon_tca, _photon_hierarchy, f_in),
        f,
    )

    # --- Massless neutrino equations of motion -------------------------------------------------------
    idxr = 9 + lmaxg + lmaxgp
    deltarprime = 4.0 / 3.0 * (-thetar_eff - 0.5 * hprime)
    f = f.at[idxr+0].set( deltarprime )
    thetarprime = kmode**2 * (0.25 * deltar_eff - shearr_eff)
    f = f.at[idxr+1].set( thetarprime )

    def _massless_nu_hierarchy(f_in):
        shearrprime = 8./15. * (thetar_eff + kmode**2 * alpha) - 0.6 * kmode * y[idxr+3]
        f_in = f_in.at[idxr+2].set( shearrprime )
        ell = jnp.arange(3, lmaxr)
        ell_2d = ell[:, None]
        f_in = f_in.at[idxr+ell].set( kmode / (2 * ell_2d + 1) * (ell_2d * y[idxr+ell-1] - (ell_2d + 1) * y[idxr+ell+1]) )

        # ... truncate moment expansion
        f_in = f_in.at[idxr+lmaxr].set( kmode * y[idxr+lmaxr-1] - (lmaxr + 1) / tau * y[idxr+lmaxr] )
        return f_in

    def _massless_nu_ufa(f_in):
        shearrprime = compute_shearprime_ufa(
            tau=tau,
            shearr=shearr_eff,
            thetar=thetar_eff,
            hprime=hprime,
        )
        f_in = f_in.at[idxr+2].set( shearrprime )
        return f_in.at[idxr+3:idxr + lmaxr + 1].set(0.0)

    def _massless_nu_rsa(f_in):
        return f_in.at[idxr:idxr + lmaxr + 1].set(0.0)

    f = jax.lax.cond(
        do_relativistic_sa,
        _massless_nu_rsa,
        lambda f_in: jax.lax.cond(do_ur_fluid, _massless_nu_ufa, _massless_nu_hierarchy, f_in),
        f,
    )

    # --- Massive neutrino equations of motion --------------------------------------------------------
    q, _ = get_neutrino_momentum_bins( nqmax )
    q = q[:, None]
    aq = a * param['amnu'] / q
    v = 1 / jnp.sqrt(1 + aq**2)
    dlfdlq = -q / (1.0 + jnp.exp(-q))  # derivative of the Fermi-Dirac distribution

    f = f.at[iq0 : iq1].set(
        -kmode * v * y[iq1 : iq2] + hprime* dlfdlq / 6.0 
    )
    f = f.at[iq1 : iq2].set(
        kmode * v * (y[iq0 : iq1] - 2.0 * y[iq2 : iq3]) / 3.0
    )
    f = f.at[iq2 : iq3].set(
        kmode * v * (2 * y[iq1 : iq2] - 3 * y[iq3 : iq4]) / 5.0 - (hprime / 15 + 2 / 5 * etaprime) * dlfdlq
    )

    ell = jnp.arange(3, lmaxnu)
    ell_2d = ell[:, None]
    vv = jnp.tile(v, (lmaxnu - 3,1))
    denl = jnp.repeat( 2*ell_2d+1, nqmax , axis=0)
    
    print("lmaxnu = {}, nqmax = {}", lmaxnu, nqmax)

    print(y.shape)
    print(vv.shape, denl.shape, y[iq0 + 2*nqmax : iq0 + (lmaxnu-1)*nqmax].shape, y[iq0 + 4*nqmax : iq0 + (lmaxnu+1)*nqmax].shape)
    f = f.at[iq0 + 3 * nqmax : iq0 + lmaxnu * nqmax].set(
        kmode * vv / denl * (
            jnp.repeat( ell_2d, nqmax ,axis=0) * y[iq0 + 2*nqmax : iq0 + (lmaxnu-1)*nqmax] 
            - jnp.repeat( ell_2d+1, nqmax ,axis=0) * y[iq0 + 4*nqmax : iq0 + (lmaxnu+1)*nqmax]
        ) 
    )

    # Truncate moment expansion.
    f = f.at[-nqmax-2 :-2].set(
        kmode * v * y[-2 * nqmax-2 : -nqmax-2] - (lmaxnu + 1) / tau * y[-nqmax-2 :-2]
    )

    # ---- Quintessence equations of motion -----------------------------------------------------------
    # ... Ballesteros & Lesgourgues (2010, BL10), arXiv:1004.5509
    f = f.at[-2].set( # BL10, eq. (3.5)
        -(1+w_Q) *(thetaq + 0.5 * hprime) - 3*(cs2_Q - w_Q) * aprimeoa * deltaq - 9*(1+w_Q)*(cs2_Q-ca2_Q)*aprimeoa**2/kmode**2 * thetaq
    )
    f = f.at[-1].set( # BL10, eq. (3.6)
        -(1-3*cs2_Q)*aprimeoa*thetaq + cs2_Q/(1+w_Q) * kmode**2 * deltaq
    )

    print("Inside ODE:",y.shape, f.shape)
    #return f.flatten()
    return f

def convert_to_output_variables(*, y, param, kmode, lmaxg, lmaxgp, lmaxr, lmaxnu, nqmax , idx):
    """Convert the synchronous gauge perturbations to the output fields.

    Parameters
    ----------
    y : array_like
        input vector of perturbations
    param : dict
        dictionary of parameters and interpolated background functions
    kmode : float
        wavenumber of mode [1/Mpc]
    lmaxg : int
        maximum photon Boltmann hierarchy moment
    lmaxgp : int
        maximum photon polarization hierarchy moment
    lmaxr : int
        maximum massless neutrino hierarchy moment
    lmaxnu : int
        maximum massive neutrino hierarchy moment
    nqmax : int
        number of momentum bins for massive neutrinos

    Returns
    -------
    yout : array_like
        output vector of perturbations:
            eta, etaprime, hprime, alpha,       # 0-3
            deltam,  thetam / (aH),             # 4-5
            deltabc, thetabc / (aH),            # 6-7
            deltac,  thetac / (aH),             # 8-9
            deltab,  thetab / (aH),             # 10-11
            deltag,  thetag / (aH),             # 12-13
            deltar,  thetar / (aH),             # 14-15
            deltanu, thetanu / (aH),            # 16-17
            deltaq,  thetaq / (aH),             # 18-19
    where aH = \\mathcal{H} = a' / a, which is the conformal Hubble rate.
    """

    
    def take_idx(x):
      return jnp.take(x, idx, mode="wrap")
    Omegac = take_idx(param['Omegam'] - param['Omegab'])

    iq0 = 10 + lmaxg + lmaxgp + lmaxr
    iq1 = iq0 + nqmax
    iq2 = iq1 + nqmax
    iq3 = iq2 + nqmax

    a = y[0]
    eta = y[2]

    # ... cdm
    deltac = y[3]
    thetac = y[4]

    # ... baryons
    deltab = y[5]
    thetab = y[6]

    # ... photons
    deltag = y[7]
    thetag = y[8]

    # ... massless neutrinos
    deltar = y[ 9 + lmaxg + lmaxgp]
    thetar = y[10 + lmaxg + lmaxgp]

    #... massive neutrinos
    rhonu = jnp.exp(take_idx(param['logrhonu_of_loga_spline'].evaluate(jnp.log(a))))
    pnu = jnp.exp(take_idx(param['logpnu_of_loga_spline'].evaluate( jnp.log(a) )) )
    rho_plus_p = rhonu + pnu

    drhonu, _, fnu, _ = nu_perturb( a, take_idx(param['amnu']), y[iq0:iq1], y[iq1:iq2], y[iq2:iq3], nqmax=nqmax )
    deltanu = drhonu / rhonu
    thetanu = kmode * fnu / rho_plus_p

    # ... quintessence field
    deltaq    = y[-2]
    thetaq    = y[-1]
    w_Q       = take_idx(param['w_DE_0'] + param['w_DE_a'] * (1.0 - a))
    rho_Q     = a**(-3*take_idx(1+param['w_DE_0']+param['w_DE_a'])) * jnp.exp(3*(a-1)*take_idx(param['w_DE_a']))
    rho_plus_p_theta_Q = (1+w_Q) * rho_Q * take_idx(param['grhom'] * param['OmegaDE']) * thetaq * a**2


    # ... background
    grho = take_idx(
        param['grhom'] * param['Omegam'] / a
        + (param['grhog'] + param['grhor'] * (param['Neff'] + param['Nmnu'] * rhonu)) / a**2
        + param['grhom'] * param['OmegaDE'] * rho_Q * a**2
        + param['grhom'] * param['Omegak']
    )

    gpres = take_idx(
        (param['grhog'] + param['grhor'] * param['Neff']) / 3.0 + param['grhor'] * param['Nmnu'] * pnu
    ) / a**2 + w_Q * take_idx(param['grhom'] * param['OmegaDE']) * rho_Q * a**2
    
    # ... compute expansion rate
    aprimeoa = jnp.sqrt(grho / 3.0)                # Friedmann I
    
    # ... metric perturbations
    dgrho = take_idx(
        param['grhom'] * (Omegac * deltac + param['Omegab'] * deltab) / a
        + (param['grhog'] * deltag + param['grhor'] * (param['Neff'] * deltar + param['Nmnu'] * drhonu)) / a**2
        + param['grhom'] * param['OmegaDE'] * deltaq * rho_Q * a**2
    )
    dgtheta = take_idx(
        param['grhom'] * (Omegac * thetac + param['Omegab'] * thetab) / a
        + 4.0 / 3.0 * (param['grhog'] * thetag + param['Neff'] * param['grhor'] * thetar) / a**2
        + param['Nmnu'] * param['grhor'] * kmode * fnu / a**2
        + rho_plus_p_theta_Q
    )
    
    hprime = (2.0 * kmode**2 * eta + dgrho) / aprimeoa
    etaprime = 0.5 * dgtheta / kmode**2
    alpha  = (hprime + 6.*etaprime)/2./kmode**2


    # total matter perturbations
    deltam = take_idx(
        ( param['grhom'] * (Omegac * deltac + param['Omegab'] * deltab) / a
        + (param['grhor'] * param['Nmnu'] * drhonu) / a**2) / (param['grhom'] * param['Omegam'] / a
        + (param['grhor'] * param['Nmnu'] * rhonu)/ a**2 )
    )
    thetam = take_idx(
        (param['grhom'] * (Omegac * thetac + param['Omegab'] * thetab) / a + param['Nmnu'] * param['grhor'] * kmode * fnu / a**2) 
        / (3.0 * (param['grhom'] * param['Omegam'] / a + param['grhor'] * param['Nmnu'] * rhonu / a**2 ))
    )

    deltabc = take_idx(param['grhom'] * (Omegac * deltac + param['Omegab'] * deltab)/ a) \
        / (param['grhom'] * param['Omegam'] / a)
    thetabc = take_idx(param['grhom'] * (Omegac * thetac + param['Omegab'] * thetab) / a) \
        / (3.0 * (param['grhom'] * param['Omegam'] / a) / a**2)
    
    #... gauge trafo from comoving (MB95 eq. 27b)
    thetam   += alpha * kmode**2
    thetabc  += alpha * kmode**2

    ##################################################################################################################

    # store fields of interest
    yout = jnp.array([
        eta, etaprime, hprime, alpha,       # 0-3
        deltam,  thetam  / aprimeoa,        # 4-5
        deltabc, thetabc / aprimeoa,        # 6-7
        deltac,  thetac  / aprimeoa,        # 8-9
        deltab,  thetab  / aprimeoa,        # 10-11
        deltag,  thetag  / aprimeoa,        # 12-13
        deltar,  thetar  / aprimeoa,        # 14-15
        deltanu, thetanu / aprimeoa,        # 16-17
        deltaq,  thetaq  / aprimeoa,        # 18-19
    ])
    
                
    return yout


def adiabatic_ics_one_mode( *, tau: float|Array, param, kmode, nvar, lmaxg, lmaxgp, lmaxr, lmaxnu, nqmax ):
    """Initial conditions for adiabatic perturbations"""
    Omegac = param['Omegam'] - param['Omegab']

    iq0 = 10 + lmaxg + lmaxgp + lmaxr
    iq1 = iq0 + nqmax
    iq2 = iq1 + nqmax
    iq3 = iq2 + nqmax
    iq4 = iq3 + nqmax

    a = param['a_of_tau_spline'].evaluate(tau)

    # .. isentropic ("adiabatic") initial conditions
    rhom  = param['grhom'] * param['Omegam'] / a**3
    rhor  = (param['grhog'] + param['grhor'] * (param['Neff'] + param['Nmnu']*jnp.exp(param['logrhonu_of_loga_spline'].evaluate(jnp.log(a))))) / a**4

    ##jax.debug.print("rhor = {rhor} ! {grhog}, {grhor}", rhor=rhor, grhog=param['grhog'], grhor=param['grhor'])
    rhonu = param['grhor'] * (param['Neff'] + param['Nmnu']*jnp.exp(param['logrhonu_of_loga_spline'].evaluate(jnp.log(a)))) / a**4

    ##jax.debug.print("nu = {rhonu} ! {grhor}, {Neff}, {Nmnu}, {logrhonu_spline}, {a}", rhonu=rhonu, grhor=param['grhor'], Neff=param['Neff'], Nmnu=param['Nmnu'], logrhonu_spline=param['logrhonu_of_loga_spline'].evaluate(jnp.log(a)), a=a)

    fracb  = param['Omegab'] / param['Omegam']
    fracg  = param['grhog'] / rhor
    fracnu = rhonu / rhor
    
    #jax.debug.print("fracnu = {fracnu} ! {rhor}, {rhonu}", fracnu=fracnu, rhonu=rhonu, rhor=rhor)
    
    #def print_callback(arg):
    #  fracnu, rhor, rhonu = arg
    #  print(f"ACTUAL VALUES:\nfracnu={fracnu}\nrhor={rhor}\nrhonu={rhonu}\n")
    #  print(f"TYPES: fracnu={fracnu.dtype}, rhor={rhor.dtype}")

    # This safely hooks into JAX's execution pipeline, ignoring tracers
    #jax.debug.callback(print_callback, (fracnu, rhor, rhonu))


    om    = a * rhom / jnp.sqrt(rhor)
    
    curvature_ini = -1.0
    s2_squared = 1.0

    #... photons
    deltag = -(kmode*tau)**2 / 3 * (1 - om * tau / 5) * curvature_ini * s2_squared
    thetag = -(kmode*tau)**3/tau /36 * (1-3*(1+5*fracb-fracnu)/20/(1-fracnu)*om*tau) * curvature_ini * s2_squared

    #... baryons
    deltab = 0.75 * deltag
    thetab = thetag

    #... CDM
    deltac = 0.75 * deltag
    thetac = jnp.zeros(deltag.shape)

    #... massless neutrinos
    deltar = deltag
    thetar = -(kmode*tau)**4/tau/36/(4*fracnu+15) * (4*fracnu+11+12 - 3*(8*fracnu*fracnu+50*fracnu+275)/20/(2*fracnu+15)*tau*om) * curvature_ini
    shearr = (kmode*tau)**2/(45+12*fracnu) * (3*s2_squared-1) * (1+(4*fracnu-5)/4/(2*fracnu+15)*tau*om) * curvature_ini

    #... massive neutrinos
    deltan = deltar
    thetan = thetar
    shearn = shearr

    # ... quintessence, Ballesteros & Lesgourgues (2010, BL20), arXiv:1004.5509
    cs2_Q  = param['cs2_DE']
    w_Q    = param['w_DE_0'] + param['w_DE_a'] * (1.0 - a)
    deltaq = (kmode*tau)**2 / 4 * (1+w_Q)*(4-3*cs2_Q)/(4-6*w_Q+3*cs2_Q) * curvature_ini * s2_squared # BL10 eq. 3.7
    thetaq = (kmode*tau)**4 / tau / 4 * cs2_Q/(4-6*w_Q+3*cs2_Q) * curvature_ini * s2_squared      # BL10 eq. 3.8

    # metric
    eta = curvature_ini * (1-(kmode*tau)**2/12/(15+4*fracnu)*(5+4*s2_squared*fracnu - (16*fracnu*fracnu+280*fracnu+325)/10/(2*fracnu+15)*tau*om))
    #jax.debug.print("initial condition eta = {eta} ! -> {kmode}, {curvature_ini}, {tau}, {fracnu}, {s2_squared}, {om}", eta=eta, kmode=kmode, curvature_ini=curvature_ini, tau=tau, fracnu=fracnu, s2_squared=s2_squared, om=om)

    ahprime = jnp.zeros(deltag.shape) # will not be evolved, only constraint


    # ... metric
    #y = y.at[0].set( a )
    #y = y.at[1].set( ahprime )
    #y = y.at[2].set( eta )

    # .. CDM
    #y = y.at[3].set( deltac )
    #y = y.at[4].set( thetac )

    # .. baryons
    #y = y.at[5].set( deltab )
    #y = y.at[6].set( thetab )

    # ... Photons (total intensity and polarization)
    #y = y.at[7].set( deltag )
    #y = y.at[8].set( thetag )
    # shear and polarization are zero at the initial time
    
    # ... massless neutrinos
    #y = y.at[ 9 + lmaxg + lmaxgp].set( deltar )
    #y = y.at[10 + lmaxg + lmaxgp].set( thetar )
    #y = y.at[11 + lmaxg + lmaxgp].set( shearr * 2.0 )
    # higher moments are zero at the initial time

    # ... massive neutrinos
    # if params.cp.Nmnu > 0:
    # q = jnp.arange(1, nqmax + 1) - 0.5  # so dq == 1 # if not using CAMB approx
    q, _ = get_neutrino_momentum_bins( nqmax )
    q = q[:, None]
    aq = (a * param['amnu']) / q
    v = 1 / jnp.sqrt(1 + aq**2)
    # akv = jnp.outer(kmode, v)
    dlfdlq = -q / (1.0 + jnp.exp(-q))
    #y = y.at[iq0:iq1].set( -0.25 * dlfdlq * deltan)
    #y = y.at[iq1:iq2].set( -dlfdlq * thetan / v / kmode / 3.0)
    #y = y.at[iq2:iq3].set( -0.5 * dlfdlq * shearn)
    # higher moments are zero at the initial time

    # ... quintessence, Ballesteros & Lesgourgues (2010, BL20), arXiv:1004.5509
    #y = y.at[-2].set( deltaq )
    #y = y.at[-1].set( thetaq )
    
    #print([(x.shape if hasattr(x,'shape') else None) for x in [a, ahprime, eta, deltac, thetac, deltab, thetab, deltag, thetag] ])
    #print([(x.shape if hasattr(x,'shape') else None) for x in [deltar, thetar, shearr * 2.0] ])
    #print(jnp.zeros(1+lmaxg+lmaxgp)[:, None].shape)
    #print(jnp.zeros(6+lmaxr)[:, None].shape)
    #print((-0.25 * dlfdlq * deltan).shape)
    #print((-dlfdlq * thetan / v / kmode / 3.0).shape)
    #print((-0.5 * dlfdlq * shearn).shape)
    #print([(x.shape if hasattr(x,'shape') else None) for x in [deltaq, thetaq] ])

    block1 = jnp.vstack([a, ahprime, eta, deltac, thetac, deltab, thetab, deltag, thetag])
    block2 = jnp.zeros((1+lmaxg+lmaxgp,1))
    block3 = jnp.vstack([deltar, thetar, shearr * 2.0])
    block4 = jnp.zeros((6+lmaxr,1))
    block5 = -0.25 * dlfdlq * deltan
    block6 = -dlfdlq * thetan / v / kmode / 3.0
    block7 = -0.5 * dlfdlq * shearn
    block8 = jnp.zeros(((lmaxnu-3)*nqmax, 1))
    block9 = jnp.vstack([deltaq, thetaq])
    
    blocks = [block1, block2, block3, block4, block5, block6, block7, block8, block9]
    ymax = max(b.shape[1] for b in blocks)

    print(iq0, iq1, iq2, iq3, iq3+2)
    blocks_aligned = [jnp.broadcast_to(b, (b.shape[0], ymax)) if b.shape[1]!=ymax else b for b in blocks]

    y = jnp.concatenate(blocks_aligned, axis=0)
    
    
    #jax.debug.print("initial condition = {y} !", y=y)
    
    return y


def determine_starting_time( *, param, k ):
    # ADOPTED from CLASS:
    # largest wavelengths start being sampled when universe is sufficiently opaque. This is quantified in terms of the ratio of thermo to hubble time scales, 
    # \f$ \tau_c/\tau_H \f$. Start when start_largek_at_tau_c_over_tau_h equals this ratio. Decrease this value to start integrating the wavenumbers earlier 
    # in time.
    start_small_k_at_tau_c_over_tau_h =  0.0004 

    # ADOPTED from CLASS:
    #  largest wavelengths start being sampled when mode is sufficiently outside Hubble scale. This is quantified in terms of the ratio of hubble time scale 
    #  to wavenumber time scale, \f$ \tau_h/\tau_k \f$ which is roughly equal to (k*tau). Start when this ratio equals start_large_k_at_tau_k_over_tau_h. 
    #  Decrease this value to start integrating the wavenumbers earlier in time. 
    start_large_k_at_tau_h_over_tau_k = 0.07 #0.05

    tau0 = param['taumin']
    tau1 = param['tau_of_a_spline'].evaluate( 0.1 ) # don't start after a=0.1
    tau_k = 1.0/k

    def get_tauc_tauH( tau, param ):
        akthom = 2.3048e-9 * (1.0 - param['YHe']) * param['Omegab'] * param['H0']**2
        xe = param['xe_of_tau_spline'].evaluate( tau )
        a = param['a_of_tau_spline'].evaluate(tau)
        opac = xe * akthom / a**2

        # Note: For starting time calculation, we use the full aprimeoa from background
        # This is slightly different from the old radiation-only approximation but more accurate
        aprimeoa = get_aprimeoa( param=param, aexp=a )
        return 1.0/opac, 1.0/aprimeoa


    def get_tauH( tau, param ):
        a = param['a_of_tau_spline'].evaluate(tau)
        aprimeoa = get_aprimeoa( param=param, aexp=a )
        return 1.0/aprimeoa

    # condition for small k: tau_c(a) / tau_H(a) < start_small_k_at_tau_c_over_tau_h
    def cond_small_k( logtau, param ):
        tau_c, tau_H = get_tauc_tauH( jnp.exp(logtau), param[0] )
        # adotoa/opac > start_small_k_at_tau_c_over_tau_h
        start_small_k_at_tau_c_over_tau_h = param[1]
        return tau_c/tau_H/start_small_k_at_tau_c_over_tau_h - 1.0

    # condition for large k: tau_H(a) / tau_k < start_large_k_at_tau_k_over_tau_h
    def cond_large_k( logtau, param ):
        tau_H = get_tauH( jnp.exp(logtau), param[0] )
        tau_k = param[2]
        start_large_k_at_tau_h_over_tau_k = param[1]
        return tau_H/tau_k/start_large_k_at_tau_h_over_tau_k - 1.0

    logtau_large_k = root_find_bisect_nocond(func=cond_large_k, xleft=jnp.log(tau0), xright=jnp.log(tau1), numit=7, param=(param,start_large_k_at_tau_h_over_tau_k,tau_k) )
    logtau_small_k = root_find_bisect_nocond(func=cond_small_k, xleft=jnp.log(tau0), xright=jnp.log(tau1), numit=7, param=(param,start_small_k_at_tau_c_over_tau_h) )

    return jnp.exp(jnp.minimum(logtau_small_k, logtau_large_k))


class VectorField(eqx.Module):
    model: eqx.Module

    def __call__(self, t, y, args):
        return self.model(t, y, args)

def rms_norm_filtered_batched(x: PyTree, filter_indices: jnp.ndarray, weights: list[jnp.ndarray]) -> Scalar:
    
    xs_weighted = jax.vmap(lambda _x, _w: _x[filter_indices] * _w, (0, 0))(x, jnp.array(weights))
    x_, _ = fu.ravel_pytree(xs_weighted)
    return _rms_norm(x_)

    # norms = jax.vmap(_rms_norm)(xs_weighted)

    # return jnp.max(norms)


def rms_norm_filtered(x: PyTree, filter_indices: jnp.ndarray, weights: jnp.ndarray) -> Scalar:
    x, _ = fu.ravel_pytree(x)
    if x.size == 0:
        return 0
    return _rms_norm(x[filter_indices] * weights)


@jax.custom_jvp
def _rms_norm(x):
    x_sq = jnp.real(x * jnp.conj(x))
    return jnp.sqrt(jnp.mean(x_sq))


@_rms_norm.defjvp
def _rms_norm_jvp(x, tx):
    (x,) = x
    (tx,) = tx
    out = _rms_norm(x)
    # Get zero gradient, rather than NaN gradient, in these cases
    pred = (out == 0) | jnp.isinf(out)
    numerator = jnp.where(pred, 0, x)
    denominator = jnp.where(pred, 1, out * x.size)
    t_out = jnp.dot(numerator / denominator, tx)
    return out, t_out
    

# @partial(jax.jit, static_argnames=('lmaxg', 'lmaxgp', 'lmaxr', 'lmaxnu', 'nqmax','max_steps'))
def evolve_one_mode( *, tau_max, tau_out, param, kmode, 
                        lmaxg : int, lmaxgp : int, lmaxr : int, lmaxnu : int,
                        nqmax : int, rtol: float, atol: float,
                        pcoeff : float, icoeff : float, dcoeff : float, factormax : float, factormin : float, max_steps : int, return_full : bool = False):

    jax.debug.print("Evolving k_mode={km}",km=kmode)
    modelX_ = VectorField(
        lambda tau, y, params : model_synchronous( idx = params[2], tau=tau, y=y, param=params[0], kmode=params[1],  
                                                   lmaxg=lmaxg, lmaxgp=lmaxgp, lmaxr=lmaxr, lmaxnu=lmaxnu, nqmax=nqmax) )
    modelX = drx.ODETerm( modelX_ )
    
    # ... determine the number of active variables (i.e. the number of equations), absent any optimizations
    nvar   = 7 + (lmaxg + 1) + (lmaxgp + 1) + (lmaxr + 1) + nqmax * (lmaxnu + 1) + 2

    # ... determine starting time
    tau_start = determine_starting_time( param=param, k=kmode )
    tau_start = 0.99 * jnp.minimum( jnp.min(tau_out), tau_start )

    # ... set adiabatic ICs
    y0 = adiabatic_ics_one_mode( tau=tau_start, param=param, kmode=kmode, nvar=nvar, 
                       lmaxg=lmaxg, lmaxgp=lmaxgp, lmaxr=lmaxr, lmaxnu=lmaxnu, nqmax=nqmax )

    #model_synchronous( tau=tau_start, y=y0, param=param, kmode=kmode, lmaxg=lmaxg, lmaxgp=lmaxgp, lmaxr=lmaxr, lmaxnu=lmaxnu, nqmax=nqmax)
    # create solver wrapper, we use the Kvaerno5 solver, which is a 5th order implicit solver
    def DEsolve_implicit(model, t0, t1, y0, kmode , idx):
        print(t0.shape, t1.shape, y0.shape, kmode.shape, idx.shape)
        return drx.diffeqsolve(
            terms=model,
            #solver=Rodas5Transformed(),
            solver=drx.Kvaerno5(),
            t0=t0.squeeze(),
            t1=t1.squeeze(),
            dt0=jnp.minimum(t0/4, 0.5*(t1-t0)).squeeze(),
            y0=y0,
            #saveat=saveat,

            ##########throw=False, ## HERE HERE HERE
            
            saveat = drx.SaveAt(dense=True),
            #saveat = drx.SaveAt(dense=True),
            stepsize_controller = drx.PIDController(rtol=rtol, atol=atol, norm=lambda t:rms_norm_filtered(t,jnp.array([0,2,3,5,6,7]), jnp.array([1,kmode**2,1,1,1/kmode**2,1])),
                                                    pcoeff=pcoeff, icoeff=icoeff, dcoeff=dcoeff, factormax=factormax, factormin=factormin),
            # default controller has icoeff=1, pcoeff=0, dcoeff=0
            max_steps=max_steps,
            args=(param, kmode, idx),
            # adjoint=drx.RecursiveCheckpointAdjoint(), # for backward differentiation
            adjoint=drx.DirectAdjoint(),  #for forward differentiation
            # adjoint=drx.BacksolveAdjoint(), # for backward differentiation
        )

    # solve before neutrinos become fluid
    #saveat = drx.SaveAt(ts=tau_out)
    # 2. Stack them into a unified PyTree layout
    #saveat = jax.tree_util.tree_stack(saveat_list)
    #sol = DEsolve_implicit( model=modelX, t0=jnp.min(tau_start), t1=jnp.max(tau_max), y0=y0, saveat=saveat, kmode=kmode )
    print("Y0 shape = ",y0.shape)

    sol_fn = jax.vmap(DEsolve_implicit, in_axes=(None, 1, 0, 1, None, 0))
    print(tau_start.shape, tau_max.shape, y0.shape,  kmode.shape, jnp.arange(y0.shape[1]).shape)
    sol = sol_fn(modelX, tau_start, tau_max, y0, kmode, jnp.arange(y0.shape[1]))

    #print(sol)
    #first_nan_idx = jnp.argmax(jnp.isinf(sol.ts))
    #last_valid_idx = first_nan_idx - 1
    #jax.debug.print("Stalled at time t = {t}", t=sol.ts[last_valid_idx])
    #jax.debug.print("State right before failure: {y}", y=jax.tree_util.tree_leaves(sol.ys)[0][last_valid_idx])

    print(tau_out.shape)
    extracted_ys = jax.vmap(
      lambda s, t_grid: jax.vmap(s.evaluate)(t_grid), 
      in_axes=(0, 0)
    )(sol, tau_out.T)
    
    print(extracted_ys.shape)

    ys_projected = jax.vmap( lambda tau, y: jax.vmap(
        lambda _tau, _y: apply_rsa_state_projection(
            y=_y,
            tau=_tau,
            kmode=kmode,
            param=param,
            lmaxg=lmaxg,
            lmaxgp=lmaxgp,
            lmaxr=lmaxr,
            nqmax=nqmax,
            nu_perturb_fn=nu_perturb,
        ))(tau, y),
    in_axes=(1, 0))(tau_out, extracted_ys)

    print(ys_projected.shape)
    if not return_full:
        # convert outputs
        yout = jax.vmap( lambda y_: jax.vmap( lambda y : convert_to_output_variables( y=y, param=param, kmode=kmode, lmaxg=lmaxg, lmaxgp=lmaxgp, lmaxr=lmaxr, lmaxnu=lmaxnu, nqmax=nqmax, idx=idx) )( y_ , idx), in_axes=(0,0))(ys_projected, jnp.arange(y0.shape[1]))
    else:
        # return full solution output
        yout = ys_projected

    print("Solution = ",yout.shape)
    return yout


#### NEW CODE FOR BATCHED CALCULATION ####

def determine_start_times_per_batch(tau_out, param, kmodes, batch_size):
    n_total = len(kmodes)

    # starting time for one moda:
    def starting_time_one_mode(p, k):
        tau_start = determine_starting_time( param=p, k=k )
        tau_start = 0.99 * jnp.minimum( jnp.min(tau_out), tau_start )

        return tau_start
    
     # ... determine starting times for each mode
    tau_start_all_modes = jax.vmap(starting_time_one_mode, (None, 0))( param, kmodes)

    # arange the starting times into batches
    tau_start_batches = jnp.array(jnp.split(tau_start_all_modes, n_total // batch_size))

    # finally pick the same starting time for each batch
    return jax.vmap(lambda a: jnp.min(a))(tau_start_batches)

def calculate_ics(tau_start_batched, kmode_batches, param, nvar, lmaxg, lmaxgp, lmaxr, lmaxnu, nqmax):
    
    def calculate_ic_one_mode(tau_start, kmode):
        return adiabatic_ics_one_mode( tau=tau_start, param=param, kmode=kmode, nvar=nvar, 
                       lmaxg=lmaxg, lmaxgp=lmaxgp, lmaxr=lmaxr, lmaxnu=lmaxnu, nqmax=nqmax )

    calc_ics_one_batch = jax.vmap(calculate_ic_one_mode, (None, 0))
     
    calc_ics_all_batches = jax.vmap(calc_ics_one_batch, (0, 0))

    # assert False, str((jnp.array(tau_start_batched).shape, jnp.array(kmode_batches).shape)) == ((32,), (32, 16))

    return calc_ics_all_batches(jnp.array(tau_start_batched), jnp.array(kmode_batches))


def reduce_state_ur_fluid(*, y, lmaxg, lmaxgp, lmaxr):
    """Remove massless-neutrino moments l>=3 from the state vector."""
    idxr = 9 + lmaxg + lmaxgp
    return jnp.concatenate((y[:idxr + 3], y[idxr + lmaxr + 1:]))


def expand_state_ur_fluid(*, y_reduced, lmaxg, lmaxgp, lmaxr):
    """Expand reduced UFA state back to full layout by padding zeroed moments."""
    idxr = 9 + lmaxg + lmaxgp
    n_removed = lmaxr - 2
    zeros = jnp.zeros((n_removed,), dtype=y_reduced.dtype)
    return jnp.concatenate((y_reduced[:idxr + 3], zeros, y_reduced[idxr + 3:]))

# @partial(jax.jit, static_argnames=('lmaxg', 'lmaxgp', 'lmaxr', 'lmaxnu', 'nqmax', 'max_steps', 'batch_size', 'return_full'))
def evolve_modes_batched( *, tau_max, tau_out, param, kmodes, 
                        lmaxg : int, lmaxgp : int, lmaxr : int, lmaxnu : int,
                        nqmax : int, rtol: float, atol: float,
                        pcoeff : float, icoeff : float, dcoeff : float, 
                        factormax : float, factormin : float, max_steps : int  , 
                        batch_size: int, return_full : bool = False):

    n_total = len(kmodes)
    n_batches = n_total // batch_size

    kmode_batches = jnp.array(jnp.split(kmodes, n_batches))

    def F(tau, y, kmode):
        return model_synchronous( tau=tau, y=y, param=param, kmode=kmode,  
                                                   lmaxg=lmaxg, lmaxgp=lmaxgp, lmaxr=lmaxr, lmaxnu=lmaxnu, nqmax=nqmax)

    # For Rodas5Batched with Diffrax 0.7.0:
    # Rodas5Batched extracts: f, _args = args
    # Then calls: terms.vf(t, y, _args)
    # We need terms.vf to vmap f over batch, but f is not passed to vf
    # Solution: Create a closure that captures F

    def vf_batched(t, y_batch, args):
        """
        Batched vector field for Rodas5Batched.

        Args:
            t: time scalar
            y_batch: (batch_size, nvars) batched states
            args: During compatibility check: tuple (f, batch_params)
                  During actual solve: just batch_params (kmodes)

        Returns:
            (batch_size, nvars) batched derivatives
        """
        # Handle both compatibility check and actual solve
        if isinstance(args, tuple) and len(args) == 2:
            # Compatibility check phase: args = (f, batch_params)
            # Return zeros for shape inference
            return jax.tree_util.tree_map(jnp.zeros_like, y_batch)
        else:
            # Actual solve phase: args = batch_params (kmodes)
            # vmap F over batch: F(t, y[i], kmode[i]) for each i
            return jax.vmap(F, in_axes=(None, 0, 0))(t, y_batch, args)

    modelX_term = drx.ODETerm(vf_batched)
    
    # ... determine the number of active variables (i.e. the number of equations), absent any optimizations
    nvar   = 7 + (lmaxg + 1) + (lmaxgp + 1) + (lmaxr + 1) + nqmax * (lmaxnu + 1) + 2

    # ... determine the starting times for each batch
    tau_start_batched = determine_start_times_per_batch(tau_out, param, kmodes, batch_size)

    # ... set adiabatic ICs for each kmode an batch together
    y0_batches = calculate_ics(tau_start_batched, kmode_batches, param, nvar, lmaxg, lmaxgp, lmaxr, lmaxnu, nqmax)

    # create solver wrapper
    def DEsolve_implicit(t0, y0, kmodes_batch):

        # This function now only returns the solution.ys and nothing else, since the other info is not used
        # at the moment. Might save some memory after compilation?

        filters_full = jax.vmap(lambda kmode: jnp.array([1,kmode**2,1,1,1/kmode**2,1]))(kmodes_batch)

        sol =  drx.diffeqsolve(
            terms=modelX_term,
            solver=Rodas5Batched(),
            t0=t0,
            t1=tau_max,
            dt0=jnp.minimum(t0/4, 0.5*(tau_max-t0)),
            y0=y0,
            saveat=drx.SaveAt(ts=tau_out),
            stepsize_controller = drx.PIDController(rtol=rtol, atol=atol,
                                                    norm=lambda t:rms_norm_filtered_batched(t, jnp.array([0,2,3,5,6,7]), filters_full),
                                                    pcoeff=pcoeff, icoeff=icoeff, dcoeff=dcoeff, factormax=factormax, factormin=factormin),
            max_steps=max_steps,
            args=(F, kmodes_batch),
            adjoint=drx.DirectAdjoint(),
        )

        if sol.result != dfx.RESULTS.successful:
          jax.debug.print("Solver failed with code: {r}",r=sol.result)

        return sol.ys


    DEsolve_implicit_vmap = jax.vmap(DEsolve_implicit, (0,0,0))

    ys = DEsolve_implicit_vmap(tau_start_batched, y0_batches, kmode_batches)

    n_batches, n_steps, _batch_size, _nvar = ys.shape

    # Transpose to (n_batches, batch_size, n_steps, nvar) then reshape to (n_total, n_steps, nvar)
    ys_transposed = jnp.transpose(ys, (0, 2, 1, 3))
    reordered_ys = jnp.reshape(ys_transposed, (n_total, n_steps, _nvar))

    projected_ys = jax.vmap(
        lambda _k, _ys: jax.vmap(
            lambda _tau, _y: apply_rsa_state_projection(
                y=_y,
                tau=_tau,
                kmode=_k,
                param=param,
                lmaxg=lmaxg,
                lmaxgp=lmaxgp,
                lmaxr=lmaxr,
                nqmax=nqmax,
                nu_perturb_fn=nu_perturb,
            )
        )(tau_out, _ys)
    )(kmodes, reordered_ys)

    if not return_full:
        # convert outputs
        def calculate_final_ys( kmode, ys): 
            return jax.vmap( lambda y : convert_to_output_variables( y=y, param=param, kmode=kmode, 
                                                       lmaxg=lmaxg, lmaxgp=lmaxgp, lmaxr=lmaxr, lmaxnu=lmaxnu, nqmax=nqmax) )( ys )
        
        return jax.vmap(calculate_final_ys, (0,0))(kmodes, projected_ys)
    else:
        # return full solution output
        return projected_ys



def evolve_perturbations( *, param, aexp_out, kmin : float, kmax : float, num_k : int,
                         lmaxg : int = 11, lmaxgp : int = 11, lmaxr : int = 11, lmaxnu : int = 8,
                         nqmax : int = 3, rtol: float = 1e-4, atol: float = 1e-4,
                         pcoeff : float = 0.25, icoeff : float = 0.80, dcoeff : float = 0.0,
                         factormax : float = 20.0, factormin : float = 0.3, max_steps : int = 2048, return_full : bool = False, k_sampling_method: str = 'camb'):
    """evolve cosmological perturbations in the synchronous gauge

    Parameters
    ----------
    param : dict
        dictionary of parameters and interpolated functions
        Optional TCA controls:
        - use_tca (bool): enable/disable tight-coupling approximation (default: False)
        - tca_tau_c_over_tau_h_trigger (float): TCA-to-full trigger threshold for tau_c/tau_h (default: 0.015)
        - tca_tau_c_over_tau_k_trigger (float): TCA-to-full trigger threshold for tau_c/tau_k (default: 0.010)
        Optional RSA controls:
        - use_rsa (bool): enable/disable radiation streaming approximation (default: True)
        - rsa_tau_c_over_tau_trigger (float): trigger threshold for tau_c/tau (default: 10.0)
        - rsa_tau_over_tau_k_trigger (float): trigger threshold for tau/tau_k = k*tau (default: 80.0)
        - use_ur_fluid (bool): enable/disable ultra-relativistic fluid approximation (default: True)
        - ur_fluid_tau_over_tau_k_trigger (float): trigger threshold for UFA k*tau (default: 120.0)
    aexp_out : jnp.ndarray
        array of scale factors at which to output
    kmin : float
        minimum wavenumber [in units 1/Mpc]
    kmax : float
        maximum wavenumber [in units 1/Mpc]
    num_k : int
        number of wavenumbers
    lmaxg : int
        maximum multipole for photon temperature
    lmaxgp : int
        maximum multipole for photon polarization
    lmaxr : int
        maximum multipole for massless neutrinos
    lmaxnu : int
        maximum multipole for massive neutrinos
    nqmax : int
        number of momentum bins for massive neutrinos
    rtol : float
        relative tolerance for ODE solver
    atol : float
        absolute tolerance for ODE solver
    k_sampling_method : str
        'log' for logarithmic, 'linear' for linear, 'camb' for CAMB-like hybrid sampling.

    Returns
    -------
    y : jnp.ndarray
        array of shape (num_k, nout, nvar) containing the perturbations
    k : jnp.ndarray
        array of shape (num_k) containing the wavenumbers [in units 1/Mpc]
    """
    if k_sampling_method == 'log':
        kmodes = jnp.geomspace(kmin, kmax, num_k)
    elif k_sampling_method == 'linear':
        kmodes = jnp.linspace(kmin, kmax, num_k)
    elif k_sampling_method == 'camb':
        if 'tau_maxvis' not in param:
            raise ValueError("param dictionary must contain 'tau_maxvis' for 'camb' k-sampling.")
        taurst = param['tau_maxvis']
        taurst = taurst.mean()
        
        # Simplified CAMB-like sampling
        q_switch1 = 8.0 / taurst
        q_switch2 = 30.0 / taurst
        
        n1 = int(num_k * 0.2)
        n2 = int(num_k * 0.4)
        n3 = num_k - n1 - n2
        
        k1 = jnp.geomspace(kmin, q_switch1, n1, endpoint=False)
        k2 = jnp.linspace(q_switch1, q_switch2, n2, endpoint=False)
        k3 = jnp.geomspace(q_switch2, kmax, n3)
        kmodes = jnp.concatenate([k1, k2, k3])
    else:
        raise ValueError(f"Unknown k_sampling_method: {k_sampling_method}")
    

    # determine output times from aexp_out
    aexp_out = jnp.atleast_1d(aexp_out)
    tau_out = jax.vmap( lambda a: param['tau_of_a_spline'].evaluate(a) )(aexp_out).squeeze(1)
    
    #global_min = jnp.min(tau_out)
    #global_max = jnp.max(tau_out)

    #ref_track = jnp.mean(tau_out, axis=1)
    #normalized_profile = (ref_track - jnp.min(ref_track)) / (jnp.max(ref_track) - jnp.min(ref_track))
    #master_tau = global_min + normalized_profile * (global_max - global_min)
    #tau_max = jnp.max(master_tau)
    tau_max = jnp.max(tau_out, axis=0)
    nout = aexp_out.shape[0]
    
    # set up ICs and solve ODEs for all the modes
    y1 = jax.lax.map(
        lambda k : evolve_one_mode( tau_max=tau_max, tau_out=tau_out, 
                                    param=param, kmode=k, lmaxg=lmaxg, lmaxgp=lmaxgp, lmaxr=lmaxr, 
                                    lmaxnu=lmaxnu, nqmax=nqmax, rtol=rtol, atol=atol,
                                    pcoeff=pcoeff, icoeff=icoeff, dcoeff=dcoeff, 
                                    factormax=factormax, factormin=factormin, max_steps=max_steps, return_full=return_full )
                    , kmodes)

    param['lmaxg'] = lmaxg
    param['lmaxgp'] = lmaxgp
    param['lmaxr'] = lmaxr
    param['lmaxnu'] = lmaxnu
    param['nqmax'] = nqmax
    param['nout'] = nout
    param['tau_out'] = tau_out
    
    return y1, kmodes, param


@partial(jax.jit, static_argnames=('kmin', 'kmax', 'num_k', 'lmaxg', 'lmaxgp', 'lmaxr', 'lmaxnu', 'nqmax', 'max_steps', 'batch_size', 'return_full', 'k_sampling_method'))
def evolve_perturbations_batched( *, param, aexp_out, kmin : float, kmax : float, num_k : int,
                         lmaxg : int = 11, lmaxgp : int = 11, lmaxr : int = 11, lmaxnu : int = 8,
                         nqmax : int = 3, rtol: float = 1e-4, atol: float = 1e-4,
                         pcoeff : float = 0.25, icoeff : float = 0.80, dcoeff : float = 0.0,
                         factormax : float = 20.0, factormin : float = 0.3, max_steps : int = 4096 , 
                         batch_size: int = 16, return_full : bool = False, k_sampling_method: str = 'camb'):
    """evolve cosmological perturbations in the synchronous gauge

    Parameters
    ----------
    param : dict
        dictionary of parameters and interpolated functions
        Optional TCA controls:
        - use_tca (bool): enable/disable tight-coupling approximation (default: False)
        - tca_tau_c_over_tau_h_trigger (float): TCA-to-full trigger threshold for tau_c/tau_h (default: 0.015)
        - tca_tau_c_over_tau_k_trigger (float): TCA-to-full trigger threshold for tau_c/tau_k (default: 0.010)
        Optional RSA controls:
        - use_rsa (bool): enable/disable radiation streaming approximation (default: True)
        - rsa_tau_c_over_tau_trigger (float): trigger threshold for tau_c/tau (default: 10.0)
        - rsa_tau_over_tau_k_trigger (float): trigger threshold for tau/tau_k = k*tau (default: 80.0)
        - use_ur_fluid (bool): enable/disable ultra-relativistic fluid approximation (default: True)
        - ur_fluid_tau_over_tau_k_trigger (float): trigger threshold for UFA k*tau (default: 120.0)
    aexp_out : jnp.ndarray
        array of scale factors at which to output
    kmin : float
        minimum wavenumber [in units 1/Mpc]
    kmax : float
        maximum wavenumber [in units 1/Mpc]
    num_k : int
        number of wavenumbers
    lmaxg : int
        maximum multipole for photon temperature
    lmaxgp : int
        maximum multipole for photon polarization
    lmaxr : int
        maximum multipole for massless neutrinos
    lmaxnu : int
        maximum multipole for massive neutrinos
    nqmax : int
        number of momentum bins for massive neutrinos
    rtol : float
        relative tolerance for ODE solver
    atol : float
        absolute tolerance for ODE solver
    batch_size: int
        number of modes to batch together for ODE solver
    return_full : bool
        if True, return full state vector; if False, return converted output variables
    k_sampling_method : str
        'log' for logarithmic, 'linear' for linear, 'camb' for CAMB-like hybrid sampling.

    Returns
    -------
    y : jnp.ndarray
        array of shape (num_k, nout, nvar) containing the perturbations
    k : jnp.ndarray
        array of shape (num_k) containing the wavenumbers [in units 1/Mpc]
    param : dict
        updated parameter dictionary with output information
    """
    if k_sampling_method == 'log':
        kmodes = jnp.geomspace(kmin, kmax, num_k)
    elif k_sampling_method == 'linear':
        kmodes = jnp.linspace(kmin, kmax, num_k)
    elif k_sampling_method == 'camb':
        if 'tau_maxvis' not in param:
            raise ValueError("param dictionary must contain 'tau_maxvis' for 'camb' k-sampling.")
        taurst = param['tau_maxvis']
        
        # Simplified CAMB-like sampling
        q_switch1 = 8.0 / taurst
        q_switch2 = 30.0 / taurst
        
        n1 = int(num_k * 0.2)
        n2 = int(num_k * 0.4)
        n3 = num_k - n1 - n2
        
        k1 = jnp.geomspace(kmin, q_switch1, n1, endpoint=False)
        k2 = jnp.linspace(q_switch1, q_switch2, n2, endpoint=False)
        k3 = jnp.geomspace(q_switch2, kmax, n3)
        kmodes = jnp.concatenate([k1, k2, k3])
    else:
        raise ValueError(f"Unknown k_sampling_method: {k_sampling_method}")
    

    # determine output times from aexp_out
    aexp_out = jnp.atleast_1d(aexp_out)
    tau_out = jax.vmap( lambda a: param['tau_of_a_spline'].evaluate(a) )(aexp_out)
    tau_max = jnp.max(tau_out)
    nout = aexp_out.shape[0]

    # do all calculations batched in here
    y1 = evolve_modes_batched(tau_max=tau_max, tau_out=tau_out, param=param, kmodes=kmodes, lmaxg=lmaxg, lmaxgp=lmaxgp, lmaxr=lmaxr, 
                                    lmaxnu=lmaxnu, nqmax=nqmax, rtol=rtol, atol=atol,
                                    pcoeff=pcoeff, icoeff=icoeff, dcoeff=dcoeff, 
                                    factormax=factormax, factormin=factormin, max_steps=max_steps,
                                    batch_size=batch_size, return_full=return_full )
    
    # Store parameters in param dict for compatibility with non-batched version
    param['lmaxg'] = lmaxg
    param['lmaxgp'] = lmaxgp
    param['lmaxr'] = lmaxr
    param['lmaxnu'] = lmaxnu
    param['nqmax'] = nqmax
    param['nout'] = nout
    param['tau_out'] = tau_out
    
    return y1, kmodes, param

@partial(jax.jit, static_argnames=('lmaxg', 'lmaxgp', 'lmaxr', 'lmaxnu', 'nqmax', 'idxcosmo'))
def compute_time_derivatives(yout, tau, kmodes, param, lmaxg, lmaxgp, lmaxr, lmaxnu, nqmax, idxcosmo):
    """Compute time derivatives by re-evaluating the ODE system.

    This function computes dy/dτ by re-evaluating the synchronous gauge
    perturbation equations at each output time and k-mode. This is useful
    for CMB source function calculations that require time derivatives.

    Parameters
    ----------
    yout : jnp.ndarray
        State vector array with shape (n_kmodes, n_times, n_vars)
    tau : jnp.ndarray
        Conformal time array with shape (n_times,)
    kmodes : jnp.ndarray
        Wavenumber array with shape (n_kmodes,)
    param : dict
        Parameter dictionary (excluding static integer parameters)
    lmaxg : int
        Maximum photon temperature multipole (static)
    lmaxgp : int
        Maximum photon polarization multipole (static)
    lmaxr : int
        Maximum massless neutrino multipole (static)
    lmaxnu : int
        Maximum massive neutrino multipole (static)
    nqmax : int
        Number of neutrino momentum bins (static)

    Returns
    -------
    jnp.ndarray
        Time derivatives with same shape as yout (n_kmodes, n_times, n_vars)

    Examples
    --------
    >>> yout, kmodes, param = evolve_perturbations_batched(...)
    >>> tau = param['tau_out']
    >>> lmaxg = param['lmaxg']
    >>> lmaxgp = param['lmaxgp']
    >>> lmaxr = param['lmaxr']
    >>> lmaxnu = param['lmaxnu']
    >>> nqmax = param['nqmax']
    >>> yprime = compute_time_derivatives(yout, tau, kmodes, param, lmaxg, lmaxgp, lmaxr, lmaxnu, nqmax)
    """

    # Create indices for vectorization
    idxtau = jnp.arange(len(tau))
    idxk = jnp.arange(len(kmodes))

    # Compute derivatives by re-evaluating the ODE system
    yprime = jax.vmap(
        lambda ik: jax.vmap(
            lambda itau: model_synchronous(
                tau=tau[itau],
                y=yout[ik, itau, :],
                param=param,
                kmode=kmodes[ik],
                lmaxg=lmaxg,
                lmaxgp=lmaxgp,
                lmaxr=lmaxr,
                lmaxnu=lmaxnu,
                nqmax=nqmax,
                idx=idxcosmo
            )
        )(idxtau)
    )(idxk)

    return yprime


# @partial(jax.jit, static_argnames=('N'))
def get_xi_from_P( *, k : jnp.ndarray, Pk : jnp.ndarray, N : int, ell : int = 0 ):
    """ get the correlation function from the power spectrum  using FFTlog, cf.
        J. D. Talman (1978). JCP, 29:35-48		
        A. J. S. Hamilton (2000).  MNRAS, 312:257-284

    Args:
        k (array_like)   : the wavenumbers [units 1/Mpc]
        Pk (array_like)  : the power spectrum
        N (int)          : length of the input vector
        ell (int)        : the multipole to compute (0,2,4,...)

    Returns:
        xi (array_like)  : the correlation function
        r (array_like)   : the radii [units in Mpc]
    """
    N = len(k)
    kmin = k[0]
    kmax = k[N-1]

    L = jnp.log(kmax/kmin)

    # FFTlog algorithm:
    fPk = jnp.fft.rfft( Pk * k**1.5 )

    ki = jnp.pi * jnp.arange( N//2+1 ) / L
    zp = (1.5+ell)/2 + 1j* ki

    theta = jax.vmap( lambda z: jnp.imag( lngamma_complex_e( z ) ) )( zp )

    fPk = fPk * jnp.exp( 2j * (theta - jnp.log(jnp.pi) * ki) )

    r  = 2*jnp.pi/k
    xi = jnp.real( 1j**ell * jnp.fft.irfft( fPk ) / (2*jnp.pi*r)**1.5 )
    return xi[::-1], r[::-1] # reverse order since 1/k is decreasing for increasing k


def get_power( *, k : jax.Array, y : jax.Array, idx : int , param : dict) -> jax.Array:
    """ compute the power spectrum from the perturbations, field indices are:

            eta, etaprime, hprime, alpha,       # 0-3
            deltam,  thetam / (aH),             # 4,5
            deltabc, thetabc / (aH),            # 6,7
            deltac,  thetac / (aH),             # 8,9
            deltab,  thetab / (aH),             # 10,11
            deltag,  thetag / (aH),             # 12,13
            deltar,  thetar / (aH),             # 14,15
            deltanu, thetanu / (aH),            # 16,17
            deltaq,  thetaq / (aH),             # 18,19
    
    Args:
        k (array_like)   : the wavenumbers [in units 1/Mpc]
        y (array_like)   : the perturbations
        idx (int)        : index of the perturbation to compute the power spectrum for
        param (dict)     : dictionary of all parameters
        
    Returns:
        Pk (array_like)  : the power spectrum
    """
    return 2 * jnp.pi**2 * param['A_s'] *(k/param['k_p'])**(param['n_s'] - 1) * k**(-3) * y[...,idx]**2


def get_power_smoothed( *, k : jax.Array, y : jax.Array, dlogk : float, idx : int , param : dict) -> jax.Array:
    """ compute Savitzky-Golay smoothed version of the power spectrum, field indices are:

            eta, etaprime, hprime, alpha,       # 0-3
            deltam,  thetam / (aH),             # 4,5
            deltabc, thetabc / (aH),            # 6,7
            deltac,  thetac / (aH),             # 8,9
            deltab,  thetab / (aH),             # 10,11
            deltag,  thetag / (aH),             # 12,13
            deltar,  thetar / (aH),             # 14,15
            deltanu, thetanu / (aH),            # 16,17
            deltaq,  thetaq / (aH),             # 18,19
    
    Args:
        k (array_like)   : the wavenumbers [in units 1/Mpc]
        y (array_like)   : the perturbations
        dlogk (float)    : the log bin width (dlogk = 1.0)
        idx (int)        : index of the perturbation to compute the power spectrum for
        param (dict)     : dictionary of all parameters
        
    Returns:
        Pk (array_like)  : the power spectrum
    """
    window_length = round(dlogk/(jnp.log(k[1])-jnp.log(k[0])))
    window_length += (window_length+1)%2

    Pm  = get_power( y=y, k=k, idx=idx, param=param )

    Pms = jnp.exp(savgol_filter(y=jnp.log(Pm), window_length=window_length, polyorder=3))

    # replace boundary affected regions with original signal
    Pms = Pms.at[:window_length//2].set( Pm[:window_length//2] )
    Pms = Pms.at[-window_length//2:].set( Pm[-window_length//2:] )

    return Pms

def power_Kaiser( *, y : jax.Array, kmodes : jax.Array, bias : float, mu_sampling : bool = True, smooth_dlogk : float = None, nmu : int, param : dict) -> tuple[jax.Array, jax.Array]:
    """ compute the anisotropic power spectrum using the Kaiser formula
    
    Args:
        y (array_like)       : input solution from the EB solver
        kmodes (array_like)  : the list of wave numbers in units of [1/Mpc]
        bias (float)         : linear tracer bias
        mu_sampling (bool)   : if True, sample the mu bins, else sample the theta bins
        smooth_dlogk (float) : if not None, use Savitzky-Golay smoothing at this log scale
        sigma_z0 (float)     : redshift error sigma_z = sigma_z0 * (1+z)
        nmu (int)            : number of mu bins
        param (dict)         : dictionary of all data

    Returns:
        P(k,mu) (array_like) : anisotropic spectrum
        mu (array_like)      : mu bins
    """
    
    if mu_sampling:
        mu = jnp.linspace(-1,1,nmu)
    else:
        theta = jnp.linspace(0,jnp.pi,nmu)
        mu = jnp.cos(theta)
    
    if smooth_dlogk is None:
        fac = 2 * jnp.pi**2 * param['A_s']
        deltam = jnp.sqrt(fac *(kmodes/param['k_p'])**(param['n_s'] - 1) * kmodes**(-3)) * y[:,4]
        thetam = jnp.sqrt(fac *(kmodes/param['k_p'])**(param['n_s'] - 1) * kmodes**(-3)) * y[:,5]
    else:
        Pdelta = get_power_smoothed( y=y, k=kmodes, dlogk=smooth_dlogk, idx=4, param=param )
        Ptheta = get_power_smoothed( y=y, k=kmodes, dlogk=smooth_dlogk, idx=5, param=param )
        deltam = jnp.sqrt(Pdelta)
        thetam = -jnp.sqrt(Ptheta)

    # thetam already contains 1/ mathcal{H} factor   -f delta = theta
    Pkmu = (bias*deltam[:,None] - mu[None,:]**2 * thetam[:,None])**2
    return Pkmu, mu



def power_multipoles( *, y : jnp.ndarray, kmodes : jnp.ndarray, b : float, param ) -> tuple[jax.Array]:
    """ compute the power spectrum multipoles (l=0,2,4)

    Args:
        y (array_like)       : input solution from the EB solver
        kmodes (array_like)  : the list of wave numbers [in units 1/Mpc]
        b (float)            : linear bias

    Returns:
        P0 (array_like)      : monopole
        P2 (array_like)      : quadrupole
        P4 (array_like)      : hexadecapole
    """
    fac = 2 * jnp.pi**2 * param['A_s']
    deltam = jnp.sqrt(fac *(kmodes/param['k_p'])**(param['n_s'] - 1) * kmodes**(-3)) * y[:,4]
    thetam = jnp.sqrt(fac *(kmodes/param['k_p'])**(param['n_s'] - 1) * kmodes**(-3)) * y[:,5]

    # powerspectrum multipoles
    P0 = b**2 * deltam**2 - 2*b/3 * deltam*thetam + 1/5*thetam**2
    P2 = -4*b/3 * deltam * thetam + 4/7 * thetam**2
    P4 = 8/35 * thetam**2

    return P0, P2, P4
# Hierarchy truncations. These are module-level integers (not runtime arguments)
# so the state-vector layout and the accelerator-compiled right-hand side treat
# them as compile-time constants. The current flat-LambdaCDM + massless-neutrino
# layout has NVAR == 50 (11-var dense core + 3x13 free-streaming hierarchies),
# matching the hand-tuned Schur-EB LU solver in :mod:`discoeb.schur_eb`.
LMAX_G = 15
"""Photon temperature hierarchy truncation (``Theta_0 ... Theta_LMAX_G``)."""

LMAX_POL = 15
"""Photon E-mode polarization hierarchy truncation (``E_2 ... E_LMAX_POL``)."""

LMAX_NR = 15
"""Massless-neutrino hierarchy truncation (``N_0 ... N_LMAX_NR``)."""

IX_ETAK = 0
"""State-vector index of the metric perturbation ``etak = k eta``."""

IX_CLXC = 1
"""State-vector index of the CDM density contrast ``clxc``."""

IX_CLXB = 2
"""State-vector index of the baryon density contrast ``clxb``."""

IX_VB = 3
"""State-vector index of the baryon velocity ``vb``."""

IX_G = 4
"""Base index of the photon temperature hierarchy (``Theta_l`` at ``IX_G + l``)."""

IX_POL = IX_G + LMAX_G + 1
"""Base index of the photon polarization hierarchy (``E_l`` at ``IX_POL + (l-2)``)."""

IX_R = IX_POL + LMAX_POL - 1
"""Base index of the massless-neutrino hierarchy (``N_l`` at ``IX_R + l``)."""

NVAR = IX_R + LMAX_NR + 1
"""Total length of the perturbation state vector."""


# ============================================================
# EINSTEIN CONSTRAINTS AND BACKGROUND TERMS
# ============================================================


def comoving_densities(
    a: float,
    grhog: float,
    grhornomass: float,
    grhoc: float,
    grhob: float,
    grhov: float,
) -> tuple[float, float, float, float, float]:
    """Return the ``8*pi*G*rho_i a^2`` density coefficients at scale factor ``a``.

    The CAMB ``grho`` coefficients satisfy ``grhoa4 = 8*pi*G*rho_i a^4``. Dividing
    by ``a^2`` gives the combinations that appear in the synchronous-gauge
    Einstein equations:

    ``rho_gamma, rho_nu ~ a^-4`` give ``grho_i a^2 = grho_i / a^2``;
    ``rho_c, rho_b ~ a^-3`` give ``grho_i a^2 = grho_i / a``;
    ``rho_Lambda ~ a^0`` gives ``grhov a^2 = grhov * a^2``.

    Returns ``(grhog_t, grhor_t, grhoc_t, grhob_t, grhov_t)``.
    """

    a2 = a * a
    return (grhog / a2, grhornomass / a2, grhoc / a, grhob / a, grhov * a2)


def expansion_rate(
    grhog_t: float,
    grhor_t: float,
    grhoc_t: float,
    grhob_t: float,
    grhov_t: float,
) -> float:
    """Return the conformal Hubble rate ``adotoa = a'/a = aH``.

    ``adotoa = sqrt((Sum grho_i a^2) / 3)``. Written as ``** 0.5`` so the helper
    traces unchanged on Python floats and CUDA device values.
    """

    return ((grhog_t + grhor_t + grhoc_t + grhob_t + grhov_t) / 3.0) ** 0.5


def density_perturbation(
    grhob_t: float,
    clxb: float,
    grhoc_t: float,
    clxc: float,
    grhog_t: float,
    clxg: float,
    grhor_t: float,
    clxr: float,
) -> float:
    """Return the total density perturbation ``dgrho = Sum grho_i a^2 delta_i``."""

    return grhob_t * clxb + grhoc_t * clxc + grhog_t * clxg + grhor_t * clxr


def momentum_perturbation(
    grhob_t: float,
    vb: float,
    grhog_t: float,
    qg: float,
    grhor_t: float,
    qr: float,
) -> float:
    """Return the total momentum-density source ``dgq = Sum grho_i a^2 (rho+p) v_i``."""

    return grhob_t * vb + grhog_t * qg + grhor_t * qr


def anisotropic_stress(
    grhog_t: float,
    pig: float,
    grhor_t: float,
    pir: float,
) -> float:
    """Return the total anisotropic-stress source ``dgpi``."""

    return grhog_t * pig + grhor_t * pir


def metric_z(dgrho: float, etak: float, adotoa: float, k: float) -> float:
    """Return the synchronous-gauge metric variable ``z = (dgrho/(2k) + k eta)/aH``."""

    return (0.5 * dgrho / k + etak) / adotoa


def shear_sigma(z: float, dgq: float, k: float) -> float:
    """Return the metric shear ``sigma = z + (3/2) dgq / k^2``."""

    return z + 1.5 * dgq / (k * k)


def newtonian_potential(
    dgrho: float,
    dgq: float,
    dgpi: float,
    adotoa: float,
    k: float,
) -> float:
    """Return the Newtonian-gauge curvature potential ``Phi`` (diagnostic).

    ``Phi = -[(dgrho + 3 aH dgq / k) + dgpi] / (2 k^2)``.
    """

    return -((dgrho + 3.0 * dgq * adotoa / k) + dgpi) / (2.0 * k * k)


# ============================================================
# ADIABATIC INITIAL CONDITIONS
# Deep in radiation domination (k tau << 1), following CAMB's initial().
# ============================================================


def neutrino_radiation_fraction(grhog: float, grhornomass: float) -> float:
    """Return the neutrino fraction of the radiation density ``R_nu``."""

    return grhornomass / (grhog + grhornomass)


def matter_radiation_parameter(
    grhob: float,
    grhoc: float,
    grhog: float,
    grhornomass: float,
) -> float:
    """Return the matter-to-radiation initial-condition parameter ``om``."""

    return (grhob + grhoc) / (3.0 * (grhog + grhornomass)) ** 0.5


def adiabatic_photon_density(k: float, tau: float, om: float) -> float:
    """Return the initial photon density contrast ``delta_gamma``."""

    x2 = (k * tau) ** 2
    return x2 / 3.0 * (1.0 - om * tau / 5.0)


def adiabatic_photon_velocity(k: float, tau: float, om: float) -> float:
    """Return the initial photon dipole ``q_gamma``."""

    x = k * tau
    return x**3 / 27.0 * (1.0 - om * tau / 5.0)


def adiabatic_initial_conditions(
    k: float,
    tau: float,
    grhog: float,
    grhornomass: float,
    grhoc: float,
    grhob: float,
) -> tuple[float, ...]:
    """Return the adiabatic initial state vector deep in radiation domination.

    Implements CAMB's ``initial()`` series for the adiabatic growing mode at
    ``k tau << 1``, normalized so the comoving curvature perturbation is unity.
    The returned tuple has length :data:`NVAR`; all multipoles above those set
    explicitly start at zero.
    """

    x = k * tau
    x2 = x * x

    Rv = neutrino_radiation_fraction(grhog, grhornomass)
    Rp15 = 4.0 * Rv + 15.0
    om = matter_radiation_parameter(grhob, grhoc, grhog, grhornomass)
    omtau = om * tau

    clxg = adiabatic_photon_density(k, tau, om)
    qg = adiabatic_photon_velocity(k, tau, om)

    y0 = [0.0] * NVAR

    # Metric perturbation: etak = k eta ~ -k at leading order.
    y0[IX_ETAK] = -k * (1.0 - x2 / 12.0 * (-10.0 / Rp15 + 1.0))

    # Photon monopole and dipole.
    y0[IX_G] = clxg
    y0[IX_G + 1] = qg

    # CDM and baryons share the photon density (3/4 factor for adiabatic mode).
    y0[IX_CLXC] = 0.75 * clxg
    y0[IX_CLXB] = 0.75 * clxg
    y0[IX_VB] = 0.75 * qg

    # Massless neutrinos.
    y0[IX_R] = clxg
    y0[IX_R + 1] = (4.0 * Rv + 23.0) / Rp15 * x2 * x / 27.0
    y0[IX_R + 2] = (
        -4.0
        / 3.0
        * x2
        / Rp15
        * (1.0 + omtau / 4.0 * (4.0 * Rv - 5.0) / (2.0 * Rv + 15.0))
    )
    if LMAX_NR >= 3:
        y0[IX_R + 3] = -4.0 / 21.0 / Rp15 * x2 * x

    return tuple(y0)


# ============================================================
# FLUID AND METRIC DERIVATIVES
# ============================================================


def metric_etak_derivative(dgq: float) -> float:
    """Return ``etak' = (1/2) dgq`` from the momentum Einstein constraint."""

    return 0.5 * dgq


def cdm_density_derivative(z: float, k: float) -> float:
    """Return ``clxc' = -k z`` (CDM is pressureless, at rest in synchronous gauge)."""

    return -k * z


def baryon_density_derivative(z: float, vb: float, k: float) -> float:
    """Return ``clxb' = -k (z + v_b)`` from baryon number conservation."""

    return -k * (z + vb)


def baryon_velocity_derivative(
    adotoa: float,
    vb: float,
    k: float,
    delta_p_b: float,
    photbar: float,
    opacity: float,
    qg: float,
) -> float:
    """Return ``v_b'`` from the full (non-tight-coupling) baryon Euler equation.

    ``v_b' = -aH v_b + k delta_p_b - (rho_gamma/rho_b) kappa' (4/3 v_b - q_gamma)``.
    """

    return -adotoa * vb + k * delta_p_b - photbar * opacity * (4.0 / 3.0 * vb - qg)


def photon_monopole_derivative(z: float, qg: float, k: float) -> float:
    """Return ``Theta_0' = -k (4/3 z + q_gamma)`` (photon continuity)."""

    return -k * (4.0 / 3.0 * z + qg)


def photon_dipole_derivative(
    vbdot: float,
    adotoa: float,
    vb: float,
    k: float,
    delta_p_b: float,
    pb43: float,
    clxg: float,
    pig: float,
) -> float:
    """Return the photon dipole derivative ``Theta_1' = q_gamma'``.

    ``q_gamma' = (4/3)(-v_b' - aH v_b + k delta_p_b)/R_gamma + (k/3) delta_gamma
    - (2k/3) pi_gamma``, with ``pb43 = R_gamma = (4/3) rho_gamma/rho_b``.
    """

    return (
        4.0 / 3.0 * (-vbdot - adotoa * vb + k * delta_p_b) / pb43
        + k / 3.0 * clxg
        - 2.0 * k / 3.0 * pig
    )


def polarization_source(pig: float, e2: float) -> float:
    """Return the polarization source ``Pi = pi_gamma/10 + (9/15) E_2``."""

    return pig / 10.0 + 9.0 / 15.0 * e2


def photon_quadrupole_derivative(
    qg: float,
    theta3: float,
    pig: float,
    polter: float,
    sigma: float,
    k: float,
    opacity: float,
) -> float:
    """Return ``Theta_2' = (2k/5) q - (3k/5) Theta_3 - kappa'(pi - Pi) + (8/15) k sigma``."""

    return (
        2.0 * k / 5.0 * qg
        - 3.0 * k / 5.0 * theta3
        - opacity * (pig - polter)
        + 8.0 / 15.0 * k * sigma
    )


def photon_temperature_multipole_derivative(
    l: int,
    theta_lm1: float,
    theta_l: float,
    theta_lp1: float,
    k: float,
    opacity: float,
) -> float:
    """Return ``Theta_l'`` for an interior multipole ``3 <= l < LMAX_G``."""

    return (
        k * l / (2 * l + 1) * theta_lm1
        - k * (l + 1) / (2 * l + 1) * theta_lp1
        - opacity * theta_l
    )


def photon_temperature_truncation(
    theta_lm1: float,
    theta_l: float,
    k: float,
    opacity: float,
    tau: float,
    lmax: int,
) -> float:
    """Return the truncated photon temperature derivative at ``l = LMAX_G``."""

    return k * theta_lm1 - (lmax + 1) / tau * theta_l - opacity * theta_l


def polarization_quadrupole_derivative(
    e2: float,
    e3: float,
    polter: float,
    k: float,
    opacity: float,
) -> float:
    """Return ``E_2' = -kappa'(E_2 - Pi) - (k/3) E_3``."""

    return -opacity * (e2 - polter) - k / 3.0 * e3


def polarization_multipole_derivative(
    l: int,
    e_lm1: float,
    e_l: float,
    e_lp1: float,
    k: float,
    opacity: float,
) -> float:
    """Return ``E_l'`` for an interior multipole ``3 <= l < LMAX_POL``."""

    polfac = (l + 3) * (l - 1) / (l + 1)
    return (
        -opacity * e_l + k * l / (2 * l + 1) * e_lm1 - polfac * k / (2 * l + 1) * e_lp1
    )


def polarization_truncation(
    e_lm1: float,
    e_l: float,
    k: float,
    opacity: float,
    tau: float,
    lmax: int,
) -> float:
    """Return the truncated E-mode derivative at ``l = LMAX_POL``."""

    return -opacity * e_l + k * lmax / (2 * lmax + 1) * e_lm1 - (lmax + 3) / tau * e_l


def neutrino_monopole_derivative(z: float, qr: float, k: float) -> float:
    """Return ``N_0' = -k (4/3 z + q_nu)`` (neutrino continuity)."""

    return -k * (4.0 / 3.0 * z + qr)


def neutrino_dipole_derivative(clxr: float, pir: float, k: float) -> float:
    """Return ``N_1' = (k/3)(delta_nu - 2 pi_nu)`` (neutrino momentum)."""

    return k / 3.0 * (clxr - 2.0 * pir)


def neutrino_quadrupole_derivative(
    qr: float,
    n3: float,
    sigma: float,
    k: float,
) -> float:
    """Return ``N_2' = (2k/5) q_nu - (3k/5) N_3 + (8/15) k sigma``."""

    return 2.0 * k / 5.0 * qr - 3.0 * k / 5.0 * n3 + 8.0 / 15.0 * k * sigma


def neutrino_multipole_derivative(
    l: int,
    n_lm1: float,
    n_l: float,
    n_lp1: float,
    k: float,
) -> float:
    """Return ``N_l'`` for an interior multipole ``3 <= l < LMAX_NR``."""

    return k * l / (2 * l + 1) * n_lm1 - k * (l + 1) / (2 * l + 1) * n_lp1


def neutrino_truncation(
    n_lm1: float,
    n_l: float,
    k: float,
    tau: float,
    lmax: int,
) -> float:
    """Return the truncated neutrino derivative ``N_L' = k N_{L-1} - (L+1) N_L / tau``."""

    return k * n_lm1 - (lmax + 1) / tau * n_l


# ============================================================
# DYNAMICAL DARK ENERGY (CPL fluid perturbations)
#
# Optional extension (enabled when w_0 != -1 or w_a != 0). The DE fluid is a
# clustering fluid with rest-frame sound speed cs2_DE, carrying two extra state
# variables clxq (density contrast) and thetaq (velocity divergence) that join
# the densely-coupled metric core. Synchronous-gauge equations follow Ballesteros
# & Lesgourgues 2010; the metric enters through ``0.5 h' = k z`` (see
# :func:`cdm_density_derivative`, where ``clxc' = -k z = -h'/2``).
# ============================================================


def dark_energy_equation_of_state(a: float, w_DE_0: float, w_DE_a: float) -> float:
    """Return the CPL equation of state ``w_Q(a) = w_0 + w_a (1 - a)``."""

    return w_DE_0 + w_DE_a * (1.0 - a)


def dark_energy_adiabatic_sound_speed(
    w_Q: float, w_Q_prime: float, adotoa: float
) -> float:
    """Return the adiabatic sound speed ``c_a^2 = w - w'/(3(1+w) aH)``."""

    return w_Q - w_Q_prime / (3.0 * (1.0 + w_Q) * adotoa)


def dark_energy_density_coefficient(grhov: float, rho_Q: float, a: float) -> float:
    """Return the comoving DE density coefficient ``8*pi*G*rho_Q a^2``.

    ``grhov = grhom * OmegaDE`` is the present DE density coefficient and
    ``rho_Q = rho_Q(a)/rho_Q(1)`` its normalized evolution, so the comoving
    coefficient is ``grhov * rho_Q * a^2`` (reducing to the flat ``grhov a^2``
    when ``rho_Q = 1``).
    """

    return grhov * rho_Q * a * a


def dark_energy_fluid_derivatives(
    clxq: float,
    thetaq: float,
    z: float,
    adotoa: float,
    k: float,
    a: float,
    w_DE_0: float,
    w_DE_a: float,
    cs2_Q: float,
) -> tuple[float, float]:
    """Return the DE fluid derivatives ``(clxq', thetaq')`` in synchronous gauge.

    ``clxq' = -(1+w)(thetaq + k z) - 3(cs2 - w) aH clxq
              - 9(1+w)(cs2 - c_a^2)(aH)^2 thetaq / k^2``,
    ``thetaq' = -(1 - 3 cs2) aH thetaq + cs2 k^2 clxq / (1+w)``,

    with ``w = w(a)``, ``w' = -w_a aH a``, and ``c_a^2`` the adiabatic sound
    speed. Valid only for ``w != -1`` (a cosmological constant carries no fluid
    perturbations and is excluded from the state layout).
    """

    w_Q = dark_energy_equation_of_state(a, w_DE_0, w_DE_a)
    w_Q_prime = -w_DE_a * adotoa * a
    ca2_Q = dark_energy_adiabatic_sound_speed(w_Q, w_Q_prime, adotoa)

    clxq_prime = (
        -(1.0 + w_Q) * (thetaq + k * z)
        - 3.0 * (cs2_Q - w_Q) * adotoa * clxq
        - 9.0 * (1.0 + w_Q) * (cs2_Q - ca2_Q) * adotoa**2 / k**2 * thetaq
    )
    thetaq_prime = (
        -(1.0 - 3.0 * cs2_Q) * adotoa * thetaq + cs2_Q / (1.0 + w_Q) * k**2 * clxq
    )
    return clxq_prime, thetaq_prime


# ============================================================
# FULL RIGHT-HAND SIDE
# ============================================================


def boltzmann_rhs(
    tau: float,
    y,
    k: float,
    a: float,
    opacity: float,
    cs2_b: float,
    grhog: float,
    grhornomass: float,
    grhoc: float,
    grhob: float,
    grhov: float,
) -> tuple[float, ...]:
    """Return the Einstein-Boltzmann right-hand side ``dy/dtau``.

    Assembles the synchronous-gauge hierarchy from the scalar helpers above. The
    background scale factor ``a``, Thomson opacity ``kappa' = opacity``, and
    baryon sound speed ``cs2_b`` are supplied as scalars evaluated at ``tau`` by
    the caller (typically from precomputed splines). The full photon,
    polarization, and neutrino hierarchies are integrated at all times; the
    early-time photon-baryon scattering enters through the exact Thomson drag
    terms, so a stiff ODE solver handles the high-opacity regime directly without
    a tight-coupling approximation.

    ``y`` is indexed positionally; the returned tuple has length :data:`NVAR`.
    """

    grhog_t, grhor_t, grhoc_t, grhob_t, grhov_t = comoving_densities(
        a, grhog, grhornomass, grhoc, grhob, grhov
    )
    adotoa = expansion_rate(grhog_t, grhor_t, grhoc_t, grhob_t, grhov_t)

    etak = y[IX_ETAK]
    clxc = y[IX_CLXC]
    clxb = y[IX_CLXB]
    vb = y[IX_VB]
    clxg = y[IX_G]
    qg = y[IX_G + 1]
    pig = y[IX_G + 2]
    clxr = y[IX_R]
    qr = y[IX_R + 1]
    pir = y[IX_R + 2]
    e2 = y[IX_POL] if LMAX_POL >= 2 else 0.0

    dgrho = density_perturbation(
        grhob_t, clxb, grhoc_t, clxc, grhog_t, clxg, grhor_t, clxr
    )
    dgq = momentum_perturbation(grhob_t, vb, grhog_t, qg, grhor_t, qr)
    z = metric_z(dgrho, etak, adotoa, k)
    sigma = shear_sigma(z, dgq, k)

    photbar = grhog_t / grhob_t
    pb43 = 4.0 / 3.0 * photbar
    delta_p_b = cs2_b * clxb

    dy = [0.0] * NVAR

    # Metric, CDM, baryon density.
    dy[IX_ETAK] = metric_etak_derivative(dgq)
    dy[IX_CLXC] = cdm_density_derivative(z, k)
    dy[IX_CLXB] = baryon_density_derivative(z, vb, k)

    polter = polarization_source(pig, e2)
    vbdot = baryon_velocity_derivative(adotoa, vb, k, delta_p_b, photbar, opacity, qg)
    dy[IX_VB] = vbdot
    dy[IX_G] = photon_monopole_derivative(z, qg, k)
    dy[IX_G + 1] = photon_dipole_derivative(
        vbdot, adotoa, vb, k, delta_p_b, pb43, clxg, pig
    )

    theta3 = y[IX_G + 3] if LMAX_G >= 3 else 0.0
    dy[IX_G + 2] = photon_quadrupole_derivative(
        qg, theta3, pig, polter, sigma, k, opacity
    )
    for l in range(3, LMAX_G):
        dy[IX_G + l] = photon_temperature_multipole_derivative(
            l, y[IX_G + l - 1], y[IX_G + l], y[IX_G + l + 1], k, opacity
        )
    dy[IX_G + LMAX_G] = photon_temperature_truncation(
        y[IX_G + LMAX_G - 1], y[IX_G + LMAX_G], k, opacity, tau, LMAX_G
    )

    # Photon polarization.
    e3 = y[IX_POL + 1] if LMAX_POL >= 3 else 0.0
    dy[IX_POL] = polarization_quadrupole_derivative(e2, e3, polter, k, opacity)
    for l in range(3, LMAX_POL):
        idx = IX_POL + l - 2
        dy[idx] = polarization_multipole_derivative(
            l, y[idx - 1], y[idx], y[idx + 1], k, opacity
        )
    idx_last = IX_POL + LMAX_POL - 2
    dy[idx_last] = polarization_truncation(
        y[idx_last - 1], y[idx_last], k, opacity, tau, LMAX_POL
    )

    # Massless neutrinos (collisionless).
    dy[IX_R] = neutrino_monopole_derivative(z, qr, k)
    dy[IX_R + 1] = neutrino_dipole_derivative(clxr, pir, k)
    n3 = y[IX_R + 3] if LMAX_NR >= 3 else 0.0
    dy[IX_R + 2] = neutrino_quadrupole_derivative(qr, n3, sigma, k)
    for l in range(3, LMAX_NR):
        dy[IX_R + l] = neutrino_multipole_derivative(
            l, y[IX_R + l - 1], y[IX_R + l], y[IX_R + l + 1], k
        )
    dy[IX_R + LMAX_NR] = neutrino_truncation(
        y[IX_R + LMAX_NR - 1], y[IX_R + LMAX_NR], k, tau, LMAX_NR
    )

    return tuple(dy)
def total_matter_density_contrast(
    grhoc_t: float,
    clxc: float,
    grhob_t: float,
    clxb: float,
) -> float:
    """Return the total-matter density contrast ``delta_m``.

    Density-weighted combination of the CDM and baryon contrasts,

    ``delta_m = (rho_c delta_c + rho_b delta_b) / (rho_c + rho_b)``,

    using the comoving density coefficients (the ``a`` factors cancel).
    """

    return (grhoc_t * clxc + grhob_t * clxb) / (grhoc_t + grhob_t)


def primordial_curvature_power(
    k: float,
    A_s: float,
    n_s: float,
    k_pivot: float,
) -> float:
    """Return the dimensionless primordial curvature power ``P_R(k)``.

    Power-law spectrum ``P_R(k) = A_s (k / k_pivot)^(n_s - 1)``.
    """

    return A_s * (k / k_pivot) ** (n_s - 1.0)


def matter_power_spectrum(
    k: float,
    delta_m: float,
    A_s: float,
    n_s: float,
    k_pivot: float,
) -> float:
    """Return the linear matter power spectrum ``P(k)`` in Mpc^3.

    With initial conditions normalized to unit comoving curvature, the evolved
    total-matter contrast ``delta_m(k)`` acts as a transfer function and

    ``P(k) = (2 pi^2 / k^3) P_R(k) delta_m(k)^2``,

    where ``P_R(k)`` is the dimensionless primordial curvature power. The contrast
    ``delta_m`` must be evaluated at the output time (e.g. ``a = 1``).
    """

    return (
        2.0
        * math.pi**2
        / k**3
        * primordial_curvature_power(k, A_s, n_s, k_pivot)
        * delta_m**2
    )
@dataclass(frozen=True)
class PerturbationLayout:
    """Enabled components and state-vector indices of the perturbation hierarchy.

    Attributes
    ----------
    lmaxg, lmaxpol, lmaxr : int
        Photon-temperature, E-mode-polarization, and massless-neutrino hierarchy
        truncations.
    nqmax : int
        Number of massive-neutrino momentum bins. ``0`` disables massive
        neutrinos entirely (no state variables allocated).
    lmaxnu : int
        Massive-neutrino multipole-hierarchy truncation (used only when
        ``nqmax > 0``).
    enable_dark_energy : bool
        Whether the dynamical dark-energy fluid perturbations are evolved.
    """

    lmaxg: int = 15
    lmaxpol: int = 15
    lmaxr: int = 15
    nqmax: int = 0
    lmaxnu: int = 15
    enable_dark_energy: bool = False

    def __post_init__(self):
        if self.lmaxg < 2 or self.lmaxpol < 2 or self.lmaxr < 2:
            raise ValueError("lmaxg, lmaxpol, lmaxr must each be >= 2")
        if self.nqmax < 0:
            raise ValueError("nqmax must be non-negative")
        if self.nqmax > 0 and self.lmaxnu < 2:
            raise ValueError("lmaxnu must be >= 2 when massive neutrinos are enabled")

    # --- Core (always present) -------------------------------------------------

    @property
    def ix_etak(self) -> int:
        """Index of the synchronous-gauge metric perturbation ``etak = k eta``."""
        return 0

    @property
    def ix_clxc(self) -> int:
        """Index of the CDM density contrast."""
        return 1

    @property
    def ix_clxb(self) -> int:
        """Index of the baryon density contrast."""
        return 2

    @property
    def ix_vb(self) -> int:
        """Index of the baryon velocity."""
        return 3

    @property
    def ix_g(self) -> int:
        """Base index of the photon temperature hierarchy (``Theta_l`` at ix_g + l)."""
        return 4

    @property
    def ix_pol(self) -> int:
        """Base index of the polarization hierarchy (``E_l`` at ix_pol + (l - 2))."""
        return self.ix_g + self.lmaxg + 1

    @property
    def ix_r(self) -> int:
        """Base index of the massless-neutrino hierarchy (``N_l`` at ix_r + l)."""
        return self.ix_pol + self.lmaxpol - 1

    @property
    def _ix_after_massless(self) -> int:
        """First index after the always-present core hierarchies."""
        return self.ix_r + self.lmaxr + 1

    # --- Optional: dynamical dark energy --------------------------------------

    @property
    def has_dark_energy(self) -> bool:
        return self.enable_dark_energy

    @property
    def ix_clxq(self) -> int:
        """Index of the dark-energy density contrast (only if dark energy enabled)."""
        if not self.enable_dark_energy:
            raise AttributeError("dark energy is not enabled in this layout")
        return self._ix_after_massless

    @property
    def ix_thetaq(self) -> int:
        """Index of the dark-energy velocity (only if dark energy enabled)."""
        if not self.enable_dark_energy:
            raise AttributeError("dark energy is not enabled in this layout")
        return self._ix_after_massless + 1

    @property
    def _ix_after_dark_energy(self) -> int:
        return self._ix_after_massless + (2 if self.enable_dark_energy else 0)

    # --- Optional: massive neutrinos ------------------------------------------

    @property
    def has_massive_neutrinos(self) -> bool:
        return self.nqmax > 0

    @property
    def ix_massive_nu(self) -> int:
        """Base index of the massive-neutrino momentum-bin hierarchies.

        The multipole ``psi_l`` of momentum bin ``q`` is stored at
        ``ix_massive_nu + q * (lmaxnu + 1) + l`` (**bin-major, multipole-minor**),
        so that each bin's free-streaming tail ``psi_3 ... psi_lmaxnu`` is
        contiguous and can be treated as one tridiagonal block by the Schur-EB
        block-LU solver.
        """
        if self.nqmax == 0:
            raise AttributeError("massive neutrinos are not enabled in this layout")
        return self._ix_after_dark_energy

    def ix_psi_base(self, q: int) -> int:
        """Return the state index of ``psi_0`` for momentum bin ``q``."""
        if self.nqmax == 0:
            raise AttributeError("massive neutrinos are not enabled in this layout")
        if not (0 <= q < self.nqmax):
            raise IndexError(f"momentum bin q={q} out of range [0, {self.nqmax})")
        return self.ix_massive_nu + q * (self.lmaxnu + 1)

    def ix_psi(self, l: int, q: int) -> int:
        """Return the state index of massive-neutrino multipole ``psi_l`` of bin ``q``."""
        if not (0 <= l <= self.lmaxnu):
            raise IndexError(f"multipole l={l} out of range [0, {self.lmaxnu}]")
        return self.ix_psi_base(q) + l

    # --- Total size -----------------------------------------------------------

    @property
    def nvar(self) -> int:
        """Total length of the trimmed perturbation state vector."""
        n = self._ix_after_dark_energy
        if self.nqmax > 0:
            n += self.nqmax * (self.lmaxnu + 1)
        return n

    @classmethod
    def from_cosmology(
        cls,
        cosmology,
        *,
        lmaxg: int = 15,
        lmaxpol: int = 15,
        lmaxr: int = 15,
        lmaxnu: int = 15,
        nqmax: int = 5,
    ) -> "PerturbationLayout":
        """Return the trimmed layout implied by a cosmology's enabled components.

        Dynamical dark energy is enabled when ``w_DE_0 != -1`` or ``w_DE_a != 0``;
        massive neutrinos are enabled (with ``nqmax`` momentum bins) when
        ``num_massive_neutrinos > 0``. Curvature does not affect the layout.
        """

        enable_de = (cosmology.w_DE_0 != -1.0) or (cosmology.w_DE_a != 0.0)
        nqmax_massive = nqmax if cosmology.num_massive_neutrinos > 0 else 0
        return cls(
            lmaxg=lmaxg,
            lmaxpol=lmaxpol,
            lmaxr=lmaxr,
            nqmax=nqmax_massive,
            lmaxnu=lmaxnu,
            enable_dark_energy=enable_de,
        )


FLAT_MASSLESS_LAYOUT = PerturbationLayout()
"""Default flat-LambdaCDM + massless-neutrino layout (NVAR == 50).

Reproduces the fixed index constants of :mod:`discoeb.perturbations`.
"""
class _ParamCosmology(NamedTuple):
    """Hashable view of the historical DISCO-EB parameter dictionary."""

    Omegam: float
    Omegab: float
    w_DE_0: float
    w_DE_a: float
    cs2_DE: float
    Omegak: float
    A_s: float
    n_s: float
    H0: float
    T_cmb: float
    Y_He: float
    Neff_massless: float
    num_massive_neutrinos: float
    mnu: float
    k_pivot: float

    @property
    def h(self):
        return self.H0 / 100.0

    @property
    def Omegac(self):
        return self.Omegam - self.Omegab

    @property
    def omega_b_h2(self):
        return self.Omegab * self.h**2

    @property
    def omega_c_h2(self):
        return self.Omegac * self.h**2

    @property
    def tau_reion(self):
        return 0.0

    @property
    def N_eff(self):
        return self.Neff_massless + self.num_massive_neutrinos

    def to_background_params(self):
        return {
            "Omegam": self.Omegam,
            "Omegab": self.Omegab,
            "w_DE_0": self.w_DE_0,
            "w_DE_a": self.w_DE_a,
            "cs2_DE": self.cs2_DE,
            "Omegak": self.Omegak,
            "A_s": self.A_s,
            "n_s": self.n_s,
            "H0": self.H0,
            "Tcmb": self.T_cmb,
            "YHe": self.Y_He,
            "Neff": self.Neff_massless,
            "Nmnu": self.num_massive_neutrinos,
            "mnu": self.mnu,
        }


def _as_cosmology(cosmology):
    """Normalize a legacy parameter dictionary for the accelerator path."""

    if not isinstance(cosmology, dict):
        return cosmology
    return _ParamCosmology(
        Omegam=float(cosmology["Omegam"]),
        Omegab=float(cosmology["Omegab"]),
        w_DE_0=float(cosmology.get("w_DE_0", -1.0)),
        w_DE_a=float(cosmology.get("w_DE_a", 0.0)),
        cs2_DE=float(cosmology.get("cs2_DE", 1.0)),
        Omegak=float(cosmology.get("Omegak", 0.0)),
        A_s=float(cosmology["A_s"]),
        n_s=float(cosmology["n_s"]),
        H0=float(cosmology["H0"]),
        T_cmb=float(cosmology["Tcmb"]),
        Y_He=float(cosmology["YHe"]),
        Neff_massless=float(cosmology["Neff"]),
        num_massive_neutrinos=float(cosmology.get("Nmnu", 0.0)),
        mnu=float(cosmology.get("mnu", 0.0)),
        k_pivot=float(cosmology.get("k_p", 0.05)),
    )

# Solver and grid presets for the accelerated matter-power path.
N_THERMO_GRID = 2048
N_BACKGROUND_GRID = 4096
PERTURB_RTOL = 1.0e-4
PERTURB_ATOL = 1.0e-4
PERTURB_FIRST_STEP = 1.0e-2
PERTURB_MAX_STEPS = 20000
TAU_START = 0.1

# Packed-parameter row layout consumed by the device rhs/jac.
#   p = (grhog, grhornomass, grhoc, grhob, grhov,
#        tau_start, tau_end, k, cosmology_idx, save_index,
#        w_DE_0, w_DE_a, cs2_DE, grhok)
# grhok = grhom * Omega_k is the (constant) comoving curvature density
# coefficient 8*pi*G*rho_K a^2; it is 0 for a flat cosmology.
IX_TAU_START = 5
IX_TAU_END = 6
IX_K = 7
IX_COSMOLOGY = 8
IX_SAVE_INDEX = 9
IX_W_DE_0 = 10
IX_W_DE_A = 11
IX_CS2_DE = 12
IX_GRHOK = 13
IX_GRHOR_NU = 14
IX_NMNU = 15
IX_AMNU = 16
N_PARAM = 17

THERMO_TABLE_DTYPE = np.float32
BATCHES_PER_BLOCK = 32

NQMAX = 3
"""Massive-neutrino momentum bins used by both the background and the hierarchy.

Each bin adds ``lmaxnu + 1 == 16`` state variables and the generated dense
Jacobian scales as ``nvar^2``, so this sets the numba compile time: 3 bins give
``nvar = 98``, 5 bins give ``nvar = 130`` and compile ~4x slower for no gain in
P(k) accuracy against CLASS.
"""


def neutrino_momentum_bins(nqmax: int = NQMAX):
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


def massive_neutrino_density_ratio(a, amnu: float, q, w) -> np.ndarray:
    """Return ``rho_nu(a)/rho_nu0_massless`` on the same momentum quadrature.

    ``rho = sum_i w_i / v_i`` with ``v_i = 1/sqrt(1 + (a amnu/q_i)^2)``, i.e.
    ``rho = sum_i w_i sqrt(1 + (a amnu/q_i)^2)``. It tends to 1 (relativistic)
    as ``a -> 0``. Using the *same* quadrature as the perturbation hierarchy
    keeps ``drho_nu/rho_nu`` consistent.
    """

    a = np.atleast_1d(np.asarray(a, dtype=np.float64))[:, None]
    return np.sum(w[None, :] * np.sqrt(1.0 + (a * amnu / q[None, :]) ** 2), axis=1)


def massive_neutrino_parameters(cosmology) -> tuple[float, float, float]:
    """Return ``(grhor_nu, N_mnu, amnu)`` for the massive-neutrino species."""

    grhor_nu = 3.39739477e-14 * cosmology.T_cmb**4
    n_mnu = float(cosmology.num_massive_neutrinos)
    amnu = 1.62581581e4 * cosmology.mnu / cosmology.T_cmb if n_mnu > 0.0 else 0.0
    return grhor_nu, n_mnu, amnu


def density_coefficients(cosmology) -> tuple[float, float, float, float, float]:
    """Return the background ``grho`` density coefficients for ``cosmology``.

    ``grhornomass`` uses the *massless* neutrino count. The dark-energy
    coefficient is fixed by closing the density budget at ``a = 1``,

    ``grhov = grhom - (grhog + grhornomass + grho_mnu(1) + grhoc + grhob + grhok)``,

    which reproduces the flat massless value and automatically accounts for
    spatial curvature and the massive-neutrino density.
    """

    grhom = 3.33795017e-11 * cosmology.H0**2
    grhog_value = 1.49594245e-13 * cosmology.T_cmb**4
    grhornomass_value = 3.39739477e-14 * cosmology.T_cmb**4 * cosmology.Neff_massless
    grhoc_value = grhom * cosmology.Omegac
    grhob_value = grhom * cosmology.Omegab
    grhok = grhom * cosmology.Omegak

    grhor_nu, n_mnu, amnu = massive_neutrino_parameters(cosmology)
    if n_mnu > 0.0:
        q, w, _ = neutrino_momentum_bins()
        grho_mnu_today = grhor_nu * n_mnu * float(
            massive_neutrino_density_ratio(1.0, amnu, q, w)[0]
        )
    else:
        grho_mnu_today = 0.0

    grhov_value = grhom - (
        grhog_value
        + grhornomass_value
        + grho_mnu_today
        + grhoc_value
        + grhob_value
        + grhok
    )
    return (grhog_value, grhornomass_value, grhoc_value, grhob_value, grhov_value)


_THERMO_TABLE_CACHE: dict = {}


def build_thermo_tables(cosmology, n_grid: int = N_THERMO_GRID):
    """Cached wrapper around :func:`_build_thermo_tables`.

    Each call runs a full RECFAST recombination solve plus the background
    quadrature, so it is far too expensive to repeat per evaluation -- a CMB batch
    needs every cosmology's ``tau0`` up front *and* again inside the solve
    preparation. The tables depend only on ``(cosmology, n_grid)``, so memoize
    them.
    """

    key = (cosmology, n_grid)
    cached = _THERMO_TABLE_CACHE.get(key)
    if cached is None:
        cached = _build_thermo_tables(cosmology, n_grid)
        _THERMO_TABLE_CACHE[key] = cached
    return cached


def _build_thermo_tables(cosmology, n_grid: int = N_THERMO_GRID):
    """Return ``(tau_grid, values, seconds, tau0)`` thermodynamics spline tables.

    ``values`` has rows ``(a, kappa', c_s,b^2, rho_nu)`` sampled on a uniform
    ``log(tau)`` grid over ``[TAU_START, tau0]``. The background and RECFAST
    histories come directly from :func:`discoeb.background.evolve_background`;
    this function only resamples its existing splines into the layout required by
    the numba-CUDA callbacks. ``seconds`` contains the matching cubic second
    derivatives with respect to ``log(tau)``.
    """

    background = evolve_background(
        param=cosmology.to_background_params(),
        thermo_module="RECFAST",
        num_thermo=n_grid,
    )
    a_background = np.geomspace(1.0e-9, 1.0, N_BACKGROUND_GRID)
    dtauda_values = np.asarray(
        dtauda(jnp.asarray(a_background), background)
    )[:, 0]
    tau_background = integrate.cumulative_trapezoid(
        dtauda_values, a_background, initial=0.0
    )
    tau_background += float(np.asarray(background["taumin"]).reshape(-1)[0])
    tau0 = float(tau_background[-1])
    log_tau_grid = np.linspace(np.log(TAU_START), np.log(tau0), n_grid)
    tau_grid = np.exp(log_tau_grid)
    a_vals = np.interp(tau_grid, tau_background, a_background)
    loga_eval = jnp.log(jnp.asarray(a_vals))

    xe_vals = np.asarray(background["xe_of_loga_spline"].evaluate(loga_eval))[:, 0]
    a_recfast_start = 1.0 / 3501.0
    xe_recfast_start = float(
        np.asarray(
            background["xe_of_loga_spline"].evaluate(
                jnp.log(jnp.asarray([a_recfast_start]))
            )
        ).reshape(-1)[0]
    )
    xe_vals = np.where(a_vals < a_recfast_start, xe_recfast_start, xe_vals)
    akthom = (
        2.3038921003709498e-9
        * (1.0 - cosmology.Y_He)
        * cosmology.Omegab
        * cosmology.H0**2
    )
    opacity_vals = xe_vals * akthom / a_vals**2
    cs2a_vals = np.asarray(
        background["cs2a_of_loga_spline"].evaluate(loga_eval)
    )[:, 0]
    cs2_vals = cs2a_vals / a_vals
    rhonu_vals = np.exp(
        np.asarray(
            background["logrhonu_of_loga_spline"].evaluate(jnp.log(a_vals))
        )[:, 0]
    )

    values = np.stack((a_vals, opacity_vals, cs2_vals, rhonu_vals))
    seconds = np.stack(
        [interpolate.CubicSpline(log_tau_grid, row)(log_tau_grid, 2) for row in values]
    )
    return tau_grid, values, seconds, tau0


def make_params(cosmology, k_values: np.ndarray, tau_end: float) -> np.ndarray:
    """Return per-mode packed perturbation parameters for the device rhs."""

    densities = density_coefficients(cosmology)
    n = len(k_values)
    rows = np.repeat(np.asarray(densities)[None, :], n, axis=0)
    tau_start = np.full(n, TAU_START, dtype=np.float64)
    tau_end_col = np.full(n, tau_end, dtype=np.float64)
    cosmology_idx = np.zeros(n, dtype=np.float64)
    save_index = np.ones(n, dtype=np.float64)
    w_de_0 = np.full(n, cosmology.w_DE_0, dtype=np.float64)
    w_de_a = np.full(n, cosmology.w_DE_a, dtype=np.float64)
    cs2_de = np.full(n, cosmology.cs2_DE, dtype=np.float64)
    grhok = 3.33795017e-11 * cosmology.H0**2 * cosmology.Omegak
    grhok_col = np.full(n, grhok, dtype=np.float64)
    grhor_nu, n_mnu, amnu = massive_neutrino_parameters(cosmology)
    grhor_nu_col = np.full(n, grhor_nu, dtype=np.float64)
    n_mnu_col = np.full(n, n_mnu, dtype=np.float64)
    amnu_col = np.full(n, amnu, dtype=np.float64)
    return np.column_stack(
        (rows, tau_start, tau_end_col, np.asarray(k_values, dtype=np.float64),
         cosmology_idx, save_index, w_de_0, w_de_a, cs2_de, grhok_col,
         grhor_nu_col, n_mnu_col, amnu_col)
    )


def make_initial_states(cosmology, k_values: np.ndarray, layout) -> np.ndarray:
    """Return adiabatic initial states for every wave mode, shape ``(n_k, nvar)``.

    The always-present core (metric + fluid + photon/polarization/massless-nu
    hierarchies) is set by :func:`adiabatic_initial_conditions`; when dark energy
    is enabled, the DE fluid is given its adiabatic value
    ``clxq = (1 + w_0) clxc``, ``thetaq = 0``.
    """

    densities = density_coefficients(cosmology)
    y0 = np.zeros((len(k_values), layout.nvar), dtype=np.float64)
    if layout.has_massive_neutrinos:
        _, _, dlfdlq = neutrino_momentum_bins(layout.nqmax)
    for i, k in enumerate(k_values):
        core = adiabatic_initial_conditions(float(k), TAU_START, *densities[:4])
        y0[i, : len(core)] = core
        if layout.enable_dark_energy:
            y0[i, layout.ix_clxq] = (1.0 + cosmology.w_DE_0) * core[IX_CLXC]
            y0[i, layout.ix_thetaq] = 0.0
        if layout.has_massive_neutrinos:
            # Deep in radiation domination the massive neutrinos are still
            # relativistic (a*amnu/q << 1, v -> 1), so their phase-space
            # perturbation reduces to the massless hierarchy:
            #   psi_l = -(1/4) N_l dln f0/dln q      (Ma & Bertschinger)
            for qi in range(layout.nqmax):
                for l in range(0, min(3, LMAX_NR) + 1):
                    y0[i, layout.ix_psi(l, qi)] = -0.25 * core[IX_R + l] * dlfdlq[qi]
    return y0


def build_numba_callbacks(layout, tau_min, inv_dtau, values, seconds, schur_solver):
    """Return numba-CUDA ``(rhs, jac, time_jac)`` device callbacks for a layout.

    The thermodynamics tables are uploaded once and read through a uniform
    ``log(tau)`` cubic-spline device evaluator; the perturbation RHS and its
    (linear, sparse) Jacobian are compiled as ``cuda.jit(device=True)`` functions.
    The dynamical-dark-energy fluid equations are included when
    ``layout.enable_dark_energy`` is set; because the flag is a compile-time
    constant, numba prunes the DE branches entirely when it is disabled (so the
    flat-LambdaCDM kernel is unchanged).

    The tables are stacked over cosmologies: ``values`` and ``seconds`` have
    shape ``(n_cosmologies, n_channels, n_grid)`` and ``tau_min`` / ``inv_dtau``
    are ``(n_cosmologies,)``. The device evaluator selects a cosmology's table by
    the ``IX_COSMOLOGY`` tag on each trajectory's parameter row, so one compiled
    kernel solves an arbitrary batch of cosmologies. A single-cosmology solve is
    just ``n_cosmologies == 1``.
    """

    from numba import cuda, float64

    NVAR = layout.nvar
    ENABLE_DE = layout.enable_dark_energy
    IX_CLXQ = layout.ix_clxq if ENABLE_DE else 0
    IX_THETAQ = layout.ix_thetaq if ENABLE_DE else 0

    ENABLE_MNU = layout.has_massive_neutrinos
    NQ = layout.nqmax if ENABLE_MNU else 0
    LMAXNU = layout.lmaxnu
    IX_MNU = layout.ix_massive_nu if ENABLE_MNU else 0
    NPSI = LMAXNU + 1

    n_grid = values.shape[-1]
    tau_min_dev = cuda.to_device(np.ascontiguousarray(tau_min, dtype=np.float64))
    inv_dtau_dev = cuda.to_device(np.ascontiguousarray(inv_dtau, dtype=np.float64))
    values_dev = cuda.to_device(np.ascontiguousarray(values, dtype=THERMO_TABLE_DTYPE))
    seconds_dev = cuda.to_device(np.ascontiguousarray(seconds, dtype=THERMO_TABLE_DTYPE))

    # Massive-neutrino momentum quadrature (device constants).
    q_np, w_np, dlf_np = neutrino_momentum_bins(NQ if ENABLE_MNU else 1)
    q_dev = cuda.to_device(np.ascontiguousarray(q_np))
    w_dev = cuda.to_device(np.ascontiguousarray(w_np))
    dlf_dev = cuda.to_device(np.ascontiguousarray(dlf_np))

    @cuda.jit(device=True)
    def spline_eval(x, p, channel):
        cosmology_idx = int(p[IX_COSMOLOGY])
        x_min = tau_min_dev[cosmology_idx]
        inv_dx = inv_dtau_dev[cosmology_idx]
        idx = int((x - x_min) * inv_dx)
        if idx < 0:
            idx = 0
        if idx > n_grid - 2:
            idx = n_grid - 2
        h = 1.0 / inv_dx
        x0 = x_min + idx * h
        x1 = x0 + h
        left = x1 - x
        right = x - x0
        y0 = values_dev[cosmology_idx, channel, idx]
        y1 = values_dev[cosmology_idx, channel, idx + 1]
        m0 = seconds_dev[cosmology_idx, channel, idx]
        m1 = seconds_dev[cosmology_idx, channel, idx + 1]
        return (
            m0 * left**3 / (6.0 * h)
            + m1 * right**3 / (6.0 * h)
            + (y0 - m0 * h**2 / 6.0) * left / h
            + (y1 - m1 * h**2 / 6.0) * right / h
        )

    @cuda.jit(device=True)
    def rhs(y, tau, p, out):
        k = p[IX_K]
        log_tau = math.log(tau)
        a = spline_eval(log_tau, p, 0)
        opacity = spline_eval(log_tau, p, 1)
        cs2 = spline_eval(log_tau, p, 2)
        if opacity < 1.0e-30:
            opacity = 1.0e-30
        if cs2 < 0.0:
            cs2 = 0.0

        a2 = a * a
        grhog_t = p[0] / a2
        grhor_t = p[1] / a2
        grhoc_t = p[2] / a
        grhob_t = p[3] / a
        if ENABLE_DE:
            w0 = p[IX_W_DE_0]
            wa = p[IX_W_DE_A]
            w_Q = w0 + wa * (1.0 - a)
            rho_Q = a ** (-3.0 * (1.0 + w0 + wa)) * math.exp(3.0 * (a - 1.0) * wa)
            grhov_t = p[4] * rho_Q * a2
        else:
            grhov_t = p[4] * a2
        if ENABLE_MNU:
            grhor_nu = p[IX_GRHOR_NU]
            n_mnu = p[IX_NMNU]
            amnu = p[IX_AMNU]
            rhonu = spline_eval(log_tau, p, 3)
            grho_mnu_t = grhor_nu * n_mnu * rhonu / a2
        else:
            grho_mnu_t = 0.0
        grhok = p[IX_GRHOK]
        adotoa = math.sqrt(
            (grhog_t + grhor_t + grho_mnu_t + grhoc_t + grhob_t + grhov_t + grhok) / 3.0
        )

        etak = y[IX_ETAK]
        clxc = y[IX_CLXC]
        clxb = y[IX_CLXB]
        vb = y[IX_VB]
        clxg = y[IX_G]
        qg = y[IX_G + 1]
        pig = y[IX_G + 2]
        e2 = y[IX_POL]
        clxr = y[IX_R]
        qr = y[IX_R + 1]
        pir = y[IX_R + 2]
        dgrho = grhob_t * clxb + grhoc_t * clxc + grhog_t * clxg + grhor_t * clxr
        dgq = grhob_t * vb + grhog_t * qg + grhor_t * qr
        if ENABLE_DE:
            clxq = y[IX_CLXQ]
            thetaq = y[IX_THETAQ]
            dgrho = dgrho + grhov_t * clxq
            dgq = dgq + grhov_t * (1.0 + w_Q) * thetaq / k
        if ENABLE_MNU:
            # Momentum integrals of the phase-space perturbation:
            #   drho_nu = sum_i w_i psi0_i / v_i ,   f_nu = sum_i w_i psi1_i
            drhonu = 0.0
            fnu = 0.0
            for qi in range(NQ):
                b = IX_MNU + qi * NPSI
                vq = 1.0 / math.sqrt(1.0 + (a * amnu / q_dev[qi]) ** 2)
                drhonu += w_dev[qi] * y[b] / vq
                fnu += w_dev[qi] * y[b + 1]
            dgrho = dgrho + grhor_nu * n_mnu * drhonu / a2
            dgq = dgq + grhor_nu * n_mnu * fnu / a2
        # Curved-geometry metric relations (CLASS perturbations.c, synchronous):
        #   h'    = (k^2 s2^2 eta + 1.5 a^2 delta_rho)/(0.5 aH),   z = h'/(2k)
        #   eta'  = (1.5 a^2 (rho+p)theta + 0.5 K h')/(k^2 s2^2)
        #   sigma = k alpha = (z + 1.5 dgq/k^2)/s2^2
        # with K = -grhok/3 and s2^2 = 1 - 3K/k^2 = 1 + grhok/k^2 (= 1 when flat).
        s2 = 1.0 + grhok / (k * k)
        Kcurv = -grhok / 3.0
        z = (0.5 * dgrho / k + s2 * etak) / adotoa
        sigma = (z + 1.5 * dgq / (k * k)) / s2
        photbar = grhog_t / grhob_t
        pb43 = 4.0 / 3.0 * photbar
        delta_p_b = cs2 * clxb
        polter = pig / 10.0 + 9.0 * e2 / 15.0
        vbdot = -adotoa * vb + k * delta_p_b - photbar * opacity * (4.0 * vb / 3.0 - qg)

        for i in range(NVAR):
            out[i] = 0.0
        out[IX_ETAK] = (0.5 * dgq + Kcurv * z) / s2
        out[IX_CLXC] = -k * z
        out[IX_CLXB] = -k * (z + vb)
        out[IX_VB] = vbdot
        out[IX_G] = -k * (4.0 * z / 3.0 + qg)
        out[IX_G + 1] = (
            4.0 * (-vbdot - adotoa * vb + k * delta_p_b) / (3.0 * pb43)
            + k * clxg / 3.0
            - 2.0 * k * pig / 3.0
        )
        out[IX_G + 2] = (
            2.0 * k * qg / 5.0
            - 3.0 * k * y[IX_G + 3] / 5.0
            - opacity * (pig - polter)
            + 8.0 * k * sigma / 15.0
        )
        for ell in range(3, LMAX_G):
            out[IX_G + ell] = (
                k * ell * y[IX_G + ell - 1] / (2 * ell + 1)
                - k * (ell + 1) * y[IX_G + ell + 1] / (2 * ell + 1)
                - opacity * y[IX_G + ell]
            )
        out[IX_G + LMAX_G] = (
            k * y[IX_G + LMAX_G - 1]
            - (LMAX_G + 1) * y[IX_G + LMAX_G] / tau
            - opacity * y[IX_G + LMAX_G]
        )

        out[IX_POL] = -opacity * (e2 - polter) - k * y[IX_POL + 1] / 3.0
        for ell in range(3, LMAX_POL):
            idx = IX_POL + ell - 2
            polfac = (ell + 3) * (ell - 1) / (ell + 1)
            out[idx] = (
                -opacity * y[idx]
                + k * ell * y[idx - 1] / (2 * ell + 1)
                - polfac * k * y[idx + 1] / (2 * ell + 1)
            )
        idx_last = IX_POL + LMAX_POL - 2
        out[idx_last] = (
            -opacity * y[idx_last]
            + k * LMAX_POL * y[idx_last - 1] / (2 * LMAX_POL + 1)
            - (LMAX_POL + 3) * y[idx_last] / tau
        )

        out[IX_R] = -k * (4.0 * z / 3.0 + qr)
        out[IX_R + 1] = k * (clxr - 2.0 * pir) / 3.0
        out[IX_R + 2] = (
            2.0 * k * qr / 5.0 - 3.0 * k * y[IX_R + 3] / 5.0 + 8.0 * k * sigma / 15.0
        )
        for ell in range(3, LMAX_NR):
            out[IX_R + ell] = k * ell * y[IX_R + ell - 1] / (2 * ell + 1) - k * (
                ell + 1
            ) * y[IX_R + ell + 1] / (2 * ell + 1)
        out[IX_R + LMAX_NR] = (
            k * y[IX_R + LMAX_NR - 1] - (LMAX_NR + 1) * y[IX_R + LMAX_NR] / tau
        )

        if ENABLE_DE:
            cs2_Q = p[IX_CS2_DE]
            w_Q_prime = -wa * adotoa * a
            ca2_Q = w_Q - w_Q_prime / (3.0 * (1.0 + w_Q) * adotoa)
            out[IX_CLXQ] = (
                -(1.0 + w_Q) * (thetaq + k * z)
                - 3.0 * (cs2_Q - w_Q) * adotoa * clxq
                - 9.0 * (1.0 + w_Q) * (cs2_Q - ca2_Q) * adotoa**2 / k**2 * thetaq
            )
            out[IX_THETAQ] = (
                -(1.0 - 3.0 * cs2_Q) * adotoa * thetaq
                + cs2_Q / (1.0 + w_Q) * k**2 * clxq
            )

        if ENABLE_MNU:
            # Massive-neutrino phase-space hierarchy (Ma & Bertschinger; CLASS
            # perturbations.c, synchronous gauge with metric_continuity = h'/2,
            # metric_euler = 0, metric_shear = k*sigma, and h' = 2 k z):
            #   psi0' = -k v psi1 + (k z / 3) dlnf0
            #   psi1' = (k v / 3) (psi0 - 2 psi2)
            #   psi2' = (k v / 5) (2 psi1 - 3 psi3) - (2/15) k sigma dlnf0
            #   psi_l' = (k v / (2l+1)) (l psi_{l-1} - (l+1) psi_{l+1})
            #   psi_L' = k v psi_{L-1} - (L+1) psi_L / tau
            for qi in range(NQ):
                b = IX_MNU + qi * NPSI
                vq = 1.0 / math.sqrt(1.0 + (a * amnu / q_dev[qi]) ** 2)
                kv = k * vq
                dl = dlf_dev[qi]
                out[b] = -kv * y[b + 1] + (k * z / 3.0) * dl
                out[b + 1] = kv / 3.0 * (y[b] - 2.0 * y[b + 2])
                out[b + 2] = (
                    kv / 5.0 * (2.0 * y[b + 1] - 3.0 * y[b + 3])
                    - (2.0 / 15.0) * k * sigma * dl
                )
                for ell in range(3, LMAXNU):
                    out[b + ell] = (
                        kv
                        / (2.0 * ell + 1.0)
                        * (ell * y[b + ell - 1] - (ell + 1.0) * y[b + ell + 1])
                    )
                out[b + LMAXNU] = (
                    kv * y[b + LMAXNU - 1] - (LMAXNU + 1.0) * y[b + LMAXNU] / tau
                )

    dense_position = {
        state_index: dense_index
        for dense_index, state_index in enumerate(schur_solver.dense_idx)
    }
    D_ETAK = dense_position[IX_ETAK]
    D_CLXC = dense_position[IX_CLXC]
    D_CLXB = dense_position[IX_CLXB]
    D_VB = dense_position[IX_VB]
    D_G = dense_position[IX_G]
    D_QG = dense_position[IX_G + 1]
    D_PIG = dense_position[IX_G + 2]
    D_POL = dense_position[IX_POL]
    D_R = dense_position[IX_R]
    D_QR = dense_position[IX_R + 1]
    D_PIR = dense_position[IX_R + 2]
    D_CLXQ = dense_position[IX_CLXQ] if ENABLE_DE else 0
    D_THETAQ = dense_position[IX_THETAQ] if ENABLE_DE else 0
    D_MNU = dense_position[layout.ix_psi(0, 0)] if ENABLE_MNU else 0

    B_G = schur_solver.k_bases.index(IX_G + 3)
    B_POL = schur_solver.k_bases.index(IX_POL + 1)
    B_R = schur_solver.k_bases.index(IX_R + 3)
    B_MNU = (
        schur_solver.k_bases.index(layout.ix_psi(3, 0)) if ENABLE_MNU else 0
    )

    D_SIZE = schur_solver.d_size
    TL = schur_solver.tridiag_len
    JAC_SIZE = schur_solver.jac_size
    J_DENSE = schur_solver.jac_dense_off
    J_DIAG = schur_solver.jac_diag_off
    J_UPPER = schur_solver.jac_upper_off
    J_LOWER = schur_solver.jac_lower_off
    J_C0 = schur_solver.jac_c0_off

    @cuda.jit(device=True)
    def jac(y, tau, p, out):
        # The system is linear in y, so only background-dependent coefficients
        # are required. Curvature corrections are intentionally omitted here:
        # Rodas5P is a Rosenbrock-W method and accepts this approximate Jacobian.
        for i in range(JAC_SIZE):
            out[i] = 0.0

        k = p[IX_K]
        log_tau = math.log(tau)
        a = spline_eval(log_tau, p, 0)
        opacity = spline_eval(log_tau, p, 1)
        cs2 = spline_eval(log_tau, p, 2)
        if opacity < 1.0e-30:
            opacity = 1.0e-30
        if cs2 < 0.0:
            cs2 = 0.0

        a2 = a * a
        grhog_t = p[0] / a2
        grhor_t = p[1] / a2
        grhoc_t = p[2] / a
        grhob_t = p[3] / a
        if ENABLE_DE:
            w0 = p[IX_W_DE_0]
            wa = p[IX_W_DE_A]
            cs2_Q = p[IX_CS2_DE]
            w_Q = w0 + wa * (1.0 - a)
            rho_Q = a ** (-3.0 * (1.0 + w0 + wa)) * math.exp(
                3.0 * (a - 1.0) * wa
            )
            grhov_t = p[4] * rho_Q * a2
        else:
            grhov_t = p[4] * a2

        if ENABLE_MNU:
            grhor_nu = p[IX_GRHOR_NU]
            n_mnu = p[IX_NMNU]
            amnu = p[IX_AMNU]
            rhonu = spline_eval(log_tau, p, 3)
            grho_mnu_t = grhor_nu * n_mnu * rhonu / a2
        else:
            grho_mnu_t = 0.0

        grhok = p[IX_GRHOK]
        adotoa = math.sqrt(
            (
                grhog_t
                + grhor_t
                + grho_mnu_t
                + grhoc_t
                + grhob_t
                + grhov_t
                + grhok
            )
            / 3.0
        )
        photbar = grhog_t / grhob_t
        pb43 = 4.0 * photbar / 3.0

        zc = cuda.local.array(D_SIZE, float64)
        sigc = cuda.local.array(D_SIZE, float64)
        for i in range(D_SIZE):
            zc[i] = 0.0
            sigc[i] = 0.0

        zc[D_ETAK] = 1.0 / adotoa
        zc[D_CLXB] = 0.5 * grhob_t / (k * adotoa)
        zc[D_CLXC] = 0.5 * grhoc_t / (k * adotoa)
        zc[D_G] = 0.5 * grhog_t / (k * adotoa)
        zc[D_R] = 0.5 * grhor_t / (k * adotoa)
        if ENABLE_DE:
            zc[D_CLXQ] = 0.5 * grhov_t / (k * adotoa)
        if ENABLE_MNU:
            for qi in range(NQ):
                d0 = D_MNU + 3 * qi
                vq = 1.0 / math.sqrt(1.0 + (a * amnu / q_dev[qi]) ** 2)
                zc[d0] = (
                    0.5
                    * (grhor_nu * n_mnu * w_dev[qi] / (vq * a2))
                    / (k * adotoa)
                )

        for i in range(D_SIZE):
            sigc[i] = zc[i]
        sigc[D_VB] = 1.5 * grhob_t / (k * k)
        sigc[D_QG] = 1.5 * grhog_t / (k * k)
        sigc[D_QR] = 1.5 * grhor_t / (k * k)
        if ENABLE_DE:
            sigc[D_THETAQ] = 1.5 * grhov_t * (1.0 + w_Q) / k**3
        if ENABLE_MNU:
            for qi in range(NQ):
                d0 = D_MNU + 3 * qi
                velmnu = grhor_nu * n_mnu * w_dev[qi] / a2
                sigc[d0 + 1] = 1.5 * velmnu / (k * k)

        # Metric row and the z/sigma outer-product couplings.
        out[J_DENSE + D_ETAK * D_SIZE + D_VB] = 0.5 * grhob_t
        out[J_DENSE + D_ETAK * D_SIZE + D_QG] = 0.5 * grhog_t
        out[J_DENSE + D_ETAK * D_SIZE + D_QR] = 0.5 * grhor_t
        if ENABLE_DE:
            out[J_DENSE + D_ETAK * D_SIZE + D_THETAQ] = (
                0.5 * grhov_t * (1.0 + w_Q) / k
            )
        if ENABLE_MNU:
            for qi in range(NQ):
                d0 = D_MNU + 3 * qi
                velmnu = grhor_nu * n_mnu * w_dev[qi] / a2
                out[J_DENSE + D_ETAK * D_SIZE + d0 + 1] = 0.5 * velmnu

        for c in range(D_SIZE):
            z_coeff = zc[c]
            sigma_coeff = sigc[c]
            out[J_DENSE + D_CLXC * D_SIZE + c] += -k * z_coeff
            out[J_DENSE + D_CLXB * D_SIZE + c] += -k * z_coeff
            out[J_DENSE + D_G * D_SIZE + c] += -(4.0 * k / 3.0) * z_coeff
            out[J_DENSE + D_R * D_SIZE + c] += -(4.0 * k / 3.0) * z_coeff
            out[J_DENSE + D_PIG * D_SIZE + c] += (
                8.0 * k / 15.0
            ) * sigma_coeff
            out[J_DENSE + D_PIR * D_SIZE + c] += (
                8.0 * k / 15.0
            ) * sigma_coeff
            if ENABLE_DE:
                out[J_DENSE + D_CLXQ * D_SIZE + c] += (
                    -(1.0 + w_Q) * k * z_coeff
                )
            if ENABLE_MNU:
                for qi in range(NQ):
                    d0 = D_MNU + 3 * qi
                    dlf = dlf_dev[qi]
                    out[J_DENSE + d0 * D_SIZE + c] += (
                        k * dlf / 3.0
                    ) * z_coeff
                    out[J_DENSE + (d0 + 2) * D_SIZE + c] += (
                        -(2.0 / 15.0) * k * dlf
                    ) * sigma_coeff

        out[J_DENSE + D_CLXB * D_SIZE + D_VB] += -k
        out[J_DENSE + D_G * D_SIZE + D_QG] += -k
        out[J_DENSE + D_R * D_SIZE + D_QR] += -k

        # Baryon velocity and photon dipole.
        vbf_vb = -adotoa - 4.0 * photbar * opacity / 3.0
        vbf_clxb = k * cs2
        vbf_qg = photbar * opacity
        dipole_factor = -4.0 / (3.0 * pb43)

        out[J_DENSE + D_VB * D_SIZE + D_VB] += vbf_vb
        out[J_DENSE + D_VB * D_SIZE + D_CLXB] += vbf_clxb
        out[J_DENSE + D_VB * D_SIZE + D_QG] += vbf_qg
        out[J_DENSE + D_QG * D_SIZE + D_VB] += dipole_factor * vbf_vb
        out[J_DENSE + D_QG * D_SIZE + D_CLXB] += dipole_factor * vbf_clxb
        out[J_DENSE + D_QG * D_SIZE + D_QG] += dipole_factor * vbf_qg
        out[J_DENSE + D_QG * D_SIZE + D_VB] += (
            -4.0 * adotoa / (3.0 * pb43)
        )
        out[J_DENSE + D_QG * D_SIZE + D_CLXB] += (
            4.0 * k * cs2 / (3.0 * pb43)
        )
        out[J_DENSE + D_QG * D_SIZE + D_G] += k / 3.0
        out[J_DENSE + D_QG * D_SIZE + D_PIG] += -2.0 * k / 3.0

        # Photon shear and temperature hierarchy.
        out[J_DENSE + D_PIG * D_SIZE + D_QG] += 2.0 * k / 5.0
        out[J_DENSE + D_PIG * D_SIZE + D_PIG] += -0.9 * opacity
        out[J_DENSE + D_PIG * D_SIZE + D_POL] += 0.6 * opacity
        out[J_C0 + B_G] = -3.0 * k / 5.0
        for ell in range(3, LMAX_G):
            i = ell - 3
            out[J_LOWER + B_G * TL + i] = k * ell / (2 * ell + 1)
            out[J_DIAG + B_G * TL + i] = -opacity
            out[J_UPPER + B_G * (TL - 1) + i] = (
                -k * (ell + 1) / (2 * ell + 1)
            )
        out[J_LOWER + B_G * TL + TL - 1] = k
        out[J_DIAG + B_G * TL + TL - 1] = -(LMAX_G + 1.0) / tau - opacity

        # E-mode polarization hierarchy.
        out[J_DENSE + D_POL * D_SIZE + D_PIG] += 0.1 * opacity
        out[J_DENSE + D_POL * D_SIZE + D_POL] += -0.4 * opacity
        out[J_C0 + B_POL] = -k / 3.0
        for ell in range(3, LMAX_POL):
            i = ell - 3
            polfac = (ell + 3.0) * (ell - 1.0) / (ell + 1.0)
            out[J_LOWER + B_POL * TL + i] = k * ell / (2 * ell + 1)
            out[J_DIAG + B_POL * TL + i] = -opacity
            out[J_UPPER + B_POL * (TL - 1) + i] = (
                -polfac * k / (2 * ell + 1)
            )
        out[J_LOWER + B_POL * TL + TL - 1] = (
            k * LMAX_POL / (2 * LMAX_POL + 1)
        )
        out[J_DIAG + B_POL * TL + TL - 1] = (
            -opacity - (LMAX_POL + 3.0) / tau
        )

        # Massless-neutrino hierarchy.
        out[J_DENSE + D_QR * D_SIZE + D_R] = k / 3.0
        out[J_DENSE + D_QR * D_SIZE + D_PIR] = -2.0 * k / 3.0
        out[J_DENSE + D_PIR * D_SIZE + D_QR] += 2.0 * k / 5.0
        out[J_C0 + B_R] = -3.0 * k / 5.0
        for ell in range(3, LMAX_NR):
            i = ell - 3
            out[J_LOWER + B_R * TL + i] = k * ell / (2 * ell + 1)
            out[J_UPPER + B_R * (TL - 1) + i] = (
                -k * (ell + 1) / (2 * ell + 1)
            )
        out[J_LOWER + B_R * TL + TL - 1] = k
        out[J_DIAG + B_R * TL + TL - 1] = -(LMAX_NR + 1.0) / tau

        if ENABLE_DE:
            w_Q_prime = -wa * adotoa * a
            ca2_Q = w_Q - w_Q_prime / (3.0 * (1.0 + w_Q) * adotoa)
            out[J_DENSE + D_CLXQ * D_SIZE + D_CLXQ] += (
                -3.0 * (cs2_Q - w_Q) * adotoa
            )
            out[J_DENSE + D_CLXQ * D_SIZE + D_THETAQ] += (
                -(1.0 + w_Q)
                - 9.0
                * (1.0 + w_Q)
                * (cs2_Q - ca2_Q)
                * adotoa**2
                / k**2
            )
            out[J_DENSE + D_THETAQ * D_SIZE + D_CLXQ] = (
                cs2_Q * k**2 / (1.0 + w_Q)
            )
            out[J_DENSE + D_THETAQ * D_SIZE + D_THETAQ] = (
                -(1.0 - 3.0 * cs2_Q) * adotoa
            )

        if ENABLE_MNU:
            for qi in range(NQ):
                d0 = D_MNU + 3 * qi
                block = B_MNU + qi
                vq = 1.0 / math.sqrt(1.0 + (a * amnu / q_dev[qi]) ** 2)
                kv = k * vq

                out[J_DENSE + d0 * D_SIZE + d0 + 1] += -kv
                out[J_DENSE + (d0 + 1) * D_SIZE + d0] = kv / 3.0
                out[J_DENSE + (d0 + 1) * D_SIZE + d0 + 2] = -2.0 * kv / 3.0
                out[J_DENSE + (d0 + 2) * D_SIZE + d0 + 1] += 2.0 * kv / 5.0
                out[J_C0 + block] = -3.0 * kv / 5.0

                for ell in range(3, LMAXNU):
                    i = ell - 3
                    out[J_LOWER + block * TL + i] = (
                        kv * ell / (2 * ell + 1)
                    )
                    out[J_UPPER + block * (TL - 1) + i] = (
                        -kv * (ell + 1) / (2 * ell + 1)
                    )
                out[J_LOWER + block * TL + TL - 1] = kv
                out[J_DIAG + block * TL + TL - 1] = -(LMAXNU + 1.0) / tau

    @cuda.jit(device=True)
    def time_jac(y, tau, p, out):
        eps = 1.0e-6 * max(abs(tau), 1.0)
        fp = cuda.local.array(NVAR, float64)
        fm = cuda.local.array(NVAR, float64)
        rhs(y, tau + eps, p, fp)
        rhs(y, tau - eps, p, fm)
        inv = 1.0 / (2.0 * eps)
        for i in range(NVAR):
            out[i] = (fp[i] - fm[i]) * inv

    return rhs, jac, time_jac


class _PreparedSolve(NamedTuple):
    """Compiled artifacts shared by every cosmology of a (possibly batched) solve."""

    cosmologies: tuple
    layout: object
    tables: tuple  # per-cosmology (tau_grid, values, seconds, tau0)
    tau0: np.ndarray  # (n_cosmologies,)
    rhs: object
    jac: object
    time_jac: object
    lu_solver: object


_PREPARED_SOLVE_CACHE: dict = {}
_PACKED_BATCH_CACHE: dict = {}


def _stack_thermo_tables(tables):
    """Stack per-cosmology thermo tables for the multi-cosmology device evaluator.

    Returns ``(tau_min, inv_dtau, values, seconds, tau0)`` where ``values`` and
    ``seconds`` are ``(n_cosmologies, n_channels, n_grid)`` and the rest are
    ``(n_cosmologies,)``. The device spline evaluator selects a cosmology's row
    by the per-trajectory ``IX_COSMOLOGY`` tag.
    """

    tau_min = np.array([np.log(t[0][0]) for t in tables], dtype=np.float64)
    inv_dtau = np.array(
        [(t[0].shape[0] - 1) / (np.log(t[0][-1]) - np.log(t[0][0])) for t in tables],
        dtype=np.float64,
    )
    values = np.stack([t[1] for t in tables])
    seconds = np.stack([t[2] for t in tables])
    tau0 = np.array([t[3] for t in tables], dtype=np.float64)
    return tau_min, inv_dtau, values, seconds, tau0


def _build_schur_solver(layout):
    """Build the Schur-EB block-LU matching a perturbation layout.

    The state splits into a densely-coupled core (metric / fluid / low multipoles
    [+ DE fluid] [+ psi0,1,2 per massive-nu bin]) bordered by free-streaming
    tridiagonal blocks (photon, polarization, massless-nu [+ one tail per bin]),
    each coupling into the core only through its lowest multipole.
    """

    from .schur_eb import DENSE_IDX, K_BASES, DENSE_C0, A_SIZE, SchurEBSolver

    dense_idx = list(DENSE_IDX)
    k_bases = list(K_BASES)
    dense_c0 = list(DENSE_C0)
    if layout.enable_dark_energy:
        dense_idx += [layout.ix_clxq, layout.ix_thetaq]
    if layout.has_massive_neutrinos:
        for qi in range(layout.nqmax):
            dense_c0.append(len(dense_idx) + 2)  # position of psi2(q) in the core
            dense_idx += [layout.ix_psi(l, qi) for l in (0, 1, 2)]
            k_bases.append(layout.ix_psi(3, qi))

    return SchurEBSolver(
        batches_per_block=BATCHES_PER_BLOCK,
        block_dim=(BATCHES_PER_BLOCK, 1, 1),
        dense_idx=tuple(dense_idx),
        k_bases=tuple(k_bases),
        dense_c0=tuple(dense_c0),
        tridiag_len=A_SIZE,
        nvar=layout.nvar,
    )


def _prepare_solve(cosmologies) -> _PreparedSolve:
    """Return (and cache) the compiled solve artifacts for a batch of cosmologies.

    The thermodynamics tables, the numba-CUDA right-hand side / Jacobian device
    functions, and the Schur-EB block-LU depend only on the cosmologies (and the
    module-level grid and quadrature presets), not on the wave modes, the save
    times, or the solver tolerances. They are also what triggers the *slow*
    numba-CUDA kernel compilation: the returned ``rhs`` / ``jac`` / ``time_jac``
    closures and the :class:`SchurEBSolver` instance are the identity keys of
    :func:`discoeb.integrators.rodas5Pnumba_solve`'s compiled-kernel cache, so
    reusing the same objects across solves of the same batch turns a repeated
    solve from a full recompile (tens of seconds) into a bare kernel launch.

    All cosmologies in a batch must share a perturbation layout (same dark-energy
    and massive-neutrino settings); one compiled kernel then solves them all,
    selecting each cosmology's thermodynamics table by the ``IX_COSMOLOGY`` tag.

    The cache is unbounded and holds device memory (one thermodynamics table set
    plus the compiled kernel per distinct batch); sweeping many batches
    accumulates GPU allocations. Call :func:`clear_prepared_solve_cache` to
    release them.
    """

    from .schur_eb import A_SIZE

    cosmologies = tuple(_as_cosmology(cosmology) for cosmology in cosmologies)
    # Include the presets that change the compiled artifacts, so a caller that
    # rebinds them (e.g. NQMAX) does not get a stale kernel.
    key = (cosmologies, NQMAX, N_THERMO_GRID, TAU_START, BATCHES_PER_BLOCK)
    cached = _PREPARED_SOLVE_CACHE.get(key)
    if cached is not None:
        return cached

    layout = PerturbationLayout.from_cosmology(cosmologies[0], nqmax=NQMAX)
    for cosmology in cosmologies[1:]:
        if PerturbationLayout.from_cosmology(cosmology, nqmax=NQMAX) != layout:
            raise ValueError(
                "all cosmologies in a batch must share a perturbation layout "
                "(same dark-energy / massive-neutrino settings)"
            )
    if layout.has_massive_neutrinos and layout.lmaxnu - 2 != A_SIZE:
        raise NotImplementedError(
            "the Schur-EB solver requires all tridiagonal blocks to share a length; "
            f"massive-nu needs lmaxnu - 2 == {A_SIZE} (got {layout.lmaxnu})"
        )

    tables = tuple(build_thermo_tables(c) for c in cosmologies)
    tau_min, inv_dtau, values, seconds, tau0 = _stack_thermo_tables(tables)
    lu_solver = _build_schur_solver(layout)
    rhs, jac, time_jac = build_numba_callbacks(
        layout, tau_min, inv_dtau, values, seconds, lu_solver
    )

    prepared = _PreparedSolve(
        cosmologies, layout, tables, tau0, rhs, jac, time_jac, lu_solver
    )
    _PREPARED_SOLVE_CACHE[key] = prepared
    return prepared


def _pack_batch(prepared, k_values, tau_save=None):
    """Return (and cache) the packed ``(params, y0, t_span, order, k_sorted)``.

    Building the packed inputs means one :func:`make_initial_states` per cosmology
    -- i.e. ``n_cosmologies * n_k`` scalar :func:`adiabatic_initial_conditions`
    calls, tens of milliseconds for a large batch. That work depends only on the
    cosmologies and the wave modes (not the solver tolerances), so it is cached
    on ``(cosmologies, k, tau_save)``; a repeated solve of the same batch then
    skips straight to the kernel launch.

    ``tau_save`` is the shared grid of conformal times at which every trajectory
    is recorded (the CMB line-of-sight integration needs a dense one). It
    defaults to just the two endpoints, i.e. the final state only.
    """

    k_arr = np.ascontiguousarray(np.asarray(k_values, dtype=np.float64))
    tau_key = None if tau_save is None else np.asarray(tau_save, np.float64).tobytes()
    key = (prepared.cosmologies, k_arr.shape, k_arr.tobytes(), tau_key)
    cached = _PACKED_BATCH_CACHE.get(key)
    if cached is not None:
        return cached

    layout = prepared.layout
    n_cosmo = len(prepared.cosmologies)
    order = np.argsort(k_arr, kind="stable")
    k_sorted = k_arr[order]
    n_k = len(k_sorted)

    # Pack trajectories as row = k_idx * n_cosmo + cosmology_idx, so a block of
    # BATCHES_PER_BLOCK consecutive rows shares one wave mode when
    # n_cosmo >= BATCHES_PER_BLOCK: neighbouring threads then take near-identical
    # adaptive steps, minimizing warp divergence.
    n_traj = n_k * n_cosmo
    params = np.empty((n_traj, N_PARAM), dtype=np.float64)
    y0 = np.empty((n_traj, layout.nvar), dtype=np.float64)
    for ci, cosmology in enumerate(prepared.cosmologies):
        base = make_params(cosmology, k_sorted, float(prepared.tau0[ci]))
        base[:, IX_COSMOLOGY] = float(ci)
        params[ci::n_cosmo] = base
        y0[ci::n_cosmo] = make_initial_states(cosmology, k_sorted, layout)
    params = np.ascontiguousarray(params)
    y0 = np.ascontiguousarray(y0)

    # Each trajectory stops at its own tau0 (IX_TAU_END); the shared save grid
    # runs to the largest tau0 and the trailing save slot holds each trajectory's
    # final state.
    if tau_save is None:
        t_span = np.asarray((TAU_START, float(np.max(prepared.tau0))), dtype=np.float64)
    else:
        t_span = np.ascontiguousarray(np.asarray(tau_save, dtype=np.float64))

    packed = (params, y0, t_span, order, k_sorted)
    _PACKED_BATCH_CACHE[key] = packed
    return packed


def clear_prepared_solve_cache() -> None:
    """Drop all cached prepared and packed solves, releasing their device memory."""

    _PREPARED_SOLVE_CACHE.clear()
    _PACKED_BATCH_CACHE.clear()
    _THERMO_TABLE_CACHE.clear()


def solve_perturbation_history(
    k_values,
    cosmology,
    *,
    tau_save=None,
    as_jax: bool = False,
    rtol: float = PERTURB_RTOL,
    atol: float = PERTURB_ATOL,
    first_step: float = PERTURB_FIRST_STEP,
    max_steps: int = PERTURB_MAX_STEPS,
):
    """Solve the perturbation hierarchy and return the saved state history.

    Parameters
    ----------
    tau_save : array_like, optional
        Strictly increasing conformal times at which to record the state. The
        first entry must be ``TAU_START`` and the last ``tau0``. Defaults to just
        the two endpoints, i.e. the final state only. The line-of-sight CMB
        integration needs a dense grid through recombination instead.
    as_jax : bool, optional
        Return the history as a device-resident JAX array instead of copying it
        back to host NumPy. The solver's output already lives on the GPU (the
        numba kernel is invoked through a JAX FFI custom call), so this keeps the
        whole downstream pipeline -- e.g. the CMB source construction and
        line-of-sight projection -- on device with no host round-trip.

    Returns
    -------
    (history, layout, tables)
        ``history`` has shape ``(n_k, len(tau_save), nvar)`` in the caller's
        original ``k`` order; ``tables`` is the ``build_thermo_tables`` tuple.
    """

    from .integrators import rodas5Pnumba_solve

    prepared = _prepare_solve((cosmology,))
    cosmology = prepared.cosmologies[0]
    layout = prepared.layout
    tau0 = float(prepared.tau0[0])

    k_arr = np.ascontiguousarray(np.asarray(k_values, dtype=np.float64))
    order = np.argsort(k_arr, kind="stable")
    k_sorted = k_arr[order]

    params = np.ascontiguousarray(make_params(cosmology, k_sorted, tau0))
    y0 = np.ascontiguousarray(make_initial_states(cosmology, k_sorted, layout))
    if tau_save is None:
        t_span = np.asarray((TAU_START, tau0), dtype=np.float64)
    else:
        t_span = np.ascontiguousarray(np.asarray(tau_save, dtype=np.float64))

    sol = rodas5Pnumba_solve(
        prepared.rhs,
        prepared.jac,
        y0,
        t_span,
        params,
        time_jac_fn=prepared.time_jac,
        lu_precision="fp32",
        custom_lu_solver=prepared.lu_solver,
        rtol=rtol,
        atol=atol,
        first_step=first_step,
        max_steps=max_steps,
        pcoeff=0.3,
        icoeff=0.4,
        batches_per_block=BATCHES_PER_BLOCK,
        tf_local_idx=IX_TAU_END,
    )
    # The solver returns trajectories in ascending-k order; undo the sort. The
    # inverse permutation is a host-side constant, so the reorder is a plain
    # gather that works equally on device (JAX) or host (NumPy).
    inv_order = np.argsort(order)
    if as_jax:
        import jax.numpy as jnp

        return jnp.asarray(sol)[jnp.asarray(inv_order)], layout, prepared.tables[0]

    hist_sorted = np.asarray(sol)  # (n_k, n_save, nvar)
    return hist_sorted[inv_order], layout, prepared.tables[0]


def solve_perturbation_history_batch(
    k_values,
    cosmologies,
    *,
    tau_save,
    rtol: float = PERTURB_RTOL,
    atol: float = PERTURB_ATOL,
    first_step: float = PERTURB_FIRST_STEP,
    max_steps: int = PERTURB_MAX_STEPS,
):
    """Solve a batch of cosmologies on a shared ``tau_save`` grid in one launch.

    All ``n_cosmologies * n_k`` trajectories are integrated by a single
    ``rodas5Pnumba`` launch. Each trajectory still stops at its *own* ``tau0``
    (via ``IX_TAU_END``), so save times beyond a cosmology's ``tau0`` simply hold
    its frozen final state -- consumers must mask them (see
    :func:`discoeb.cmb.cmb_spectrum_from_sources`).

    Returns
    -------
    (history, k_sorted, prepared)
        ``history`` is a device-resident JAX array of shape
        ``(n_k, n_cosmologies, len(tau_save), nvar)``, with the wave modes in
        **ascending** order (``k_sorted``).

        The history is deliberately left in this layout: it is the solver's own
        packing, so the reshape is a free view. Transposing to a cosmology-major
        layout, or gathering the wave modes back into the caller's order, would
        each need a full second copy of the array -- 6.5 GB for a 128-cosmology,
        1000-save-point batch, enough to exhaust a 12 GB GPU on its own. Ascending
        ``k`` is what the downstream spline and the ``dln k`` quadrature want
        anyway.
    """

    import jax.numpy as jnp

    from .integrators import rodas5Pnumba_solve

    prepared = _prepare_solve(tuple(cosmologies))
    layout = prepared.layout
    n_cosmo = len(prepared.cosmologies)

    params, y0, t_span, order, k_sorted = _pack_batch(prepared, k_values, tau_save)
    n_k = len(k_sorted)

    sol = rodas5Pnumba_solve(
        prepared.rhs,
        prepared.jac,
        y0,
        t_span,
        params,
        time_jac_fn=prepared.time_jac,
        lu_precision="fp32",
        custom_lu_solver=prepared.lu_solver,
        rtol=rtol,
        atol=atol,
        first_step=first_step,
        max_steps=max_steps,
        pcoeff=0.3,
        icoeff=0.4,
        batches_per_block=BATCHES_PER_BLOCK,
        tf_local_idx=IX_TAU_END,
    )
    # Rows are packed k-major (row = k_idx * n_cosmo + cosmology_idx), so this
    # reshape is a free view -- no copy of the (potentially multi-GB) history.
    hist = jnp.asarray(sol).reshape(n_k, n_cosmo, len(t_span), layout.nvar)
    return hist, k_sorted, prepared


def solve_perturbations(k_values, cosmology, **solve_kwargs):
    """Solve the perturbation hierarchy for each ``k`` and return final states.

    Returns
    -------
    np.ndarray
        Final perturbation state for each wave mode, shape ``(n_k, nvar)``,
        evaluated at ``tau0`` (today).
    """

    solve_kwargs.pop("tau_save", None)
    hist, _, _ = solve_perturbation_history(k_values, cosmology, **solve_kwargs)
    return hist[:, -1, :]


def solve_matter_power_spectrum(
    k_values,
    cosmology,
    **solve_kwargs,
) -> np.ndarray:
    """Return the linear matter power spectrum ``P(k)`` in Mpc^3 for ``cosmology``."""

    cosmology = _as_cosmology(cosmology)
    k_arr = np.ascontiguousarray(np.asarray(k_values, dtype=np.float64))
    y_final = solve_perturbations(k_arr, cosmology, **solve_kwargs)
    densities = density_coefficients(cosmology)
    grhoc_val, grhob_val = densities[2], densities[3]

    pk = np.empty(len(k_arr), dtype=np.float64)
    for i, k in enumerate(k_arr):
        delta_m = total_matter_density_contrast(
            grhoc_val, y_final[i, IX_CLXC], grhob_val, y_final[i, IX_CLXB]
        )
        pk[i] = matter_power_spectrum(
            float(k), delta_m, cosmology.A_s, cosmology.n_s, cosmology.k_pivot
        )
    return pk


def solve_matter_power_spectrum_batch(
    k_values,
    cosmologies,
    *,
    rtol: float = PERTURB_RTOL,
    atol: float = PERTURB_ATOL,
    first_step: float = PERTURB_FIRST_STEP,
    max_steps: int = PERTURB_MAX_STEPS,
) -> np.ndarray:
    """Return ``P(k)`` for a batch of cosmologies in one GPU solve.

    All cosmologies must share a perturbation layout. The ``n_cosmologies * n_k``
    trajectories are integrated by a single ``rodas5Pnumba`` launch, so the whole
    batch amortizes one kernel compilation and one set of device buffers.

    Returns
    -------
    np.ndarray
        ``P(k)`` in Mpc^3, shape ``(n_cosmologies, n_k)``, in the caller's
        original cosmology and ``k`` order.
    """

    from .integrators import rodas5Pnumba_solve

    prepared = _prepare_solve(tuple(cosmologies))
    cosmologies = prepared.cosmologies
    layout = prepared.layout
    n_cosmo = len(cosmologies)

    params, y0, t_span, order, k_sorted = _pack_batch(prepared, k_values)
    n_k = len(k_sorted)

    sol = rodas5Pnumba_solve(
        prepared.rhs,
        prepared.jac,
        y0,
        t_span,
        params,
        time_jac_fn=prepared.time_jac,
        lu_precision="fp32",
        custom_lu_solver=prepared.lu_solver,
        rtol=rtol,
        atol=atol,
        first_step=first_step,
        max_steps=max_steps,
        pcoeff=0.3,
        icoeff=0.4,
        batches_per_block=BATCHES_PER_BLOCK,
        tf_local_idx=IX_TAU_END,
    )
    y_final = np.asarray(sol)[:, -1, :].reshape(n_k, n_cosmo, layout.nvar)

    pk_sorted = np.empty((n_cosmo, n_k), dtype=np.float64)
    for ci, cosmology in enumerate(cosmologies):
        densities = density_coefficients(cosmology)
        grhoc_val, grhob_val = densities[2], densities[3]
        for ki in range(n_k):
            yv = y_final[ki, ci]
            delta_m = total_matter_density_contrast(
                grhoc_val, yv[IX_CLXC], grhob_val, yv[IX_CLXB]
            )
            pk_sorted[ci, ki] = matter_power_spectrum(
                float(k_sorted[ki]), delta_m, cosmology.A_s, cosmology.n_s, cosmology.k_pivot
            )

    pk = np.empty_like(pk_sorted)
    pk[:, order] = pk_sorted
    return pk
