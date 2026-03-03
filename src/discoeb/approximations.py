import jax
import jax.numpy as jnp

from functools import partial

from .background import get_aprimeoa


def get_approximation_settings(param):
    """Return approximation configuration with defaults from parameter dictionary."""
    return {
        'use_rsa': param.get('use_rsa', True),
        'tau_c_over_tau_trigger': float(param.get('rsa_tau_c_over_tau_trigger', 10.0)),
        'tau_over_tau_k_trigger': float(param.get('rsa_tau_over_tau_k_trigger', 80.0)),
        'use_ur_fluid': param.get('use_ur_fluid', True),
        'ur_fluid_tau_over_tau_k_trigger': float(param.get('ur_fluid_tau_over_tau_k_trigger', 120.0)),
    }


def get_rsa_settings(param):
    """Backward-compatible alias for approximation settings."""
    return get_approximation_settings(param)


@partial(jax.jit, inline=True)
def compute_fields_rsa(*, kmode, aprimeoa, hprime, eta, deltab, thetab, cs2_b, tau_c, tau_c_prime):
    """Relativistic species in radiation streaming approximation (BLT11 Eq. 4.12)."""
    safe_k2 = jnp.maximum(kmode**2, 1e-30)
    safe_tauc = jnp.maximum(tau_c, 1e-30)

    # Photons
    deltag = 4.0 / safe_k2 * (aprimeoa * hprime - safe_k2 * eta) + 4.0 / (safe_k2 * safe_tauc) * (thetab + 0.5 * hprime)
    thetag = -0.5 * hprime + 3.0 / (safe_k2 * safe_tauc) * (
        -tau_c_prime / safe_tauc * (thetab + 0.5 * hprime)
        + (-aprimeoa * thetab + cs2_b * safe_k2 * deltab + safe_k2 * eta)
    )
    shearg = 0.0

    # Massless neutrinos
    deltar = 4.0 / safe_k2 * (aprimeoa * hprime - safe_k2 * eta)
    thetar = -0.5 * hprime
    shearr = 0.0

    return deltag, thetag, shearg, deltar, thetar, shearr


@partial(jax.jit, inline=True)
def in_rsa_regime(*, tau, kmode, tau_c, tau_c_over_tau_trigger, tau_over_tau_k_trigger):
    """CLASS-inspired RSA trigger with configurable thresholds."""
    return jnp.logical_and(
        tau_c / jnp.maximum(tau, 1e-30) > tau_c_over_tau_trigger,
        tau * kmode > tau_over_tau_k_trigger,
    )


@partial(jax.jit, inline=True)
def in_ur_fluid_regime(*, tau, kmode, tau_over_tau_k_trigger):
    """CLASS-inspired UFA trigger based on horizon entry (k*tau)."""
    return tau * kmode > tau_over_tau_k_trigger


@partial(jax.jit, inline=True)
def compute_shearprime_ufa(*, tau, shearr, thetar, hprime):
    """UFA closure for massless-neutrino shear evolution."""
    return -3.0 / jnp.maximum(tau, 1e-30) * shearr + 4.0 / 15.0 * (thetar + 0.5 * hprime)


def apply_rsa_state_projection(*, y, tau, kmode, param, lmaxg, lmaxgp, lmaxr, nqmax, nu_perturb_fn):
    """Replace relativistic hierarchy variables with RSA fields at a given output time."""

    def to_scalar(x):
        return jnp.ravel(x)[0]

    rsa_settings = get_approximation_settings(param)
    use_rsa = rsa_settings['use_rsa']
    tau_c_over_tau_trigger = rsa_settings['tau_c_over_tau_trigger']
    tau_over_tau_k_trigger = rsa_settings['tau_over_tau_k_trigger']

    y = jnp.ravel(y)
    tau = to_scalar(tau)
    kmode = to_scalar(kmode)

    a = to_scalar(y[0])
    loga = jnp.log(a)
    eta = to_scalar(y[2])
    deltab = to_scalar(y[5])
    thetab = to_scalar(y[6])

    # Background + thermodynamics
    aprimeoa = to_scalar(get_aprimeoa(param=param, aexp=a))
    cs2 = to_scalar(param['cs2a_of_loga_spline'].evaluate(loga) / a)
    xe = to_scalar(param['xe_of_loga_spline'].evaluate(loga))
    xeprime = to_scalar(param['xe_of_loga_spline'].derivative(loga) * aprimeoa)
    akthom = 2.3038921003709498e-9 * (1.0 - param['YHe']) * param['Omegab'] * param['H0']**2
    opac = to_scalar(xe * akthom / a**2)
    tau_c = to_scalar(1.0 / jnp.maximum(opac, 1e-30))
    tau_c_prime = to_scalar(tau_c * (2.0 * aprimeoa - xeprime / jnp.maximum(xe, 1e-30)))

    # Massive neutrino + DE pieces for hprime estimate
    Omegac = param['Omegam'] - param['Omegab']
    iq0 = 10 + lmaxg + lmaxgp + lmaxr
    iq1 = iq0 + nqmax
    iq2 = iq1 + nqmax
    iq3 = iq2 + nqmax
    drhonu, _, _, _ = nu_perturb_fn(a, param['amnu'], y[iq0:iq1], y[iq1:iq2], y[iq2:iq3], nqmax=nqmax)

    rho_Q = a**(-3 * (1 + param['w_DE_0'] + param['w_DE_a'])) * jnp.exp(3 * (a - 1) * param['w_DE_a'])
    deltaq = y[-2]
    dgrho_wo_rel = (
        param['grhom'] * (Omegac * y[3] + param['Omegab'] * deltab) / a
        + param['grhor'] * param['Nmnu'] * drhonu / a**2
        + param['grhom'] * param['OmegaDE'] * deltaq * rho_Q * a**2
    )

    hprime = to_scalar((2.0 * kmode**2 * eta + dgrho_wo_rel) / jnp.maximum(aprimeoa, 1e-30))
    deltag, thetag, shearg, deltar, thetar, shearr = compute_fields_rsa(
        kmode=kmode,
        aprimeoa=aprimeoa,
        hprime=hprime,
        eta=eta,
        deltab=deltab,
        thetab=thetab,
        cs2_b=cs2,
        tau_c=tau_c,
        tau_c_prime=tau_c_prime,
    )

    idxg = 7
    idxgp = 7 + (lmaxg + 1)
    idxr = 9 + lmaxg + lmaxgp

    y_rsa = y
    y_rsa = y_rsa.at[idxg + 0].set(deltag)
    y_rsa = y_rsa.at[idxg + 1].set(thetag)
    y_rsa = y_rsa.at[idxg + 2].set(2.0 * shearg)
    y_rsa = y_rsa.at[idxg + 3:idxg + lmaxg + 1].set(0.0)
    y_rsa = y_rsa.at[idxgp:idxgp + lmaxgp + 1].set(0.0)

    y_rsa = y_rsa.at[idxr + 0].set(deltar)
    y_rsa = y_rsa.at[idxr + 1].set(thetar)
    y_rsa = y_rsa.at[idxr + 2].set(2.0 * shearr)
    y_rsa = y_rsa.at[idxr + 3:idxr + lmaxr + 1].set(0.0)

    do_rsa = jnp.logical_and(
        jnp.asarray(use_rsa),
        in_rsa_regime(
            tau=tau,
            kmode=kmode,
            tau_c=tau_c,
            tau_c_over_tau_trigger=tau_c_over_tau_trigger,
            tau_over_tau_k_trigger=tau_over_tau_k_trigger,
        ),
    )

    return jax.lax.cond(
        do_rsa,
        lambda _: y_rsa,
        lambda _: y,
        operand=None,
    )
