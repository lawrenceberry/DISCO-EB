"""
This module provides functions to compute the CMB temperature anisotropy
source functions from background and perturbation evolution output.
"""

import jax
import jax.numpy as jnp
from functools import partial

from .background import compute_background_quantities, evolve_background
from .cosmo import get_aprimeoa
from .perturbations import compute_time_derivatives, evolve_perturbations_batched, nu_perturb, nu_perturb_prime
from .util import spherical_bessel
from .spline_interpolation import spline_interpolation


def extract_perturbations(yout, youtprime, lmaxg, lmaxgp, lmaxr):
    """Extract perturbation variables from the state vector.

    Parameters
    ----------
    yout : jnp.ndarray
        State vector array with shape (n_kmodes, n_times, n_vars)
    youtprime : jnp.ndarray
        Time derivatives with shape (n_kmodes, n_times, n_vars)
    lmaxg : int
        Maximum photon temperature multipole
    lmaxgp : int
        Maximum photon polarization multipole
    lmaxr : int
        Maximum massless neutrino multipole

    Returns
    -------
    dict
        Dictionary containing extracted perturbation variables
    """
    idxb = 5
    idxg = 7
    idxgp = 7 + (lmaxg + 1)
    idxr = 9 + lmaxg + lmaxgp

    iq0 = 10 + lmaxg + lmaxgp + lmaxr

    perturbations = {
        # Indices
        'idxb': idxb,
        'idxg': idxg,
        'idxgp': idxgp,
        'idxr': idxr,
        'iq0': iq0,

        # Metric
        'eta': yout[..., 2],

        # CDM
        'deltac': yout[..., 3],
        'thetac': yout[..., 4],

        # Baryons
        'deltab': yout[..., idxb],
        'thetab': yout[..., idxb + 1],
        'deltabprime': youtprime[..., idxb],
        'thetabprime': youtprime[..., idxb + 1],

        # Photons
        'deltag': yout[..., idxg],
        'thetag': yout[..., idxg + 1],
        'shearg': yout[..., idxg + 2] / 2.0,
        'deltagprime': youtprime[..., idxg],
        'thetagprime': youtprime[..., idxg + 1],
        'sheargprime': youtprime[..., idxg + 2] / 2.0,

        # Massless neutrinos
        'deltar': yout[..., idxr],
        'thetar': yout[..., idxr + 1],
        'shearr': yout[..., idxr + 2] / 2.0,
        'deltarprime': youtprime[..., idxr],
        'thetarprime': youtprime[..., idxr + 1],
        'shearrprime': youtprime[..., idxr + 2] / 2.0,

        # Dark energy
        'deltaq': yout[..., -2],
        'thetaq': yout[..., -1],
    }

    return perturbations


def compute_visibility_functions(tau, param):
    """Create splines and evaluate visibility functions at given conformal times.

    This function creates cubic spline interpolations for the visibility function
    and related quantities from the thermal history computed by evolve_background(),
    then evaluates them at the requested conformal times. These functions are
    essential for CMB source function calculations.

    Parameters
    ----------
    tau : jnp.ndarray
        Conformal time values where visibility functions are needed
    param : dict
        Parameter dictionary containing thermal history arrays:
        - tau: Conformal time array from thermal history
        - opac: Opacity κ = σ_T n_e a^2
        - gvis: Visibility function g(τ) = κ exp(-τ_c)
        - gvisprime: First derivative g'(τ)
        - gvispprime: Second derivative g''(τ)
        - optical_depth: Optical depth τ_c to conformal time today

    Returns
    -------
    dict
        Dictionary with keys:
        - 'opac': Opacity κ at requested times
        - 'opacprime': Opacity derivative dκ/dτ at requested times
        - 'gvis': Visibility function g(τ) at requested times
        - 'gvisprime': First derivative g'(τ) at requested times
        - 'gvispprime': Second derivative g''(τ) at requested times
        - 'optical_depth': Optical depth τ_c at requested times

    Examples
    --------
    >>> param = evolve_background(param=param, ...)
    >>> tau = jnp.linspace(100, 14000, 100)  # Conformal times in Mpc
    >>> vis_funcs = compute_visibility_functions(tau, param)
    >>> gvis = vis_funcs['gvis']  # Visibility function g(τ)

    Notes
    -----
    The visibility function g(τ) = κ(τ) exp(-τ_c(τ)) peaks at last scattering
    and determines when photons last interacted with matter. Its derivatives
    are used in the Doppler and polarization source terms.

    The opacity is κ = σ_T n_e a^2 where σ_T is the Thomson cross section
    and n_e is the free electron density.
    """
    # Create spline for opacity and evaluate
    opacspline = spline_interpolation(jnp.log(param['tau']), param['opac'])
    opac = opacspline.evaluate(jnp.log(tau))
    opacprime = opacspline.derivative(jnp.log(tau)) / tau

    # Create splines for visibility function and derivatives
    gvis_spline = spline_interpolation(jnp.log(param['tau']), param['gvis'])
    gvisprime_spline = spline_interpolation(jnp.log(param['tau']), param['gvisprime'])
    gvispprime_spline = spline_interpolation(jnp.log(param['tau']), param['gvispprime'])

    gvis = gvis_spline.evaluate(jnp.log(tau))
    gvisprime = gvisprime_spline.evaluate(jnp.log(tau))
    gvispprime = gvispprime_spline.evaluate(jnp.log(tau))

    # Create spline for optical depth (log-space with floor for numerical stability)
    optical_depth_spline = spline_interpolation(
        jnp.log(param['tau']),
        jnp.log(jnp.maximum(param['optical_depth'], 1e-10))
    )
    optical_depth = jnp.exp(optical_depth_spline.evaluate(jnp.log(tau)))

    return {
        'opac': opac,
        'opacprime': opacprime,
        'gvis': gvis,
        'gvisprime': gvisprime,
        'gvispprime': gvispprime,
        'optical_depth': optical_depth,
    }


def compute_neutrino_perturbations(yout, yprime, aexp_out, param, nqmax, iq0):
    """Compute massive neutrino perturbations by integrating over momentum bins.

    Parameters
    ----------
    yout : jnp.ndarray
        State vector array
    yprime : jnp.ndarray
        Time derivatives
    aexp_out : jnp.ndarray
        Output scale factors
    param : dict
        Cosmological parameters
    nqmax : int
        Number of neutrino momentum bins
    iq0 : int
        Starting index for neutrino perturbations

    Returns
    -------
    dict
        Dictionary with neutrino perturbation quantities
    """
    n_kmodes = yout.shape[0]
    n_times = yout.shape[1]

    idxtau = jnp.arange(n_times)
    idxk = jnp.arange(n_kmodes)

    iq1 = iq0 + nqmax
    iq2 = iq1 + nqmax
    iq3 = iq2 + nqmax

    # Compute neutrino perturbations
    drhonu, dpnu, fnu, shearnu = jax.vmap(
        lambda ik: jax.vmap(
            lambda ia: nu_perturb(
                aexp_out[ia], param['amnu'],
                yout[ik, ia, iq0:iq1],
                yout[ik, ia, iq1:iq2],
                yout[ik, ia, iq2:iq3],
                nqmax=nqmax
            )
        )(idxtau)
    )(idxk)

    # Compute conformal Hubble rate at each output time
    aprimeoa = jax.vmap(lambda ia: get_aprimeoa(param=param, aexp=aexp_out[ia]))(idxtau)

    # Compute time derivatives
    rho_nu_prime, shear_nu_prime = jax.vmap(
        lambda ik: jax.vmap(
            lambda ia: nu_perturb_prime(
                a=aexp_out[ia], amnu=param['amnu'], aprimeoa=aprimeoa[ia],
                psi0=yout[ik, ia, iq0:iq1],
                psi2=yout[ik, ia, iq2:iq3],
                psi0prime=yprime[ik, ia, iq0:iq1],
                psi2prime=yprime[ik, ia, iq2:iq3],
                nqmax=nqmax
            )
        )(idxtau)
    )(idxk)

    return {
        'drhonu': drhonu,
        'dpnu': dpnu,
        'fnu': fnu,
        'shearnu': shearnu,
        'rho_nu_prime': rho_nu_prime,
        'shear_nu_prime': shear_nu_prime,
        'aprimeoa': aprimeoa,
    }


def compute_metric_perturbations(perturbations, neutrinos, background_quantities,
                                 param, kmodes, aexp_out):
    """Compute metric perturbations from Einstein equations.

    Parameters
    ----------
    perturbations : dict
        Extracted perturbation variables
    neutrinos : dict
        Neutrino perturbation quantities
    background_quantities : dict
        Background density and pressure quantities
    param : dict
        Cosmological parameters
    kmodes : jnp.ndarray
        Wavenumber array
    aexp_out : jnp.ndarray
        Output scale factors

    Returns
    -------
    dict
        Metric perturbations (alpha, alphaprime, alphapprime, etc.)
    """
    kmode = kmodes[:, None]
    a = aexp_out

    Omegac = param['Omegam'] - param['Omegab']

    # Extract quantities
    rhonu = background_quantities['rhonu']
    rho_Q = background_quantities['rho_Q']
    w_Q = background_quantities['w_Q']
    cs2_Q = param['cs2_DE']

    aprimeoa = neutrinos['aprimeoa']

    # Quintessence velocity term
    rho_plus_p_theta_Q = (1 + w_Q) * rho_Q * param['grhom'] * param['OmegaDE'] * perturbations['thetaq'] * a**2

    # Quintessence EOS time derivatives
    w_Q_prime = -param['w_DE_a'] * aprimeoa * a
    ca2_Q = w_Q - w_Q_prime / 3 / ((1 + w_Q) + 1e-6) / aprimeoa

    # Total background energy density derivative
    grhoprime = (
        -param['grhom'] * param['Omegam'] / a
        - 2 * (param['grhog'] + param['grhor'] * (param['Neff'] + param['Nmnu'] * rhonu)) / a**2
        + param['grhom'] * param['OmegaDE'] * (
            -3 * (1 + param['w_DE_0'] + (1 - a) * param['w_DE_a']) * rho_Q * a**2
            + 2 * rho_Q * a**2
        )
    ) * aprimeoa + param['grhor'] * param['Nmnu'] * neutrinos['rho_nu_prime'] / a**2

    aprimeoaprime = grhoprime / (6 * aprimeoa)

    # Total perturbations
    dgrho = (
        param['grhom'] * (Omegac * perturbations['deltac'] + param['Omegab'] * perturbations['deltab']) / a
        + (param['grhog'] * perturbations['deltag'] + param['grhor'] * (
            param['Neff'] * perturbations['deltar'] + param['Nmnu'] * neutrinos['drhonu']
        )) / a**2
        + param['grhom'] * param['OmegaDE'] * perturbations['deltaq'] * rho_Q * a**2
    )

    dgtheta = (
        param['grhom'] * (Omegac * perturbations['thetac'] + param['Omegab'] * perturbations['thetab']) / a
        + 4.0 / 3.0 * (param['grhog'] * perturbations['thetag'] + param['Neff'] * param['grhor'] * perturbations['thetar']) / a**2
        + param['Nmnu'] * param['grhor'] * kmode * neutrinos['fnu'] / a**2
        + rho_plus_p_theta_Q
    )

    dgshear = (
        4.0 / 3.0 * (param['grhog'] * perturbations['shearg'] + param['Neff'] * param['grhor'] * perturbations['shearr']) / a**2
        + param['Nmnu'] * param['grhor'] * neutrinos['shearnu'] / a**2
    )

    dgshearprime = (
        4.0 / 3.0 * (param['grhog'] * perturbations['sheargprime']
                     + param['Neff'] * param['grhor'] * perturbations['shearrprime']) / a**2
        + param['Nmnu'] * param['grhor'] * neutrinos['shear_nu_prime'] / a**2
        - 2 * aprimeoa * dgshear
    )

    # Metric perturbations from Einstein equations
    hprime = (2.0 * kmode**2 * perturbations['eta'] + dgrho) / aprimeoa
    etaprime = 0.5 * dgtheta / kmode**2
    alpha = (hprime + 6.0 * etaprime) / 2.0 / kmode**2
    alphaprime = -3 * dgshear / (2 * kmode**2) + perturbations['eta'] - 2 * aprimeoa * alpha
    alphapprime = (-3 * dgshearprime / (2 * kmode**2) + etaprime
                   - 2 * (aprimeoaprime * alpha + aprimeoa * alphaprime))

    return {
        'hprime': hprime,
        'etaprime': etaprime,
        'alpha': alpha,
        'alphaprime': alphaprime,
        'alphapprime': alphapprime,
        'aprimeoa': aprimeoa,
        'aprimeoaprime': aprimeoaprime,
    }


def compute_polarization_terms(perturbations, metric, visibility_functions,
                               yout, yprime, kmodes, lmaxg, lmaxgp):
    """Compute polarization-related quantities for CMB source function.

    Parameters
    ----------
    perturbations : dict
        Extracted perturbation variables
    metric : dict
        Metric perturbations
    visibility_functions : dict
        Visibility function and derivatives
    yout : jnp.ndarray
        Full state vector
    yprime : jnp.ndarray
        Time derivatives
    kmodes : jnp.ndarray
        Wavenumber array
    lmaxg : int
        Maximum photon temperature multipole
    lmaxgp : int
        Maximum photon polarization multipole

    Returns
    -------
    dict
        Dictionary with polarization_term, polarization_termprime, polarization_termpprime
    """
    kmode = kmodes[:, None]
    idxg = perturbations['idxg']
    idxgp = perturbations['idxgp']

    # Coupling term derivative
    couplprime = (8 * (perturbations['thetagprime'] + kmode**2 * metric['alphaprime']) / 15
                  - kmode * 0.6 * (yprime[:, :, idxg + 3] + yprime[:, :, idxgp + 1] + yprime[:, :, idxgp + 3]))

    # Polarization multipole combinations
    polarization_term = yout[:, :, idxg + 2] + yout[:, :, idxgp + 0] + yout[:, :, idxgp + 2]
    polarization_termprime = yprime[:, :, idxg + 2] + yprime[:, :, idxgp + 0] + yprime[:, :, idxgp + 2]
    polarization_termpprime = (couplprime
                               - 0.3 * (visibility_functions['opacprime'] * polarization_term
                                       + visibility_functions['opac'] * polarization_termprime))

    return {
        'polarization_term': polarization_term,
        'polarization_termprime': polarization_termprime,
        'polarization_termpprime': polarization_termpprime,
    }


def compute_source_term_isw(metric, visibility_functions):
    """Compute the Integrated Sachs-Wolfe (ISW) source term.

    The ISW effect arises from photons gaining or losing energy as they
    traverse time-evolving gravitational potentials during radiation-matter
    transition and dark energy domination.

    S1 = exp(-τ) × (Ψ' - Φ')

    Parameters
    ----------
    metric : dict
        Metric perturbations including etaprime, alphapprime
    visibility_functions : dict
        Visibility function and optical depth

    Returns
    -------
    jnp.ndarray
        ISW source term S1
    """
    # Time derivative of potential difference (Ψ' - Φ')
    s1 = metric['etaprime'] + metric['alphapprime']

    # Suppress before recombination (universe opaque)
    expmmu = jnp.exp(-visibility_functions['optical_depth'])

    S1 = expmmu * s1

    return S1


def compute_source_term_sachs_wolfe(perturbations, metric, visibility_functions,
                                    polarization_terms, kmodes):
    """Compute the Sachs-Wolfe source term at last scattering.

    The SW effect is the dominant contribution from temperature and potential
    perturbations at the last scattering surface, weighted by the visibility function.

    S2 = g(τ) × [Θ₀/4 + 2Φ + polarization + baryon velocity]

    Parameters
    ----------
    perturbations : dict
        Extracted perturbation variables
    metric : dict
        Metric perturbations
    visibility_functions : dict
        Visibility function
    polarization_terms : dict
        Polarization-related quantities
    kmodes : jnp.ndarray
        Wavenumber array

    Returns
    -------
    jnp.ndarray
        Sachs-Wolfe source term S2
    """
    kmode = kmodes[:, None]

    # Potential contribution
    s2 = 2 * metric['alphaprime']

    S2 = visibility_functions['gvis'] * (
        0.25 * perturbations['deltag']  # Temperature monopole
        + s2  # Gravitational potential
        + polarization_terms['polarization_term'] / 16  # E-mode coupling
        + (perturbations['thetabprime'] + 3 / 16 * polarization_terms['polarization_termpprime']) / kmode**2  # Baryon velocity
    )

    return S2


def compute_source_term_doppler(perturbations, metric, visibility_functions,
                                polarization_terms, kmodes):
    """Compute the Doppler source term.

    The Doppler effect arises from bulk motion of baryons at last scattering,
    creating a velocity-induced anisotropy pattern (acoustic peaks).

    S3 = g'(τ) × [Φ + v_b/k² + polarization]

    Parameters
    ----------
    perturbations : dict
        Extracted perturbation variables
    metric : dict
        Metric perturbations
    visibility_functions : dict
        Visibility function derivative
    polarization_terms : dict
        Polarization-related quantities
    kmodes : jnp.ndarray
        Wavenumber array

    Returns
    -------
    jnp.ndarray
        Doppler source term S3
    """
    kmode = kmodes[:, None]

    S3 = visibility_functions['gvisprime'] * (
        metric['alpha']  # Gravitational potential
        + (perturbations['thetab'] + 3 / 8 * polarization_terms['polarization_termprime']) / kmode**2  # Baryon velocity
    )

    return S3


def compute_source_term_polarization(visibility_functions, polarization_terms, kmodes):
    """Compute the polarization coupling source term.

    This term describes the generation of E-mode polarization through
    Thomson scattering of anisotropic radiation.

    S4 = g''(τ) × (3/16) × polarization / k²

    Parameters
    ----------
    visibility_functions : dict
        Second derivative of visibility function
    polarization_terms : dict
        Polarization-related quantities
    kmodes : jnp.ndarray
        Wavenumber array

    Returns
    -------
    jnp.ndarray
        Polarization source term S4
    """
    kmode = kmodes[:, None]

    S4 = visibility_functions['gvispprime'] * 3 / 16 * polarization_terms['polarization_term'] / kmode**2

    return S4


def compute_source_function(perturbations, metric, visibility_functions,
                            yout, yprime, kmodes, lmaxg, lmaxgp):
    """Compute the CMB temperature anisotropy source function.

    Computes the four source terms:
    - S1: Integrated Sachs-Wolfe (ISW) effect
    - S2: Sachs-Wolfe effect at last scattering
    - S3: Doppler effect
    - S4: Polarization coupling

    Parameters
    ----------
    perturbations : dict
        Extracted perturbation variables
    metric : dict
        Metric perturbations
    visibility_functions : dict
        Visibility function and derivatives (gvis, gvisprime, gvispprime, optical_depth)
    yout : jnp.ndarray
        Full state vector
    yprime : jnp.ndarray
        Time derivatives
    kmodes : jnp.ndarray
        Wavenumber array
    lmaxg : int
        Maximum photon temperature multipole
    lmaxgp : int
        Maximum photon polarization multipole

    Returns
    -------
    dict
        Source function components (S1, S2, S3, S4, S) and polarization_terms
    """
    # Compute polarization terms (needed by multiple source terms)
    polarization_terms = compute_polarization_terms(
        perturbations, metric, visibility_functions, yout, yprime, kmodes, lmaxg, lmaxgp
    )

    # Compute each source term
    S1 = compute_source_term_isw(metric, visibility_functions)

    S2 = compute_source_term_sachs_wolfe(
        perturbations, metric, visibility_functions, polarization_terms, kmodes
    )

    S3 = compute_source_term_doppler(
        perturbations, metric, visibility_functions, polarization_terms, kmodes
    )

    S4 = compute_source_term_polarization(
        visibility_functions, polarization_terms, kmodes
    )

    # Total source function
    S = S1 + S2 + S3 + S4

    return {
        'S1': S1,  # ISW
        'S2': S2,  # Sachs-Wolfe
        'S3': S3,  # Doppler
        'S4': S4,  # Polarization
        'S': S,    # Total
        'polarization_terms': polarization_terms,
    }


@partial(jax.jit, static_argnames=['ellmax', 'nk_fine', 'chunk_size', 'k_chunk_size'])
def compute_theta_ell(ellmax, kmodes, tau, S, tau0, nk_fine=2048, chunk_size=500, k_chunk_size=1024):
    """Compute multipole moments Θ_ℓ(k) via line-of-sight integration.

    The line-of-sight integral computes:

        Θ_ℓ(k) = ∫₀^τ₀ S(k,τ) j_ℓ(k(τ₀-τ)) dτ

    where S(k,τ) is the CMB source function, j_ℓ is the spherical Bessel function,
    and τ₀ is the conformal time today.

    Parameters
    ----------
    ellmax : int
        Maximum multipole moment
    kmodes : jnp.ndarray
        Wavenumber array (shape: n_kmodes)
    tau : jnp.ndarray
        Conformal time array (shape: n_times)
    S : jnp.ndarray
        Source function (shape: n_kmodes × n_times)
    tau0 : float
        Conformal time today
    nk_fine : int, optional
        Number of k points for fine integration grid (default: 2048)
    chunk_size : int, optional
        Chunk size for multipole processing (default: 500)
    k_chunk_size : int, optional
        Chunk size for k-mode processing (default: 1024)

    Returns
    -------
    jnp.ndarray
        Multipole moments Θ_ℓ(k) with shape (nk_fine, ellmax+1)
    """
    # Create fine k-grid for better integration accuracy
    kmin = jnp.min(kmodes)
    kmax = jnp.max(kmodes)
    kmodes_fine = jnp.geomspace(kmin, kmax, nk_fine)
    log_kmodes_fine = jnp.log(kmodes_fine)

    # Interpolate S to fine k-grid
    def interpolate_S_slice(S_slice):
        spline = spline_interpolation(jnp.log(kmodes), S_slice)
        return spline.evaluate(log_kmodes_fine)

    S_fine = jax.vmap(interpolate_S_slice, in_axes=1, out_axes=1)(S)

    # Process in chunks to reduce memory usage
    def process_chunk(ell_start, ell_end, k_start, k_end):
        kmodes_chunk = kmodes_fine[k_start:k_end]
        S_chunk = S_fine[k_start:k_end, :]

        # Argument for Bessel functions: k(τ₀ - τ)
        ktau_chunk = kmodes_chunk[:, None] * (tau0 - tau[None, :])

        # Compute spherical Bessel functions for all needed ℓ
        sj_chunk = spherical_bessel(ell_end - 1, ktau_chunk.flatten())[:, 0, :]
        sj_chunk = sj_chunk.reshape(len(kmodes_chunk), len(tau), ell_end)
        sj_chunk = sj_chunk[:, :, ell_start:ell_end]

        # Line-of-sight integral: ∫ S(k,τ) j_ℓ(k(τ₀-τ)) dτ
        integrand_chunk = sj_chunk * S_chunk[:, :, None]
        theta_l_chunk = jnp.trapezoid(integrand_chunk, x=tau, axis=1)

        return theta_l_chunk

    # Process in chunks for both ℓ and k
    num_ell_chunks = (ellmax + 1 + chunk_size - 1) // chunk_size
    num_k_chunks = (nk_fine + k_chunk_size - 1) // k_chunk_size

    ell_chunks = []
    for i in range(num_ell_chunks):
        ell_start = i * chunk_size
        ell_end = min((i + 1) * chunk_size, ellmax + 1)
        if ell_start < ell_end:
            k_chunks = []
            for j in range(num_k_chunks):
                k_start = j * k_chunk_size
                k_end = min((j + 1) * k_chunk_size, nk_fine)
                if k_start < k_end:
                    k_chunk_result = process_chunk(ell_start, ell_end, k_start, k_end)
                    k_chunks.append(k_chunk_result)
            # Concatenate k chunks
            ell_chunk_result = jnp.concatenate(k_chunks, axis=0)
            ell_chunks.append(ell_chunk_result)

    # Concatenate ℓ chunks
    theta_ell = jnp.concatenate(ell_chunks, axis=1)

    return theta_ell, kmodes_fine


@jax.jit
def compute_Cell(theta_ell, kmodes_fine, n_s, k_p):
    """Compute CMB angular power spectrum C_ℓ from multipole moments.

    The angular power spectrum is computed by integrating over k:

        C_ℓ ∝ ∫ k^(n_s-1) |Θ_ℓ(k)|² dk

    where n_s is the primordial spectral index and k_p is the pivot scale.

    Parameters
    ----------
    theta_ell : jnp.ndarray
        Multipole moments Θ_ℓ(k) with shape (nk_fine, ellmax+1)
    kmodes_fine : jnp.ndarray
        Fine wavenumber grid (shape: nk_fine)
    n_s : float
        Primordial spectral index
    k_p : float
        Pivot scale (typically 0.05 Mpc⁻¹)

    Returns
    -------
    jnp.ndarray
        Angular power spectrum C_ℓ (shape: ellmax+1)
    """
    log_kmodes_fine = jnp.log(kmodes_fine)

    # Integrate over k: ∫ (k/k_p)^(n_s-1) |Θ_ℓ(k)|² d(log k)
    Cell = jnp.trapezoid(
        (kmodes_fine[:, None] / k_p)**(n_s - 1) * theta_ell**2,
        x=log_kmodes_fine,
        axis=0
    )

    return Cell


@partial(jax.jit, static_argnames=['ellmax'])
def compute_Dell(Cell, A_s, Tcmb, ellmax=None):
    """Convert C_ℓ angular power spectrum to D_ℓ = ℓ(ℓ+1)C_ℓ/(2π) in μK².

    This function converts the dimensionless C_ℓ spectrum to the commonly-used
    D_ℓ representation that includes the primordial amplitude and temperature
    normalization.

    Parameters
    ----------
    Cell : jnp.ndarray
        Angular power spectrum C_ℓ (dimensionless, shape: ellmax+1)
    A_s : float
        Primordial amplitude (typically ~2.1e-9)
    Tcmb : float
        CMB temperature in Kelvin (typically 2.7255 K)
    ellmax : int, optional
        Maximum multipole. If None, inferred from Cell.shape[0] - 1

    Returns
    -------
    ell : jnp.ndarray
        Multipole moments starting from ell=2
    D_ell : jnp.ndarray
        Temperature power spectrum D_ℓ = ℓ(ℓ+1)C_ℓ/(2π) in μK²

    Examples
    --------
    >>> Cell = compute_Cell(theta_ell, kmodes_fine, n_s=0.96, k_p=0.05)
    >>> ell, D_ell = compute_D_ell(Cell, A_s=2.1e-9, Tcmb=2.7255)
    >>> # D_ell now contains the power spectrum in μK²

    Notes
    -----
    The conversion formula is:
        D_ℓ = ℓ(ℓ+1) × C_ℓ × A_s × 2 × T_CMB²

    The factor of 2 comes from the convention difference between dimensionless
    and dimensional power spectra. We start from ell=2 since monopole (ℓ=0)
    and dipole (ℓ=1) are typically removed or unmeasured in CMB observations.
    """
    if ellmax is None:
        ellmax = Cell.shape[0] - 1

    ell = jnp.arange(ellmax + 1)

    # Apply ell(ell+1) factor, primordial amplitude, and temperature normalization
    # Start from ell=2 (monopole and dipole not included)
    # Convert from K² to μK² by multiplying by (10^6)² = 1e12
    D_ell = ell[2:] * (ell[2:] + 1) * Cell[2:] * A_s * 2 * Tcmb**2 * 1e12

    return ell[2:], D_ell


@partial(jax.jit, static_argnames=['ellmax', 'nmodes', 'kmin', 'kmax'])
def compute_Cell_spectrum_from_cosmo_params(
    param_dict,
    ellmax=2500,
    nmodes=512,
    kmin=1e-4,
    kmax=1.0
):
    """Compute CMB C_ell spectrum using DISCO-EB.

    This function follows the same pipeline as DISCOEB_CMB_spectrum_simple.ipynb
    to compute the temperature power spectrum.

    Parameters
    ----------
    param_dict : dict
        Dictionary of cosmological parameters
    ellmax : int, optional
        Maximum multipole to compute. Default: 2500
    nmodes : int, optional
        Number of k-modes. Default: 512
    kmin : float, optional
        Minimum wavenumber in 1/Mpc. Default: 1e-4
    kmax : float, optional
        Maximum wavenumber in 1/Mpc. Default: 1.0

    Returns
    -------
    ell : jnp.ndarray
        Multipole moments (starting from ell=2)
    C_ell : jnp.ndarray
        Temperature power spectrum in μK^2
    """
    # 1. Background evolution
    param = param_dict.copy()
    param = evolve_background(param=param, thermo_module='RECFAST', num_thermo=1024)

    # 2. Perturbation evolution
    aexp_out = jnp.concatenate([
        jnp.geomspace(3e-4, 5e-3, 256, endpoint=False),
        jnp.geomspace(5e-3, 1.0, 64)
    ])

    yout, kmodes, param = evolve_perturbations_batched(
        param=param,
        kmin=kmin,
        kmax=kmax,
        num_k=nmodes,
        aexp_out=aexp_out,
        rtol=1e-4,
        atol=1e-4,
        return_full=True,
        dologk=True,
    )

    # 3. Time derivatives
    tau = param['tau_out']
    yprime = compute_time_derivatives(yout, tau, kmodes, param)

    # 4. Extract parameters
    lmaxg = param['lmaxg']
    lmaxgp = param['lmaxgp']
    lmaxr = param['lmaxr']
    nqmax = param['nqmax']

    # 5. Extract perturbations
    perturbations = extract_perturbations(yout, yprime, lmaxg, lmaxgp, lmaxr)

    # 6. Compute background quantities
    background_quantities = compute_background_quantities(aexp_out, param)

    # 7. Compute visibility functions
    tau = param['tau_of_a_spline'].evaluate(aexp_out)
    visibility_functions = compute_visibility_functions(tau, param)

    # 8. Compute neutrino perturbations
    neutrinos = compute_neutrino_perturbations(
        yout, yprime, aexp_out, param, nqmax, perturbations['iq0']
    )

    # 9. Compute metric perturbations
    metric = compute_metric_perturbations(
        perturbations, neutrinos, background_quantities, param, kmodes, aexp_out
    )

    # 10. Compute source function
    source_results = compute_source_function(
        perturbations, metric, visibility_functions, yout, yprime, kmodes, lmaxg, lmaxgp
    )
    S = source_results['S']

    # 11. Line-of-sight integration
    tau0 = param['tau_of_a_spline'].evaluate(1.0)
    theta_ell, kmodes_fine = compute_theta_ell(
        ellmax=ellmax,
        kmodes=kmodes,
        tau=tau,
        S=S,
        tau0=tau0,
        # The following parameters are hardware dependent as they can cause memory overflow
        # TODO: make the batching dynamic according to how much memory is available on the user's GPU
        nk_fine=512,
        chunk_size=32,
        k_chunk_size=32
    )

    # 12. Compute C_ell
    Cell = compute_Cell(
        theta_ell=theta_ell,
        kmodes_fine=kmodes_fine,
        n_s=param['n_s'],
        k_p=param['k_p']
    )

    return Cell, param
