"""
This is a module for the basics of the astrophyscial expressions

"""

from __future__ import division
import numpy as np

from numpy import pi, sqrt, exp, power, log, log10
from scipy.integrate import trapz
from scipy.optimize import fsolve, bisect, brentq
from scipy.special import erf, lambertw
from inspect import getargspec

import os

# related to the map handling
import healpy as hp
from astropy_healpix import HEALPix
from astropy.coordinates import SkyCoord
# from astropy.io import fits
from astropy.coordinates import Galactic
# from astropy import units as u

import constants as ct

#########################################
# List of required parameters for the sources:


source_id = ['name', 'longitude', 'latitude', 'distance', 'size']
pars_always = ['alpha', 'nu_pivot', 'gamma']
pars_early = {'eff': ['L_peak', 't_peak'],
              'thy': ['L_norm', 'K2', 'beta', 'delta']
              }
pars_late = ['t_trans', 't_age', 'L_today']
# lightcurve params
pars_lightcurve = {model: pars_early[model] +
                   ['gamma']+pars_late for model in ['eff', 'thy']}

pars_required = {('eff', 'L_today'): ['L_peak', 't_peak', 'gamma', 't_trans', 't_age'],
                 ('eff', 'L_peak'): ['t_peak', 'gamma', 't_trans', 't_age', 'L_today'],
                 ('eff', 't_age'): ['L_peak', 't_peak', 'gamma', 't_trans', 'L_today'],
                 ('eff', 't_trans'): ['L_peak', 't_peak', 'gamma', 'L_today', 't_age'],
                 ('thy', 'L_today'): ['L_norm', 'K2', 'beta', 'delta', 't_trans', 'gamma', 't_age'],
                 ('thy', 'L_norm'): ['K2', 'beta', 'delta', 't_trans', 'gamma', 't_age', 'L_today']
                 }

# all params:
source_input = {key: source_id+pars_always+value +
                pars_late for key, value in pars_early.items()}


# default time array [years]
t_arr_default = np.logspace(-4, 11, 10001)


#########################################
# Dark matter and galactic functions:

def rho_NFW(r, rs, rhos):
    """
    The density profile [GeV/cm**3] of an NFW halo.

    Parameters
    ----------
    r : the distance from the center [kpc]
    rs : the NFW r_s parameter [kpc]
    rhos : the NFW rho_s parameter [GeV/cm**3]
    """
    res = rhos/(r/rs)/(1.+r/rs)**2
    return res


def rho_MW(r):
    """
    The density [GeV/cm**3] of the Milky Way.

    Parameters
    ----------
    x : distance from the center [kpc]
    """
    rs = ct._rs_of_MW_
    rho_at_sun = ct._rho_local_DM_
    rsun = ct._Sun_to_gal_center_
    rhos = rho_at_sun * (rsun/rs) * (1. + rsun/rs)**2
    res = rho_NFW(r, rs, rhos)
    try:
        res = res.astype(float)
    except:
        pass
    return res


def theta_gal_ctr(l, b, output_radians=True):
    """
    Computes the theta angle [radians/degrees] that a point of galactic coordinates (l, b) makes with the galactic center.

    Parameters
    ----------
    l : galactic longitude [deg]
    b : galactic latitude [deg]
    output_radians : whether the result is given in radians (default: True)
    """

    theta = np.arccos(np.cos(l*pi/180.)*np.cos(b*pi/180.))

    if output_radians:
        return theta
    else:
        return theta*180./pi


def r_to_gal(x, th, r0=ct._Sun_to_gal_center_):
    """
    The radius [kpc] of a point to the galactic center.

    Parameters
    ----------
    x : distance of a point on the line of sight [kpc]
    th : angle between galaxy center and light source [radian]
    r0 : distance from the Sun to the galaxy center [kpc]  (default: 8.1)
    """
    r2 = x**2 + r0**2 - 2.*x*r0*np.cos(th)
    r = np.sqrt(r2)
    try:
        r = r.astype(float)
    except:
        pass
    return r


#########################################
# Light-related functions:


def flux_density_to_psd(nu, Snu, Omega):
    """
    Convert flux density S_nu to phase space density f. Returns Energy [eV] and f = S_\nu/Omega * 2*pi^2/E^3.

    Parameters
    ----------
    nu : array of the frequency [GHz]
    Snu : array of S_\nu [Jy]
    Omega :  the solid angle the source subtends [sr]
    """
    Snu_in_eV3 = Snu * ct._Jy_over_eV3_
    E = nu * ct._GHz_over_eV_
    res = Snu_in_eV3 / Omega * 2. * np.pi**2 / E**3
    return (E.astype(float), res.astype(float))


def psd_to_flux_density(E, f, Omega):
    """
    Converting phase space density f to flux density S_nu. Returns frequency nu [GHz] and spectral irradiance S_nu = E^3/2./pi^2*f*Omega [Jy].

    Parameters
    ----------
    E : array of energy [eV]
    f : p.s.d 
    Omega : the angle the source subtends [sr]
    """
    Snu = E**3 / 2. / np.pi**2 * f * Omega / ct._Jy_over_eV3_
    nu = E / ct._GHz_over_eV_
    return (nu, Snu)


def flux(nu, Snu):
    """
    Integrate S_\nu w.r.t. nu to get S [eV^4]. Returns the flux (irradiance) [eV^4].

    Parameters
    ----------
    nu : the range of the frequency to be integrated over [GHz]
    Snu : the flux density [Jy]
    """
    # fix unit
    nu_in_eV = nu * ct._GHz_over_eV_
    Snu_in_eV3 = Snu * ct._Jy_over_eV3_
    res = trapz(Snu_in_eV3, nu_in_eV)
    return res


def irrad(distance, Lum):
    """
    Returns the flux density (spectral irradiance) [Jy] of a source.

    Parameters
    ----------
    distance : distance to source [kpc]
    Lum : spectral luminosity of source [erg * s^-1 * Hz^-1]
    """

    Area = 4.*pi * \
        (distance*ct._kpc_over_cm_)**2.  # [cm^2] sphere covering the source
    irrad = Lum * 1.e23 / Area  # [Jy]

    return irrad


def S_cygA(nu):
    """
    The flux density [Jy] S_\nu of Cygnus A.

    Parameters
    ----------
    nu : frequency, can be scalar or array [GHz]
    """

    log_res = []
    nu_lst = np.asarray(nu) * 1.e3  # converting to MHz
    flg_scalar = False
    if nu_lst.ndim == 0:
        nu_lst = nu_lst[None]
        flg_scalar = True
    for nui in nu_lst:
        if nui < 2000.:
            a = 4.695
            b = 0.085
            c = -0.178
        else:
            a = 7.161
            b = -1.244
            c = 0.
        log_res.append(a + b * np.log10(nui) + c * np.log10(nui)**2)
    log_res = np.asarray(log_res)
    res = 10.**(log_res)
    if flg_scalar:
        res = np.squeeze(res)
    return res


#########################################
# Source model functions:


def FreeErrorMssg(model):
    """
    A custom error message for different free-expansion evolutions.
    """

    if model in pars_early.keys():
        return "For model=={}, we need the arguments: {}.".format(model, ['t']+pars_early[model])
    else:
        return "Currently, this function can only support model=='eff' (Bietenholz et al., arXiv:2011.11737) or model=='thy'/'thy-alt' (Weiler et al., 1986ApJ...301..790W). model={} is not supported.".format(str(model))


def AdiabaticErrorMssg():
    """
    A custom error message for the adiabatic evolution.
    """

    return "L_adiab needs the arguments: ['t_ref', 'L_ref', 'gamma']."


# adiabatic and spectral indices
def gamma_from_alpha(alpha):
    """
    Adiabatic index from spectral index, Sedov-Taylor expansion.

    Parameters
    ----------
    alpha : spectral index
    """
    gamma = (4./5)*(2.*alpha + 1.)
    return gamma


def alpha_from_gamma(gamma):
    """
    Spectral index from adiabatic index, Sedov-Taylor expansion.

    Parameters
    ----------
    gamma : adiabatic index
    """
    alpha = ((5./4)*gamma - 1.)/2.
    return alpha


def nu_factor(nu, nu_pivot, alpha):
    """
    Frequency rescaling factor, depending on the spectral index.

    Parameters
    ----------
    nu : frequency [GHz]
    nu_pivot : pivot (i.e. reference) frequency [GHz]
    alpha : spectral index
    """

    return (nu/nu_pivot)**-alpha


# SNR light-curves
def L_eff(t, L_peak, t_peak):
    """
    Returns the early evolution (i.e. during free expansion) of the SNR radio spectral luminosity [erg * s^-1 * Hz^-1] as a function of time after explosion [years] and frequency [GHz], according to the effective model from Bietenholz et al., arXiv:2011.11737. NOTE: frequency dependence prefactor, usually (nu/nu_pivot)^-alpha, is factorized out.

    Parameters
    ----------
    t : time after explosion [years]
    L_peak : peak spectral luminosity [erg * s^-1 * Hz^-1]
    t_peak : peak time [days]
    """

    t_days = t*365.  # [years] -> [days]

    return L_peak * exp(-1.5 * (t_peak/t_days - 1.)) * (t_days/t_peak)**-1.5


def L_thy(t, L_norm, beta, K2, delta, tau_factor=1.):
    """
    Returns the early evolution (i.e. during free expansion) of the SNR radio spectral luminosity [erg * s^-1 * Hz^-1] as a function of time after explosion [years] and frequency [GHz], according to the theoretical model from Weiler et al., 1986ApJ...301..790W. NOTE: frequency dependence prefactor, usually (nu/nu_pivot)^-alpha, is factorized out.

    Parameters
    ----------
    t : time after explosion [years]
    L_norm : normalization spectral luminosity [erg * s^-1 * Hz^-1]
    beta : post-peak free expansion evolution [defined in the exponent with a minus sign]
    K2 : optical depth normalization
    delta : optical depth time evolution [defined in the exponent with a minus sign]
    tau_factor : frequency-dependent power for opacity tau; in Weiler's it's (nu/5.)**-2.1 (default: 1.)
    """
    # In Weiler's, factor = (nu/5)^-alpha, while tau_factor = (nu/5.)**-2.1

    t_days = t*365.  # [years] -> [days]
    tau = tau_factor * K2 * t_days**(-delta)

    return L_norm * t_days**(-beta) * exp(-tau)


def L_free(t, model='eff', **kwargs):
    """
    Returns the free expansion evolution of the SNR radio spectral luminosity [erg * s^-1 * Hz^-1] as a function of time after explosion [years] and frequency [GHz]. NOTE: frequency dependence prefactor, usually (nu/nu_pivot)^-alpha, is factorized out.

    Parameters
    ----------
    t : time after explosion [years]
    model : model for the free expansion evolution, either 'eff' or 'thy' (default: 'eff')
    """

    # Checking kwargs:
    # subset of kwargs that is in pars_early[model]
    pars_inter = set(pars_early[model]).intersection(kwargs.keys())
    if pars_inter != set(pars_early[model]):
        raise KeyError("The kwargs passed do not contain all the arguments for the model requested. For model={} these are: {}".format(
            model, pars_early[model]))

    # reading off the arguments of the early evolution functions
    if model == 'eff':
        Learly_pars = getargspec(L_eff)[0]
    elif model == 'thy':
        Learly_pars = getargspec(L_thy)[0]

    # building the kwargs for the free expansion lightcurve
    free_kwargs = {}
    for par in Learly_pars:
        if par in kwargs.keys():
            free_kwargs[par] = kwargs[par]

    try:
        if model == 'eff':
            Lf = L_eff(t, **free_kwargs)

        elif model == 'thy':
            # in principle, L_norm could be absent from free_kwargs; instead there being 'K1' and 'distance'. An old version of this code allowed for that. We have decided to restrict it.
            Lf = L_thy(t, **free_kwargs)

    except TypeError:
        raise TypeError(FreeErrorMssg(model=model))

    return Lf


def L_adiab(t, t_ref, L_ref, gamma):
    """
    Returns the adiabatic (Sedov-Taylor) evolution of the SNR radio spectral luminosity [erg * s^-1 * Hz^-1] as a function of time after explosion [years] and frequency [GHz]. NOTE: frequency dependence prefactor, usually (nu/nu_pivot)^-alpha, is factorized out.

    Parameters
    ----------
    t : time after explosion [years]
    t_ref : reference time after explosion [years]
    L_ref : spectral luminosity at reference time [erg * s^-1 * Hz^-1]
    gamma : adiabatic expansion evolution [defined in the exponent with a minus sign]
    """

    return L_ref * (t/t_ref)**(-gamma)


def sandwich_logeqn(Lpk, tpk, L0, t0, gamma, tt):
    """
    Returns the equation to minimize in order to find the transition time between the effective free expansion and the adiabatic expansion.

    Parameters
    ----------
    Lpk : peak spectral luminosity [erg * s^-1 * Hz^-1]
    tpk : peak time [days]
    L0 : SNR spectral luminosity today [erg * s^-1 * Hz^-1]
    t0 : SNR age [years]
    gamma : adiabatic index
    tt : free-adiabatic phases transition time [years]
    """

    tpk_yr = tpk/365.

    eqn = log10(Lpk/L0) + (1.5)*(1. - tpk_yr/tt)*log10(np.e) + \
        (gamma-1.5)*log10(tt) + 1.5*log10(tpk_yr) - gamma*log10(t0)

    return eqn


def crossings(fn, arr):
    """
    Find where a function crosses 0. Returns the zeroes of the function.

    Parameters
    ----------
    fn : function
    arr : array of arguments for function
    """

    fn_arr = fn(arr)

    where_arr = np.where(np.logical_or((fn_arr[:-1] < 0.) * (fn_arr[1:] > 0.),
                                       (fn_arr[:-1] > 0.) * (fn_arr[1:] < 0.))
                         )[0]

    cross_arr = []
    if len(where_arr) > 0:
        for i in range(len(where_arr)):
            cross_arr.append(
                brentq(fn, arr[where_arr[i]],
                       arr[where_arr[i] + 1])
            )

    cross_arr = np.array(cross_arr)

    return cross_arr


def tage_compute(Lpk, tpk, t_trans, L_today, gamma):
    """
    Returns the age [years] of a SNR given Lpk, tpk, t_trans, L_today, and gamma.

    Parameters
    ----------
    Lpk : peak spectral luminosity [erg * s^-1 * Hz^-1]
    tpk : peak time [days]
    t_trans : free-adiabatic phases transition time [years]
    L_today : SNR spectral luminosity today [erg * s^-1 * Hz^-1]
    gamma : adiabatic index
    """

    def Lf_fn(tt): return L_eff(tt, Lpk, tpk)
    L_trans = Lf_fn(t_trans)  # L_trans

    t_age = (L_today/L_trans)**(-1/gamma) * t_trans  # t_age

    return t_age


def L_source(t, model='eff', output_pars=False, **kwargs):
    """
    Returns the SNR radio luminosity [erg * s^-1 * Hz^-1] as a function of time after explosion [years] and frequency [GHz]. NOTE: frequency dependence prefactor, usually (nu/nu_pivot)^-alpha, is factorized out.

    Parameters
    ----------
    t : time after explosion [years]
    model : model for the free expansion evolution, either 'eff' or 'thy' (default: 'eff')
    output_pars : whether to output a dictionary with the parameters used
    """

    if not 'gamma' in kwargs.keys():
        raise KeyError(
            "'gamma' is not in kwargs. Please pass 'gamma' as an argument.")
    gamma = kwargs['gamma']

    # list of lightcurve parameters that are known (i.e. present in kwargs)
    known = {}
    # list of lightcurve parameters that are to be deduced (i.e. are missing in kwargs)
    to_deduce = []
    for par in pars_lightcurve[model]:
        if par in kwargs.keys():
            known[par] = kwargs[par]
        else:
            to_deduce.append(par)
    if (model == 'thy') and ('tau_factor' in kwargs.keys()):  # special case
        known['tau_factor'] = kwargs['tau_factor']

    if len(to_deduce) > 1:  # too many missing parameters
        raise ValueError(
            "to_deduce={} is too large. Please include more parameters in kwargs.".format(to_deduce))

    to_deduce = to_deduce[0]  # extracting the only parameter to be deduced
    # the parameters required in kwargs
    required = pars_required[(model, to_deduce)]
    if not set(required).issubset(set(known.keys())):
        raise ValueError("known={} is too short. It should be a subset of the required parameters, which are {}. Please include more parameters in kwargs.".format(
            known.keys(), required))

    if model == 'eff':

        if to_deduce == 'L_today':  # based on peak info, deduce L_today

            t_today = known['t_age']  # age [years]
            t_trans = known['t_trans']  # transition time [years]
            # computing transition luminosity
            L_trans = L_free(t_trans, model=model, **known)
            adiab_kwargs = {'L_ref': L_trans, 't_ref': t_trans, 'gamma': gamma}

            L_today = L_adiab(t_today, **adiab_kwargs)
            known.update({'L_today': L_today})

        elif to_deduce == 'L_peak':  # based on t_peak and today's info, deduce L_peak

            t_age = known['t_age']  # age [years]
            t_trans = known['t_trans']  # transition time [years]
            L_today = known['L_today']  # spectral luminosity [today]
            adiab_kwargs = {'L_ref': L_today, 't_ref': t_age, 'gamma': gamma}
            eff_kwargs = {}
            eff_pars = getargspec(L_eff)[0]
            for par, val in known.items():
                if par in eff_pars:
                    eff_kwargs[par] = val

            # computing transition luminosity
            L_trans = L_adiab(t_trans, **adiab_kwargs)

            def LogLeff_fn(LogLpk): return log10(
                L_eff(t_trans, L_peak=10**LogLpk, **eff_kwargs)) - log10(L_trans)
            L_peak = 10.**fsolve(LogLeff_fn, log10(L_trans)+1.)
            known.update({'L_peak': np.squeeze(L_peak)})

        elif to_deduce == 't_trans':  # based on t_age and both today's and peak info, deduce t_trans

            t_peak = known['t_peak']
            t_age = known['t_age']  # age
            L_peak = known['L_peak']
            L_today = known['L_today']  # spectral luminosity today
            adiab_kwargs = {'L_ref': L_today, 't_ref': t_age, 'gamma': gamma}

            def fn(Ltt): return sandwich_logeqn(L_peak, t_peak, L_today,
                                                t_age, gamma, 10.**Ltt)  # function to minimize
            try:
                Lt_cross = crossings(fn, log10(t_arr_default))
                t_cross = 10.**Lt_cross
                t_trans = max(t_cross)
            except ValueError:
                t_trans = 1.e11  # made-up, very large value
            known.update({'t_trans': t_trans})

        elif to_deduce == 't_age':  # based on t_trans and both today's and peak info, deduce t_age

            t_peak = known['t_peak']
            t_trans = known['t_trans']  # transition time
            L_peak = known['L_peak']
            L_today = known['L_today']  # spectral luminosity today

            # simplified free expansion function
            def Lf_fn(tt): return L_free(tt, model=model, **known)

            L_trans = Lf_fn(t_trans)  # L_trans
            adiab_kwargs = {'L_ref': L_trans, 't_ref': t_trans, 'gamma': gamma}
            t_age = tage_compute(L_peak, t_peak, t_trans,
                                 L_today, gamma)  # age
            known.update({'t_age': t_age})

        else:
            raise ValueError(
                "to_deduce={}, known={} is currently not supported.".format(to_deduce, known))

    elif model == 'thy':

        if 'L_today' in to_deduce:

            t_today = known['t_age']  # age [years]
            t_trans = known['t_trans']  # transition time [years]
            # computing transition luminosity
            L_trans = L_free(t_trans, model=model, **known)
            adiab_kwargs = {'L_ref': L_trans, 't_ref': t_trans, 'gamma': gamma}

            L_today = L_adiab(t_today, **adiab_kwargs)
            known.update({'L_today': L_today})

        elif 'L_norm' in to_deduce:

            t_age = known['t_age']  # age [years]
            t_trans = known['t_trans']  # transition time [years]
            L_today = known['L_today']  # spectral luminosity [today]
            adiab_kwargs = {'L_ref': L_today, 't_ref': t_age, 'gamma': gamma}
            thy_kwargs = {}
            thy_pars = getargspec(L_thy)[0]
            for par, val in known.items():
                if par in thy_pars:
                    thy_kwargs[par] = val
            # computing transition luminosity
            L_trans = L_adiab(t_trans, **adiab_kwargs)
            def LogLthy_fn(LogLnorm): return log10(
                L_thy(t_trans, L_norm=10**LogLnorm, **thy_kwargs)) - log10(L_trans)
            L_norm = 10.**fsolve(LogLthy_fn, log10(L_trans)+1.)
            known.update({'L_norm': np.squeeze(L_norm)})

    if output_pars:
        return_pars = {}
        return_pars.update(known)
        return_pars.update(adiab_kwargs)

    Lf = L_free(t, model=model, **known)  # computing early-times luminosity
    La = L_adiab(t, **adiab_kwargs)  # computing adiabatic luminosity

    Lum = Lf*np.heaviside(t_trans - t, 1.) + La*np.heaviside(t - t_trans, 0.)

    if output_pars:
        return Lum, return_pars
    else:
        return Lum


# SNR light-curve analytic formulas, assuming the Bietenholz effective model.
def Snu_supp(gamma, frac_tpk, frac_tt):
    """
    Fractional suppression of spectral irradiance (i.e. S0/S_peak = L0/L_peak).

    Parameters
    ----------
    gamma : adiabatic index
    frac_tpk : ratio of peak day to SNR age
    frac_tt : ratio of transition time to SNR age
    """

    first = exp(1.5 - 1.5*frac_tpk/frac_tt)
    second = (frac_tt/frac_tpk)**-1.5
    third = (1./frac_tt)**-gamma

    return first*second*third


def ftt(gamma, frac_tpk, sup):
    """
    Fractional transition time (i.e. t_trans/t_age).

    Parameters
    ----------
    gamma : adiabatic index
    frac_tpk : ratio of peak day to SNR age
    sup : spectral irradiance suppression
    """

    const = 1./(2.*gamma/3. - 1.)

    arg = const * exp(const) * frac_tpk**(const+1.) * sup**(-2.*const/3.)

    num = const*frac_tpk
    den = lambertw(arg)

    return np.real(num/den)


def dimless_free(frac_tpk, tau):
    """
    Dimensionless free expansion lightcurve.

    Parameters
    ----------
    frac_tpk : ratio of peak day to SNR age
    tau : ratio of time to SNR age
    """

    return exp(1.5 - 1.5*frac_tpk/tau) * (tau/frac_tpk)**-1.5


def dimless_adiab(gamma, sup, tau):
    """
    Dimensionless adiabatic expansion lightcurve.

    Parameters
    ----------
    gamma : adiabatic index
    sup : spectral irradiance suppression
    tau : ratio of time to SNR age
    """

    return sup*(tau)**-gamma


def dimless_lum(gamma, frac_tpk, sup, tau):
    """
    Dimensionless full lightcurve.

    Parameters
    ----------
    gamma : adiabatic index
    frac_tpk : ratio of peak day to SNR age
    sup : spectral irradiance suppression
    tau : ratio of time to SNR age
    """

    frac_tt = ftt(gamma, frac_tpk, sup)

    free = dimless_free(frac_tpk, tau)
    adiab = dimless_adiab(gamma, sup, tau)

    return free*np.heaviside(frac_tt-tau, 0.) + adiab*np.heaviside(tau-frac_tt, 1.)


#########################################
# Noise and signal functions:


def SKA_exper_nu(nu):
    """
    Returns the SKA experiment mode (low/mid) sensitive to the given frequency nu [GHz].

    Parameters
    ----------
    nu : frequency [GHz]
    """

    if (nu < ct._nu_min_ska_low_):  # frequency below SKA low lower threshold
        exper_mode = None  # just a placeholder, won't matter
    elif (nu <= ct._nu_max_ska_low_):  # frequency within SKA low range
        exper_mode = 'SKA low'
    elif (nu <= ct._nu_max_ska_mid_):  # frequency within SKA mid range
        exper_mode = 'SKA mid'
    else:  # frequency above SKA mid upper threshold
        exper_mode = None  # just a placeholder, won't matter

    return exper_mode


def SKA_specs(nu, exper_mode):
    """
    Returns the specifications (area [m^2], window, receiver noise brightness temperature [K]) of the SKA experiment mode, for the given frequency [GHz].

    Parameters
    ----------
    nu : frequency [GHz]
    exper_mode : mode in which the experiment is working
    """

    if exper_mode == None:
        area, window, Tr = 0., 0., 0.
    elif exper_mode == 'SKA low':
        area = ct._area_ska_low_
        window = np.heaviside(nu - ct._nu_min_ska_low_, 1.) * \
            np.heaviside(ct._nu_max_ska_low_ - nu, 1.)
        Tr = 40.
    elif exper_mode == 'SKA mid':
        area = ct._area_ska_mid_
        window = np.heaviside(nu - ct._nu_min_ska_mid_, 0.) * \
            np.heaviside(ct._nu_max_ska_mid_ - nu, 1.)
        Tr = 20.

    return area, window, Tr


def bg_408_temp(l, b, size=None, average=False, verbose=False):
    """
    Reads the background brightness temperature at 408 MHz at galactic coordinates l and b.

    Parameters
    ----------
    l : longitude [deg]
    b : latitude [deg]
    size : angular size [sr] of the region of interest (default: None)
    average : whether the brightness temperature is averaged over the size of the region of interest (default: False)
    """

    try:
        # when we run inside ./
        path = './data/'
        path = os.path.join(os.getcwd(), path, 'lambda_haslam408_dsds.fits')
        map_allsky_408 = hp.read_map(path, verbose=verbose)
    except FileNotFoundError:
        # useful when we run inside ./workspace_notebooks/
        try:
            path = '../data/'
            path = os.path.join(os.getcwd(), path,
                                'lambda_haslam408_dsds.fits')
            map_allsky_408 = hp.read_map(path, verbose=verbose)
        except FileNotFoundError:
            raise Exception(
                'Haslam map (lambda_haslam408_dsds.fits) is not found.')

    healp = HEALPix(nside=512, order='ring', frame=Galactic())

    pos_coords = SkyCoord(frame="galactic", l=l, b=b, unit='deg')
    pos_pix = healp.skycoord_to_healpix(pos_coords)

    if average and size != None:
        vec = hp.pix2vec(nside=512, ipix=pos_pix)
        new_pos_pix = hp.query_disc(
            nside=512, vec=vec, radius=ct.solid_angle_to_angle(size))
    else:
        new_pos_pix = pos_pix

    # query the background temperature for the echo

    bg_T408 = np.average(map_allsky_408[new_pos_pix])

    return bg_T408


def T_noise(nu, Tbg_at_408=27, beta=-2.55, Tr=0.):
    """
    The background noise temperature [K]

    Parameters
    ----------
    nu: frequency [GHz]
    Tbg_at_408: the MW background at 408 MHz [K] (default: 27, for Cygnus A gegenschein position)
    beta: the index for the Milky (default: -2.55 from Ghosh paper)
    Tr: the receiver's noise brightness temperature (default: 0.)
    """

    Tcmb = 2.7255  # cmb brightness [K]
    Ta = 3.  # atmospheric brightness [K]
    # Tbg_0 = 60.  # at 0.3 GHz [K]
    # Tbg = Tbg_0 * (nu/0.3)**beta
    Tbg = Tbg_at_408 * (nu/0.408)**beta

    res = Tcmb + Ta + Tr + Tbg

    return res


def P_noise(T_noise, delnu, tobs):
    """
    The power of the noise [eV^2].

    Parameters
    ----------
    T_noise: the temperature of the noise [K]
    delnu: the bandwidth of the detector [GHz]
    tobs: the total observation time [hour]
    """

    res = 2. * T_noise * ct._K_over_eV_ * \
        np.sqrt(delnu * ct._GHz_over_eV_/(tobs * ct._hour_eV_))
    return res


def P_signal(S, A, eta=ct._eta_ska_, f_Delta=1.):
    """
    The signal power, assuming given bandwidth [eV^2].

    Parameters
    ----------
    S: the (integrated) flux (irradiance) [eV^4]
    A: the area of the detector [m^2]
    eta: the detector efficiency (default: 0.8)
    f_Delta: the fraction of signal falling withing the bandwidth
    """
    res = S * eta * A * f_Delta
    # fix units
    res *= ct._m_eV_**2
    return res
