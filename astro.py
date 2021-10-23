"""
This is a module for the basics of the astrophysical expressions

"""

from __future__ import division
import numpy as np

from numpy import pi, sqrt, exp, power, log, log10
from scipy.integrate import trapz
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

import tools as tl
import constants as ct
import particle as pt
import ska as sk

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


def rho_MW(r, DM_profile="NFW"):
    """
    The density [GeV/cm**3] of the Milky Way.

    Parameters
    ----------
    x : distance from the center [kpc]
    """
    if DM_profile == "NFW":
        rs = ct._rs_of_MW_
        rho_at_sun = ct._rho_local_DM_
        rsun = ct._Sun_to_gal_center_
        rhos = rho_at_sun * (rsun/rs) * (1. + rsun/rs)**2
        res = rho_NFW(r, rs, rhos)
        try:
            res = res.astype(float)
        except:
            pass
    elif DM_profile == "Burkert":
        x = r / ct._r_H_
        res = ct._rho_H_ / (1. + x) / (1. + x**2)
    else:
        raise Exception(
            "The DM profile can only be NFW or Burkert. Please check the input.")
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
    r = sqrt(r2)
    try:
        r = r.astype(float)
    except:
        pass
    return r


#########################################
# Light-related functions:


def flux_density_to_psd(nu, Snu, Omega):
    """
    Convert flux density S_nu to phase space density f. Returns Energy [eV] and f = S_nu/Omega * 2*pi^2/E^3.

    Parameters
    ----------
    nu : array of the frequency [GHz]
    Snu : array of S_nu [Jy]
    Omega :  the solid angle the source subtends [sr]
    """
    Snu_in_eV3 = Snu * ct._Jy_over_eV3_
    E = ct._h_ * nu * ct._GHz_over_eV_
    res = Snu_in_eV3 / Omega * 2. * np.pi**2 / E**3

    return (E, res)


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
    nu = E / ct._GHz_over_eV_ / ct._h_
    return (nu, Snu)


def flux(nu, Snu):
    """
    Integrate S_nu w.r.t. nu to get S [eV^4]. Returns the flux (irradiance) [eV^4].

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


def irrad(distance, lumin):
    """
    Returns the flux density (spectral irradiance) [Jy] of a source.

    Parameters
    ----------
    distance : distance to source [kpc]
    lumin : spectral luminosity of source [erg * s^-1 * Hz^-1]
    """

    area = 4.*pi * \
        (distance*ct._kpc_over_cm_)**2.  # [cm^2] sphere covering the source
    irrad = (lumin/area) / ct._Jy_over_cgs_irrad_  # [Jy]

    return irrad


def lumin(distance, irrad):
    """
    Returns the spectral luminosity [erg * s^-1 * Hz^-1] of a source.

    Parameters
    ----------
    distance : distance to source [kpc]
    irrad : flux density (spectral irradiance) of source [Jy]
    """

    area = 4.*pi * \
        (distance*ct._kpc_over_cm_)**2.  # [cm^2] sphere covering the source
    lumin = (irrad*ct._Jy_over_cgs_irrad_)*area  # [erg * s^-1 * Hz^-1]

    return lumin


def S_cygA(nu):
    """
    The flux density [Jy] S_nu of Cygnus A.

    Parameters
    ----------
    nu : frequency, can be scalar or array [GHz]
    """

    log_res = []

    nu_lst, flg_scalar = tl.treat_as_arr(nu)  # scalar --> array trick
    nu_lst *= 1.e3  # converting to MHz

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

    try:
        to_deduce = to_deduce[0]  # extracting the only parameter to be deduced

        # the parameters required in kwargs
        required = pars_required[(model, to_deduce)]
        if not set(required).issubset(set(known.keys())):
            raise ValueError("known={} is too short. It should be a subset of the required parameters, which are {}. Please include more parameters in kwargs.".format(
                known.keys(), required))

    except IndexError:
        to_deduce = None  # no parameter to deduce!

    if model == 'eff':

        if (to_deduce == 'L_today') or (to_deduce == None):  # based on peak info, deduce L_today

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

            L_peak = tl.zeros(LogLeff_fn, log10(L_arr_default))
            L_peak = np.squeeze(10.**L_peak)

            known.update({'L_peak': L_peak})

        elif to_deduce == 't_trans':  # based on t_age and both today's and peak info, deduce t_trans

            t_peak = known['t_peak']
            t_age = known['t_age']  # age
            L_peak = known['L_peak']
            L_today = known['L_today']  # spectral luminosity today
            adiab_kwargs = {'L_ref': L_today, 't_ref': t_age, 'gamma': gamma}

            def fn(Ltt): return sandwich_logeqn(L_peak, t_peak, L_today,
                                                t_age, gamma, 10.**Ltt)  # function to minimize
            try:
                # NOTE: added 'np.squeeze'
                Lt_cross = np.squeeze(tl.zeros(fn, log10(t_arr_default)))
                t_cross = 10.**Lt_cross

                try:
                    t_trans = max(t_cross)
                except:
                    t_trans = t_cross
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

        if (to_deduce == 'L_today') or (to_deduce == None):

            t_today = known['t_age']  # age [years]
            t_trans = known['t_trans']  # transition time [years]
            # computing transition luminosity
            L_trans = L_free(t_trans, model=model, **known)
            adiab_kwargs = {'L_ref': L_trans, 't_ref': t_trans, 'gamma': gamma}

            L_today = L_adiab(t_today, **adiab_kwargs)
            known.update({'L_today': L_today})

        elif to_deduce == 'L_norm':

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

            L_norm = tl.zeros(LogLthy_fn, log(L_arr_default))
            L_norm = np.squeeze(10.**L_norm)

            known.update({'L_norm': L_norm})

    if output_pars:
        return_pars = {}
        return_pars.update(known)
        return_pars.update(adiab_kwargs)

    Lf = L_free(t, model=model, **known)  # computing early-times luminosity
    La = L_adiab(t, **adiab_kwargs)  # computing adiabatic luminosity

    if not 'use_free_expansion' in kwargs.keys():
        use_free_expansion = 1.
    else:
        if kwargs['use_free_expansion'] is False:
            use_free_expansion = 0.
        else:
            use_free_expansion = 1.

    Lum = Lf*np.heaviside(t_trans - t, 1.) * \
        use_free_expansion + La*np.heaviside(t - t_trans, 0.)

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


###############################
# SNR evolution analytic models
###############################

# ---------------------------------
# Phenomenological evolution model

def R_pheno(t, Rst=3.8, tst=360., eta1=1., eta2=0.4):
    """
    SNR radius [pc] as a simple phenomenological broken power law. The default values are taken from Tables 2 & 3 of the Truelove-McKee '99 paper for n=0 ejecta, with M_ej = 1 M_sun, E_sn = 1.e51 erg, and n0 = 0.2 cm^-3.

    Parameters
    ----------
    t : SNR age [years]
    Rst : radius [pc] at the start of the Sedov-Taylor (adiabatic) expansion phase (default: 3.8)
    tst : age [years] at the start of the Sedov-Taylor (adiabatic) expansion phase (default 360.)
    eta1 : the power scaling R~t^eta1 during the Ejecta-Dominated phase (default: 1.)
    eta2 : the power scaling R`t^eta2 during the Sedov-Taylor expansion phase (default: 2/5 = 0.4)
    """

    ed = Rst * (t/tst)**eta1 * np.heaviside(tst-t, 0.)
    ad = Rst * (t/tst)**eta2 * np.heaviside(t-tst, 1.)

    return ed+ad


def pheno_age(R, Rst=3.8, tst=360., eta1=1., eta2=0.4):
    """
    SNR age [years] as deduced from a phenomenological broken power-law function of the SNR blast radius [pc]. The default values are taken from Tables 2 & 3 of the Truelove-McKee '99 paper for n=0 ejecta, with M_ej = 1 M_sun, E_sn = 1.e51 erg, and n0 = 0.2 cm^-3.

    Parameters
    ----------
    R : SNR radius [pc]
    Rst : radius [pc] at the start of the Sedov-Taylor (adiabatic) expansion phase (default: 3.8)
    tst : age [years] at the start of the Sedov-Taylor (adiabatic) expansion phase (default 360.)
    eta1 : the power scaling R~t^eta1 during the Ejecta-Dominated phase (default: 1.)
    eta2 : the power scaling R`t^eta2 during the Sedov-Taylor expansion phase (default: 2/5 = 0.4)
    """

    def logDelRt(t, r): return log10(
        R_pheno(t, Rst, tst, eta1=eta1, eta2=eta2)) - log10(r)

    R_arr, _ = tl.treat_as_arr(R)
    age = np.array([tl.zeros(logDelRt, t_arr_default, r) for r in R_arr])
    age = np.squeeze(age)

    return age


# -------------------------
# Physical evolution model


def ED_fn(t, t_bench, R_bench, model):
    """
    Blast radius [pc] as a function of time for the Ejecta-Dominated era. Based on Truelove & McKee 1999 (TM99).

    Parameters
    ----------
    t : time [years]
    t_bench : benchmark time [years]
    R_bench : benchmark radius [pc]
    model : model of the expansion ('TM99-simple'/'TM99-0' for a simplified version of TM99 or for the case with n=0 ejecta.)
    """

    tstar = t/t_bench

    if model == 'TM99-simple':
        # power law R~t
        Rstar = tstar
    elif model == 'TM99-0':
        # TM99, Table 5
        Rstar = 2.01*tstar*(1. + 1.72 * tstar**(3./2.))**(-2./3.)

    return R_bench*Rstar


def ST_fn(t, t_bench, R_bench, model):
    """
    Blast radius [pc] as a function of time for the Sedov-Taylor era. Based on Truelove & McKee 1999 (TM99).

    Parameters
    ----------
    t : time [years]
    t_bench : benchmark time [years]
    R_bench : benchmark radius [pc]
    model : model of the expansion ('TM99-simple'/'TM99-0' for a simplified version of TM99 or for the case with n=0 ejecta.)
    """

    tstar = t/t_bench

    if model == 'TM99-simple':
        # power law R~t^(2/5)
        Rstar = tstar**(2./5.)
    elif model == 'TM99-0':
        # TM99, Table 5
        arg = (1.42*tstar - 0.254)
        # regularizing:
        arg = np.where(arg <= 0., 1.e-100, arg)
        Rstar = arg**(2./5.)

    return R_bench*Rstar


def Rb_TM99(t, t_bench, R_bench, model):
    """
    Blast radius [pc] as a function of SNR age [years]. Based on Truelove & McKee 1999 (TM99).

    Parameters
    ----------
    t : time [years]
    t_bench : benchmark time [years]
    R_bench : benchmark radius [pc]
    model : model of the expansion ('TM99-simple'/'TM99-0' for a simplified version of TM99 or for the case with n=0 ejecta.)
    """
    # broken function:
    if model == 'TM99-simple':
        two_phase = np.minimum.reduce(
            [ED_fn(t, t_bench, R_bench, model), ST_fn(t, t_bench, R_bench, model)])

    elif model == 'TM99-0':
        two_phase = np.maximum.reduce(
            [ED_fn(t, t_bench, R_bench, model), ST_fn(t, t_bench, R_bench, model)])

    return two_phase


def physics_age(R, model='estimate', M_ej=1., E_sn=1., rho0=1.):
    """
    SNR age [years] as deduced from a physically-motivated a function of the SNR blast radius [pc]. Using either a simple model, or formulas by Truelove & McKee 1999 (TM99).

    Parameters
    ----------
    R : SNR radius today [pc]
    model : 'estimate'/'TM99-simple'/TM99-0'': whether the simple one-phase 'estimate' model is used, or instead the two-phase Truelove-McKee model (ED-ST, or Ejecta-Dominated -- Sedov-Taylor), either in a simplified form, or for n=0 (uniform) ejecta profile.
    M_ej : Mass of the ejecta [M_sun] (default: 1.)
    E_sn : Energy of the SNR [1.e51 ergs] (default: 1.)
    rho0 : Mass density of surrounding medium [m_proton/cm^3] (default: 1.)
    """
    if not model in ['estimate', 'TM99-simple', 'TM99-0']:
        raise ValueError("model={} is currently unsupported.".format(model))

    if model == 'estimate':
        # https://chandra.harvard.edu/edu/formal/age_snr/pencil_paper.pdf

        R_m = R*(ct._kpc_over_m_/1000.)  # [m]
        vol_m3 = (4./3.) * pi * R_m**3.  # [m^3]

        m_proton = 0.93827208816*ct._GeV_over_g_ * 1.e-3  # [kg]
        density = (m_proton*rho0)*1.e6  # [kg/m^3]
        mass = density*vol_m3  # [kg]

        E_sn_J = (E_sn*1.e51)*(1.e-7)  # [J]
        ke = 0.25*E_sn_J  # [J]

        vel = sqrt(2.*ke / mass)  # [m/s]
        age = (R_m/vel)/ct._year_over_s_  # [years]

        return age

    elif model in ['TM99-simple', 'TM99-0']:
        # Article: https://iopscience.iop.org/article/10.1086/313176
        # Erratum: https://iopscience.iop.org/article/10.1086/313385
        # Revisited: arXiv:2109.03612

        n0 = rho0/1.  # number density [cm^-3]

        # Characteristic scales: Table 2
        R_ch = 3.07 * M_ej**(1./3.) * n0**(-1./3.)  # [pc]
        t_ch = 423. * E_sn**(-1./2.) * M_ej**(5./6.) * n0**(-1./3.)  # [years]

        # Sedov-Taylor scales: Table 3 (rounding up, for various ejecta power-law indices)

        if model == 'TM99-simple':
            tstar_ST = 0.4
            Rstar_ST = 0.7
        elif model == 'TM99-0':
            tstar_ST = 0.4950
            Rstar_ST = 0.7276

        # Sedov-Taylor radius and age
        t_ST = tstar_ST*t_ch
        R_ST = Rstar_ST*R_ch

        if model == 'TM99-simple':
            t_bench = t_ST
            R_bench = R_ST
        elif model == 'TM99-0':
            t_bench = t_ch
            R_bench = R_ch

        # defining the evolution in the two phases
        def LogDelRb(t, r):
            """
            Function whose zeros we need to find in order to solve for the time [years] as a function of the radius [pc].
            """
            return log10(Rb_TM99(t, t_bench, R_bench, model)) - log10(r)

        R_arr, _ = tl.treat_as_arr(R)
        age = np.array([tl.zeros(LogDelRb, t_arr_default, r) for r in R_arr])
        age = np.squeeze(age)

        return age


#########################################
# Noise and signal

def bg_408_temp(l, b, size=None, average=False, verbose=True, load_on_the_fly=False):
    """
    Reads the background brightness temperature at 408 MHz at galactic coordinates l and b.

    Parameters
    ----------
    l : longitude [deg]
    b : latitude [deg]
    size : angular size [sr] of the region of interest (default: None)
    average : whether the brightness temperature is averaged over the size of the region of interest (default: False)
    verbose: control of warning output. Switch to False only if you know what you are doing.
    load_on_the_fly: flag to load the map on the fly. It's for debugging purpose. Switch to True only if you know what you are doing.
    """

    if load_on_the_fly:
        if verbose:
            print('You requested to load Haslem every time bg_408_temp is called. This is highly time-consuming. You should use this only for debugging purpose.')
        # initialize the haslem map
        try:
            path = os.path.dirname(os.path.abspath(__file__))+'/data/'
            path = os.path.join(path, 'lambda_haslam408_dsds.fits')
            global map_allsky_408
            map_allsky_408 = hp.read_map(path)
        except FileNotFoundError:
            raise Exception(
                'Haslam map (lambda_haslam408_dsds.fits) is not found.')

    healp = HEALPix(nside=512, order='ring', frame=Galactic())

    pos_coords = SkyCoord(frame="galactic", l=l, b=b, unit='deg')
    pos_pix = healp.skycoord_to_healpix(pos_coords)
    vec = hp.pix2vec(nside=512, ipix=pos_pix)

    if size is not None:

        size, is_scalar = tl.treat_as_arr(size)  # scalar --> array trick

        bg_T408 = []

        if average:
            for size_val in size:
                radial_angle = ct.solid_angle_to_angle(
                    size_val)/2.  # need to divide by 2 to get radius
                new_pos_pix = hp.query_disc(
                    nside=512, vec=vec, radius=radial_angle)
                bg_T408.append(np.average(map_allsky_408[new_pos_pix]))
        else:
            # this should be used only for debugging
            # i.e. manually switch average == False
            print('Warning: you are setting average flag to False')
            for size_val in size:
                new_pos_pix = pos_pix
                bg_T408.append(np.average(map_allsky_408[new_pos_pix]))

        if is_scalar:
            bg_T408 = np.squeeze(bg_T408)
        else:
            bg_T408 = np.array(bg_T408)

    else:
        new_pos_pix = pos_pix
        bg_T408 = np.average(map_allsky_408[new_pos_pix])

    # query the background temperature for the echo

    return bg_T408


def T_atm(nu):
    """Compute the atmospheric temperature [K]

    :param nu: Frequency [GHz]

    """

    nu, is_scalar = tl.treat_as_arr(nu)
    nu_low = nu[nu <= ct._nu_min_ska_mid_]
    nu_mid = nu[nu > ct._nu_min_ska_mid_]
    res_low = Tatm_low_fn(nu_low)
    res_mid = Tatm_mid_fn(nu_mid)
    res = np.concatenate((res_low, res_mid))
    if is_scalar:
        res = np.squeeze(res)
    return res


def Tatm_mid_fn(nu):
    """The atmospheric temperature for SKA1-Mid frequency

    :param nu: frequency [GHz] 

    """

    nu, is_scalar = tl.treat_as_arr(nu)
    fn = tl.interp_fn(Tsky_mid)
    Tsky_arr = fn(nu)
    # Braun 10th percentile
    Tgal_arr = 17.1 * (nu/0.408)**ct._MW_spectral_beta_
    res = Tsky_arr - ct._Tcmb_ - Tgal_arr
    if is_scalar:
        res = np.squeeze(res)
    return res


def Tatm_low_fn(nu):
    nu, is_scalar = tl.treat_as_arr(nu)
    fn = tl.interp_fn(Tsky_low)
    Tsky_arr = fn(nu)
    # Tgal_arr = 20. * (0.408/nu)**2.55
    Tgal_arr = 20. * (nu/0.408)**ct._MW_spectral_beta_  # Braun 2017
    res = Tsky_arr - ct._Tcmb_ - Tgal_arr
    if is_scalar:
        res = np.squeeze(res)
    return res


def T_sys_mid(nu):
    """System noise temperature [K] of a single dish for SKA-Mid. It's defined by Treceiver + Tspillover + Tsky, where Tsky = Tcmb + Tgal + Tatm. Note that this function is only used to compute eta from the table of Aeff/Tsys in Braun et al. It is not used in the noise computation. 

    :param nu: frequency [GHz]

    """
    nu, is_scalar = tl.treat_as_arr(nu)
    Tsky_arr = np.interp(nu, Tsky_mid[:, 0], Tsky_mid[:, 1])

    # combine MeerKAT with SKA dishes
    # if there's only SKA dish, use SKA dish
    # if there are both, use geometric mean
    Trec_arr = []
    for nui in nu:
        val1 = sk.Trec_mid_MeerKAT(nui)
        val2 = sk.Trec_mid_SKA(nui)
        if np.isinf(val1):
            val1 = val2
        val = np.sqrt(val1*val2)
        Trec_arr.append(val)
    Trec_arr = np.array(Trec_arr)
    res = ct._T_spill_mid_ + Tsky_arr + Trec_arr
    if is_scalar:
        res = np.squeeze(res)
    return res


def T_sys_low(nu):
    """System noise temperature [K] of a single station SKA-Low. It's defined by Treceiver + Tspillover + Tsky, where Tsky = Tcmb + Tgal + Tatm. Note that this function is only used to compute eta from the table of Aeff/Tsys in Braun et al. It is not used in the real noise computation. 
"""

    nu, is_scalar = tl.treat_as_arr(nu)

    Tsky_arr = np.interp(nu, Tsky_low[:, 0], Tsky_low[:, 1])

    Trec_arr = np.ones_like(nu) * 40.
    res = ct._T_spill_low_ + Tsky_arr + Trec_arr
    # res = Tsky_arr
    if is_scalar:
        res = np.squeeze(res)
    return res


def T_sys(nu, Tbg_at_408=27, beta=ct._MW_spectral_beta_, Tr=None):
    """
    The system temperature [K] seen by a single unit instantaneously. This function can be used for any experiments but if Tr is not specified, it will be automatically computed based on SKA.

    Parameters
    ----------
    nu: frequency [GHz]
    Tbg_at_408: the MW background at 408 MHz [K] (default: 27, for Cygnus A gegenschein position)
    beta: the index for the Milky (default: -2.55)
    Tr: the receiver's noise brightness temperature (default: 0.)
    """

    nu, is_scalar = tl.treat_as_arr(nu)
    if Tr is None:
        Tr = np.ones_like(nu)
        Tr[(nu > ct._nu_max_ska_low_)] = ct._Tr_ska_mid_
        Tr[(nu < ct._nu_max_ska_low_)] = ct._Tr_ska_low_

    Tcmb = ct._Tcmb_  # cmb brightness [K]
    Ta = T_atm(nu)  # atmospheric brightness [K]
    Tbg = Tbg_at_408 * (nu/0.408)**beta

    res = Tcmb + Ta + Tr + Tbg

    if is_scalar:
        res = np.squeeze(res)

    return res


def T_noise(T_sys, delnu, tobs, Omega_obs, Omega_res, nu, correlation_mode):
    """
    The noise rms temperature [K] of the array. 

    Based on Sec. 3 of arXiv:1811.08436.

    Parameters
    ----------
    T_sys: the system temperature [K]
    delnu: the bandwidth of the detector [GHz]
    tobs: the total observation time [hour]
    Omega_obs: the observation solid angle [sr]
    Omega_res: the resolution solid angle [sr]
    correlation_mode: the correlation mode, "single dish" or "interferometry".
    """

    # Photons have two polarizations:
    npol = 2.

    # We now look for all the factors that will affect T_sys:

    # 1. The number of independent measurements in time
    N_time = (delnu*1.e9)*(tobs*3600.)  # [GHz --> 1/s], [hours --> s]

    # 2. Any angular size-dependent factors:
    # Angle of observation
    theta_obs = ct.solid_angle_to_angle(Omega_obs)

    # Starting with a size factor of 1...
    size_factor = 1.
    if correlation_mode == "single dish":

        # Even though we always feed P_noise() with Omega_obs >= Omega_res
        # I'm adding an extra check here just in case I get sloppy in the
        # future. The noise increase due to larger angles is:
        try:
            factor = max(sqrt(Omega_obs/Omega_res), 1)
        except ValueError:
            factor = np.array([max(x, 1) for x in sqrt(Omega_obs/Omega_res)])

        size_factor *= factor

    elif correlation_mode == "interferometry":
        # NOTE: according to Eq. (3.9) of 1811.08436, there should be a factor of (Omega_pix / Omega) * sqrt(N_pix) where N_pix is the number of pixels of size Omega_pix (given by theta_sinth) in the observed primary beam angle Omega. It seems to me that this factor amounts to 1/sqrt(N_pix), but that seems to be the opposite of what they're saying. Also, we are not computing theta_sinth yet. I'm making this factor to be 1 provisionally.
        size_factor *= 1.

    else:
        raise Exception(
            "Correlation mode can only be 'single dish' or 'interferometry'.\
 You assigned %s" % correlation_mode)

    # 3. the number of independent array measurements

    nu, is_scalar = tl.treat_as_arr(nu)  # scalar --> array trick

    if is_scalar:
        # determine what exp we are looking at
        exper_mode = sk.SKA_exper_nu(nu)
        _, _, _, _, _, _, number_of_measurements = \
            sk.SKA_specs(nu,
                         exper_mode,
                         correlation_mode=correlation_mode,
                         theta_sig=theta_obs)

        # the total independent number of measurements
        N_meas = npol*np.squeeze(number_of_measurements)

    else:
        N_meas = []
        for i, nu_i in enumerate(nu):
            # determine what exp we are looking at
            exper_mode = sk.SKA_exper_nu(nu_i)
            _, _, _, _, _, _, number_of_measurements = \
                sk.SKA_specs(nu_i,
                             exper_mode,
                             correlation_mode=correlation_mode,
                             theta_sig=theta_obs[i])

            # the total independent number of measurements
            N_meas.append(npol*np.squeeze(number_of_measurements))
        N_meas = np.array(N_meas)

    res = T_sys*size_factor / sqrt(N_time) / sqrt(N_meas)

    return res


def P_noise(T_sys, delnu, tobs, Omega_obs, Omega_res, nu, correlation_mode):
    """
    The power of the noise [eV^2] of the array.

    Parameters
    ----------
    T_sys: the temperature of the noise [K]
    delnu: the bandwidth of the detector [GHz]
    tobs: the total observation time [hour]
    Omega_obs: the observation solid angle [sr]
    Omega_res: the resolution solid angle [sr]
    correlation_mode: the correlation mode, "single dish" or "interferometry".
    """

    # IMPORTANT: NOTA BENE: In converting the T_sys into energy (in Joules [J], after multiplying it by Boltzmann's constant) and then multiplying by the bandwidth frequency, the [Hz] units in the bandwidth play the role of the [1/s] in the SI units of power [W] = [J/s]. In other words [W] = [J*Hz]. Only after this one has to convert the SI power [W] into natural units [eV^2]. Converting the bandwidth frequency [Hz] into [eV] first *makes an error of 2pi*. The reason is that there's not energy associated with the bandwidth! The energy of the photons is *not* the bandwidth. The bandwidth is simply a rate time.

    # From temperature and bandwidth to power:
    # Tnu_to_P = (T_sys * ct._K_over_eV_) * (delnu * ct._GHz_over_eV_) # N.B.: WRONG!!!!!!!!
    Tnu_to_P = (T_sys*ct._k_B_)*(delnu*1.e9)  # [W] = [J/s]
    Tnu_to_P *= ct._J_over_eV_/ct._s_eV_  # [W] --> eV^2

    # number of independent measurements in time
    N_time = (delnu*1.e9)*(tobs*3600.)  # [GHz --> 1/s], [hours --> s]

    # house keeping
    theta_obs = ct.solid_angle_to_angle(Omega_obs)

    # First, we compute the noise of a single unit
    if correlation_mode == "single dish":
        # this is the noise of a single dish
        res = sqrt(2.)*Tnu_to_P/sqrt(N_time)
        # res = 2.*Tnu_to_P/sqrt(N_time) # NOTE: SEE BELOW!!!
        # NOTA BENE: previously, the prefactor above was 2. But according to Caputo's 1811.08436, there should be a sqrt(npol) = sqrt(2) factor in the denominator due to the npol=2 polarizations of the photon.

        # Even though we always feed P_noise() with Omega_obs >= Omega_res
        # I'm adding an extra check here just in case I get sloppy in the
        # future
        try:
            factor = max(sqrt(Omega_obs/Omega_res), 1)
        except ValueError:
            factor = np.array([max(x, 1) for x in sqrt(Omega_obs/Omega_res)])
        res *= factor
    elif correlation_mode == "interferometry":
        # this is the noise of a single baseline. c.f. 7-33 of Napier and Crane 1982
        res = sqrt(2.)*Tnu_to_P/sqrt(N_time)
    else:
        raise Exception(
            "Correlation mode can only be 'single dish' or 'interferometry'.\
 You assigned %s" % correlation_mode)

    # Second, we compute the noise of the array

    nu, is_scalar = tl.treat_as_arr(nu)  # scalar --> array trick

    if is_scalar:
        # determine what exp we are looking at
        exper_mode = sk.SKA_exper_nu(nu)
        _, _, _, _, _, number_of_dishes, number_of_measurements = \
            sk.SKA_specs(nu,
                         exper_mode,
                         correlation_mode=correlation_mode,
                         theta_sig=theta_obs)
        # convert to the noise of all dishes combined
        res *= number_of_dishes/np.sqrt(number_of_measurements)

        res = np.squeeze(res)

    else:
        for i, nu_i in enumerate(nu):
            # determine what exp we are looking at
            exper_mode = sk.SKA_exper_nu(nu_i)
            _, _, _, _, _, number_of_dishes, number_of_measurements = \
                sk.SKA_specs(nu_i,
                             exper_mode,
                             correlation_mode=correlation_mode,
                             theta_sig=theta_obs[i])
            res[i] *= number_of_dishes / np.sqrt(number_of_measurements)

    return res


def T_signal(Snu, A, eta=None, f_Delta=1.):
    """
    The signal temperature, also called the antenna temperature [K].

    Based on Sec. 3 of arXiv:1811.08436.

    Parameters
    ----------
    Snu: the average flux density (spectral irradiance) over the bandwidth [Jy]
    A: the area of the detector [m^2]
    eta: the detector efficiency (default: 0.8)
    f_Delta: the fraction of signal falling withing the bandwidth
    """

    res = f_Delta * (Snu * ct._Jy_over_SI_) * eta * A / 2. / ct._k_B_

    return res


def P_signal(S, A, eta=None, f_Delta=1.):
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


##################
# global variables
##################
# Defining local path
# --------------------
local_path = os.path.dirname(os.path.abspath(__file__))

# List of required parameters
# for the sources:
# ---------------------------

source_id = ['name', 'longitude', 'latitude', 'distance']
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
# -----------
source_input = {key: source_id+pars_always+value +
                pars_late for key, value in pars_early.items()}


# default time array [years]
# --------------------------
t_arr_default = np.logspace(-4, 11, 50001)
L_arr_default = np.logspace(-40, 60, 50001)

# initialize the haslem map
# -------------------------
try:
    path = os.path.dirname(os.path.abspath(__file__))+'/data/'
    path = os.path.join(path, 'lambda_haslam408_dsds.fits')
    map_allsky_408 = hp.read_map(path)
except FileNotFoundError:
    raise Exception(
        'Haslam map (lambda_haslam408_dsds.fits) is not found.')


# load the sky temperature (Tsky=Tcmb + Tgal + Tatm)
# from Braun et al. 2017
# and SKA-TEL-SKO-0000308_SKA1_System_Baseline_v2_DescriptionRev01-part-1-signed.
# Note that this Tsky is used for two things
# 1. for computing eta
# 2. for extracting Tatm
# -------------------------

Tsky_mid = np.loadtxt(local_path+"/data/Tsky_mid.csv", delimiter=',')
Tsky_low = np.loadtxt(local_path+"/data/Tsky_low.csv", delimiter=',')
