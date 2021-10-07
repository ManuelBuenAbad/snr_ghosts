from __future__ import division

import numpy as np

from numpy import pi, sqrt, log, log10, power, exp
from scipy.integrate import trapz

import constants as ct
import astro as ap
import particle as pt
import echo as ec


# Default Quantities
def_D = 1.  # [kpc]
def_A = 4.*pi*(def_D*ct._kpc_over_cm_)**2.  # [cm^2] area
def_S0 = 1.  # [Jy]
# [erg * s^-1 * Hz^-1] spectral luminosity
def_L0 = (def_S0*ct._Jy_over_cgs_irrad_)*def_A

def_t0 = 1000.  # [years]
def_tt = 10.  # [years]
def_tpk = 100.  # [days]
def_tmin = def_tpk/365.  # [years]

def_alpha = 0.5
def_nu_pivot = 1.  # [GHz]
def_l = 0.  # [deg] galactic center longitude
def_b = 0.  # [deg] galactic center latitude

# Deefault Dictionaries
# source_input:
default_source_input = {'longitude': def_l,  # [deg]
                        'latitude': def_b,  # [deg]
                        'distance': def_D,  # [kpc]
                        'force_Omega_size_compute': True,  # compute source size
                        'force_Omega_disp_compute': True,  # compute DM dispersion angle
                        't_age': def_t0,  # [years]
                        'alpha': def_alpha,
                        'nu_pivot': def_nu_pivot,
                        # Sedov-Taylor analytic formula
                        'gamma': ap.gamma_from_alpha(def_alpha),
                        'model': 'eff',
                        'L_today': def_L0,  # [erg * s^-1 * Hz^-1]
                        't_trans': def_tt,  # [years]
                        't_peak': def_tpk  # [days]
                        }

# axion_input:


def ax_in(ma, ga):
    """
    Returns a dictionary with axion parameters.

    Parameters
    ----------
    ma : axion mass [eV]
    ga : axion-photon coupling [GeV^-1]
    """

    axion_input = {'ma': ma, 'ga': ga}
    return axion_input


# data:
default_data = {'deltaE_over_E': 1.e-3,
                'f_Delta': 0.721,
                'exper': 'SKA',
                'total_observing_time': 100.,
                'average': True,
                'DM_profile': 'NFW',
                'verbose': 0
                }


# Snu_echo_kwargs:

age_steps = abs(int(1000*(log10(def_t0) - log10(def_tpk/365.)) + 1))
max_steps = 1000001
default_Snu_echo_kwargs = {'tmin_default': None,
                           # for a fine enough array
                           'Nt': min(age_steps, max_steps),
                           'xmin': ct._au_over_kpc_,
                           'xmax_default': 100.,
                           'use_quad': False,
                           'lin_space': False,
                           # for a fine enough array
                           'Nint': min(age_steps, max_steps),
                           't_extra_old': 0.
                           }


# check default dictionaries
ec.check_source(default_source_input)
ec.check_data(default_data)
# Update source_input with:
# size
ec.Omega_size(default_source_input)
# Omega_dispersion
Omdisp_kwargs = {key: value
                 for key, value in default_Snu_echo_kwargs.items()
                 if key in ['tmin_default', 'xmax_default', 't_extra_old']}
ec.Omega_dispersion(default_source_input, default_data, **Omdisp_kwargs)


# Routine Functions

def fixed_axion_routine(ga_ref, output,
                        source_input=default_source_input,
                        data=default_data,
                        Snu_echo_kwargs=default_Snu_echo_kwargs
                        ):
    """
    Computes the full echo routine of both source and echo for a fixed reference axion-photon coupling ga [GeV^-1] and a fixed reference axion mass ma [eV] corresponding to nu_pivot [GHz].

    Parameters
    ----------
    ga_ref : reference axion-photon coupling [GeV^-1]
    output : dictionary in which output will be saved
    source_input : dictionary with source input parameters (default: default_source_input)
    data : dictionary with environmental, experimental, and observational data (default: default_data)
    Snu_echo_kwargs : Snu_echo() keyword arguments (default: default_Snu_echo_kwargs)
    """

    # reference axion properties:
    nu_ref = source_input['nu_pivot']  # [GHz] reference frequency
    ma_ref = pt.ma_from_nu(nu_ref)  # [eV] reference axion mass
    axion_input = ax_in(ma_ref, ga_ref)  # axion input

    # source spectral irradiance
    ec.Snu_source(ap.t_arr_default, nu_ref, source_input, output=output)
    # echo spectral irradiance
    ec.Snu_echo(source_input, axion_input, data,
                recycle_output=(True, output), **Snu_echo_kwargs)
    # echo signal
    ec.signal(source_input, axion_input, data,
              recycle_output=(True, output), **Snu_echo_kwargs)
    # noise
    Omdisp_kwargs = {key: value
                     for key, value in Snu_echo_kwargs.items()
                     if key in ['tmin_default', 'xmax_default', 't_extra_old']}

    ec.noise(source_input, axion_input, data,
             recycle_output=(True, output), **Omdisp_kwargs)

    # S/N ratio
    signal_power = output['signal_power']
    noise_power = output['noise_power']
    ec.sn_ratio(signal_power, noise_power, output=output, verbose=0)

    return ma_ref, ga_ref, output


def Snu_rescale_axion(ma, ga, ma_ref, ga_ref, source_input=default_source_input):
    """
    Computes the rescale factor for different axion parameters.

    Parameters
    ----------
    ma : axion mass [eV]
    ga : axion-photon coupling [GeV^-1]
    ma_ref : reference axion mass [eV]
    ga_ref : reference axion-photon coupling [GeV^-1]
    source_input : dictionary with source input parameters (default: default_source_input)
    """

    nu = pt.nu_from_ma(ma)  # [GHz] new frequency
    nu_ref = pt.nu_from_ma(ma_ref)  # [GHz] pivot frequency
    alpha = source_input['alpha']  # spectral index

    ax_pref = ec.axion_pref(ma, ga)/ec.axion_pref(ma_ref,
                                                  ga_ref)  # axion prefactors
    nu_fac = ap.nu_factor(nu, nu_ref, alpha)  # frequency-dependence

    return nu_fac * ax_pref


def SKA_rescaled_specs(ma, data=default_data):
    """
    Returns the SKA specs for the rescaled axion parameters.

    Parameters
    ----------
    ma : axion mass [eV]
    ma_ref : reference axion mass [eV]
    data : dictionary with environmental, experimental, and observational data (default: default_data)
    """

    nu = pt.nu_from_ma(ma)  # [GHz] new frequency
    nu = np.array(nu)  # converting into array for handling
    if nu.ndim == 0:
        nu = nu[None]
    
    exper = data['exper']  # requested experiment

    # computing the collecting area and the frequency sensitivity window of the experiment mode
    if exper == 'SKA':  # in case the range is frequency-dependent

        nu_flat = nu.flatten()

        area, window, Tr, Omega_res = [], [], [], []
        for nn in nu_flat:

            exper_mode = ap.SKA_exper_nu(nn)
            aa, ww, tr, od = ap.SKA_specs(nn, exper_mode, eta=ct._eta_ska_)

            area.append(aa)
            window.append(ww)
            Tr.append(tr)
            Omega_res.append(od)

        area, window, Tr, Omega_res = np.array(area), np.array(
            window), np.array(Tr), np.array(Omega_res)
        area, window, Tr, Omega_res = np.reshape(area, nu.shape), np.reshape(
            window, nu.shape), np.reshape(Tr, nu.shape), np.reshape(Omega_res, nu.shape)
        area, window, Tr, Omega_res = np.squeeze(area), np.squeeze(
            window), np.squeeze(Tr), np.squeeze(Omega_res)

    elif exper in ['SKA low', 'SKA mid']:  # in case the range was fixed by hand
        exper_mode = exper
        area, window, Tr, Omega_res = ap.SKA_specs(
            nu, exper_mode, eta=ct._eta_ska_)

    else:
        raise ValueError(
            "data['exper'] must be either 'SKA', 'SKA low', or 'SKA mid'. Please update accordingly.")

    return area, window, Tr, Omega_res


def rescale_routine(ma, ga, ma_ref, ga_ref, ref_dict,
                    source_input=default_source_input,
                    data=default_data,
                    Snu_echo_kwargs=default_Snu_echo_kwargs,
                    beta=-2.55):
    """
    Compute the rescaling echo routine for any axion parameters (ma, ga) based on pre-computed reference echo quantities. The idea is to make as few integrations as possible: only the single theory point with ma_ref and ga_ref is integrated and the rest signal from (ma, ga) can be computed through the rescaling of prefactor. 

    Parameters
    ----------
    ma : axion mass [eV]
    ga : axion-photon coupling [GeV^-1]
    ma_ref : reference axion mass [eV]
    ga_ref : reference axion-photon coupling [GeV^-1]
    ref_dict : dictionary with output results from reference values
    source_input : dictionary with source input parameters (default: default_source_input)
    data : dictionary with environmental, experimental, and observational data (default: default_data)
    Snu_echo_kwargs : Snu_echo() keyword arguments (default: default_Snu_echo_kwargs)
    beta: the index for the Milky (default: -2.55 from Ghosh paper)
    """

    # reference frequency:
    nu_ref = pt.nu_from_ma(ma_ref)
    # new frequency:
    nu = pt.nu_from_ma(ma)

    # sanity check:
    if (np.abs(nu_ref/ref_dict['signal_nu'] - 1.) > 1.e-10):
        raise ValueError("There seems to be an inconsistency in the value of nu_ref ={} GHz. It should ne the same as nu_pivot and ref_dict['signal_nu'] = {} GHz.".format(
            nu_ref, ref_dict['signal_nu']))

    # computing rescale factors
    nu_rescale = (nu/nu_ref)  # frequency rescale
    # for echo's spectral irradiance
    factors = Snu_rescale_axion(
        ma, ga, ma_ref, ga_ref, source_input=source_input)
    area, window, Tr, Omega_res = SKA_rescaled_specs(
        ma, data=data)  # SKA specs

    # echo's location
    l_echo = source_input['longitude'] + \
        180.  # [deg] galactic longitude of echo
    b_echo = -source_input['latitude']  # [deg] galactic latitude of echo

    # data
    f_Delta = data['f_Delta']  # fraction of signal that falls in bandwidth
    obs_time = data['total_observing_time']  # total observation time [hr]

    # rescaled output
    new_output = {}

    # signal:
    new_output['echo_Snu'] = ref_dict['echo_Snu']*factors
    new_output['signal_nu'] = ref_dict['signal_nu']*nu_rescale
    new_output['signal_delnu'] = ref_dict['signal_delnu']*nu_rescale
    new_output['signal_Omega'] = ref_dict['signal_Omega']
    new_output['signal_Snu'] = ref_dict['signal_Snu']*factors
    new_output['signal_S_echo'] = ref_dict['signal_S_echo']*factors*nu_rescale
    new_output['signal_power'] = ap.P_signal(
        new_output['signal_S_echo'], area, f_Delta=f_Delta)*window

    # noise:
    new_output['noise_nu'] = ref_dict['noise_nu']*nu_rescale
    new_output['noise_delnu'] = ref_dict['noise_delnu']*nu_rescale
    new_output['noise_Omega_res'] = Omega_res
    new_output['noise_Omega_obs'] = np.maximum.reduce(
        [new_output['signal_Omega']*np.ones_like(new_output['noise_Omega_res']), new_output['noise_Omega_res']])
    new_output['noise_T408'] = ap.bg_408_temp(
        l_echo, b_echo, size=new_output['noise_Omega_obs'], average=data['average'])
    new_output['noise_Tnu'] = ap.T_noise(
        nu, Tbg_at_408=new_output['noise_T408'], beta=beta, Tr=Tr)
    new_output['noise_power'] = ap.P_noise(
        new_output['noise_Tnu'], new_output['noise_delnu'], tobs=obs_time, Omega_obs=new_output['noise_Omega_obs'], Omega_res=new_output['noise_Omega_res'], nu=nu)

    # S/N:
    new_output['S/N'] = new_output['signal_power']/new_output['noise_power']

    # axion:
    new_output['ma'] = ma
    new_output['ga'] = ga

    # return final output
    return new_output


def full_routine(ma, ga, ga_ref, output,
                 source_input=default_source_input,
                 data=default_data,
                 Snu_echo_kwargs=default_Snu_echo_kwargs,
                 beta=-2.55):
    """
    Compute the full echo routine for any axion parameters (ma, ga).

    Parameters
    ----------
    ma : axion mass [eV]
    ga : axion-photon coupling [GeV^-1]
    ga_ref : reference axion-photon coupling [GeV^-1]
    output : dictionary in which useful output will be saved
    source_input : dictionary with source input parameters (default: default_source_input)
    data : dictionary with environmental, experimental, and observational data (default: default_data)
    Snu_echo_kwargs : Snu_echo() keyword arguments (default: default_Snu_echo_kwargs)
    beta: the index for the Milky (default: -2.55 from Ghosh paper)
    """

    # pre-computed reference quantities:
    reference_quantities = fixed_axion_routine(ga_ref, output,
                                               source_input=source_input,
                                               data=data,
                                               Snu_echo_kwargs=Snu_echo_kwargs)

    # reference axion parameters and echo quantities:
    ma_ref, ga_ref, ref_dict = reference_quantities

    # rescaling results from the reference quantities
    new_output = rescale_routine(ma, ga, ma_ref, ga_ref, ref_dict,
                                 source_input=source_input,
                                 data=data,
                                 Snu_echo_kwargs=Snu_echo_kwargs,
                                 beta=beta)

    # test:
    if data['verbose'] > 2:
        print("routines.py::source_input['size']: %s" % source_input['size'])

    # return final output
    return new_output
