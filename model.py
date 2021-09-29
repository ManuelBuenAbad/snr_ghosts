"""module that has some simple models for testing
"""

import numpy as np
import constants as ct
import particle as pt
import astro as ap
import echo as ec
import routines as rt



def snr_routine(ma, ga,
                sn_remnant,
                gamma=None,
                nu_pivot=None,
                lightcurve_params=None,
                snu_echo_kwargs=None,
                data=None,
                output_all=False,
                verbose=0):
    """
    Computes the full echo routine for any axion parameters (ma, ga) and a particular Supernova Remnant object.
    
    Parameters
    ----------
    ma : axion mass [eV]
    ga : axion-photon coupling [GeV^-1]
    sn_remnant: the SNR object
    gamma : SNR adiabatic index (default: None)
    nu_pivot : pivot frequency [GHz] (default: None)
    lightcurve_params : parameters required to describe the SNR lightcurve: 4 out of the following 5: ['t_peak', 'L_peak', 't_trans', 't_age', 'L_today']. If sn_remnant.get_flux_density()/sn_remnant.snu_at_1GHz is available then passing only 3 parameters is allowed.
    snu_echo_kwargs : keyword args that go into snu_echo computation (default: None)
    data : experiment and detection data dictionary (default: None)
    output_all : False
    """
    
    # Preparing source_input dictionary from SNR object
    # a) coordinates
    l, b = sn_remnant.get_coord()  # [degree]
    # b) extracting source's size (solid angle)
    size = sn_remnant.get_size()  # [sr]
    if size == None:
        force_size = True # there is no known size: compute it
        if verbose > 1:
            print("SNR size=None, will be forced to compute it from first principles.")
    else:
        force_size = False # there is a known size: do not compute it
        if verbose > 1:
            print("SNR size=%.1e. Value will be respected." % size)
    
    # c) distance
    try:
        distance = sn_remnant.get_distance()
    except:
        return -1 # NECESSARY!
    # d) spectral index
    alpha = sn_remnant.get_spectral_index()
    if alpha is None:
        return -1 # NECESSARY!
    # e) adiabatic index
    if gamma is None:
        gamma = ap.gamma_from_alpha(alpha)
    # f) nu_pivot
    if nu_pivot is None:
        nu_pivot = 1. # [GHz]
    # g) lightcurve_params:
    if len(lightcurve_params) > 4:
        raise ValueError("'lightcure_params' has too many parameters. Please refer to astro.pars_required.")
    elif len(lightcurve_params) == 4:
        pass
    elif len(lightcurve_params) == 3:
        if 'L_today' in lightcurve_params.keys():
            raise KeyError("'lightcurve_params' has 3 parameters and 'L_today' is one of them. Either pass 4 parameters, or pass 3 without 'L_today', which will be read directly from 'sn_remnant'.")
        
        try:
            S0 = sn_remnant.get_flux_density() # [Jy] at 1 GHz
            area = 4.*np.pi*(distance*ct._kpc_over_cm_)**2. # [cm^2] area
            L_today = S0 * area * ct._Jy_over_cgs_irrad_ # [erg * s^-1 * Hz^-1]
            lightcurve_params['L_today'] = L_today
        
        except:
            raise ValueError("sn_remnant.get_flux_density() does not work: i.e. the 'sn_remnant' does not have a known flux density. Please fix this, or use a different suite of 'lightcurve_params'.")
        
    else:
        raise ValueError("'lightcurve_params' is not in the correct form. Please refer to astro.pars_required.")
    
    # source_input
    source_input = {
        'model': 'eff',
        'name': sn_remnant.name,
        'longitude': l,
        'latitude': b,
        'distance': distance,
        'size': size,
        'force_Omega_size_compute': force_size,
        'alpha': alpha,
        'nu_pivot': nu_pivot,
        'gamma': gamma
    }
    source_input.update(lightcurve_params)
    
    # compute signal
    ga_ref = 1.e-10 # [GeV]
    output = {}
    
    new_output = rt.full_routine(ma, ga, ga_ref, output,
                                 source_input=source_input,
                                 data=data,
                                 Snu_echo_kwargs=snu_echo_kwargs,
                                 beta=-2.55)
    
    signal_power = new_output['signal_power']
    noise_power = new_output['noise_power']
    signal_noise_ratio = new_output['S/N']

    if verbose > 0:
        print('signal power:{}'.format(signal_power))
        print('noise power:{}'.format(noise_power))
        print('s/n: {}\n'.format(signal_noise_ratio))
    
    if output_all:
        return signal_noise_ratio, new_output
    else:
        return signal_noise_ratio