"""
TODO: include optical depth?

This is a module to compute the basics of the stimulated decay
a la Boltzmann equations. The three main structures are:

        1. 'source_input',
        2. 'axion_input',
        3. 'data', and
        4. 'output'


1. The 'source_input' dict includes the following keys, defined in ap.source_input:
        'name' :        name of source; can be customized
        'longitude' :   galactic longitude of source [kpc]
        'latitude' :    galactic latitude of source [deg]
        'distance' :    distance to source [kpc]
        'alpha' :       spectral index of source
        'nu_pivot' :    pivot frequency of source's spectrum
        'gamma' :       adiabatic expansion index of source
        'size' :        solid angle size of source [sr]

        # computable keys and values:
        'Omega_dispersion' : solid angle due to DM velocity dispersion [sr]

        # modeling keys and values:
        'model' :       model of the time-evolution of source: 'eff'/'thy'.
            Each 'model'=='eff' ('thy') has 6 (8) parameters, and supports one of them being a dependent variable and 5 (7) required known parameters/quantities. For more information on these, see ap.pars_required.
            ... 'eff' (6 parameters):
                'L_peak' :  peak luminosity [erg*s^-1*Hz^-1]
                't_peak' :  peak time [days]
                't_trans' : free/adiabatic expansion transition time [years]
                'gamma' :   adiabatic expansion index of source
                't_age' :   age of source [years]
                'L_today' : today's luminosity [erg*s^-1*Hz^-1]
            ... 'thy' (8 parameters):
                'L_norm' :  luminosity scale near peak time [erg*s^-1*Hz^-1]
                'K2' :      opacity coefficient
                'beta' :    free expansion time power law
                'delta' :   opacity time power law
                't_trans' : free/adiabatic expansion transition time [years]
                'gamma' :   adiabatic expansion index of source
                't_age' :   age of source [years]
                'L_today' : luminosity today [erg*s^-1*Hz^-1]

        # Optional:
            'force_Omega_disp_compute' : whether the dispersion solid angle is computed (default: True)
            'use_free_expansion' : set the contribution of the free expansion part to be zero, and only account the signal from the adiabatic phase.

2. The 'axion_input' dict includes the following keys:
        'ma' :              the axion mass [eV]
        'ga' :              the axion-photon coupling [GeV^-1]


3. The 'data' dict includes the following keys:
        # environment
        'deltaE_over_E' :           the width of the line (default: 0.00145326)
        'DM_profile' :              the DM profile to be used (default: 'NFW')

        # experiment
        'f_Delta' :                 the fraction of flux density that falls in the optimal bandwidth (default: 0.83848)
        'exper' :                   the experiment ('SKA low'/'SKA mid', or simply 'SKA' for either, depending on the frequency; default: 'SKA')
        'total_observing_time' :    the total time of experimental observation (default: 100.)
        'average' :                 whether the background noise brightness temperature will be averaged over the angular size of the source (default: True)

        # optional
        'verbose' :                 verbosity (default: 0)


4. The 'output' dict includes the following keys:
        't-nu-Snu' :                    time-frequency-spectralirradiance of source [years-GHz-Jy]
        'echo_Snu' :                    spectral irradiance of the echo [Jy]
        'signal_nu' :                   signal frequency [GHz]
        'signal_delnu' :                signal line width [GHz]
        'signal_Omega' :                signal solid angle [sr]
        'signal_Snu' :                  spectral irradiance of echo in frequency array [Jy]
        'signal_S_echo' :               total irradiance of echo [ev^4]
        'signal_Tant' :                 signal antenna temperature [K]
        'signal_power' :                signal power of echo [eV^2]
        'noise_nu' :                    noise frequency [GHz]
        'noise_delnu' :                 noise line width [GHz]
        'noise_Omega_obs' :             noise observation solid angle [sr]
        'noise_T408' :                  noise background brightness temperature at 408 MHz [K]
        'noise_Tsys' :                  noise system temperature at frequency nu [K]
        'noise_Trms' :                  noise rms temperature at frequency nu [K]
        'noise_power' :                 noise power [eV^2]
        'S/N_power' :                   signal-to-noise ratio, computed with power ratio
        'S/N_temp' :                    signal-to-noise ratio, computed with temperature ratio
"""
from __future__ import division
import os
import numpy as np

from numpy import pi, sqrt, exp, power, log, log10
from scipy.integrate import quad, trapz
from scipy.interpolate import interp1d
from scipy.special import erf, lambertw

import tools as tl
import constants as ct
import particle as pt
import ska as sk
import astro as ap


# a default array of radio frequencies
nu_array_default = np.logspace(-2, 2, 5001)  # [GHz]

# example background brightness temperatures at 408 MHz
Tbg_408_avg = ap.bg_408_temp(0., 0., size=(
    4*pi), average=True)  # total sky average
# antipodal point to galactic center
Tbg_408_antipodal = ap.bg_408_temp(180., 0., average=False)


#########################################
# Check functions


def check_source(source_input, custom_name='custom', verbose=False):
    """
    Checks if the 'source_input' dictionary for a source is in the necessary format.

    Parameters
    ----------
    source_input : dictionary with source input parameters
    custom_name : a custom name for the source (default: 'custom')
    verbose : verbosity (default: 0)
    """

    if not 'name' in source_input.keys():
        # update source 'source_input' dictionary to contain some name
        source_input['name'] = custom_name

    if (not 'force_Omega_disp_compute' in source_input.keys()) or (not 'Omega_dispersion' in source_input.keys()):
        # DM's dispersion size hasn't been passed, and there is no explicit request on whether to compute it
        source_input['force_Omega_disp_compute'] = True

    if not 'use_free_expansion' in source_input.keys():
        # haven't passed whether the free expansion phase will be included or not; default will be yes
        source_input['use_free_expansion'] = True

    has_all_source_id = set(ap.source_id).issubset(set(source_input.keys()))
    if not has_all_source_id:
        raise KeyError(
            "'source_input' does not contain all the keys in source_id={}. Please update.".format(ap.source_id))

    has_all_model_always = set(ap.pars_always).issubset(
        set(source_input.keys()))
    if not has_all_model_always:
        raise KeyError(
            "'source_input' does not contain all the keys in pars_always={}. Please update.".format(ap.pars_always))

    is_there_model = ('model' in source_input.keys())
    if not is_there_model:
        raise KeyError(
            "'source_input' does not have a model. Please update and make sure source_input['model'] is either 'eff' or 'thy'.")
    else:
        model = source_input['model']
        if model in ['eff', 'thy']:

            # list of lightcurve parameters that are known (i.e. present in kwargs)
            known = []
            # list of lightcurve parameters that are to be deduced (i.e. are missing in kwargs)
            unknown = []
            for par in ap.pars_lightcurve[model]:
                if par in source_input.keys():
                    known.append(par)
                else:
                    unknown.append(par)

            if len(unknown) > 1:  # too many missing parameters
                raise ValueError(
                    "unknown={} is too large. Please update and include more parameters in source_input.".format(unknown))

            elif len(unknown) == 1:  # one unknown parameter

                # extracting the only parameter to be deduced
                unknown_par = unknown[0]
                if verbose:
                    print("Unknown parmeter: {}".format(unknown_par))

                try:
                    # the parameters required in source_input
                    required = ap.pars_required[(model, unknown_par)]
                except:
                    raise KeyError("Currently the code does not support having {} as an unknown parameter, only {}. Please update.".format(
                        unknown_par, ap.pars_required.keys()))

                if not set(required).issubset(set(known)):
                    raise ValueError(
                        "known={} is too short. It should be a subset of the required parameters, which are {}. Please include more parameters in kwargs.".format(known, required))

                # computing unknown parameters, at nu_pivot:
                local_source = {key: value for key,
                                value in source_input.items()}
                if model == 'thy':
                    # Weiler's frequency-dependent opacity correction factor
                    tau_factor = (local_source['nu_pivot']/5.)**-2.1
                    local_source['tau_factor'] = tau_factor

                # computing the unknown parameter:
                _, pars_out = ap.L_source(1., output_pars=True, **local_source)

                # updating source_input with new known parameter
                source_input.update({unknown_par: pars_out[unknown_par]})

            else:  # no unknown parameters!
                if verbose:
                    print("No unknown parameters.")
                pass

        else:
            raise KeyError(
                "source_input['model']={} is neither 'eff' nor 'thy', which are the only two options. Please update.".format(model))

    return


def check_axion(axion_input):
    """
    Checks if the 'axion_input' dictionary is in the necessary format.

    Parameters
    ----------
    axion_input : dictionary with axion input parameters
    """

    necessary_keys = ['ma', 'ga']
    has_all_params = all([key in axion_input.keys() for key in necessary_keys])
    if not has_all_params:
        raise KeyError(
            "'axion_input' dictionary does not contain all the keys: {}. Please update.".format(necessary_keys))

    return


def check_data(data, deltaE_over_E=ct._deltaE_over_E_, f_Delta=ct._f_Delta_, exper='SKA', total_observing_time=100., average=True, DM_profile='NFW', correlation_mode='single dish', verbose=0):
    """
    Checks if the 'data' dictionary of a source is in the necessary format.

    Parameters
    ----------
    data : dictionary with environmental, experimental, and observational data
    deltaE_over_E : width of the signal line (default: 0.00145326)
    f_Delta : fraction of flux density that falls in the optimal bandwidth (default: 0.83848)
    exper : experiment under consideration (default: 'SKA')
    total_observing_time : total time of observation [hours] (default: 100.)
    average : whether the background noise brightness temperature will be averaged over the angular size of the source (default: True)
    DM_profile : the DM density profile (default: 'NFW')
    correlation_mode: the correlation mode of the telescope, either "single dish" or "interferometry".
    verbose : verbosity (default: 0)
    """

    # I'm not setting default to expose all old calls through Exceptions.
    # TODO: We can enable the following after all ws/scripts are updated

    # if not 'correlation_mode' in data.keys():
    #     # running mode, "single dish" or "interferometry"
    #     data['correlation_mode'] = correlation_mode

    if not 'deltaE_over_E' in data.keys():
        # update 'data' with default value for deltaE_over_E
        data['deltaE_over_E'] = deltaE_over_E

    if not 'DM_profile' in data.keys():
        # update 'data' with default value for 'DM_profile'
        data['DM_profile'] = DM_profile

    if not 'f_Delta' in data.keys():
        # update 'data' with default value of f_Delta
        data['f_Delta'] = f_Delta

    if not 'exper' in data.keys():
        data['exper'] = exper  # update 'data' with default value for experiment

    if not 'total_observing_time' in data.keys():
        # update 'data' with default value for deltaE_over_E
        data['total_observing_time'] = total_observing_time

    if not 'average' in data.keys():
        # update 'data' with default value for deltaE_over_E
        data['average'] = average

    necessary_keys = ['deltaE_over_E', 'DM_profile', 'f_Delta',
                      'exper', 'total_observing_time', 'average']
    has_all_params = all([key in data.keys() for key in necessary_keys])
    if not has_all_params:
        raise KeyError(
            "'data' dictionary does not contain all the keys: {}. Please update.".format(necessary_keys))

    try:
        data['verbose']
    except:
        data['verbose'] = verbose

    return


#########################################
# Source and echo numerical computations

def Omega_size(source_input, verbose=0):
    """
    Computes the solid angle [sr] of the source, based on its other properties. Can be called only after only after check_source()

    Parameters
    ----------
    source_input : dictionary with source input parameters
    verbose: verbosity (default: 0)
    """

    try:
        Omega_size = source_input['size']
        if Omega_size is None:
            raise KeyError

    except KeyError:
        # catches either 'size' is not in source_input.key(), or it's there but None
        check_source(source_input)
        # v_free = ct._v_hom_ # speed of the free expansion [c]
        # speed of the free expansion, according to TM99 [c]
        v_free = ct._v_TM99_
        t_trans = source_input['t_trans']
        t_end = source_input['t_age']
        distance = source_input['distance']

        if verbose > 0:
            print('v_free: %.1e [c]' % v_free)
            print('t_trans: %.1e [c]' % t_trans)
            print('t_end: %.1e [c]' % t_end)
            print('distance: %.1e kpc' % distance)
            print('R_trans: %.1e kpc' % R_trans)

        if t_trans >= t_end:
            # radius at the end of the free expansion phase [kpc]
            R_end = (v_free*t_end)/ct._kpc_over_lightyear_
            if verbose > 1:
                print('R ~ t during free expansion')
        else:
            # radius at the end of the free expansion phase [kpc]
            R_trans = (v_free*t_trans)/ct._kpc_over_lightyear_
            R_end = R_trans * (t_end/t_trans)**(2./5.)
            if verbose > 1:
                print('R ~ t during free expansion')
                print('R ~ t^(2/5) during adiabatic expansion')

        if verbose > 0:
            print('R_end: %.1e kpc' % R_end)

        # angle subtended (twice angular radius) [rad]
        theta_source = 2.*(R_end/distance)
        Omega_size = ct.angle_to_solid_angle(theta_source)  # [sr]

        if verbose > 0:
            print('theta_source: %.1e rad' % theta_source)
            print('size: %.1e sr' % Omega_size)

    return Omega_size


def Omega_dispersion(source_input, data, tmin_default=None, xmax_default=100., t_extra_old=0., verbose=0):
    """
    Computes the solid angle [sr] from which the echo signal is emitted due to the dark matter velocity dispersion.

    Parameters
    ----------
    source_input : dictionary with source input parameters
    data : dictionary with environmental, experimental, and observational data
    tmin_default : the default cutoff minimum time [years] (i.e. the youngest age) of the SNR we will consider (default: None)
    xmax_default : the default maximum value of the integration variable x [kpc] (default: 100.)
    t_extra_old : extra time [years] added to the SNR source to make it older; they do not contribute to the lightcurve but merely displace the limits of the l.o.s. integral (default: 0.)
    verbose: verbosity (default: 0)
    """

    try:
        force_Omega_disp_compute = source_input['force_Omega_disp_compute']
    except KeyError:  # should not be necessary, as this function is always called after check_source()
        force_Omega_disp_compute = True

    if (not ('Omega_dispersion' in source_input.keys())) or (force_Omega_disp_compute is True):

        # checking if the dictionaries are in the correct format
        check_source(source_input)
        check_data(data)

        # duration of free+adiabatic phases [years]
        t_age = source_input['t_age']
        distance = source_input['distance']  # distance to SNR source [kpc]
        sigma_v = ct._sigma_v_  # velocity dispersion
        if verbose > 0:
            print('sigma_v: %.1e' % sigma_v)

        # calculating the cutoff minimum time (i.e. the youngest age under consideration)
        if tmin_default == None:
            try:
                tmin = source_input['t_peak']/365.  # [years]
            except:
                tmin = 1./365.  # 1 day [years]
        else:
            tmin = tmin_default

        # correcting tmin in non-sensical case
        if tmin > t_age:
            tmin = t_age/2.

        # l.o.s. offset
        x_offset = t_extra_old/(2.*ct._kpc_over_lightyear_)
        if verbose > 0:
            print('x_offset: %.1e' % x_offset)

        # the upper limit: xmax
        # the location of the wave front
        xmax_tmp = (t_age - tmin)/(2.*ct._kpc_over_lightyear_)
        xmax_tmp += x_offset  # adding offset
        # truncate it with xmax_default
        x_wavefront = min([xmax_default, xmax_tmp])

        # multiply by 2 to get full arc subtended
        theta_sig = 2.*((x_wavefront+distance)/distance)*sigma_v
        if verbose > 0:
            print('theta sig: %.1e' % theta_sig)
        Omega_sig = ct.angle_to_solid_angle(theta_sig)
        if verbose > 0:
            print('Omega sig: %.1e\n' % Omega_sig)
        source_input['Omega_dispersion'] = Omega_sig

        # compute the aberration angel due to motion of the source here
        # first, motion of the source
        ds = sigma_v * x_wavefront
        # then aberration angle
        # this is the entire angle subtended by the smearing of the signal
        theta_ab = ds / distance
        Omega_ab = ct.angle_to_solid_angle(theta_ab)
        source_input['Omega_aberration'] = Omega_ab

    return Omega_sig, Omega_ab


def axion_pref(ma, ga):
    """
    Axion prefactor [eV^-3] for echo.

    Parameters
    ----------
    ma : axion mass [eV]
    ga : axion-photon coupling [GeV^-1]
    """

    E = ma/2.
    prefactor = pi**2 / E**3 * pt.Gamma(ma, ga) / E

    return prefactor


def pref(ma, ga, deltaE_over_E=ct._deltaE_over_E_):
    """
    Axion and dark matter prefactor [1/kpc] for echo.

    Parameters
    ----------
    ma : axion mass [eV]
    ga : axion-photon coupling [GeV^-1]
    deltaE_over_E : the line width (default: 0.00145326)
    """

    rho_at_Earth = ct._rho_local_DM_ / ct._eV_over_GeV_ / ct._cm_eV_**3
    prefactor = axion_pref(ma, ga) / deltaE_over_E

    return ct._kpc_eV_ * prefactor * rho_at_Earth


def Snu_source(t, nu, source_input, output=None):
    """
    Returns the spectral irradiance (flux density) [Jy] of the SNR. Saves to output.

    Parameters
    ----------
    t : time after explosion [years]
    nu : frequency [GHz]
    source_input : dictionary with source input parameters
    output : output dictionary (default: None)
    """

    # checking if the source_input is in the correct format
    check_source(source_input)

    # source's size is not used here

    alpha = source_input['alpha']  # spectral index
    nu_pivot = source_input['nu_pivot']  # [GHz] pivot/reference frequency
    distance = source_input['distance']  # [kpc] distance to SNR
    model = source_input['model']  # SNR light curve model

    # creating local copy of the source input parameters (to be edited)
    # dictionary of parameters necessary to compute the luminosity of the source
    local_source = {key: value for key, value in source_input.items()}

    # analyzing properties of t & nu arguments:
    # a more sophisticated variant of the scalar --> array trick,
    # necessary because we need to treat both t & nu as orthogonal arrays
    # TODO: simply use a variant of tl.treat_as_arr() to turn t & nu into grids

    try:
        if t.ndim == 0:
            t_is_scalar = True
            t = t[None]
        else:
            t_is_scalar = False
    except AttributeError:
        t_is_scalar = True

    try:
        if nu.ndim == 0:
            nu_is_scalar = True
            nu = nu[None]
        else:
            nu_is_scalar = False
    except AttributeError:
        nu_is_scalar = True

    # computing the source's luminosity:
    if not nu_is_scalar:  # nu is an array

        Lum = []
        for nn in nu:

            if model == 'thy':
                # Weiler's frequency-dependent opacity correction factor
                tau_factor = (nn/5.)**-2.1
                local_source['tau_factor'] = tau_factor

            # frequency-dependent correction factor
            factor = ap.nu_factor(nn, nu_pivot, alpha)
            # luminosity [erg * s^-1 * Hz^-1] w/ frequency-dependent factor
            Lnu = factor*ap.L_source(t, output_pars=False, **local_source)
            Lum.append(Lnu)
        Lum = np.array(Lum)

    elif not t_is_scalar:  # nu is a scalar but t is an array

        if model == 'thy':
            # Weiler's frequency-dependent opacity correction factor
            tau_factor = (nu/5.)**-2.1
            local_source['tau_factor'] = tau_factor

        # frequency-dependent correction factor
        factor = ap.nu_factor(nu, nu_pivot, alpha)
        # luminosity [erg * s^-1 * Hz^-1] w/ frequency-dependent factor
        Lum = np.squeeze(
            factor*ap.L_source(t, output_pars=False, **local_source))

    else:  # nu and t are scalars

        if model == 'thy':
            # Weiler's frequency-dependent opacity correction factor
            tau_factor = (nu/5.)**-2.1
            local_source['tau_factor'] = tau_factor

        # frequency-dependent correction factor
        factor = ap.nu_factor(nu, nu_pivot, alpha)
        # luminosity [erg * s^-1 * Hz^-1] w/ frequency-dependent factor
        Lum = factor*ap.L_source(t, output_pars=False, **local_source)

    # converting spectral luminosity into spectral irradiance (flux density):
    Snu = ap.irrad(distance, Lum)  # spectral irradiance [Jy]

    # saving computation to output
    if type(output) == dict:
        output['source_t-nu-Snu'] = t, nu, Snu

    return Snu


def dSnu_echo(x, theta,  # position of echo differential element
              tobs,  # time of observation
              axion_prefactor,  # axion-dependent prefactor
              Snu_fn,  # source flux density function
              rho, delE_over_E,  # environment
              verbose=False,
              DM_profile="NFW"):
    """
    The differential contribution to the echo spectral irradiance (flux density) along the l.o.s. [Jy/kpc]

    Parameters
    ----------
    x : the position of the differential contribution along the l.o.s. [kpc]
    theta : the angle of the differential contribution w.r.t. the galaxy center [degree]
    tobs : the time at which the source is being observed [years]
    axion_prefactor : a numerical prefactor that depends on the axion properties [eV^-3], given by an.axion_pref()
    Snu_fn : the source's flux density [Jy], as a function of time, already evaluated at the right frequency
    rho : the density function from the galaxy center [GeV/cm**3]
    delE_over_E : the line width
    verbose : verbosity (default: False)
    DM_profile: the profile of DM in MW, can be 'NFW' or 'Burkert' (default: 'NFW')
    """

    # [years] time at which echo differential element ought to be evaluated
    t = tobs - 2.*x*ct._kpc_over_lightyear_
    # [kpc] distance of echo differential element to galactic center
    dist_from_gal_center = ap.r_to_gal(x, th=theta)

    # [eV^4] dark matter density at location of echo differential element
    rho_at_x = rho(dist_from_gal_center, DM_profile) / \
        ct._eV_over_GeV_/ct._cm_eV_**3

    try:
        if t.ndim == 0:
            t = t[None]

        # evaluating with inverted order in t (which from above we can see its in decreasing order). Will re-order later.
        Snu_eval = Snu_fn(t[::-1])
        # squeezing; then returning to original order; this is a feature of interpolation functions for the case when their arguments are in decreasing order instead of increasing.s
        Snu_eval = np.squeeze(Snu_eval)[::-1]

    except:
        Snu_eval = Snu_fn(t)

    if verbose:
        print('t\t%s' % t)
        print('rho_at_ax\t%s' % rho_at_x)
        print('Snu(t)\t%s' % (Snu_fn(t)))

    res = (axion_prefactor/delE_over_E) * rho_at_x * Snu_eval  # [Jy*eV]
    # convert to [Jy/kpc]
    res *= ct._kpc_eV_

    return res


def Snu_echo(source_input, axion_input, data,
             recycle_output=(False, None),
             tmin_default=None,
             Nt=1001,
             xmin=ct._au_over_kpc_,
             xmax_default=100.,
             use_quad=False,
             lin_space=False,
             Nint=50001,
             t_extra_old=0.):
    """
    Computes the spectral irradiance (flux density) [Jy] of the echo. Saves to output.

    Parameters
    ----------
    source_input : dictionary with source input parameters
    axion_input : dictionary with axion parameters
    data : dictionary with environmental, experimental, and observational data
    recycle_output : whether we recycle a previous computation; and the location where it is stored (default: (False, None))
    tmin_default : the default cutoff minimum time [years] (i.e. the youngest age) of the SNR we will consider (default: None)
    Nt : number of time points over which we interpolate the source's Snu
    xmin : the closest integration distance [kpc] we will consider (default: 1 AU)
    xmax_default : the default maximum value of the integration variable x [kpc] (default: 100.)
    use_quad : whether the integration routine used is quad; if False use trapz (default: False)
    lin_space : whether a linear space array is used for trapz routine; if False use logspace (Default: False).
    Nint : number of integration steps for integration routine (default: 50001).
    t_extra_old : extra time [years] added to the SNR source to make it older; they do not contribute to the lightcurve but merely displace the limits of the l.o.s. integral (default: 0.)
    """

    # checking if the dictionaries are in the correct format
    check_source(source_input)
    check_axion(axion_input)
    check_data(data)

    # # Update source_input with:
    # size
    # source's size is not used here
    # Omega_dispersion
    Omega_dispersion(source_input, data,
                     tmin_default=tmin_default,
                     xmax_default=xmax_default,
                     t_extra_old=t_extra_old)

    # source location parameters:
    l_source = source_input['longitude']  # [deg] galactic longitude of source
    b_source = source_input['latitude']  # [deg] galactic latitude of source
    # [rad] angle of source to galactic center
    theta_source = ap.theta_gal_ctr(l_source, b_source, output_radians=True)

    # echo location parameters:
    l_echo = l_source + 180.  # [deg] galactic longitude of echo
    b_echo = -b_source  # [deg] galactic latitude of echo
    theta_echo = pi-theta_source  # [rad] angle of echo to galactic center

    # time at which we perform the observations: the age of the SNR
    if 't_age' in source_input.keys():  # already among parameters
        t_age = source_input['t_age']

    else:  # compute from t_trans
        if (source_input['model'] != 'eff'):
            raise KeyError(
                "'t_age' is not among the keys of source_input, and yet the model in source_input['model']={} is not 'eff'. This should not have happened.")

        t_age = ap.tage_compute(source_input['L_peak'],
                                source_input['t_peak'],
                                source_input['t_trans'],
                                source_input['L_today'],
                                source_input['gamma'])

    tage_extended = t_age + t_extra_old  # [years] extended age of SNR

    # calculating the cutoff minimum time (i.e. the youngest age under consideration)
    if tmin_default == None:
        try:
            tmin = source_input['t_peak']/365.  # [years]
        except:
            tmin = 1./365.  # 1 day [years]
    else:
        tmin = tmin_default

    # correcting tmin in non-sensical case
    if tmin > t_age:
        tmin = t_age/2.

    # axion parameters
    ma = axion_input['ma']
    ga = axion_input['ga']
    axion_prefactor = axion_pref(ma, ga)

    # data parameters
    deltaE_over_E = data['deltaE_over_E']

    # checking for recycling:
    recycle, output = recycle_output

    if (not recycle) or (type(output) != dict):  # computing from scratch

        nu = pt.nu_from_ma(ma)
        # [years] array of times at which we compute the source's Snu
        tArr = np.logspace(log10(tmin), log10(t_age), Nt)
        SnuArr = Snu_source(tArr, nu, source_input, output=None)  # [Jy]

    # recycling previously-computed sources
    elif recycle and ('source_t-nu-Snu' in output.keys()):
        # Snu for source: t, nu, Snu
        tArr, nu, SnuArr = output['source_t-nu-Snu']
        nu_ma = pt.nu_from_ma(ma)
        diff = np.abs(nu_ma/nu - 1.)

        if (diff > 1.e-10):
            raise ValueError(
                "Upon recycling the previously computed output['source_t-nu-Snu'], we found it to be inconsistent with the requested axion mass ma: the frequencies do not match: nu = {} GHz != {} GHz = ma/2. Please re-compute Snu_source with the correct frequency; or simply demand recycle=False.".format(nu, nu_ma))

    else:
        raise ValueError(
            "'recycle_output' is not of the form (bool, dict). Please update.")

    tS = np.vstack((tArr, SnuArr)).T
    Snu_source_fn = tl.interp_fn(tS)
    del tS

    # DM_profile is part of 'data' structure, or fall back to NFW
    try:
        DM_profile = data['DM_profile']
    except KeyError:  # should not be necessary, as this function is passed after check_data()
        DM_profile = "NFW"

    # defining the integrand:
    def integrand(x):
        dSnu = dSnu_echo(x=x,  # NOTE: offset position differential element of echo
                         theta=theta_echo,
                         tobs=tage_extended,  # NOTE: we have offset tobs to tage_extended = t_age + t_extra_old
                         axion_prefactor=axion_prefactor,
                         Snu_fn=Snu_source_fn,
                         rho=ap.rho_MW,
                         delE_over_E=deltaE_over_E,
                         DM_profile=DM_profile)
        return dSnu

    # defining the limits of the integral: xmin & xmax. NOTE: we need to also offset the l.o.s.
    # the offset in the l.o.s. from the extra SNR age
    x_offset = t_extra_old/(2.*ct._kpc_over_lightyear_)

    # the lower limit: xmin
    xmin += x_offset  # adding offset

    # the upper limit: xmax
    # the location of the wave front
    xmax_tmp = (t_age - tmin)/(2.*ct._kpc_over_lightyear_)
    xmax_tmp += x_offset  # adding offset
    xmax = min([xmax_default, xmax_tmp])  # truncate it with xmax_default

    if use_quad:
        res, error = quad(integrand, xmin, xmax)
        if data['verbose'] > 1:
            print('Snu_echo = {:.3}, error={:.3}\n'.format(res, error))

    else:
        if lin_space:
            x_arr = np.linspace(xmin, xmax, Nint)
        else:
            # we will be using logspace. It is better to define the log array in time (the natural variable of the lightcurve) and then convert into a log array in x-space to avoid clustering x-points at low x (large t)
            # in order to do this we need both the lowest and the highest t points to be sampled in the lightcurve.
            if xmax < xmax_default:
                t_lo = tmin  # lowest lightcurve time probed by l.o.s.
            else:
                # lowest lightcurve time probed by l.o.s.
                t_lo = tage_extended - xmax*2.*ct._kpc_over_lightyear_

            # highest lightcurve time probed by l.o.s.
            t_hi = t_age - (xmin - x_offset)*(2.*ct._kpc_over_lightyear_)

            # array of times, without the offset
            t_arr = np.logspace(log10(t_lo), log10(t_hi), Nint)
            # corresponding array of l.o.s. positions; inverted to go from lowest to largest, without the offset
            x_arr = ((t_age - t_arr)/(2.*ct._kpc_over_lightyear_))[::-1]
            x_arr += x_offset  # putting back the offset

        int_arr = integrand(x_arr)

        res = trapz(int_arr, x_arr)

    if recycle and (type(output) == dict):
        output['echo_Snu'] = res

    return res


# simple analytic formulas

def echo_ad_fn(gamma, frac_tpk, frac_tt):
    """
    Analytic dimensionless echo from adiabatic era, assuming constant DM density. NOTE: the frequency dependence prefactor, usually (nu/nu_pivot)^-alpha, is factorized out.

    Parameters
    ----------
    gamma : adiabatic index
    frac_tpk : ratio of peak day to SNR age
    frac_tt : ratio of transition time to SNR age
    """

    factors = exp(1.5)/2./(1.-gamma)
    first = exp(-1.5*frac_tpk/frac_tt) * (frac_tt/frac_tpk)**-1.5
    second = ((1./frac_tt)**-gamma - frac_tt)

    return factors*first*second


def echo_free_fn(frac_tpk, frac_tt):
    """
    Analytic dimensionless echo from free expansion era, assuming constant DM density. NOTE: the frequency dependence prefactor, usually (nu/nu_pivot)^-alpha, is factorized out.

    Parameters
    ----------
    frac_tpk : ratio of peak day to SNR age
    frac_tt : ratio of transition time to SNR age
    """

    factors = exp(1.5) * sqrt(pi/6.)
    first = frac_tpk
    second = erf(sqrt(1.5)) - erf(sqrt(1.5) * sqrt(frac_tpk/frac_tt))

    return factors*first*second


def echo_tot_fn(gamma, frac_tpk, frac_tt):
    """
    Analytic dimensionless total echo, assuming constant DM density. NOTE: frequency dependence prefactor, usually (nu/nu_pivot)^-alpha, is factorized out.

    Parameters
    ----------
    gamma : adiabatic index
    frac_tpk : ratio of peak day to SNR age
    frac_tt : ratio of transition time to SNR age
    """

    ad = echo_ad_fn(gamma, frac_tpk, frac_tt)
    free = echo_free_fn(frac_tpk, frac_tt)

    return ad + free


def echo_an(ma, ga, Lpk, dist, t_age, gamma, tpk, tt, deltaE_over_E=ct._deltaE_over_E_):
    """
    Analytic formula for the echo spectral irradiance [Jy], with transition time as input, and assuming constant DM density. NOTE: the frequency dependence prefactor, usually (nu/nu_pivot)^-alpha, is factorized out.

    Parameters
    ----------
    ma : axion mass [eV]
    ga : axion-photon coupling [GeV^-1]
    Lpk : peak spectral luminosity [erg * s^-1 * Hz^-1]
    dist : distance to source [kpc]
    t_age : age of SNR [years]
    gamma : adiabatic index
    tpk : peak time [days]
    tt : free-adiabatic phases transition time [years]
    deltaE_over_E : the line width (default: 0.00145326)
    """

    frac_tpk = (tpk/365.)/t_age
    frac_tt = tt/t_age

    dimless_int = echo_tot_fn(gamma, frac_tpk, frac_tt)

    area = 4.*pi*(dist*ct._kpc_over_cm_)**2.  # [cm^2]
    Spk = Lpk/area/ct._Jy_over_cgs_irrad_  # [Jy]

    A_fac = pref(ma, ga, deltaE_over_E=deltaE_over_E)  # [1/kpc]
    x_age = t_age/ct._kpc_over_lightyear_  # [kpc]

    Se = A_fac*Spk*x_age*dimless_int

    return Se


def echo_an_sup(ma, ga, Lpk, dist, S0, t_age, gamma, tpk, deltaE_over_E=ct._deltaE_over_E_):
    """
    Analytic formula for the echo spectral irradiance [Jy], with the suppression as input, and assuming constant DM density. NOTE: the frequency dependence prefactor, usually (nu/nu_pivot)^-alpha, is factorized out.

    Parameters
    ----------
    ma : axion mass [eV]
    ga : axion-photon coupling [GeV^-1]
    Lpk : peak spectral luminosity [erg * s^-1 * Hz^-1]
    dist : distance to source [kpc]
    S0 : SNR spectral irradiance (flux density) today [Jy]
    t_age : age of SNR [years]
    gamma : adiabatic index
    tpk : peak time [days]
    deltaE_over_E : the line width (default: 0.00145326)
    """

    area = 4.*pi*(dist*ct._kpc_over_cm_)**2.  # [cm^2]
    Spk = Lpk/area/ct._Jy_over_cgs_irrad_  # [Jy]
    sup = S0/Spk

    frac_tpk = (tpk/365.)/t_age
    frac_tt = ap.ftt(gamma, frac_tpk, sup)
    tt = frac_tt*t_age

    frac_tt_mask = 1.*np.heaviside(frac_tt-1., 1.) + frac_tt*np.heaviside(
        1.-frac_tt, 0.)*np.heaviside(frac_tt-frac_tpk, 0.) + frac_tpk*np.heaviside(frac_tpk-frac_tt, 1.)

    dimless_int = echo_tot_fn(gamma, frac_tpk, frac_tt_mask)

    A_fac = pref(ma, ga, deltaE_over_E=deltaE_over_E)  # [1/kpc]
    x_age = t_age/ct._kpc_over_lightyear_  # [kpc]

    Se = A_fac*Spk*x_age*dimless_int

    return Se


#########################################
# Signal and noise computations


def signal(source_input, axion_input, data,
           recycle_output=(False, None),
           **Snu_echo_kwargs):
    """
    Returns:
    the signal frequency [GHz] ('signal_nu'),
    signal line width [GHz] ('signal_delnu'),
    signal solid angle [sr] ('signal_Omega'),
    flux density/spectral irradiance [Jy] ('signal_Snu'),
    signal flux/irradiance [eV^4] ('signal_S_echo'),
    signal temperature [K] ('signal_Tant'), and
    signal power [eV^2] ('signal_power').
    Saves to output.

    NOTE: we ignore the signal's spectral irradiance variation over the very narrow bandwidth.

    Parameters
    ----------
    source_input : dictionary with source input parameters
    axion_input : dictionary with axion parameters
    data : dictionary with environmental, experimental, and observational data
    recycle_output : whether we recycle a previous computation; and the location where it is stored (default: (False, None))
    Snu_echo_kwargs : Snu_echo() keyword arguments
    """

    # checking if the dictionaries are in the correct format
    check_source(source_input)
    check_axion(axion_input)
    check_data(data)

    # # Update source_input with:
    # # size
    Omega_source = Omega_size(source_input)
    # Omega_dispersion
    Omega_dispersion(source_input, data,
                     tmin_default=Snu_echo_kwargs['tmin_default'],
                     xmax_default=Snu_echo_kwargs['xmax_default'],
                     t_extra_old=Snu_echo_kwargs['t_extra_old'])

    # axion parameters
    ma = axion_input['ma']  # [eV] axion mass
    # [GHz] frequency corresponding to half the axion mass
    nu = pt.nu_from_ma(ma)

    # data parameters
    deltaE_over_E = data['deltaE_over_E']  # fractional bandwidth
    delnu = nu*deltaE_over_E  # [GHz] bandwidth
    f_Delta = data['f_Delta']  # the fraction of flux in the bandwidth
    exper = data['exper']  # experiment requested
    # "single dish" or "interferometry"
    correlation_mode = data['correlation_mode']

    # solid angle of signal
    signal_Omega = max(
        Omega_source, source_input['Omega_dispersion'], source_input['Omega_aberration'])
    theta_sig = ct.solid_angle_to_angle(signal_Omega)
    try:
        verbose = data['verbose']
        if verbose > 2:
            print("echo.py: : size=%.1e, \n\
\tsource_input['Omega_dispersion']=% .1e, \n\
\tsource_input['Omega_aberration']= % .1e" % (
                Omega_source,
                source_input['Omega_dispersion'],
                source_input['Omega_aberration']))
    except KeyError:
        pass

    # # test
    # print("echo.py: source_input['Omega_aberration']=%.1e" %
    #       source_input['Omega_aberration'])
    # print("echo.py: source_input['Omega_dispersion']=%.1e" %
    #       source_input['Omega_dispersion'])

    # finding the experimental range
    if exper == 'SKA':  # in case the range is frequency-dependent
        # could be either 'SKA low', 'SKA mid', or None, depending on frequency nu.
        exper_mode = sk.SKA_exper_nu(nu)
    elif exper in ['SKA low', 'SKA mid']:  # in case the range was fixed by hand
        exper_mode = exper
    else:
        raise ValueError(
            "data['exper'] must be either 'SKA', 'SKA low', or 'SKA mid'. Please update accordingly.")

    # computing the collecting area and the frequency sensitivity window of the experiment mode
    area, window, _, eta, _, Ndishes, _ = sk.SKA_specs(
        nu, exper_mode, correlation_mode=correlation_mode, theta_sig=theta_sig)

    # checking for recycle:
    recycle, output = recycle_output

    if (not recycle) or (type(output) != dict):
        signal_Snu = Snu_echo(source_input, axion_input, data,
                              recycle_output=recycle_output,
                              **Snu_echo_kwargs)  # [Jy]

    elif recycle and ('echo_Snu' in output.keys()):
        signal_Snu = output['echo_Snu']  # [Jy]

    else:
        raise ValueError(
            "'recycle_output' is not of the form (bool, dict). Please update.")

    # compute the integrated flux S for signal (ignoring the non-uniform dependence of Snu on nu; which is OK for a narrow band)
    # [eV^4] irradiance (flux) over the bandwidth
    signal_S_echo = (signal_Snu*ct._Jy_over_SI_) * \
        (delnu*1.e9)  # [W/m^2] = [J/s/m^2]
    signal_S_echo *= ct._J_over_eV_ / \
        (ct._m_eV_**2. * ct._s_eV_)  # [W/m^2] --> eV^4
    # signal_S_echo = (signal_Snu*ct._Jy_over_eV3_)*(delnu*ct._GHz_over_eV_) # equivalent to the previous one...

    # compute the signal power
    # [eV^2], assuming the default SKA efficiency of eta = 0.8

    signal_power = ap.P_signal(
        S=signal_S_echo, A=area, eta=eta, f_Delta=f_Delta)

    # truncate the power according to the SKA freq range window
    signal_power *= window

    # computing signal (antenna) temperature:
    dish_area = area/Ndishes
    signal_Tant = ap.T_signal(signal_Snu, dish_area, eta=eta, f_Delta=f_Delta)

    if recycle and (type(output) == dict):

        output['signal_nu'] = nu
        output['signal_delnu'] = delnu
        output['signal_Omega'] = signal_Omega
        output['signal_Snu'] = signal_Snu
        output['signal_S_echo'] = signal_S_echo
        output['signal_Tant'] = signal_Tant
        output['signal_power'] = signal_power

        # output
        if data['verbose'] > 0:
            print(
                "Signal computed and saved in:\n\noutput['signal_nu']\t[GHz]\noutput['signal_delnu']\t[GHz]\noutput['signal_Omega']\t[sr]\\noutput['signal_Snu']\t[Jy]\noutput['signal_S_echo']\t\t[eV^4]\noutput['signal_Tant']\t[K]\noutput['signal_power']\t[eV^2]\n")

    return (nu,
            delnu,
            signal_Omega,
            signal_Snu,
            signal_S_echo,
            signal_Tant,
            signal_power)


def noise(source_input, axion_input, data,
          recycle_output=(False, None), **Omdisp_kwargs):
    """
    Returns:
    the noise frequency [GHz] ('noise_nu'),
    noise line width [GHz] ('noise_delnu'),
    observation solid angle [sr] ('noise_Omega_obs'),
    background noise brightness temperature at 408 MHz [K] ('noise_T408'),
    system temperature [K] ('noise_Tsys'),
    noise rms temperature [K] (noise_Trms), and
    noise power [eV^2] ('noise_power').
    Saves to output.

    Parameters
    ----------
    source_input : dictionary with source input parameters
    axion_input : dictionary with axion parameters
    data : dictionary with environmental, experimental, and observational data
    recycle_output : whether we recycle a previous computation; and the location where it is stored (default: (False, None))
    Omdisp_kwargs : Omega_dispersion() keyword arguments

    NOTE: we ignore the noise's brightness temperature T_sys variation over the very narrow bandwidth.
    """

    # checking if the dictionaries are in the correct format
    check_source(source_input)
    check_axion(axion_input)
    check_data(data)

    # Update source_input with:
    # Omega_dispersion
    Omega_dispersion(source_input, data, **Omdisp_kwargs)

    # source parameters
    l_source = source_input['longitude']  # [deg] galactic longitude of source
    b_source = source_input['latitude']  # [deg] galactic latitude of source
    Omega_source = Omega_size(source_input)  # [sr] solid angle of source

    # TODO: compute the angular properties of the echo
    l_echo = l_source + 180.  # [deg] galactic longitude of echo
    b_echo = -b_source  # [deg] galactic latitude of echo
    # [sr] solid angle of echo
    signal_Omega = max(source_input['Omega_dispersion'],
                       Omega_source, source_input['Omega_aberration'])
    theta_sig = ct.solid_angle_to_angle(signal_Omega)

    # axion parameters
    ma = axion_input['ma']  # [eV] axion mass
    # [GHz] frequency corresponding to half the axion mass
    nu = pt.nu_from_ma(ma)

    # data parameters
    deltaE_over_E = data['deltaE_over_E']
    delnu = nu*deltaE_over_E
    f_Delta = data['f_Delta']  # the fraction of flux in the bandwidth
    obs_time = data['total_observing_time']  # total observation time [hr]
    exper = data['exper']  # experiment requested
    # whether we average the noise brightness temperature over the angular size of the source
    average = data['average']
    correlation_mode = data['correlation_mode']

    # finding the experimental range:
    if exper == 'SKA':  # in case the range is frequency-dependent
        # could be either 'SKA low', 'SKA mid', or None, depending on frequency nu.
        exper_mode = sk.SKA_exper_nu(nu)
    elif exper in ['SKA low', 'SKA mid']:
        exper_mode = exper  # in case the range was fixed by hand
    else:
        raise ValueError(
            "data['exper'] must be either 'SKA', 'SKA low', or 'SKA mid'. Please update accordingly.")

    # generate the noise power at nu
    # we ignore the non-uniform dependence of T_sys on nu; which is OK for a narrow band:
    # dlog T_sys/dlog nu ~ -beta*deltaE_over_E ~ -0.00255)

    # reading out the receiver's noise brightness temperature and solid angle resolution
    _, _, Tr, _, Omega_res, _, _ = sk.SKA_specs(
        nu, exper_mode, correlation_mode=correlation_mode, theta_sig=theta_sig)

    # the observation solid angle [sr]
    # only meaningful for single dish mode. Interferometry mode doesn't change anything as Omega_res is given Omega_sig
    Omega_obs = max(Omega_res, signal_Omega)

    # compute background brightness temperature at 408 MHz at the location of the echo.
    Tbg_408_at_echo_loc = ap.bg_408_temp(
        l=l_echo, b=b_echo, size=Omega_obs, average=average, verbose=False)  # [K]

    # computing system temperature
    T_sys = np.squeeze(ap.T_sys(
        nu, Tbg_at_408=Tbg_408_at_echo_loc, Tr=Tr))  # [K]

    # computing the noise rms temperature
    Trms = ap.T_noise(T_sys=T_sys,
                      delnu=delnu,
                      tobs=obs_time,
                      Omega_obs=Omega_obs,
                      Omega_res=Omega_res,
                      nu=nu,
                      correlation_mode=correlation_mode)  # [K]

    # computing the noise power
    noise_power = ap.P_noise(T_sys=T_sys,
                             delnu=delnu,
                             tobs=obs_time,
                             Omega_obs=Omega_obs,
                             Omega_res=Omega_res,
                             nu=nu,
                             correlation_mode=correlation_mode)  # [eV^2]

    # checking for recycle:
    recycle, output = recycle_output

    if recycle and (type(output) == dict):

        output['noise_nu'] = nu
        output['noise_delnu'] = delnu
        output['noise_Omega_res'] = Omega_res
        output['noise_Omega_obs'] = Omega_obs
        output['noise_T408'] = Tbg_408_at_echo_loc
        output['noise_Tsys'] = T_sys
        output['noise_Trms'] = Trms
        output['noise_power'] = noise_power

        # output
        if data['verbose'] > 0:
            print(
                "Noise computed and saved in:\n\noutput['noise_nu']\t[GHz]\noutput['noise_delnu']\t[GHz]\noutput['noise_Omega_res']\t[sr]\noutput['noise_Omega_obs']\t[sr]\noutput['noise_T408']\t[K]\noutput['noise_Tsys']\t[K]\noutput['noise_Trms']\t[K]\noutput['noise_power']\t[eV^2]\n")

    return (nu,
            delnu,
            Omega_res,
            Omega_obs,
            Tbg_408_at_echo_loc,
            T_sys,
            Trms,
            noise_power)


def sn_ratio(signal_power, noise_power,
             output=None,
             verbose=0):
    """
    Returns the signal-to-noise ratio.

    Parameters
    ----------
    signal_power : power of signal [eV^2]
    noise_power : power of noise [eV^2]
    output : output dictionary (default: None)
    verbose : verbosity (default: 0)
    """

    # ratio
    res = signal_power / noise_power

    if type(output) == dict:

        # saving to ouput dictionary
        output['S/N_power'] = res

        if verbose > 0:
            print("S/N computed and saved in output['S/N_power'].")

    return res


def sn_temp_ratio(signal_Tant, noise_Trms,
                  output=None,
                  verbose=0):
    """
    Returns the signal-to-noise ratio.

    Parameters
    ----------
    signal_Tant : signal (antenna) temperature [K]
    noise_Trms : noise rms temperature [K]
    output : output dictionary (default: None)
    verbose : verbosity (default: 0)
    """

    # ratio
    res = signal_Tant / noise_Trms

    if type(output) == dict:

        # saving to ouput dictionary
        output['S/N_temp'] = res

        if verbose > 0:
            print("S/N_temp computed and saved in output['S/N_temp'].")

    return res


def snr_fn(Secho, nu, delta_nu, Omega_obs=1.e-4, Tbg_408=Tbg_408_avg, eta=None, f_Delta=ct._f_Delta_, tobs=100., correlation_mode=None, theta_sig=None, beta=ct._MW_spectral_beta_):
    """
    Simpler signal-to-noise ratio formula. [DEPRECATED]

    Parameters
    ----------
    Secho : echo flux density (spectral irradiance) [Jy]
    nu : signal frequency [GHz]
    delta_nu : the bandwidth of the detector [GHz]
    Omega_obs : the observation solid angle [sr]
    Tbg_408 : the MW background at 408 MHz [K] (default: full sky average)
    eta: the detector efficiency (default: 0.8)
    f_Delta : the fraction of signal falling withing the bandwidth (default: 0.83848)
    tobs : the total observation time [hours] (default: 100)
    correlation_mode : the correlation mode of the experiment <"single dish"|"interferometry"> (default: None)
    theta_sig: the size of the signal to be observed, relevant for the interferometry mode [radian]. (default: None)
    """

    delta_nu_in_eV = delta_nu * ct._GHz_over_eV_
    Secho_in_eV3 = Secho * ct._Jy_over_eV3_
    Stot = Secho_in_eV3*delta_nu_in_eV

    # experiment
    exper_mode = sk.SKA_exper_nu(nu)
    # area, window, Tr, eta, Omega_res, _, _ = sk.SKA_specs(
    #     nu, exper_mode, eta=eta, correlation_mode=correlation_mode, theta_sig=theta_sig)
    area, window, Tr, eta, Omega_res, _, _ = sk.SKA_specs(
        nu, exper_mode, correlation_mode=correlation_mode, theta_sig=theta_sig)

    # if Omega_obs > Omega_max:
    #     Psig = 0.
    # else:
    #    Psig = ap.P_signal(Stot, area, eta=eta, f_Delta=f_Delta)
    # print exper_mode
    Psig = ap.P_signal(Stot, area, eta=eta, f_Delta=f_Delta)

    Tsys = ap.T_sys(nu, Tbg_at_408=Tbg_408, beta=beta, Tr=Tr)
    Pnoise = ap.P_noise(Tsys, delta_nu, tobs, Omega_obs,
                        Omega_res, nu, correlation_mode)

    return Psig/Pnoise


def ga_reach(sn_val, sn_ref, ga_ref):
    """
    Computes the reach in the axion-photon coupling ga [GeV^-1].

    Parameters
    ----------
    sn_val : the threshold value of the signal-to-noise ratio at which the reach is defined
    sn_ref : a reference signal-to-noise ratio
    ga_ref : the reference axion-photon coupling at which the reference signal-to-noise ratio was computed [GeV^-1]
    """
    try:
        for i in range(len(sn_ref)):
            if sn_ref[i] == 0:
                sn_ref[i] = 1.e-100
                # print("zero found at %d-th entry:" % i)
    except:
        pass
    return ga_ref * sqrt(sn_val/sn_ref)


def ma_ga_bound(sn_val, ma_arr, sn_ref_arr, ga_ref):
    """
    Returns the (ma, ga) coordinates of a constraint in axion parameter space for a given threshold value of signal-to-noise ratio.

    Parameters
    ----------
    sn_val : the threshold value of the signal-to-noise ratio at which the bound is defined
    ma_arr : the array of axion masses for which the reference signal-to-noise ratios were computed [eV]
    sn_ref_arr : the array of signal-to-noise ratios computed for the given axion masses and reference axion-photon coupling
    ga_ref : the reference axion-photon coupling at which the reference signal-to-noise ratios were computed [GeV^-1]
    """

    if len(ma_arr) != len(sn_ref_arr):
        raise ValueError(
            "The arguments 'ma_arr' and 'sn_ref_arr' must be of the same length, since the latter are the signal-to-noise ratios computed for the axion masses given by the former.")

    ga_arr = ga_reach(sn_val, sn_ref_arr, ga_ref)
    ma_ga = np.vstack((ma_arr, ga_arr)).T

    return ma_ga
