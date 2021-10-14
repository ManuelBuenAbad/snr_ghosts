from __future__ import division

import numpy as np
from numpy import pi, sqrt, exp, power, log, log10

import os

import constants as ct
import particle as pt


# Getting SKA baselines

def get_baseline(r, dist_r_arr, dist_frac_arr, Ntot=None, exper_mode=None):
    """Compute the averaged baseline distance assuming they distribute according to dist_r_arr and dist_frac_arr

    :param r: radius from the center for telescopes whose baseline length is to be estimated
    :param dist_r_arr: grid's radius from the center
    :param dist_frac_arr: the distribution (of fraction of total telescopes) as a function of dist_r_arr.
    :param Ntot: total number of telescopes
    :param exper_mode: "SKA low" or "SKA mid"
    :returns: the baseline length of telescopes with a distance r from the center
    :rtype:

    """

    fraction = np.interp(np.log10(r), np.log10(dist_r_arr), (dist_frac_arr))
    number_of_tel = Ntot * fraction
    B_eff = 2.*r / np.sqrt(number_of_tel)

    if exper_mode == "SKA mid":
        minimal_baseline = ct._SKA1Mid_dish_diameter_
        maximal_baseline = ct._SKA1Mid_maximal_baseline_
    elif exper_mode == "SKA low":
        minimal_baseline = ct._SKALow_dish_diameter_
        maximal_baseline = ct._SKALow_maximal_baseline_

    # treating the empty bins, which leads to np.inf in baseline
    try:
        if r < 1.e3 and number_of_tel == 0:
            B_eff = minimal_baseline
        if r > 1.e3 and number_of_tel == 0:
            B_eff = maximal_baseline
    except ValueError:
        mask1 = np.where(number_of_tel == 0, True, False)
        mask2 = np.where(r < 1.e3, True, False)
        mask_small = mask1 * mask2
        B_eff[mask_small] = minimal_baseline

        mask2 = np.where(r > 1.e3, True, False)
        mask_large = mask1 * mask2
        B_eff[mask_large] = maximal_baseline

    return B_eff


##############################
# Preparing SKA configurations
##############################

SKA_conf = {}

#--------------
# SKA-low

path = os.path.dirname(os.path.abspath(__file__))+"/data/SKA1-low_accumu.csv"
SKA_conf['low'] = np.loadtxt(path, delimiter=',')

x = SKA_conf['low'][:, 0]
y = SKA_conf['low'][:, 1]
bins = np.logspace(1, 5, 20)  # bin it
hist = np.interp(np.log10(bins), np.log10(
    x), y, left=0)  # sample at the bin edges
dist_r_arr = bins[1:]  # get the bin edges
dist_frac_arr = (hist[1:] - hist[:-1])  # get the distribution
baseline_arr = get_baseline(
    dist_r_arr, dist_r_arr=dist_r_arr, dist_frac_arr=dist_frac_arr, exper_mode="SKA low", Ntot=ct._SKALow_number_of_stations_)
SKA_conf['low baseline distribution'] = (baseline_arr, dist_frac_arr)
SKA_conf['low baseline cumulative'] = (baseline_arr, np.cumsum(dist_frac_arr))

# garbage collection to avoid errors
SKA_conf['debug low'] = (x, y, bins, hist, dist_r_arr,
                         dist_frac_arr, baseline_arr)
del x
del y
del bins
del hist
del dist_r_arr
del dist_frac_arr
del baseline_arr

#--------------
# SKA-mid
path = os.path.dirname(os.path.abspath(__file__))+"/data/SKA1-mid_accumu.csv"
SKA_conf['mid'] = np.loadtxt(path, delimiter=',')

x = SKA_conf['mid'][:, 0]
y = SKA_conf['mid'][:, 1]
bins = np.logspace(1, 5, 20)  # bin it
hist = np.interp(np.log10(bins), np.log10(
    x), y, left=0)  # sample at the bin edges
dist_r_arr = bins[1:]  # get the bin edges
dist_frac_arr = (hist[1:] - hist[:-1])  # get the distribution
baseline_arr = get_baseline(
    dist_r_arr, dist_r_arr=dist_r_arr, dist_frac_arr=dist_frac_arr, exper_mode="SKA mid", Ntot=ct._SKA1Mid_number_of_dishes_)
SKA_conf['mid baseline distribution'] = (baseline_arr, dist_frac_arr)
SKA_conf['mid baseline cumulative'] = (baseline_arr, np.cumsum(dist_frac_arr))

# garbage collection to avoid errors
SKA_conf['debug mid'] = (x, y, bins, hist, dist_r_arr,
                         dist_frac_arr, baseline_arr)
del x
del y
del bins
del hist
del dist_r_arr
del dist_frac_arr
del baseline_arr


################
# SKA properties
################

def SKA_get_active_baseline(length, exper_mode):
    """Get the active number of baselines in the interferometry mode

    :param length: critical baseline below which the signal can be resolved
    :param exper_mode: "SKA low" or "SKA mid"
    :returns: number of baselines that sees the signal

    """
    if exper_mode == "SKA low":
        baseline_arr, cumu_frac_arr = SKA_conf['low baseline cumulative']
    if exper_mode == "SKA mid":
        baseline_arr, cumu_frac_arr = SKA_conf['mid baseline cumulative']

    # baseline_arr[baseline_arr == np.inf] = 1.e-100
    res = np.interp(np.log(length), np.log(baseline_arr),
                    cumu_frac_arr, left=0, right=1)
    # return baseline_arr, cumu_frac_arr
    return res



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


def SKA_specs(nu, exper_mode, eta=ct._eta_ska_, correlation_mode=None, theta_sig=None):
    """
    Returns the specifications (area [m^2], window, receiver noise brightness temperature [K], and solid angle resolution [sr], number_of_dishes, number_of_measurements) of the SKA experiment mode, for the given frequency [GHz].

    Parameters
    ----------
    nu : frequency [GHz]
    exper_mode : mode in which the experiment is working
    eta: the detector efficiency (default: 0.8)
    correlation_mode: whether to run in interferometry mode or single dish mode. Default None is meant to raise error if not assigned explicitly.
    theta_sig: the signal size we want to observe [radian]
    """

    if exper_mode == None:
        # area, window, Tr, Omega_res, Omega_max, number_of_dishes = 0., 0., 0., 1.e-100, np.inf, 1e100
        area, window, Tr, Omega_res, number_of_dishes, number_of_measurements = 0., 0., 0., 1.e-100, 0., 0.  # set to zero so it will raise error if not treated

    elif exper_mode == 'SKA low' and correlation_mode == "single dish":
        area = ct._area_ska_low_
        window = np.heaviside(nu - ct._nu_min_ska_low_, 1.) * \
            np.heaviside(ct._nu_max_ska_low_ - nu, 1.)
        Tr = 40.

        # finding resolution:
        wavelength = pt.lambda_from_nu(nu)/100.  # wavelength [m]
        # angular size of pixel resolution [rad]
        # theta_res = (1.02*wavelength)/sqrt(eta*4.*area/pi) # assuming this is the aperture angle and not the radial angle
        # assuming this is the aperture angle and not the radial angle
        theta_res = (1.02*wavelength)/ct._SKALow_dish_diameter_/sqrt(eta)
        Omega_res = ct.angle_to_solid_angle(
            theta_res)  # solid angle of resolution [sr]
        number_of_dishes = ct._area_ska_low_ / \
            (np.pi * ct._SKALow_dish_diameter_**2 / 4.)
        number_of_measurements = number_of_dishes
        # Omega_max = np.inf  # being sloppy here but we never reach FOV

    elif exper_mode == 'SKA low' and correlation_mode == "interferometry":
        window = np.heaviside(nu - ct._nu_min_ska_low_, 1.) * \
            np.heaviside(ct._nu_max_ska_low_ - nu, 1.)
        Tr = 40.

        # get the required baseline length for nu
        wavelength = pt.lambda_from_nu(nu) / 100.  # wavelength [m]
        critical_baseline_length = (
            1.02*wavelength) / (theta_sig)\
            * ct._SKA_factor_lose_signal_  # fudge factor could be ~ 2 or 3 to deem the signal cannot be observed
        # get the active number of baselines
        active_fraction_of_baselines = SKA_get_active_baseline(
            critical_baseline_length, exper_mode='SKA low')
        # taking the resolution to be exactly the signal size
        # penalty is taken care of through active_fraction_of_baselines
        theta_res = theta_sig
        Omega_res = ct.angle_to_solid_angle(
            theta_res)  # solid angle of resolution [sr]
        # for interferometry mode noise has 1/sqrt(number of active baselines) factor
        number_of_measurements = active_fraction_of_baselines * ct._SKALow_total_baselines_
        # assuming the fraction of baseline is the same for the fraction of stations
        area = ct._area_ska_low_ * active_fraction_of_baselines
        number_of_dishes = ct._area_ska_low_ / \
            (np.pi * ct._SKALow_dish_diameter_**2 / 4.) * \
            active_fraction_of_baselines
        # print("SKA-low, nu=%.1e GHz, critical_baseline=%.1em" %
        #       (nu, critical_baseline_length))

    elif exper_mode == 'SKA mid' and correlation_mode == "single dish":
        area = ct._area_ska_mid_
        window = np.heaviside(nu - ct._nu_min_ska_mid_, 0.) * \
            np.heaviside(ct._nu_max_ska_mid_ - nu, 1.)
        Tr = 20.

        # finding resolution:
        wavelength = pt.lambda_from_nu(nu)/100.  # wavelength [m]
        # angular size of pixel resolution [rad]
        # assuming this is the aperture angle and not the radial angle
        # theta_res = (1.02*wavelength)/sqrt(eta*4.*area/pi)
        theta_res = (1.02*wavelength)/ct._SKA1Mid_dish_diameter_/sqrt(eta)
        Omega_res = ct.angle_to_solid_angle(
            theta_res)  # solid angle of resolution [sr]
        number_of_dishes = ct._area_ska_mid_ / \
            (np.pi * ct._SKA1Mid_dish_diameter_**2 / 4.)
        number_of_measurements = number_of_dishes
        Omega_max = np.inf  # being sloppy here but we never reach FOV

    elif exper_mode == 'SKA mid' and correlation_mode == "interferometry":

        area = ct._area_ska_mid_
        window = np.heaviside(nu - ct._nu_min_ska_mid_, 0.) * \
            np.heaviside(ct._nu_max_ska_mid_ - nu, 1.)
        Tr = 20.

        # get the required baseline length for nu
        wavelength = pt.lambda_from_nu(nu) / 100.  # wavelength [m]
        critical_baseline_length = (
            1.02*wavelength) / (theta_sig)\
            * ct._SKA_factor_lose_signal_  # fudge factor 2 to deem the signal cannot be observed
        # print("SKA1-mid, nu=%.1e GHz, critical_baseline=%.1em" %
        #       (nu, critical_baseline_length))
        # get the active number of baselines
        active_fraction_of_baselines = SKA_get_active_baseline(
            critical_baseline_length, exper_mode='SKA mid')
        # taking the resolution to be exactly the signal size
        # penalty is taken care of through active_num_of_baselines
        theta_res = theta_sig
        Omega_res = ct.angle_to_solid_angle(
            theta_res)  # solid angle of resolution [sr]
        # for interferometry mode noise has 1/sqrt(number of active baselines) factor
        number_of_measurements = active_fraction_of_baselines * ct._SKA1Mid_total_baselines_
        # assuming the fraction of baseline is the same for the fraction of dishes
        area = ct._area_ska_mid_ * active_fraction_of_baselines
        number_of_dishes = ct._area_ska_mid_ / \
            (np.pi * ct._SKA1Mid_dish_diameter_**2 / 4.) * \
            active_fraction_of_baselines

    # in case the number of baselines is zero
    if number_of_measurements == 0:
        number_of_measurements = 1e-100

    return area, window, Tr, Omega_res, number_of_dishes, number_of_measurements
