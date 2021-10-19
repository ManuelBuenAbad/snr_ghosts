from __future__ import division

import numpy as np
from numpy import pi, sqrt, exp, power, log, log10

import os

import constants as ct
import particle as pt
import tools as tl


##############################
# Preparing SKA configurations
##############################
def main():
    SKA_conf = {}

    # --------------
    # SKA-low
    for exp in ['low', 'mid']:
        if exp == "low":
            path = os.path.dirname(os.path.abspath(__file__)) + \
                "/data/SKA1-low_accumu.csv"

        elif exp == "mid":
            path = os.path.dirname(os.path.abspath(__file__)) + \
                "/data/SKA1-mid_accumu.csv"

        data_raw = np.loadtxt(path, delimiter=',')
        radius = data_raw[:, 0]
        fraction = data_raw[:, 1]
        bins_radius = np.logspace(1, 5, 20)  # bin it
        hist_radius = np.interp(np.log10(bins_radius), np.log10(
            radius), fraction, left=0)  # sample at the bin edges

        if exp == "low":
            # compute the x-y coordinates of all units
            x_arr, y_arr = get_telescope_coordinate(
                fraction*ct._SKALow_number_of_stations_, radius, SKA=exp)
            # save it
            SKA_conf['low radius'] = (data_raw, x_arr, y_arr, bins_radius,
                                      hist_radius)
        elif exp == "mid":
            x_arr, y_arr = get_telescope_coordinate(
                fraction*ct._SKA1Mid_number_of_dishes_, radius, SKA=exp)
            SKA_conf['mid radius'] = (data_raw, x_arr, y_arr, bins_radius,
                                      hist_radius)

        # get baseline distribution
        baseline_arr = get_baseline(x_arr, y_arr)
        hist_baseline, bins_baseline = np.histogram(
            baseline_arr, bins=np.logspace(1, 5, 20))
        hist_baseline_cumsum = np.cumsum(hist_baseline)
        # save it
        if exp == "low":
            SKA_conf['low baseline'] = (
                baseline_arr, hist_baseline, bins_baseline, hist_baseline_cumsum)
        elif exp == "mid":
            SKA_conf['mid baseline'] = (
                baseline_arr, hist_baseline, bins_baseline, hist_baseline_cumsum)

    return SKA_conf

################
# SKA properties
################


def SKA_get_active_baseline(length, exper_mode):
    """Get the active number of baselines in the interferometry mode

    :param length: critical baseline below which the signal can be resolved
    :param exper_mode: "SKA low" or "SKA mid"
    :returns: number of baselines that sees the signal

    """
    length_arr, is_scalar = tl.treat_as_arr(length)

    if exper_mode == "SKA low":
        (baseline_arr, hist_baseline, bins_baseline,
         hist_baseline_cumsum) = SKA_conf['low baseline']
    if exper_mode == "SKA mid":
        (baseline_arr, hist_baseline, bins_baseline,
         hist_baseline_cumsum) = SKA_conf['mid baseline']

    res = np.interp(np.log(length_arr), np.log(bins_baseline[:-1]),
                    hist_baseline_cumsum, left=ct._zero_)

    if exper_mode == "SKA low":
        res[length_arr < ct._SKALow_station_diameter_] = ct._zero_
    if exper_mode == "SKA mid":
        res[length_arr < ct._SKA1Mid_dish_diameter_] = ct._zero_

    if is_scalar:
        res = np.squeeze(res)
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
        # assuming this is the aperture angle and not the radial angle
        theta_res = (1.02*wavelength)/ct._SKALow_station_diameter_/sqrt(eta)
        Omega_res = ct.angle_to_solid_angle(
            theta_res)  # solid angle of resolution [sr]
        number_of_dishes = ct._area_ska_low_ / \
            (np.pi * ct._SKALow_station_diameter_**2 / 4.)
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
        active_number_of_baselines = SKA_get_active_baseline(
            critical_baseline_length, exper_mode='SKA low')
        # taking the resolution to be exactly the signal size
        # penalty is taken care of through active_number_of_baselines
        theta_res = theta_sig
        Omega_res = ct.angle_to_solid_angle(
            theta_res)  # solid angle of resolution [sr]
        # for interferometry mode noise has 1/sqrt(number of active baselines) factor
        number_of_measurements = active_number_of_baselines
        # assuming all dishes/stations contribute
        # since S and N scale the same with reception area, S/N cancels out
        # in the end only the number of measurements (baselines) matter
        area = ct._area_ska_low_
        number_of_dishes = ct._SKALow_number_of_stations_
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
        active_number_of_baselines = SKA_get_active_baseline(
            critical_baseline_length, exper_mode='SKA mid')
        # taking the resolution to be exactly the signal size
        # penalty is taken care of through active_num_of_baselines
        theta_res = theta_sig
        Omega_res = ct.angle_to_solid_angle(
            theta_res)  # solid angle of resolution [sr]
        # for interferometry mode noise has 1/sqrt(number of active baselines) factor
        number_of_measurements = active_number_of_baselines
        # assuming all dishes/stations contribute
        # since S and N scale the same with reception area, S/N cancels out
        # in the end only the number of measurements (baselines) matter
        area = ct._area_ska_mid_
        number_of_dishes = ct._SKA1Mid_number_of_dishes_

    # in case the number of baselines is zero
    if number_of_measurements == 0:
        number_of_measurements = 1e-100

    return area, window, Tr, Omega_res, number_of_dishes, number_of_measurements


def get_telescope_coordinate(tel_arr, r_arr, SKA):
    """Generate an array with coordinate of each telescope computed

    :param tele_arr: the array of telescope index from 1 to (number of telescope)
    :param radius_arr: the radius of each telescope
    :param SKA: "low" or "mid"

    """
    if SKA == "low":
        tel_fine_arr = np.arange(ct._SKALow_number_of_stations_)
        r_core = ct._SKALow_r_core_
    elif SKA == "mid":
        tel_fine_arr = np.arange(ct._SKA1Mid_number_of_dishes_)
        r_core = ct._SKA1Mid_r_core_
    r_fine_arr = np.interp(tel_fine_arr, tel_arr, r_arr)

    # fix see as we don't really need the randomness
    np.random.seed(123)
    theta_arr = np.random.random(size=len(r_fine_arr)) * np.pi * 2.

    # over write the arm part
    mask = np.where(r_fine_arr > r_core, True, False)
    for i in tel_fine_arr:
        if r_fine_arr[int(i)] > r_core:
            theta_arr[int(i)] = int(i) % 3 * 2. * np.pi / 3.

    x_arr = r_fine_arr * np.cos(theta_arr)
    y_arr = r_fine_arr * np.sin(theta_arr)

    return x_arr, y_arr


def get_baseline(x_arr, y_arr):
    """Given array coordinates x, y, compute lengths of each pair. Returns the array of pair lengths.

    :param x_arr: x coordinate of all units
    :param y_arr: y coordinates of all units

    """
    n_unit = len(x_arr)
    n_baseline = int(n_unit * (n_unit - 1) / 2.)
    baseline_arr = np.zeros((n_unit, n_unit))
    for i in range(n_unit):
        for j in range(n_unit):
            # print("x[i]=%s, y[j]=%s" % (x_arr[i], y_arr[j]))

            dist = np.sqrt((x_arr[i] - x_arr[j])**2 + (y_arr[i] - y_arr[j])**2)
            baseline_arr[i, j] = dist
            # baseline_arr[j, i] = dist
    baseline_arr = baseline_arr.reshape(-1)
    baseline_arr = baseline_arr[baseline_arr > 0]
    return baseline_arr

##################################


# The global variable to be saved
SKA_conf = main()
