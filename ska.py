from __future__ import division

import numpy as np
from numpy import pi, sqrt, exp, power, log, log10

import os

import constants as ct
import particle as pt
import tools as tl

#####################
# Defining local path
#####################

local_path = os.path.dirname(os.path.abspath(__file__))


##############################
# Preparing SKA configurations
##############################

def main():
    """This routine is supposed to be run only once, therefore\
 the I/O is not optimized for speed concerns.

    """

    SKA_conf = {}

    # --------------
    # SKA-low
    for exper in ['low', 'mid']:
        if exper == "low":
            path = local_path + "/data/SKA1-low_accumu.csv"

        elif exper == "mid":
            path = local_path + "/data/SKA1-mid_accumu.csv"

        data_raw = np.loadtxt(path, delimiter=',')
        radius = data_raw[:, 0]
        fraction = data_raw[:, 1]
        bins_radius = np.logspace(1, 5, 20)  # bin it
        hist_radius = np.interp(np.log10(bins_radius), np.log10(
            radius), fraction, left=0)  # sample at the bin edges

        if exper == "low":
            # compute the x-y coordinates of all units
            x_arr, y_arr = get_telescope_coordinate(
                fraction*ct._SKALow_number_of_stations_, radius, SKA=exper)
            # save it
            SKA_conf['low radius'] = (data_raw, x_arr, y_arr, bins_radius,
                                      hist_radius)
        elif exper == "mid":
            x_arr, y_arr = get_telescope_coordinate(
                fraction*ct._SKA1Mid_number_of_dishes_, radius, SKA=exper)
            SKA_conf['mid radius'] = (data_raw, x_arr, y_arr, bins_radius,
                                      hist_radius)

        # get baseline distribution
        baseline_arr = get_baseline(x_arr, y_arr)
        hist_baseline, bins_baseline = np.histogram(
            baseline_arr, bins=np.logspace(1, 5, 20))
        hist_baseline_cumsum = np.cumsum(hist_baseline)
        # save it
        if exper == "low":
            SKA_conf['low baseline'] = (
                baseline_arr, hist_baseline, bins_baseline, hist_baseline_cumsum)
        elif exper == "mid":
            SKA_conf['mid baseline'] = (
                baseline_arr, hist_baseline, bins_baseline, hist_baseline_cumsum)

        # about effective area

        if exper == "low":
            path = local_path + "/data/SKA1-low_Aeff_over_Tsys.txt"
            data_raw = np.loadtxt(path)
            # low is given in MHz, convert to GHz
            data_raw[:, 0] = data_raw[:, 0] * 1.e-3
            SKA_conf['low A/T'] = data_raw

        elif exper == "mid":
            path = local_path + "/data/SKA1-mid_Aeff_over_Tsys.txt"
            data_raw = np.loadtxt(path)
            SKA_conf['mid A/T'] = data_raw
    SKA_conf['A/T'] = np.concatenate((SKA_conf['low A/T'],
                                     SKA_conf['mid A/T']))
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



def SKA_specs(nu, exper_mode, correlation_mode=None, theta_sig=None):
    """
    Returns the SKA specifications for the given experiment mode and frequency [GHz]:

    area [m^2],
    window,
    receiver noise brightness temperature [K],
    efficiency,
    solid angle resolution [sr],
    number_of_dishes, and
    number_of_measurements.

    Parameters
    ----------
    nu : frequency [GHz]
    exper_mode : mode in which the experiment is working
    correlation_mode: whether to run in interferometry mode or single dish mode. Default None is meant to raise error if not assigned explicitly.
    theta_sig: the signal size we want to observe [radian]
    """

    if exper_mode == None:
        area, window, Tr, eta, Omega_res, number_of_dishes, number_of_measurements = 0., 0., 0., 0., 1.e-100, 0., 0.  # set to zero so it will raise error if not treated

    elif exper_mode == 'SKA low' and correlation_mode == "single dish":
        area = ct._area_ska_low_
        window = np.heaviside(nu - ct._nu_min_ska_low_, 1.) * \
            np.heaviside(ct._nu_max_ska_low_ - nu, 1.)
        Tr = ct._Tr_ska_low_
        eta = eta_nu(nu, exper_mode)

        # finding resolution:
        wavelength = pt.lambda_from_nu(nu)/100.  # wavelength [m]
        # angular size of pixel resolution [rad]
        # assuming this is the aperture angle and not the radial angle
        theta_res = (1.22*wavelength) / \
            ct._SKALow_station_diameter_  # /sqrt(eta)
        Omega_res = ct.angle_to_solid_angle(
            theta_res)  # solid angle of resolution [sr]
        number_of_dishes = ct._area_ska_low_ / \
            (np.pi * ct._SKALow_station_diameter_**2 / 4.)
        number_of_measurements = number_of_dishes
        # Omega_max = np.inf  # being sloppy here but we never reach FOV

    elif exper_mode == 'SKA low' and correlation_mode == "interferometry":
        window = np.heaviside(nu - ct._nu_min_ska_low_, 1.) * \
            np.heaviside(ct._nu_max_ska_low_ - nu, 1.)
        Tr = ct._Tr_ska_low_
        eta = eta_nu(nu, exper_mode)

        # get the required baseline length for nu
        wavelength = pt.lambda_from_nu(nu) / 100.  # wavelength [m]
        critical_baseline_length = (
            1.22*wavelength) / (theta_sig)\
            * ct._SKA_factor_lose_signal_  # fudge factor for when invisible

        # get the active number of baselines
        active_number_of_baselines = SKA_get_active_baseline(
            critical_baseline_length, exper_mode='SKA low')
        # taking the resolution to be exactly the signal size
        # penalty is taken care of through active_number_of_baselines
        theta_res = theta_sig
        Omega_res = ct.angle_to_solid_angle(
            theta_res)  # solid angle of resolution [sr]

        # for interferometry mode noise has 1/sqrt(number of active baselines)
        number_of_measurements = active_number_of_baselines

        # assuming all dishes/stations contribute
        # since S and N scale the same with reception area, S/N cancels out
        # in the end only the number of measurements (baselines) matter
        area = ct._area_ska_low_
        number_of_dishes = ct._SKALow_number_of_stations_

    elif exper_mode == 'SKA mid' and correlation_mode == "single dish":
        area = ct._area_ska_mid_
        window = np.heaviside(nu - ct._nu_min_ska_mid_, 0.) * \
            np.heaviside(ct._nu_max_ska_mid_ - nu, 1.)
        Tr = ct._Tr_ska_mid_
        eta = eta_nu(nu, exper_mode)

        # finding resolution:
        wavelength = pt.lambda_from_nu(nu)/100.  # wavelength [m]

        # angular size of pixel resolution [rad]
        # assuming this is the aperture angle and not the radial angle
        # theta_res = (1.22*wavelength)/sqrt(eta*4.*area/pi)
        theta_res = (1.22*wavelength)/ct._SKA1Mid_dish_diameter_  # /sqrt(eta)
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
        Tr = ct._Tr_ska_mid_
        eta = eta_nu(nu, exper_mode)

        # get the required baseline length for nu
        wavelength = pt.lambda_from_nu(nu) / 100.  # wavelength [m]
        critical_baseline_length = (
            1.22*wavelength) / (theta_sig)\
            * ct._SKA_factor_lose_signal_  # fudge factor

        # get the active number of baselines
        active_number_of_baselines = SKA_get_active_baseline(
            critical_baseline_length, exper_mode='SKA mid')
        # taking the resolution to be exactly the signal size
        # penalty is taken care of through active_num_of_baselines
        theta_res = theta_sig
        Omega_res = ct.angle_to_solid_angle(
            theta_res)  # solid angle of resolution [sr]

        # for interferometry mode noise has 1/sqrt(number of active baselines)
        number_of_measurements = active_number_of_baselines

        # assuming all dishes/stations contribute
        # since S and N scale the same with reception area, S/N cancels out
        # in the end only the number of measurements (baselines) matter
        area = ct._area_ska_mid_
        number_of_dishes = ct._SKA1Mid_number_of_dishes_

    # in case the number of baselines is zero
    if number_of_measurements == 0:
        number_of_measurements = 1e-100

    return area, window, Tr, eta, Omega_res, number_of_dishes, number_of_measurements



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



############
# Efficiency
############

#--------------------------------------
# (nu, eta) arrays for SKA1 low and mid
nu_eta_low = np.loadtxt(local_path+"/data/eta_low.csv", delimiter=",")
nu_eta_mid = np.loadtxt(local_path+"/data/eta_mid.csv", delimiter=",")

#------------------------
# interpolating the data:
eta_low_fn = tl.interp_fn(nu_eta_low)
eta_mid_fn = tl.interp_fn(nu_eta_mid)

#--------------------------------
# defining the general efficiency

def eta_nu(nu, exper_mode):
    """Returns the efficiency eta.

    nu : frequency [GHz]
    exper_mode : mode in which the experiment is working
    """
    if exper_mode == None:
        eta = 0.
    elif exper_mode == 'SKA low':
        eta = eta_low_fn(nu)
    elif exper_mode == 'SKA mid':
        eta = eta_mid_fn(nu)

    return eta


###############
# Aeff
###############

# def get_eta_eff(nu, Tsys, SKA_conf):
#     """Compute the effective area [m^2] for a given Tsys
#
#     :param Tsys: systematic noise temperature [K]
#     :param SKA_conf: the dictionary that stores the SKA configurations
#
#     """
#     nu_arr = SKA_conf['A/T'][:, 0]
#     area_arr = SKA_conf['A/T'][:, 1] * Tsys
#     nu, is_scalar = tl.treat_as_arr(nu)
#     eta = []
#
#     for nu_i in nu:
#         area_eff = np.interp(nu_i, nu_arr, area_arr)
#
#         if nu_i > ct._nu_max_ska_low_:
#             area_geo = np.pi * (ct._SKA1Mid_dish_diameter_ / 2.)**2
#         else:
#             area_geo = np.pi * (ct._SKALow_station_diameter_ / 2.)**2
#         eta.append(min(area_eff / area_geo, 1))  # eta cannot be larger than 1
#     eta = np.array(eta)
#
#     if is_scalar:
#         return np.squeeze(eta)
#     else:
#         return eta


##################################

# The global variable to be saved
SKA_conf = main()
