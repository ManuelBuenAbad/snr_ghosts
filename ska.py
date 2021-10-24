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

def initialize():
    """This routine is supposed to be run only once, \
i.e. when the module is loaded, therefore\
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
        # correcting the over-counting of baseline pair
        hist_baseline = hist_baseline/2.
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

    # computing efficiency
    # make a nu grid
    Nsteps = 2001

    nulow = np.logspace(log10(ct._nu_min_ska_low_), log10(
        ct._nu_max_ska_low_), Nsteps//2)[1:]
    # ... and SKA mid...
    numid = np.logspace(log10(ct._nu_min_ska_mid_), log10(
        ct._nu_max_ska_mid_), Nsteps - Nsteps//2)[1:]

    Aeff_over_Tsys = SKA_conf['A/T']

    # Mid
    nu_arr = numid
    Aeff_over_Tsys_arr = np.interp(
        nu_arr, Aeff_over_Tsys[:, 0], Aeff_over_Tsys[:, 2])
    Tsys_arr = T_sys_mid(nu_arr)
    eta_arr = Aeff_over_Tsys_arr * Tsys_arr / ct._area_ska_mid_
    SKA_conf['eta mid'] = (nu_arr, eta_arr)

    # Low
    nu_arr = nulow
    Aeff_over_Tsys_arr = np.interp(
        nu_arr, Aeff_over_Tsys[:, 0], Aeff_over_Tsys[:, 2])
    Tsys_arr = T_sys_low(nu_arr)
    eta_arr = Aeff_over_Tsys_arr * Tsys_arr / ct._area_ska_low_
    SKA_conf['eta low'] = (nu_arr, eta_arr)

    # combined storage
    nu_arr = np.concatenate((SKA_conf['eta low'][0], SKA_conf['eta mid'][0]))
    eta_arr = np.concatenate((SKA_conf['eta low'][1], SKA_conf['eta mid'][1]))
    SKA_conf['eta'] = (nu_arr, eta_arr)

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
        # Tr = ct._Tr_ska_low_ # DEPRECATED
        Tr = Trec_low(nu)
        eta = eta_nu(nu)

        # finding resolution:
        wavelength = pt.lambda_from_nu(nu)/100.  # wavelength [m]
        # angular size of pixel resolution [rad]
        # assuming this is the aperture angle and not the radial angle
        theta_res = (1.22*wavelength) / \
            ct._SKALow_station_diameter_  # /sqrt(eta)
        Omega_res = ct.angle_to_solid_angle(
            theta_res)  # solid angle of resolution [sr]
        number_of_dishes = ct._SKALow_number_of_stations_
        number_of_measurements = number_of_dishes
        # Omega_max = np.inf  # being sloppy here but we never reach FOV

    elif exper_mode == 'SKA low' and correlation_mode == "interferometry":
        window = np.heaviside(nu - ct._nu_min_ska_low_, 1.) * \
            np.heaviside(ct._nu_max_ska_low_ - nu, 1.)
        # Tr = ct._Tr_ska_low_ # DEPRECATED
        Tr = Trec_low(nu)
        eta = eta_nu(nu)

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
        # Tr = ct._Tr_ska_mid_ # DEPRECATED, AND INCONSISTENT
        Tr = Trec_mid(nu)
        eta = eta_nu(nu)

        # finding resolution:
        wavelength = pt.lambda_from_nu(nu)/100.  # wavelength [m]

        # angular size of pixel resolution [rad]
        # assuming this is the aperture angle and not the radial angle
        # theta_res = (1.22*wavelength)/sqrt(eta*4.*area/pi)
        theta_res = (1.22*wavelength)/ct._SKA1Mid_dish_diameter_  # /sqrt(eta)
        Omega_res = ct.angle_to_solid_angle(
            theta_res)  # solid angle of resolution [sr]

        number_of_dishes = ct._SKA1Mid_number_of_dishes_
        number_of_measurements = number_of_dishes
        # Omega_max = np.inf  # being sloppy here but we never reach FOV

    elif exper_mode == 'SKA mid' and correlation_mode == "interferometry":

        area = ct._area_ska_mid_
        window = np.heaviside(nu - ct._nu_min_ska_mid_, 0.) * \
            np.heaviside(ct._nu_max_ska_mid_ - nu, 1.)
        # Tr = ct._Tr_ska_mid_ # DEPRECATED, AND INCONSISTENT
        Tr = Trec_mid(nu)
        eta = eta_nu(nu)

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

    # fix seed as we don't really need the randomness
    np.random.seed(123)
    theta_arr = np.random.random(size=len(r_fine_arr)) * np.pi * 2.

    # over write the arm part
    # mask = np.where(r_fine_arr > r_core, True, False)
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
    # n_baseline = int(n_unit * (n_unit - 1) / 2.)
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


def Trec_mid_MeerKAT(nu):
    """Receiver noise temperature [K] of a MeerKAT dish (13.5m-diameter type)

    :param nu: frequency [GHz]

    """

    nu, is_scalar = tl.treat_as_arr(nu)
    res = []
    for nui in nu:
        if 0.58 < nui < 1.02:
            res.append(11 - 4.5*(nui-0.58))
        elif 1.02 < nui < 1.67:
            res.append(7.5 + 6.8 * np.abs(nui - 1.65)**1.5)
        elif 1.65 < nui < 3.05:
            res.append(7.5)
        else:
            res.append(np.inf)
    if is_scalar:
        res = np.squeeze(res)
    return np.array(res)


def Trec_mid_SKA(nu):
    """Receiver noise temperature [K] of a SKA dish (15m-diameter type)

    :param nu: frequency [GHz]

    """
    nu, is_scalar = tl.treat_as_arr(nu)
    res = []
    for nui in nu:
        if 0.35 < nui < 0.95:
            res.append(15 + 30*(nui-0.75)**2)
        elif 0.95 < nui < 4.6:
            res.append(7.5)
        elif 4.6 < nui < 50:
            res.append(4.4+0.69 * nui)
        else:
            res.append(np.inf)
    if is_scalar:
        res = np.squeeze(res)
    return np.array(res)


def Trec_mid(nu):
    """Receiver noise temperature [K] of a typical SKA1-mid dish. Combines MeerKAT with SKA dishes. If there's only SKA dish, use that one; if there are both, use a weighted mean.

    :param nu: frequency [GHz]

    """

    nu, is_scalar = tl.treat_as_arr(nu)
    Trec_arr = []

    for nui in nu:
        val1 = Trec_mid_MeerKAT(nui)
        val2 = Trec_mid_SKA(nui)
        if np.isinf(val1):
            val1 = val2
        # val = np.sqrt(val1*val2) # NOTE: geometric mean puts them on equal footing, even if there was but a single MeerKAT telescope!!!
        val = (val1*64. + val2*133.)/(133.+64.)  # weighted mean: seems fairer
        Trec_arr.append(val)
    Trec_arr = np.array(Trec_arr)

    if is_scalar:
        Trec_arr = np.squeeze(Trec_arr)

    return Trec_arr


def Trec_low(nu):
    """Receiver noise temperature [K] of a typical SKA1-low dish.

    :param nu: frequency [GHz]

    """

    nu, is_scalar = tl.treat_as_arr(nu)
    Trec_arr = np.ones_like(nu) * ct._Tr_ska_low_

    if is_scalar:
        Trec_arr = np.squeeze(Trec_arr)

    return Trec_arr


def Trec(nu):
    """The receiver noise [K] for both SKA1-Mid and SKA1-Low

    :param nu: frequency [GHz]

    """

    nu, is_scalar = tl.treat_as_arr(nu)
    res = np.zeros_like(nu)
    low_idx = np.where(nu <= ct._nu_max_ska_low_)
    mid_idx = np.where(nu > ct._nu_max_ska_low_)
    res[low_idx] = Trec_low(nu[low_idx])
    res[mid_idx] = Trec_mid(nu[mid_idx])
    if is_scalar:
        res = np.squeeze(res)
    return res


def T_sys_mid(nu):
    """System noise temperature [K] of a single dish for SKA-Mid. It's defined by Treceiver + Tspillover + Tsky, where Tsky = Tcmb + Tgal + Tatm. Note that this function is only used to compute eta from the table of Aeff/Tsys in Braun et al. It is not used in the noise computation.

    :param nu: frequency [GHz]

    """
    nu, is_scalar = tl.treat_as_arr(nu)

    Tsky_arr = np.interp(nu, Tsky_mid[:, 0], Tsky_mid[:, 1])
    Trec_arr = Trec_mid(nu)

    res = ct._T_spill_mid_ + Tsky_arr + Trec_arr

    if is_scalar:
        res = np.squeeze(res)

    return res


def T_sys_low(nu):
    """System noise temperature [K] of a single station SKA-Low. It's defined by Treceiver + Tspillover + Tsky, where Tsky = Tcmb + Tgal + Tatm. Note that this function is only used to compute eta from the table of Aeff/Tsys in Braun et al. It is not used in the real noise computation.
"""

    nu, is_scalar = tl.treat_as_arr(nu)

    Tsky_arr = np.interp(nu, Tsky_low[:, 0], Tsky_low[:, 1])
    Trec_arr = Trec_low(nu)

    res = ct._T_spill_low_ + Tsky_arr + Trec_arr

    if is_scalar:
        res = np.squeeze(res)

    return res

# #
# # efficiency related global vairiables
# # --------------------------------------
# # (nu, eta) arrays for SKA1 low and mid
# nu_eta_low = np.loadtxt(local_path+"/data/eta_low.csv", delimiter=",")
# nu_eta_mid = np.loadtxt(local_path+"/data/eta_mid.csv", delimiter=",")

# # interpolating the data:
# eta_low_fn = tl.interp_fn(nu_eta_low)
# eta_mid_fn = tl.interp_fn(nu_eta_mid)

# # --------------------------------
# # defining the general efficiency

# def eta_nu(nu, exper_mode):
#     """Returns the efficiency eta.

#     nu : frequency [GHz]
#     exper_mode : mode in which the experiment is working
#     """
#     if exper_mode == None:
#         eta = 0.
#     elif exper_mode == 'SKA low':
#         eta = eta_low_fn(nu)
#     elif exper_mode == 'SKA mid':
#         eta = eta_mid_fn(nu)

#     return eta


def eta_nu(nu):
    """Returns the efficiency eta.

    nu : frequency [GHz]
    """
    nu_arr, eta_arr = SKA_conf['eta']
    nu, is_scalar = tl.treat_as_arr(nu)
    res = np.interp(nu, nu_arr, eta_arr)
    if is_scalar:
        res = np.squeeze(res)
    return res


##################
# global variables
##################

# Defining local path
# --------------------
local_path = os.path.dirname(os.path.abspath(__file__))

# load the sky temperature (Tsky=Tcmb + Tgal + Tatm)
# from Braun et al. 2017
# and SKA-TEL-SKO-0000308_SKA1_System_Baseline_v2_DescriptionRev01-part-1-signed.
# Note that this Tsky is used for two things
# 1. for computing eta
# 2. for extracting Tatm
# -------------------------

Tsky_mid = np.loadtxt(local_path+"/data/Tsky_mid.csv", delimiter=',')
Tsky_low = np.loadtxt(local_path+"/data/Tsky_low.csv", delimiter=',')

# The global variable of SKA
# configuration saved into SKA_conf
# ---------------------------------
SKA_conf = initialize()
