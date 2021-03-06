"""
This is a module for the contants.
Requires MyUnit.py
"""

from __future__ import division
from MyUnit import NaturalUnit
import numpy as np
import os

###############################################################
# functions converting between (diameter) angle and solid angle
###############################################################


def angle_to_solid_angle(theta):
    """convert the angle to solid angle

    :param theta: angle that the object subtends [radian]
    :returns: solid angle of the object [sr]

    """

    Omega = 2.*np.pi * (1.-np.cos(theta/2.))
    return Omega


def solid_angle_to_angle(Omega):
    """convert solid angle to angle

    :param Omega: soild angle that the object subtends [sr]
    :returns: angle of the object [radian]

    """

    theta = 2. * np.arccos(1 - Omega/(2.*np.pi))
    return theta


# TODO: remember to harden the numbers to speed up the code


################
# useful numbers
################

# to avoid division by zero:
_zero_ = 1e-100

# a huge number:
_huge_ = 1.e100


###################
# Constants & Units
###################

# ------------------
# Constants:

_light_speed_ = 299792458.  # [m/s]
_k_B_ = NaturalUnit('K/J').val  # Boltzmann constant 1.3806503e-23 [J/K]
_h_ = 2.*np.pi  # Planck's constant in natural unit

# ----------------
# Unit conversion

_year_over_s_ = NaturalUnit('year/s').val  # 31557600
_GeV_over_g_ = 1000.*NaturalUnit('GeV/kg').val  # 5.6095888453893324e+23
_eV_over_GeV_ = 1.e-9
_cm_eV_ = NaturalUnit('cm*eV').val
_kpc_over_cm_ = NaturalUnit('kpc/cm').val  # 3.0857e21
_kpc_over_m_ = NaturalUnit('kpc/m').val  # 3.0857e19
_kpc_eV_ = NaturalUnit('kpc*eV').val
_kpc_over_pc_ = 1000.
_kpc_over_lightyear_ = NaturalUnit('kpc/year').val
# _GHz_over_eV_ = (NaturalUnit('1.e9/s/eV').val * 2.*np.pi)  # 4.13566773306e-06 NOTE: the crucial factor of 2*pi coming from Planck's Law: E = h*nu = hbar * 2pi * nu, in order to convert a photon's frequency into its energy
# note that this is a pure unit conversion. It does not convert frequency to energy. Remember h=2pi when converting to energy.
_GHz_over_eV_ = (NaturalUnit('1.e9/s/eV').val)
_Jy_over_SFU_ = 1.e-4
_Jy_over_eV3_ = (NaturalUnit('1e-26*J/s/m**2*s/eV**3').val /
                 (2.*np.pi))  # 3.867966373282915e-22
_SFU_over_eV3_ = _Jy_over_eV3_ / _Jy_over_SFU_
_Jy_over_cgs_irrad_ = 1.e-23  # [Jy] -> [erg * s^-1 * cm^-2 * Hz^-1]
_Jy_over_SI_ = 1.e-26  # [Jy] -> [W * m^-2 * Hz^-1]
_au_over_kpc_ = NaturalUnit('au/kpc').val  # 4.8481e-9
_K_over_eV_ = NaturalUnit('K/eV').val
_s_eV_ = NaturalUnit('s*eV').val  # 1519267407871140
_J_over_eV_ = NaturalUnit('J/eV').val  # 6.24150934326018e+18
_hour_eV_ = NaturalUnit('60.*60.*s*eV').val  # 5.4693626683361e+18
_m_eV_ = NaturalUnit('m*eV').val  # 5067730.58270578
_arcmin_over_radian_ = 1./60.*np.pi/180

##################
# Astro quantities
##################

# --------------------
# Galactic quantities

_Sun_to_gal_center_ = 8.1  # [kpc]
_rho_local_DM_ = 0.4  # [GeV/cm**3]
_sigma_v_ = 0.000519026  # [c] 155.6 km/s # DM velocity dispersion
# NFW profile
_rs_of_MW_ = 20.  # [kpc]
# Burkert profile 1304.5127
_rho_H_ = 1.568  # [GeV/cm**3]
_r_H_ = 9.26  # [kpc]
# _r_H_ = 4.  # [kpc]
_MW_spectral_beta_ = -2.55  # 2015 SKA design report v2
# _MW_spectral_beta_ = -2.75 # from Braun et al. 2019

# -----------------------------------------
# Properties of some astrophysical objects

# Sun:
_solar_solid_angle_ = NaturalUnit('3.14159*Rsun**2/au**2').val  # 6.8001e-5
_solar_solid_angle_G_ = NaturalUnit('3.14159*Rsun**2/(8.1*kpc)**2').val

# Cassiopeia A:
_casA_angle_ = 5.  # [arcmin]
_casA_solid_angle_ = angle_to_solid_angle(
    _casA_angle_ * _arcmin_over_radian_)  # 1.66e-06 [sr]

# Cygnus A:
_cygA_angle_ = 3.  # [arcmin]
_cygA_solid_angle_ = angle_to_solid_angle(
    _cygA_angle_ * _arcmin_over_radian_)  # 5.98e-7[sr]
_cygA_theta_ = 76.26 * np.pi/180.  # [radian]
_cygA_bg_T_at_408Hz_ = 27.  # [K]

######################
# Noitse temperatures
######################
# background temperatures
_Tcmb_ = 2.7255  # cmb brightness [K]
_Tatm_ = 3.  # atmospheric brightness for SKA1-mid freq range [K]

# -------------------
# SKA receiver noise

_Tr_ska_low_ = 40.  # [K]
# _Tr_ska_mid_ = 20.  # [K] # NOTE: should not be used, since we compute it on the fly, for consistency
# _Tr_ska_mid_avg_ = 10.55 # [K] average value of computer SKA1-mid Tr over the mid frequency band

_T_spill_mid_ = 3.  # [K]
_T_spill_low_ = 0.  # [K]

##############################
# SN fits from Bietenholz 2021
##############################

# -------
# L_peak

_mu_log10_Lpk_, _sig_log10_Lpk_ = 25.5, 1.5  # all types, D<100Mpc
_mu_log10_Lpk_50_, _sig_log10_Lpk_50_ = 25.5, 1.5  # all types, D<50Mpc
_mu_log10_Lpk_IIb_, _sig_log10_Lpk_IIb_ = 26.8, 0.5  # type IIb

# -------
# t_peak

_mu_log10_tpk_, _sig_log10_tpk_ = 1.7, 0.9  # all types, D<100Mpc
_mu_log10_tpk_50_, _sig_log10_tpk_50_ = 1.6, 0.8  # all types, D<50Mpc
_mu_log10_tpk_IIb_, _sig_log10_tpk_IIb_ = 1.5, 0.6  # type IIb

###############
# SN properties
###############

_time_of_phase_two_ = 1.e4  # [year]
_v_hom_ = 2.e7 / _light_speed_  # = 0.0667128 [c] speed of homologous expansion
# approximate speed during free expansion, according to TM99 (for M_ej = 1 Msun, E_sn = 1.e51 erg, and rho0 = proton_mass/cm^3) 0.0414252 [c]
_v_TM99_ = ((0.7*3.07*_kpc_over_m_/1000.) /
            (0.4*423.*_year_over_s_) / _light_speed_)


######################################
# Optimal bandwidth
# N.B.: disagreement with Ghosh et al.
######################################

# bandwidth of top-hat profile, in units of standard deviations, which yields optimal S/N ratio (if S is Gaussian distributed over frequency and N scales like the square root of the frequency bandwidth)

_tophat_width_ = 2.79997
_f_Delta_ = 0.83848  # fraction of Gaussian area contained within that tophat width
_deltaE_over_E_ = 0.00145326  # _tophat_width_ * _sigma_v_

##################
# Experiment specs
##################

# ------------------
# SKA aperture area

# _area_ska_low_ = 419000.  # [m^2] Ghosh et al
# _area_ska_mid_ = 1.e6  # [m^2] Ghosh et al

# area of SKA-low:
# num_of_stations * pi * D**2 / 4
# num_of_stations: 512
# station effective size: D = 38 m (summary v4, instead of 35 m)
_area_ska_low_ = 580667.  # [m^2]

# area of SKA1-mid:
# pi/4 * (133 * 15**2 + 64 * 13.5**2)
# 133 SKA 15m-dishes, 64 MeerKAT 13.5-dishes
_area_ska_mid_ = 32663.9  # [m^2]

# ----------------
# SKA frequencies

_band_width_ska_low_ = 0.3  # [GHz]
_band_width_ska_mid_ = 15.05  # [GHz]

_nu_min_ska_low_ = 0.05  # [GHz]
_nu_max_ska_low_ = 0.35  # [GHz]

_nu_min_ska_mid_ = 0.35  # [GHz] # both SKA1/2
_nu_max_ska_mid_ = 15.4  # [GHz] # this is SKA1-mid
_nu_max_ska2_mid_ = 30.  # [GHz] # this is SKA2-mid c.f. Caputo


# ------------------
# SKA beam geometry

_SKALow_station_diameter_ = 38.  # [m]
_SKA1Mid_dish_diameter_ = 14.53  # [m] weighted average

# ------------------------
# SKA array configuration


_SKA1Mid_maximal_baseline_ = 150000.  # [m]
_SKA1Mid_number_of_dishes_ = (133. + 64.)
_SKA1Mid_total_baselines_ = _SKA1Mid_number_of_dishes_ * \
    (_SKA1Mid_number_of_dishes_-1.) / 2.
_SKA1Mid_r_core_ = 1000  # [m]

_SKALow_maximal_baseline_ = 80000.  # [m]
_SKALow_number_of_stations_ = 512.
_SKALow_total_baselines_ = _SKALow_number_of_stations_ * \
    (_SKALow_number_of_stations_ - 1.) / 2.
_SKALow_r_core_ = 300  # [m]

# ----------------------------------------------------------------------------
# the value of theta_sig/theta_b at which the given baseline loses the signal
# fudge factor */ 2
_SKA_factor_lose_signal_ = 1.
