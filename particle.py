"""
This is a module for the basics of the particle physics
"""

from __future__ import division
import numpy as np
import constants as ct
import tools as tl

# decay constant


def Gamma(ma, ga):
    """
    The decay width [eV] of axions to two photons.

    Parameters
    ----------
    ma : axion mass [eV]
    ga : axion-photon coupling [GeV^-1]
    """
    res = (ga * ct._eV_over_GeV_)**2 * ma**3. / 64. / np.pi
    return res


# mass and frequency
def nu_from_ma(ma):
    """
    Frequency of light [GHz] from axion of mass ma [eV].

    Parameters
    ----------
    ma : axion mass [eV]
    """

    ma, is_scalar = tl.treat_as_arr(ma)
    E = ma/2.
    res = E/ct._GHz_over_eV_/ct._h_
    if is_scalar:
        res = np.squeeze(res)
    return res


def ma_from_nu(nu):
    """
    Axion mass ma [eV] from frequency of light [GHz].

    Parameters
    ----------
    nu : photon frequency [GHz]
    """
    nu, is_scalar = tl.treat_as_arr(nu)

    E = ct._h_ * nu * ct._GHz_over_eV_
    res = 2*E

    if is_scalar:
        res = np.squeeze(res)

    return res


# wavelength
def lambda_from_nu(nu):
    """
    Wavelength [cm] from frequency [GHz].

    Parameters
    ----------
    nu: photon frequency [GHz]
    """

    period = 1./(nu*1.e9)  # [sec]
    wl = 100.*ct._light_speed_*period  # [cm]

    return wl

# p.s.d. of axion DM


def fa(ma, va, rho_a=ct._rho_local_DM_):
    """Compute the phase space distribution of axion DM

    :param ma: axion mass [eV]
    :param va: DM velocity [c]
    :param rho_a: DM energy density [GeV/cm**3]

    """
    va, is_scalar = tl.treat_as_arr(va)
    factor = rho_a * 1.e9 / ct._cm_eV_**3 / \
        ma**4 * (2.*np.pi)**1.5 / ct._sigma_v_**3
    exponent = - va**2/2./ct._sigma_v_**2
    res = factor * np.exp(exponent)
    if is_scalar:
        res = np.squeeze(res)
    return res
