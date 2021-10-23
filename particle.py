"""
This is a module for the basics of the particle physics
"""

from __future__ import division
import numpy as np
import constants as ct


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

    E = ma/2.
    return E/ct._GHz_over_eV_/ct._h_


def ma_from_nu(nu):
    """
    Axion mass ma [eV] from frequency of light [GHz].

    Parameters
    ----------
    nu : photon frequency [GHz]
    """

    E = ct._h_ * nu * ct._GHz_over_eV_

    return 2*E


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
