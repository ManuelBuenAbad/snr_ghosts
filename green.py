from __future__ import division
import numpy as np
from numpy import pi, sqrt, log, log10, power, exp
from scipy.interpolate import interp1d, interp2d
import os

# current directory
current_dir = os.getcwd()

import constants as ct
import particle as pt
import astro as ap
import echo as ec
import data as dt

# -------------------------------------------------

###############
# DIRECTORIES #
###############

green_path = os.path.dirname(os.path.abspath(__file__))+"/output/green_snr/"



# -------------------------------------------------

##########
# ARRAYS #
##########

# loading pre-computed arrays
pre_Lpk_arr = np.loadtxt(green_path+"Lpk_arr.txt", delimiter=",")
pre_tpk_arr = np.loadtxt(green_path+"tpk_arr.txt", delimiter=",")

# peak luminosity and time grids
pre_Lpk_Gr, pre_tpk_Gr = np.meshgrid(pre_Lpk_arr, pre_tpk_arr, indexing='xy')

# normal (0, 1) variables from pre-computed arrays:
normal_Lpk_arr = (log10(pre_Lpk_arr)-ct._mu_log10_Lpk_)/ct._sig_log10_Lpk_
normal_tpk_arr = (log10(pre_tpk_arr)-ct._mu_log10_tpk_)/ct._sig_log10_tpk_

# sigmas contours on pre-computed grids
sigs_Gr = np.sqrt(((log10(pre_Lpk_Gr)-ct._mu_log10_Lpk_)/ct._sig_log10_Lpk_)**2 + ((log10(pre_tpk_Gr)-ct._mu_log10_tpk_)/ct._sig_log10_tpk_)**2)

# -------------------------------------------------

###############
# SNR CATALOG #
###############

# Loading Green's catalog:
# First let's parse snrs.list.html
# Names:
snr_name_arr = dt.snr_name_arr
# Catalog:
snrs_dct = dt.snrs_dct
snrs_cut = dt.snrs_cut
snrs_age = dt.snrs_age

# -------------------------------------------------

#############
# FUNCTIONS #
#############

# a small number, but not too small for it to cause overflows when manipulated
very_small = 1.e-100


def load_green_results(name, r=None, tex=0., nuB=1.):
    """
    Function that loads the CCV numerical results for the SNRs from Green's Catalog.
    """
    
    if not name in snrs_cut.keys():
        raise ValueError("name={} not available in results.".format(name))
    
    if (r == None) and (not name in snrs_age.keys()):
        raise ValueError("name={} not available in SNR results of known age.".format(name))
    
    if r == None:
        r_str = "_wage"
    else:
        r_str = "_r-{}".format(int(r))
    
    tex_str = "_tex-{}".format(int(tex))
    nuB_str = "_nuB-{}".format(int(nuB))
    
    folder = green_path+name+"/"
    file = "{}_Lpk-tpk{}{}{}.txt".format(name, r_str, tex_str, nuB_str)
    
    sn = np.loadtxt(folder+"sn_"+file, delimiter=",")
    echo = np.loadtxt(folder+"echo_"+file, delimiter=",")
    
    if r != None:
        tgrid = np.loadtxt(folder+"tage_"+file, delimiter=",")
    else:
        tgrid = np.loadtxt(folder+"ttrans_"+file, delimiter=",")
        
    return sn, echo, tgrid



def snr_reach(name, r=None, nuB=1., tex=0., sn_ratio_threshold=1., nu_pivot=1., ga_ref=1.e-10, full_output=False):
    """
    Returns an interpolated function of the discovery reach of the axion-photon coupling ga [GeV^-1] as a function of the Bietenholz parameters (be that in terms of normalized (0, 1) variables (variables='normal'), or in terms of the raw parameters t_peak and L_peak themselves) for a certain signal-to-noise ratio and SNR name. If full_output == True, it also returns other important quantities.
    """
    
    if (not name in snrs_cut.keys()):
        raise ValueError("name={} not available in results.".format(name))
    
    if (r == None) and (not name in snrs_age.keys()):
        raise ValueError("name={} not available in SNR results of known age.".format(name))
    
    snr = snrs_cut[name]
    # SNR results
    sn_Gr, _, tgrid = load_green_results(name, r=r, tex=tex, nuB=nuB)
    # N.B.: tgrid is t_trans if r==None, and t_age if r!= None
    
    # SNR properties:
    alpha = snr.alpha
    gamma = ap.gamma_from_alpha(alpha)
    S0 = snr.get_flux_density() # [Jy] spectral irrad. today
    L0 = snr.get_luminosity() # [cgs]
    size = snr.sr

    # going from Bietenholz peak Luminosities to pivot luminosities (at 1 GHz):
    # conversion factor:
    from_Bieten_to_pivot = (nu_pivot/nuB)**-alpha
    # copying arrays and grids:
    Lpk_arr, tpk_arr = np.copy(pre_Lpk_arr), np.copy(pre_tpk_arr)
    Lpk_Gr, tpk_Gr = np.copy(pre_Lpk_Gr), np.copy(pre_tpk_Gr)
    # correcting with conversion factor:
    Lpk_arr *= from_Bieten_to_pivot
    Lpk_Gr *= from_Bieten_to_pivot
    
    # normal_Lpk value where the cut takes place
    normal_Lpk_cut = (log10(L0/from_Bieten_to_pivot)-ct._mu_log10_Lpk_)/ct._sig_log10_Lpk_
    
    # computing the forbidden parameter space region
    nonsense_lum = (L0 >= Lpk_Gr).astype(int) # points where L0 >= Lpk
    if r == None:
#         nonsense_time = np.zeros_like(tpk_Gr) # TODO: change?
        nonsense_time = (tgrid < (tpk_Gr/365.)).astype(int) # t_trans < tpk
    else:
        nonsense_time = np.zeros_like(tpk_Gr) # TODO: change?
#         tt_Gr = r*(tpk_Gr/365.)
        nonsense_time = (tgrid < (tpk_Gr/365.)).astype(int) # t_age < tpk
    
    nonsense_params = np.logical_or(nonsense_lum, nonsense_time).astype(int)
    
    regularized_sn_Gr = np.where(sn_Gr < very_small, very_small, sn_Gr) # converting 0s to a small number
    
    ga_Gr = ec.ga_reach(sn_ratio_threshold, regularized_sn_Gr, ga_ref)
    ga_Gr = np.nan_to_num(ga_Gr)
    
    def ga_fn(nL, nt): return 10.**interp2d(normal_Lpk_arr, normal_tpk_arr, log10(ga_Gr))(nL, nt)
    
    if full_output:
        return ga_fn, ga_Gr, normal_Lpk_arr, normal_tpk_arr, nonsense_params
    else:
        return ga_fn