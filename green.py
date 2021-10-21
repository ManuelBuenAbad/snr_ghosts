from __future__ import division
import numpy as np
from numpy import pi, sqrt, log, log10, power, exp
from scipy.interpolate import interp1d, interp2d
import os

# current directory
current_dir = os.getcwd()

import tools as tl
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
try:

    Lpk_arr = np.loadtxt(green_path+"Lpk_arr.txt", delimiter=",")
    tpk_arr = np.loadtxt(green_path+"tpk_arr.txt", delimiter=",")

    # peak luminosity and time grids
    Lpk_Gr, tpk_Gr = np.meshgrid(Lpk_arr, tpk_arr, indexing='xy')

    # normal (0, 1) variables from pre-computed arrays:
    normal_Lpk_arr = (log10(Lpk_arr)-ct._mu_log10_Lpk_)/ct._sig_log10_Lpk_
    normal_tpk_arr = (log10(tpk_arr)-ct._mu_log10_tpk_)/ct._sig_log10_tpk_

    # sigmas contours on pre-computed grids
    sigs_Gr = np.sqrt(((log10(Lpk_Gr)-ct._mu_log10_Lpk_)/ct._sig_log10_Lpk_)**2 + ((log10(tpk_Gr)-ct._mu_log10_tpk_)/ct._sig_log10_tpk_)**2)

except:
    Lpk_arr, tpk_arr = None, None

try:
    ma_arr = np.loadtxt(green_path+"ma_arr.txt", delimiter=",")
except:
    ma_arr = None

try:
    tpk_arr = np.loadtxt(green_path+"tpk_arr.txt", delimiter=",")
    ttr_arr = np.loadtxt(green_path+"ttr_arr.txt", delimiter=",")
except:
    pass

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
snrs_age_only = dt.snrs_age_only

# -------------------------------------------------

#############
# FUNCTIONS #
#############

# a small number, but not too small for it to cause overflows when manipulated
very_small = 1.e-100


def load_green_results(name, run_id=None):
    """
    Function that loads the numerical results for the SNRs from Green's Catalog.
    """

    if not name in snrs_cut.keys():
        raise ValueError("name={} not available in results.".format(name))

    # loading lines of log file
    log_file = "run_%d_log.txt" % run_id
    with open(green_path+log_file, 'r') as log_info:
        log_lines = [line.rstrip('\n') for line in log_info]

    # looking in the log what the parameter slice was:
    slice_idx = [('slice:' in line) for line in log_lines].index(True)
    slice = log_lines[slice_idx].split()[-1]

    # file paths:
    folder = green_path+name+"/"
    file = name+"_run-"+str(run_id)+".txt"

    # loading parameters
    if slice == "ma-ga":
        params = ma_arr
    elif slice == "Lpk-tpk":
        params = (Lpk_arr, tpk_arr)
    elif slice == "ttr-tpk":
        params = (ttr_arr, tpk_arr)

    # loading results:
    results = {}

    results['echo'] = np.loadtxt(folder+"echo_"+file, delimiter=",")
    results['sn'] = np.loadtxt(folder+"sn_"+file, delimiter=",")
    results['ga'] = np.loadtxt(folder+"ga_"+file, delimiter=",")
    results['tage'] = np.loadtxt(folder+"tage_"+file, delimiter=",")

    try:
        results['ttrans'] = np.loadtxt(folder+"ttrans_"+file, delimiter=",")
    except:
        pass

    try:
        results['Lpk'] = np.loadtxt(folder+"Lpk_"+file, delimiter=",")
    except:
        pass

    return log_lines, params, results
