from __future__ import division
import numpy as np
from numpy import pi, sqrt, log, log10, power, exp
from scipy.interpolate import interp1d
from tqdm import tqdm
import os
import argparse

# current directory
current_dir = os.getcwd()

import constants as ct
import particle as pt
import astro as ap
import echo as ec
import routines as rt
import data as dt
import model as md

# -------------------------------------------------

###############
# DIRECTORIES #
###############

# Making directories:
try:
    os.makedirs("./output/green_snr/")
except:
    pass


# -------------------------------------------------

#############
# ARGUMENTS #
#############

# Defining arguments
parser = argparse.ArgumentParser(description="Computes the reach for the SNR in the Green's Catalog")

# Arguments with numerical values:
parser.add_argument('-N', '--Nsteps', default=None,
                    type=int, help="The number of steps in the parameter space arrays (default: None)")
parser.add_argument('-r', '--tt_ratio', '--ratio', default=None,
                    type=float, help="The ratio of t_trans/t_pk (default: None)")
parser.add_argument('-x', '--t_extra', '--extra', default=None,
                    type=float, help="The extra age of the SNR, after the adiabatic phase [years] (default: None)")
parser.add_argument('-n', '--nuB', '--nu_Bietenholz', default=None,
                    type=float, help="The Bietenholz frequency [GHz] (default: None)")
# True/False flags
parser.add_argument('-a', '--known_age', action='store_true', help="Whether the age of the SNR is known (default: False)")
parser.add_argument('-v', '--verbose', action='store_true', help="Verbosity (default: False)")
    
# Parsing arguments:
args = parser.parse_args()
    
# Defining appropriate variables
Nsteps = args.Nsteps
tt_ratio = args.tt_ratio
t_extra = args.t_extra
nuB = args.nuB

flg_r = bool(args.tt_ratio)
known_age = args.known_age
verbose = args.verbose

# Making sure the parametere have the right values
if flg_r and known_age:
    raise Exception("Cannot pass both --tt_ratio and --known_age. Pick one.")

if t_extra == None or nuB == None or Nsteps == None:
    raise Exception("Pass a value for --Nsteps, for --t_extra, and for --nuB.")

if (not known_age) and (not flg_r):
    raise Exception("Pass a value for --tt_ratio or simply pass --known_age.")
else:
    pass

# file header
if flg_r:
    header = "_Lpk-tpk_r-{}_tex-{}_nuB-{}.txt".format(int(tt_ratio), int(t_extra), int(nuB))
elif known_age:
    header = "_Lpk-tpk_wage_tex-{}_nuB-{}.txt".format(int(t_extra), int(nuB))

if verbose:
    print("tt_ratio={}, t_extra={}, nuB={}, known_age={}".format(tt_ratio, t_extra, nuB, known_age))
    print(header)

# -------------------------------------------------

##########
# ARRAYS #
##########

# SNR early-time evolution: from Bietenholz et al., Table 4.
# from quantities in constants.py
# tpk and Lpk arrays:
Nsigs = 3. # number of standard deviations from the Bietenholz's mean to scan
tpk_arr = np.logspace(ct._mu_log10_tpk_-Nsigs*ct._sig_log10_tpk_, ct._mu_log10_tpk_+Nsigs*ct._sig_log10_tpk_, Nsteps+1)
Lpk_arr = np.logspace(ct._mu_log10_Lpk_-Nsigs*ct._sig_log10_Lpk_, ct._mu_log10_Lpk_+Nsigs*ct._sig_log10_Lpk_, Nsteps+2)

# Saving arrays
if os.access("./output/green_snr/tpk_arr.txt", os.R_OK):
    pass
else:
    np.savetxt("./output/green_snr/tpk_arr.txt", tpk_arr)

if os.access("./output/green_snr/Lpk_arr.txt", os.R_OK):
    pass
else:
    np.savetxt("./output/green_snr/Lpk_arr.txt", Lpk_arr)


# -------------------------------------------------

###############
# SNR CATALOG #
###############

# Loading Green's catalog:
# First let's parse snrs.list.html
# Names:
snr_name_arr = dt.load_Green_catalogue_names()
# Catalog:
snrs_dct = dt.load_Green_catalogue(snr_name_arr)

# Curating Green's catalog
snrs_cut = {}
for name, snr in snrs_dct.items():
    
    try: snr.distance
    except: continue
    
    try: snr.alpha
    except: continue
    
    if known_age:
        try: snr.age
        except: continue
    else: pass
    
    if snr.get_flux_density() == -1:
        print("no flux density: "+str(name))
        continue
    
    if not snr.is_flux_certain:
        print("uncertain flux: "+str(name))
        continue
    
    snrs_cut[name] = snr
    # Creating SNR directories:
    try:
        os.makedirs("./output/green_snr/"+name+"/")
    except:
        pass

print("Total no. of SNRs: "+str(len(snrs_cut))+"\n")

# -------------------------------------------------

###########
# ROUTINE #
###########

# Axion fixed params:
ga_ref = 1.e-10 # [GeV^-1]

# Results dictionaries:
sn_results = {}
snu_results = {}
if not known_age:
    age_results = {}

counter = 0

sorted_names = snrs_cut.keys()
sorted_names.sort()
for i, name in tqdm(enumerate(sorted_names)):
    snr = snrs_cut[name]
# for name, snr in snrs_cut.items():
    
    file_name = name+header # name of file
    
    if verbose:
        print(name)
    
    # Reading some important SNR properties:
    gamma = ap.gamma_from_alpha(snr.alpha) # Sedov-Taylor analytic formula
    area = 4.*pi*(snr.distance*ct._kpc_over_cm_)**2. # [cm^2]
    L0 = (snr.snu_at_1GHz*ct._Jy_over_cgs_irrad_)*area # [cgs]
    from_Bieten_to_pivot = (1./nuB)**-snr.alpha # correction from the fact that the Bietenholz frequency is not the pivot frequency [1 GHz]
    new_Lpk_arr = np.copy(Lpk_arr) # copying peak luminosity array
    new_Lpk_arr *= from_Bieten_to_pivot  # correcting L_peak by switching from the Bietenholz to the pivot frequencies
    
    if known_age:
        t_age = snr.age
    
    # preparing the arrays to be filled:
    sn_results[name] = []
    snu_results[name] = []
    if not known_age:
        age_results[name] = []
    
    # start!
    for tpk in tpk_arr:
        
        row_a = []
        row_b = []
        if not known_age:
            row_c = []
        
        for Lpk in new_Lpk_arr:
            
            if L0 >= Lpk: # sensible luminosities
                continue
            
            lightcurve_params = {'t_peak': tpk,
                                 'L_peak': Lpk,
                                 'L_today': L0
                                }
            if flg_r:
                t_trans = tt_ratio*(tpk/365.)
                lightcurve_params.update({'t_trans': t_trans})
                # Computing t_age
                t_age = ap.tage_compute(Lpk, tpk, t_trans, L0, gamma)
                
                if t_age < t_trans: # sensible t_trans (i.e. tpk)
                    continue
            else:
                lightcurve_params.update({'t_age': t_age})
            
            if verbose:
                print(t_age)
            
            # Snu kwargs
            max_steps = int(3*(t_age) + 1)
            snu_echo_kwargs = {'tmin_default': None,
                               'Nt': min(max_steps, 100001),
                               'xmin': ct._au_over_kpc_,
                               'xmax_default': 100.,
                               'use_quad': False,
                               'lin_space': False,
                               'Nint': min(max_steps, 100001),
                               't_extra_old': t_extra
                              }
            
            # data:
            data = {'deltaE_over_E': 1.e-3,
                    'f_Delta': 0.721,
                    'exper': 'SKA',
                    'total_observing_time': 100.,
                    'verbose': 0,
                    'DM_profile': 'NFW',
                    'average': True
                   }
            
            # computing routine
            z, new_output = md.snr_routine(pt.ma_from_nu(1.), ga_ref,
                                           snr,
                                           lightcurve_params=lightcurve_params,
                                           snu_echo_kwargs=snu_echo_kwargs,
                                           data=data,
                                           output_all=True)
            
            signal_Snu = new_output['signal_Snu']
            del new_output
            
            if verbose:
                print("S/N= "+str(z))
            
            # building rows
            row_a.append(z) # signal-to-noise ratio
            row_b.append(signal_Snu) # signal S_nu
            if not known_age:
                row_c.append(t_age) # t_age
            
            # end of routine for fixed Lpk
        
        # appending finished Lpk rows
        sn_results[name].append(row_a)
        snu_results[name].append(row_b)
        if not known_age:
            age_results[name].append(row_c)
        
        # end of routine for fixed tpk
    
    # converting grids to arrays
    sn_results[name] = np.array(sn_results[name])
    snu_results[name] = np.array(snu_results[name])
    if not known_age:
        age_results[name] = np.array(age_results[name])
    
    # saving grids
    np.savetxt("./output/green_snr/"+name+"/sn_"+file_name, sn_results[name], delimiter=",")
    np.savetxt("./output/green_snr/"+name+"/echo_"+file_name, snu_results[name], delimiter=",")
    if not known_age:
        np.savetxt("./output/green_snr/"+name+"/tage_"+file_name, age_results[name], delimiter=",")
    
    counter += 1
    
    # end of routine for fixed snr

print(counter)
