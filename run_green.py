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
import data as dt
import model as md

# -------------------------------------------------

###############
# DIRECTORIES #
###############

# Making directories:

folder = os.path.dirname(os.path.abspath(__file__))+"/output/green_snr/"
try:
    os.makedirs(folder)
except:
    pass


# -------------------------------------------------

#############
# ARGUMENTS #
#############

# Defining arguments
parser = argparse.ArgumentParser(description="Computes the reach for the SNR in the Green's Catalog, for a specific approach to their lightcurve and their age.")

# Arguments with numerical values:
parser.add_argument("-i", "--run_id", "--run", "--id", default=0,
                    type=int, help="The run ID number (default: 0)")
parser.add_argument("-x", "--t_extra", "--extra", default=0.,
                    type=float, help="The extra age of the SNR, after the adiabatic phase [years] (default: None)")
parser.add_argument("-c", "--correlation_mode", "--correl", default=None,
                    type=str, choices=["single dish", "interferometry"], help="The running mode of SKA.")
parser.add_argument("-a", "--age_mode", default=None,
                    type=str, choices=["known_age", "size_age", "ratio_age"], help="The way in which the age of the SNR will be treated.")
# arguments for args.age_mode == "size_age":
parser.add_argument('-m', '--method', default=None,
                    choices=['TM99-0', 'TM99-simple', 'estimate', 'lin', 'log'],
                    type=str, help="The method to compute the age. Based on either Truelove-McKee '99 (TM99; for n=0 ejecta or a simplified version), a quick-and-dirty estimate, or a linear regression performed on the (linear/log) data of SNR with known age (default: None).")
parser.add_argument('--M_ej', default=1.,
                    type=float, help="The mass [Msun] of the SNR ejecta (default: 1). Only relevant if method=='TM99-0'/'TM99-simple'.")
parser.add_argument('--E_sn', default=1.,
                    type=float, help="The energy output [1.e51 ergs] of the SNR (default: 1). Only relevant if method=='TM99-0'/'TM99-simple'.")
parser.add_argument('--rho0', default=1.,
                    type=float, help="The density [proton mass/cm^3] of the interstellar medium surrounding the SNR (default: 1). Only relevant if method=='TM99-0'/'TM99-simple'.")
# arguments for args.age_mode == "ratio_age":
parser.add_argument("-r", "--tt_ratio", "--ratio", default=None,
                    type=float, help="The ratio of t_trans/t_pk (default: None)")

# verbosity
parser.add_argument("-v", "--verbose", action="store_true", help="Verbosity (default: False)")

# Defining the subparsers and sending their names to .lightcurve attributes
subparsers = parser.add_subparsers(dest="lightcurve",
                    help="Subcommand options determining the way in which the lightcurve of the SNR will be treated.")


#######################
# Lightcurve subparsers
#######################
#--------------------
# CASE 1: Adiabatic expansion-only lightcurve
adiab = subparsers.add_parser("adiabatic_only", help="Only the adiabatic expansion part of the lightcurve will be used in our computations.")
adiab.add_argument('--t_trans', default=None,
                    type=float, help="The duration [years] of the free expansion phase (default: None).")
adiab.add_argument("-N", "--Nsteps", default=None,
                    type=int, help="The number of steps in the axion mass parameter space arrays (default: None)")

# N.B. FOR DEBUGGING PURPOSES ONLY:
adiab.add_argument("-k", "--t_peak", default=10**(ct._mu_log10_tpk_),
                    type=float, help="N.B. FOR DEBUGGING PURPOSES ONLY: The time [days] of peak SNR luminosity. Since the free expansion will be ignored, its precise value is irrelevant (default: 10^1.7).")

#--------------------
# CASE 2: Free+Adiabatic expansions lightcurve
free_adiab = subparsers.add_parser("free+adiabatic", help="Both the free expansion and adiabatic expansion parts of the lightcurve will be used in our computations.")
free_adiab.add_argument("-N", "--Nsteps", default=None,
                        type=int, help="The number of steps in the Lpk-tpk parameter space arrays (default: None)")
free_adiab.add_argument("-n", "--nuB", "--nu_Bietenholz", default=None,
                        type=float, help="The Bietenholz frequency [GHz] (default: None)")
free_adiab.add_argument("-f", "--fixed_free", "--fixed_Lpk-tpk",
                        action="store_true", help="Whether we fix the free expansion parameters L_peak & t_peak, as opposed to scanning them (default: False).")
free_adiab.add_argument("-L", "--L_peak", default=None,
                        type=float, help="The peak luminosity [erg/s/Hz] (default: None)")
free_adiab.add_argument("-k", "--t_peak", default=None,
                        type=float, help="The time [days] of peak SNR luminosity (default: None).")


# Parsing arguments:
args = parser.parse_args()

# Defining appropriate variables
run_id = args.run_id
correlation_mode = args.correlation_mode
t_extra = args.t_extra
verbose = args.verbose

# Checking basic values:
if correlation_mode == None:
    raise Exception("Pass a value for --correlation_mode.")
if t_extra == None:
    raise Exception("Pass a value for --t_extra.")

# Checking age_mode values
age_is_known = False # will need to compute age: either from R or from size
if args.age_mode == "known_age":
    age_is_known = True # since age is known, no need to compute it
elif args.age_mode == "size_age":
    if not args.method in ['TM99-0', 'TM99-simple', 'estimate', 'lin', 'log']:
        raise ValueError("Please make sure the argument --method is one of the allowed options.")
elif args.age_mode == "ratio_age":
    if args.tt_ratio == None:
        raise ValueError("Pass a value for --tt_ratio.")
    tt_ratio = args.tt_ratio
else:
    pass

# inconsistent choice:
if args.age_mode == "ratio_age" and args.lightcurve == "adiabatic_only":
    raise ValueError("If the age is to be deduced from t_trans/t_peak, then we need the full 'free+adiabatic' lightcurve evolution.")
else:
    pass

# Checking lightcurve values:
include_free = False

if args.lightcurve == "adiabatic_only":

    if args.t_trans == None:
        raise ValueError("Pass a value for --t_trans")
    if args.t_peak == None:
        raise ValueError("Pass a value for --t_peak")
    if args.Nsteps == None :
        raise ValueError("Pass a value for --Nsteps")

    t_trans = args.t_trans
    t_peak = args.t_peak
    Nsteps = args.Nsteps

elif args.lightcurve == "free+adiabatic":

    include_free = True

    if args.Nsteps == None :
        raise ValueError("Pass a value for --Nsteps")
    if args.nuB == None:
        raise ValueError("Pass a value for --nuB")

    Nsteps = args.Nsteps
    nuB = args.nuB
    fixed_free = args.fixed_free

    if fixed_free: # running with fixed parameters

        if (args.t_peak == None) or (args.L_peak == None):
            raise ValueError("If you want to run with fixed values for the free expansion parameters L_peak and t_peak, then you better pass both of them!")

else:
    pass

# will we scan over the axion mass?
# TODO: make it so we always do! (see problem with 3D-param space in next TODO flag)
scan_ma = ((not include_free) or fixed_free)

# file tail
if correlation_mode == "single dish":
    corr_str = "_SD"
elif correlation_mode == "interferometry":
    corr_str = "_IN"

tail = "_run-"+str(run_id)+corr_str+".txt"

####################################
# save log file for future reference
####################################

# axion-photon coupling
ga_ref = 1.e-10  # [GeV^-1]

log_file = os.path.join(folder, "run_%d_log.txt" % run_id)
with open(log_file, 'w') as f:
    f.write('#\n#-------Run info\n#\n')
    f.write('run_id: %d\n' % run_id)
    f.write('ga_ref: %e\n' % ga_ref)
    f.write('scan_ma: %s\n' % scan_ma)
    f.write('#\n#-------Detailed log\n#\n')
    for key, entry in vars(args).items():
        f.write('%s: %s\n' % (key, entry))

# -------------------------------------------------

##########
# ARRAYS #
##########

if include_free and (not fixed_free):
    # SNR early-time evolution: from Bietenholz et al., Table 4.
    # from quantities in constants.py
    # including free expansion but scanning over Lpk & tpk
    # we will NOT scan over ma, because that's a large dimensionality;
    # rather, we will fix ma = ma @ nu_pivot of 1 GHz
    # TODO: figure out a way to scan over ma as well!

    # tpk and Lpk arrays:
    Nsigs = 3. # number of standard deviations from the Bietenholz's mean to scan
    tpk_arr = np.logspace(ct._mu_log10_tpk_-Nsigs*ct._sig_log10_tpk_, ct._mu_log10_tpk_+Nsigs*ct._sig_log10_tpk_, Nsteps+1)
    Lpk_arr = np.logspace(ct._mu_log10_Lpk_-Nsigs*ct._sig_log10_Lpk_, ct._mu_log10_Lpk_+Nsigs*ct._sig_log10_Lpk_, Nsteps+2)

    # Saving arrays
    if os.access(folder+"tpk_arr.txt", os.R_OK):
        pass
    else:
        np.savetxt(folder+"tpk_arr.txt", tpk_arr)

    if os.access(folder+"Lpk_arr.txt", os.R_OK):
        pass
    else:
        np.savetxt(folder+"Lpk_arr.txt", Lpk_arr)

else:
    # either adiabatic_only, or include_free but fixed_free.
    # at any rate, there is only 1D param space: ma, so we will scan over it.

    # Defining a fine array of frequencies
    # for SKA low...
    nulow = np.logspace(log10(ct._nu_min_ska_low_), log10(ct._nu_max_ska_low_), Nsteps//2)[1:]
    # ... and SKA mid...
    numid = np.logspace(log10(ct._nu_min_ska_mid_), log10(ct._nu_max_ska_mid_), Nsteps - Nsteps//2)[1:]
    # ... concatenating...
    nu_arr = np.concatenate((nulow, numid))
    # ... and converting into axion masses
    ma_arr = pt.ma_from_nu(nu_arr)

    # Saving arrays
    if os.access(folder+"ma_arr.txt", os.R_OK):
        pass
    else:
        np.savetxt(folder+"ma_arr.txt", ma_arr)


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

# Creating SNR directories:
for name in snrs_cut.keys():
    try:
        os.makedirs(folder+name+"/")
    except:
        pass

# -------------------------------------------------

###########
# ROUTINE #
###########

# A small number:
ridiculous = 1.e-100

# Maximum number of steps:
max_steps = 1000001

# data:
data = {'deltaE_over_E': ct._deltaE_over_E_,
        'f_Delta': ct._f_Delta_,
        'exper': 'SKA',
        'total_observing_time': 100.,
        'verbose': 0,
        'DM_profile': 'NFW',
        'correlation_mode': correlation_mode,
        'average': True}

# Results dictionaries:
sn_results = {}
snu_results = {}
if include_free:
    time_results = {} # t_age for args.age_mode == "ratio_age", t_trans for args.age_mode == "known_age"/"size_age"

# SNR counter
counter = 0

# Sorting the SNR names for easier tracking
sorted_names = snrs_cut.keys()
sorted_names.sort()
for i, name in tqdm(enumerate(sorted_names)):

    # SNR object:
    snr = snrs_cut[name]

    # running only for those SNR with known ages
    if age_is_known:
        try:
            # for SNR with known age:
            t_age = snr.age # [years]
        except:
            continue

    # SNR folder
    snr_folder = folder+name+"/"
    # name of file
    file_name = name+tail

    if verbose:
        print(name)

    # Reading some important SNR properties:
    gamma = ap.gamma_from_alpha(snr.alpha) # Sedov-Taylor analytic formula
    L0 = snr.get_luminosity() # [cgs]
    R = snr.get_radius() # [pc]

    # defining the lightcurve parameters
    lightcurve_params = {'L_today': L0,
                         'use_free_expansion': include_free}

    # adiabatic_only computation:
    if not include_free:
        # if age is not known it needs to be computed:
        if not age_is_known:
            # computed from the SNR radius
            if args.age_mode == "size_age":
                t_age = dt.age_from_radius(R,
                                           method=args.method,
                                           M_ej=args.M_ej,
                                           E_sn=args.E_sn,
                                           rho0=args.rho0)
            elif args.age_mode == "ratio_age":
                raise Error("args.lightcurve=='adiabatic_only' and yet args.age_mode=='ratio_age'. This should not have happened.")

        # computing L_peak
        _, computed_pars = ap.L_source(t_age, model='eff',
                                       output_pars=True,
                                       gamma=gamma,
                                       t_peak=t_peak, t_trans=t_trans,
                                       L_today=L0, t_age=t_age)

        L_peak = computed_pars['L_peak']
        del computed_pars

        # updating lightcurve parameters
        lightcurve_params.update({'t_age':t_age,
                                  't_trans':t_trans,
                                  't_peak':t_peak})

        # Snu kwargs
        age_steps = abs(int(1000*(log10(t_age) - log10(t_peak/365.)) + 1))
        snu_echo_kwargs = {'tmin_default': None,
                           'Nt': min(age_steps, max_steps),
                           'xmin': ct._au_over_kpc_,
                           'xmax_default': 100.,
                           'use_quad': False,
                           'lin_space': False,
                           'Nint': min(age_steps, max_steps),
                           't_extra_old': t_extra}

        # computing routine
        z, new_output = md.snr_routine(ma_arr, ga_ref,
                                       snr,
                                       lightcurve_params=lightcurve_params,
                                       snu_echo_kwargs=snu_echo_kwargs,
                                       data=data,
                                       output_all=True)

        # keeping echo's spectral irradiance
        signal_Snu = new_output['signal_Snu']
        del new_output

        if verbose:
            print("signal_Snu = "+str(signal_Snu))
            print("S/N= "+str(z))

        # saving S/N ratio and spectral irradiance of echo (Snu_echo)
        np.savetxt(snr_folder+"sn_"+file_name, z, delimiter=",")
        np.savetxt(snr_folder+"echo_"+file_name, signal_Snu, delimiter=",")

        tage_file = os.path.join(snr_folder, "tage_"+file_name)
        with open(tage_file, 'w') as f:
            f.write(str(t_age))

        ttrans_file = os.path.join(snr_folder, "ttrans_"+file_name)
        with open(ttrans_file, 'w') as f:
            f.write(str(t_trans))

    # free+adiabatic, fixed Lpk-tpk free expansion parameters
    elif include_free and fixed_free:
        # need to change Lpk @ nuB ---> Lpk @ 1 GHz
        from_Bieten_to_pivot = (1./nuB)**-snr.alpha # correction from the fact that the Bietenholz frequency

        tpk = args.t_peak
        Lpk = args.L_peak*from_Bieten_to_pivot

        # if age is not known it needs to be computed:
        if not age_is_known:
            # computed from the SNR radius
            if args.age_mode == "size_age":
                t_age = dt.age_from_radius(R,
                                           method=args.method,
                                           M_ej=args.M_ej,
                                           E_sn=args.E_sn,
                                           rho0=args.rho0)
                lightcurve_params.update({'t_age':t_age})

                # finding t_trans
                _, computed_pars = ap.L_source(t_age, model='eff',
                                               output_pars=True,
                                               gamma=gamma,
                                               t_peak=tpk, L_peak=Lpk,
                                               L_today=L0, t_age=t_age)

                t_trans = computed_pars['t_trans']
                del computed_pars

            # computed from tt_ratio = t_peak/t_trans
            elif args.age_mode == "ratio_age":
                t_trans = tt_ratio*(tpk/365.)
                lightcurve_params.update({'t_trans':t_trans})
                t_age = ap.tage_compute(Lpk, tpk, t_trans, L0, gamma)

        # updating lightcurve parameters
        lightcurve_params.update({'t_peak':tpk,
                                  'L_peak':Lpk})

        # Snu kwargs
        age_steps = abs(int(1000*(log10(t_age) - log10(tpk/365.)) + 1))
        snu_echo_kwargs = {'tmin_default': None,
                            'Nt': min(age_steps, max_steps),
                            'xmin': ct._au_over_kpc_,
                            'xmax_default': 100.,
                            'use_quad': False,
                            'lin_space': False,
                            'Nint': min(age_steps, max_steps),
                            't_extra_old': t_extra}

        # computing routine
        z, new_output = md.snr_routine(ma_arr, ga_ref,
                                   snr,
                                   lightcurve_params=lightcurve_params,
                                   snu_echo_kwargs=snu_echo_kwargs,
                                   data=data,
                                   output_all=True)

        signal_Snu = new_output['signal_Snu']
        del new_output

        if verbose:
            print("signal_Snu = "+str(signal_Snu))
            print("S/N= "+str(z))

        # saving S/N ratio and spectral irradiance of echo (Snu_echo)
        np.savetxt(snr_folder+"sn_"+file_name, z, delimiter=",")
        np.savetxt(snr_folder+"echo_"+file_name, signal_Snu, delimiter=",")

        tage_file = os.path.join(snr_folder, "tage_"+file_name)
        with open(tage_file, 'w') as f:
            f.write(str(t_age))

        ttrans_file = os.path.join(snr_folder, "ttrans_"+file_name)
        with open(ttrans_file, 'w') as f:
            f.write(str(t_trans))

    # free+adiabatic, scan over Lpk-tpk free expansion parameters
    else:
        # need to change Lpk @ nuB ---> Lpk @ 1 GHz
        from_Bieten_to_pivot = (1./nuB)**-snr.alpha # correction from the fact that the Bietenholz frequency is not the pivot frequency [1 GHz]
        new_Lpk_arr = np.copy(Lpk_arr) # copying peak luminosity array
        new_Lpk_arr *= from_Bieten_to_pivot  # correcting L_peak by switching from the Bietenholz to the pivot frequencies

        # preparing the arrays to be filled:
        sn_results[name] = []
        snu_results[name] = []
        tage_results[name] = []
        ttrans_results[name] = []

        # start!
        for tpk in tpk_arr:

            row_a = []
            row_b = []
            row_c = []
            row_d = []

            for Lpk in new_Lpk_arr:

                try:
                    # if age is not known it needs to be computed:
                    if not age_is_known:
                        # computed from the SNR radius
                        if args.age_mode == "size_age":
                            t_age = dt.age_from_radius(R,
                                                    method=args.method,
                                                    M_ej=args.M_ej,
                                                    E_sn=args.E_sn,
                                                    rho0=args.rho0)
                            lightcurve_params.update({'t_age':t_age})

                            # finding t_trans
                            _, computed_pars = ap.L_source(t_age, model='eff',
                                               output_pars=True,
                                               gamma=gamma,
                                               t_peak=tpk, L_peak=Lpk,
                                               L_today=L0, t_age=t_age)

                            t_trans = computed_pars['t_trans']
                            del computed_pars

                        # computed from tt_ratio = t_peak/t_trans
                        elif args.age_mode == "ratio_age":
                            t_trans = tt_ratio*(tpk/365.)
                            lightcurve_params.update({'t_trans':t_trans})
                            t_age = ap.tage_compute(Lpk, tpk, t_trans, L0, gamma)

                    # updating lightcurve parameters
                    lightcurve_params.update({'t_peak':tpk,
                                              'L_peak':Lpk})

                    # Snu kwargs
                    age_steps = abs(int(1000*(log10(t_age) - log10(tpk/365.)) + 1))
                    snu_echo_kwargs = {'tmin_default': None,
                                        'Nt': min(age_steps, max_steps),
                                        'xmin': ct._au_over_kpc_,
                                        'xmax_default': 100.,
                                        'use_quad': False,
                                        'lin_space': False,
                                        'Nint': min(age_steps, max_steps),
                                        't_extra_old': t_extra}

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
                        print("signal_Snu = "+str(signal_Snu))
                        print("S/N= "+str(z))

                    # building rows
                    row_a.append(z) # signal-to-noise ratio
                    row_b.append(signal_Snu) # signal S_nu
                    row_c.append(t_age) # t_age
                    row_d.append(t_trans) # t_trans

                except:
                    # nonsense results; append some ridiculous value
                    row_a.append(ridiculous)
                    row_b.append(ridiculous)
                    row_c.append(ridiculous)
                    row_d.append(ridiculous)

                # end of routine for fixed Lpk

            # appending finished Lpk rows
            sn_results[name].append(row_a)
            snu_results[name].append(row_b)
            tage_results[name].append(row_c)
            ttrans_results[name].append(row_d)

            # end of routine for fixed tpk

        # converting grids to arrays
        sn_results[name] = np.array(sn_results[name])
        snu_results[name] = np.array(snu_results[name])
        tage_results[name] = np.array(tage_results[name])
        ttrans_results[name] = np.array(ttrans_results[name])

        # saving grids
        np.savetxt(snr_folder+"sn_"+file_name, sn_results[name], delimiter=",")
        np.savetxt(snr_folder+"echo_"+file_name, snu_results[name], delimiter=",")
        np.savetxt(snr_folder+"tage_"+file_name, tage_results[name], delimiter=",")
        np.savetxt(snr_folder+"ttrans_"+file_name, ttrans_results[name], delimiter=",")

    counter += 1

    # end of routine for fixed snr

print(counter)
