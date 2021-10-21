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


################
# BASIC PARAMS #
################

# Some basic axion parameters:
ga_ref = 1.e-10 # [GeV^-1]
nu_pivot = 1. # [GHz]
ma_pivot = pt.ma_from_nu(1.) # [eV]

# A small number:
very_small = 1.e-100

# A large number
very_large = 1.e100

# Maximum number of steps:
max_steps = 1000001


# -------------------------------------------------

#############
# ARGUMENTS #
#############

# Defining arguments
parser = argparse.ArgumentParser(description="Computes the reach for the SNR in the Green's Catalog, for a specific approach to their lightcurve and their age.")

# Arguments with numerical values:
parser.add_argument("-i", "--run_id", "--run", "--id", default=0,
                    type=int, help="The run ID number (default: 0)")
parser.add_argument("-N", "--Nsteps", default=None,
                    type=int, help="The number of steps in parameter space arrays (default: None)")
parser.add_argument("-x", "--t_extra", "--extra", default=0.,
                    type=float, help="The extra age of the SNR, after the adiabatic phase [years] (default: 0.)")
parser.add_argument("-z", "--sn_th", "--signal_noise_ratio", default=1.,
                    type=float, help="The threshold signal-to-noise ratio to be considered (default: 1)")
parser.add_argument("-c", "--SKA_mode", "--correlation_mode", "--correl", default=None,
                    type=str, choices=["single_dish", "interferometry"], help="The running mode of SKA.")
parser.add_argument("-a", "--age_mode", default=None,
                    type=str, choices=["known_age", "size_age", "ratio_age"], help="The way in which the age of the SNR will be treated.")

# AGES CASE A: arguments for args.age_mode == "size_age":
parser.add_argument("--method", default=None,
                    choices=['TM99-0', 'TM99-simple', 'estimate', 'lin', 'log', 'pheno', 'phenomenological'],
                    type=str, help="The method to compute the age. Based on either Truelove-McKee '99 (TM99; for n=0 ejecta or a simplified version), a quick-and-dirty estimate, a linear regression performed on the (linear/log) data of SNR with known age (default: None), or a phenomenological model with a broken power law.")

# A.1: arguments for 'TM99-0' & 'TM99-simple'
parser.add_argument('--M_ej', default=1.,
                    type=float, help="The mass [Msun] of the SNR ejecta (default: 1). Only relevant if method=='TM99-0'/'TM99-simple'.")

# A.2: arguments for 'TM99-0', 'TM99-simple', and 'estimate'
parser.add_argument('--E_sn', default=1.,
                    type=float, help="The energy output [1.e51 ergs] of the SNR (default: 1). Only relevant if method=='TM99-0'/'TM99-simple'.")
parser.add_argument('--rho0', default=1.,
                    type=float, help="The density [proton mass/cm^3] of the interstellar medium surrounding the SNR (default: 1). Only relevant if method=='TM99-0'/'TM99-simple'.")

# A.3: arguments for 'phenomenological'
parser.add_argument('--Rst', default=3.8,
                    type=float, help="Radius [pc] at the start of the Sedov-Taylor (adiabatic) expansion phase (default: 3.8).")
parser.add_argument('--tst', default=360.,
                    type=float, help="Age [years] at the start of the Sedov-Taylor (adiabatic) expansion phase (default 360.).")
parser.add_argument('--eta1', default=1.,
                    type=float, help="Power scaling R~t^eta1 during the Ejecta-Dominated phase (default: 1.).")
parser.add_argument('--eta2', default=0.4,
                    type=float, help="Power scaling R`t^eta2 during the Sedov-Taylor expansion phase (default: 2/5 = 0.4).")

# AGES CASE B: arguments for args.age_mode == "ratio_age":
parser.add_argument("-r", "--tt_ratio", "--ratio", default=None,
                    type=float, help="The ratio of t_trans/t_pk (default: None)")

# verbosity
parser.add_argument("-v", "--verbose", action="store_true", help="Verbosity (default: False)")


# defining the subparsers, and sending their names to .slice attribute
slice_subparsers = parser.add_subparsers(dest="slice", description="The following subcommand options determine the parameter space slice to be explored. NOTA BENE: A slice is denoted by ParX-ParY, in (x,y) axis ordering. ParX is the x-array and will have Nsteps+1 points; ParY is the y-array will have Nsteps+2 points. The routine starts iterating over the y-array (rows), and then proceeds to iterate over the x-array (columns), for easier plotting.")

# SLICE 1: ma-ga slice
mg_subparser = slice_subparsers.add_parser("ma-ga", help="ma-ga parameter space slice [eV, GeV^-1].")

# SLICE 1: ma-ga slice
#-----------
# Lightcurve subparsers
# N.B.: if we want to have ma-ga parameter space slice, we need to fixed all the lightcurve parameters.
# Otherwise, the dimensionality would be too large.

# Defining the subparsers and sending their names to .lightcurve attributes
lc_subparser = mg_subparser.add_subparsers(dest="lightcurve",
                            help="Subcommand options determining the way in which the lightcurve of the SNR will be treated.")

# 1.1 Adiabatic expansion-only lightcurve
adiab = lc_subparser.add_parser("adiabatic_only", help="Only the adiabatic expansion part of the lightcurve will be used in our computations.")
adiab.add_argument('--t_trans', default=None,
                    type=float, help="The duration [years] of the free expansion phase (default: None).")

# N.B. FOR DEBUGGING PURPOSES ONLY:
adiab.add_argument("-k", "--t_peak", default=10**(ct._mu_log10_tpk_),
                    type=float, help="N.B. FOR DEBUGGING PURPOSES ONLY: The time [days] of peak SNR luminosity. Since the free expansion will be ignored, its precise value is irrelevant (default: 10^1.7).")

# 1.2: Free+Adiabatic expansions lightcurve
free_adiab = lc_subparser.add_parser("free+adiabatic", help="Both the free expansion and adiabatic expansion parts of the lightcurve will be used in our computations.")
free_adiab.add_argument("-n", "--nuB", "--nu_Bietenholz", default=None,
                        type=float, help="The Bietenholz frequency [GHz] (default: None)")
free_adiab.add_argument("-L", "--L_peak", default=None,
                        type=float, help="The peak luminosity [erg/s/Hz] (default: None)")
free_adiab.add_argument("-k", "--t_peak", default=None,
                        type=float, help="The time [days] of peak SNR luminosity (default: None).")



# SLICE 2: Lpk-tpk slice
Lt_subparser = slice_subparsers.add_parser("Lpk-tpk", help="L_peak-t_peak parameter space slice [ergs/s/Hz, days].")

Lt_subparser.add_argument("-m", "--ma", default=ma_pivot,
                            type=float, help="The benchmark axion mass [eV] (default: {:.1e}, for 1 GHz)".format(ma_pivot))
Lt_subparser.add_argument("-n", "--nuB", "--nu_Bietenholz", default=None,
                            type=float, help="The Bietenholz frequency [GHz] (default: None)")



# SLICE 3: ttr-tpk slice
tt_subparser = slice_subparsers.add_parser("ttr-tpk", help="t_trans-t_peak parameter space slice [years, days].")

tt_subparser.add_argument("-m", "--ma", default=ma_pivot,
                            type=float, help="The benchmark axion mass [eV] (default: {:.1e}, for 1 GHz)".format(ma_pivot))


###################
# Reading Arguments
###################

#-------------------
# Parsing
args = parser.parse_args()

# Defining appropriate variables
run_id = args.run_id
Nsteps = args.Nsteps
SKA_mode = (args.SKA_mode).replace('_', ' ')
t_extra = args.t_extra
sn_th = args.sn_th
verbose = args.verbose

#-----------------------
# Checking basic values:
if run_id == None:
    raise Exception("Pass a value for --run_id.")
if SKA_mode == None:
    raise Exception("Pass a value for --SKA_mode.")
if Nsteps == None:
    raise Exception("Pass a value for --Nsteps.")
if t_extra == None:
    raise Exception("Pass a value for --t_extra.")
if sn_th == None:
    raise Exception("Pass a value for --sn_th.")

#-------------------------
# Checking age_mode values
age_is_known = False # will need to compute age: either from R or from size
if args.age_mode == "known_age":
    age_is_known = True # since age is known, no need to compute it
elif args.age_mode == "size_age":
    if not args.method in ['TM99-0', 'TM99-simple', 'estimate', 'lin', 'log', 'pheno', 'phenomenological']:
        raise ValueError("Please make sure the argument --method is one of the allowed options.")
elif args.age_mode == "ratio_age":
    if args.tt_ratio == None:
        raise ValueError("Pass a value for --tt_ratio.")
    tt_ratio = args.tt_ratio
else:
    pass

# Defining age_kwargs
age_kwargs = {}
if args.age_mode == "size_age":
    if (args.method in ['TM99-0', 'TM99-simple', 'estimate']):
        age_kwargs.update({'M_ej':args.M_ej,
                           'E_sn':args.E_sn,
                           'rho0':args.rho0})
    elif (args.method in ['pheno', 'phenomenological']):
        age_kwargs.update({'Rst':args.Rst,
                           'tst':args.tst,
                           'eta1':args.eta1,
                           'eta2':args.eta2})

# inconsistent choice:
if args.age_mode == "ratio_age":
    if args.slice == "ttr-tpk":
        raise ValueError("If the age is to be deduced from r = t_trans/t_peak, then we cannot use the parameter space ttr-tpk (t_trans, t_peak). Please choose another parameter space slice.")
    if (args.slice == "ma-ga") and (args.lightcurve == "adiabatic_only"):
        raise ValueError("If the age is to be deduced from r = t_trans/t_peak, then we need the full 'free+adiabatic' lightcurve evolution.")
else:
    pass


#----------------------------
# Checking lightcurve values:
include_free = True

if args.slice == "ma-ga":
    if args.lightcurve == "adiabatic_only":

        include_free = False

        if args.t_trans == None:
            raise ValueError("Pass a value for --t_trans")
        if args.t_peak == None:
            raise ValueError("Pass a value for --t_peak")

        t_trans = args.t_trans
        t_peak = args.t_peak

    elif args.lightcurve == "free+adiabatic":

        if (args.t_peak == None) or (args.L_peak == None):
            raise ValueError("If you want to run with fixed values for the free expansion parameters L_peak and t_peak, then you better pass both of them!")

        if args.nuB == None:
            raise ValueError("Pass a value for --nuB")

        nuB = args.nuB

    else:
        pass


# shorthand
if args.slice in ["Lpk-tpk", "ttr-tpk"]:
    ma = args.ma
if args.slice == "Lpk-tpk":
    nuB = args.nuB

#############################
# File handling preliminaries
#############################

#----------
# file tail
tail = "_run-"+str(run_id)+".txt"

#-----------------------------------
# save log file for future reference

log_file = os.path.join(folder, "run_%d_log.txt" % run_id)
with open(log_file, 'w') as f:
    f.write('#\n#-------Run info\n#\n')
    f.write('run_id: %d\n' % run_id)
    f.write('ga_ref: %e\n' % ga_ref)
    f.write('nu_pivot: %e\n' % nu_pivot)
    f.write('ma_pivot: %e\n' % ma_pivot)
    f.write('#\n#-------Detailed log\n#\n')
    for key, entry in vars(args).items():
        f.write('%s: %s\n' % (key, entry))


##########
# ARRAYS #
##########


#---------
# SLICE 1:
if args.slice == "ma-ga":
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


#---------
# SLICE 2:
elif args.slice == "Lpk-tpk":
    # SNR early-time evolution: from Bietenholz et al., Table 4.
    # from quantities in constants.py
    # we will NOT scan over ma, because that's a large dimensionality;
    # rather ma will be fixed.

    # t_peak and L_peak arrays:
    Nsigs = 3. # number of standard deviations from the Bietenholz's mean to scan
    # x-array: L_peak
    Lpk_arr = np.logspace(ct._mu_log10_Lpk_-Nsigs*ct._sig_log10_Lpk_, ct._mu_log10_Lpk_+Nsigs*ct._sig_log10_Lpk_, Nsteps+1)
    # y-array: t_peak
    tpk_arr = np.logspace(ct._mu_log10_tpk_-Nsigs*ct._sig_log10_tpk_, ct._mu_log10_tpk_+Nsigs*ct._sig_log10_tpk_, Nsteps+2)

    # Saving arrays
    if os.access(folder+"Lpk_arr.txt", os.R_OK):
        pass
    else:
        np.savetxt(folder+"Lpk_arr.txt", Lpk_arr)

    if os.access(folder+"tpk_arr.txt", os.R_OK):
        pass
    else:
        np.savetxt(folder+"tpk_arr.txt", tpk_arr)


#---------
# SLICE 3:
elif args.slice == "ttr-tpk":
    # t_trans and t_peak arrays
    Nsigs = 3. # number of standard deviations from the Bietenholz's mean to scan
    # x-array: t_trans
    ttr_arr = np.linspace(100, 1000, Nsteps+1)
    # y-array: t_peak
    tpk_arr = np.logspace(ct._mu_log10_tpk_-Nsigs*ct._sig_log10_tpk_, ct._mu_log10_tpk_+Nsigs*ct._sig_log10_tpk_, Nsteps+2)

    # Saving arrays
    if os.access(folder+"ttr_arr.txt", os.R_OK):
        pass
    else:
        np.savetxt(folder+"ttr_arr.txt", ttr_arr)

    if os.access(folder+"tpk_arr.txt", os.R_OK):
        pass
    else:
        np.savetxt(folder+"tpk_arr.txt", tpk_arr)

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


###########
# ROUTINE #
###########

# data:
data = {'deltaE_over_E': ct._deltaE_over_E_,
        'f_Delta': ct._f_Delta_,
        'exper': 'SKA',
        'total_observing_time': 100.,
        'verbose': 0,
        'DM_profile': 'NFW',
        'correlation_mode': SKA_mode,
        'average': True}


# Sorting the SNR names for easier tracking
sorted_names = snrs_cut.keys()
sorted_names.sort()

# SNR counter
counter = 0
for i, name in tqdm(enumerate(sorted_names)):

    # SNR object:
    snr = snrs_cut[name]

    # Reading some important SNR properties:
    alpha = snr.get_spectral_index() # spectral index
    gamma = snr.get_gamma() # Sedov-Taylor analytic formula
    L0 = snr.get_luminosity() # [cgs]
    R = snr.get_radius() # [pc]

    # Defining the lightcurve parameters
    lightcurve_params = {'L_today': L0,
                         'use_free_expansion': include_free}

    # Running only for those SNR with known ages...
    if age_is_known:
        # For SNR with known age:
        t_age = snr.get_age() # [years]
        if t_age == None:
            continue

        # Updating with age:
        lightcurve_params.update({'t_age':t_age})

    elif args.age_mode == "size_age":
        # For SNR whose age will be computed from their radii:
        t_age = dt.age_from_radius(R,
                    method=args.method,
                    **age_kwargs)
        # Updating with age:
        lightcurve_params.update({'t_age':t_age})


    #-------------------
    # SNR file handling:
    # SNR folder
    snr_folder = folder+name+"/"
    # Name of file
    file_name = name+tail

    # Printing SNR name!
    if verbose:
        print(name)

    #---------
    # SLICE 1:
    if args.slice == "ma-ga":
        #............................
        # adiabatic_only computation:
        if not include_free:
            # If age is not known it needs to be computed:
            if args.age_mode == "ratio_age":
                raise Error("args.lightcurve=='adiabatic_only' (available for args.slice=='ma-ga') and yet args.age_mode=='ratio_age'. This should not have happened.")

            # Computing L_peak
            _, computed_pars = ap.L_source(t_age, model='eff',
                                           output_pars=True,
                                           gamma=gamma,
                                           t_peak=t_peak, t_trans=t_trans,
                                           L_today=L0, t_age=t_age)

            L_peak = computed_pars['L_peak']
            del computed_pars

            # Updating lightcurve parameters
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

            # Performing routine
            z, new_output = md.snr_routine(ma_arr, ga_ref,
                                           snr,
                                           lightcurve_params=lightcurve_params,
                                           snu_echo_kwargs=snu_echo_kwargs,
                                           data=data,
                                           output_all=True)

            # Keeping echo's spectral irradiance
            signal_Snu = new_output['signal_Snu']
            del new_output

            if verbose:
                print("signal_Snu = "+str(signal_Snu))
                print("S/N= "+str(z))

            # Regularizing the signal-to-noise ratio:
            reg_z = np.nan_to_num(z)
            reg_z = np.where(reg_z < very_small, very_small, reg_z) # converting 0s to a small number

            # Finding reach
            ga_reach = ec.ga_reach(sn_th, reg_z, ga_ref)
            ga_reach = np.nan_to_num(ga_reach)

            # Saving spectral irradiance of echo (Snu_echo), S/N ratio, and ga reach
            np.savetxt(snr_folder+"echo_"+file_name, signal_Snu, delimiter=",")
            np.savetxt(snr_folder+"sn_"+file_name, reg_z, delimiter=",")
            np.savetxt(snr_folder+"ga_"+file_name, ga_reach, delimiter=",")

            # Saving age...
            tage_file = os.path.join(snr_folder, "tage_"+file_name)
            with open(tage_file, 'w') as f:
                f.write(str(t_age))

            # ... and t_trans...
            ttrans_file = os.path.join(snr_folder, "ttrans_"+file_name)
            with open(ttrans_file, 'w') as f:
                f.write(str(t_trans))

            # ... and L_peak...
            Lpk_file = os.path.join(snr_folder, "Lpk_"+file_name)
            with open(Lpk_file, 'w') as f:
                f.write(str(L_peak))

        #........................................................
        # free+adiabatic, fixed Lpk-tpk free expansion parameters
        else:
            # Need to change Lpk @ nuB ---> Lpk @ 1 GHz
            from_Bieten_to_pivot = (nu_pivot/nuB)**-alpha # correction from the fact that the Bietenholz frequency

            # Peak parameters
            tpk = args.t_peak
            Lpk = args.L_peak*from_Bieten_to_pivot

            # Updating lightcurve parameters
            lightcurve_params.update({'t_peak':tpk,
                                      'L_peak':Lpk})

            try:
                # Age computed from tt_ratio = t_peak/t_trans
                if args.age_mode == "ratio_age":
                    t_trans = tt_ratio*(tpk/365.)
                    lightcurve_params.update({'t_trans':t_trans})
                    t_age = ap.tage_compute(Lpk, tpk, t_trans, L0, gamma)

                # Age was already computed from size; now computing t_trans
                elif (args.age_mode == "size_age") or (args.age_mode == "known_age"):
                    _, computed_pars = ap.L_source(t_age, model='eff',
                                                    output_pars=True,
                                                    gamma=gamma,
                                                    t_peak=tpk, L_peak=Lpk,
                                                    L_today=L0, t_age=t_age)

                    t_trans = computed_pars['t_trans']
                    del computed_pars

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

                # Regularizing the signal-to-noise ratio:
                reg_z = np.nan_to_num(z)
                reg_z = np.where(reg_z < very_small, very_small, reg_z) # converting 0s to a small number

                # Finding reach
                ga_reach = ec.ga_reach(sn_th, reg_z, ga_ref)
                ga_reach = np.nan_to_num(ga_reach)

            except:
                signal_Snu = very_small*np.ones_like(ma_arr)
                reg_z = very_small*np.ones_like(ma_arr)
                ga_reach = very_large*np.ones_like(ma_arr)
                t_trans = very_small

            # Saving spectral irradiance of echo (Snu_echo), S/N ratio, and ga reach
            np.savetxt(snr_folder+"echo_"+file_name, signal_Snu, delimiter=",")
            np.savetxt(snr_folder+"sn_"+file_name, reg_z, delimiter=",")
            np.savetxt(snr_folder+"ga_"+file_name, ga_reach, delimiter=",")

            # Saving age...
            tage_file = os.path.join(snr_folder, "tage_"+file_name)
            with open(tage_file, 'w') as f:
                f.write(str(t_age))

            # ... and t_trans
            ttrans_file = os.path.join(snr_folder, "ttrans_"+file_name)
            with open(ttrans_file, 'w') as f:
                f.write(str(t_trans))


    #---------
    # SLICE 2:
    elif args.slice == "Lpk-tpk":
        # need to change Lpk @ nuB ---> Lpk @ 1 GHz
        from_Bieten_to_pivot = (nu_pivot/nuB)**-alpha # correction from the fact that the Bietenholz frequency is not the pivot frequency [1 GHz]
        new_Lpk_arr = np.copy(Lpk_arr) # copying peak luminosity array
        new_Lpk_arr *= from_Bieten_to_pivot  # correcting L_peak by switching from the Bietenholz to the pivot frequencies

        # preparing the arrays to be filled:
        echo_gr = []
        sn_gr = []
        ga_gr = []
        tage_gr = []
        ttrans_gr = []

        # start!
        for tpk in tpk_arr:

            row_a = []
            row_b = []
            row_c = []
            row_d = []
            row_e = []

            for Lpk in new_Lpk_arr:

                # Updating lightcurve parameters
                lightcurve_params.update({'t_peak':tpk,
                                          'L_peak':Lpk})

                try:
                    # Computing age from tt_ratio = t_peak/t_trans
                    if args.age_mode == "ratio_age":
                        t_trans = tt_ratio*(tpk/365.)
                        lightcurve_params.update({'t_trans':t_trans})
                        t_age = ap.tage_compute(Lpk, tpk, t_trans, L0, gamma)

                    # Age was already computed from the SNR radius; now finding t_trans
                    elif (args.age_mode == "size_age") or (args.age_mode == "known_age"):
                        _, computed_pars = ap.L_source(t_age, model='eff',
                                           output_pars=True,
                                           gamma=gamma,
                                           t_peak=tpk, L_peak=Lpk,
                                           L_today=L0, t_age=t_age)

                        t_trans = computed_pars['t_trans']
                        del computed_pars

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
                    z, new_output = md.snr_routine(ma, ga_ref,
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

                    # Regularizing the signal-to-noise ratio:
                    reg_z = np.nan_to_num(z)
                    reg_z = np.where(reg_z < very_small, very_small, reg_z) # converting 0s to a small number

                    # Finding reach
                    ga_reach = ec.ga_reach(sn_th, reg_z, ga_ref)
                    ga_reach = np.nan_to_num(ga_reach)

                    # building rows
                    row_a.append(signal_Snu) # signal S_nu
                    row_b.append(reg_z) # signal-to-noise ratio
                    row_c.append(ga_reach) # ga reach
                    row_d.append(t_age) # t_age
                    row_e.append(t_trans) # t_trans

                except:
                    # nonsense results; append some very small/large value
                    row_a.append(very_small)
                    row_b.append(very_small)
                    row_c.append(very_large)
                    row_d.append(very_small)
                    row_e.append(very_small)

                # end of routine for fixed Lpk

            # appending finished Lpk rows
            echo_gr.append(row_a)
            sn_gr.append(row_b)
            ga_gr.append(row_c)
            tage_gr.append(row_d)
            ttrans_gr.append(row_e)

            # end of routine for fixed tpk

        # converting grids to arrays
        echo_gr = np.array(echo_gr)
        sn_gr = np.array(sn_gr)
        ga_gr = np.array(ga_gr)
        tage_gr = np.array(tage_gr)
        ttrans_gr = np.array(ttrans_gr)

        # saving grids
        np.savetxt(snr_folder+"echo_"+file_name, echo_gr, delimiter=",")
        np.savetxt(snr_folder+"sn_"+file_name, sn_gr, delimiter=",")
        np.savetxt(snr_folder+"ga_"+file_name, ga_gr, delimiter=",")
        np.savetxt(snr_folder+"tage_"+file_name, tage_gr, delimiter=",")
        np.savetxt(snr_folder+"ttrans_"+file_name, ttrans_gr, delimiter=",")


    #---------
    # SLICE 3:
    elif args.slice == "ttr-tpk":
        # preparing the arrays to be filled:
        echo_gr = []
        sn_gr = []
        ga_gr = []
        tage_gr = []
        Lpk_gr = []

        # start!
        for tpk in tpk_arr:

            row_a = []
            row_b = []
            row_c = []
            row_d = []
            row_e = []

            for t_trans in ttr_arr:
                # Updating lightcurve parameters
                lightcurve_params.update({'t_peak':tpk,
                                          't_trans':t_trans})

                try:
                    # Computing age from tt_ratio = t_peak/t_trans
                    if args.age_mode == "ratio_age":
                        raise Error("args.slice=='ttr-tpk' and yet args.age_mode=='ratio_age'. This should not have happened.")

                    # Age was already computed from the SNR radius; now finding L_peak
                    _, computed_pars = ap.L_source(t_age, model='eff',
                                       output_pars=True,
                                       gamma=gamma,
                                       t_peak=tpk, t_trans=t_trans,
                                       L_today=L0, t_age=t_age)

                    Lpk = computed_pars['L_peak']
                    del computed_pars

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
                    z, new_output = md.snr_routine(ma, ga_ref,
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

                    # Regularizing the signal-to-noise ratio:
                    reg_z = np.nan_to_num(z)
                    reg_z = np.where(reg_z < very_small, very_small, reg_z) # converting 0s to a small number

                    # Finding reach
                    ga_reach = ec.ga_reach(sn_th, reg_z, ga_ref)
                    ga_reach = np.nan_to_num(ga_reach)

                    # building rows
                    row_a.append(signal_Snu) # signal S_nu
                    row_b.append(reg_z) # signal-to-noise ratio
                    row_c.append(ga_reach) # ga reach
                    row_d.append(t_age) # t_age
                    row_e.append(Lpk) # L_peak

                except:
                    # nonsense results; append some very small/large value
                    row_a.append(very_small)
                    row_b.append(very_small)
                    row_c.append(very_large)
                    row_d.append(very_small)
                    row_e.append(very_small)

                # end of routine for fixed ttr

            # appending finished ttr rows
            echo_gr.append(row_a)
            sn_gr.append(row_b)
            ga_gr.append(row_c)
            tage_gr.append(row_d)
            Lpk_gr.append(row_e)

            # end of routine for fixed tpk

        # converting grids to arrays
        echo_gr = np.array(echo_gr)
        sn_gr = np.array(sn_gr)
        ga_gr = np.array(ga_gr)
        tage_gr = np.array(tage_gr)
        Lpk_gr = np.array(Lpk_gr)

        # saving grids
        np.savetxt(snr_folder+"echo_"+file_name, echo_gr, delimiter=",")
        np.savetxt(snr_folder+"sn_"+file_name, sn_gr, delimiter=",")
        np.savetxt(snr_folder+"ga_"+file_name, ga_gr, delimiter=",")
        np.savetxt(snr_folder+"tage_"+file_name, tage_gr, delimiter=",")
        np.savetxt(snr_folder+"Lpk_"+file_name, Lpk_gr, delimiter=",")

    #-----------
    counter += 1
    # end of routine for fixed snr
print(counter)
