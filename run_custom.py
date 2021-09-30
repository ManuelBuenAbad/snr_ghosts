from __future__ import division
import model as md
import data as dt
import routines as rt
import echo as ec
import astro as ap
import particle as pt
import constants as ct
import numpy as np
from numpy import pi, sqrt, log, log10, power, exp
from scipy.interpolate import interp1d
from tqdm import tqdm
import os
import argparse

# current directory
current_dir = os.getcwd()


# -------------------------------------------------

#############
# ARGUMENTS #
#############


# defining the higher-level parser: parameter space slices
parser = argparse.ArgumentParser(
    description="Computing reach for custom SNR for a specific parameter space slice.")
parser.add_argument('-N', '--Nsteps',
                    default=None,
                    type=int,
                    help="The number of steps in the parameter space arrays (default: None)")
parser.add_argument('-a', '--alpha',
                    default=0.5,
                    type=float,
                    help="The SNR spectral index (default: 0.5)")
parser.add_argument('-n', '--nuB', '--nu_Bietenholz',
                    default=None,
                    type=float,
                    help="The Bietenholz frequency [GHz] (default: None)")
parser.add_argument('-z', '--sz', '--size',
                    default=None,
                    type=float,
                    help="The size of the source [sr] (default: None; will be calculated from first principles)")
parser.add_argument('-v', '--verbose',
                    action="store_true",
                    help="Verbosity (default: False)")
parser.add_argument('-i', '--run',
                    default=0,
                    type=int,
                    help="The run ID number (default: 0)")


# defining the subparsers, and sending their names to .slice attribute
subparsers = parser.add_subparsers(
    dest="slice", description="The following subcommand options determine the parameter space slice to be explored. NOTA BENE: A slice is denoted by ParX-ParY, in (x,y) axis ordering. ParX is the x-array and will have Nsteps+1 points; ParY is the y-array will have Nsteps+2 points. The routine starts iterating over the y-array (rows), and then proceeds to iterate over the x-array (columns), for easier plotting.")


# CASE 1: Lpk-tpk slice
Lt_parser = subparsers.add_parser(
    'Lpk-tpk', help="Lpk-tpk parameter space slice")

Lt_parser.add_argument('-D', '--distance', '--dist',
                       default=1.,
                       type=float,
                       help="The distance to the source [kpc] (default: 1)")
Lt_parser.add_argument('-r', '--tt_ratio', '--ratio',
                       default=30.,
                       type=float,
                       help="The ratio of t_trans/t_pk (default: 30)")
Lt_parser.add_argument('-s', '--S0', '--irrad', '--flux',
                       default=None,
                       type=float,
                       help="The SNR spectral irradiance at the end of the adiabatic phase [Jy] (default: None)")
Lt_parser.add_argument('-t', '--t_signal', '--t0',
                       default=None,
                       type=float,
                       help="The age of the SNR signal [years] (default: None)")
Lt_parser.add_argument('-x', '--t_extra', '--extra',
                       default=0.,
                       type=float,
                       help="The extra age of the SNR, after the adiabatic phase [years] (default: 0)")
Lt_parser.add_argument('-lb', '--coords', '--long_lat',
                       default=(0., 0.),
                       type=float,
                       nargs=2,
                       help="The galactic coordinates of the SNR [deg] (default: (0, 0))")


# CASE 2: tsig-r slice
tr_parser = subparsers.add_parser(
    'tsig-r', help="t_signal-ratio parameter space slice")

tr_parser.add_argument('-L', '--Lpk', '--L_peak',
                       default=(10.**ct._mu_log10_Lpk_),
                       type=float,
                       help="The peak luminosity of the SNR, at the Bietenholz frequency [erg/s/Hz] (default: 10^25.5)")
tr_parser.add_argument('-p', '--tpk', '--t_peak',
                       default=(10.**ct._mu_log10_tpk_),
                       type=float,
                       help="The peak time of the SNR [days] (default: 10^1.7)")
tr_parser.add_argument('-D', '--distance', '--dist',
                       default=1.,
                       type=float,
                       help="The distance to the source [kpc] (default: 1)")

# WARNING: when tsig < 1e4, it means the SNR is still going through
# the second phase. Therefore, we should not add extra age to it in
# general. The only use case for this flag is to test the impact of
# adiabatic phase duration on extra old SNRs

tr_parser.add_argument('-x', '--t_extra', '--extra',
                       default=0.,
                       type=float,
                       help="Debug only: the extra age of the SNR, after the adiabatic phase [years] (default: 0)")

# end of warning

tr_parser.add_argument('-lb', '--coords', '--long_lat',
                       default=(0., 0.),
                       type=float,
                       nargs=2,
                       help="The galactic coordinates of the SNR [deg] (default: (0, 0))")


# CASE 3: tex-r slice
xr_parser = subparsers.add_parser(
    'tex-r', help="t_extra-ratio parameter space slice")

xr_parser.add_argument('-L', '--Lpk', '--L_peak',
                       default=(10.**ct._mu_log10_Lpk_),
                       type=float,
                       help="The peak luminosity of the SNR, at the Bietenholz frequency [erg/s/Hz] (default: 10^25.5)")
xr_parser.add_argument('-p', '--tpk', '--t_peak',
                       default=(10.**ct._mu_log10_tpk_),
                       type=float,
                       help="The peak time of the SNR [days] (default: 10^1.7)")
xr_parser.add_argument('-D', '--distance', '--dist',
                       default=1.,
                       type=float,
                       help="The distance to the source [kpc] (default: 1)")

# WARNING: when we allow t_extra to be non-zero, that means t_age
# needs to saturate the end of adiabatic phase. The only use case for
# this flag is to test the impact of adiabatic phase duration on extra
# old SNRs

xr_parser.add_argument('-s', '--S0', '--irrad', '--flux',
                       default=None,
                       type=float,
                       help="Debug only: the SNR spectral irradiance at the end of the adiabatic phase [Jy] (default: None)")

xr_parser.add_argument('-t', '--t_signal', '--t0',
                       default=1.e4,  # set to 10k years here
                       type=float,
                       help="Debug only: the age of the SNR signal [years] (default: 10^4)")

# end of warning

xr_parser.add_argument('-lb', '--coords', '--long_lat',
                       default=(0., 0.),
                       type=float,
                       nargs=2,
                       help="The galactic coordinates of the SNR [deg] (default: (0, 0))")


# CASE 4: l-D slice
lD_parser = subparsers.add_parser(
    'l-D', help="l-distance parameter space slice")

lD_parser.add_argument('-L', '--Lpk', '--L_peak',
                       default=(10.**ct._mu_log10_Lpk_),
                       type=float,
                       help="The peak luminosity of the SNR, at the Bietenholz frequency [erg/s/Hz] (default: 10^25.5)")
lD_parser.add_argument('-p', '--tpk', '--t_peak',
                       default=(10.**ct._mu_log10_tpk_),
                       type=float,
                       help="The peak time of the SNR [days] (default: 10^1.7)")
lD_parser.add_argument('-r', '--tt_ratio', '--ratio',
                       default=30.,
                       type=float,
                       help="The ratio of t_trans/t_pk (default: 30)")
lD_parser.add_argument('-s', '--S0', '--irrad', '--flux',
                       default=None,
                       type=float,
                       help="The SNR spectral irradiance at the end of the adiabatic phase [Jy] (default: None)")
lD_parser.add_argument('-t', '--t_signal', '--t0',
                       default=None,
                       type=float,
                       help="The age of the SNR signal [years] (default: None)")
lD_parser.add_argument('-x', '--t_extra', '--extra',
                       default=0.,
                       type=float,
                       help="The extra age of the SNR, after the adiabatic phase [years] (default: 0)")
lD_parser.add_argument('-b', '--lat', '--latitude',
                       default=0.,
                       type=float,
                       help="The galactic latitude of the SNR [deg] (default: 0)")


# CASE 5: l-b slice
lb_parser = subparsers.add_parser(
    'l-b', help="longitude-latitude paremeter space slice")

lb_parser.add_argument('-L', '--Lpk', '--L_peak',
                       default=(10.**ct._mu_log10_Lpk_),
                       type=float,
                       help="The peak luminosity of the SNR, at the Bietenholz frequency [erg/s/Hz] (default: 10^25.5)")
lb_parser.add_argument('-p', '--tpk', '--t_peak',
                       default=(10.**ct._mu_log10_tpk_),
                       type=float,
                       help="The peak time of the SNR [days] (default: 10^1.7)")
lb_parser.add_argument('-D', '--distance', '--dist',
                       default=1.,
                       type=float,
                       help="The distance to the source [kpc] (default: 1)")
lb_parser.add_argument('-r', '--tt_ratio', '--ratio',
                       default=30.,
                       type=float,
                       help="The ratio of t_trans/t_pk (default: 30)")
lb_parser.add_argument('-s', '--S0', '--irrad', '--flux',
                       default=None,
                       type=float,
                       help="The SNR spectral irradiance at the end of the adiabatic phase [Jy] (default: None)")
lb_parser.add_argument('-t', '--t_signal', '--t0',
                       default=None,
                       type=float,
                       help="The age of the SNR signal [years] (default: None)")
lb_parser.add_argument('-x', '--t_extra', '--extra',
                       default=0.,
                       type=float,
                       help="The extra age of the SNR, after the adiabatic phase [years] (default: 0)")


# CASE 6: t-D slice
tD_parser = subparsers.add_parser(
    't-D', help="distance-t_total parameter space slice")

tD_parser.add_argument('-L', '--Lpk', '--L_peak',
                       default=(10.**ct._mu_log10_Lpk_),
                       type=float,
                       help="The peak luminosity of the SNR, at the Bietenholz frequency [erg/s/Hz] (default: 10^25.5)")
tD_parser.add_argument('-p', '--tpk', '--t_peak',
                       default=(10.**ct._mu_log10_tpk_),
                       type=float,
                       help="The peak time of the SNR [days] (default: 10^1.7)")
tD_parser.add_argument('-r', '--tt_ratio', '--ratio',
                       default=30.,
                       type=float,
                       help="The ratio of t_trans/t_pk (default: 30)")
tD_parser.add_argument('-s', '--S0', '--irrad', '--flux',
                       default=None,
                       type=float,
                       help="Debug only: the SNR spectral irradiance at the end of the adiabatic phase [Jy] (default: None)")
tD_parser.add_argument('-lb', '--coords', '--long_lat',
                       default=(0., 0.),
                       type=float,
                       nargs=2,
                       help="The galactic coordinates of the SNR [deg] (default: (0, 0))")


# CASE 7: l-t slice
lt_parser = subparsers.add_parser(
    'l-t', help="longitude-t_total parameter space slice")

lt_parser.add_argument('-L', '--Lpk', '--L_peak',
                       default=(10.**ct._mu_log10_Lpk_),
                       type=float,
                       help="The peak luminosity of the SNR, at the Bietenholz frequency [erg/s/Hz] (default: 10^25.5)")
lt_parser.add_argument('-p', '--tpk', '--t_peak',
                       default=(10.**ct._mu_log10_tpk_),
                       type=float,
                       help="The peak time of the SNR [days] (default: 10^1.7)")
lt_parser.add_argument('-r', '--tt_ratio', '--ratio',
                       default=30.,
                       type=float,
                       help="The ratio of t_trans/t_pk (default: 30)")
lt_parser.add_argument('-D', '--distance', '--dist',
                       default=1.,
                       type=float,
                       help="The distance to the source [kpc] (default: 1)")
lt_parser.add_argument('-s', '--S0', '--irrad', '--flux',
                       default=None,
                       type=float,
                       help="Debug only: the SNR spectral irradiance at the end of the adiabatic phase [Jy] (default: None)")
lt_parser.add_argument('-b', '--lat', '--latitude',
                       default=0.,
                       type=float,
                       help="The galactic latitude of the SNR [deg] (default: 0)")


# CASE 8: t-b slice
tb_parser = subparsers.add_parser(
    't-b', help="longitude-t_total parameter space slice")

tb_parser.add_argument('-L', '--Lpk', '--L_peak',
                       default=(10.**ct._mu_log10_Lpk_),
                       type=float,
                       help="The peak luminosity of the SNR, at the Bietenholz frequency [erg/s/Hz] (default: 10^25.5)")
tb_parser.add_argument('-p', '--tpk', '--t_peak',
                       default=(10.**ct._mu_log10_tpk_),
                       type=float,
                       help="The peak time of the SNR [days] (default: 10^1.7)")
tb_parser.add_argument('-r', '--tt_ratio', '--ratio',
                       default=30.,
                       type=float,
                       help="The ratio of t_trans/t_pk (default: 30)")
tb_parser.add_argument('-D', '--distance', '--dist',
                       default=1.,
                       type=float,
                       help="The distance to the source [kpc] (default: 1)")
tb_parser.add_argument('-s', '--S0', '--irrad', '--flux',
                       default=None,
                       type=float,
                       help="Debug only: the SNR spectral irradiance at the end of the adiabatic phase [Jy] (default: None)")
tb_parser.add_argument('-l', '--longitude',
                       default=0.,
                       type=float,
                       help="The galactic longitude of the SNR [deg] (default: 0)")


# Parsing arguments:
args = parser.parse_args()

if args.verbose:  # Printing arguments
    print("Parameter slice: ", args.slice)
    print("Arguments: ", args._get_kwargs())


# Need nuB!
if args.nuB == None:
    raise Exception(
        "Please pass a float value for the Bietenholz frequency --nuB [GHz].")
if args.Nsteps == None:
    raise Exception(
        "Please pass an int value for the number of steps --Nsteps.")

if args.slice in ["Lpk-tpk", "tex-r", "l-D", "l-b"]:
    if args.S0 != None and args.t_signal != None:
        raise Exception(
            "Cannot pass both --S0 and --t_signal. One is solved in terms of the other. Pick one.")
    if args.S0 == None and args.t_signal == None:
        raise Exception("Need to pass either --S0 and --t_signal. Pick one.")
    if args.S0 != None and args.t_signal == None:
        flg_s, flg_t = True, False
    elif args.S0 == None and args.t_signal != None:
        flg_s, flg_t = False, True

# Defining the Run ID variable
run_id = args.run

# -------------------------------------------------

###########################
# DIRECTORIES & FILE NAME #
###########################

# Making directories:
try:
    os.makedirs("./output/custom_snr/")
except:
    pass


# Creating the appropriate slice directory:
folder = "./output/custom_snr/"+args.slice+"/"
try:
    os.makedirs(folder)
except:
    pass


# Defining the filename
filename = "custom_"+args.slice

if args.verbose:
    print(filename)

# -------------------------------------------------

############
# ROUTINES #
############

# Defining a custom SNR
snr = dt.SuperNovaRemnant()

snr.set_name('custom')
snr.set_spectral(args.alpha, is_certain='yes')
snr.set_sr(args.sz)

# Defining some useful quantities:
# Sedov-Taylor analytic formula for adiabatic expansion index:
gamma = ap.gamma_from_alpha(args.alpha)
# Correction from the fact that the Bietenholz frequency is not the pivot frequency [1 GHz]:
from_Bieten_to_pivot = (1./args.nuB)**-args.alpha
# Axion fixed params:
ga_ref = 1.e-10  # [GeV^-1]

# (rough) number of steps in arrays:
Nsteps = args.Nsteps

# TODO: for each slice perform the parameter scan routine

# A slice parA-parB denotes the x-y axis; x-array will have Nsteps+1 points, while y-array will have Nsteps+2 points.
# Obviously, we need to start with the y-array (rows) and then proceed to the x-array (columns) for easier plotting

# CASE 1:
if args.slice == "Lpk-tpk":
    # Defining the arrays
    Nsigs = 3.  # number of standard deviations from the Bietenholz's mean to scan
    # x-array:
    Lpk_arr = np.logspace(ct._mu_log10_Lpk_-Nsigs*ct._sig_log10_Lpk_,
                          ct._mu_log10_Lpk_+Nsigs*ct._sig_log10_Lpk_, Nsteps+1)
    # y-array:
    tpk_arr = np.logspace(ct._mu_log10_tpk_-Nsigs*ct._sig_log10_tpk_,
                          ct._mu_log10_tpk_+Nsigs*ct._sig_log10_tpk_, Nsteps+2)
    new_Lpk_arr = np.copy(Lpk_arr)  # copying peak luminosity array
    # correcting L_peak by switching from the Bietenholz to the pivot frequencies
    new_Lpk_arr *= from_Bieten_to_pivot

    # Saving arrays
    np.savetxt(folder+"run_%d_Lpk_arr_x.txt" % run_id, Lpk_arr)
    np.savetxt(folder+"run_%d_tpk_arr_y.txt" % run_id, tpk_arr)

    # Updating SNR parameters:
    snr.set_coord(l=args.coords[0], b=args.coords[1], sign=None)
    snr.set_distance(args.distance)  # no_dist will be off automatically

    area = 4.*pi*(snr.distance*ct._kpc_over_cm_)**2.  # [cm^2]

    if flg_t:
        snr.set_age(args.t_signal)

    elif flg_s:
        snr.set_flux_density(
            args.S0, is_flux_certain='let us assume it is certain')
        L0 = snr.get_luminosity()  # [cgs]

    # Result grids:
    sn_Gr = []
    snu_Gr = []
    if flg_s:
        tsig_Gr = []
    elif flg_t:
        s0_Gr = []

    # Commence the routine:
    # y-array:
    for tpk in tqdm(tpk_arr):

        t_trans = args.tt_ratio*(tpk/365.)

        row_a, row_b, row_c = [], [], []

        # x-array:
        for new_Lpk in new_Lpk_arr:

            lightcurve_params = {'t_peak': tpk,
                                 'L_peak': new_Lpk,
                                 't_trans': t_trans}

            if flg_t:
                # Updating lightcurve parameters
                t_signal = args.t_signal
                lightcurve_params.update({'t_age': t_signal})
                # Computing L0 and S0
                local_source = {'gamma': gamma, 't_age': t_signal,
                                'L_peak': new_Lpk, 't_peak': tpk, 't_trans': t_trans}
                L0 = ap.L_source(t_signal, **local_source)  # [cgs]
                S0 = L0/ct._Jy_over_cgs_irrad_/area  # [Jy]

            elif flg_s:
                # Computing t_age
                t_signal = ap.tage_compute(new_Lpk, tpk, t_trans, L0, gamma)
                # Updating lightcurve parameters
                lightcurve_params.update({'L_today': L0})

            # Skipping non-sensical parameters:
            if L0 > new_Lpk:  # non-sensical luminosities
                continue
            if t_signal < t_trans:  # non-sensical t_trans (i.e. tpk)
                continue

            # Snu kwargs
            max_steps = int(3*(t_signal) + 1)
            snu_echo_kwargs = {'tmin_default': None,
                               'Nt': min(max_steps, 100001),
                               'xmin': ct._au_over_kpc_,
                               'xmax_default': 100.,
                               'use_quad': False,
                               'lin_space': False,
                               'Nint': min(max_steps, 100001),
                               't_extra_old': args.t_extra}

            # data:
            data = {'deltaE_over_E': 1.e-3,
                    'f_Delta': 0.721,
                    'exper': 'SKA',
                    'total_observing_time': 100.,
                    'verbose': 0,
                    'average': True}

            # computing routine
            z, new_output = md.snr_routine(pt.ma_from_nu(1.), ga_ref,
                                           snr,
                                           lightcurve_params=lightcurve_params,
                                           snu_echo_kwargs=snu_echo_kwargs,
                                           data=data,
                                           output_all=True)

            signal_Snu = new_output['signal_Snu']
            del new_output

            if args.verbose:
                print("S/N=", z)

            # building rows
            row_a.append(z)  # signal-to-noise ratio
            row_b.append(signal_Snu)  # signal S_nu
            if flg_t:
                row_c.append(S0)  # S0
            elif flg_s:
                row_c.append(t_signal)  # t_signal

            # end of routine for fixed Lpk

        # appending finished Lpk rows
        sn_Gr.append(row_a)
        snu_Gr.append(row_b)
        if flg_t:
            s0_Gr.append(row_c)
        elif flg_s:
            tsig_Gr.append(row_c)

        # end of routine for fixed tpk

    # converting grids to arrays
    sn_Gr = np.array(sn_Gr)
    snu_Gr = np.array(snu_Gr)
    if flg_t:
        s0_Gr = np.array(s0_Gr)
    if flg_s:
        tsig_Gr = np.array(tsig_Gr)

    # saving grids
    np.savetxt(folder+"/run_%d_sn_" % run_id + filename+".txt", sn_Gr,
               delimiter=",", fmt="%s")  # signal/noise
    np.savetxt(folder+"/run_%d_echo_" % run_id+filename+".txt",
               snu_Gr, delimiter=",", fmt="%s")  # Snu of echo signal
    if flg_t:
        np.savetxt(folder+"/run_%d_s0_" % run_id + filename+".txt", s0_Gr,
                   delimiter=",", fmt="%s")  # S0
    if flg_s:
        np.savetxt(folder+"/run_%d_tsig_" % run_id+filename+".txt", tsig_Gr,
                   delimiter=",", fmt="%s")  # t signal

    # end of Lpk-tpk routine


# CASE 2:
elif args.slice == "tsig-r":
    # Defining the arrays
    # x-array:
    tsig_arr = np.logspace(log10(10.*(args.tpk/365.)), 4., Nsteps+1)
    # y-array:
    ratio_arr = np.linspace(10., 100., Nsteps+2)

    # Saving arrays
    np.savetxt(folder+"run_%d_tsig_arr_x.txt" % run_id, tsig_arr)
    np.savetxt(folder+"run_%d_ratio_arr_y.txt" % run_id, ratio_arr)

    # New L_peak (corrected from Bietenholz frequency to to 1 GHz):
    new_Lpk = args.Lpk*from_Bieten_to_pivot

    # Updating SNR parameters:
    snr.set_coord(l=args.coords[0], b=args.coords[1], sign=None)
    snr.set_distance(args.distance)  # no_dist will be off automatically

    area = 4.*pi*(snr.distance*ct._kpc_over_cm_)**2.  # [cm^2]

    # Result grids:
    sn_Gr = []
    snu_Gr = []
    s0_Gr = []

    # Commence the routine:
    # y-array:
    for tt_ratio in tqdm(ratio_arr):

        t_trans = tt_ratio*(args.tpk/365.)

        row_a, row_b, row_c = [], [], []

        # x-array:
        for t_signal in tsig_arr:

            # defining lightcurve parameters:
            lightcurve_params = {'t_peak': args.tpk,
                                 'L_peak': new_Lpk,
                                 't_trans': t_trans,
                                 't_age': t_signal}

            # computing L0 and S0
            local_source = {'gamma': gamma, 't_age': t_signal,
                            'L_peak': new_Lpk, 't_peak': args.tpk, 't_trans': t_trans}
            L0 = ap.L_source(t_signal, **local_source)  # [cgs]
            S0 = L0/ct._Jy_over_cgs_irrad_/area  # [Jy]
            snr.set_age(t_signal)
            snr.set_flux_density(S0)

            # Snu kwargs
            max_steps = int(3*(t_signal) + 1)
            snu_echo_kwargs = {'tmin_default': None,
                               'Nt': min(max_steps, 100001),
                               'xmin': ct._au_over_kpc_,
                               'xmax_default': 100.,
                               'use_quad': False,
                               'lin_space': False,
                               'Nint': min(max_steps, 100001),
                               't_extra_old': args.t_extra}

            # data:
            data = {'deltaE_over_E': 1.e-3,
                    'f_Delta': 0.721,
                    'exper': 'SKA',
                    'total_observing_time': 100.,
                    'verbose': 0,
                    'average': True}

            # computing routine
            z, new_output = md.snr_routine(pt.ma_from_nu(1.), ga_ref,
                                           snr,
                                           lightcurve_params=lightcurve_params,
                                           snu_echo_kwargs=snu_echo_kwargs,
                                           data=data,
                                           output_all=True)

            signal_Snu = new_output['signal_Snu']
            del new_output

            if args.verbose:
                print("S/N=", z)

            # building rows
            row_a.append(z)  # signal-to-noise ratio
            row_b.append(signal_Snu)  # signal S_nu
            row_c.append(S0)  # S0

            # end of routine for fixed t_signal

        # appending finished t_signal rows
        sn_Gr.append(row_a)
        snu_Gr.append(row_b)
        s0_Gr.append(row_c)

        # end of routine for fixed tt_ratio

    # converting grids to arrays
    sn_Gr = np.array(sn_Gr)
    snu_Gr = np.array(snu_Gr)
    s0_Gr = np.array(s0_Gr)

    # saving grids
    np.savetxt(folder+"/run_%d_sn_" %
               run_id + filename+".txt", sn_Gr, delimiter=",")
    np.savetxt(folder+"/run_%d_echo_" %
               run_id+filename+".txt", snu_Gr, delimiter=",")
    np.savetxt(folder+"/run_%d_s0_" %
               run_id+filename+".txt", s0_Gr, delimiter=",")

    print(len(sn_Gr))

    # end of tsig-r routine


# CASE 3:
elif args.slice == "tex-r":
    # Defining the arrays
    # x-array:
    tex_arr = np.logspace(3, 5., Nsteps+1)
    # y-array:
    ratio_arr = np.logspace(1., 2., Nsteps+2)

    # Saving arrays
    np.savetxt(folder+"run_%d_tex_arr_x.txt" % run_id, tex_arr)
    np.savetxt(folder+"run_%d_ratio_arr_y.txt" % run_id, ratio_arr)

    # New L_peak (corrected from Bietenholz frequency to to 1 GHz):
    new_Lpk = args.Lpk*from_Bieten_to_pivot

    # Updating SNR parameters:
    snr.set_coord(l=args.coords[0], b=args.coords[1], sign=None)
    snr.set_distance(args.distance)  # no_dist will be off automatically

    area = 4.*pi*(snr.distance*ct._kpc_over_cm_)**2.  # [cm^2]

    if flg_t:
        snr.set_age(args.t_signal)

    elif flg_s:
        snr.set_flux_density(args.S0)
        L0 = snr.get_luminosity()  # [cgs]

    # Result grids:
    sn_Gr = []
    snu_Gr = []
    if flg_s:
        tsig_Gr = []
    elif flg_t:
        s0_Gr = []

    # y-array:
    for tt_ratio in tqdm(ratio_arr):

        t_trans = tt_ratio*(args.tpk/365.)

        # defining lightcurve parameters:
        lightcurve_params = {'t_peak': args.tpk,
                             'L_peak': new_Lpk,
                             't_trans': t_trans}

        if flg_t:
            # Updating lightcurve parameters
            t_signal = args.t_signal
            lightcurve_params.update({'t_age': t_signal})
            # Computing L0 and S0
            local_source = {'gamma': gamma, 't_age': t_signal,
                            'L_peak': new_Lpk, 't_peak': args.tpk, 't_trans': t_trans}
            L0 = ap.L_source(t_signal, **local_source)  # [cgs]
            S0 = L0/ct._Jy_over_cgs_irrad_/area  # [Jy]
            # Appending rows
            s0_Gr.append(S0*np.ones_like(tex_arr))  # S0

        elif flg_s:
            # Computing t_age
            t_signal = ap.tage_compute(new_Lpk, args.tpk, t_trans, L0, gamma)
            # Updating lightcurve parameters
            lightcurve_params.update({'L_today': L0})
            # Appending rows
            tsig_Gr.append(t_signal*np.ones_like(tex_arr))  # t_signal

        # Skipping non-sensical parameters:
        if L0 > new_Lpk:  # non-sensical luminosities
            continue
        if t_signal < t_trans:  # non-sensical t_trans (i.e. tpk)
            continue

        # Snu kwargs
        max_steps = int(3*(t_signal) + 1)
        snu_echo_kwargs = {'tmin_default': None,
                           'Nt': min(max_steps, 100001),
                           'xmin': ct._au_over_kpc_,
                           'xmax_default': 100.,
                           'use_quad': False,
                           'lin_space': False,
                           'Nint': min(max_steps, 100001)}

        # data:
        data = {'deltaE_over_E': 1.e-3,
                'f_Delta': 0.721,
                'exper': 'SKA',
                'total_observing_time': 100.,
                'verbose': 0,
                'average': True}

        row_a, row_b = [], []

        # x-array:
        for t_extra in tex_arr:

            snu_echo_kwargs.update({'t_extra_old': t_extra})

            # computing routine
            z, new_output = md.snr_routine(pt.ma_from_nu(1.), ga_ref,
                                           snr,
                                           lightcurve_params=lightcurve_params,
                                           snu_echo_kwargs=snu_echo_kwargs,
                                           data=data,
                                           output_all=True)

            signal_Snu = new_output['signal_Snu']
            del new_output

            if args.verbose:
                print("S/N=", z)

            # building rows
            row_a.append(z)  # signal-to-noise ratio
            row_b.append(signal_Snu)  # signal S_nu

            # end of routine for fixed t_extra

        # appending finished t_extra rows
        sn_Gr.append(row_a)
        snu_Gr.append(row_b)

        # end of routine for fixed tt_ratio

    # converting grids to arrays
    sn_Gr = np.array(sn_Gr)
    snu_Gr = np.array(snu_Gr)

    # saving grids
    np.savetxt(folder+"/run_%d_sn_" %
               run_id+filename+".txt", sn_Gr, delimiter=",")
    np.savetxt(folder+"/run_%d_echo_" %
               run_id+filename+".txt", snu_Gr, delimiter=",")
    if flg_t:
        np.savetxt(folder+"/run_%d_s0_" %
                   run_id+filename, s0_Gr, delimiter=",")
    if flg_s:
        np.savetxt(folder+"/run_%d_tsig_" %
                   run_id+filename, tsig_Gr, delimiter=",")

    # end of tex-r routine

# CASE 4:
elif args.slice == "l-D":
    # Defining the arrays
    # x-array:
    long_arr = np.linspace(0., 360., Nsteps+1)
    # y-array:
    dist_arr = np.logspace(-1, 0.5, Nsteps+2)

    # Saving arrays
    np.savetxt(folder+"run_%d_long_arr_x.txt" % run_id, long_arr)
    np.savetxt(folder+"run_%d_dist_arr_y.txt" % run_id, dist_arr)

    # New L_peak (corrected from Bietenholz frequency to to 1 GHz):
    new_Lpk = args.Lpk*from_Bieten_to_pivot

    # Updating SNR parameters:
    snr.set_coord(b=args.lat, l=None, sign=None)
    # the following is to turn off no_dist but still able to cause an exception if not updated again
    snr.set_distance(0)

    if flg_t:
        snr.set_age(args.t_signal)
    elif flg_s:
        snr.set_flux_density(args.S0)

    # defining lightcurve parameters:
    t_trans = (args.tt_ratio)*(args.tpk/365.)
    lightcurve_params = {'t_peak': args.tpk,
                         'L_peak': new_Lpk,
                         't_trans': t_trans}

    # Result grids:
    sn_Gr = []
    snu_Gr = []
    if flg_s:
        tsig_Gr = []
    elif flg_t:
        s0_Gr = []

    # y-array:
    for D in tqdm(dist_arr):

        snr.set_distance(D)
        area = 4.*pi*(snr.distance*ct._kpc_over_cm_)**2.  # [cm^2]

        row_a, row_b, row_c = [], [], []

        # x-array:
        for l in long_arr:

            snr.set_coord(l=l, b=None, sign=None)

            if flg_t:
                # Updating lightcurve parameters
                t_signal = args.t_signal
                lightcurve_params.update({'t_age': t_signal})
                # Computing L0 and S0
                local_source = {'gamma': gamma, 't_age': t_signal,
                                'L_peak': new_Lpk, 't_peak': args.tpk, 't_trans': t_trans}
                L0 = ap.L_source(t_signal, **local_source)  # [cgs]
                S0 = L0/ct._Jy_over_cgs_irrad_/area  # [Jy]

            elif flg_s:
                # Computing L0
                L0 = (snr.snu_at_1GHz*ct._Jy_over_cgs_irrad_)*area  # [cgs]
                # Computing t_age
                t_signal = ap.tage_compute(
                    new_Lpk, args.tpk, t_trans, L0, gamma)
                # Updating lightcurve parameters
                lightcurve_params.update({'L_today': L0})

            # Skipping non-sensical parameters:
            if L0 > new_Lpk:  # non-sensical luminosities
                continue
            if t_signal < t_trans:  # non-sensical t_trans (i.e. tpk)
                continue

            # Snu kwargs
            max_steps = int(3*(t_signal) + 1)
            snu_echo_kwargs = {'tmin_default': None,
                               'Nt': min(max_steps, 100001),
                               'xmin': ct._au_over_kpc_,
                               'xmax_default': 100.,
                               'use_quad': False,
                               'lin_space': False,
                               'Nint': min(max_steps, 100001),
                               't_extra_old': args.t_extra}

            # data:
            data = {'deltaE_over_E': 1.e-3,
                    'f_Delta': 0.721,
                    'exper': 'SKA',
                    'total_observing_time': 100.,
                    'verbose': 0,
                    'average': True}

            # computing routine
            z, new_output = md.snr_routine(pt.ma_from_nu(1.), ga_ref,
                                           snr,
                                           lightcurve_params=lightcurve_params,
                                           snu_echo_kwargs=snu_echo_kwargs,
                                           data=data,
                                           output_all=True)

            signal_Snu = new_output['signal_Snu']
            del new_output

            if args.verbose:
                print("S/N=", z)

            # building rows
            row_a.append(z)  # signal-to-noise ratio
            row_b.append(signal_Snu)  # signal S_nu
            if flg_t:
                row_c.append(S0)  # S0
            elif flg_s:
                row_c.append(t_signal)  # t_signal

            # end of routine for fixed l

        # appending finished l rows
        sn_Gr.append(row_a)
        snu_Gr.append(row_b)
        if flg_t:
            s0_Gr.append(row_c)
        elif flg_s:
            tsig_Gr.append(row_c)

        # end of routine for fixed D

    # converting grids to arrays
    sn_Gr = np.array(sn_Gr)
    snu_Gr = np.array(snu_Gr)
    if flg_t:
        s0_Gr = np.array(s0_Gr)
    if flg_s:
        tsig_Gr = np.array(tsig_Gr)

    # saving grids
    np.savetxt(folder+"/run_%d_sn_" %
               run_id+filename+".txt", sn_Gr, delimiter=",")
    np.savetxt(folder+"/run_%d_echo_" %
               run_id+filename+".txt", snu_Gr, delimiter=",")
    if flg_t:
        np.savetxt(folder+"/run_%d_s0_" %
                   run_id+filename, s0_Gr, delimiter=",")
    if flg_s:
        np.savetxt(folder+"/run_%d_tsig_" %
                   run_id+filename, tsig_Gr, delimiter=",")

    # end of l-D routine


# CASE 5:
elif args.slice == "l-b":
    # Defining the arrays
    # x-array:
    long_arr = np.linspace(0., 360., Nsteps+1)
    # y-array:
    lat_arr = np.linspace(-90., 90., Nsteps+2)

    # Saving arrays
    np.savetxt(folder+"run_%d_long_arr_x.txt" % run_id, long_arr)
    np.savetxt(folder+"run_%d_lat_arr_y.txt" % run_id, lat_arr)

    # New L_peak (corrected from Bietenholz frequency to to 1 GHz):
    new_Lpk = args.Lpk*from_Bieten_to_pivot

    # Updating SNR parameters:
    snr.set_distance(args.distance)

    if flg_t:
        snr.set_age(args.t_signal)
    elif flg_s:
        snr.set_flux_density(args.S0)

    area = 4.*pi*(snr.distance*ct._kpc_over_cm_)**2.  # [cm^2]
    # defining lightcurve parameters:
    t_trans = (args.tt_ratio)*(args.tpk/365.)
    lightcurve_params = {'t_peak': args.tpk,
                         'L_peak': new_Lpk,
                         't_trans': t_trans}

    if flg_t:
        # Updating lightcurve parameters
        t_signal = args.t_signal
        lightcurve_params.update({'t_age': t_signal})
        # Computing L0 and S0
        local_source = {'gamma': gamma, 't_age': t_signal,
                        'L_peak': new_Lpk, 't_peak': args.tpk, 't_trans': t_trans}
        L0 = ap.L_source(t_signal, **local_source)  # [cgs]
        S0 = L0/ct._Jy_over_cgs_irrad_/area  # [Jy]

    elif flg_s:
        # Computing L0
        L0 = (snr.snu_at_1GHz*ct._Jy_over_cgs_irrad_)*area  # [cgs]
        # Computing t_age
        t_signal = ap.tage_compute(new_Lpk, args.tpk, t_trans, L0, gamma)
        # Updating lightcurve parameters
        lightcurve_params.update({'L_today': L0})

    # Skipping non-sensical parameters:
    if L0 > new_Lpk:  # non-sensical luminosities
        sys.exit("EXITING: L0 = {} > Lpk = {}".format(L0, new_Lpk))
    if t_signal < t_trans:  # non-sensical t_trans (i.e. tpk)
        sys.exit("EXITING: t_signal = {} < t_trans = {}".format(t_signal, t_trans))

    # Snu kwargs
    max_steps = int(3*(t_signal) + 1)
    snu_echo_kwargs = {'tmin_default': None,
                       'Nt': min(max_steps, 100001),
                       'xmin': ct._au_over_kpc_,
                       'xmax_default': 100.,
                       'use_quad': False,
                       'lin_space': False,
                       'Nint': min(max_steps, 100001),
                       't_extra_old': args.t_extra}

    # data:
    data = {'deltaE_over_E': 1.e-3,
            'f_Delta': 0.721,
            'exper': 'SKA',
            'total_observing_time': 100.,
            'verbose': 0,
            'average': True}

    # Result grids:
    sn_Gr = []
    snu_Gr = []

    # y-array:
    for b in tqdm(lat_arr):

        snr.set_coord(b=b, l=None, sign=None)

        row_a, row_b = [], []

        # x-array
        for l in long_arr:

            snr.set_coord(l=l, b=None, sign=None)

            # computing routine
            z, new_output = md.snr_routine(pt.ma_from_nu(1.), ga_ref,
                                           snr,
                                           lightcurve_params=lightcurve_params,
                                           snu_echo_kwargs=snu_echo_kwargs,
                                           data=data,
                                           output_all=True)

            signal_Snu = new_output['signal_Snu']
            del new_output

            if args.verbose:
                print("S/N=", z)

            # building rows
            row_a.append(z)  # signal-to-noise ratio
            row_b.append(signal_Snu)  # signal S_nu

            # end of routine for fixed l

        # appending finished l rows
        sn_Gr.append(row_a)
        snu_Gr.append(row_b)

        # end of routine for fixed b

    # converting grids to arrays
    sn_Gr = np.array(sn_Gr)
    snu_Gr = np.array(snu_Gr)

    # saving grids
    np.savetxt(folder+"/run_%d_sn_" %
               run_id+filename+".txt", sn_Gr, delimiter=",")
    np.savetxt(folder+"/run_%d_echo_" %
               run_id+filename+".txt", snu_Gr, delimiter=",")

    # end of l-b routine


# CASE 6:
elif args.slice == "t-D":
    # Defining the arrays
    # x-array:
    t_arr = np.logspace(log10(10.*(args.tpk/365.)), 5., Nsteps+1)
    # y-array:
    dist_arr = np.logspace(-1, 0.5, Nsteps+2)

    # Saving arrays:
    np.savetxt(folder+"run_%d_t_arr_x.txt" % run_id, t_arr)
    np.savetxt(folder+"run_%d_dist_arr_y.txt" % run_id, dist_arr)

    # New L_peak (corrected from Bietenholz frequency to to 1 GHz):
    new_Lpk = args.Lpk*from_Bieten_to_pivot

    snr.set_coord(l=args.coords[0], b=args.coords[1], sign=None)

    # set the light curve params
    t_trans = (args.tt_ratio)*(args.tpk/365.)
    lightcurve_params = {'t_peak': args.tpk,
                         'L_peak': new_Lpk,
                         't_trans': t_trans}

    # Result grids:
    sn_Gr = []
    snu_Gr = []
    s0_Gr = []

    # Commence the routine:
    # y-array:
    for D in tqdm(dist_arr):

        # set distance to SNR, which will be used to construct input_param through model.snr_routine()
        snr.set_distance(D)
        area = 4.*pi*(snr.distance*ct._kpc_over_cm_)**2.  # [cm^2]

        row_a, row_b, row_c = [], [], []

        # x-array:
        for t in t_arr:
            if t < ct._time_of_phase_two_:
                t_signal = t
                t_extra = 0.

            else:
                t_signal = ct._time_of_phase_two_
                t_extra = t - ct._time_of_phase_two_

            lightcurve_params['t_age'] = t_signal

            # computing L0 and S0
            local_source = {'gamma': gamma, 't_age': t_signal,
                            'L_peak': new_Lpk, 't_peak': args.tpk, 't_trans': t_trans}
            L0 = ap.L_source(t_signal, **local_source)  # [cgs]
            S0 = L0/ct._Jy_over_cgs_irrad_/area  # [Jy]

            snr.set_age(t_signal)
            snr.set_flux_density(S0)

            # Snu kwargs
            max_steps = int(3*(t_signal) + 1)
            snu_echo_kwargs = {'tmin_default': None,
                               'Nt': min(max_steps, 100001),
                               'xmin': ct._au_over_kpc_,
                               'xmax_default': 100.,
                               'use_quad': False,
                               'lin_space': False,
                               'Nint': min(max_steps, 100001),
                               't_extra_old': t_extra}

            # data:
            data = {'deltaE_over_E': 1.e-3,
                    'f_Delta': 0.721,
                    'exper': 'SKA',
                    'total_observing_time': 100.,
                    'verbose': 0,
                    'average': True}

            # computing routine
            z, new_output = md.snr_routine(pt.ma_from_nu(1.), ga_ref,
                                           snr,
                                           lightcurve_params=lightcurve_params,
                                           snu_echo_kwargs=snu_echo_kwargs,
                                           data=data,
                                           output_all=True)

            signal_Snu = new_output['signal_Snu']
            del new_output

            if args.verbose:
                print("S/N=", z)

            # building rows
            row_a.append(z)  # signal-to-noise ratio
            row_b.append(signal_Snu)  # signal S_nu
            row_c.append(S0)  # S0

            # end of routine for fixed t

        # appending finished t rows
        sn_Gr.append(row_a)
        snu_Gr.append(row_b)
        s0_Gr.append(row_c)

        # end of routine for fixed D

    # converting grids to arrays
    sn_Gr = np.array(sn_Gr)
    snu_Gr = np.array(snu_Gr)
    s0_Gr = np.array(s0_Gr)

    # saving grids
    np.savetxt(folder+"/run_%d_sn_" %
               run_id + filename+".txt", sn_Gr, delimiter=",")
    np.savetxt(folder+"/run_%d_echo_" %
               run_id+filename+".txt", snu_Gr, delimiter=",")
    np.savetxt(folder+"/run_%d_s0_" %
               run_id+filename+".txt", s0_Gr, delimiter=",")

    # end of t-D routine


# CASE 7:
elif args.slice == "l-t":
    # Defining the arrays
    # x-array:
    long_arr = np.linspace(0., 360., Nsteps+1)
    # y-array:
    t_arr = np.logspace(log10(10.*(args.tpk/365.)), 5., Nsteps+2)

    np.savetxt(folder+"run_%d_long_arr_x.txt" % run_id, long_arr)
    np.savetxt(folder+"run_%d_t_arr_y.txt" % run_id, t_arr)

    # New L_peak (corrected from Bietenholz frequency to to 1 GHz):
    new_Lpk = args.Lpk*from_Bieten_to_pivot

    # geometry
    snr.set_distance(args.distance)
    area = 4.*pi*(snr.distance*ct._kpc_over_cm_)**2.  # [cm^2]

    # set the light curve params
    t_trans = (args.tt_ratio)*(args.tpk/365.)
    lightcurve_params = {'t_peak': args.tpk,
                         'L_peak': new_Lpk,
                         't_trans': t_trans}

    # Result grids:
    sn_Gr = []
    snu_Gr = []
    s0_Gr = []

    # Commence the routine:
    # y-array:
    for t in tqdm(t_arr):

        if t < ct._time_of_phase_two_:
            t_signal = t
            t_extra = 0.

        else:
            t_signal = ct._time_of_phase_two_
            t_extra = t - ct._time_of_phase_two_

        # Updating lightcurve parameters:
        lightcurve_params['t_age'] = t_signal

        # computing L0 and S0
        local_source = {'gamma': gamma, 't_age': t_signal,
                        'L_peak': new_Lpk, 't_peak': args.tpk, 't_trans': t_trans}
        L0 = ap.L_source(t_signal, **local_source)  # [cgs]
        S0 = L0/ct._Jy_over_cgs_irrad_/area  # [Jy]
        snr.set_age(t_signal)
        snr.set_flux_density(S0)

        # Snu kwargs
        max_steps = int(3*(t_signal) + 1)
        snu_echo_kwargs = {'tmin_default': None,
                           'Nt': min(max_steps, 100001),
                           'xmin': ct._au_over_kpc_,
                           'xmax_default': 100.,
                           'use_quad': False,
                           'lin_space': False,
                           'Nint': min(max_steps, 100001),
                           't_extra_old': t_extra}

        # data:
        data = {'deltaE_over_E': 1.e-3,
                'f_Delta': 0.721,
                'exper': 'SKA',
                'total_observing_time': 100.,
                'verbose': 0,
                'average': True}

        row_a, row_b, row_c = [], [], []

        # x-array:
        for longitude in (long_arr):

            snr.set_coord(l=longitude, b=args.lat, sign=None)

            # computing routine
            z, new_output = md.snr_routine(pt.ma_from_nu(1.), ga_ref,
                                           snr,
                                           lightcurve_params=lightcurve_params,
                                           snu_echo_kwargs=snu_echo_kwargs,
                                           data=data,
                                           output_all=True)

            signal_Snu = new_output['signal_Snu']
            del new_output

            if args.verbose:
                print("S/N=", z)

            # building rows
            row_a.append(z)  # signal-to-noise ratio
            row_b.append(signal_Snu)  # signal S_nu
            row_c.append(S0)  # S0

            # end of routine for fixed l

        # appending finished l rows
        sn_Gr.append(row_a)
        snu_Gr.append(row_b)
        s0_Gr.append(row_c)

        # end of routine for fixed t

    # converting grids to arrays
    sn_Gr = np.array(sn_Gr)
    snu_Gr = np.array(snu_Gr)
    s0_Gr = np.array(s0_Gr)

    # saving grids
    np.savetxt(folder+"/run_%d_sn_" %
               run_id + filename+".txt", sn_Gr, delimiter=",")
    np.savetxt(folder+"/run_%d_echo_" %
               run_id+filename+".txt", snu_Gr, delimiter=",")
    np.savetxt(folder+"/run_%d_s0_" %
               run_id+filename+".txt", s0_Gr, delimiter=",")

    # end of l-t routine


# CASE 8:
elif args.slice == "t-b":
    # Defining the arrays
    # x-array:
    t_arr = np.logspace(log10(10.*(args.tpk/365.)), 5., Nsteps+1)
    # y-array:
    lat_arr = np.linspace(-90., 90., Nsteps+2)

    np.savetxt(folder+"run_%d_t_arr_x.txt" % run_id, t_arr)
    np.savetxt(folder+"run_%d_lat_arr_y.txt" % run_id, lat_arr)

    # New L_peak (corrected from Bietenholz frequency to to 1 GHz):
    new_Lpk = args.Lpk*from_Bieten_to_pivot

    # geometry
    snr.set_distance(args.distance)
    area = 4.*pi*(snr.distance*ct._kpc_over_cm_)**2.  # [cm^2]

    # set the light curve params
    t_trans = (args.tt_ratio)*(args.tpk/365.)
    lightcurve_params = {'t_peak': args.tpk,
                         'L_peak': new_Lpk,
                         't_trans': t_trans}

    # Result grids:
    sn_Gr = []
    snu_Gr = []
    s0_Gr = []

    # Commence the routine:
    # y-array:
    for latitude in tqdm(lat_arr):

        # set distance to SNR, which will be used to construct input_param through model.snr_routine()
        row_a, row_b, row_c = [], [], []
        snr.set_coord(l=args.longitude, b=latitude, sign=None)

        # x-array:
        for t in t_arr:
            if t < ct._time_of_phase_two_:
                t_signal = t
                t_extra = 0.

            else:
                t_signal = ct._time_of_phase_two_
                t_extra = t - ct._time_of_phase_two_

            lightcurve_params['t_age'] = t_signal

            # computing L0 and S0
            local_source = {'gamma': gamma, 't_age': t_signal,
                            'L_peak': new_Lpk, 't_peak': args.tpk, 't_trans': t_trans}
            L0 = ap.L_source(t_signal, **local_source)  # [cgs]
            S0 = L0/ct._Jy_over_cgs_irrad_/area  # [Jy]
            # snr.__dict__.update({'age': t_signal, 'snu_at_1GHz': S0})
            snr.set_age(t_signal)
            snr.set_flux_density(S0)

            # Snu kwargs
            max_steps = int(3*(t_signal) + 1)
            snu_echo_kwargs = {'tmin_default': None,
                               'Nt': min(max_steps, 100001),
                               'xmin': ct._au_over_kpc_,
                               'xmax_default': 100.,
                               'use_quad': False,
                               'lin_space': False,
                               'Nint': min(max_steps, 100001),
                               't_extra_old': t_extra}

            # data:
            data = {'deltaE_over_E': 1.e-3,
                    'f_Delta': 0.721,
                    'exper': 'SKA',
                    'total_observing_time': 100.,
                    'verbose': 0,
                    'average': True}

            # computing routine
            z, new_output = md.snr_routine(pt.ma_from_nu(1.), ga_ref,
                                           snr,
                                           lightcurve_params=lightcurve_params,
                                           snu_echo_kwargs=snu_echo_kwargs,
                                           data=data,
                                           output_all=True)

            signal_Snu = new_output['signal_Snu']
            del new_output

            if args.verbose:
                print("S/N=", z)

            # building rows
            row_a.append(z)  # signal-to-noise ratio
            row_b.append(signal_Snu)  # signal S_nu
            row_c.append(S0)  # S0

            # end of routine for fixed t

        # appending finished t rows
        sn_Gr.append(row_a)
        snu_Gr.append(row_b)
        s0_Gr.append(row_c)

        # end of routine for fixed b

    # converting grids to arrays
    sn_Gr = np.array(sn_Gr)
    snu_Gr = np.array(snu_Gr)
    s0_Gr = np.array(s0_Gr)

    # saving grids
    np.savetxt(folder+"/run_%d_sn_" %
               run_id + filename+".txt", sn_Gr, delimiter=",")
    np.savetxt(folder+"/run_%d_echo_" %
               run_id+filename+".txt", snu_Gr, delimiter=",")
    np.savetxt(folder+"/run_%d_s0_" %
               run_id+filename+".txt", s0_Gr, delimiter=",")

    # end of t-b routine

####################################
# save log file for future reference
####################################
log_file = os.path.join(folder, "run_%d_log.txt" % run_id)
with open(log_file, 'w') as f:
    f.write('#\n#-------Run info\n#\n')
    f.write('run_id: %d\n' % run_id)
    f.write('running_mode: %s\n' % args.slice)
    f.write('ga_ref: %e\n' % ga_ref)
    # f.write('current_dir: %s\n' % current_dir)

    f.write('#\n#-------SNe Remnant info\n#\n')
    f.write('# Note that for the SNR properties being scanned over only the last value of the scan will also be recorded.')
    for key, entry in snr.__dict__.items():
        f.write('%s: %s\n' % (key, entry))

    f.write('#\n#-------detailed log\n#\n')
    for key, entry in vars(args).items():
        f.write('%s: %s\n' % (key, entry))

#
# examples of the use cases
#
# CASE1:
# python ./run_custom.py --run 1 --nuB 8 --Nsteps 30 Lpk-tpk --dist 0.5 --tt_ratio 30 --t0 1e4 --t_extra 4e4 --long_lat 175 0

# CASE2:
# python ./run_custom.py --run 1 --nuB 8 --Nsteps 30 tsig-r --Lpk 3.16e28 --tpk 50.1 --dist 0.5 --long_lat 175 0

# CASE3:
# python ./run_custom.py --run 1 --nuB 8 --Nsteps 30 tex-r --Lpk 3.16e28 --tpk 50.1 --dist 0.5 --long_lat 175 0

# CASE4:
# python ./run_custom.py --run 1 --nuB 8 --Nsteps 100 l-D --Lpk 3.16e28 --tpk 50.1 --tt_ratio 30 --t0 1e4 --t_extra 4e4 --lat 0

# CASE5:
# python ./run_custom.py --run 1 --nuB 8 --Nsteps 30 l-b --Lpk 3.16e28 --tpk 50.1 --dist 0.5 --tt_ratio 30 --t0 1e4 --t_extra 4e4

# CASE6:
# python ./run_custom.py --run 1 --nuB 8 --Nsteps 30 t-D --Lpk 3.16e28 --tpk 50.1 --tt_ratio 30 -lb 175 0

# CASE7:
# python ./run_custom.py --run 1 --nuB 8 --Nsteps 100 l-t --Lpk 3.16e28 --tpk 50.1 --tt_ratio 30 -D 0.5 -b 0

# CASE8:
# python ./run_custom.py --run 1 --nuB 8 --Nsteps 100 t-b --Lpk 3.16e28 --tpk 50.1 --tt_ratio 30 -D 0.5 -l 175
