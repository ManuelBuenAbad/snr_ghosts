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
                    help="The number of steps in the parameter space arrays")
parser.add_argument('-a', '--alpha',
                    default=0.5,
                    type=float,
                    help="The SNR spectral index")
parser.add_argument('-n', '--nuB', '--nu_Bietenholz',
                    default=None,
                    type=float,
                    help="The Bietenholz frequency [GHz]")
parser.add_argument('-z', '--sz', '--size',
                    default=1.e-4,
                    type=float,
                    help="The size of the source [sr]")
parser.add_argument('-v', '--verbose',
                    action="store_true",
                    help="Verbosity")
parser.add_argument('--run', default=0, type=int, help="The run number")

# defining the subparsers, and sending their names to .slice attribute
subparsers = parser.add_subparsers(
    dest="slice", description="The following subcommand options determine the parameter space slice to be explored.")

# CASE 1: Lpk-tpk slice
Lt_parser = subparsers.add_parser(
    'Lpk-tpk', help="Lpk-tpk parameter space slice")

Lt_parser.add_argument('-D', '--distance', '--dist',
                       default=1.,
                       type=float,
                       help="The distance to the source [kpc]")
Lt_parser.add_argument('-r', '--tt_ratio', '--ratio',
                       default=30.,
                       type=float,
                       help="The ratio of t_trans/t_pk")
Lt_parser.add_argument('-s', '--S0', '--irrad', '--flux',
                       default=None,
                       type=float,
                       help="The SNR spectral irradiance at the end of the adiabatic phase [Jy]")
Lt_parser.add_argument('-t', '--t_signal', '--t0',
                       default=None,
                       type=float,
                       help="The age of the SNR signal [years]")
Lt_parser.add_argument('-x', '--t_extra', '--extra',
                       default=0.,
                       type=float,
                       help="The extra age of the SNR, after the adiabatic phase [years]")
Lt_parser.add_argument('-lb', '--coords', '--long_lat',
                       default=(0., 0.),
                       type=float,
                       nargs=2,
                       help="The galactic coordinates of the SNR [deg]")

# CASE 2: r-tsig slice
tr_parser = subparsers.add_parser(
    'r-tsig', help="ratio-t_signal parameter space slice")

tr_parser.add_argument('-L', '--Lpk', '--L_peak',
                       default=(10.**ct._mu_log10_Lpk_),
                       type=float,
                       help="The peak luminosity of the SNR, at the Bietenholz frequency [erg/s/Hz]")
tr_parser.add_argument('-p', '--tpk', '--t_peak',
                       default=(10.**ct._mu_log10_tpk_),
                       type=float,
                       help="The peak time of the SNR [days]")
tr_parser.add_argument('-D', '--distance', '--dist',
                       default=1.,
                       type=float,
                       help="The distance to the source [kpc]")

# WARNING: when tsig < 1e4, it means the SNR is still going through
# the second phase. Therefore, we should not add extra age to it in
# general. The only use case for this flag is to test the impact of
# adiabatic phase duration on extra old SNRs

tr_parser.add_argument('-x', '--t_extra', '--extra',
                       default=0.,
                       type=float,
                       help="Debug only: the extra age of the SNR, after the adiabatic phase [years]")

# end of warning

tr_parser.add_argument('-lb', '--coords', '--long_lat',
                       default=(0., 0.),
                       type=float,
                       nargs=2,
                       help="The galactic coordinates of the SNR [deg]")

# CASE 3: r-tex slice
xr_parser = subparsers.add_parser(
    'r-tex', help="ratio-t_extra parameter space slice")

xr_parser.add_argument('-L', '--Lpk', '--L_peak',
                       default=(10.**ct._mu_log10_Lpk_),
                       type=float,
                       help="The peak luminosity of the SNR, at the Bietenholz frequency [erg/s/Hz]")
xr_parser.add_argument('-p', '--tpk', '--t_peak',
                       default=(10.**ct._mu_log10_tpk_),
                       type=float,
                       help="The peak time of the SNR [days]")
xr_parser.add_argument('-D', '--distance', '--dist',
                       default=1.,
                       type=float,
                       help="The distance to the source [kpc]")

# WARNING: when we allow t_extra to be non-zero, that means t_age
# needs to saturate the end of adiabatic phase. The only use case for
# this flag is to test the impact of adiabatic phase duration on extra
# old SNRs

xr_parser.add_argument('-s', '--S0', '--irrad', '--flux',
                       default=None,
                       type=float,
                       help="Debug only: the SNR spectral irradiance at the end of the adiabatic phase [Jy]")

xr_parser.add_argument('-t', '--t_signal', '--t0',
                       default=1.e4,  # set to 10k years here
                       type=float,
                       help="Debug only: the age of the SNR signal [years]")

# end of warning

xr_parser.add_argument('-lb', '--coords', '--long_lat',
                       default=(0., 0.),
                       type=float,
                       nargs=2,
                       help="The galactic coordinates of the SNR [deg]")

# CASE 4: l-D slice
lD_parser = subparsers.add_parser(
    'l-D', help="l-distance parameter space slice")

lD_parser.add_argument('-L', '--Lpk', '--L_peak',
                       default=(10.**ct._mu_log10_Lpk_),
                       type=float,
                       help="The peak luminosity of the SNR, at the Bietenholz frequency [erg/s/Hz]")
lD_parser.add_argument('-p', '--tpk', '--t_peak',
                       default=(10.**ct._mu_log10_tpk_),
                       type=float,
                       help="The peak time of the SNR [days]")
lD_parser.add_argument('-r', '--tt_ratio', '--ratio',
                       default=30.,
                       type=float,
                       help="The ratio of t_trans/t_pk")
lD_parser.add_argument('-s', '--S0', '--irrad', '--flux',
                       default=None,
                       type=float,
                       help="The SNR spectral irradiance at the end of the adiabatic phase [Jy]")
lD_parser.add_argument('-t', '--t_signal', '--t0',
                       default=None,
                       type=float,
                       help="The age of the SNR signal [years]")
lD_parser.add_argument('-x', '--t_extra', '--extra',
                       default=0.,
                       type=float,
                       help="The extra age of the SNR, after the adiabatic phase [years]")
lD_parser.add_argument('-b', '--lat', '--latitude',
                       default=0.,
                       type=float,
                       help="The galactic latitude of the SNR [deg]")

# CASE 5: l-b slice
lb_parser = subparsers.add_parser(
    'l-b', help="longitude-latitude paremeter space slice")

lb_parser.add_argument('-L', '--Lpk', '--L_peak',
                       default=(10.**ct._mu_log10_Lpk_),
                       type=float,
                       help="The peak luminosity of the SNR, at the Bietenholz frequency [erg/s/Hz]")
lb_parser.add_argument('-p', '--tpk', '--t_peak',
                       default=(10.**ct._mu_log10_tpk_),
                       type=float,
                       help="The peak time of the SNR [days]")
lb_parser.add_argument('-D', '--distance', '--dist',
                       default=1.,
                       type=float,
                       help="The distance to the source [kpc]")
lb_parser.add_argument('-r', '--tt_ratio', '--ratio',
                       default=30.,
                       type=float,
                       help="The ratio of t_trans/t_pk")
lb_parser.add_argument('-s', '--S0', '--irrad', '--flux',
                       default=None,
                       type=float,
                       help="The SNR spectral irradiance at the end of the adiabatic phase [Jy]")
lb_parser.add_argument('-t', '--t_signal', '--t0',
                       default=None,
                       type=float,
                       help="The age of the SNR signal [years]")
lb_parser.add_argument('-x', '--t_extra', '--extra',
                       default=0.,
                       type=float,
                       help="The extra age of the SNR, after the adiabatic phase [years]")

# CASE 6: D-t slice
Dt_parser = subparsers.add_parser(
    'D-t', help="distance-t_total parameter space slice")

Dt_parser.add_argument('-L', '--Lpk', '--L_peak',
                       default=(10.**ct._mu_log10_Lpk_),
                       type=float,
                       help="The peak luminosity of the SNR, at the Bietenholz frequency [erg/s/Hz]")
Dt_parser.add_argument('-p', '--tpk', '--t_peak',
                       default=(10.**ct._mu_log10_tpk_),
                       type=float,
                       help="The peak time of the SNR [days]")
Dt_parser.add_argument('-r', '--tt_ratio', '--ratio',
                       default=30.,
                       type=float,
                       help="The ratio of t_trans/t_pk")
# Dt_parser.add_argument('-D', '--distance', '--dist',
#                        default=1.,
#                        type=float,
#                        help="The distance to the source [kpc]")

Dt_parser.add_argument('-s', '--S0', '--irrad', '--flux',
                       default=None,
                       type=float,
                       help="Debug only: the SNR spectral irradiance at the end of the adiabatic phase [Jy]")

# Dt_parser.add_argument('-t', '--t_total',
#                        default=None,  # total time
#                        type=float,
#                        help="the total age of the SNR signal [years]")

Dt_parser.add_argument('-lb', '--coords', '--long_lat',
                       default=(0., 0.),
                       type=float,
                       nargs=2,
                       help="The galactic coordinates of the SNR [deg]")


# CASE 7: l-t slice
lt_parser = subparsers.add_parser(
    'l-t', help="longitude-t_total parameter space slice")

lt_parser.add_argument('-L', '--Lpk', '--L_peak',
                       default=(10.**ct._mu_log10_Lpk_),
                       type=float,
                       help="The peak luminosity of the SNR, at the Bietenholz frequency [erg/s/Hz]")
lt_parser.add_argument('-p', '--tpk', '--t_peak',
                       default=(10.**ct._mu_log10_tpk_),
                       type=float,
                       help="The peak time of the SNR [days]")
lt_parser.add_argument('-r', '--tt_ratio', '--ratio',
                       default=30.,
                       type=float,
                       help="The ratio of t_trans/t_pk")

lt_parser.add_argument('-D', '--distance', '--dist',
                       default=1.,
                       type=float,
                       help="The distance to the source [kpc]")

lt_parser.add_argument('-s', '--S0', '--irrad', '--flux',
                       default=None,
                       type=float,
                       help="Debug only: the SNR spectral irradiance at the end of the adiabatic phase [Jy]")

# lt_parser.add_argument('-t', '--t_total',
#                        default=None,  # total time
#                        type=float,
#                        help="the total age of the SNR signal [years]")

lt_parser.add_argument('-b', '--lat', '--latitude',
                       default=0.,
                       type=float,
                       help="The galactic latitude of the SNR [deg]")

# lt_parser.add_argument('-lb', '--coords', '--long_lat',
#                        default=(0., 0.),
#                        type=float,
#                        nargs=2,
#                        help="The galactic coordinates of the SNR [deg]")


# CASE 8: b-t slice
bt_parser = subparsers.add_parser(
    'b-t', help="longitude-t_total parameter space slice")

bt_parser.add_argument('-L', '--Lpk', '--L_peak',
                       default=(10.**ct._mu_log10_Lpk_),
                       type=float,
                       help="The peak luminosity of the SNR, at the Bietenholz frequency [erg/s/Hz]")
bt_parser.add_argument('-p', '--tpk', '--t_peak',
                       default=(10.**ct._mu_log10_tpk_),
                       type=float,
                       help="The peak time of the SNR [days]")
bt_parser.add_argument('-r', '--tt_ratio', '--ratio',
                       default=30.,
                       type=float,
                       help="The ratio of t_trans/t_pk")

bt_parser.add_argument('-D', '--distance', '--dist',
                       default=1.,
                       type=float,
                       help="The distance to the source [kpc]")

bt_parser.add_argument('-s', '--S0', '--irrad', '--flux',
                       default=None,
                       type=float,
                       help="Debug only: the SNR spectral irradiance at the end of the adiabatic phase [Jy]")

# bt_parser.add_argument('-t', '--t_total',
#                        default=None,  # total time
#                        type=float,
#                        help="the total age of the SNR signal [years]")

bt_parser.add_argument('-l', '--longitude',
                       default=0.,
                       type=float,
                       help="The galactic longitude of the SNR [deg]")

# bt_parser.add_argument('-lb', '--coords', '--long_lat',
#                        default=(0., 0.),
#                        type=float,
#                        nargs=2,
#                        help="The galactic coordinates of the SNR [deg]")


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

if args.slice in ["Lpk-tpk", "r-tex", "l-D", "l-b"]:
    if args.S0 != None and args.t_signal != None:
        raise Exception(
            "Cannot pass both --S0 and --t_signal. One is solved in terms of the other. Pick one.")
    if args.S0 == None and args.t_signal == None:
        raise Exception("Need to pass either --S0 and --t_signal. Pick one.")
    if args.S0 != None and args.t_signal == None:
        flg_s, flg_t = True, False
    elif args.S0 == None and args.t_signal != None:
        flg_s, flg_t = False, True

run_id = args.run
# -------------------------------------------------

###############
# DIRECTORIES #
###############

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


# -------------------------------------------------

#############
# FILE NAME #
#############

# Preparing the filename. Order: a-Lpk-tpk-D-r-S0/tsig-tex-lb-nuB-sz
filename = "custom_"+args.slice+"_"

if args.slice == "Lpk-tpk":
    if args.S0 != None:
        filename += "a-{}_D-{}_r-{}_S0-{}_tex-{}_lb-{}-{}_nuB-{}_sz-{:.0e}".format(args.alpha, args.distance, int(
            args.tt_ratio), int(args.S0), int(args.t_extra), int(args.coords[0]), int(args.coords[1]), int(args.nuB), args.sz)
    else:
        filename += "a-{}_D-{}_r-{}_tsig-{}_tex-{}_lb-{}-{}_nuB-{}_sz-{:.0e}".format(args.alpha, args.distance, int(
            args.tt_ratio), int(args.t_signal), int(args.t_extra), int(args.coords[0]), int(args.coords[1]), int(args.nuB), args.sz)

elif args.slice == "r-tsig":
    filename += "a-{}_Lpk-{:.0e}_tpk-{:.0e}_D-{}_tex-{}_lb-{}-{}_nuB-{}_sz-{:.0e}".format(
        args.alpha, args.Lpk, args.tpk, args.distance, int(args.t_extra), int(args.coords[0]), int(args.coords[1]), int(args.nuB), args.sz)

elif args.slice == "r-tex":
    if args.S0 != None:
        filename += "a-{}_Lpk-{:.0e}_tpk-{:.0e}_D-{}_S0-{}_lb-{}-{}_nuB-{}_sz-{:.0e}".format(
            args.alpha, args.Lpk, args.tpk, args.distance, int(args.S0), int(args.coords[0]), int(args.coords[1]), int(args.nuB), args.sz)
    else:
        filename += "a-{}_Lpk-{:.0e}_tpk-{:.0e}_D-{}_tsig-{}_lb-{}-{}_nuB-{}_sz-{:.0e}".format(
            args.alpha, args.Lpk, args.tpk, args.distance, int(args.t_signal), int(args.coords[0]), int(args.coords[1]), int(args.nuB), args.sz)

elif args.slice == "l-D":
    if args.S0 != None:
        filename += "a-{}_Lpk-{:.0e}_tpk-{:.0e}_r-{}_S0-{}_tex-{}_b-{}_nuB-{}_sz-{:.0e}".format(
            args.alpha, args.Lpk, args.tpk, int(args.tt_ratio), int(args.S0), int(args.t_extra), int(args.lat), int(args.nuB), args.sz)
    else:
        filename += "a-{}_Lpk-{:.0e}_tpk-{:.0e}_r-{}_tsig-{}_tex-{}_b-{}_nuB-{}_sz-{:.0e}".format(args.alpha, args.Lpk, args.tpk, int(
            args.tt_ratio), int(args.t_signal), int(args.t_extra), int(args.lat), int(args.nuB), args.sz)

elif args.slice == "l-b":
    if args.S0 != None:
        filename += "a-{}_Lpk-{:.0e}_tpk-{:.0e}_D-{}_r-{}_S0-{}_tex-{}_nuB-{}_sz-{:.0e}".format(
            args.alpha, args.Lpk, args.tpk, args.distance, int(args.tt_ratio), int(args.S0), int(args.t_extra), int(args.nuB), args.sz)
    else:
        filename += "a-{}_Lpk-{:.0e}_tpk-{:.0e}_D-{}_r-{}_tsig-{}_tex-{}_nuB-{}_sz-{:.0e}".format(
            args.alpha, args.Lpk, args.tpk, args.distance, int(args.tt_ratio), int(args.t_signal), int(args.t_extra), int(args.nuB), args.sz)

elif args.slice == "D-t":
    pass

elif args.slice == "l-t":
    pass

elif args.slice == "b-t":
    pass

if args.verbose:
    print(filename)

# -------------------------------------------------

############
# ROUTINES #
############

# Defining a custom SNR
snr = dt.SuperNovaRemnant()
# snr.__dict__ = {'name': 'custom',
#                 'alpha': args.alpha,
#                 'sr': args.sz}
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
# CASE 1:
if args.slice == "Lpk-tpk":
    # Defining the arrays
    Nsigs = 3.  # number of standard deviations from the Bietenholz's mean to scan
    tpk_arr = np.logspace(ct._mu_log10_tpk_-Nsigs*ct._sig_log10_tpk_,
                          ct._mu_log10_tpk_+Nsigs*ct._sig_log10_tpk_, Nsteps+1)
    Lpk_arr = np.logspace(ct._mu_log10_Lpk_-Nsigs*ct._sig_log10_Lpk_,
                          ct._mu_log10_Lpk_+Nsigs*ct._sig_log10_Lpk_, Nsteps+2)
    new_Lpk_arr = np.copy(Lpk_arr)  # copying peak luminosity array
    # correcting L_peak by switching from the Bietenholz to the pivot frequencies
    new_Lpk_arr *= from_Bieten_to_pivot

    # Saving arrays
    # if os.access(folder+"tpk_arr.txt", os.R_OK):
    #     pass
    # else:
    #     np.savetxt(folder+"tpk_arr.txt", tpk_arr)
    # if os.access(folder+"Lpk_arr.txt", os.R_OK):
    #     pass
    # else:
    #     np.savetxt(folder+"Lpk_arr.txt", Lpk_arr)
    # Saving arrays
    np.savetxt(folder+"run_%d_tpk_arr_y.txt" % run_id, tpk_arr)
    np.savetxt(folder+"run_%d_Lpk_arr_x.txt" % run_id, Lpk_arr)

    # Updating SNR parameters:
    # snr.__dict__.update({'l': args.coords[0],
    #                      'b': args.coords[1],
    #                      'distance': args.distance,
    #                      'no_dist': False})
    snr.set_coord(l=args.coords[0], b=args.coords[1], sign=None)
    snr.set_distance(args.distance)  # no_dist will be off automatically

    area = 4.*pi*(snr.distance*ct._kpc_over_cm_)**2.  # [cm^2]

    if flg_t:
        # snr.__dict__.update({'age': args.t_signal})
        snr.set_age(args.t_signal)

    elif flg_s:
        # snr.__dict__.update({'snu_at_1GHz': args.S0})
        snr.set_flux_density(
            args.S0, is_flux_certain='let us assume it is certain')
        # L0 = (snr.snu_at_1GHz*ct._Jy_over_cgs_irrad_)*area  # [cgs]
        L0 = snr.get_luminosity()  # [cgs]

    # Result grids:
    sn_Gr = []
    snu_Gr = []
    if flg_s:
        tsig_Gr = []
    elif flg_t:
        s0_Gr = []

    # Commence the routine:
    for tpk in tqdm(tpk_arr):

        t_trans = args.tt_ratio*(tpk/365.)

        row_a, row_b, row_c = [], [], []

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
            data = {'deltaE_over_E': 1.e-3 * 2.17,
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
elif args.slice == "r-tsig":
    # Defining the arrays
    ratio_arr = np.linspace(10., 100., Nsteps+1)
    tsig_arr = np.logspace(log10(10.*(args.tpk/365.)), 4., Nsteps+2)

    # Saving arrays
    # if os.access(folder+"/ratio_arr.txt", os.R_OK):
    #     pass
    # else:
    #     np.savetxt(folder+"/ratio_arr.txt", ratio_arr)
    # if os.access(folder+"/tsig_arr.txt", os.R_OK):
    #     pass
    # else:
    #     np.savetxt(folder+"/tsig_arr.txt", tsig_arr)
    # Saving arrays
    np.savetxt(folder+"run_%d_ratio_arr_y.txt" % run_id, ratio_arr)
    np.savetxt(folder+"run_%d_tsig_arr_x.txt" % run_id, tsig_arr)

    # New L_peak (corrected from Bietenholz frequency to to 1 GHz):
    new_Lpk = args.Lpk*from_Bieten_to_pivot

    # Updating SNR parameters:
    # snr.__dict__.update({'l': args.coords[0],
    #                      'b': args.coords[1],
    #                      'distance': args.distance,
    #                      'no_dist': False})
    snr.set_coord(l=args.coords[0], b=args.coords[1], sign=None)
    snr.set_distance(args.distance)  # no_dist will be off automatically

    area = 4.*pi*(snr.distance*ct._kpc_over_cm_)**2.  # [cm^2]

    # Result grids:
    sn_Gr = []
    snu_Gr = []
    s0_Gr = []

    # Commence the routine:
    for tt_ratio in tqdm(ratio_arr):

        t_trans = tt_ratio*(args.tpk/365.)

        row_a, row_b, row_c = [], [], []

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
                               't_extra_old': args.t_extra}

            # data:
            data = {'deltaE_over_E': 1.e-3 * 2.17,
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

    # end of r-tsig routine


# CASE 3:
elif args.slice == "r-tex":
    # Defining the arrays
    ratio_arr = np.logspace(1., 2., Nsteps+1)
    tex_arr = np.logspace(3, 5., Nsteps+2)

    # Saving arrays
    # if os.access(folder+"ratio_arr.txt", os.R_OK):
    #     pass
    # else:
    #     np.savetxt(folder+"ratio_arr.txt", ratio_arr)
    # if os.access(folder+"tex_arr.txt", os.R_OK):
    #     pass
    # else:
    #     np.savetxt(folder+"tex_arr.txt", tex_arr)
    # Saving arrays
    np.savetxt(folder+"run_%d_ratio_arr_y.txt" % run_id, ratio_arr)
    np.savetxt(folder+"run_%d_tex_arr_x.txt" % run_id, tex_arr)

    # New L_peak (corrected from Bietenholz frequency to to 1 GHz):
    new_Lpk = args.Lpk*from_Bieten_to_pivot

    # Updating SNR parameters:
    # snr.__dict__.update({'l': args.coords[0],
    #                      'b': args.coords[1],
    #                      'distance': args.distance,
    #                      'no_dist': False})
    snr.set_coord(l=args.coords[0], b=args.coords[1], sign=None)
    snr.set_distance(args.distance)  # no_dist will be off automatically

    area = 4.*pi*(snr.distance*ct._kpc_over_cm_)**2.  # [cm^2]

    if flg_t:
        # snr.__dict__.update({'age': args.t_signal})
        snr.set_age(args.t_signal)

    elif flg_s:
        # snr.__dict__.update({'snu_at_1GHz': args.S0})
        snr.set_flux_density(args.S0)
        # L0 = (snr.snu_at_1GHz*ct._Jy_over_cgs_irrad_)*area  # [cgs]
        L0 = snr.get_luminosity()  # [cgs]

    # Result grids:
    sn_Gr = []
    snu_Gr = []
    if flg_s:
        tsig_Gr = []
    elif flg_t:
        s0_Gr = []

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
        data = {'deltaE_over_E': 1.e-3 * 2.17,
                'f_Delta': 0.721,
                'exper': 'SKA',
                'total_observing_time': 100.,
                'verbose': 0,
                'average': True}

        row_a, row_b = [], []

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

    # end of r-tex routine

# CASE 4:
elif args.slice == "l-D":
    # Defining the arrays
    long_arr = np.linspace(0., 360., Nsteps+1)
    dist_arr = np.logspace(-1, 0.5, Nsteps+2)

    # # Saving arrays
    # if os.access(folder+"long_arr.txt", os.R_OK):
    #     pass
    # else:
    #     np.savetxt(folder+"long_arr.txt", long_arr)
    # if os.access(folder+"dist_arr.txt", os.R_OK):
    #     pass
    # else:
    #     np.savetxt(folder+"dist_arr.txt", dist_arr)
    # Saving arrays
    np.savetxt(folder+"run_%d_long_arr_y.txt" % run_id, long_arr)
    np.savetxt(folder+"run_%d_dist_arr_x.txt" % run_id, dist_arr)

    # New L_peak (corrected from Bietenholz frequency to to 1 GHz):
    new_Lpk = args.Lpk*from_Bieten_to_pivot

    # Updating SNR parameters:
    # snr.__dict__.update({'b': args.lat,
    #                      'no_dist': False})
    snr.set_coord(b=args.lat, l=None, sign=None)
    # the following is to turn off no_dist but still able to cause an exception if not updated again
    snr.set_distance(0)

    if flg_t:
        # snr.__dict__.update({'age': args.t_signal})
        snr.set_age(args.t_signal)
    elif flg_s:
        # snr.__dict__.update({'snu_at_1GHz': args.S0})
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

    for l in tqdm(long_arr):

        row_a, row_b, row_c = [], [], []

        # snr.__dict__.update({'l': l})
        snr.set_coord(l=l, b=None, sign=None)

        for D in dist_arr:

            # snr.__dict__.update({'distance': D})
            snr.set_distance(D)
            area = 4.*pi*(snr.distance*ct._kpc_over_cm_)**2.  # [cm^2]

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
            data = {'deltaE_over_E': 1.e-3 * 2.17,
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

            # end of routine for fixed D

        # appending finished D rows
        sn_Gr.append(row_a)
        snu_Gr.append(row_b)
        if flg_t:
            s0_Gr.append(row_c)
        elif flg_s:
            tsig_Gr.append(row_c)

        # end of routine for fixed l

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
    long_arr = np.linspace(0., 360., Nsteps+1)
    lat_arr = np.linspace(-90., 90., Nsteps+2)

    # # Saving arrays
    # if os.access(folder+"long_arr.txt", os.R_OK):
    #     pass
    # else:
    #     np.savetxt(folder+"long_arr.txt", long_arr)
    # if os.access(folder+"lat_arr.txt", os.R_OK):
    #     pass
    # else:
    #     np.savetxt(folder+"lat_arr.txt", lat_arr)
    # Saving arrays
    np.savetxt(folder+"run_%d_long_arr_y.txt" % run_id, long_arr)
    np.savetxt(folder+"run_%d_lat_arr_x.txt" % run_id, lat_arr)

    # New L_peak (corrected from Bietenholz frequency to to 1 GHz):
    new_Lpk = args.Lpk*from_Bieten_to_pivot

    # Updating SNR parameters:
    # snr.__dict__.update({'distance': args.distance,
    #                      'no_dist': False})
    snr.set_distance(args.distance)

    if flg_t:
        # snr.__dict__.update({'age': args.t_signal})
        snr.set_age(args.t_signal)
    elif flg_s:
        # snr.__dict__.update({'snu_at_1GHz': args.S0})
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
    data = {'deltaE_over_E': 1.e-3 * 2.17,
            'f_Delta': 0.721,
            'exper': 'SKA',
            'total_observing_time': 100.,
            'verbose': 0,
            'average': True}

    # Result grids:
    sn_Gr = []
    snu_Gr = []

    for l in tqdm(long_arr):

        # snr.__dict__.update({'l': l})
        snr.set_coord(l=l, b=None, sign=None)

        row_a, row_b = [], []

        for b in lat_arr:

            # snr.__dict__.update({'b': b})
            snr.set_coord(b=b, l=None, sign=None)

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

            # end of routine for fixed b

        # appending finished b rows
        sn_Gr.append(row_a)
        snu_Gr.append(row_b)

        # end of routine for fixed l

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
elif args.slice == "D-t":
    # combine r-tsig and r-tex
    # Defining the arrays
    dist_arr = np.logspace(-1, 0.5, Nsteps+1)
    t_arr = np.logspace(log10(10.*(args.tpk/365.)), 5., Nsteps+2)

    np.savetxt(folder+"run_%d_dist_arr_y.txt" % run_id, dist_arr)
    np.savetxt(folder+"run_%d_t_arr_x.txt" % run_id, t_arr)

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
    for D in tqdm(dist_arr):

        # set distance to SNR, which will be used to construct input_param through model.snr_routine()
        snr.set_distance(D)
        area = 4.*pi*(snr.distance*ct._kpc_over_cm_)**2.  # [cm^2]

        row_a, row_b, row_c = [], [], []

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
            data = {'deltaE_over_E': 1.e-3 * 2.17,
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

    # end of D-t routine


# CASE 7:
elif args.slice == "l-t":
    # t combines r-tsig and r-tex
    # Defining the arrays
    long_arr = np.linspace(0., 360., Nsteps+1)
    t_arr = np.logspace(log10(10.*(args.tpk/365.)), 5., Nsteps+2)

    np.savetxt(folder+"run_%d_long_arr_y.txt" % run_id, long_arr)
    np.savetxt(folder+"run_%d_t_arr_x.txt" % run_id, t_arr)

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
    for longitude in tqdm(long_arr):

        # set distance to SNR, which will be used to construct input_param through model.snr_routine()
        row_a, row_b, row_c = [], [], []
        snr.set_coord(l=longitude, b=args.lat, sign=None)

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
            data = {'deltaE_over_E': 1.e-3 * 2.17,
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

    # end of l-t routine


# CASE 8:
elif args.slice == "b-t":
    # t combines r-tsig and r-tex
    # Defining the arrays
    lat_arr = np.linspace(-90., 90., Nsteps+1)
    t_arr = np.logspace(log10(10.*(args.tpk/365.)), 5., Nsteps+2)

    np.savetxt(folder+"run_%d_lat_arr_y.txt" % run_id, lat_arr)
    np.savetxt(folder+"run_%d_t_arr_x.txt" % run_id, t_arr)

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
    for latitude in tqdm(lat_arr):

        # set distance to SNR, which will be used to construct input_param through model.snr_routine()
        row_a, row_b, row_c = [], [], []
        snr.set_coord(l=args.longitude, b=latitude, sign=None)

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
            data = {'deltaE_over_E': 1.e-3 * 2.17,
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

    # end of b-t routine

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
# python ./run_custom.py --run 0 --nuB 8 --Nsteps 30 Lpk-tpk --dist 0.5 --tt_ratio 30 --t0 1e4 --t_extra 4e4 --long_lat 178 0

# CASE2:
# python ./run_custom.py --run 0 --nuB 8 --Nsteps 30 r-tsig --Lpk 3.16e28 --tpk 50.1 --dist 0.5 --long_lat 178 0

# CASE3:
# python ./run_custom.py --run 0 --nuB 8 --Nsteps 30 r-tex --Lpk 3.16e28 --tpk 50.1 --dist 0.5 --long_lat 178 0

# CASE4:
# python ./run_custom.py --run 0 --nuB 8 --Nsteps 100 l-D --Lpk 3.16e28 --tpk 50.1 --tt_ratio 30 --t0 1e4 --t_extra 4e4 --lat 0

# CASE5:
# python ./run_custom.py --run 0 --nuB 8 --Nsteps 30 l-b --Lpk 3.16e28 --tpk 50.1 --dist 0.5 --tt_ratio 30 --t0 1e4 --t_extra 4e4

# CASE6:
# python ./run_custom.py --run 0 --nuB 8 --Nsteps 30 D-t --Lpk 3.16e28 --tpk 50.1 --tt_ratio 30 -lb 178 0

# CASE7:
# python ./run_custom.py --run 0 --nuB 8 --Nsteps 100 l-t --Lpk 3.16e28 --tpk 50.1 --tt_ratio 30 -D 0.5 -b 0

# CASE8:
# python ./run_custom.py --run 0 --nuB 8 --Nsteps 100 b-t --Lpk 3.16e28 --tpk 50.1 --tt_ratio 30 -D 0.5 -l 178
