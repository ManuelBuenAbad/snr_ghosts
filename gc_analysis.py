from __future__ import division
import numpy as np
from numpy import pi, sqrt, log, log10, power, exp
from scipy.interpolate import interp1d, interp2d
from scipy.interpolate import RectBivariateSpline as RBS
from tqdm import tqdm
import os
import argparse

import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import rc

rcParams['figure.figsize'] = (10, 10)
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['New Times Roman']
rc('text', usetex=True)

# Current directory
current_dir = os.getcwd()

import constants as ct
import particle as pt
import astro as ap
import echo as ec
import routines as rt
import data as dt
import model as md
import green as gr
import mcmc as mcmc

# -------------------------------------------------

###############
# DIRECTORIES #
###############

green_path = os.path.dirname(os.path.abspath(__file__))+"/output/green_snr/"

# -------------------------------------------------

#############
# ARGUMENTS #
#############

# Defining arguments
parser = argparse.ArgumentParser(description="Computes the +-1sigma band reach for a specific SNR from the Green's Catalog")

# Arguments with values:
parser.add_argument('-i', '--id', '--name', default=None,
                    type=str, help="The name of the SNR (default: None)")
parser.add_argument('-r', '--tt_ratio', '--ratio', default=None,
                    type=float, help="The ratio of t_trans/t_pk (default: None)")
parser.add_argument('-x', '--t_extra', '--extra', default=None,
                    type=float, help="The extra age of the SNR, after the adiabatic phase [years] (default: None)")
parser.add_argument('-n', '--nuB', '--nu_Bietenholz', default=None,
                    type=float, help="The Bietenholz frequency [GHz] (default: None)")
parser.add_argument('-Z', '--sn_th', '--signal_to_noise_ratio', default=None,
                    type=float, help="The signal-to-noise ratio threshold (default: None)")

parser.add_argument('-N', '--Nsteps', default=5000,
                    type=int, help="The number of steps in the parameter space arrays (default: None)")
parser.add_argument('-w', '--walkers', default=50,
                    type=int, help="The number of steps in the parameter space arrays (default: None)")
parser.add_argument('-b', '--burn', default=1000,
                    type=int, help="The number of steps in the parameter space arrays (default: None)")

# True/False flags
parser.add_argument('-a', '--known_age', action='store_true', help="Whether the age of the SNR is known (default: False)")
parser.add_argument('-v', '--verbose', action='store_true', help="Verbosity (default: False)")
    
# Parsing arguments:
args = parser.parse_args()
    
# Defining appropriate variables
name = args.id
tt_ratio = args.tt_ratio
t_extra = args.t_extra
nuB = args.nuB
sn_th = args.sn_th

nsteps = args.Nsteps
nwalkers = args.walkers
nburn = args.burn

flg_r = bool(args.tt_ratio)
known_age = args.known_age
verbose = args.verbose

# Making sure the parametere have the right values
if flg_r and known_age:
    raise Exception("Cannot pass both --tt_ratio and --known_age. Pick one.")

if t_extra == None or nuB == None or nsteps == None:
    raise Exception("Pass a value for --Nsteps, for --t_extra, and for --nuB.")

if (not known_age) and (not flg_r):
    raise Exception("Pass a value for --tt_ratio or simply pass --known_age.")
else:
    pass

# File tail
if flg_r:
    ident = "r-{}_tex-{}_nuB-{}".format(int(tt_ratio), int(t_extra), int(nuB))
    tail = "_Lpk-tpk_"+ident+".txt"
elif known_age:
    ident = "wage_tex-{}_nuB-{}".format(int(t_extra), int(nuB))
    tail = "_Lpk-tpk_"+ident+".txt"

if verbose:
    print("\ntt_ratio={}, t_extra={}, nuB={}, known_age={}".format(tt_ratio, t_extra, nuB, known_age))
    print("Identifier: "+ident)

# SNR folder and files:
folder = green_path+name+"/"
file = name+tail



# -------------------------------------------------

###################
# SNR INFORMATION #
###################

# The SNR we are interested in:
snr = gr.snrs_dct[name]

# Loading the S/N, Snu_echo, and t_age grids:
sn_Gr, echo_Gr, time_Gr = gr.load_green_results(name, r=tt_ratio, tex=t_extra, nuB=nuB)

# Quantities for SNR reach
ga_fn, ga_Gr, normal_Lpk_arr, normal_tpk_arr, nonsense_params = gr.snr_reach(name, r=tt_ratio, nuB=nuB, tex=t_extra, sn_ratio_threshold=sn_th, full_output=True)

if verbose:
    print "\nS/N: min=%.1e, max=%.1e" % (sn_Gr.min(), sn_Gr.max())
    print "Snu_echo: min=%.1e, max=%.1e" % (echo_Gr.min(), echo_Gr.max())
    print "time: min=%.1e, max=%.1e" % (time_Gr.min(), time_Gr.max())
    print "ga: min=%.1e, max=%.1e" % (ga_Gr.min(), ga_Gr.max())
    print "param: allowed=%.1e, forbidden:%.1e" % (nonsense_params.min(), nonsense_params.max())

# -------------------------------------------------

#################
# MCMC ANALYSIS #
#################

# Number of sigmas in Lpk and tpk to be scanned
nL_limits = (-3., 3.)
nt_limits = (-3., 3.)

# Performing the MCMC scane!
if verbose:
    print("\nReady to run MCMC!")

sample, ga_reach = mcmc.snr_emcee_routine(ga_fn, nL_limits=nL_limits, nt_limits=nt_limits,
                                          nwalkers=nwalkers, nburn=nburn, nsteps=nsteps,
                                          xarr=normal_Lpk_arr,
                                          yarr=normal_tpk_arr,
                                          nonsense_grid=nonsense_params)

if verbose:
    print("\nga distribution: min={}, max={}".format(ga_reach.min(), ga_reach.max()))

# Refining ga_reach distribution: forbidding nonsensical values
refined_ga = np.nan_to_num(ga_reach)
refined_ga = refined_ga[refined_ga < 1.e-2]

# Defining the list of quantiles to look for...:
quant_list = [0.023, 0.159, 0.5, 0.841, 0.977]
# ... and their corresponding sigma fluctuations keywords:
sigs_list = ['-2s', '-1s', '0s', '+1s', '+2s']

# A function computing the quantiles on the ga distribution:
def ga_quantile(quant): return np.quantile(refined_ga, quant)

# The list of ga quantiles:
ga_list = map(ga_quantile, quant_list)

# Zipping sigma keywords and ga quantiles into a dictionary:
ga_band = dict(zip(sigs_list, ga_list))

if verbose:
    print("\nga_bands: {}".format({sig:ga_band[sig] for sig in sigs_list}))


# -------------------------------------------------

#############
# RESCALING #
#############

def nu_dependence_fn(nu, sn_remnant):
    """
    A function approximating the frequency nu [GHz] dependence of the axion-photon coupling ga [GeV^-1] reach, for a given SNR. It is only an approximation, since in principle there might be some un-factorizable frequency dependence coming from the fact that the observation solid angle might depend on it and on the lightcurve parameters (for example, the age of the SNR).
    
    Parameters
    ----------
    nu : frequency [GHz]
    sn_remnan : SuperNovaRemnant object
    """
    
    # Compute a mock approximation to the solid angle of the signal:
    # First, the SNR solid angle size
    Omega_size = sn_remnant.get_size()
    # The dispersion solid angle, exact for x_wavefront << distance: this is the smallest possible dispersion
    Omega_dispersion = ct.angle_to_solid_angle(2.*(1.e-3 / 2.17))
    # The approximate signal solid angle. This is the smallest possible signal: Omega_dispersion and Omega_aberration (and thus Omega_signal) could be larger
    mock_Omega_signal = max(Omega_size, Omega_dispersion)
    
    # Axion mass [eV]
    ma = pt.ma_from_nu(nu)
    # SKA specs
    area, window, Tr, Omega_res = rt.SKA_rescaled_specs(ma, data={'exper':'SKA'})
    # Mock observation solid angle
    mock_Omega_obs = np.maximum.reduce([mock_Omega_signal*np.ones_like(Omega_res), Omega_res])
    
    # SNR coordinates
    l, b = sn_remnant.get_coord()
    # Echo coordinates
    l_echo, b_echo = l + 180., -b
    
    # Background brightness temperature at 408 MHz
    Tbg_408 = ap.bg_408_temp(l_echo, b_echo, size=mock_Omega_obs, average=True)
    # Noise temperature at frequency nu
    Tn = ap.T_noise(nu, Tbg_at_408=Tbg_408, beta=-2.55, Tr=Tr)
    
    # SNR spectral index
    alpha = sn_remnant.alpha
    
    # The nu functional dependence of the ga reach
    f = nu**(-alpha-0.5) * area / Tn / sqrt(mock_Omega_obs/Omega_res)
    
    return f



# Defining a fine array of frequencies
Nma = 201
# For SKA low...
nulow = np.logspace(log10(ct._nu_min_ska_low_), log10(ct._nu_max_ska_low_), Nma//2)[1:]
# ... and SKA mid...
numid = np.logspace(log10(ct._nu_min_ska_mid_), log10(ct._nu_max_ska_mid_), Nma - Nma//2)[1:]
# ... concatenating...
nu_all = np.concatenate((nulow, numid))
# ... and converting into axion masses
ma_all = pt.ma_from_nu(nu_all)

# The reach nu-dependent rescale factor. Note that nu_pivot = 1 GHz
rescale = sqrt(nu_dependence_fn(1., snr)/nu_dependence_fn(nu_all, snr))

# The ga(ma) band, for median, +-1 sigmas, and +- 2 sigmas.
ma_ga_band = np.vstack((ma_all,
                        ga_band['0s']*rescale,
                        ga_band['+1s']*rescale,
                        ga_band['-1s']*rescale,
                        ga_band['+2s']*rescale,
                        ga_band['-2s']*rescale)).T

# -------------------------------------------------

############
# PLOTTING #
############

# Some sizes:
title_sz = 20.
label_sz = 15.
legend_sz = 15.

# The CAST bound
ga_cast = 6.e-11 # [GeV]

# Lpk-tpk distribution:
fig, ax = plt.subplots()
ax.hist(sample[:,0], bins=100, histtype="stepfilled", alpha=0.3, density=True, label=r"$L_{\rm peak}$")
ax.hist(sample[:,1], bins=100, histtype="stepfilled", alpha=0.3, density=True, label=r"$t_{\rm peak}$")
ax.set_xlim(-3., 3.);
ax.set_xlabel(r'$Z$ [normalized variable]', fontsize=label_sz);
ax.tick_params("both", which="both", labelsize=label_sz, direction="in", length=10.)
ax.legend(fontsize=legend_sz);

fig.suptitle(r'Distribution of normalized $L_{\rm peak}$ and $t_{\rm peak}$ for '+name, fontsize=title_sz);
fig.tight_layout()
plt.savefig(green_path+"{0}/Lpk-tpk_dist_{0}_{1}.pdf".format(name, ident))

# ga distribution:
fig, ax = plt.subplots()
ax.hist(log10(refined_ga), bins=100, histtype="stepfilled", alpha=0.3, density=True, color='C2', zorder=-1)
ax.set_xlim(-12, -6);
ax.set_xlabel(r'$\log_{10}\big( g_{a\gamma\gamma} \cdot \mathrm{GeV} \big)$', fontsize=label_sz);

ax.axvline(log10(ga_band['0s']), ls="-", color="k", zorder=-1);
ax.axvline(log10(ga_band['-1s']), ls="--", color="b", zorder=-1);
ax.axvline(log10(ga_band['+1s']), ls="--", color="b", zorder=-1);
ax.axvline(log10(ga_band['-2s']), ls=":", color="r", zorder=-1);
ax.axvline(log10(ga_band['+2s']), ls=":", color="r", zorder=-1);
ax.tick_params("both", which="both", labelsize=label_sz, direction="in", length=10.)

fig.suptitle(r'Distribution of $g_{a\gamma\gamma}$ reach for '+name, fontsize=title_sz);
fig.tight_layout()
plt.savefig(green_path+"{0}/ga_dist_{0}_{1}.pdf".format(name, ident))

# ga(ma) bands:
fig, ax = plt.subplots()
ax.fill_between(ma_ga_band[:,0], ma_ga_band[:,-1], ma_ga_band[:,-2], color='C0', alpha=0.2, zorder=-1)
ax.fill_between(ma_ga_band[:,0], ma_ga_band[:,2], ma_ga_band[:,3], color='C0', alpha=0.2, zorder=-1)
ax.loglog(ma_ga_band[:,0], ma_ga_band[:,1], color='C0', lw=2., zorder=-1)
ax.axhline(ga_cast, color='C2', zorder=-1)
ax.set_xlim(1.e-7, 1.e-3);
ax.set_ylim(1.e-12, 1.e-4);
ax.set_xlabel(r'$m_a \quad [\mathrm{eV}^{-1}]$', fontsize=label_sz);
ax.set_ylabel(r'$g_{a\gamma\gamma} \quad [\mathrm{GeV}^{-1}]$', fontsize=label_sz);
ax.tick_params("both", which="both", labelsize=label_sz, direction="in", length=10.)

fig.suptitle(r'Reach $g_{a\gamma\gamma}$ bands for '+name, fontsize=title_sz);
fig.tight_layout()
plt.savefig(green_path+"{0}/{0}_ga_bands_{1}.pdf".format(name, ident))

# Saving ga(ma) bands
np.savetxt(green_path+"{0}/ma_ga_bands_{0}_{1}.txt".format(name, ident), ma_ga_band, delimiter=",")
