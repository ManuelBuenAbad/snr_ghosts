from __future__ import division
import numpy as np
from numpy import pi, sqrt, log, log10, power, exp
from scipy.interpolate import interp2d
import os
import emcee

# current directory
current_dir = os.getcwd()



# -------------------------------------------------

#################
# MCMC ANALYSIS #
#################

# a large number, but not too large for it to cause overflows when manipulated
very_large = 1.e100



def normal_prior(x, limits):
    """
    Prior on normalized variable x, with limits.
    """
    
    return -very_large*np.logical_not(np.heaviside(x-limits[0], 1.)*np.heaviside(limits[1]-x, 1.))



def forbidden_prior(x, y, xarr, yarr, nonsense_grid):
    """
    Prior for forbidden/nonsensical parameters, obtained from a grid whose values are 0 (allowed) or 1 (forbidden).
    """
    
    def nonsense_fn(x, y): return interp2d(xarr, yarr, nonsense_grid)(x, y)
    
    return -very_large*nonsense_fn(x, y)



def log_likelihood(nL, nt):
    """
    Log-likelihood for a 2D normal distribution.
    """
    
    return -0.5*(nL**2. + nt**2.)



def log_posterior(nL, nt, nL_limits, nt_limits, blob_fn, forbidden=False, **nonsense_kwargs):
    """
    Computes the posterior from the priors and the log-likelihood.
    """
    
    nL_prior = normal_prior(nL, nL_limits)
    nt_prior = normal_prior(nt, nt_limits)
    
    if forbidden:
        no_prior = forbidden_prior(nL, nt, **nonsense_kwargs)
    else:
        no_prior = 0.
    
    loglkl = log_likelihood(nL, nt)    
    
    posterior = nL_prior + nt_prior + no_prior + loglkl
    
#     print nL_prior, nt_prior, no_prior, loglkl, posterior
    
    blob = blob_fn(nL, nt)[0]
    
    return posterior, blob



def snr_emcee_routine(ga_fn,
                      nL_limits=(None, None),
                      nt_limits=(None, None),
                      forbidden=False,
                      nwalkers=50, nburn=1000, nsteps=5000,
                      **nonsense_kwargs):
    """
    The emcee routine to analyze the Green's Catalog SNR results.
    """
    
    # limits of the priors
    nL_center, nL_size = np.mean(nL_limits), nL_limits[1] - nL_limits[0]
    nt_center, nt_size = np.mean(nt_limits), nt_limits[1] - nt_limits[0]
    
    # two dimensions
    ndim = 2
    
    # starting guesses
    p0 = np.random.rand(nwalkers, ndim)-0.5
    # for normalized Lpk:
    p0[:,0] *= nL_size # with correct size...
    p0[:,0] += nL_center # ... and center
    # for normalized tpk:
    p0[:,1] *= nt_size # with correct size...
    p0[:,1] += nt_center # ... and center
    
    # preparing the data type of the blobs
    dtype = [("ga_reach", float)]
    
    # defining the posterior probability
    def log_post(x):
        return log_posterior(x[0], x[1],
                             nL_limits, nt_limits,
                             ga_fn,
                             forbidden=forbidden,
                             **nonsense_kwargs)
    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_post,
                                    blobs_dtype=dtype)
    sampler.run_mcmc(p0, nsteps)
    
    sample = sampler.chain # shape = (nwalkers, nsteps, ndim)
    sample = sampler.chain[:, nburn:, :] # discard burn-in points
    sample = sample.reshape(-1, ndim)
    
    blobs = sampler.get_blobs()
    ga_reach = blobs["ga_reach"]
    
    ga_reach = ga_reach[nburn:,:].ravel()
    
    return sample, ga_reach