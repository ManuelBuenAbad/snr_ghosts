from __future__ import division
import numpy as np

from numpy import pi, sqrt, exp, power, log, log10
from scipy.interpolate import interp1d
from scipy.special import erf, lambertw
from scipy.optimize import brentq

#########################################

def interp_fn(array):
    """
    An interpolator for log-arrays spanning many orders of magnitude.

    Parameters
    ----------
    array : An array of shape (N, 2) from which to interpolate.
    """

    array[array < 1.e-300] = 1.e-300  # regularizing small numbers

    def fn(x): return 10**interp1d(log10(array[:, 0]),
                                   log10(array[:, 1]), fill_value='extrapolate')(log10(x))

    return fn



def zeros(fn, arr):
    """
    Find where a function crosses 0. Returns the zeroes of the function.

    Parameters
    ----------
    fn : function
    arr : array of arguments for function
    """

    fn_arr = fn(arr)

    where_arr = np.where(np.logical_or((fn_arr[:-1] < 0.) * (fn_arr[1:] > 0.),
                                       (fn_arr[:-1] > 0.) * (fn_arr[1:] < 0.))
                         )[0]

    cross_arr = []
    if len(where_arr) > 0:
        for i in range(len(where_arr)):
            cross_arr.append(
                brentq(fn, arr[where_arr[i]],
                       arr[where_arr[i] + 1])
            )

    cross_arr = np.array(cross_arr)

    return cross_arr