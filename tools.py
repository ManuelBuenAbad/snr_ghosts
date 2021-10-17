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

    # looking where the function changes sign...
    sign_change_arr = np.where(np.logical_or((fn_arr[:-1] < 0.) * (fn_arr[1:] > 0.),
                                             (fn_arr[:-1] > 0.) * (fn_arr[1:] < 0.))
                               )[0]

    # or, just in case, where it is exactly 0!
    exact_zeros_arr = np.where(fn_arr == 0.)[0]

    # defining the array of 0-crossings:
    cross_arr = []

    # first, interpolating between the sign changes
    if len(sign_change_arr) > 0:
        for i in range(len(sign_change_arr)):
            cross_arr.append(
                brentq(fn, arr[sign_change_arr[i]],
                       arr[sign_change_arr[i] + 1])
            )

    # and then adding those places where it is exactly 0
    if len(exact_zeros_arr) > 0:
        for i in range(len(exact_zeros_arr)):
            cross_arr.append(arr[exact_zeros_arr[i]])

    # sorting the crossings in increasing order:
    cross_arr = np.sort(np.array(cross_arr))

    return cross_arr


def treat_as_arr(arg):
    """
    A routine to cleverly return scalars as (temporary and fake) arrays. True arrays are returned unharmed. Thanks to Chen!
    """

    arr = np.asarray(arg)
    is_scalar = False

    # making sure scalars are treated properly
    if arr.ndim == 0:  # it is really a scalar!
        arr = arr[None]  # turning scalar into temporary fake array
        is_scalar = True  # keeping track of its scalar nature

    return arr, is_scalar


def load_dct(dct, key):
    """Used to load and determine if dict has a key

    :param dct: the dictionary to be interogated 
    :param key: the key to be tried

    """

    try:
        res = dct[key]
        is_success = True
    except KeyError:
        res = None
        is_success = False
    return res, is_success
