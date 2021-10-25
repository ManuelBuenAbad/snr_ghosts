from __future__ import division
import model as md
import data as dt
import routines as rt
import echo as ec
import astro as ap
import particle as pt
import constants as ct
import tools as tl
import numpy as np
from numpy import pi, sqrt, log, log10, power, exp
from scipy.interpolate import interp1d
from tqdm import tqdm
import os
import sys
import argparse

# current directory
current_dir = os.getcwd()

# class DataPoint(object):
#     """A single data point to construct the lightcurve and snr object

#     """
#     def __init__(self):
#         super(DataPoint, self).__init__()
#         self.args = args


class ParamAxis(object):
    """Parameter Axis

    """

    def __init__(self, name, x0, x1=None, steps=None, is_log=None):
        if x1 is not None:
            self.is_scalar = False
            if is_log:
                self.points = np.logspace(np.log10(x0), np.log10(x1), steps)
            else:
                self.points = np.linspace(x0, x1, steps)
        else:
            self.points = np.array([x0])
            self.is_scalar = True
        self.name = name

    def __call__(self):
        return self.points

    def __str__(self):
        return "<axis instance along %s>" % self.name

    def __repr__(self):
        return "<axis instance along %s>" % self.name

    def get_name(self):
        return self.name

    def is_scan(self):
        """Determine if the axes should be scanned over

        """
        if self.is_scalar is True:
            res = False  # can't scan it
        elif self.is_scalar is False:
            res = True  # should scan it
        return res

    def get_point(self):
        """Used to return the single point when the axis is a scalar

        """
        if self.is_scan():
            raise Exception(
                'The axis you try to get point is not a scalar axis.')
        return self.points[0]

    def get_min(self):
        """Used to return the minimal point

        """
        return min(self.points)

    def get_max(self):
        """Used to return the maximal point

        """
        return max(self.points)

    def get_length(self):
        """Get the length of the axis

        """

        return len(self.points)


class ParamSpaceSlice2D(object):
    """The parameter space to be scanned over

    """

    def __init__(self, axes):
        """
        :param axes: all the axes of the parameter space
        """
        # sanity check: only two axes should be axis.is_scan == True
        axes_scan = []

        for axis in axes:
            if axis.is_scan():
                axes_scan.append(axis)
        if len(axes_scan) != 2:
            raise Exception(
                'Only two axes should be non-scalar. You have %d axes to be non-scalar.' % len(axes_scan))

        self.axes = axes
        self.axes_scan = axes_scan

        # the following should be initizalied every time a new slice
        # is constructed therefore it is not safe to re-use
        # ParamSpaceSlice2D instances for different slices

        # some go into the lightcurve dict
        self.lightcurve_params = {}
        self.snu_echo_kwargs = {}
        # some go into the snr object
        self.snr = dt.SuperNovaRemnant()
        self.snr.set_name('custom')

    def get_dimension(self):
        return len(self.axes)

    def save(self):
        """Save the param space 2D slice

        """
        pass

    def scan(self):
        """Perform scan over the 2D slice

        """

        # ref
        self.ga_ref = 1.e-10  # [GeV^-1]
        # Maximum number of steps:
        max_steps = 1000001

        # then scan over the two axes
        axis0 = self.axes_scan[0]
        axis1 = self.axes_scan[1]
        mesh0, mesh1 = np.meshgrid(axis0(), axis1(), indexing='ij')
        flat0, flat1 = (mesh0.reshape(-1), mesh1.reshape(-1))
        sig_noi_flat = []
        signal_Snu_flat = []
        blob_flat = []

        # verbosity
        try:
            verbose = self.verbose
        except AttributeError:
            verbose = 0
        if verbose > 0:
            tqdm_disable = False
        else:
            tqdm_disable = True

        for i, _ in enumerate(tqdm(flat0, disable=tqdm_disable)):
            value0 = flat0[i]
            value1 = flat1[i]

            point_dct = {}

            name = axis0.get_name()
            point_dct[name] = value0

            name = axis1.get_name()
            point_dct[name] = value1

            # fill in the fixed values
            for axis in self.axes:
                if axis.is_scan():
                    continue
                name = axis.get_name()
                value = axis.get_point()
                # self.construct(name, value)
                point_dct[name] = value

            self.construct(point_dct)

            #
            # start the real computation
            #

            # Snu kwargs
            # print(self.snr.get_age())
            # print(self.lightcurve_params['t_peak'])
            age_steps = abs(
                int(1000*(log10(self.snr.get_age()) - log10(self.lightcurve_params['t_peak']/365.)) + 1))
            # already has t extra in it
            self.snu_echo_kwargs.update({'tmin_default': None,
                                         'Nt': min(age_steps, max_steps),
                                         'xmin': ct._au_over_kpc_,
                                         'xmax_default': 100.,
                                         'use_quad': False,
                                         'lin_space': False,
                                         'Nint': min(age_steps, max_steps)
                                         # 't_extra_old': None
                                         })

            # data:
            z_arr = []
            signal_Snu_arr = []

            for SKA_mode in ['interferometry', 'single dish']:
                # for SKA_mode in ['single dish']:
                # for SKA_mode in ['interferometry']:
                data = {'deltaE_over_E': ct._deltaE_over_E_,
                        'f_Delta': ct._f_Delta_,
                        'exper': 'SKA',
                        'total_observing_time': 100.,
                        'verbose': 0,
                        'average': True,
                        'correlation_mode': SKA_mode}

                # computing routine
                z, new_output = md.snr_routine(
                    # pt.ma_from_nu(1.)
                    1.e-6,  # [eV]
                    self.ga_ref,
                    self.snr,
                    lightcurve_params=self.lightcurve_params,
                    snu_echo_kwargs=self.snu_echo_kwargs,
                    data=data,
                    output_all=True)

                signal_Snu = new_output['signal_Snu']
                if args.verbose:
                    print("S/N=%.1e, S=%.1e, N=%.1e (%s)" %
                          (z, new_output['signal_power'], new_output['noise_power'], SKA_mode))
                del new_output

                z_arr.append(np.squeeze(z))
                signal_Snu_arr.append(np.squeeze(signal_Snu))

            # building rows
            sig_noi_flat.append(max(z_arr))  # signal-to-noise ratio
            signal_Snu_flat.append(max(signal_Snu_arr))  # signal S_nu

            try:
                Lpk = ap.L_source(t=self.lightcurve_params['t_peak']/365.,
                                  L_today=self.lightcurve_params['L_today'],
                                  t_age=self.lightcurve_params['t_age'],
                                  t_peak=self.lightcurve_params['t_peak'],
                                  t_trans=self.lightcurve_params['t_trans'],
                                  gamma=self.snr.get_gamma()
                                  )

                blob_flat.append(Lpk / self.from_Bieten_to_pivot)
            except KeyError:
                pass

        # save it
        self.x_mesh = mesh0
        self.y_mesh = mesh1
        self.sig_noi_mesh = np.array(sig_noi_flat).reshape(mesh0.shape)
        self.signal_Snu_mesh = np.array(signal_Snu_flat).reshape(mesh0.shape)
        try:
            self.blob_mesh = np.array(blob_flat).reshape(mesh0.shape)
        except ValueError:
            self.blob_mesh = np.array([-1])

    def construct(self, point_dct):
        """Construct lightcurve dict and assemble the snr object. Some quantities depend on the rest. We treat the two groups separately. 

        :param point_dct: the dictionary that contains names and values of a given single data point

        """
        try:
            args.verbose
        except ValueError:
            args.verbose = 0
        if args.verbose > 1:
            string = ""
            for name, val in point_dct.items():
                string += "%s: %s" % (name, val)
            print(string)

        #
        # quantities that do not depend on others
        # we only do simple ambiguity check here, as check_data() and
        # check_input() does a much more thorough check
        #
        # light curve
        self.lightcurve_params['t_peak'] = point_dct['t peak']
        self.nuB = point_dct['nu Bietenholz']

        # SNR
        # t age and t extra separately
        t_total, flag1 = tl.load_dct(point_dct, 't age total')
        t_age, flag2 = tl.load_dct(point_dct, 't age')
        t_extra, flag3 = tl.load_dct(point_dct, 't extra')

        if flag1 and flag2 and flag3:
            raise Exception("Ambiguity encounterred. ")

        if flag1:
            t_extra = t_total - ct._time_of_phase_two_
            if t_extra > 0:
                self.snr.set_age(ct._time_of_phase_two_)
                self.lightcurve_params['t_age'] = ct._time_of_phase_two_
                self.snu_echo_kwargs['t_extra_old'] = t_extra
            else:
                self.snr.set_age(t_total)
                self.lightcurve_params['t_age'] = t_total
                self.snu_echo_kwargs['t_extra_old'] = 0.
        else:
            if flag2:
                self.snr.set_age(t_age)
                self.lightcurve_params['t_age'] = t_age

            if flag3:
                self.snu_echo_kwargs['t_extra_old'] = t_extra
            else:
                self.snu_echo_kwargs['t_extra_old'] = 0.

        self.snr.set_coord(l=point_dct['galaxy l'], b=None, sign=None)
        self.snr.set_coord(l=None, b=point_dct['galaxy b'], sign=None)
        self.snr.set_distance(point_dct['distance'])
        self.snr.set_sr(point_dct['square radians'])
        self.snr.set_spectral(point_dct['spectral index'], is_certain='yes')
        self.from_Bieten_to_pivot = (
            1./args.nuB)**(-1.*point_dct['spectral index'])

        #
        # things that depends on others
        #
        # light curve
        value, flag = tl.load_dct(point_dct, 'L peak Bietenholz')
        if flag:
            self.lightcurve_params['L_peak'] = value * \
                self.from_Bieten_to_pivot

        value, flag = tl.load_dct(point_dct, 't trans over t peak')
        if flag:
            self.lightcurve_params['t_trans'] = value * \
                self.lightcurve_params['t_peak']/365.

        # SNR
        S0, flg1 = tl.load_dct(point_dct, "S0")
        L0, flg2 = tl.load_dct(point_dct, "L0")
        if flg1 and flg2:
            raise Exception("Ambiguity found: you specified S0 and L0.")

        if flg1:
            self.snr.set_flux_density(
                S0, is_flux_certain='let us assume it is certain')
            L0 = self.snr.get_luminosity()  # [cgs] depends on distance
            self.lightcurve_params.update({'L_today': L0})
        if flg2:
            self.lightcurve_params.update({'L_today': L0})


class Run(object):
    """Class for a complete run. Mostly deals with I/O stuff.

    """

    def __init__(self):
        # super(Run, self).__init__()
        self.run_id = None
        self.mode = None

    def __call__(self, mode, run_id=None, verbose=1):
        # run id:
        if run_id is not None:
            self.run_id = run_id

        if self.run_id is None:
            import uuid
            self.run_id = str(uuid.uuid4())

        print('%s:%s\n' % (mode, self.run_id))
        # specify mode, i.e. name of slice say 'Lpk-tpk':
        self.mode = mode
        # takes care of paths
        self.init_dir()
        # get a slice
        self.param_space = gen_slice(mode)
        # verbosity
        self.param_space.verbose = verbose
        # scan the slice
        self.param_space.scan()
        # save it
        self.export()
        # log it
        self.log()

        return {mode: self.run_id}

    def init_dir(self):
        # directory initialize
        try:
            os.makedirs(os.path.dirname(os.path.abspath(__file__)) +
                        "/output/custom_snr/")
        except Exception:
            pass

        # Creating the appropriate slice directory:
        self.folder = os.path.dirname(os.path.abspath(__file__)) + \
            "/output/custom_snr/"+self.mode+"/"
        try:
            os.makedirs(self.folder)
        except Exception:
            pass

        # # Defining the filename
        # self.filename = "custom_"

        # if args.verbose:
        #     print(self.filename)

    def export(self):
        param_space = self.param_space

        # save grid
        path = os.path.join(self.folder, "run_%s_x.txt" % (self.run_id))
        np.savetxt(path, param_space.x_mesh, delimiter=",")
        path = os.path.join(self.folder, "run_%s_y.txt" % (self.run_id))
        np.savetxt(path, param_space.y_mesh, delimiter=",")

        # save result
        # print('shape:%s' % (param_space.sig_noi_mesh.shape,))
        path = os.path.join(self.folder, "run_%s_sn.txt" % self.run_id)
        np.savetxt(path, param_space.sig_noi_mesh,
                   delimiter=",")  # signal/noise
        path = os.path.join(self.folder, "run_%s_echo.txt" % self.run_id)
        np.savetxt(path, param_space.signal_Snu_mesh,
                   delimiter=",")  # Snu of echo signal
        path = os.path.join(self.folder, "run_%s_blob.txt" % self.run_id)
        np.savetxt(path, param_space.blob_mesh,
                   delimiter=",")  # Snu of echo signal

    def log(self):
        """Log the info of the run

        """

        path = os.path.join(self.folder, "run_%s_log.txt" % self.run_id)
        with open(path, 'w') as f:
            f.write('#\n#----------------------------------------BASIC LOG\n#\n')
            f.write('#\n#-------Meta info\n#\n')
            f.write('run_id: %s\n' % self.run_id)
            f.write('running_mode: %s\n' % self.mode)
            f.write('ga_ref: %e\n' % self.param_space.ga_ref)
            # f.write('current_dir: %s\n' % current_dir)

            f.write('#\n#-------SNe Remnant info\n#\n')
            for key, entry in self.param_space.snr.__dict__.items():
                f.write('%s: %s\n' % (key, entry))

            f.write('#\n#-------lightcurve info\n#\n')
            for key, entry in self.param_space.lightcurve_params.items():
                f.write('%s: %s\n' % (key, entry))

            f.write('#\n#-------snu_echo_kwargs info\n#\n')
            for key, entry in self.param_space.snu_echo_kwargs.items():
                f.write('%s: %s\n' % (key, entry))

            f.write('#\n#----------------------------------------VERBOSE LOG\n#\n')
            for i, axis in enumerate(self.param_space.axes):
                if axis.is_scan() is False:
                    name = axis.get_name()
                    value = axis.get_point()
                    f.write('axis %d; %s: %s\n' % (i, name, value))
            for axis in self.param_space.axes_scan:
                name = axis.get_name()
                value_min = axis.get_min()
                value_max = axis.get_max()
                length = axis.get_length()
                f.write('long axis; %s: from %e to %e, with %d sample points\n' % (
                    name, value_min, value_max, length))

            f.write(
                '#\n#----------------------------------------VERY VERBOSE LOG\n#\n')
            f.write('#\n#-------detailed log of all inputs\n#\n')
            for key, entry in vars(args).items():
                f.write('%s: %s\n' % (key, entry))


def gen_slice(slice_mode):
    """Prepare the 2D param space

    :returns: ParamSpaceSlice2D instance

    """

    axes = []
    total_axes_name_arr = ["L peak Bietenholz",
                           "t peak",
                           "t trans over t peak",
                           "t age total",
                           "galaxy l",
                           "galaxy b",
                           "distance",
                           "square radians",
                           "spectral index",
                           "nu Bietenholz"
                           ]
    #
    # if one specifies one of the special slices
    #
    # CASE 1:
    if slice_mode == "Lpk-tpk":
        # Defining the arrays
        Nsigs = 3.  # number of standard deviations from the Bietenholz's mean to scan
        # x-array:
        axis = ParamAxis("L peak Bietenholz",
                         x0=10**(ct._mu_log10_Lpk_-Nsigs*ct._sig_log10_Lpk_),
                         x1=10**(ct._mu_log10_Lpk_+Nsigs*ct._sig_log10_Lpk_),
                         steps=args.Nsteps+1,
                         is_log=True)
        axes.append(axis)
        axis = ParamAxis("t peak",
                         x0=10**(ct._mu_log10_tpk_-Nsigs*ct._sig_log10_tpk_),
                         x1=10**(ct._mu_log10_tpk_+Nsigs*ct._sig_log10_tpk_),
                         steps=args.Nsteps+2,
                         is_log=True)
        axes.append(axis)

        axis = ParamAxis("square radians", args.sr)
        axes.append(axis)
        axis = ParamAxis("spectral index", args.alpha)
        axes.append(axis)
        axis = ParamAxis("galaxy l", args.coords[0])
        axes.append(axis)
        axis = ParamAxis("galaxy b", args.coords[1])
        axes.append(axis)
        axis = ParamAxis("distance", args.distance)
        axes.append(axis)
        try:
            axis = ParamAxis("t age", args.t_signal)
            axes.append(axis)
        except Exception:
            axis = ParamAxis("S0", args.S0)
            axes.append(axis)

        # light curve params
        # i'm making nuB also a param here, in case we also want to slice it later
        axis = ParamAxis("nu Bietenholz", args.nuB)
        axes.append(axis)
        axis = ParamAxis("t trans over t peak", args.tt_ratio)
        axes.append(axis)

    # CASE 2:
    elif slice_mode == "tsig-r":
        axis = ParamAxis("t age",
                         x0=10.*(args.tpk/365.),
                         x1=1.e4,
                         steps=args.Nsteps+1,
                         is_log=True)
        axes.append(axis)
        axis = ParamAxis("t trans over t peak",
                         x0=10,
                         x1=100,
                         steps=args.Nsteps+2,
                         is_log=False)
        axes.append(axis)
        axis = ParamAxis("square radians", args.sr)
        axes.append(axis)
        axis = ParamAxis("spectral index", args.alpha)
        axes.append(axis)
        axis = ParamAxis("galaxy l", args.coords[0])
        axes.append(axis)
        axis = ParamAxis("galaxy b", args.coords[1])
        axes.append(axis)
        axis = ParamAxis("distance", args.distance)
        axes.append(axis)

        # light curve params
        # i'm making nuB also a param here, in case we also want to slice it later
        axis = ParamAxis("nu Bietenholz", args.nuB)
        axes.append(axis)
        # axis = ParamAxis("t trans over t peak", args.tt_ratio)
        # axes.append(axis)
        axis = ParamAxis("L peak Bietenholz", args.Lpk)
        axes.append(axis)
        axis = ParamAxis("t peak", args.tpk)
        axes.append(axis)
        # axis = ParamAxis("t age", args.t_signal)
        # axes.append(axis)

    # CASE 3:
    elif slice_mode == "tex-r":
        axis = ParamAxis("t extra",
                         x0=1.e3,
                         x1=1.e5,
                         steps=args.Nsteps+1,
                         is_log=True)
        axes.append(axis)
        axis = ParamAxis("t trans over t peak",
                         x0=10,
                         x1=100,
                         steps=args.Nsteps+2,
                         is_log=False)
        axes.append(axis)
        axis = ParamAxis("square radians", args.sr)
        axes.append(axis)
        axis = ParamAxis("spectral index", args.alpha)
        axes.append(axis)
        axis = ParamAxis("galaxy l", args.coords[0])
        axes.append(axis)
        axis = ParamAxis("galaxy b", args.coords[1])
        axes.append(axis)
        axis = ParamAxis("distance", args.distance)
        axes.append(axis)

        # light curve params
        # i'm making nuB also a param here, in case we also want to slice it later
        axis = ParamAxis("nu Bietenholz", args.nuB)
        axes.append(axis)
        # axis = ParamAxis("t trans over t peak", args.tt_ratio)
        # axes.append(axis)
        axis = ParamAxis("L peak Bietenholz", args.Lpk)
        axes.append(axis)
        axis = ParamAxis("t peak", args.tpk)
        axes.append(axis)
        axis = ParamAxis("t age", args.t_signal)
        axes.append(axis)

    # CASE 4:
    elif slice_mode == "l-D":
        axis = ParamAxis("galaxy l",
                         x0=0.,
                         x1=360.,
                         steps=args.Nsteps+1,
                         is_log=False)
        axes.append(axis)
        axis = ParamAxis("distance",
                         x0=0.1,
                         x1=3.,
                         steps=args.Nsteps+2,
                         is_log=True)
        axes.append(axis)
        axis = ParamAxis("square radians", args.sr)
        axes.append(axis)
        axis = ParamAxis("spectral index", args.alpha)
        axes.append(axis)
        # axis = ParamAxis("galaxy l", args.coords[0])
        # axes.append(axis)
        axis = ParamAxis("galaxy b", args.coords[1])
        axes.append(axis)
        # axis = ParamAxis("distance", args.distance)
        # axes.append(axis)

        # light curve params
        # i'm making nuB also a param here, in case we also want to slice it later
        axis = ParamAxis("nu Bietenholz", args.nuB)
        axes.append(axis)
        axis = ParamAxis("t extra", args.t_extra)
        axes.append(axis)
        axis = ParamAxis("t trans over t peak", args.tt_ratio)
        axes.append(axis)
        axis = ParamAxis("L peak Bietenholz", args.Lpk)
        axes.append(axis)
        axis = ParamAxis("t peak", args.tpk)
        axes.append(axis)
        axis = ParamAxis("t age", args.t_signal)
        axes.append(axis)

    # CASE 5:
    elif slice_mode == "l-b":
        axis = ParamAxis("galaxy l",
                         x0=0.,
                         x1=360.,
                         steps=args.Nsteps+1,
                         is_log=False)
        axes.append(axis)
        axis = ParamAxis("galaxy b",
                         x0=-90.,
                         x1=90.,
                         steps=args.Nsteps+2,
                         is_log=False)
        axes.append(axis)
        axis = ParamAxis("square radians", args.sr)
        axes.append(axis)
        axis = ParamAxis("spectral index", args.alpha)
        axes.append(axis)
        # axis = ParamAxis("galaxy l", args.coords[0])
        # axes.append(axis)
        # axis = ParamAxis("galaxy b", args.coords[1])
        # axes.append(axis)
        axis = ParamAxis("distance", args.distance)
        axes.append(axis)

        # light curve params
        # i'm making nuB also a param here, in case we also want to slice it later
        axis = ParamAxis("nu Bietenholz", args.nuB)
        axes.append(axis)
        axis = ParamAxis("t extra", args.t_extra)
        axes.append(axis)
        axis = ParamAxis("t trans over t peak", args.tt_ratio)
        axes.append(axis)
        axis = ParamAxis("L peak Bietenholz", args.Lpk)
        axes.append(axis)
        axis = ParamAxis("t peak", args.tpk)
        axes.append(axis)
        axis = ParamAxis("t age", args.t_signal)
        axes.append(axis)

    # CASE 6:
    # this total age case will be caught automatically by constructor
    # DONE: add 't total age' to the constructor
    elif slice_mode == "t-D":
        axis = ParamAxis("t age total",
                         x0=(args.tpk/365.),
                         x1=1.e5,
                         steps=args.Nsteps+1,
                         is_log=True)
        axes.append(axis)
        axis = ParamAxis("distance",
                         x0=0.1,
                         x1=3.,
                         steps=args.Nsteps+2,
                         is_log=True)
        axes.append(axis)
        axis = ParamAxis("square radians", args.sr)
        axes.append(axis)
        axis = ParamAxis("spectral index", args.alpha)
        axes.append(axis)
        axis = ParamAxis("galaxy l", args.coords[0])
        axes.append(axis)
        axis = ParamAxis("galaxy b", args.coords[1])
        axes.append(axis)
        # axis = ParamAxis("distance", args.distance)
        # axes.append(axis)

        # light curve params
        axis = ParamAxis("nu Bietenholz", args.nuB)
        axes.append(axis)
        # axis = ParamAxis("t extra", args.t_extra)
        # axes.append(axis)
        axis = ParamAxis("t trans over t peak", args.tt_ratio)
        axes.append(axis)
        axis = ParamAxis("L peak Bietenholz", args.Lpk)
        axes.append(axis)
        axis = ParamAxis("t peak", args.tpk)
        axes.append(axis)
        # axis = ParamAxis("t age", args.t_signal)
        # axes.append(axis)

    # CASE 7:
    # this total age case will be caught automatically by constructor
    elif slice_mode == "l-t":
        axis = ParamAxis("galaxy l",
                         x0=0.,
                         x1=360.,
                         steps=args.Nsteps+1,
                         is_log=False)
        axes.append(axis)
        axis = ParamAxis("t age total",
                         x0=10.*(args.tpk/365.),
                         x1=1.e5,
                         steps=args.Nsteps+2,
                         is_log=True)
        axes.append(axis)
        axis = ParamAxis("square radians", args.sr)
        axes.append(axis)
        axis = ParamAxis("spectral index", args.alpha)
        axes.append(axis)
        # axis = ParamAxis("galaxy l", args.coords[0])
        # axes.append(axis)
        axis = ParamAxis("galaxy b", args.coords[1])
        axes.append(axis)
        axis = ParamAxis("distance", args.distance)
        axes.append(axis)

        # light curve params
        axis = ParamAxis("nu Bietenholz", args.nuB)
        axes.append(axis)
        # axis = ParamAxis("t extra", args.t_extra)
        # axes.append(axis)
        axis = ParamAxis("t trans over t peak", args.tt_ratio)
        axes.append(axis)
        axis = ParamAxis("L peak Bietenholz", args.Lpk)
        axes.append(axis)
        axis = ParamAxis("t peak", args.tpk)
        axes.append(axis)
        # axis = ParamAxis("t age", args.t_signal)
        # axes.append(axis)

    # CASE 8:
    # this total age case will be caught automatically by constructor
    elif slice_mode == "t-b":
        axis = ParamAxis("t age total",
                         x0=10.*(args.tpk/365.),
                         x1=1.e5,
                         steps=args.Nsteps+1,
                         is_log=True)

        axes.append(axis)
        axis = ParamAxis("galaxy b",
                         x0=-90.,
                         x1=90.,
                         steps=args.Nsteps+2,
                         is_log=False)
        axes.append(axis)

        axis = ParamAxis("square radians", args.sr)
        axes.append(axis)
        axis = ParamAxis("spectral index", args.alpha)
        axes.append(axis)
        axis = ParamAxis("galaxy l", args.coords[0])
        axes.append(axis)
        # axis = ParamAxis("galaxy b", args.coords[1])
        # axes.append(axis)
        axis = ParamAxis("distance", args.distance)
        axes.append(axis)

        # light curve params
        axis = ParamAxis("nu Bietenholz", args.nuB)
        axes.append(axis)
        # axis = ParamAxis("t extra", args.t_extra)
        # axes.append(axis)
        axis = ParamAxis("t trans over t peak", args.tt_ratio)
        axes.append(axis)
        axis = ParamAxis("L peak Bietenholz", args.Lpk)
        axes.append(axis)
        axis = ParamAxis("t peak", args.tpk)
        axes.append(axis)
        # axis = ParamAxis("t age", args.t_signal)
        # axes.append(axis)

    # CASE 9:
    # this total age case will be caught automatically by constructor
    elif slice_mode == "t-S0":
        axis = ParamAxis("t age total",
                         x0=10.*(args.tpk/365.),
                         x1=1.e5,
                         steps=args.Nsteps+1,
                         is_log=True)

        axes.append(axis)
        axis = ParamAxis("S0",
                         x0=1.e-5,
                         x1=1.e5,
                         steps=args.Nsteps+2,
                         is_log=True)
        axes.append(axis)

        axis = ParamAxis("square radians", args.sr)
        axes.append(axis)
        axis = ParamAxis("spectral index", args.alpha)
        axes.append(axis)
        axis = ParamAxis("galaxy l", args.coords[0])
        axes.append(axis)
        axis = ParamAxis("galaxy b", args.coords[1])
        axes.append(axis)
        axis = ParamAxis("distance", args.distance)
        axes.append(axis)

        # light curve params
        axis = ParamAxis("nu Bietenholz", args.nuB)
        axes.append(axis)
        # axis = ParamAxis("t extra", args.t_extra)
        # axes.append(axis)
        axis = ParamAxis("t trans over t peak", args.tt_ratio)
        axes.append(axis)
        # axis = ParamAxis("L peak Bietenholz", args.Lpk)
        # axes.append(axis)
        axis = ParamAxis("t peak", args.tpk)
        axes.append(axis)
        # axis = ParamAxis("t age", args.t_signal)
        # axes.append(axis)

    else:
        # now we start the "generic slicing"
        # one can assign any pair of axes to slice in the format
        # say "t peak, galaxy b"

        # error message
        err_msg = "The slice mode you provide is %s. It\
 is not one of the special slice cases, nor can it be\
 parsed as a generic slice. " % slice_mode

        # catch generic modes in the format of say "galaxy l, galaxy b"
        try:
            slice_mode_arr = slice_mode.split(",")
            for i, x in enumerate(slice_mode_arr):
                slice_mode_arr[i] = x.strip()
        except AttributeError:
            raise Exception(err_msg)

        # catch the two long axes
        for ax_name in slice_mode_arr:
            # remove it from total axes
            total_axes_name_arr.remove(ax_name)

            # check what axis is this
            if ax_name == "L peak Bietenholz":
                Nsigs = 3.  # number of standard deviations from the Bietenholz's mean to scan
                # x-array:
                axis = ParamAxis("L peak Bietenholz",
                                 x0=10**(ct._mu_log10_Lpk_ -
                                         Nsigs*ct._sig_log10_Lpk_),
                                 x1=10**(ct._mu_log10_Lpk_ +
                                         Nsigs*ct._sig_log10_Lpk_),
                                 steps=args.Nsteps+1,
                                 is_log=True)
                axes.append(axis)
            elif ax_name == "t peak":
                Nsigs = 3.  # number of standard deviations from the Bietenholz's mean to scan
                axis = ParamAxis("t peak",
                                 x0=10**(ct._mu_log10_tpk_ -
                                         Nsigs*ct._sig_log10_tpk_),
                                 x1=10**(ct._mu_log10_tpk_ +
                                         Nsigs*ct._sig_log10_tpk_),
                                 steps=args.Nsteps+2,
                                 is_log=True)
                axes.append(axis)

            elif ax_name == "t trans over t peak":
                axis = ParamAxis("t trans over t peak",
                                 x0=10,
                                 x1=100,
                                 steps=args.Nsteps+2,
                                 is_log=False)
                axes.append(axis)

            elif ax_name == "t age total":
                axis = ParamAxis("t age total",
                                 x0=10.*(args.tpk/365.),
                                 x1=1.e5,
                                 steps=args.Nsteps+1,
                                 is_log=True)
                axes.append(axis)

            elif ax_name == "galaxy l":
                axis = ParamAxis("galaxy l",
                                 x0=0.,
                                 x1=360.,
                                 steps=args.Nsteps+1,
                                 is_log=False)
                axes.append(axis)

            elif ax_name == "galaxy b":
                axis = ParamAxis("galaxy b",
                                 x0=-90.,
                                 x1=90.,
                                 steps=args.Nsteps+2,
                                 is_log=False)
                axes.append(axis)

            elif ax_name == "distance":
                axis = ParamAxis("distance",
                                 x0=0.1,
                                 x1=3.,
                                 steps=args.Nsteps+2,
                                 is_log=True)
                axes.append(axis)

        # for the rest of unspecified axes, assign the fixed value
        for ax_name in total_axes_name_arr:
            if ax_name == "L peak Bietenholz":
                axis = ParamAxis("L peak Bietenholz", args.Lpk)
                axes.append(axis)
            elif ax_name == "t peak":
                axis = ParamAxis("t peak", args.tpk)
                axes.append(axis)
            elif ax_name == "t trans over t peak":
                axis = ParamAxis("t trans over t peak", args.tt_ratio)
                axes.append(axis)
            elif ax_name == "t age total":
                axis = ParamAxis("t age total", args.tage)
                axes.append(axis)
            elif ax_name == "galaxy l":
                axis = ParamAxis("galaxy l", args.coords[0])
                axes.append(axis)
            elif ax_name == "galaxy b":
                axis = ParamAxis("galaxy b", args.coords[1])
                axes.append(axis)
            elif ax_name == "distance":
                axis = ParamAxis("distance", args.distance)
                axes.append(axis)
            elif ax_name == "square radians":
                axis = ParamAxis("square radians", args.sr)
                axes.append(axis)
            elif ax_name == "spectral index":
                axis = ParamAxis("spectral index", args.alpha)
                axes.append(axis)
            elif ax_name == "nu Bietenholz":
                axis = ParamAxis("nu Bietenholz", args.nuB)
                axes.append(axis)

    # generate 2D parameter space
    param_space = ParamSpaceSlice2D(axes)

    return param_space


# -------------------------------------------------


#############
# ARGUMENTS #
#############

if __name__ == "__main__":

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

    # CASE 9: t-S0 slice
    tS0_parser = subparsers.add_parser(
        't-S0', help="distance-t_total parameter space slice")

    tS0_parser.add_argument('-p', '--tpk', '--t_peak',
                            default=(10.**ct._mu_log10_tpk_),
                            type=float,
                            help="The peak time of the SNR [days] (default: 10^1.7)")
    tS0_parser.add_argument('-r', '--tt_ratio', '--ratio',
                            default=30.,
                            type=float,
                            help="The ratio of t_trans/t_pk (default: 30)")
    tS0_parser.add_argument('-D', '--distance', '--dist',
                            default=1.,
                            type=float,
                            help="The distance to the source [kpc] (default: 1)")
    tS0_parser.add_argument('-lb', '--coords', '--long_lat',
                            default=(0., 0.),
                            type=float,
                            nargs=2,
                            help="The galactic coordinates of the SNR [deg] (default: (0, 0))")

    # Parsing arguments:
    args = parser.parse_args()

    if args.verbose:  # Printing arguments
        print("Parameter slice: ", args.slice)
        print("Arguments: ", args._get_kwargs())

    # Defining the Run ID variable
    run_id = args.run

    # -------------------------------------------------


#
# examples of the use cases (to be deprecated)
# using it inside jupyter is maintained
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

# CASE9:
# python ./run_custom.py --run 1 --nuB 8 --Nsteps 30 t-S0 --tpk 50.1 --tt_ratio 30 -D 0.5 -lb 175 0
