"""
Module that loads the data components
"""

import numpy as np
#import requests
from astroquery.simbad import Simbad
from astropy.coordinates import SkyCoord
from datetime import datetime
import re
import os
import constants as ct

# location of the data path
data_path = os.path.dirname(os.path.abspath(__file__))+'/data/'

class SuperNova(object):
    """Class of SN. Each SN of the Bietenholz dataset is one instance of this class

    """

    def __init__(self):
        self.name = ''
        self.is_limit = np.asarray([], dtype=bool)  # whether it's a limit
        self.year = np.asarray([])  # year of data point
        self.month = np.asarray([])  # month of the data point
        self.day = np.asarray([])  # day of the data point
        # days since explosion, will be updated in gen_time_axis()
        self.t = np.asarray([])
        self.telescope = np.asarray([])  # the telescope id
        # the freq of observation,4-10 GHz, except for 1987A (2.3 GHz)
        self.freq = np.asarray([])
        self.flux = np.asarray([])  # [mJy]
        self.dflux = np.asarray([])  # [mJy]
        self.comment = np.asarray([])  # extra comment

        # extended info
        self.type = None
        self.galaxy = None  # galaxy of residence
        self.dist = None  # dist [Mpc]
        self.explosion_date = None  # date of explosion [yyyy mm dd]
        self.number_of_measurements = None
        self.detected = None
        self.has_explosion_time = False

    def sanity_check():
        # TODO: add some sanity checks
        return

    def gen_time_axis(self):
        """Generate time array, each point is the time since explosion [day]

        """
        try:
            t_explosion = datetime.strptime(self.explosion_date, "%Y %m %d")
            t_since_explosion = []
            for i in range(len(self.year)):
                t_i = datetime.strptime(
                    "%d-%d-%d" % (self.year[i], self.month[i], self.day[i]), "%Y-%m-%d")
                t_since_explosion.append((t_i - t_explosion).days)
            self.t = np.asarray(t_since_explosion)
            self.has_explosion_time = True
            return np.asarray(t_since_explosion)
        except TypeError:
            pass


def load_Bietenholz(path):
    """Load Bietenholz table 1 (note that it's not all 294 SNe. only those 100+ new SNe)

    :param path: path of the data file

    """

    res = {}
    num_of_SN = 0
    ex = ""

    with open(path, 'r') as f:
        for i in range(30):
            next(f)
        for line in f:
            words = line.split()
            current_SN_name = words[0]
            # determine if it's a new SN
            if current_SN_name != ex:
                if num_of_SN > 0:
                    res[ex] = SN  # save previous SN
                SN = SuperNova()
                num_of_SN += 1
                ex = words[0]

            SN.name = words[0]
            if ('L' in line[10]):
                SN.is_limit = np.append(SN.is_limit, True)
            else:
                SN.is_limit = np.append(SN.is_limit, False)
            SN.year = np.append(SN.year, int(line[12:16]))
            SN.month = np.append(SN.month, int(line[17:19]))
            SN.day = np.append(SN.day, float(line[20:25]))
            SN.telescope = np.append(SN.telescope, line[26:33])
            SN.freq = np.append(SN.freq, float(line[35:40]))
            SN.flux = np.append(SN.flux, float(line[41:49]))
            SN.dflux = np.append(SN.dflux, float(line[50:56]))
            SN.comment = np.append(SN.comment, line[57:63])
    res[words[0]] = SN
    return res


def load_table2(path):
    """load Bietenholz Table 2 data. Returns array rows

    """
    res = []
    with open(path, 'r') as f:
        for line in f:
            words = line.split('&')

            # clean up white spaces
            words_clean = np.asarray([x.strip() for x in words], dtype=object)

            # clean up the white space in name
            name_str = words_clean[0]
            name_arr = name_str.split()
            name_new = ''
            for name_part in name_arr:
                name_new = name_new + name_part
            words_clean[0] = name_new

            # clean up distance to number
            dist = float(words_clean[3])
            words_clean[3] = dist

            # save
            res.append(words_clean)
    res = np.asarray(res, dtype=object)

    # save into a dictionary
    res_dct = {}
    for i in range(len(res)):
        res_dct[res[i, 0]] = res[i]

    # return np.asarray(res, dtype=object)
    return res_dct


def update_Bietenholz_with_table2(SNe_dct, table2_dct):
    """Update the result of Table 1 with meta-info from Table 2

    :param SNe_dct: the result from load_Bietenholz() return
    :param table2_dct: the dict from load_table2() return

    """

    for key, SN in SNe_dct.items():
        try:
            entry = table2_dct[key]
            SN.type = entry[1]
            SN.galaxy = entry[2]
            SN.distance = entry[3]
            SN.explosion_date = entry[4]
        except KeyError:
            print('%s failed updating' % key)
    return


def clean_white_spaces(string):
    res = ''
    words = string.split()
    for word in words:
        res = res + str(word)
    return res


def update_Bietenholz_with_coord(SNe_dct, use_Simbad=False):
    """query simbad to get SN coordinates. Requires network when use_Simbad is True

    """

    names = SNe_dct.keys()
    if use_Simbad:
        # use online query
        query = Simbad.query_objects(names)
        query_res = query.as_array()
        with open(data_path+'Table_SN_coord.txt', 'w') as f:
            for i in range(len(query_res)):
                entry = query_res[i]
                name = entry[0]
                name = clean_white_spaces(name)
                RA = entry[1]
                DEC = entry[2]

                RA = RA
                DEC = DEC
                try:
                    SN = SNe_dct[name]
                except:
                    print('%s failed the query' % (names[i]))
                SN.RA = RA
                SN.DEC = DEC
                try:
                    SN.l, SN.b = simbad_to_galactic(RA, DEC)
                except TypeError:
                    print(name)
                    print(RA)
                    print(DEC)

                # TODO: save the query result
                f.write('%s, %s, %s\n' % (name, RA, DEC))
    else:
        # use offline file
        i = 0
        try:
            with open(data_path+'Table_SN_coord.txt', 'r') as f:
                for line in f:
                    words = line.split(',')
                    name = words[0].strip()
                    RA = words[1].strip()
                    DEC = words[2].strip()
                    try:
                        SN = SNe_dct[name]
                    except:
                        print('%s failed the query' % (names[i]))
                    SN.RA = RA
                    SN.DEC = DEC
                    SN.l, SN.b = simbad_to_galactic(RA, DEC)
                    i += 1
        except:
            raise Exception(
                'Error in using local cache. Try to run with use_Simbad=True to regenerate the cache in {}Table_SN_coord.txt'.format(data_path))
    return


def simbad_to_galactic(RA, DEC):
    """For simbad by default returns RA "h:m:s" and DEC "d:m:s" in ICRS(ep=J2000,eq=2000), this is a utility function that converts (RA, DEC) to (l, b) in galactic coordinate system.

    """
    RA_arr = RA.split()
    DEC_arr = DEC.split()
    if len(DEC_arr) == 3:
        coord = SkyCoord('%sh%sm%ss' % tuple(RA_arr),
                         '%sd%sm%ss' % tuple(DEC_arr))
    else:
        # some of the records (SN1982F and SN1980O) only have d-m not d-m-s
        coord = SkyCoord('%sh%sm%ss' % tuple(RA_arr),
                         '%sd%sm' % tuple(DEC_arr))
    coord_gal = coord.galactic
    return (coord_gal.l, coord_gal.b)


# def test():
#     url = 'http://simbad.u-strasbg.fr/simbad/sim-basic?Ident=SN1997eg&submit=SIMBAD+search'
#     #myobj = {'somekey': 'somevalue'}
#     x = requests.get(url)  # , data=myobj)
#     print(x.text)
#     return x


def test2():
    res = Simbad.query_objects(
        ["SN1995ad", "SN1996L", "SN2012a", "SN1987A", "SN1982F", "SN1980O"])
    return res


def gen_SN_with_table2(table2_dct):
    SNe_dct = {}
    for key, entry in table2_dct.items():
        try:
            SN = SuperNova()
            SN.type = entry[1]
            SN.galaxy = entry[2]
            SN.distance = entry[3]
            SN.explosion_date = entry[4]
            SNe_dct[key] = SN
        except KeyError:
            print('%s failed updating' % key)
    return SNe_dct


#
# SNR
#

class SuperNovaRemnant(object):
    """ The SNR class
    """

    def __init__(self):
        self.no_dist = True  # whether the record has a distance meas
        self.no_flux = True  # whether the record has a flux meas.
        self.is_complete = False  # the completeness condition from Green 2005
        self.is_spectral_certain = False
        self.is_flux_certain = False
        self.is_type_certain = False

    def set_coord(self, l, sign, b):
        """set the snr coordinate

        :param l: longitude in galactic frame
        :param b: height in galactic frame

        """
        self.l = float(l)
        self.b = float(b)
        if sign == '-':
            self.b *= -1.

    def set_flux_density(self, Snu, is_flux_certain):
        """Set the flux density according 

        :param Snu: flux density at 1 GHz [Jy]
        :param is_flux_certain: flag to show whether it has large uncertainty in distance

        """

        self.snu_at_1GHz = float(Snu)
        if is_flux_certain == "?":
            self.is_flux_certain = False
        else:
            self.is_flux_certain = True
        self.no_flux = False
        return

    def set_size(self, length, width=None):
        """Set the size of the SNR. Some data points are in a x b format, i.e. rectangle, while some have only one number. In the former case, we take the geometric mean as the angular size [arcmin]. The sr value is also updated
        :param length: the length in [arcmin]
        :param width: the width [arcmin]
        """

        length = float(length)
        try:
            width = float(width)
        except:
            pass
        if width is not None:
            self.ang_size = np.sqrt(length * width)
        else:
            self.ang_size = length

        ang_size_in_rad = self.ang_size / 60 * np.pi / 180
        self.sr = ct.angle_to_solid_angle(ang_size_in_rad)

    def set_distance(self, dist_arr):
        """Some data points have multiple distance measurements. We take average in these cases

        :param dist_arr: distance measurement array from different tracers [kpc]

        """
        dist_arr = np.asarray(dist_arr).astype(float)
        dist_arr_clean = []
        for x in dist_arr:
            # some sanity check to filter out rare junk that arises from the regex parsing
            if x < 50:
                dist_arr_clean.append(x)
        res = np.mean(dist_arr)
        self.distance = res
        self.no_dist = False

    def set_name(self, name):
        """name the SNR

        :param name: string of name

        """
        self.name = name

    def set_type(self, snr_type, is_certain):
        """set SNR type, S: shell, F: fill, C: combination

        :param snr_type: type of SNR 
        :param is_certain: flag to show if it's certain. "?" set to False. 

        """

        self.type = snr_type
        if is_certain is not None:
            if is_certain != "?":
                # print(is_certain)
                self.is_type_certain = True

    def set_spectral(self, alpha, is_certain):
        """set the spectral index alpha, as defined in S_nu\propto \nu^{-\alpha}

        :param alpha: spectral index
        :param is_certain: flag to show if it's certain

        """
        self.alpha = float(alpha)
        if is_certain == "?":
            self.is_spectral_certain = False
        else:
            self.is_spectral_certain = True

    def set_age(self, age):
        """set up the age of the SNR
        """
        self.age = float(age)

    def get_age(self):
        try:
            return self.age
        except Exception:
            return None

    def get_luminosity(self):
        """get the luminosity of the SNR in [erg/s/Hz]

        """

        if self.no_dist is False and self.no_flux is False:
            dist = self.distance

            snu = self.snu_at_1GHz
            lum = snu * 4.*np.pi * dist**2 * \
                ct._Jy_over_cgs_irrad_ * (ct._kpc_over_cm_)**2
            self.lum = lum
        else:
            self.lum = -1  # use -1 to indicate unknown luminosity
        return self.lum

    def get_diameter(self):
        """compute the diameter of the SNR [pc]

        """

        if self.no_dist is False and self.no_flux is False:
            dist = self.distance
            diam = dist * self.ang_size / 60. * np.pi/180. * ct._kpc_over_pc_
            self.diam = diam
        else:
            self.diam = -1  # use -1 to indicate unknown diameter

        return self.diam

    def get_SB(self):
        """get the surface brightness [Jy/sr]

        """
        sr = self.sr
        Sigma = self.snu_at_1GHz / sr
        self.Sigma = Sigma
        # set the flag of SB completeness
        Sigma_in_SI = Sigma * ct._Jy_over_SI_
        if Sigma_in_SI > 1.e-20:
            self.is_complete = True
        return Sigma

    def get_spectral_index(self):
        """get the spectral index alpha

        """
        try:
            return self.alpha
        except AttributeError:
            return None

    def get_type(self):
        try:
            return self.type
        except AttributeError:
            return None

    def get_distance(self):
        if self.no_dist is False:
            return self.distance
        else:
            raise Exception('no distance for %s SNR' % self.name)

    def get_coord(self):
        return (self.l, self.b)

    def get_longitude(self):
        return self.l

    def get_latitude(self):
        return self.b

    def get_size(self):
        try:
            return self.sr
        except AttributeError:
            return None

    def get_flux_density(self):
        """get the flux density [Jy]

        """
        if self.no_flux is False:
            return self.snu_at_1GHz
        else:
            return -1


def load_Green_catalogue_names(path=data_path+'snr_website/www.mrao.cam.ac.uk/surveys/snrs/snrs.list.html'):
    # first load the file list
    file_snrs_list = []
    with open(path, 'r') as f:
        for i in range(49):
            next(f)
        for line in f:
            file_snrs_list.append(line)

    # now load the names
    snr_name_arr = []
    for entry in file_snrs_list:
        m = re.search('.*snrs\.(.*\..*)\.html.*', entry)
        if m is not None:
            # print(m.group(1))
            name = m.group(1)
            snr_name_arr.append(name)

    return snr_name_arr


def load_Green_catalogue(snr_name_arr, pathroot=data_path+'snr_website/www.mrao.cam.ac.uk/surveys/snrs/', verbose=0):
    """Load the Green 2019 catalog

    :param snr_name_arr: array of SNR names
    :param pathroot: directory where the SNR catalog is saved

    """

    snrs_dct = {}
    # pathroot = '/home/chen/Downloads/snr_website/www.mrao.cam.ac.uk/surveys/snrs/'
    for snr_name in snr_name_arr:
        path = os.path.join(pathroot, 'snrs.'+snr_name+'.html')

        try:
            snr_obj = snrs_dct[snr_name]
        except KeyError:
            flg_new = True
            snr_obj = SuperNovaRemnant()
            snr_obj.set_name(snr_name)

        # parse coordinate
        m = re.search(
            'G(\d\d\d\.\d|\d\d\.\d|\d\.\d)([+-])(\d\d\.\d|\d\.\d)', snr_name)
        l = m.group(1)
        sign = m.group(2)
        b = m.group(3)
        snr_obj.set_coord(l, sign, b)

        with open(path, 'r') as f:
            for line in f:
                # parse coordinate
                #             m = re.search('.*Right Ascension:(.*\d).*Declination:.*(&minus|\+).*(\d\d \d\d)\n', line)
                #             if m is not None:
                #                 ra =  m.group(1)
                #                 if m.group(2) == '&minus':
                #                     dec = '-' + m.group(3)
                #                 else:
                #                     dec = '+' + m.group(3)
                #                 snr_obj.set_coord(ra, dec)

                # parse flux density
                m = re.search(
                    '.*Flux density at 1 GHz \(/Jy\): (\d+\.\d+|\d+)([?]*).*', line)
                if m is not None:
                    snu_at_1GHz = m.group(1)
                    is_flux_certain = m.group(2)
                    snr_obj.set_flux_density(snu_at_1GHz, is_flux_certain)

                # parse size aa x bb format for rectangle
                is_square = True
                m = re.search(
                    'Size \(/arcmin\): (\d+\.\d+|\d+)&#215;(\d+\.\d+|\d+)', line)
                if m is not None:
                    is_square = False
                    length = m.group(1)
                    width = m.group(2)
                    snr_obj.set_size(length, width)

                # parse size for square data points
                if is_square:
                    m = re.search('Size \(/arcmin\): (\d+\.\d+|\d+)', line)
                    if m is not None:
                        length = m.group(1)
                        snr_obj.set_size(length)

                # parse distance
                #m = re.search('Distance.*?(\d+.\d+|\d+)&nbsp;kpc.*?(\d+.\d+|\d+)&nbsp;kpc.*?(\d+.\d+|\d+)&nbsp;kpc', line)
                #m = re.search('Distance.*?(\d+.\d+|\d+)&nbsp;kpc.*?(\d+.\d+|\d+)&nbsp;kpc', line)

                # three distances
                m = re.search(
                    '^Distance.*?(\d+\.\d+).*?(\d+\.\d+).*?(\d+\.\d+)(.*?)$', line)
                if m is not None:
                    dist1 = m.group(1)
                    dist2 = m.group(2)
                    dist3 = m.group(3)
                    snr_obj.set_distance([dist1, dist2, dist3])
                    if verbose > 4:
                        print(m.group(4))

                # two distances
                if snr_obj.no_dist is True:
                    m = re.search(
                        'Distance.*?(\d+\.\d+).*?(\d+\.\d+)(.*?)$', line)
                    if m is not None:
                        dist1 = m.group(1)
                        dist2 = m.group(2)
                        snr_obj.set_distance([dist1, dist2])
                        if verbose > 4:
                            print(m.group(3))

                # one distance
                if snr_obj.no_dist is True:
                    m = re.search('Distance.*?(\d+\.\d+)(.*?)$', line)
                    if m is not None:
                        dist1 = m.group(1)
                        snr_obj.set_distance([dist1])
                        if verbose > 4:
                            print(m.group(2))

                # one distance, with \pm sign
                if snr_obj.no_dist is True:
                    m = re.search('Distance.*?(\d+)&#177;\d+(.*?)$', line)
                    if m is not None:
                        dist1 = m.group(1)
                        snr_obj.set_distance([dist1])
                        if verbose > 4:
                            print(m.group(2))

                # parse type - first, all that have flux set to certain ones:
                m = re.search('Type: (S|C|F)', line)
                if m is not None:
                    snr_type = m.group(1)
                    snr_obj.set_type(snr_type, is_certain=True)

                # parse type - next, find the uncertain ones
                m = re.search('Type: (S|C|F)(\?)', line)
                if m is not None:
                    snr_type = m.group(1)
                    is_certain = m.group(2)
                    snr_obj.set_type(snr_type, is_certain)

                # parse snr spectral index
                # x.yy type
                m = re.search('Spectral Index: (\d.\d\d)([?]*).*', line)
                if m is not None:
                    alpha = m.group(1)
                    is_certain = m.group(2)
                    snr_obj.set_spectral(alpha, is_certain)
                    # snr_obj.set_is_spectral_certain(is_certain)
                else:
                    # x.y type
                    mm = re.search('Spectral Index: (\d.\d)([?]*).*', line)
                    if mm is not None:
                        alpha = mm.group(1)
                        is_certain = mm.group(2)
                        snr_obj.set_spectral(alpha, is_certain)

                # parse the remnant with known age
                m = re.search('AD</SPAN>(\d\d\d\d).*$', line)
                if m is not None:
                    print('%s is suggested to be related to SN explosion at AD:%s' % (snr_name, m.group(1)))
                    snr_obj.set_age(2021 - float(m.group(1)))
                else:
                    m = re.search('AD</SPAN>(\d\d\d).*$', line)
                    if m is not None:
                        print('%s is suggested to be related to SN explosion at AD:%s' % (snr_name, m.group(1)))
                        snr_obj.set_age(2021 - float(m.group(1)))
                    # the following doesn't yield anything so skip it for now.
                    # else:
                    #     m = re.search('AD</SPAN>(\d\d).*$', line)
                    #     if m is not None:
                    #         print('%s exploded at AD:%s' %
                    #               (snr_name, m.group(1)))
                    #     else:
                    #         m = re.search('AD</SPAN>(\d).*$', line)
                    #         if m is not None:
                    #             print('%s exploded at AD:%s' %
                    #                   (snr_name, m.group(1)))
                    else:
                        m = re.search('.*remnant of(.*)', line)
                        if m is not None:
                            print('%s could be related to %s' %(snr_name, m.group(1)))
                # the special one, Cas A, not in standard format
                if snr_obj.name == 'G111.7-2.1':
                    snr_obj.set_age(2021 - 1700)

        if flg_new:
            snrs_dct[snr_name] = snr_obj
    
        age = snr_obj.get_age()
        if age is not None:
            print('it is about %.0f years old.' %age)

    return snrs_dct
