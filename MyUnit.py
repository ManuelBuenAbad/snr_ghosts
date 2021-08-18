"""
.. module:: MyUnit
    :synopsis: Python version to convert SI Units and Natural Units. Created on 12/19/2019.
.. moduleauthor:: Chen Sun <chensun@mail.tau.ac.il>

This module defines a classes called SIUnit and NaturalUnit. It is based on class Unit and utilizes Sympy to parse and simplify expressions. 

# CS 12/21/2019: TODO: check whether it's possible to modify __repr__() method to return symbols directly, instead of 
using the __call__() method.

# CS 06/25/2020: added Gauss

"""

import numpy as np
from sympy import symbols
from sympy.parsing.sympy_parser import parse_expr
# from sympy import simplify
# from sympy import refine, Q
# from sympy import factor
# the factor way, without args same as expr
# with args it's not good
# self.expr = np.array(factor(input).args)
# import re
# list_expr = (re.findall(r"[\w]+",input))
# from sympy import degree
# from sympy import poly
# from sympy import sympify


class Unit:
    def __init__(self, input):
        try:
            # assume text
            self.symb = parse_expr(input)
        except:
            # assume Unit instance
            try:
                self.symb = input.symb
            except:
                # assume symbols
                self.symb = input

        # the string form
        self.str = str(self.symb)

    def __mul__(self, other):
        mul = self.str + '*' + other.str

        # the parsed symbo form
        symb = parse_expr(mul)

        # the string form
        string = str(symb)
        return Unit(string)

    def __div__(self, other):
        div = self.str + '/(' + other.str + ')'

        # the parsed symbo form
        symb = parse_expr(div)

        # the string form
        string = str(symb)
        return Unit(string)

    def __call__(self):
        return self.symb

    def __str__(self):
        return self.str

#     def __repr__(self):
#         self.symb
#         return ''

# # example
# test1 = Unit('10*(m)**2/s**2*s')
# test2 = Unit('(10*m)**3/s**2*s')
# test1 / test2
# test3 = test1 / test2
# test3()


class SIUnit(Unit):
    def __init__(self, input):
        Unit.__init__(self, input)

        # target units
        kg, m, s, K, A = symbols("kg m s K A", positive=True)
        list_atoms = [kg, m, s, K, A]

        # quantum
        h_val = 6.626070040*10**(-34)*kg*m**2/s  # Js
        hbar_val = 1.054571800*10**(-34)*kg*m**2/s  # Js
        eV_val = 1.602176565*10**(-19)*kg*m**2/s**2  # J
        c_val = 299792458*m/s  # m/s
        kB_val = 1.38064852*10**(-23)*kg*m**2/s**2/K  # J/K
        Alfa_val = 1/137.035
        GN_val = 6.67408*10**(-11)*m**3/kg/s**2  # m**3 kg**(-1) s**(-2)
        # E&M
        mu0_val = 1.2566370614*10**(-6)*kg*m/s**2/A**2  # N/A**2=kg m/s**2/A**2
        # A**2 s**4 kg**-1 m**-3
        eps0_val = 8.854187817620*10**(-12)*A**2*s**4/kg/m**3
        e_val = 1.6021766208*10**(-19)*A*s  # C
        T_val = kg/A/s**2
        G_val = 10**(-4)*kg/A/s**2

        list_symbol = np.array(
            symbols("h, hbar, eV, c, kB, Alfa, GN, mu0, eps0, e, T, G"))
        list_val = np.array([h_val, hbar_val, eV_val, c_val, kB_val,
                             Alfa_val, GN_val, mu0_val, eps0_val, e_val, T_val, G_val])

        for ind, symbol in enumerate(list_symbol):
            self.symb = self.symb.subs(symbol, list_val[ind])

        # update str
        self.str = str(self.symb)
        # combine powers such as 1/s and 1/s**2
        self.symb = parse_expr(self.str)
        self.str = str(self.symb)
# # example
# test = SIUnit('hbar/s')
# test()


class NaturalUnit(Unit):
    def __init__(self, input):
        Unit.__init__(self, input)

        # target units
        eV = symbols("eV")

        # quantum
        h_val = 2*np.pi
        hbar_val = 1

        # consts
        c_val = 1
        kB_val = 1
        mu0_val = 1
        eps0_val = 1

        J_val = (1.602176565*10**(-19))**(-1) * eV
        e_val = np.sqrt(4*np.pi*1/137.035)  # at q=0
        Alfa_val = 1/137.035  # at q=0
        T_val = np.sqrt((1.054571800*10**(-34))**3*(299792458)**3/(1.2566370614 *
                                                                   10**-6)/(1.602176565*10**-19)**4)*eV**2
        G_val = 1e-4*T_val
        kg_val = ((299792458)**2/(1.602176565*10**-19))*eV
        s_val = ((1.602176565*10**-19)/(1.054571800*10**-34))*eV**-1
        m_val = ((1.602176565*10**-19)/(1.054571800*10**-34)/(
            299792458))*eV**-1
        cm_val = 10**-2*((1.602176565*10**-19)/(1.054571800*10**-34)/(
            299792458))*eV**-1
        GN_val = (6.67408*10**-11) *\
            ((1.602176565*10**-19)/(1.054571800*10**-34)/(299792458))**3 /\
            ((299792458)**2/(1.602176565*10**-19)) / \
            ((1.602176565*10**-19)/(1.054571800*10**-34))**2*eV**-2
        AlphaMZ_val = 1/127.950
        GF_val = 1.1663787*10**(-5)*10**(-18)*eV**(-2)
        K_val = (11604.52)**(-1)*eV
        A_val = 1244.06*eV

        # order of mag
        # length
        fm_val = 1e-15 * m_val
        nm_val = 1e-9 * m_val
        cm_val = 1e-2 * m_val
        km_val = 1000 * m_val
        au_val = 149597870700*m_val
        ly_val = 9.4607*10**15*m_val
        pc_val = 3.0857*10**16*m_val
        kpc_val = 3.0857*10**19*m_val
        Mpc_val = 3.0857*10**22*m_val
        Gpc_val = 3.0857*10**25*m_val
        year_val = 365.25*24*60*60 * s_val

        # eV
        keV_val = 1e3 * eV
        MeV_val = 1e6 * eV
        GeV_val = 1e9 * eV
        TeV_val = 1e12 * eV

        # phys quantities
        # astro
        Rsun_val = 6.96*10**8 * m_val
        Msun_val = 1.98855*10**30*kg_val
        RsgrA_val = 17.*Rsun_val
        MsgrA_val = 4e6 * Msun_val
        Rearth_val = 6371*10**3*m_val
        Mearth_val = 5.97237*10**24*kg_val
        # particle
        sw_val = np.sqrt(0.2386)
        cosCabibbo2_val = 0.9746  # +/- 0.0008  astro - ph/0302055
        mproton_val = 938.2720813*10**6 * eV  # Wiki
        mneutron_val = 939.5654133*10**6 * eV  # Wiki
        melectron_val = 0.511*10**6 * eV
        mmuon_val = 105.65837*10**6 * eV
        mpi0_val = 134.977*10**6 * eV  # neutral pion mass
        mpiplus_val = 139.57061 * 10**6 * eV  # pi plus mass
        mk0_val = 497.611*10**6 * eV  # Kaon zero
        vev_val = 246221000000 * eV
        Mz_val = 91187600000 * eV
        NA_val = 6.022*10**23 / symbols('mol')  # Avogadro
        Mplr_val = 2.435*10**18*10**9 * eV  # Reduced
        Mpl_val = 1.220910*10**19*10**9 * eV
        barn_val = 10**-24 * cm_val**2

        # cosmology
        Om_m_val = 0.308
        Om_L_val = 0.692
        Om_b_val = 0.04842
        Om_c_val = 0.2580
        # aquired by integrating 2.726K black body  c.f. Hubble_BAO.nb for example
        Om_g_val = 0.0000538357
        Om_nu_val = 3.4*10**-5
        h0_val = 0.678
        H0_val = 67.8*10**3*m_val/s_val/(3.0857e22 * m_val)  # arXiv:1502.01589
        # 3H0**2/(8\[Pi]*GN)/.physQuantity/.physSI2Natural*)
        rho_c_val = 3.721623621707084e-11 * (eV**4)

        # first round of sub
        list_symbol = np.array(symbols("h, hbar, c, kB, mu0, eps0, J, e, Alfa,\
                            T, G, kg, s, m, cm, GN, AlphaMZ, GF, K, A, fm, nm,\
                            cm, km, au, ly, pc, kpc, Mpc, Gpc, year,\
                            keV, MeV, GeV, TeV"))
        list_val = np.array([h_val, hbar_val, c_val, kB_val, mu0_val, eps0_val, J_val, e_val, Alfa_val,
                             T_val, G_val, kg_val, s_val, m_val, cm_val, GN_val, AlphaMZ_val, GF_val, K_val, A_val, fm_val, nm_val,
                             cm_val, km_val, au_val, ly_val, pc_val, kpc_val, Mpc_val, Gpc_val, year_val,
                             keV_val, MeV_val, GeV_val, TeV_val])
        for ind, symbol in enumerate(list_symbol):
            self.symb = self.symb.subs(symbol, list_val[ind])

        # second round of sub
        list_symbol = np.array(symbols("Rsun,Msun,RsgrA,MsgrA,Rearth,Mearth,sw,\
                             cosCabibbo2,mproton,mneutron,melectron,mmuon,\
                             mpi0,mpiplus,mk0,vev,Mz,NA,Mplr,Mpl,\
                             barn,Om_m,Om_L,Om_b,Om_c,Om_g,Om_nu,h0,\
                             H0,rho_c"))
        list_val = np.array([Rsun_val, Msun_val, RsgrA_val, MsgrA_val, Rearth_val, Mearth_val, sw_val,
                             cosCabibbo2_val, mproton_val, mneutron_val, melectron_val, mmuon_val,
                             mpi0_val, mpiplus_val, mk0_val, vev_val, Mz_val, NA_val, Mplr_val, Mpl_val,
                             barn_val, Om_m_val, Om_L_val, Om_b_val, Om_c_val, Om_g_val, Om_nu_val, h0_val,
                             H0_val, rho_c_val])
        for ind, symbol in enumerate(list_symbol):
            self.symb = self.symb.subs(symbol, list_val[ind])

        # update str
        self.str = str(self.symb)
        # combine powers such as 1/eV and 1/eV**2
        self.symb = parse_expr(self.str)
        self.str = str(self.symb)

        # number form for dim-less quant
        try:
            self.val = float(self.str)
        except:
            self.val = self.str

    def mass(self):
        return self.symb.subs(symbols('eV'), symbols('eV')/symbols('c')**2)

    def mom(self):
        return self.symb.subs(symbols('eV'), symbols('eV')/symbols('c'))

    def T(self):
        return self.symb.subs(symbols('eV'), symbols('eV')/symbols('kB'))

    def time(self):
        return self.symb.subs(symbols('eV'), symbols('eV')/symbols('hbar'))

    def length(self):
        return self.symb.subs(symbols('eV'), symbols('eV')/symbols('hbar')/symbols('c'))

    def power(self):
        return self.symb.subs(symbols('eV'), symbols('eV')/(symbols('hbar'))**(1./2.))

    def GeV(self):
        return self.symb.subs(symbols('eV'), symbols('GeV')/1e9)

    def Mpl(self):
        return self.symb.subs(symbols('eV'), symbols('Mpl')*8.19061192061659e-29)

    def Mpc(self):
        return self.symb.subs(symbols('eV'), symbols('Mpc')**-1. * 1.56374962590552e29)


# # # example for testing
# test = NaturalUnit('c*year')
# test()
# # str(test.mass())
# test2 = SIUnit(test)
# test3 = NaturalUnit(test2)
# print test3()
# print SIUnit(test3.length())()
# print SIUnit(test3.mass())()
# print SIUnit(test3.T())()
# print SIUnit(test3.mom())()
# print SIUnit(test3.time())()
# print NaturalUnit(SIUnit(test3.power())())
# print SIUnit(test3.power())
# print test3.GeV()
# print test3.Mpc()
# print test3.Mpl()
# NaturalUnit('Mpl').Mpl()

# should give
# 4.79444331506341e+22/eV
# 9.46073047258078e+15*m
# 2.68948557400311e+58/kg
# 4.13153032678779e+18/K
# 8.9711582204083e+49*s/(kg*m)
# 31557599.9999999*s
# 4.79444331506341e+22*eV**(-0.5)*(1/eV)**0.5
# 3.07302356212126e+24*kg**(-0.5)*m**(-1.0)*s**1.5
# 4.79444331506341e+31/GeV
# 3.06599166237184e-7*Mpc**1.0
# 5.85358378779407e+50/Mpl
# JavaScript: 1.0 Mpl
