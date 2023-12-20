#!/usr/bin/env python
# coding: utf-8

import os, sys
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
import scipy.integrate
from scipy.integrate import quad
from scipy.integrate import dblquad
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar
import pandas as pd
from astropy.io import fits
from astropy import constants as ct
from astropy import units as u
import bisect
import time


################################################################


################ constants & data import ################

# constants
g_A = 1.27 # axial form factor [unitless]
R_odot = 8.5 #* u.kpc # Sun's distance from centre of MW [kpc]
rho_s = 0.51 # DM radial scale for MW NFW profile [GeV/cm^3]; https://arxiv.org/abs/1906.08419
r_s = 8.1 # DM radial scale for MW NFW profile [kpc]; https://arxiv.org/abs/1906.08419
M_bulge = 1.5e10 * ct.M_sun.to_value(u.kg) # [kg]
c_bulge = 0.6 # bulge scale radius [kpc]
M_disk = 7e10 * ct.M_sun.to_value(u.kg) # [kg]
b_disk = 4 # disk scale radius [kpc]
energyScale = 10 # [MeV]

# dictionaries
# {nucleus: [mass [MeV], spin [unitless]]}
nuc_dict = {'He4': [3758.26, 0.0], 'C12': [11274.78, 0.0], 'N14': [13153.91, 0.0], 'O16': [15033.04, 0.0]}
nuc_unitless_dict = {'He4': {'mass_unitless': 3758.26 / energyScale, 'spin': 0.0}, 'C12': {'mass_unitless': 11274.78 / energyScale, 'spin': 0.0}, \
                     'N14': {'mass_unitless': 13153.91 / energyScale, 'spin': 0.0}, 'O16': {'mass_unitless': 15033.04 / energyScale, 'spin': 0.0}}


# importing C12 and O16 data: excitation energies [MeV] and GT strengths [unitless]
C12_data = '{}_{}_{}.txt'.format('C12', nuc_dict['C12'][0], nuc_dict['C12'][1])
O16_data = '{}_{}_{}.txt'.format('O16', nuc_dict['O16'][0], nuc_dict['O16'][1])
Cdf = pd.read_csv(C12_data, sep='\t', names=['E_gamma [MeV]', 'GT strength'], skiprows=1)
Odf = pd.read_csv(O16_data, sep='\t', names=['E_gamma [MeV]', 'GT strength'], skiprows=1)
dEs_dict = {'C12': Cdf['E_gamma [MeV]'], 'O16': Odf['E_gamma [MeV]']}
GTs_dict = {'C12': Cdf['GT strength'], 'O16': Odf['GT strength']}


# FOR NOW, using optimal m_chi values at r = R_odot
def read_optimal_m_chis(nucleus):
    df = pd.read_csv('optimal_m_chis_{}_r_8500pc.txt'.format(nucleus), \
                     sep='\t', \
                     names=['Nuclear excitation energy [MeV]', 'Optimal DM mass [MeV]'], \
                     skiprows=1)
    return [df['Nuclear excitation energy [MeV]'], df['Optimal DM mass [MeV]']]
m_chis_dEs_C12 = read_optimal_m_chis('C12')
m_chis_C12 = m_chis_dEs_C12[1]
m_chis_dEs_O16 = read_optimal_m_chis('O16')
m_chis_O16 = m_chis_dEs_O16[1]
allCOMasses = np.append(m_chis_C12, m_chis_O16)
allCOdEs = np.append(dEs_dict['C12'], dEs_dict['O16'])


################################################################


################ GALPROP density data ################

# - https://galprop.stanford.edu/download/manuals/galprop_v54.pdf
# - https://fits.gsfc.nasa.gov/users_guide/usersguide.pdf
# - https://astropy4cambridge.readthedocs.io/en/latest/_static/Astropy%20-%20Handling%20FITS%20files.html

hdul_CO = fits.open('massdensity_CO.fits')
# fits file header info:
# hdul_CO[0].header

# radial data
# hdul_CO[1].data

data = hdul_CO[0].data # [kg/cm^3] density data, see below for explanation
data_unitless = hdul_CO[0].data / (rho_s * u.GeV * (1/u.cm)**3).to_value(u.kg * (1/u.cm)**3, u.mass_energy()) # [unitless] density data, see below for explanation
"""
density data: hdul_CO[0].data[r bin index, b index, l index]
    r bin index: 0 through 8 (axis 3)
    b index: 0 through 359 (axis 2)
    l index: 0 through 719 (axis 1)
    example: hdul_CO[0].data[8, 359, 719]
    gives the density in kg/cm^3 for a given (r, b, l)
    note GALPROP specifies Rsun = 8.5 kpc, Vsun = 220 km/s 
"""
bins = hdul_CO[1].data # [kpc] radial bin boundaries

r_bins = [bins[0][0]] # r bin smallest value
for i, j in bins: # creating r bins from GALPROP increments
    r_bins.append(j)
# r_bins

r_unitless_bins = [bins[0][0] / r_s] # r bin smallest value
for i, j in bins: # creating r bins from GALPROP increments
    r_unitless_bins.append(j / r_s)
# r_unitless_bins


# creating b and l arrays
b_len, l_len = hdul_CO[0].header['NAXIS2'], hdul_CO[0].header['NAXIS1'] # length of b, l arrays
b_crval, l_crval = hdul_CO[0].header['CRVAL2'], hdul_CO[0].header['CRVAL1'] # central values of b,l
b_delta, l_delta = hdul_CO[0].header['CDELT2'], hdul_CO[0].header['CDELT1'] # increments for b, l

bs = list(np.arange(b_crval, b_crval + b_len*b_delta, b_delta))
ls = list(np.arange(l_crval, l_crval + l_len*l_delta, l_delta))

b_crval_rad, l_crval_rad = np.radians(hdul_CO[0].header['CRVAL2']), np.radians(hdul_CO[0].header['CRVAL1']) # central values of b,l
b_delta_rad, l_delta_rad = np.radians(hdul_CO[0].header['CDELT2']), np.radians(hdul_CO[0].header['CDELT1']) # increments for b, l

bs_rad = list(np.arange(b_crval_rad, b_crval_rad + b_len*b_delta_rad, b_delta_rad))
ls_rad = list(np.arange(l_crval_rad, l_crval_rad + l_len*l_delta_rad, l_delta_rad))


################################################################


################ rotation curve from Clemens (1985) ################

# see https://articles.adsabs.harvard.edu/pdf/1985ApJ...295..422C

R_0, theta_0 = R_odot, 220 # [kpc], [km/s]
R_i = [0.0, 0.09 * R_0, 0.45 * R_0, 1.6 * R_0, 50.0]
A_i = [0.0, 3069.81, -15809.8, +43980.1, -68287.3, +54904., -17731.] # R/R_0 < 0.09
B_i = [+325.0912, -248.1467, +231.87099, -110.73531, +25.073006, -2.110625]
C_i = [-2342.6564, +2507.60391, -1024.068760, +224.562732, -28.4080026, +2.0697271, -0.08050808, +0.00129348]
D_i = [234.88]
dr = 0.1
A_R = np.arange(R_i[0], R_i[1] + dr, dr)
B_R = np.arange(R_i[1], R_i[2] + dr, dr)
C_R = np.arange(R_i[2], R_i[3] + dr, dr)
D_R = np.arange(R_i[3], R_i[4] + dr, dr)

theta_A, theta_B, theta_C, theta_D = np.zeros(len(A_R)), np.zeros(len(B_R)), np.zeros(len(C_R)), np.zeros(len(D_R))
for i in range(len(A_i)):
    theta_A += A_i[i] * A_R**i
for i in range(len(B_i)):
    theta_B += B_i[i] * B_R**i
for i in range(len(C_i)):
    theta_C += C_i[i] * C_R**i
for i in range(len(D_i)):
    theta_D += D_i[i]
    
# radius and rotational velocity to be used to calculate circular velocity v_circ(r) at any r:
r_rot = [*A_R, *B_R, *C_R, *D_R]
r_rot_unitless = [r / r_s for r in r_rot]
v_rot = [*theta_A, *theta_B, *theta_C, *theta_D]
v_rot_unitless = [v / (ct.c.to_value(u.km / u.s)) for v in v_rot]
# v_circ = interp1d(r_rot, v_rot, fill_value='extrapolate') # ~9x faster than putting it in a function
v_circ_unitless = interp1d(r_rot_unitless, v_rot_unitless, fill_value='extrapolate')


################################################################


################ radius & maximum line of sight distance ################


def s_max(b, l, R_halo):
    """
    returns: [kpc] maximum value of line-of-sight coordinate, as a function of angles b, l, and the halo scale R_halo
    ********
    b: [degrees] galactic latitude
    l: [degrees] galactic longitude
    R_halo: [kpc] MW halo radius - chosen by user
    """
    psi = np.arccos( np.cos(np.radians(b)) * np.cos(np.radians(l)) ) # [radians]
    return (np.sqrt(R_halo**2 - R_odot**2 * (np.sin(psi))**2) + R_odot * np.cos(psi)) # [kpc]


################################################################


################ density ################

density_unitless_interpolator = []
for i in range(len(r_unitless_bins)-1):
    density_unitless_interpolator.append(RegularGridInterpolator((bs,ls), data_unitless[i])) # ~2x faster than putting it in a function


def density_unitless(nucleus, r, b, l):
    """
    returns: [unitless] mass density as a function of radius from galactic centre
    **********
    particle: 'DM', 'He4', 'C12', 'N14', 'O16', 'CO', 'Fe57'
    r: [unitless] radius from galactic centre
    b: [degrees] galactic latitude
    l: [degrees] galactic longitude
    """
    r_unitless_index = bisect.bisect(r_unitless_bins, r) - 1 # find radial bin index
    # print(r_unitless_index)
    rho_DM = 1 / ( (r) * (1 + r)**2 ) # [unitless]
    # print(rho_DM* (rho_s * u.GeV * (1/u.cm)**3).to_value())
    if r >= r_unitless_bins[-1]: # if radius is beyond what is provided by GALPROP, return zero for baryon density
        rho_b = 0
        return(rho_b, rho_DM)

    # print(nuc_unitless_dict[nucleus]['mass_unitless'] / (nuc_unitless_dict['C12']['mass_unitless'] + nuc_unitless_dict['O16']['mass_unitless']))
    # print(density_unitless_interpolator[r_unitless_index](np.array([b, l]))[0] * (rho_s * u.GeV * (1/u.cm)**3).to_value(u.kg * (1/u.cm)**3, u.mass_energy()))
    rho_b = (nuc_unitless_dict[nucleus]['mass_unitless'] / (nuc_unitless_dict['C12']['mass_unitless'] + nuc_unitless_dict['O16']['mass_unitless']) \
             * density_unitless_interpolator[r_unitless_index](np.array([b, l])))[0]
    return (rho_b, rho_DM)


################################################################


################ velocity dispersions $\sigma(r, \psi)$ ################


def vDispersion(r):
    """
    returns: [unitless, in units of c] 3D velocity dispersion 
    **********
    r: [unitless, in units of r_s] radius from galactic centre 
    
    DM dispersion function taken from 2111.03076 Figure 1: r in units kpc, velocity dispersion in units km/s
    """
    disp_b = 10 / (ct.c.to_value(u.km / u.s)) # [unitless, in units of c]
    disp_DM = (-42.4 * np.log10(r * r_s) + 382.2) / (ct.c.to_value(u.km / u.s)) # [unitless, in units of c]
    
    return (disp_b, disp_DM)
        

################################################################


################ escape velocity ################

# note! the escape velocity is currently 3 * the usual escape velocity formula - probably will change
# - Compare $v_\text{esc}$ with Figures 2 and 3 of https://ui.adsabs.harvard.edu/abs/2021A%26A...649A.136K/abstract (try https://www.aanda.org/articles/aa/pdf/2018/08/aa33748-18.pdf or https://pure.rug.nl/ws/portalfiles/portal/196810757/aa38777_20.pdf)
# - also see https://www.aanda.org/articles/aa/pdf/2018/08/aa33748-18.pdf


def v_esc(r):
    """
    returns: [unitless, in units of r_s] escape velocity as function of [unitless] radius
    ********
    r: [unitless, in units of r_s] radius from galactic centre
    """
    c_bulge_unitless = c_bulge / r_s
    b_disk_unitless = b_disk / r_s
    M_baryon = ((M_bulge * r**2 * (c_bulge_unitless + r)**(-2)) \
                + M_disk * (1 - (1 + r/b_disk_unitless) * np.exp(-r/b_disk_unitless)))
    # print(M_baryon)
    M_DM = (4 * np.pi * (np.log(1 + r) - r/(1+r)) \
            * r_s**3 \
            * (rho_s * u.GeV * (1/u.cm)**3).to_value(u.kg * (1/u.kpc)**3, u.mass_energy()))
    # print(M_DM)
    M_MW = M_baryon + M_DM # [kg]

    return (3 * np.sqrt(2 * ct.G.to_value() * M_MW / ((r * r_s * u.kpc).to_value(u.m))) \
            * (u.m * 1/u.s).to_value(u.km * 1/u.s) / (ct.c.to_value(u.km / u.s)))


################################################################


################ DM velocity distribution integral ################

def norm_chi(sigma_chi, V_ESC):
    """
    returns: [unitless] normalization factor for DM distribution function, 
    checked with function Int_f_chi(s, b, l)
    **********
    sigma_chi: 
    V_ESC: 
    """
    norm = (2*np.pi * sigma_chi**2)**(-3/2) * (4 * np.pi) \
    * ( np.sqrt(np.pi / 2) * sigma_chi**3 * scipy.special.erf(V_ESC / (np.sqrt(2) * sigma_chi) ) \
       - sigma_chi**2 * V_ESC * np.exp(-V_ESC**2 / (2 * sigma_chi**2)))
    return norm


def Int_v_chi(m_chi, m_N, dE, r, sigma_chi, V_ESC, v_n):
    """
    returns: [unitless] integral of DM velocity distribution [MeV^-1 km s^-1]
    **********
    dE: [unitless] excitation energy
    m_chi: [unitless] DM mass 
    m_N: [unitless] nucleus mass
    v_n: [unitless] nucleus velocity
    r: [unitless] radius from galactic centre
    """
    if (dE - m_chi - (1/2) * m_N * v_n**2) < 0: # process is not possible
        return 0

    v_chi_plus = np.sqrt( (2 * (dE - m_chi - (1/2) * m_N * v_n**2) ) / (m_chi) ) # [unitless]
    if v_chi_plus > V_ESC:
        return 0
  
    E_chi = ( m_chi + (1/2) * m_chi * v_chi_plus**2 ) # [unitless]
    
    norm = norm_chi(sigma_chi, V_ESC) # [unitless]
    if norm<0:
        print("something has gone wrong!")
        return

    return ( (1 / norm) * (1 / E_chi) * v_chi_plus * np.exp(-v_chi_plus**2 / (2 * sigma_chi**2)) ) # [unitless]
    # to get [MeV^-1 km/s] multiply by: (ct.c.to_value(u.km / u.s)) / energyScale


################################################################


################ baryon velocity distribution integral ################

def v_N_integralBounds(m_chi, m_N, dE, V_ESC):
    """
    returns: [unitless] min and max bounds on v_N
    **********
    m_chi: [unitless]
    m_N: [unitless]
    dE: [unitless]
    V_ESC: [unitless]
    """
    v_N_min = np.sqrt(2 * (dE - m_chi - (1/2) * m_chi * V_ESC**2) / m_N) # [unitless]
    v_N_max = np.sqrt(2 * (dE - m_chi) / m_N) # [unitless]
    
    if v_N_min > V_ESC:
        v_N_min = V_ESC
    if v_N_max > V_ESC:
        v_N_max = V_ESC
    if v_N_min > v_N_max:
        print('v_N bounds make no sense!')
        
    return (v_N_min, v_N_max) # ([unitless], [unitless])


def norm_N(sigma_b, VBAR, V_ESC):
    """
    returns: [unitless] normalization factor for baryon distribution function, checked with function Int_f_b(s, b, l)
    """
    norm = (4 * np.pi) * (2*np.pi * sigma_b**2)**(-3/2) \
    * (1/2) * sigma_b * (np.sqrt(2 * np.pi) * (VBAR**2 + sigma_b**2) * scipy.special.erf(VBAR / (np.sqrt(2) * sigma_b)) \
                        - np.sqrt(2 * np.pi)  * (VBAR**2 + sigma_b**2) * scipy.special.erf((VBAR - V_ESC) / (np.sqrt(2) * sigma_b)) \
                        + 2 * sigma_b * VBAR * np.exp(- VBAR**2 / (2 * sigma_b**2)) \
                        - 2 * sigma_b * (VBAR + V_ESC) * np.exp(- (VBAR - V_ESC)**2 / (2 * sigma_b**2)) )
    return norm


def Int_v_N(m_chi, m_N, dE, r, sigma_b, sigma_chi):
    """
    returns: [MeV^-2 km^4 s^-4] integral of baryon velocity distribution
    **********
    dE: [unitless, in units of energyScale] excitation energy
    m_chi: [unitless, in units of energyScale] DM mass
    m_N: [unitless, in units of energyScale] nucleus mass
    nucleus: scattering target
    r: [unitless, in units of r_s] radius from galactic centre
    """
    VBAR = v_circ_unitless(r) # [unitless]
    V_ESC = v_esc(r) # [unitless]
    # if VBAR>V_ESC:
    #     return 0
    if (dE - m_chi - (1/2) * m_chi * V_ESC**2) < 0: # v_N_min expression
        return 0

    norm = norm_N(sigma_b, VBAR, V_ESC) # [unitless]
    if norm<0:
        print("something has gone wrong!")
        return
        
    v_N_bounds = v_N_integralBounds(m_chi, m_N, dE, V_ESC) # ([unitless], [unitless])
    v_N_min, v_N_max = v_N_bounds[0], v_N_bounds[1] # [unitless], [unitless]
    
    def f(v_n): # v_n units [unitless]
        E_N = ( m_N + (1/2) * m_N * v_n**2 ) # [MeV]
        integrand = ((2 * np.pi) * (1 / E_N) * sigma_b**2 * (v_n / VBAR) \
        * (np.exp(- (v_n - VBAR)**2 / (2 * sigma_b**2) ) - np.exp(- (v_n + VBAR)**2 / (2 * sigma_b**2) )) \
        * Int_v_chi(m_chi, m_N, dE, r, sigma_chi, V_ESC, v_n))
        return integrand # [km^4/s^4]
        
    integral, err = quad(f, v_N_min, v_N_max) # integration gives factor of [unitless]
    return (1/norm) * integral # [unitless]
    # to get [MeV^-2 km^4 s^-4] multiply by: energyScale**(-2) * ct.c.to_value(u.km/u.s)**4


################################################################


################ line of sight integral ################

def Int_LOS(m_chi, nucleus, m_N, dE, b, l, R_max):
    """
    returns: [cm^-5] line of sight integral
    ********
    m_chi:
    nucleus:
    dE:
    b:
    l:
    R_max: [unitless, in units of r_s]
    """
    psi = np.arccos( np.cos(np.radians(b)) * np.cos(np.radians(l)) ) # [radians]
    s_max = (np.sqrt(R_max * R_max - (R_odot/r_s) * (R_odot/r_s) * np.sin(psi) * np.sin(psi)) + (R_odot/r_s) * np.cos(psi)) # [unitless, in units of r_s]
        
    def f(s):
        r = np.sqrt((R_odot/r_s) * (R_odot/r_s) + s**2 - 2 * (R_odot/r_s) * s * np.cos(psi)) # [unitless, in units of r_s]

        disp = vDispersion(r) # ([unitless], [unitless])
        sigma_b, sigma_chi = disp[0], disp[1] # [unitless], [unitless]
        
        v_N_integral = Int_v_N(m_chi, m_N, dE, r, sigma_b, sigma_chi)
        if v_N_integral == 0:
            return 0
        
        rho = density_unitless(nucleus, r, b, l) # ([unitless, in units of rho_s], [unitless, in units of rho_s])
        rho_b, rho_chi = rho[0], rho[1] # [unitless, in units of rho_s], [unitless, in units of rho_s]
        integrand = rho_b * rho_chi * ((2*np.pi) * sigma_b * sigma_chi)**(-3) * v_N_integral
        return integrand

    # integral, err = quad(f, 0, s_max)#, epsrel = 1e-5, epsabs = 0, limit=200)
    # return integral
    esses = np.linspace(0, s_max, 2000)
    g = list(map(lambda ess: f(ess), esses))
    integral = cumtrapz(g, esses, initial=0)
    return integral[-1] # unitless
    #* (rho_s * rho_s * ct.c.to_value(u.km/u.s)**(-6)) * energyScale**(-2) * ct.c.to_value(u.km/u.s)**4 \
    #* ct.c.to_value(u.km/u.s)**2 * 1000**2 * (1 * u.kpc).to_value(u.cm)
    

losDimensionfulIntegralFactor = (rho_s * rho_s * ct.c.to_value(u.km/u.s)**(-6)) * energyScale**(-2) * ct.c.to_value(u.km/u.s)**4 * ct.c.to_value(u.km/u.s)**2 * 1000**2 * (1 * u.kpc).to_value(u.cm)
# losDimensionfulIntegralFactor

################################################################

################ flux ################

fluxDimensionfulFactor = ( 1/energyScale * 1/energyScale \
              * losDimensionfulIntegralFactor \
              * 1/energyScale \
              * (ct.c.to_value(u.cm/u.s)) * (ct.hbar.to_value(u.MeV * u.s) * ct.c.to_value(u.cm/u.s))**2 )
# fluxDimensionfulFactor


def flux_w_convolution(g_chi, m_chi, nucleus, m_N, J_N, b, l, R_max, epsilon, nucl_exc, GT_sum, E):
    """
    returns: differential flux in ?[cm^-2 s^-1 (sr^-1) MeV^-1]?
    **********
    g_chi: [unitless, in units of 1/energyScale]
    m_chi:
    nucleus:
    b:
    l:
    R_max: [unitless, in units of r_s] maximum radius (50.0 kpc)
    epsilon:
    dEs: [unitless, in units of energyScale]
    GTs: [unitless]
    E: [unitless, in units of energyScale] observed photon energy
    """
    fluxes, flux_tot = [], 0
    GT_sum_new = 0
    
    for dE, GT in nucl_exc:
        # do some checks to save time:
        if m_chi > dE:
            flux_tot += 0
            # GT_sum -= GT
            continue
        
        expTerm = np.exp(-(E - dE)**2 / (2 * epsilon**2 * dE**2))
        if expTerm == 0:
            flux_tot += 0
            # GT_sum -= GT
            continue
        
        los_integral = Int_LOS(m_chi, nucleus, m_N, dE, b, l, R_max) # [unitless] 
        # multiply los_integral by losDimensionfulIntegralFactor to get [cm^-5]
        if los_integral == 0:
            flux_tot += 0
            # GT_sum -= GT
            continue

        R = (np.sqrt(2 * np.pi) * epsilon * dE)**(-1) * expTerm # [unitless], needs to be multiplied by 1/energyScale to get [MeV^-1]
        GT_sum_new += GT # do it this way, rather than using GT_sum
        dNdgamma_unnormalized = GT # branching ratio, normalized to be [unitless] at the end
        flux = ((1/24 * g_chi * g_chi * g_A * g_A / (2*J_N + 1)) * (m_N + dE)/m_N * los_integral * dNdgamma_unnormalized * R)
        # unitless, needs to be multiplied by fluxDimensionfulFactor to get cm^-2 s^-1 MeV^-1 sr^-1:
        # energyScale**-2 \ for g_chi
        # losDimensionfulIntegralFactor
        # * 1/energyScale for R
        # * (ct.c.to_value(u.cm/u.s)) * (ct.hbar.to_value(u.MeV * u.s) * ct.c.to_value(u.cm/u.s))**2) to get flux units
        fluxes.append(flux)
        flux_tot += flux

    if flux_tot == 0:
        return flux_tot
    flux_tot_normalized = flux_tot/GT_sum_new
    return flux_tot_normalized


def bIntegral(g_chi, m_chi, nucleus, b_min, b_max, l, R_max, epsilon, E):
    """
    g_chi: [MeV^-1]
    m_chi: [MeV]
    nucleus: 'C12' or 'O16'
    l: [deg]
    R_max: [kpc]
    epsilon: [unitless]
    E: [MeV]
    b_min: [deg]
    b_max: [deg]
    """
    m_N, J_N = nuc_unitless_dict[nucleus]['mass_unitless'], nuc_unitless_dict[nucleus]['spin']
    dEs = dEs_dict[nucleus] / energyScale
    GTs = GTs_dict[nucleus]
    nucl_exc = list(zip(dEs, GTs))
    GT_sum = np.sum(GTs)
    diffFlux = []

    def f(b):
        print("b = {}".format(b))
        x = flux_w_convolution(g_chi * energyScale, m_chi / energyScale, nucleus, m_N, J_N, b, l, R_max / r_s, epsilon, nucl_exc, GT_sum, E / energyScale)
        print("b Integral flux [unitless]: {}\n".format(x))
        diffFlux.append(x * fluxDimensionfulFactor)
        integrand = np.sin(b * np.pi/180 + np.pi/2) * x
        return integrand 
    bees = np.linspace(b_min, b_max, 20) # 20
    g = list(map(lambda bee: f(bee), bees))
    integral = cumtrapz(g, bees, initial=0)

    #### write to file ####
    filename = 'fluxData/{}/mchi_{}MeV_l_{}deg_epsilon_{}.txt'.format(nucleus, m_chi, l, 100*epsilon)
    metadata = "# R_odot [kpc] = {}, \n\
# R_max [kpc] = {} kpc\n\
# rho_s [GeV cm^-3] = {}\
# r_s [kpc] = {}\n\
# M_bulge [solar masses] = {}\n\
# c_bulge [kpc] = {}\n\
# M_disk [solar masses] = {}\n\
# b_disk [kpc] = {}\n\
# g_A = {}\n\n".format(R_odot, R_max, rho_s, r_s, M_bulge, c_bulge, M_disk, b_disk, g_A)
    with open(filename, 'w') as fp:
        fp.write(metadata)
    dat = {'b [deg]': bees, 'Differential flux [cm^-2 s^-1 MeV^-1 sr^-1]': diffFlux}
    df = pd.DataFrame(dat)
    df.to_csv(filename, sep='\t', index=False, mode='a')
    ################
    
    # integral1, err1 = quad(f, -5, 5, limit=10)
    return integral[-1] * np.pi/180


def lIntegral(g_chi, m_chi, nucleus, b_min, b_max, l_min1, l_max1, l_min2, l_max2, R_max, epsilon, E):
    bInt = []
    def f(l):
        print("l = {}".format(l))
        integrand = bIntegral(g_chi, m_chi, nucleus, b_min, b_max, l, R_max, epsilon, E)
        bInt.append(integrand * fluxDimensionfulFactor)
        return integrand

    ells1 = np.linspace(l_min1, l_max1, 60) # 60
    g1 = list(map(lambda ell1: f(ell1), ells1))
    integral1 = cumtrapz(g1, ells1, initial=0)

    ells2 = np.linspace(l_min2, l_max2, 60) # 60
    g2 = list(map(lambda ell2: f(ell2), ells2))
    integral2 = cumtrapz(g2, ells2, initial=0)
    # integral = integral1[-1] + integral2[-1]

    #### write to file ####
    filename = 'bIntegralData/{}/mchi_{}MeV_epsilon_{}.txt'.format(nucleus, m_chi, 100*epsilon)
    metadata = "# b range [deg] = [{}, {}], \n\
# R_odot [kpc] = {}, \n\
# R_max [kpc] = {} kpc\n\
# rho_s [GeV cm^-3] = {}\
# r_s [kpc] = {}\n\
# M_bulge [solar masses] = {}\n\
# c_bulge [kpc] = {}\n\
# M_disk [solar masses] = {}\n\
# b_disk [kpc] = {}\n\
# g_A = {}\n\n".format(b_min, b_max, R_odot, R_max, rho_s, r_s, M_bulge, c_bulge, M_disk, b_disk, g_A)
    with open(filename, 'w') as fp:
        fp.write(metadata)
    dat = {'l [deg]': [*ells1, *ells2], 'Differential flux [cm^-2 s^-1 MeV^-1 rad^-1]': bInt}
    df = pd.DataFrame(dat)
    df.to_csv(filename, sep='\t', index=False, mode='a')
    ################
    
    return (integral1[-1] * np.pi/180, integral2[-1] * np.pi/180)


def integrateSolidAngleFlux(g_chi, m_chi, nucleus, R_max, epsilon):
    l_min1, l_max1 = 330.25, ls[-1]
    l_min2, l_max2 = ls[0], 29.75
    b_min, b_max = -4.75, 4.75
    energy = np.linspace((m_chi - 1 * epsilon * m_chi), (m_chi + 1 * epsilon * m_chi), 21)
    integratedFlux1, integratedFlux2 = [], []
    integratedFlux = []
    
    for E in energy:
        print("Photon energy = {}".format(E))
        intFlux1, intFlux2 = lIntegral(g_chi, m_chi, nucleus, b_min, b_max, l_min1, l_max1, l_min2, l_max2, R_max, epsilon, E)
        integratedFlux1.append(intFlux1 * fluxDimensionfulFactor)
        integratedFlux2.append(intFlux2 * fluxDimensionfulFactor)
        integratedFlux.append((intFlux1 + intFlux2) * fluxDimensionfulFactor)

    def f(b, l):
        integrand = np.sin(b * np.pi/180 + np.pi/2)
        return integrand
    integrateSolidAngle1 = dblquad(f, l_min1, l_max1, b_min, b_max)[0] * np.pi/180 * np.pi/180 # [sr]
    integrateSolidAngle2 = dblquad(f, l_min2, l_max2, b_min, b_max)[0] * np.pi/180 * np.pi/180 # [sr]
    integrateSolidAngle = integrateSolidAngle1 + integrateSolidAngle2
    averagedFlux = [m / integrateSolidAngle for m in integratedFlux]
    
    #### write to file ####
    filename = 'solidAngleIntegralData/{}/mchi_{}MeV_epsilon_{}.txt'.format(nucleus, m_chi, 100*epsilon)
    metadata = "# Integral 1 l bounds [deg] = [{}, {}], \n\
# Integral 2 l bounds [deg] = [{}, {}], \n\
# b range [deg] = [{}, {}], \n\
# R_odot [kpc] = {}, \n\
# R_max [kpc] = {} kpc\n\
# rho_s [GeV cm^-3] = {}\
# r_s [kpc] = {}\n\
# M_bulge [solar masses] = {}\n\
# c_bulge [kpc] = {}\n\
# M_disk [solar masses] = {}\n\
# b_disk [kpc] = {}\n\
# g_A = {}\n\n".format(l_min1, l_max1, l_min2, l_max2, b_min, b_max, R_odot, R_max, rho_s, r_s, M_bulge, c_bulge, M_disk, b_disk, g_A)
    with open(filename, 'w') as fp:
        fp.write(metadata)
    dat = {'Photon energy [MeV]': energy, 'Solid-angle-averaged integrated flux [cm^-2 s^-1 MeV^-1 sr^-1]': averagedFlux, 'Integrated flux [cm^-2 s^-1 MeV^-1]': integratedFlux, \
           'Integral 1 [cm^-2 s^-1 MeV^-1]': integratedFlux1, 'Integral 2 [cm^-2 s^-1 MeV^-1]': integratedFlux2}
    df = pd.DataFrame(dat)
    df.to_csv(filename, sep='\t', index=False, mode='a') 
    ################


def IntSolidAngleFluxAllCOMasses(g_chi, R_max, epsilon):
    # for i in range(len(m_chis_C12)):
    #     integrateSolidAngleFlux(i, g_chi, m_chis_C12[i], 'C12', R_max, epsilon)
    for m in m_chis_C12:
        integrateSolidAngleFlux(g_chi, m, 'C12', R_max, epsilon)
        # IntSolidAngleFlux(g_chi, m, 'O16', R_max, epsilon)
        

#bIntegral(1, m_chis_C12[3], 'C12', -4.75, 4.75, l_crval, 50.0, 0.1, dEs_dict['C12'][3])
#lIntegral(1, m_chis_C12[3], 'C12', -4.75, 4.75, 330.25, ls[-1], ls[0], 29.75, 50.0, 0.1, dEs_dict['C12'][3])
#integrateSolidAngleFlux(1, m_chis_C12[0], 'C12', 2*R_odot, 0.1) # energy = [dEs_dict['C12'][0]]
#IntSolidAngleFluxAllCOMasses(1, 2*R_odot, 0.10)


def fluxData(g_chi, m_chi, nucleus, b, l, R_max, epsilon):
    """
    Calculates flux and writes to .txt file.
    ****************
    g_chi: [unitless]
    m_chi: [unitless, in units of energyScale]
    nucleus: 'C12' or 'O16'
    b: [deg]
    l: [deg]s
    R_max: [unitless, in units of r_s]
    epsilon: [unitless] % of convolution as decimal from 0 to 1
    """
    m_N, J_N = nuc_unitless_dict[nucleus]['mass_unitless'], nuc_unitless_dict[nucleus]['spin']
    dEs = dEs_dict[nucleus] / energyScale
    GTs = GTs_dict[nucleus]
    nucl_exc = list(zip(dEs, GTs))
    GT_sum = np.sum(GTs)
    flux_tot = []
    energy = np.linspace((m_chi - 1 * epsilon * m_chi), (m_chi + 1 * epsilon * m_chi), 21)
    flux_tot = [(flux_w_convolution(g_chi * energyScale, m_chi / energyScale, \
                                   nucleus, m_N, J_N, b, l, R_max / r_s, epsilon, \
                                   nucl_exc, GT_sum, E / energyScale) * fluxDimensionfulFactor) for E in energy]
    # for E in energy:
    #     # print(E)
    #     flux_tot.append(flux_w_convolution(g_chi * energyScale, m_chi / energyScale, \
    #                                nucleus, m_N, J_N, b, l, R_max / r_s, epsilon, \
    #                                nucl_exc, GT_sum, E / energyScale) * fluxDimensionfulFactor)

    filename = 'fluxData/{}/mchi_{}MeV_l_{}deg_b_{}deg_epsilon_{}.txt'.format(nucleus, m_chi, l, b, 100*epsilon)
    metadata = "# R_odot [kpc] = {}, \n\
# R_max [kpc] = {} kpc\n\
# rho_s [GeV cm^-3] = {}\
# r_s [kpc] = {}\n\
# M_bulge [solar masses] = {}\n\
# c_bulge [kpc] = {}\n\
# M_disk [solar masses] = {}\n\
# b_disk [kpc] = {}\n\
# g_A = {}\n\n".format(R_odot, R_max, rho_s, r_s, M_bulge, c_bulge, M_disk, b_disk, g_A)

    with open(filename, 'w') as fp:
        fp.write(metadata)
    dat = {'Photon energy [MeV]': energy, '{} flux [cm^-2 s^-1 sr^-1 MeV^-1]'.format(nucleus): flux_tot}
    df = pd.DataFrame(dat)
    df.to_csv(filename, sep='\t', index=False, mode='a')
    return flux_tot


################ finding optimum DM mass values ################

def negInt_v_N(m_chi, m_N, dE, r, sigma_b, sigma_chi):
    # disp = vDispersion(r)
    # sigma_b, sigma_chi = disp[0], disp[1]
    return np.negative(Int_v_N(m_chi, m_N, dE, r, sigma_b, sigma_chi))


def minimizeInt_v_N(nucleus, dE, r):
    # m_chi_initialGuess = 16#np.arange(deltaE - 1/10, deltaE, 1e-2)
    # print(m_chi_initialGuess)
    disp = vDispersion(r)
    sigma_b, sigma_chi = disp[0], disp[1]
    res = minimize_scalar(negInt_v_N, bounds=(dE-1/10, dE), \
                   args=(nuc_dict[nucleus][0], dE, r, sigma_b, sigma_chi))
    return(res.x)

# np.negative(Int_v_N(m_chi = 16.73, m_N = nuc_dict['O16'][0], deltaE = 16.732241, r = 10))
# minimizeInt_v_N('O16', 16.732240999999995, 10)


def optimal_m_chis(r):
    """
    creates .txt files of optimal m_chis for all C12, O16 excitation energies as function of radius
    files have already been generated, no need to run function...
    UNLESS any of the main functions are changed
    - was rerun after normalization factor fixed
    """
    for i in ['C12', 'O16']:
        m_chis = []
        all_dEs = []
        for dE in dEs_dict[i]:
            all_dEs.append(dE)
            m_chis.append(minimizeInt_v_N(i, dE, r))
        dat = {'Nuclear excitation energy [MeV]': all_dEs, 'Optimal DM mass [MeV]': m_chis}
        df = pd.DataFrame(dat)
        df.to_csv('optimal_m_chis_unitlessCalculation/{}/{}_optimal_m_chis_r_{}kpc.txt'.format(i, i, r), sep='\t', index=False)
    return

# arrs = [0.01, 0.1, 1, 2, 4, 6, 8, 10, 20, 30, 40, 50]
# for r in arrs:
#     optimal_m_chis(r)


def optimal_m_chis_array(arrs):
    """
    creates .txt files of optimal m_chis for all C12, O16 excitation energies for each excitation energy 
    files have already been generated, no need to run function...
    UNLESS any of the main functions are changed
    or you want to include additional radii
    - was rerun after normalization factor fixed
    """
    for nuc in ['C12', 'O16']:
        m_chis = [[] for _ in range(len(dEs_dict[nuc]))]
        for i in range(len(dEs_dict[nuc])):
            for j in range(len(arrs)):
                mass_by_r_data = 'optimal_m_chis_by_radius/{}/{}_optimal_m_chis_r_{}kpc.txt'.format(nuc, nuc, arrs[j])
                mass_by_r_df = pd.read_csv(mass_by_r_data, sep='\t', names=['Nuclear excitation energy [MeV]',	'Optimal DM mass [MeV]'], skiprows=1)
                m_chis[i].append(mass_by_r_df['Optimal DM mass [MeV]'][i])
            mass_by_dE_dat = {'Radius from GC [kpc]': arrs, 'Optimal DM mass [MeV]': m_chis[i]}
            mass_by_dE_df = pd.DataFrame(mass_by_dE_dat)
            mass_by_dE_df.to_csv('optimal_m_chis_by_dE/{}/{}_optimal_m_chis_dE_{}_MeV.txt'.format(nuc, nuc, dEs_dict[nuc][i]), sep='\t', index=False)

# optimal_m_chis_array([0.01, 0.1, 1, 2, 4, 6, 8, 10, 20, 30, 40, 50])


################################################################


if __name__=="__main__":
    print("hello!")
    for m in m_chis_C12:
        print("m_chi = {}\n".format(m))
        integrateSolidAngleFlux(1, m, 'C12', 50.0, 0.05)

