#!/usr/bin/env python
# coding: utf-8

# last update: APRIL 5 2024:
## updated CO denstiy to new .fits file: 'massdensity_CO_ACTUALLYcorrectedlol.fits'
## converted new CO density in units of [kg/m^3] to [kg/cm^3] to be consistent with the rest of the code
## removed unused constants: M_bulge, c_bulge, M_disk, b_disk
## removed plotting functions
## removed importing of unused modules: sys, math, matplotlib, matplotlib.pyplot, astropy.wcs.WCS, scipy.integrate.dblquad, scipy.optimize.minimize, scipy.optimize.minimize_scalar, time
## removed random commented out print statements

import os
import numpy as np
import scipy.integrate
from scipy.integrate import quad
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d
from scipy.interpolate import RegularGridInterpolator
import pandas as pd
from astropy.io import fits
from astropy import constants as ct
from astropy import units as u
import bisect
from pathlib import Path
from multiprocessing import Pool
import click


# ################ constants & data import ################

# constants
g_A = 1.27 # [unitless] axial form factor
################
R_odot = 8.5 # [kpc] Sun's distance from centre of MW, from Clemens 1985
rho_s = 0.51 # [GeV/cm^3] DM radial scale for MW NFW profile; https://arxiv.org/abs/1906.08419
r_s = 8.1 # [kpc] DM radial scale for MW NFW profile; https://arxiv.org/abs/1906.08419; also used as distance scale for creating unitless quantities
################
energyScale = 10 # [MeV] energy scale for creating unitless quantities

# importing C12 and O16 data: excitation energies [MeV] and GT strengths [unitless]
C12_data = 'C12_dEs_GTs.txt'
O16_data = 'O16_dEs_GTs.txt'
C12df = pd.read_csv(C12_data, sep='\t', names=['dE [MeV]', 'GT'], skiprows=1, float_precision='round_trip')
O16df = pd.read_csv(O16_data, sep='\t', names=['dE [MeV]', 'GT'], skiprows=1, float_precision='round_trip')

# nuclear info dictionary
nuc_dict = {'C12': {'mass [MeV]': 11274.78, 'mass [unitless]': 11274.78/energyScale, 'spin': 0.0, 'dEs [MeV]': C12df['dE [MeV]'], 'GTs': C12df['GT']}, \
            'O16': {'mass [MeV]': 15033.04, 'mass [unitless]': 15033.04/energyScale, 'spin': 0.0, 'dEs [MeV]': O16df['dE [MeV]'], 'GTs': O16df['GT']}}
# 'He4': {'mass_MeV': 3758.26, 'mass [unitless]': 3758.26/energyScale, 'spin': 0.0}
# 'N14': {'mass_MeV': 13153.91, 'mass [unitless]': 13153.91/energyScale, 'spin': 0.0}


# ################ GALPROP density data ################

# galprop manual: 
# - https://galprop.stanford.edu/download/manuals/galprop_v54.pdf

# handling .fits files: 
# - https://fits.gsfc.nasa.gov/users_guide/usersguide.pdf
# - https://astropy4cambridge.readthedocs.io/en/latest/_static/Astropy%20-%20Handling%20FITS%20files.html

hdul_CO = fits.open('massdensity_CO_ACTUALLYcorrectedlol.fits')


def print_density_header():
    """
    prints header for CO density data
    """
    return hdul_CO[0].header # .fits file header


def print_radial_bins():
    """
    prints radial bins for CO density data
    """
    return hdul_CO[1].data # radial bins


# density data
"""
density data: hdul_CO[0].data[r bin index, b index, l index]
    r bin index: 0 through 8 (axis 3)
    b index: 0 through 359 (axis 2)
    l index: 0 through 719 (axis 1)
    example: hdul_CO[0].data[8, 359, 719]
    gives the density in kg/cm^3 for a given (r, b, l)
    note GALPROP specifies Rsun = 8.5 kpc, Vsun = 220 km/s 
"""
density_data = hdul_CO[0].data * (1/100)**3 # [kg/m^3] >> [kg/cm^3]
density_data_unitless = density_data / ((rho_s * u.GeV * (1/u.cm)**3).to_value(u.kg * (1/u.cm)**3, u.mass_energy())) # [unitless, in units of rho_s]
# to convert x from unitless density to units of [GeV/cm^3]: x * (rho_s * u.GeV * (1/u.cm)**3).to_value()
# to convert x from unitless density to units of [kg/cm^3]: x * (rho_s * u.GeV * (1/u.cm)**3).to_value(u.kg * (1/u.cm)**3, u.mass_energy())

bins = hdul_CO[1].data # [kpc] radial bin boundaries
r_bins = [bins[0][0]] # [kpc] r bin smallest value
r_unitless_bins = [bins[0][0] / r_s] # [unitless] r bin smallest value
for i, j in bins: # creating r bins from GALPROP increments
    r_bins.append(j)
    r_unitless_bins.append(j / r_s)


# creating b and l arrays
b_len, l_len = hdul_CO[0].header['NAXIS2'], hdul_CO[0].header['NAXIS1'] # length of b, l arrays
b_crval, l_crval = hdul_CO[0].header['CRVAL2'], hdul_CO[0].header['CRVAL1'] # [deg], [deg]; values of b,l at reference point CRPIX2, CRPIX1
b_delta, l_delta = hdul_CO[0].header['CDELT2'], hdul_CO[0].header['CDELT1'] # [deg], [deg]; increments for b, l

bs = list(np.arange(b_crval, b_crval + b_len*b_delta, b_delta)) # [deg]; list of bs
ls = list(np.arange(l_crval, l_crval + l_len*l_delta, l_delta)) # [deg]; list of ls


# ################ rotation curve from Clemens (1985) ################

# https://articles.adsabs.harvard.edu/pdf/1985ApJ...295..422C
R_0, theta_0 = R_odot, 220 # [kpc], [km/s]
R_i = [0.0, 0.09 * R_0, 0.45 * R_0, 1.6 * R_0, 50.0] # [kpc]; bins for composite curve; see Eq. (5)
# coefficients of rotation curves:
A_i = [0.0, 3069.81, -15809.8, +43980.1, -68287.3, +54904., -17731.] # R/R_0 < 0.09
B_i = [+325.0912, -248.1467, +231.87099, -110.73531, +25.073006, -2.110625] # R/R_0 = 0.09-0.45
C_i = [-2342.6564, +2507.60391, -1024.068760, +224.562732, -28.4080026, +2.0697271, -0.08050808, +0.00129348] # R/R_0 = 0.45-1.6
D_i = [234.88] # R/R_0 > 1.60

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
    
# radius and rotational velocity, to be used to calculate circular velocity v_circ(r) at any r:
r_rot = [*A_R, *B_R, *C_R, *D_R] # [kpc]
r_rot_unitless = [r / r_s for r in r_rot] # [unitless, in units of r_s]
v_rot = [*theta_A, *theta_B, *theta_C, *theta_D] # [km/s]
v_rot_unitless = [v / (ct.c.to_value(u.km / u.s)) for v in v_rot] # [unitless, in units of c]

# circular velocity interpolating function (~9x faster than putting it in a function)
# v_circ = interp1d(r_rot, v_rot, fill_value='extrapolate') # [km/s]
v_circ_unitless = interp1d(r_rot_unitless, v_rot_unitless, fill_value='extrapolate') # [unitless, in units of c]


# ################ density calculation & interpolation for DM and baryons ################

# to convert x from kg/cm^3 to GeV/cm^3: x * (u.kg * (1/u.cm)**3).to_value(u.GeV * (1/u.cm)**3, u.mass_energy())

# dimensionful density interpolator [kg/cm^3], for comparison (~2x faster than putting it in a function)
density_interpolator = []
for i in range(len(r_bins)-1):
    density_interpolator.append(RegularGridInterpolator((bs,ls), density_data[i]))


# unitless density interpolator, in units of rho_s (~2x faster than putting it in a function)
density_unitless_interpolator = []
for i in range(len(r_unitless_bins)-1):
    density_unitless_interpolator.append(RegularGridInterpolator((bs,ls), density_data_unitless[i]))


def density(nucleus, r, b, l):
    """
    returns: [GeV/cm^3] (nucleus, DM) mass density as a function of radius from galactic centre
    **********
    nucleus: 'C12' or 'O16'
    r_kpc: [kpc] radius from galactic centre
    b: [degrees] galactic latitude
    l: [degrees] galactic longitude
    """
    r_index = bisect.bisect(r_bins, r) - 1 # find radial bin index
    rho_DM = rho_s / ( (r/r_s) * (1 + r/r_s)**2 ) # [GeV/cm^3]; NFW profile
    if r >= r_bins[-1]: # if radius is beyond what is provided by GALPROP, return zero for baryon density
        rho_b = 0
        return(rho_b, rho_DM) # ([GeV/cm^3], [GeV/cm^3])

    rho_b = ( nuc_dict[nucleus]['mass [MeV]'] / (nuc_dict['C12']['mass [MeV]'] + nuc_dict['O16']['mass [MeV]']) \
             * density_interpolator[r_index](np.array([b, l]))[0] * (u.kg * (1/u.cm)**3).to_value(u.GeV * (1/u.cm)**3, u.mass_energy()) ) # [GeV/cm^3]
    return (rho_b, rho_DM) # ([GeV/cm^3], [GeV/cm^3])


def density_unitless(nucleus, r, b, l):
    """
    returns: [unitless, in units of rho_s], [unitless, in units of rho_s] (nucleus, DM) mass density as a function of radius from galactic centre
    **********
    nucleus: 'C12' or 'O16'
    r: [unitless, in units of r_s] radius from galactic centre
    b: [degrees] galactic latitude
    l: [degrees] galactic longitude
    """
    r_unitless_index = bisect.bisect(r_unitless_bins, r) - 1 # find unitless radial bin index
    # print(r_unitless_index)
    rho_DM = 1 / ( (r) * (1 + r)**2 ) # [unitless]; NFW profile
    # print(rho_DM * (rho_s * u.GeV * (1/u.cm)**3).to_value())
    if r >= r_unitless_bins[-1]: # if radius is beyond what is provided by GALPROP, return zero for baryon density
        rho_b = 0
        return(rho_b, rho_DM)

    rho_b = (nuc_dict[nucleus]['mass [unitless]'] / (nuc_dict['C12']['mass [unitless]'] + nuc_dict['O16']['mass [unitless]']) \
             * density_unitless_interpolator[r_unitless_index](np.array([b, l])))[0]
    return (rho_b, rho_DM) # ([unitless, in units of rho_s], [unitless, in units of rho_s])
        

# ################ velocity dispersions ################

def velocity_dispersions_unitless(r):
    """
    returns: ([unitless, in units of c], [unitless, in units of c]) (baryon velocity dispersion, DM velocity dispersion)
    ****************
    r: [unitless, in units of r_s] radius from galactic centre 
    ****************
    baryon dispersion function taken from 1707.00743 Figure 11 top right (orange circles) using automeris.io (data points selected manually)
    DM dispersion function taken from 2111.03076 Figure 1 upper right for r <~ 50 kpc using automeris.io (Distance 76, Delta X = 4 Px, Delta Y = 4 Px)
    """
    disp_DM = (1/np.sqrt(3)) * (-43 * np.log(r) + 288) / (ct.c.to_value(u.km / u.s)) # [unitless, in units of c]
    if r <= 1:
        disp_b = (-2.0 * np.log(r) + 3.3) / (ct.c.to_value(u.km / u.s)) # [unitless, in units of c]
    elif r > 1:
        disp_b = 3.3 / (ct.c.to_value(u.km / u.s)) # [unitless, in units of c]

    return (disp_b, disp_DM) # ([unitless, in units of c], [unitless, in units of c])
        

# ################ DM velocity integral ################

def norm_chi(sigma_chi, V_ESC):
    """
    returns: [unitless] normalization factor for DM distribution function;
    !!! CHECK AGAIN with function Int_f_chi(s, b, l) once unitless
    **********
    sigma_chi: [unitless, in units of c] DM velocity dispersion
    V_ESC: [unitless, in units of c] escape velocity
    """
    norm = (2 * np.pi * sigma_chi**2)**(-3/2) * (4 * np.pi) \
    * ( np.sqrt(np.pi / 2) * sigma_chi**3 * scipy.special.erf(V_ESC / (np.sqrt(2) * sigma_chi) ) \
       - sigma_chi**2 * V_ESC * np.exp(- V_ESC**2 / (2 * sigma_chi**2)))
    return norm # [unitless]


def v_chi_integral(m_chi, m_n, dE, r, sigma_chi, V_ESC, v_n):
    """
    returns: [unitless] integral of DM velocity distribution;
    to get unitful dimensions [MeV^-1 km/s] multiply by: (ct.c.to_value(u.km / u.s))/energyScale
    **********
    m_chi: [unitless, in units of energyScale] DM mass
    m_n: [unitless, in units of energyScale] nucleus mass
    dE: [unitless, in units of energyScale] nuclear excitation energy
    r: [unitless, in units of r_s] radius from galactic centre
    sigma_chi: [unitless, in units of c] DM velocity dispersion
    V_ESC: [unitless, in units of c] escape velocity
    v_n: [unitless, in units of c] nucleus velocity
    """
    if (dE - m_chi - (1/2) * m_n * v_n**2) < 0: # process is not possible
        return 0

    # analytical solution for v_chi integral:
    v_chi_plus = np.sqrt( (2 * (dE - m_chi - (1/2) * m_n * v_n**2) ) / (m_chi) ) # [unitless]
    if v_chi_plus > V_ESC:
        return 0
  
    E_chi = ( m_chi + (1/2) * m_chi * v_chi_plus**2 ) # [unitless]
    
    norm = norm_chi(sigma_chi, V_ESC) # [unitless]
    if norm<0:
        print("something has gone wrong!")
        return

    return ( (1 / norm) * (1 / E_chi) * v_chi_plus * np.exp(-(v_chi_plus**2) / (2 * sigma_chi**2)) ) # [unitless]

vchiIntegralFactor = (ct.c.to_value(u.km / u.s))/energyScale


# ################ baryon velocity integral ################

def v_n_integral_bounds(m_chi, m_n, dE, V_ESC):
    """
    returns: ([unitless, in units of c], [unitless, in units of c]) min and max bounds on v_n for baryon distribution function integral
    ****************
    m_chi: [unitless, in units of energyScale] DM mass
    m_n: [unitless, in units of energyScale] nucleus mass
    dE: [unitless, in units of energyScale] nuclear excitation energy
    V_ESC: [unitless, in units of c] escape velocity
    """
    v_n_min = np.sqrt(2 * (dE - m_chi - (1/2) * m_chi * V_ESC**2) / m_n) # [unitless]
    v_n_max = np.sqrt(2 * (dE - m_chi) / m_n) # [unitless]
    
    if v_n_min > V_ESC:
        v_n_min = V_ESC
    if v_n_max > V_ESC:
        v_n_max = V_ESC
    if v_n_min > v_n_max:
        print('v_n bounds make no sense!')
        
    return (v_n_min, v_n_max) # ([unitless, in units of c], [unitless, in units of c])


def norm_n(sigma_b, VBAR, V_ESC):
    """
    returns: [unitless] normalization factor for baryon distribution function;
    !!! CHECK AGAIN with function Int_f_b(s, b, l)
    ****************
    sigma_b: [unitless, in units of c] baryon velocity dispersion
    VBAR: [unitless, in units of c] circular baryon velocity, from Clemens 1985 rotation curves
    V_ESC: [unitless, in units of c] escape velocity
    """
    norm = (2 * np.pi * sigma_b**2)**(-3/2) * (4 * np.pi) \
    * (1/2) * sigma_b * (np.sqrt(2 * np.pi) * (VBAR**2 + sigma_b**2) * scipy.special.erf(VBAR / (np.sqrt(2) * sigma_b)) \
                        - np.sqrt(2 * np.pi) * (VBAR**2 + sigma_b**2) * scipy.special.erf((VBAR - V_ESC) / (np.sqrt(2) * sigma_b)) \
                        + 2 * sigma_b * VBAR * np.exp(- VBAR**2 / (2 * sigma_b**2)) \
                        - 2 * sigma_b * (VBAR + V_ESC) * np.exp(- (VBAR - V_ESC)**2 / (2 * sigma_b**2)) )
    return norm # [unitless]


def v_n_integral(m_chi, m_n, dE, r, sigma_b, sigma_chi):
    """
    returns: [unitless] integral of baryon velocity distribution;
    to get unitful dimensions [MeV^-2 km^4 s^-4] multiply by: energyScale**(-2) * ct.c.to_value(u.km/u.s)**4
    ****************
    m_chi: [unitless, in units of energyScale] DM mass
    m_n: [unitless, in units of energyScale] nucleus mass
    dE: [unitless, in units of energyScale] excitation energy
    r: [unitless, in units of r_s] radius from galactic centre
    sigma_b: [unitless, in units of c] baryon velocity dispersion
    sigma_chi: [unitless, in units of c] DM velocity dispersion
    """
    VBAR = v_circ_unitless(r) # [unitless, in units of c]
    V_ESC = np.sqrt(2) * VBAR # [unitless, in units of c]
    # if VBAR>V_ESC:
    #     return 0
    if (dE - m_chi - (1/2) * m_chi * V_ESC**2) < 0: # square root expression in v_n_min; process is not possible
        return 0

    norm = norm_n(sigma_b, VBAR, V_ESC) # [unitless]
    if norm<0:
        print("something has gone wrong!")
        return
        
    v_n_bounds = v_n_integral_bounds(m_chi, m_n, dE, V_ESC) # ([unitless, in units of c], [unitless, in units of c])
    v_n_min, v_n_max = v_n_bounds[0], v_n_bounds[1] # [unitless, in units of c], [unitless, in units of c]
    
    def f(v_n): 
        # v_n: [unitless]
        E_N = ( m_n + (1/2) * m_n * v_n**2 ) # [unitless]
        integrand = ((2 * np.pi) * (1 / E_N) * sigma_b**2 * (v_n / VBAR) \
        * (np.exp(- (v_n - VBAR)**2 / (2 * sigma_b**2) ) - np.exp(- (v_n + VBAR)**2 / (2 * sigma_b**2) )) \
        * v_chi_integral(m_chi, m_n, dE, r, sigma_chi, V_ESC, v_n))
        return integrand # [unitless]
        
    integral, err = quad(f, v_n_min, v_n_max) # integration gives factor of [unitless, in units of c]
    return (1/norm) * integral # [unitless]

vnIntegralFactor = vchiIntegralFactor * energyScale**(-1) * ct.c.to_value(u.km/u.s)**3


# ################ line of sight integral ################

def line_of_sight_integral(m_chi, nucleus, m_n, dE, b, l, R_max):
    """
    returns: [unitless] line of sight integral;
    to get unitful dimensions [cm^-5] multiply by losIntegralFactor:
        (rho_s * rho_s * ct.c.to_value(u.km/u.s)**(-6)) * energyScale**(-2) * ct.c.to_value(u.km/u.s)**4 \
        * ct.c.to_value(u.km/u.s)**2 * 1000**2 * (1 * u.kpc).to_value(u.cm)
    ****************
    m_chi: [unitless, in units of energyScale] DM mass
    nucleus: 'C12' or 'O16'
    m_n: [unitless, in units of energyScale] nucleus mass
    dE: [unitless, in units of energyScale] excitation energy
    b: [deg] galactic latitude
    l: [deg] galactic longitude
    R_max: [unitless, in units of r_s] maximum galactic radius
    """
    psi = np.arccos( np.cos(np.radians(b)) * np.cos(np.radians(l)) ) # [radians]
    s_max = (np.sqrt(R_max * R_max - (R_odot/r_s) * (R_odot/r_s) * np.sin(psi) * np.sin(psi)) + (R_odot/r_s) * np.cos(psi)) # [unitless, in units of r_s]
        
    def f(s):
        # s: [unitless, in units of r_s]
        r = np.sqrt((R_odot/r_s) * (R_odot/r_s) + s * s - 2 * (R_odot/r_s) * s * np.cos(psi)) # [unitless, in units of r_s]

        disp = velocity_dispersions_unitless(r) # ([unitless, in units of c], [unitless, in units of c])
        sigma_b, sigma_chi = disp[0], disp[1] # [unitless, in units of c], [unitless, in units of c]
        
        V_N_integral = v_n_integral(m_chi, m_n, dE, r, sigma_b, sigma_chi) # [unitless]
        if V_N_integral == 0:
            return 0 # save time
        
        rho = density_unitless(nucleus, r, b, l) # ([unitless, in units of rho_s], [unitless, in units of rho_s])
        rho_b, rho_chi = rho[0], rho[1] # [unitless, in units of rho_s], [unitless, in units of rho_s]
        integrand = rho_b * rho_chi * ((2 * np.pi) * sigma_b * sigma_chi)**(-3) * V_N_integral
        return integrand

    esses = np.linspace(0, s_max, 2000) # [unitless, in units of r_s]
    g = list(map(lambda ess: f(ess), esses))
    integral = cumtrapz(g, esses, initial=0) # [unitless], integral gives factor of [unitless, in units of r_s]
    return integral[-1] # unitless


losIntegralFactor = (rho_s * rho_s * ct.c.to_value(u.km/u.s)**(-6)) * vnIntegralFactor * ct.c.to_value(u.km/u.s)**2 * 1000 * 1000 * (1 * u.kpc).to_value(u.cm)
# 1000 * 1000 converts rho_s * rho_s to [MeV/cm^3] * [MeV/cm^3]


# ################ differential flux ################

def diff_flux_for_single_dE(g_chi, m_chi, nucleus, m_n, J_n, dE, GT, b, l, R_max):
    """
    returns: [unitless] differential flux with NO convolution;
    to get unitful dimensions [cm^-2 s^-1 MeV^-1 sr^-1] multiply by fluxFactorNoConv:
        ( 1/energyScale * 1/energyScale \
         * losIntegralFactor \
         * (ct.c.to_value(u.cm/u.s)) * (ct.hbar.to_value(u.MeV * u.s) * ct.c.to_value(u.cm/u.s))**2 )
    ****************
    g_chi: [unitless, in units of 1/energyScale] DM-nucleus coupling constant
    m_chi: [unitless, in units of energyScale] DM mass
    nucleus: 'C12' or 'O16'
    m_n: [unitless, in units of energyScale] nucleus mass
    J_n: [unitless] nucleus spin
    dE: [unitless, in units of energyScale] excitation energy
    GT: [unitless] GT strength for excitation energy dE
    b: [deg] galactic latitude
    l: [deg] galactic longitude
    R_max: [unitless, in units of r_s] maximum galactic radius
    """
    fluxes, flux_tot = [], 0
    
    # do some checks to save time:
    if m_chi > dE:
        # print("m_chi > dE")
        # flux_tot = 0
        return flux_tot

    los_integral = line_of_sight_integral(m_chi, nucleus, m_n, dE, b, l, R_max) # [unitless] 
    # multiply los_integral by losIntegralFactor to get [cm^-5]
    if los_integral == 0:
        # print("los == 0")
        # flux_tot = 0
        return flux_tot

    flux = ((np.pi/6 * g_chi * g_chi * g_A * g_A / (2*J_n + 1)) * (m_n + dE)/m_n * los_integral * GT) # [unitless]
    fluxes.append(flux)
    flux_tot += flux
    # print("flux = {}".format(flux_tot))
    return flux_tot # [unitless]


fluxFactorNoConv = ( 1/energyScale * 1/energyScale \
              * losIntegralFactor \
              * (ct.c.to_value(u.cm/u.s)) * (ct.hbar.to_value(u.MeV * u.s) * ct.c.to_value(u.cm/u.s))**2 )
# note extra factor of MeV^-1 comes from delta function of dN/dE_gamma


def write_diff_flux_to_file(g_chi, m_chi, nucleus, m_n, J_n, dE, GT, b_min, b_max, delta_b, l_min1, l_max1, l_min2, l_max2, delta_l, R_max):
    """
    returns: [unitless * rad] differential flux integrated over galactic latitude from b_min to b_max
    ****************
    g_chi: [unitless, in units of 1/energyScale] DM-nucleus coupling constant
    m_chi: [unitless, in units of energyScale] DM mass
    nucleus: 'C12' or 'O16'
    m_n: [unitless, in units of energyScale] nucleus mass
    J_n: [unitless] nucleus spin
    dE: [unitless, in units of energyScale] excitation energy
    GT: [unitless] GT strength for excitation energy dE
    b_min: [deg] lower integral bound for galactic latitude
    b_max: [deg] upper integral bound for galactic latitude
    l_min1: [deg] lower bound for first galactic longitude integral
    l_max1: [deg] upper bound for first galactic longitude integral
    l_min2: [deg] lower bound for second galactic longitude integral
    l_max2: [deg] upper bound for second galactic longitude integral
    R_max: [unitless, in units of r_s] maximum galactic radius
    """
    try: # check that this mass hasn't already been run
        Path("data/{}/dE_{}MeV/m_chi_{}MeV".format(nucleus, dE*energyScale, m_chi*energyScale)).mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print("\n**** m_chi = {} MeV file has already created! Checking each value of l.\n".format(m_chi*energyScale))
        # return
    
    print("m_chi = {} running".format(m_chi*energyScale))

    ells1 = np.arange(l_min1, l_max1 + delta_l, delta_l)
    ells2 = np.arange(l_min2, l_max2 + delta_l, delta_l)
    bees = np.arange(b_min, b_max + delta_b, delta_b)
    
    for l in [*ells1, *ells2]:
        filename = 'data/{}/dE_{}MeV/m_chi_{}MeV/l_{}deg.txt'.format(nucleus, dE*energyScale, m_chi*energyScale, l)
        if os.path.isfile(filename):
            print("    **** m_chi = {} MeV has already been run for l = {} deg! Trying next l.".format(m_chi*energyScale, l))
            continue
        # print("l = {} deg".format(l))
        diffFluxNoConv = []
        for b in bees:
            # print("    b = {} deg".format(b))
            x = diff_flux_for_single_dE(g_chi, m_chi, nucleus, m_n, J_n, dE, GT, b, l, R_max) # [unitless]
            diffFluxNoConv.append(x * fluxFactorNoConv)
        
        #### write differential flux to file for each value of b ####
        metadata = "# Differential flux.\n\
# m_chi [MeV] = {}\n\
# nucleus = {}\n\
# m_n [MeV] = {}\n\
# J_n = {}\n\
# dE [MeV] = {}\n\
# GT = {}\n\
# \n\
# g_chi [MeV^-1] = {}\n\
# g_A = {}\n\
# \n\
# galactic latitude l [deg] = {}\n\
# \n\
# R_odot [kpc] = {} \n\
# R_max [kpc] = {}\n\
# rho_s [GeV cm^-3] = {}\n\
# r_s [kpc] = {}\n#\n".format(m_chi*energyScale, nucleus, m_n*energyScale, J_n, dE*energyScale, GT, g_chi/energyScale, g_A, l, R_odot, R_max*r_s, rho_s, r_s)
        with open(filename, 'w') as fp:
            fp.write(metadata)
        dat = {'# b [deg]': bees, 'Differential flux [cm^-2 s^-1 MeV^-1 sr^-1]': diffFluxNoConv}
        df = pd.DataFrame(dat)
        df.to_csv(filename, sep='\t', index=False, mode='a')
        ################
        

@click.command()
@click.option(
    "--num_cores", type=int, default=int(1e0), help="number of cores being used (& DM masses being run at once)"
)

     
def main_no_conv(num_cores):
    nucleus = 'C12'
    g_chi = 1 # [MeV^-1]
    R_max = 50.0 # [kpc]
    m_n, J_n = nuc_dict[nucleus]['mass [unitless]'], nuc_dict[nucleus]['spin'] # [unitless, in units of energyScale], [unitless]

    problem_dEs = []

    cores = num_cores
    max_dict = {'C12': 33, 'O16': 38}
    print("number of cores: {}".format(num_cores))
    
    l_min1, l_max1 = 330, ls[-1] # [deg], [deg]; bounds for first galactic longitude integral
    l_min2, l_max2 = ls[0], 30 # [deg], [deg]; bounds for second galactic longitude integral
    delta_l = 0.25
    b_min, b_max = -5, 5 # [deg], [deg]; bounds for galactic latitude integral
    delta_b = 0.25
    
    print(nucleus)
    
    for dE in nuc_dict[nucleus]['dEs [MeV]'][0:max_dict[nucleus]:4]:
        try:
            # comptel data only goes to 30 MeV: 
                # nuc_dict['C12']['dEs [MeV]'][32] < 30 MeV
                # nuc_dict['O16']['dEs [MeV]'][37] < 30 MeV
            # beyond 30 MeV, at 80 for C12 & 75 need to account for both C and O being excited by the same mass
            i = list(nuc_dict[nucleus]['dEs [MeV]']).index(dE)
            GT = nuc_dict[nucleus]['GTs'][i] # [unitless]
            print("\ni = {}:    dE = {} MeV, GT = {}".format(i, dE, GT))

            # retrieve DM masses that maximize the v_n_integral for a given dE:
            mass_by_dE_data = 'optimal_m_chis_by_dE/{}/new_baryon_dispersion/{}_optimal_m_chis_dE_{}_MeV.txt'.format(nucleus, nucleus, dE)
            mass_by_dE_df = pd.read_csv(mass_by_dE_data, sep='\t', names=['Radius from GC [kpc]', 'Optimal DM mass [MeV]'], skiprows=1, float_precision='round_trip')
            M_CHIS = mass_by_dE_df['Optimal DM mass [MeV]']
            
            # use these masses:
            m_chis = list(dict.fromkeys(M_CHIS))[0:-1:2] # convert to dictionary & back to list to remove any duplicates

            # or use an array of masses:
            # delta_m = (np.max(M_CHIS) - np.min(M_CHIS)) # for all dEs, delta_m =~ 0.0038
            # m_chis = np.arange(np.min(M_CHIS) - 0.004, dE, 1e-3) # np.arange(np.min(M_CHIS) - 0.004, dE, 1e-3) this seems to work well
            #m_chis = np.arange(round(np.min(M_CHIS),3) - 0.01, dE, 1e-3) 

            with Pool(cores) as pool:
                x = pool.starmap(write_diff_flux_to_file, \
                                 [(g_chi*energyScale, m_chi/energyScale, nucleus, m_n, J_n, dE/energyScale, GT, \
                                   b_min, b_max, delta_b, l_min1, l_max1, l_min2, l_max2, delta_l, \
                                   R_max/r_s) for m_chi in m_chis])
                pool.close()
                pool.join()

        except:
            print("\n**** problem with i = {}:    dE = {} MeV, GT = {}\n".format(i, dE, GT))
            problem_dEs.append(dE)
            continue

    print(problem_dEs)


if __name__=="__main__":
    main_no_conv()


# run example:
# python3 C12-0-new_CO_density.py --num_cores 16
