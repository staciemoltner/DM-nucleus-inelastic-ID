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
import bisect


################ constants & data import ################

# natural & conversion constants
c = 2.998e8 # speed of light [m/s]
G = 6.674e-11 # [N m^2 kg^-2] = [m^3 s^-2 kg^-1]
hbar = 6.582e-16 # [eV s]
GeV_per_kg = 5.625e26
cm_per_kpc = 3.086e21


# constants
rho_s = 0.51 # DM radial scale for MW NFW profile [GeV/cm^3]; https://arxiv.org/abs/1906.08419
r_s = 8.1 # DM radial scale for MW NFW profile [kpc]; https://arxiv.org/abs/1906.08419
M_sun = 2e30 # [kg]
M_bulge = 1.5e10 * M_sun # [kg]
c_bulge = 0.6 # bulge scale radius [kpc]
M_disk = 7e10 * M_sun # [kg]
b_disk = 4 # disk scale radius [kpc]
R_odot = 8.5 # Sun's distance from centre of MW [kpc]
M_SMBH = 4.154e6 * M_sun # [kg]
g_A = 1.27 # axial form factor [unitless]


# nuclear info dictionary
nuc_info = {'He4': [3758.26, 0.0], 'C12': [11274.78, 0.0], 'N14': [13153.91, 0.0], 'O16': [15033.04, 0.0]}


# importing C12 and O16 data: excitation energies and GT strengths
C12_data = '{}_{}_{}.txt'.format('C12', nuc_info['C12'][0], nuc_info['C12'][1])
O16_data = '{}_{}_{}.txt'.format('O16', nuc_info['O16'][0], nuc_info['O16'][1])
Cdf = pd.read_csv(C12_data, sep='\t', names=['E_gamma [MeV]', 'GT strength'], skiprows=1)
Odf = pd.read_csv(O16_data, sep='\t', names=['E_gamma [MeV]', 'GT strength'], skiprows=1)
CdEs, OdEs = Cdf['E_gamma [MeV]'], Odf['E_gamma [MeV]']
dEs_dict = {'C12': Cdf['E_gamma [MeV]'], 'O16': Odf['E_gamma [MeV]']}
GTs_dict = {'C12': Cdf['GT strength'], 'O16': Odf['GT strength']}


# optimal m_chi values, calculated at r = R_odot
def read_optimal_m_chis(nucleus):
    df = pd.read_csv('optimal_m_chis_{}_r_8500pc.txt'.format(nucleus), \
                     sep='\t', \
                     names=['Nuclear excitation energy [MeV]', 'Optimal DM mass [MeV]'], \
                     skiprows=1)
    return [df['Nuclear excitation energy [MeV]'], df['Optimal DM mass [MeV]']]
m_chis_dEs_C12 = read_optimal_m_chis('C12')
dEs_C12, m_chis_C12 = m_chis_dEs_C12[0], m_chis_dEs_C12[1]
m_chis_dEs_O16 = read_optimal_m_chis('O16')
dEs_O16, m_chis_O16 = m_chis_dEs_O16[0], m_chis_dEs_O16[1]
allCOMasses = np.append(m_chis_C12, m_chis_O16)
allCOdEs = np.append(dEs_C12, dEs_O16)


################ GALPROP density data ################

# https://galprop.stanford.edu/download/manuals/galprop_v54.pdf
# https://fits.gsfc.nasa.gov/users_guide/usersguide.pdf
# https://astropy4cambridge.readthedocs.io/en/latest/_static/Astropy%20-%20Handling%20FITS%20files.html

hdul_CO = fits.open('massdensity_CO.fits')
data = hdul_CO[0].data # [kg/cm^3] density data, see below for explanation
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


# creating b and l arrays
b_len, l_len = hdul_CO[0].header['NAXIS2'], hdul_CO[0].header['NAXIS1'] # length of b, l arrays
b_crval, l_crval = hdul_CO[0].header['CRVAL2'], hdul_CO[0].header['CRVAL1'] # central values of b,l
b_delta, l_delta = hdul_CO[0].header['CDELT2'], hdul_CO[0].header['CDELT1'] # increments for b, l
bs = list(np.arange(b_crval, b_crval + b_len*b_delta, b_delta))
ls = list(np.arange(l_crval, l_crval + l_len*l_delta, l_delta))


def plotCOdensity(n):
    """
    plots CO density for bin number n in [0, 8]
    """
    fig = plt.figure(figsize = (10,15))
    ax = plt.axes()
    im = ax.imshow(hdul_CO[0].data[n], cmap='magma', norm=matplotlib.colors.LogNorm())
    A, B = 0.02, 0.04
    cax = fig.add_axes([ax.get_position().x1 + A, ax.get_position().y0, B, ax.get_position().height])
    plt.colorbar(im, cax=cax)


#### rotation curve from Clemens (1985) ####

R_0 = R_odot # [kpc]
theta_0 = 220 # [km/s]

R_i = [0.0, 0.09 * R_0, 0.45 * R_0, 1.6 * R_0, 50.0]

A_i = [0.0, 3069.81, -15809.8, +43980.1, -68287.3, +54904., -17731.] # R/R_0 < 0.09
B_i = [+325.0912, -248.1467, +231.87099, -110.73531, +25.073006, -2.110625]
C_i = [-2342.6564, +2507.60391, -1024.068760, +224.562732, -28.4080026, +2.0697271, -0.08050808, +0.00129348]
D_i = [234.88]

interval = 0.1
A_R = np.arange(R_i[0], R_i[1] + interval, interval)
B_R = np.arange(R_i[1], R_i[2] + interval, interval)
C_R = np.arange(R_i[2], R_i[3] + interval, interval)
D_R = np.arange(R_i[3], R_i[4] + interval, interval)

theta_A = np.zeros(len(A_R))
theta_B = np.zeros(len(B_R))
theta_C = np.zeros(len(C_R))
theta_D = np.zeros(len(D_R))
for i in range(7):
    theta_A += A_i[i] * A_R**i
for i in range(6):
    theta_B += B_i[i] * B_R**i
for i in range(8):
    theta_C += C_i[i] * C_R**i
for i in range(1):
    theta_D += D_i[i]

# radius and rotational velocity to be used to calculate circular velocity v_circ(r) at any r:
r_rot = np.append(np.append(np.append(A_R, B_R), C_R), D_R)
v_rot = np.append(np.append(np.append(theta_A, theta_B), theta_C), theta_D)


def plotRotationCurves():
    plt.figure(figsize=(8, 6))
    plt.xlim(-1, D_R[-1]); plt.ylim(0, 275)
    plt.xlabel('Radius from galactic centre [kpc]', fontsize=14)
    plt.ylabel('Rotation speed [km/s]', fontsize=14)
    plt.plot(A_R, theta_A)
    plt.plot(B_R, theta_B)
    plt.plot(C_R, theta_C)
    plt.plot(D_R, theta_D)


def v_circ(r):
    interp_func = interp1d(r_rot, v_rot, fill_value='extrapolate')
    v = interp_func(r)
    return v


################ galactic radius & maximum line of sight distance ################

def R(s, b, l):
    """
    returns: [kpc] radius, as a function of s, b, l
    ********
    s: [kpc] line of sight coordinate
    b: [degrees] galactic latitude
    l: [degrees] galactic longitude
    """
    psi = np.arccos( np.cos(np.radians(b)) * np.cos(np.radians(l)) ) # [radians]
    return np.sqrt(R_odot**2 + s**2 - 2 * R_odot * s * np.cos(psi)) # [kpc]


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


def plotRadiusVsLOS(R_max):
    plt.figure(figsize = (12, 8))
    plt.grid(which = 'both', linestyle = 'dotted')
    plt.xticks(size = 16); plt.yticks(size=16);
    plt.xlabel(r'Line of sight distance [kpc]', fontsize = 16); plt.ylabel(r'Radius [kpc]', fontsize = 16)
    plt.xlim(0, s_max(0, 0, R_max))
    for deg in [0, 5, 15, 30, 90]:
        bb = 0 # in galactic plane
        ll = deg
        S_MAX = s_max(bb, ll, 2 * R_max)
        # print(S_MAX)
        ss = np.linspace(0, S_MAX, 400)
        rs = list(map(lambda s: R(s, bb, ll), ss))
        plt.plot(ss, rs, label="b, l = {}$^\circ$, {}$^\circ$".format(bb, ll))
    plt.plot()
    plt.legend(fontsize=14)
    plt.title('radial coordinate as function of line of sight', fontsize = 20)


################ density ################

def densityInterpolate(r, b, l):
    """
    returns: [kg/cm^3] density at a given radius, interpolated for values (b, l) from gridded data
    """
    r_index = bisect.bisect(r_bins, r) - 1 # find radial bin index
    values_grid = hdul_CO[0].data[r_index] # reduces 3D grid (r, b, l) to 2D (b, l)
    interpolating_fx = RegularGridInterpolator((bs,ls), values_grid)
    return interpolating_fx(np.array([b, l]))


def densityLOS(nucleus, s, b, l):
    """
    returns: [GeV/cm^3] mass density as a function of line of sight distance
    **********
    particle: 'DM', 'He4', 'C12', 'N14', 'O16', 'CO', 'Fe57'
    s: [kpc] line of sight distance
    b: [degrees] galactic latitude
    l: [degrees] galactic longitude
    """
    r = R(s, b, l) # [kpc] # finds radius for a given (s, b, l) in order to find the dispersion at those coordinates
    if r > r_bins[-1]: # if radius is beyond what is provided by GALPROP, return zero
        return 0
    
    rho_b = nuc_info[nucleus][0] * densityInterpolate(r, b, l)[0] / (nuc_info['C12'][0] + nuc_info['O16'][0]) * GeV_per_kg # [GeV/cm^3]
    rho_DM = rho_s / ( (r/r_s) * (1 + r/r_s)**2 ) # [GeV/cm^3]
    return [rho_b, rho_DM]


def density(nucleus, r, b, l):
    """
    returns: [GeV/cm^3] mass density as a function of radius from galactic centre
    **********
    particle: 'DM', 'He4', 'C12', 'N14', 'O16', 'CO', 'Fe57'
    r: [kpc] radius from galactic centre
    b: [degrees] galactic latitude
    l: [degrees] galactic longitude
    """
    m_N_MeV = {'He4': 3758.26, 'C12': 11274.78, 'N14': 13153.91, 'O16': 15033.04}
    rho_DM = rho_s / ( (r/r_s) * (1 + r/r_s)**2 ) # [GeV/cm^3]
    if r >= r_bins[-1]: # if radius is beyond what is provided by GALPROP, return zero
        #print('no baryon density for this radius, try reducing R_max')
        rho_b = 0
        return(rho_b, rho_DM)    
    
    rho_b = nuc_info[nucleus][0] * densityInterpolate(r, b, l)[0] / (nuc_info['C12'][0] + nuc_info['O16'][0]) * GeV_per_kg # [GeV/cm^3]
    return [rho_b, rho_DM]


def plotLOSDMDensity(R_max):
    plt.figure(figsize = (12, 8))
    plt.xlabel(r'Line of sight distance [kpc]', fontsize = 16)
    plt.ylabel(r'DM density [GeV/cm$^3$]', fontsize = 16)
    plt.xlim(0, s_max(b_crval, l_crval + 90, N_odot * R_odot))
    for deg in [5, 15, 30, 90]:#, 45]:
        bb = 0
        ll = l_crval + deg
        S_MAX = s_max(bb, ll, R_max)
        ss = np.linspace(0, S_MAX, 400)
        rho_DM = list(map(lambda s:densityLOS('C12', s, bb, ll)[1], ss))
        plt.plot(ss, rho_DM, label="b, l = {}$^\circ$, {}$^\circ$".format(bb, ll))
    plt.plot()
    plt.legend()
    plt.title('dm density as function of line of sight', fontsize = 24)


def plotLOSBaryonDensity(R_max):
    fig, ax1 = plt.subplots(figsize = (12, 8))
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax1.set_xlabel(r'Line of sight distance [kpc]', fontsize = 16)
    ax1.set_ylabel(r'Baryon density [kg/cm$^3$]', fontsize = 16)
    ax2.set_ylabel(r'Baryon density [GeV/cm$^3$]', fontsize = 16)  # we already handled the x-label with ax1
    ax2.set_xlim(0,35)
    
    for deg in [5, 15, 30, 90]:#, 45]:
        bb = 0
        ll = l_crval + deg
        S_MAX = s_max(bb, ll, R_max)
        ss = np.linspace(0, S_MAX, 400)
        rho_b_kg = list(map(lambda s:densityLOS('C12', s, bb, ll)[0] * 1/GeV_per_kg, ss))
        rho_b_GeV = list(map(lambda s:densityLOS('C12', s, bb, ll)[0], ss))
        ax1.plot(ss, rho_b_kg, label="b, l = {}$^\circ$, {}$^\circ$".format(bb, ll))
        ax2.plot(ss, rho_b_GeV, label="b, l = {}$^\circ$, {}$^\circ$".format(bb, ll), color='black', linestyle='')
        ax1.legend()
    
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title('baryon density as function of line of sight', fontsize = 24)


def plotRadialBaryonDensity(R_max):
    fig, ax1 = plt.subplots(figsize = (12, 8))
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax1.set_xlabel(r'Radius [kpc]', fontsize = 16)
    ax1.set_ylabel(r'Baryon density [kg/cm$^3$]', fontsize = 16)
    ax2.set_ylabel(r'Baryon density [GeV/cm$^3$]', fontsize = 16)  # we already handled the x-label with ax1
    ax2.set_xlim(0,35)
    
    for deg in [5, 15, 30, 90]:#, 45]:
        bb = 0
        ll = l_crval + deg
        arrs = np.linspace(1e-3, R_max, 400)
        rho_b_kg = list(map(lambda arr:density('C12', arr, bb, ll)[0] * 1/GeV_per_kg, arrs))
        rho_b_GeV = list(map(lambda arr:density('C12', arr, bb, ll)[0], arrs))
        ax1.plot(arrs, rho_b_kg, label="b, l = {}$^\circ$, {}$^\circ$".format(bb, ll))
        ax2.plot(arrs, rho_b_GeV, label="b, l = {}$^\circ$, {}$^\circ$".format(bb, ll), color='black', linestyle='')
        ax1.legend()
    
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title('baryon density as function of radius from GC', fontsize = 24);


def vDispersion(r):
    """
    returns: 3D velocity dispersion [km/s]
    **********
    r: radius from galactic centre [kpc]
    
    DM dispersion function taken from 2111.03076 Figure 1: r in units kpc, velocity dispersion in units km/s
    """
    disp_b = 10 # [km/s]
    disp_DM = (-42.4 * np.log10(r) + 382.2) # [km/s]
    
    return (disp_b, disp_DM)
        


################ escape velocity ################

def radialCOMassDensity():
    rho_r_data = pd.read_csv('massDensity_CO_r.csv', names=['Radius [kpc]', 'Mass density [kg/m^3]'], skiprows=1)
    rho_radius, rho_rho = rho_r_data['Radius [kpc]'], rho_r_data['Mass density [kg/m^3]']
    plt.figure(figsize = (8,6))
    plt.grid(which = 'both', linestyle = 'dotted')
    plt.xticks(size = 16); plt.yticks(size=16);
    plt.xlabel(r'Radius [kpc]', fontsize=16); plt.ylabel(r'Mass density of CO [kg/m$^3$]', fontsize=16)
    plt.xlim(rho_radius[0], rho_radius.iat[-1])
    plt.plot(rho_radius, rho_rho)


def M_b(r):
    """
    returns: [kg] baryon mass within radius r
    assumes spherical baryon potential given by bulge + spherically symmetrized disk (see arXiv:1805.08379)
    """
    def f(R):
        return R**2 * ( (c_bulge * M_bulge) / (2 * np.pi * R * (c_bulge + R)**3) \
                       + (M_disk * np.exp(-R / b_disk)) / (4 * np.pi * b_disk**2 * R) ) # [kg/kpc^3 * kpc^2]
    result = quad(f, 0, r)
    mass_kg = 4 * np.pi * result[0] # [kg]
    return mass_kg


def M_DM(r):
    """
    returns: [kg] DM mass of MW within radius r_esc [kpc]
    r: [kpc]
    """
    def f(R):
        """
        returns: [GeV/cm^3 * kpc^3] DM mass
        """
        return 2*np.pi * 2 * R**2 * rho_s / ( (R/r_s) * (1 + R/r_s)**2 ) # [GeV/cm^3] * kpc^3
    # convert to kg
    mass_kg = quad(f, 0, r)[0] * cm_per_kpc**3 * 1/GeV_per_kg # [kg]

    return mass_kg


def M_MW(r):
    z = 1 # [kpc]
    # M_total = M_CO(r_esc, z) + M_DM(r_esc) # [kg]
    M_total = M_b(r) + M_DM(r) # [kg]
    mass_odot = M_total / M_sun # [solar masses]
    # print("{:2.2e} solar masses".format(mass_odot))
    # print("MW galaxy should be approximately 1.2e12 solar masses")
    return M_total


def v_esc(r):
    """
    returns: [km/s] escape velocity as function of radius
    *** CURRENTLY ONLY ASSUMES DM MAKING UP MASS ***
    """
    M = M_MW(r) # [kg]
    # M = 1.2e12 * M_sun
    # convert r from kpc to m
    # convert v_esc to km/s
    return 3 * np.sqrt(2 * G * M / (r * cm_per_kpc / 100)) / 1000 # [km/s]


def plotEscapeVelocity():
    arrs = np.linspace(0.1, 50.0, 100)
    plt.figure(figsize=(8, 6))
    plt.xlim(-1, 20.0); plt.ylim(v_esc(arrs[0]), 900)
    plt.xlabel('Radius from galactic centre [kpc]', fontsize=14)
    plt.ylabel('Escape velocity [km/s]', fontsize=14)
    plt.plot(arrs, list(map(lambda arr: v_esc(arr), arrs)))


################ DM velocity distribution integral ################

def norm_chi(V_ESC, sigma_chi):
    """
    returns: normalization factor for DM distribution function, checked with function Int_f_chi(s, b, l)
    """
    norm = (2*np.pi * sigma_chi**2)**(-3/2) * (4 * np.pi) \
    * ( np.sqrt(np.pi / 2) * sigma_chi**3 * scipy.special.erf(V_ESC / (np.sqrt(2) * sigma_chi) ) \
       - sigma_chi**2 * V_ESC * np.exp(-V_ESC**2 / (2 * sigma_chi**2)))
    return norm


def Int_v_chi(m_chi, m_N, deltaE, r, v_n):
    """
    returns: integral of DM velocity distribution [MeV^-1 km s^-1]
    **********
    deltaE: excitation energy [MeV]
    m_chi: DM mass [MeV] 
    m_N: nucleus mass [MeV]
    v_i: nucleus velocity [km/s]
    r: radius from galactic centre [kpc]
    """
    if m_chi < 0:
        return 0
    
    V_N = v_n / (c/1000) # convert from [km/s] to [unitless]
    if (deltaE - m_chi - (1/2) * m_N * V_N**2) < 0: # process is not possible
        return 0

    v_chi_plus = np.sqrt( (2 * (deltaE - m_chi - (1/2) * m_N * V_N**2) ) / (m_chi) ) * (c/1000) # [km/s]
    V_ESC = v_esc(r) # [km/s]
    if v_chi_plus > V_ESC:
        return 0
        
    V_CHI = v_chi_plus / (c/1000) # convert from [km/s] to [unitless]        
    E_chi = ( m_chi + (1/2) * m_chi * V_CHI**2 ) # [MeV]
    sigma_chi = vDispersion(r)[1] # [km/s]
    norm = norm_chi(V_ESC, sigma_chi)

    return ( (1 / norm) * (1 / E_chi) * v_chi_plus * np.exp(-v_chi_plus**2 / (2 * sigma_chi**2)) ) # [MeV^-1 km/s]


################################################################


################ baryon velocity distribution integral ################

def v_N_integralBounds(m_N, m_chi, deltaE, V_ESC):
    """
    returns: [km/s] min and max bounds on v_N
    """
    v_N_min = np.sqrt(2 * (deltaE - m_chi - (1/2) * m_chi * (V_ESC/(c/1000))**2) / m_N) * (c/1000) # [km/s]
    v_N_max = np.sqrt(2 * (deltaE - m_chi) / m_N) * (c/1000) # [km/s]
    
    if v_N_min > V_ESC:
        v_N_min = V_ESC
    if v_N_max > V_ESC:
        v_N_max = V_ESC
    if v_N_min > v_N_max:
        print('v_N bounds make no sense!')
        
    return [v_N_min, v_N_max] # [km/s]


def norm_N(VBAR, V_ESC, sigma_b):
    """
    returns: [unitless] normalization factor for baryon distribution function, checked with function Int_f_b(s, b, l)
    """
    norm = (4 * np.pi) * (2*np.pi * sigma_b**2)**(-3/2) \
    * (1/2) * sigma_b * (np.sqrt(2 * np.pi) * (VBAR**2 + sigma_b**2) * scipy.special.erf(VBAR / (np.sqrt(2) * sigma_b)) \
                        - np.sqrt(2 * np.pi)  * (VBAR**2 + sigma_b**2) * scipy.special.erf((VBAR - V_ESC) / (np.sqrt(2) * sigma_b)) \
                        + 2 * sigma_b * VBAR * np.exp(- VBAR**2 / (2 * sigma_b**2)) \
                        - 2 * sigma_b * (VBAR + V_ESC) * np.exp(- (VBAR - V_ESC)**2 / (2 * sigma_b**2)) )
    return norm


def Int_v_N(m_chi, m_N, deltaE, r):
    """
    returns: [MeV^-2 km^4 s^-4] integral of baryon velocity distribution
    **********
    deltaE: [MeV] excitation energy
    m_chi: [MeV] DM mass
    nucleus: scattering target
    r: [kpc] radius from galactic centre
    """
    if m_chi > deltaE: # process not possible
        return 0

    VBAR = v_circ(r) # [km/s]
    V_ESC = v_esc(r) # [km/s]
    # if VBAR>V_ESC:
    #     return 0
    if (deltaE - m_chi - (1/2) * m_chi * (V_ESC/(c/1000))**2) < 0: # v_N_min expression
        return 0
    
    sigma_b = vDispersion(r)[0] # [km/s]
    norm = norm_N(VBAR, V_ESC, sigma_b) # [unitless]
    if norm<0:
        print("something has gone wrong!")
        return
        
    v_N_bounds = v_N_integralBounds(m_N, m_chi, deltaE, V_ESC) # ([km/s], [km/s])
    v_N_min, v_N_max = v_N_bounds[0], v_N_bounds[1] # [km/s], [km/s]
    
    def f(v_n): # v_n units [km/s]
        V_N = v_n / (c/1000) # convert from [km/s] to [unitless]
        E_N = ( m_N + (1/2) * m_N * V_N**2 ) # [MeV]
        
        integrand = ((2 * np.pi) * (1 / E_N) * sigma_b**2 * (v_n / VBAR) \
        * (np.exp(- (v_n - VBAR)**2 / (2 * sigma_b**2) ) - np.exp(- (v_n + VBAR)**2 / (2 * sigma_b**2) )) \
        * Int_v_chi(m_chi, m_N, deltaE, r, v_n)) 
        # [MeV^-1] * [km/s]^2  *  [unitless]  *  [MeV^-1 km/s]
        return integrand # [MeV^-2 * km^3/s^3]
        
    integral, err = quad(f, v_N_min, v_N_max) # integration gives factor of [km/s]
    return (1/norm) * integral # [MeV^-2 km^4 s^-4]


def plot_f_v_N(m_chi, m_N, deltaE):
    for r in [7, 8.5, 9, 10]:
        if m_chi > deltaE: # process not possible
            return 0
    
        VBAR = v_circ(r) # [km/s]
        V_ESC = v_esc(r) # [km/s]
        # if VBAR>V_ESC:
        #     return 0
        if (deltaE - m_chi - (1/2) * m_chi * (V_ESC/(c/1000))**2) < 0: # v_N_min expression
            return 0
        
        sigma_b = vDispersion(r)[0] # [km/s]
        norm = norm_N(VBAR, V_ESC, sigma_b)
        if norm<0:
            return 0
            
        v_N_bounds = v_N_integralBounds(m_N, m_chi, deltaE, V_ESC)
        v_N_min, v_N_max = v_N_bounds[0], v_N_bounds[1] # [km/s], [km/s]
        vees = np.linspace(v_N_min, v_N_max, 100)
    
        def f(v_n):
            V_N = v_n / (c/1000) # convert from km/s to unitless
            E_N = ( m_N + (1/2) * m_N * V_N**2 ) # [MeV]
            # integrand = (1 / E_N) * sigma_b**2 * (v_n / VBAR) * (np.exp(- (v_n - VBAR)**2 / (2 * sigma_b**2) ) - np.exp(- (v_n + VBAR)**2 / (2 * sigma_b**2) ))            
            integrand = ((2 * np.pi) * (1 / E_N) * sigma_b**2 * (v_n / VBAR) \
            * (np.exp(- (v_n - VBAR)**2 / (2 * sigma_b**2) ) - np.exp(- (v_n + VBAR)**2 / (2 * sigma_b**2) )) \
            * Int_v_chi(m_chi, m_N, deltaE, r, v_n))
            # [MeV^-1] * [km/s]^2  *  [unitless]  *  [MeV^-1 km/s]
            return integrand # [MeV^-2 * km^3/s^3]

        def f_220(v_n):
            V_N = v_n / (c/1000) # convert from km/s to unitless
            E_N = ( m_N + (1/2) * m_N * V_N**2 ) # [MeV]
            # integrand = (1 / E_N) * sigma_b**2 * (v_n / 220) * (np.exp(- (v_n - 220)**2 / (2 * sigma_b**2) ) - np.exp(- (v_n + 220)**2 / (2 * sigma_b**2) ))
            integrand = ((2 * np.pi) * (1 / E_N) * sigma_b**2 * (v_n / 220) \
            * (np.exp(- (v_n - 220)**2 / (2 * sigma_b**2) ) - np.exp(- (v_n + 220)**2 / (2 * sigma_b**2) )) \
            * Int_v_chi(m_chi, m_N, deltaE, r, v_n))
            # [MeV^-1] * [km/s]^2  *  [unitless]  *  [MeV^-1 km/s]
            return integrand # [MeV^-2 * km^3/s^3]
        
        plt.figure(figsize=(8, 6))
        plt.xlim(vees[0], vees[-1])
        plt.xlabel('Baryon velocity [km/s]'); plt.ylabel('Ratio of distribution functions: f(v_circ(r)) / f(220)')
        plt.plot(vees[1:-2], list(map(lambda vee: f(vee)/f_220(vee), vees[1:-2])), label='r = {}'.format(r))
        # plt.plot(vees, list(map(lambda vee: f(vee), vees)), label='r = {}'.format(r))
        # plt.plot(vees, list(map(lambda vee: f_220(vee), vees)), label='r = {}'.format(r), linestyle = 'dashed', color='black')
    plt.legend()
    return


def logInt_v_N(m_chi, m_N, deltaE, r):
    """
    returns: [MeV^-2 km^4 s^-4] integral of baryon velocity distribution
    **********
    deltaE: [MeV] excitation energy
    m_chi: [MeV] DM mass
    nucleus: scattering target
    r: [kpc] radius from galactic centre
    """
    if m_chi > deltaE: # process not possible
        return 0

    VBAR = v_circ(r) # [km/s]
    V_ESC = v_esc(r) # [km/s]
    # if VBAR>V_ESC:
    #     return 0
    if (deltaE - m_chi - (1/2) * m_chi * (V_ESC/(c/1000))**2) < 0: # v_N_min expression
        return 0
    
    sigma_b = vDispersion(r)[0] # [km/s]
    norm = norm_N(VBAR, V_ESC, sigma_b) # [unitless]
    if norm<0:
        return 0
        
    v_N_bounds = v_N_integralBounds(m_N, m_chi, deltaE, V_ESC)
    v_N_min, v_N_max = v_N_bounds[0], v_N_bounds[1] # [km/s], [km/s]
    
    def f(v_n):
        V_N = v_n / (c/1000) # convert from km/s to unitless
        E_N = ( m_N + (1/2) * m_N * V_N**2 ) # [MeV]
        
        integrand = ((2 * np.pi) * (1 / E_N) * sigma_b**2 * (v_n / VBAR) \
        * (np.exp(- (v_n - VBAR)**2 / (2 * sigma_b**2) ) - np.exp(- (v_n + VBAR)**2 / (2 * sigma_b**2) )) \
        * Int_v_chi(m_chi, m_N, deltaE, r, v_n)) 
        # [MeV^-1] * [km/s]^2  *  [unitless]  *  [MeV^-1 km/s]
        return integrand # [MeV^-2 * km^3/s^3]
    
    def g(u):
        return np.exp(u) * f(np.exp(u))
        
    integral, err = quad(g, np.log(v_N_min), np.log(v_N_max))
    return (1/norm) * integral # [MeV^-2 km^3 s^-3 * km/s] = [MeV^-2 km^4 s^-4]


################################################################


################ line of sight $s$ integral ################

def Int_LOS(m_chi, nucleus, deltaE, b, l, R_max):
    """
    returns: [cm^-5] line of sight integral
    ********
    """
    # nuc_info = {'He4': [3758.26, 0.0], 'C12': [11274.78, 0.0], 'N14': [13153.91, 0.0], 'O16': [15033.04, 0.0]}
    m_N = nuc_info[nucleus][0]
    if m_chi > deltaE:
        return 0
    psi = np.arccos( np.cos(np.radians(b)) * np.cos(np.radians(l)) ) # [radians]
    s_max = (np.sqrt(R_max**2 - R_odot**2 * (np.sin(psi))**2) + R_odot * np.cos(psi)) # [kpc]
    
    # scale to make integrand ~O(1):
    rhoScale = density(nucleus, R_odot, bs[180], l_crval) # needs to be nonzero; using value at Sun location
    sigmaScale = vDispersion(R_odot)
    DELTAE = min(allCOdEs, key=lambda x:abs(x-m_chi))
    integralScale = Int_v_N(m_chi, m_N, DELTAE, R_odot) * rhoScale[0] * rhoScale[1] * (sigmaScale[0] * sigmaScale[1])**(-3)
    # print(integralScale)
    
    def f(s):
        r = np.sqrt(R_odot**2 + s**2 - 2 * R_odot * s * np.cos(psi)) # [kpc]
        rho = density(nucleus, r, b, l) # ([GeV/cm^3], [GeV/cm^3])
        rho_b, rho_chi = rho[0], rho[1] # [GeV/cm^3], [GeV/cm^3]
        disp = vDispersion(r) # ([km/s], [km/s])
        sigma_b, sigma_chi = disp[0], disp[1] # [km/s], [km/s]
        integrand = rho_b * rho_chi * ( (2*np.pi) * sigma_b * sigma_chi )**(-3) \
        * Int_v_N(m_chi, m_N, deltaE, r) / integralScale
        # print(integrand)
        return integrand
        # [GeV/cm^3] * [GeV/cm^3] * [km/s]^-3 * [km/s]^-3 * [MeV^-2 km^4 s^-4] = [GeV^2 cm^-6 km^-2 s^2 MeV^-2]

    # integral, err = quad(f, 0, s_max)#, epsrel = 1e-5, epsabs = 0, limit=200)
    esses = np.linspace(0, s_max, 500)
    g = list(map(lambda ess: f(ess), esses))
    integral = cumtrapz(g, esses, initial=0) # integrating over kpc gives a factor of kpc
    # return integral * (c/1000)**2 * 1000**2 * cm_per_kpc * integralScale # [cm^-5]
    return integral[-1] * (c/1000)**2 * 1000**2 * cm_per_kpc * integralScale # [cm^-5]


def logInt_LOS(s_min, m_chi, nucleus, deltaE, b, l, R_max):
    """
    returns: [cm^-5] line of sight integral
    ********
    """
    # nuc_info = {'He4': [3758.26, 0.0], 'C12': [11274.78, 0.0], 'N14': [13153.91, 0.0], 'O16': [15033.04, 0.0]}
    m_N = nuc_info[nucleus][0]
    if m_chi > deltaE:
        return 0
        
    psi = np.arccos( np.cos(np.radians(b)) * np.cos(np.radians(l)) ) # [radians]
    def f(s):
        r = np.sqrt(R_odot**2 + s**2 - 2 * R_odot * s * np.cos(psi)) # [kpc]
        
        rho = density(nucleus, r, b, l)
        rho_b, rho_chi = rho[0], rho[1] # [GeV/cm^3], [GeV/cm^3]
        
        disp = vDispersion(r) # km/s
        sigma_b, sigma_chi = disp[0], disp[1] # km/s
        
        integrand = rho_b * rho_chi * ( (2*np.pi) * sigma_b * sigma_chi )**(-3) * Int_v_N(m_chi, m_N, deltaE, r)* (c/1000)**2 * 1000**2 * cm_per_kpc
        return integrand
        # [GeV/cm^3] * [GeV/cm^3] * [km/s]^-3 * [km/s]^-3 * [MeV^-2 km^4 s^-4] = [GeV^2 cm^-6 km^-2 s^2 MeV^-2]

    def g(u):
        return np.exp(u) * f(np.exp(u))
        
    s_max = (np.sqrt(R_max**2 - R_odot**2 * (np.sin(psi))**2) + R_odot * np.cos(psi)) # [kpc]
    
    # esses = np.linspace(s_min, s_max, 100)
    # plt.xlim(esses[0], esses[-1])
    # plt.plot(esses, list(map(lambda ess: np.log(f(ess)), esses)))
    
    # logesses = np.linspace(np.log(s_min), np.log(s_max), 100)
    # plt.xlim(logesses[0], logesses[-1])
    # plt.plot(logesses, list(map(lambda logess: np.log(g(logess)), logesses)))

    integral, err = quad(g, np.log(s_min), np.log(s_max), epsabs = 1e-3)
    print(integral, err)
    return #integral * (c/1000)**2 * 1000**2 * cm_per_kpc # [cm^-5]
    # return (integral * 1000**2 * (c/1000)**2 * cm_per_kpc) # [cm^-5]
    # integrating over kpc gives a factor of kpc


################################################################


################ differential flux ################

def flux_noConv(g_chi, m_chi, nucleus, b, l, R_max, epsilon):
    """
    returns: differential flux in [cm^-2 s^-1 (sr^-1) MeV^-1]
    **********
    g_chi: [MeV^-1]
    E: observed photon energy [MeV]
    """
    fluxes, flux_tot = [], 0
    m_N, J_N = nuc_info[nucleus][0], nuc_info[nucleus][1]
    dEs = dEs_dict[nucleus]
    GTs = GTs_dict[nucleus]
    nucl_exc = zip(dEs, GTs)
    GT_sum = np.sum(GTs)

    for dE, GT in nucl_exc:
        Int = Int_LOS(m_chi, nucleus, dE, b, l, R_max) # [cm^-5]
        dNdgamma = GT / GT_sum # branching ratio, [unitless]
        # R = R_epsilon(epsilon, E, dE) # [MeV^-1]
        flux = ((1/24 * g_chi**2 * g_A**2 / (2*J_N + 1)) * (m_N + dE)/m_N * Int * dNdgamma * R * (c * 100) * (hbar * 1e-6 * c * 100)**2)
        fluxes.append(flux) # [MeV^-2 * cm^-5 * MeV^-1 * cm/s * cm^2 * MeV^2] = [cm^-2 s^-1 (sr^-1) MeV^-1]
        flux_tot += flux

    return flux_tot


def flux_wConv(g_chi, m_chi, nucleus, b, l, R_max, epsilon, E):
    """
    returns: differential flux in [cm^-2 s^-1 (sr^-1) MeV^-1]
    **********
    g_chi: [MeV^-1]
    E: observed photon energy [MeV]
    """
    fluxes, flux_tot = [], 0
    m_N, J_N = nuc_info[nucleus][0], nuc_info[nucleus][1]
    dEs = dEs_dict[nucleus]
    GTs = GTs_dict[nucleus]
    nucl_exc = zip(dEs, GTs)
    GT_sum = np.sum(GTs)

    for dE, GT in nucl_exc:
        Int = Int_LOS(m_chi, nucleus, dE, b, l, R_max) # [cm^-5]
        R = (np.sqrt(2 * np.pi) * epsilon * dE)**(-1) * np.exp(-(E - dE)**2 / (2 * epsilon**2 * dE**2)) # [MeV^-1]
        dNdgamma = GT / GT_sum # branching ratio, [unitless]
        # R = R_epsilon(epsilon, E, dE) # [MeV^-1]
        flux = ((1/24 * g_chi**2 * g_A**2 / (2*J_N + 1)) * (m_N + dE)/m_N * Int * dNdgamma * R * (c * 100) * (hbar * 1e-6 * c * 100)**2)
        fluxes.append(flux) # [MeV^-2 * cm^-5 * MeV^-1 * cm/s * cm^2 * MeV^2] = [cm^-2 s^-1 (sr^-1) MeV^-1]
        flux_tot += flux

    return flux_tot


def fluxData(g_chi, m_chi, nucleus, b, l, R_max, epsilon):
    """
    Calculates flux and writes to .txt file.
    """
    energy = np.linspace((m_chi - 5 * epsilon * m_chi), (m_chi + 5 * epsilon * m_chi), 21)
    flux_tot = [flux_wConv(g_chi, m_chi, nucleus, b, l, R_max, epsilon, E) for E in energy]

    dat = {'Photon energy [MeV]': energy, '{} flux [cm^-2 s^-1 sr^-1 MeV^-1]'.format(nucleus): flux_tot}
    df = pd.DataFrame(dat)
    df.to_csv('fluxData/{}/mchi_{}MeV_b_{}deg_l_{}deg_Rmax_{}kpc_epsilon_{}.txt'.format(nucleus, m_chi, b, l, R_max, 100*epsilon), sep='\t', index=False)


def plotFlux(g_chi, m_chi, nucleus, b, l, R_max, epsilon, save):
    """
    plots from .txt file
    """
    df = pd.read_csv('fluxData/{}/mchi_{}MeV_b_{}deg_l_{}deg_Rmax_{}kpc_epsilon_{}.txt'.format(nucleus, m_chi, b, l, R_max, 100*epsilon),\
                     sep='\t', \
                     names=['Photon energy [MeV]', 'Flux [cm^-2 s^-1 MeV^-1]'],\
                     skiprows=1)
    plt.figure(figsize = (12, 8))
    # plt.xscale('log'); 
    plt.yscale('log')
    plt.yticks(fontsize = 14)
    plt.grid(which = 'both', linestyle = 'dotted')
    plt.xlim(df['Photon energy [MeV]'][0], df['Photon energy [MeV]'].iat[-1])
    # plt.ylim(1e-10)
    plt.xlabel('Observed photon energy  [MeV]', fontsize = 14)
    plt.ylabel(r'$E^2 ~ d^2\Phi ~/~ dE_\gamma ~d\Omega$  [cm$^{-2}$ s$^{-1}$ sr$^{-1}$ MeV]', fontsize = 14)
    plt.title(r'{}: $m_\chi$ = {} MeV,  $g_A$ = {},  $g_\chi$ = {}, $\epsilon$ = {}%'.format(nucleus, m_chi, 1.27, g_chi, 100*epsilon), fontsize = 16)
    plt.plot(df['Photon energy [MeV]'], (df['Photon energy [MeV]'])**2 * df['Flux [cm^-2 s^-1 MeV^-1]'], color = 'k')
    plt.axvline(m_chi, color = 'red')
    plt.xticks(size = 16); plt.yticks(size=16);
    if save:
        plt.savefig('fluxFigures/mchi_{}MeV_epsilon_{}.pdf'.format(nucleus, m_chi, epsilon))


def flux_CO(g_chi, m_chi, b, l, R_max, epsilon):
    """
    computes CO flux - takes a few minutes
    ********
    epsilon: decimal from 0 to 1
    """
    energy = np.linspace((m_chi - 5 * epsilon * m_chi), (m_chi + 5 * epsilon * m_chi), 101)
    fluxes_C12 = []
    fluxes_O16 = []
    fluxes_CO = []
    for E in energy:
        flux_C12 = flux_wConv(g_chi, m_chi, 'C12', b, l, R_max, epsilon, E)
        flux_O16 = flux_wConv(g_chi, m_chi, 'O16', b, l, R_max, epsilon, E)
        fluxes_C12.append(flux_C12)
        fluxes_O16.append(flux_O16)
        fluxes_CO.append(flux_C12 + flux_O16)

    dat_C12 = {'Photon energy [MeV]': energy, 'C12 flux [cm^-2 s^-1 sr^-1 MeV^-1]': fluxes_C12}
    df_C12 = pd.DataFrame(dat_C12)
    df_C12.to_csv('fluxData/C12/mchi_{}MeV_b_{}deg_l_{}deg_Rmax_{}kpc_epsilon_{}.txt'.format(m_chi, b, l, R_max, 100*epsilon), sep='\t', index=False)

    dat_O16 = {'Photon energy [MeV]': energy, 'O16 flux [cm^-2 s^-1 sr^-1 MeV^-1]': fluxes_O16}
    df_O16 = pd.DataFrame(dat_O16)
    df_O16.to_csv('fluxData/O16/mchi_{}MeV_b_{}deg_l_{}deg_Rmax_{}kpc_epsilon_{}.txt'.format(m_chi, b, l, R_max, 100*epsilon), sep='\t', index=False)
    
    dat = {'Photon energy [MeV]': energy, 'CO flux [cm^-2 s^-1 sr^-2 MeV^-1]': fluxes_CO}
    df = pd.DataFrame(dat)
    df.to_csv('fluxData/CO/mchi_{}MeV_b_{}deg_l_{}deg_Rmax_{}kpc_epsilon_{}.txt'.format(m_chi, b, l, R_max, 100*epsilon), sep='\t', index=False)
    
    return fluxes_CO


################################################################


################ solid angle integral ################

def bIntegral(g_chi, m_chi, nucleus, l, R_max, epsilon, E):
    def f(b):
        integrand = np.sin(b * np.pi/180 + np.pi/2) * flux_wConv(g_chi, m_chi, nucleus, b, l, R_max, epsilon, E)
        return integrand 
    bees = np.linspace(-5.25, 5.25, 22)
    g = list(map(lambda bee: f(bee), bees))
    integral = cumtrapz(g, bees, initial=0)
    # integral1, err1 = quad(f, -5, 5, limit=10)
    return integral[-1] * np.pi/180


def lIntegral(g_chi, m_chi, nucleus, R_max, epsilon, E):
    def f(l):
        integrand = bIntegral(g_chi, m_chi, nucleus, l, R_max, epsilon, E)
        return integrand
    ells1 = np.linspace(330.25, ls[-1], 60)
    g1 = list(map(lambda ell1: f(ell1), ells1))
    integral1 = cumtrapz(g1, ells1, initial=0)

    ells2 = np.linspace(ls[0], 29.75, 60)
    g2 = list(map(lambda ell2: f(ell2), ells2))
    integral2 = cumtrapz(g2, ells2, initial=0)
    # integral = integral1[-1] + integral2[-1]
    return (integral1[-1] * np.pi/180, integral2[-1] * np.pi/180)


def IntSolidAngleFlux(g_chi, m_chi, nucleus, R_max, epsilon):
    energy = np.linspace((m_chi - 3 * epsilon * m_chi), (m_chi + 3 * epsilon * m_chi), 3)
    IntFluxes = []
    for E in energy:
        IntFlux1, IntFlux2 = lIntegral(g_chi, m_chi, nucleus, R_max, epsilon, E)
        IntFluxes.append(IntFlux1 + IntFlux2)
    
    filename = 'IntSolidAngleFluxData/{}/mchi_{}MeV_epsilon_{}.txt'.format(nucleus, m_chi, 100*epsilon)
    metadata = "# Integral 1 l bounds [deg] = [330.25, {}], \n\
# Integral 2 l bounds [deg] = [{}, 29.75], \n\
# b range [deg] = [-5.25, 5.25], \n\
# R_odot [kpc] = {}, \n\
# R_max [kpc] = {} kpc\n\n".format(ls[-1], ls[0], R_odot, R_max)

    with open(filename, 'w') as fp:
        fp.write(metadata)
    dat = {'Photon energy [MeV]': energy, 'Integrated flux [cm^-2 s^-1 MeV^-1]': IntFluxes, \
           'Integral 1 [cm^-2 s^-1 MeV^-1]': IntFlux1, 'Integral 2 [cm^-2 s^-1 MeV^-1]': IntFlux2 }
    df = pd.DataFrame(dat)
    df.to_csv(filename, sep='\t', index=False, mode='a') 


def IntSolidAngleFluxAllCOMasses(g_chi, R_max, epsilon):
    for m in allCOMasses:
        IntSolidAngleFlux(g_chi, m, 'C12', R_max, epsilon)
        IntSolidAngleFlux(g_chi, m, 'O16', R_max, epsilon)
        

################################################################
################################################################
################################################################
################################################################


################ test & other functions used along the way ################

#### finding optimum $m_\chi$ values ####

# Results for the optimal values of $m_\chi$ can be found in the 'optimal_m_chis_{}.txt' files.
# The functions below do not need to be run, but are included here for posterity.
# Based on some testing, the optimal values of $m_\chi$ are affected by the choice of $r$ on the order of 0.1%.

def negInt_v_N(m_chi, m_N, deltaE, r):
    return np.negative(Int_v_N(m_chi, m_N, deltaE, r))


def minimizeInt_v_N(nucleus, deltaE, r):
    # m_chi_initialGuess = 16#np.arange(deltaE - 1/10, deltaE, 1e-2)
    # print(m_chi_initialGuess)
    res = minimize_scalar(negInt_v_N, bounds=(deltaE-1/10, deltaE), \
                   args=(nuc_info[nucleus][0], deltaE, r))
    return(res.x)


def optimal_m_chis(r):
    """
    creates .txt files of optimal m_chis for all C12, O16 excitation energies 
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
        df.to_csv('optimal_m_chis_{}_r_{}kpc.txt'.format(i, r), sep='\t', index=False)
    return


################################################################


#### test: $m_\chi$ vs $\int d^3 v_N$ ####
# Finding $m_chi$ values for which the $v_N$ integral is maximized.

def Int_v_N_plot(DeltaE):
    Int_v_N_list = []
    m_chi_list = np.linspace(DeltaE - 0.1, DeltaE, 20)
    for i in m_chi_list:
        Int_v_N_list.append(Int_v_N(m_chi = i, m_N = nuc_info['O16'][0], deltaE = DeltaE, r = 3))
    plt.figure(figsize = (8, 6))
    plt.xlabel(r'$m_\chi [MeV]$', fontsize = 18); plt.ylabel(r'$\int d^3 v_b ~f_b ~\int d^3 v_\chi ~f_\chi \sigma v$', fontsize = 18)
    plt.xticks(fontsize = 14); plt.yticks(fontsize = 14)
    plt.xlim(m_chi_list[0], m_chi_list[-1])
    plt.plot(m_chi_list, Int_v_N_list)
    plt.title("O16 excitation energy: {:2.6f} MeV".format(DeltaE), fontsize = 18)
    print(m_chi_list[Int_v_N_list.index(max(Int_v_N_list))])


def v_b_test(nucleus, m_chi, deltaE, r):
    # testing to make sure a velocity in the (v_N_min, v_N_max) range returns a nonzero answer
    m_N = nuc_info[nucleus][0]
    V_ESC = v_esc(r)
    v_N_minMax = v_N_integralBounds(m_N, m_chi, deltaE, V_ESC) # [km/s]
    print((v_N_minMax[1] + v_N_minMax[0])/2)
    return Int_v_chi(m_chi, m_N, deltaE, r, (v_N_minMax[1] + v_N_minMax[0])/2)
                     #m_N, m_chi, deltaE, s, b, l, (v_N_minMax[1] - v_N_minMax[0])/2)


################################################################


#### test: distribution functions ####

def Int_f_chi(r, b, l):
    """
    integrated DM velocity distribution function - used to check normalization factor
    """
    V_ESC = v_esc(r) # [km/s]
    sigma_chi = vDispersion(r)[1] # [km/s]
    norm = norm_chi(V_ESC, sigma_chi) # [unitless]
    if norm<0:
        print("something has gone wrong!")
        return
    def f(v_chi):
        return 1/norm * (2*np.pi * sigma_chi**2)**(-3/2) * v_chi**2 * np.exp(-v_chi**2 / (2 * sigma_chi**2))
    integral = 4 * np.pi * quad(f, 0, V_ESC)[0] # [unitless]
    
    return integral


def Int_f_b(r, b, l):
    """
    integrated baryon velocity distribution function - used to check normalization factor
    """
    VBAR = v_circ(r) # [km/s]
    V_ESC = v_esc(r) # [km/s]
    # if V_ESC < VBAR:
    #     return 0
    sigma_b = vDispersion(r)[0] # [km/s]
    norm = norm_N(VBAR, V_ESC, sigma_b) # [unitless]
    if norm<0:
        print("something has gone wrong!")
        return
    
    def f(v_b):
        return 1/norm * (2*np.pi * sigma_b**2)**(-3/2) * v_b**2 * np.exp(-(v_b - VBAR)**2 / (2 * sigma_b**2))
    integral = 4 * np.pi * quad(f, 0, V_ESC)[0] # [unitless]

    return integral


def Int_f_b_LOS(s, b, l):
    """
    integrated baryon velocity distribution function as a function of line of sight - used to check normalization factor
    """
    r = R(s, b, l)
    VBAR = v_circ(r) # [km/s]
    V_ESC = v_esc(r) # [km/s]
    # if V_ESC<VBAR:
    #     return 0
    sigma_b = vDispersion(r)[0] # [km/s]
    norm = norm_N(VBAR, V_ESC, sigma_b) # [unitless]
    if norm<0:
        return 0
    
    def f(v_b):
        return 1/norm * (2*np.pi * sigma_b**2)**(-3/2) * v_b**2 * np.exp(-(v_b - VBAR)**2 / (2 * sigma_b**2))
    # print(norm)
    integral = 4 * np.pi * quad(f, 0, V_ESC)[0] # [unitless]

    return integral


def Int_f_b_noNorm(r, b, l):
    """
    integrated baryon velocity distribution function with NO NORMALIZATION
    """
    VBAR = v_circ(r) # [km/s]
    V_ESC = v_esc(r) # [km/s]
    # if V_ESC < VBAR:
    #     return 0
    sigma_b = vDispersion(r)[0] # [km/s]
    norm = 1
    
    def f(v_b):
        return 1/norm * (2*np.pi * sigma_b**2)**(-3/2) * v_b**2 * np.exp(-(v_b - VBAR)**2 / (2 * sigma_b**2))
    integral = 4 * np.pi * quad(f, 0, V_ESC)[0] # [unitless]

    return integral
 

def plotVelocityVsBaryonDistFx():
    # plotting baryon distribution function f_b(v_b) with escape velocities for various radii
    plt.figure(figsize=(8, 6))
    V_ESC = v_esc(0.02) # [km/s]
    sigma_b = vDispersion(1)[0] # use any radius, since sigma_b is currently not a function of r
    vees = np.linspace(0, V_ESC, 200)
    plt.xlim(vees[0], vees[-1])
    plt.xlabel('Baryon velocity [km/s]', fontsize=14)
    plt.ylabel('Baryon distribution function', fontsize=14)
    plt.axvline(v_circ(0.001), label=r"$r = 0.01$ kpc", color='red')
    plt.axvline(v_circ(0.01), label=r"$r = 0.01$ kpc", color='blue')
    plt.plot(vees, (2*np.pi * sigma_b**2)**(-3/2) * vees**2 * np.exp(-(vees - v_circ(0.001))**2 / (2 * sigma_b**2)), color='black')
    plt.plot(vees, (2*np.pi * sigma_b**2)**(-3/2) * vees**2 * np.exp(-(vees - v_circ(0.01))**2 / (2 * sigma_b**2)), color='black', linestyle='dashed')
    plt.plot(vees, (2*np.pi * sigma_b**2)**(-3/2) * vees**2 * np.exp(-(vees - v_circ(0.02))**2 / (2 * sigma_b**2)), color='black', linestyle='dotted')
    legend = plt.legend(loc=2)
    legend.set_title(r'escape velocity')


def plotRadiusVsBaryonDistFx():
    arrs = np.linspace(0.001,0.1, 300)
    norms = list(map(lambda arr: norm_N(v_circ(arr), v_esc(arr), sigma_b), arrs))
    Int_fs = list(map(lambda arr: Int_f_b(arr, 0, 0.25), arrs))
    
    fig, ax1 = plt.subplots(figsize = (8, 6))
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax1.set_xlabel(r'Radius [kpc]', fontsize = 16)
    ax1.set_ylabel(r'Normalization factor', fontsize = 16, color='red')
    ax2.set_ylabel(r'Distribution function integral', fontsize = 16)  # we already handled the x-label with ax1
    ax2.set_xlim(arrs[0], arrs[-1])
    ax2.set_ylim(0,1.1)
    ax1.plot(arrs, np.zeros(len(arrs)), color='grey')
    ax1.plot(arrs, norms, label="Normalization".format(bb, ll), color='red')
    ax2.plot(arrs, Int_fs, label = 'Distribution function integral', color='black')
    ax1.plot(arrs, np.ones(len(arrs)), color='grey')
    # ax1.legend()
    # fig.tight_layout()  # otherwise the right y-label is slightly clipped
    # plt.title('baryon density as function of line of sight', fontsize = 24);


def plotEscapeVelocityVsBaryonDistFx():
    plt.figure(figsize=(8, 6))
    VBAR = v_circ(1) # [km/s]
    # vbar # [km/s]
    sigma_b = vDispersion(1)[0] # use any radius, since sigma_b is currently not a function of r
    vees = np.linspace(0, 300, 300)
    plt.xlim(vees[0], vees[-1])
    plt.xlabel('Baryon velocity [km/s]', fontsize=14)
    plt.ylabel('Baryon distribution function', fontsize=14)
    plt.axvline(v_esc(0.01), label=r"$r = 0.01$ kpc", color='red')
    plt.axvline(v_esc(0.012), label=r"$r = 0.012$ kpc", color='green')
    plt.axvline(v_esc(0.015), label=r"$r = 0.015$ kpc", color='blue')
    plt.plot(vees, (2*np.pi * sigma_b**2)**(-3/2) * vees**2 * np.exp(-(vees - VBAR)**2 / (2 * sigma_b**2)), color='black')
    legend = plt.legend(loc=2)
    legend.set_title(r'escape velocity')


def plotRadiusVsNorm():
    arrs = np.linspace(0.01,10,30)
    VBAR = v_circ(arrs) # [km/s]
    # vbar # [km/s]
    v_escs, sigma_bs, norm_bs, Int_f_bs, Int_f_cs = [], [], [], [], []
    
    for i in arrs:
        V_ESC = v_esc(i)
        v_escs.append(V_ESC)
        sigma_b = vDispersion(i)[0]
        norm = norm_N(VBAR, V_ESC, sigma_b)
        norm_bs.append(norm)
        Int_f = Int_f_b(i, 0, l_crval)
        Int_f_bs.append(Int_f)
        Int_f_c = Int_f_chi(i, 0, l_crval)
        Int_f_cs.append(Int_f_c)
    
    plt.figure(figsize=(8, 6))
    plt.xlim(arrs[0], arrs[-1])
    plt.ylim(0.75, 1.25); #plt.yscale('log')
    plt.xlabel('Radius [kpc]', fontsize=14)
    plt.ylabel(r'Distribution function $f(v)$', fontsize=14)
    plt.plot(arrs, np.ones(len(arrs)), color='black')
    plt.plot(arrs, Int_f_bs, label=r'$f_b(r)$')
    plt.plot(arrs, Int_f_cs, label=r'$f_\chi(r)$')
    plt.legend(fontsize=12)


def plotEscapeVelocityVsNorm():
    plt.figure(figsize=(8, 6))
    plt.xlabel('Escape velocity [km/s]', fontsize=14)
    plt.ylabel(r'Distribution function $f(v)$', fontsize=14)
    plt.xlim(v_escs[0], v_escs[-1])
    plt.ylim(0.75, 1.25)
    plt.plot(v_escs, Int_f_bs)
    # plt.legend(fontsize=12)


################################################################


#### test: effective baryon-DM J-factor ####

def Jfactor_chi(b, l):
    # https://arxiv.org/pdf/1012.4515.pdf
    r_odot = R_odot # 8.33 # [kpc] pg 33
    rho_odot = rho_s # 0.3 # [GeV/cm^3] pg 33

    def f(s):
        r = R(s, b, l)
        rho_chi = density('C12', r, b, l)[1] # [GeV/cm^3]
        integrand = (1/r_odot) * rho_chi / rho_odot
        return integrand

    return quad(f, 0, 10)[0] # [km^3 s^-3 MeV^-2]


def Jfactor_b(b, l):
    # https://arxiv.org/pdf/1012.4515.pdf
    r_odot = R_odot # 8.33 # [kpc] pg 33
    rho_odot = rho_s # 0.3 # [GeV/cm^3] pg 33

    def f(s):
        r = R(s, b, l)
        rho_b = density('C12', r, b, l)[0] # [GeV/cm^3]
        integrand = (1/r_odot) * rho_b / rho_odot
        return integrand

    return quad(f, 0, 10, limit=100)[0] # [km^3 s^-3 MeV^-2]


def plotEffectiveJfactorDensityComponents():
    """
    Plots the baryon and DM density components of the "effective baryon-DM J-factor" separately to see their individual behaviour.
    """
    b = 0
    plt.figure(figsize = (12, 8))
    plt.xlabel(r'$\psi$ [degrees]', fontsize = 16)
    plt.ylabel(r'$J(\psi)$', fontsize = 16)
    plt.xlim(0, 180)
    ells_b = np.linspace(l_crval, ls[-1], 20)
    ells = np.linspace(l_crval, ls[-1], 100)
    Jfactors_chi = list(map(lambda ell: Jfactor_chi(b, ell), ells))
    Jfactors_b = list(map(lambda ell: Jfactor_b(b, ell), ells_b))
    plt.yscale('log')
    plt.plot(ells, Jfactors_chi, label=r'$\chi$')
    plt.plot(ells_b, Jfactors_b, label = r'b')
    plt.legend(fontsize = 16)
    plt.title('Effective baryon-DM J-factor density components', fontsize=20);
    # plt.savefig('effectiveJ-factorDensityComponents.pdf')
    return


def Jfactor_chi_b(b, l):
    S_MAX = s_max(b, l, 2 * R_odot) # [kpc]
    rho_chi_odot = 0.3 # [GeV/cm^3]
    rho_b_odot = 2 # [GeV/cm^3] approximate average value chosen from baryon density plot above
    r_odot = R_odot
    
    def f(s):
        r = R(s, b, l)
        DM_integral = Int_f_chi(s, b, l) # [unitless]
        baryon_integral = Int_f_b(s, b, l) # [unitless]
        rho = density('O16', r, b, l) # [GeV/cm^3]
        rho_b, rho_chi = rho[0], rho[1] # [GeV/cm^3], [GeV/cm^3]
        integrand = rho_chi / rho_chi_odot * DM_integral * rho_b / rho_b_odot * baryon_integral # [unitless]
        return integrand
    return 1/r_odot * quad(f, 0, S_MAX, limit=100)[0] # [unitless]


def plotEffectiveJfactor():
    """
    Plots an "effective baryon-DM J factor" of baryon densty times DM density.
    """
    ells = np.linspace(l_crval, ls[-1], 100)
    J_factor_list = []
    for i in ells:
        J_factor_list.append(Jfactor_chi_b(0.25, i))
    plt.figure(figsize = (12, 8))
    plt.xlabel(r'$\psi$ [degrees]', fontsize = 16); plt.ylabel(r'$J(\psi)$', fontsize = 16)
    plt.xlim(0, 180)
    plt.yscale('log')
    plt.plot(ells, J_factor_list, label = r'$b \approx 0, \psi \approx l$')
    plt.title('Effective baryon-DM J-factor', fontsize=20);
    # plt.savefig('effectiveBaryonDMJ-factor.pdf')
    return


################################################################


#### test: normalization of $dN/dE_\gamma$ ####

def dNdE_gamma(g_chi, nucleus):
    """
    returns: sum of dNdgamma weighted by their GT strengths (should equal 1)
    **********
    g_chi: [MeV^-1]
    g_A: axial form factor [unitless]
    """
    # nuc_info = {'He4': [3758.26, 0.0], 'C12': [11274.78, 0.0], 'N14': [13153.91, 0.0], 'O16': [15033.04, 0.0]}
    m_N, J_N = nuc_info[nucleus][0], nuc_info[nucleus][1]
    dEs, GTs = dEs_dict[nucleus], GTs_dict[nucleus]
    nucl_exc = zip(dEs, GTs)
    GT_sum = np.sum(GTs)
    dNdgamma = 0

    for dE, GT in nucl_exc:
        dNdgamma += GT / GT_sum # branching ratio, [unitless]

    return dNdgamma # [cm^-2 s^-1 MeV^-1]


################################################################


if __name__=="__main__":
    x = Int_v_chi(16.73, nuc_info['O16'][0], 16.732241, 16, 163.5)
    print(x)
