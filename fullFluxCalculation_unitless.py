#!/usr/bin/env python
# coding: utf-8

# # Jan 2

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
from astropy.wcs import WCS
from astropy import constants as ct
from astropy import units as u
# from spectral_cube import SpectralCube
import bisect
import time
from pathlib import Path


# # constants & data import

# constants
g_A = 1.27 # [unitless] axial form factor
################
R_odot = 8.5 # [kpc] Sun's distance from centre of MW, from Clemens 1985
rho_s = 0.51 # [GeV/cm^3] DM radial scale for MW NFW profile; https://arxiv.org/abs/1906.08419
r_s = 8.1 # [kpc] DM radial scale for MW NFW profile; https://arxiv.org/abs/1906.08419; also used as distance scale for creating unitless quantities
################
# M_bulge = 1.5e10 * ct.M_sun.to_value(u.kg) # [kg]
# c_bulge = 0.6 # [kpc] bulge scale radius
# M_disk = 7e10 * ct.M_sun.to_value(u.kg) # [kg]
# b_disk = 4 # [kpc] disk scale radius
################
energyScale = 10 # [MeV] energy scale for creating unitless quantities

# importing C12 and O16 data: excitation energies [MeV] and GT strengths [unitless]
C12_data = 'C12_dEs_GTs.txt'
O16_data = 'O16_dEs_GTs.txt'
C12df = pd.read_csv(C12_data, sep='\t', names=['dE [MeV]', 'GT'], skiprows=1)
O16df = pd.read_csv(O16_data, sep='\t', names=['dE [MeV]', 'GT'], skiprows=1)

# nuclear info dictionary
nuc_dict = {'C12': {'mass [MeV]': 11274.78, 'mass [unitless]': 11274.78/energyScale, 'spin': 0.0, 'dEs [MeV]': C12df['dE [MeV]'], 'GTs': C12df['GT']}, \
            'O16': {'mass [MeV]': 15033.04, 'mass [unitless]': 15033.04/energyScale, 'spin': 0.0, 'dEs [MeV]': O16df['dE [MeV]'], 'GTs': O16df['GT']}}
# 'He4': {'mass_MeV': 3758.26, 'mass [unitless]': 3758.26/energyScale, 'spin': 0.0}
# 'N14': {'mass_MeV': 13153.91, 'mass [unitless]': 13153.91/energyScale, 'spin': 0.0}


# # GALPROP density data

# galprop manual: 
# - https://galprop.stanford.edu/download/manuals/galprop_v54.pdf
# 
# handling .fits files: 
# - https://fits.gsfc.nasa.gov/users_guide/usersguide.pdf
# - https://astropy4cambridge.readthedocs.io/en/latest/_static/Astropy%20-%20Handling%20FITS%20files.html

hdul_CO = fits.open('massdensity_CO.fits')

def print_density_header():
    """
    prints header for CO density data
    """
    return hdul_CO[0].header # .fits file header
    # return hdul_CO[1].data # radial bins

# print_density_header()


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
density_data = hdul_CO[0].data # [kg/cm^3]
density_data_unitless = density_data / ((rho_s * u.GeV * (1/u.cm)**3).to_value(u.kg * (1/u.cm)**3, u.mass_energy())) # [unitless, in units of rho_s]

bins = hdul_CO[1].data # [kpc] radial bin boundaries
r_bins = [bins[0][0]] # r bin smallest value
r_unitless_bins = [bins[0][0] / r_s] # r bin smallest value
for i, j in bins: # creating r bins from GALPROP increments
    r_bins.append(j)
    r_unitless_bins.append(j / r_s)
# r_bins
# r_unitless_bins


# density_data[4]


# creating b and l arrays
b_len, l_len = hdul_CO[0].header['NAXIS2'], hdul_CO[0].header['NAXIS1'] # length of b, l arrays
b_crval, l_crval = hdul_CO[0].header['CRVAL2'], hdul_CO[0].header['CRVAL1'] # [deg], [deg]; central values of b,l
b_delta, l_delta = hdul_CO[0].header['CDELT2'], hdul_CO[0].header['CDELT1'] # [deg], [deg]; increments for b, l

bs = list(np.arange(b_crval, b_crval + b_len*b_delta, b_delta)) # [deg]
ls = list(np.arange(l_crval, l_crval + l_len*l_delta, l_delta)) # [deg]


# def plot_CO_density(r, save):
#     """
#     plots CO density for radial bin corresponding to galactocentric radius r
#     ****************
#     r: [kpc] galactocentric radius
#     """
#     r_index = bisect.bisect(r_bins, r) - 1 # find radial bin index
#     fig = plt.figure(figsize = (10,15))
#     ax = plt.axes()
#     im = ax.imshow(density_data[r_index], cmap='magma', norm=matplotlib.colors.LogNorm())
#     X, Y = 0.02, 0.04
#     cax = fig.add_axes([ax.get_position().x1 + X, ax.get_position().y0, Y, ax.get_position().height])
#     if save: 
#         plt.savefig('plots/CO_density_r_bin_{}'.format(r_index), bbox_inches="tight")
#     plt.xlabel('Galactic longitude')
#     plt.colorbar(im, cax=cax)

# plot_CO_density(9, False)


def plot_CO_density(r, save):
    """
    plots CO density for radial bin corresponding to galactocentric radius r
    ****************
    r: [kpc] galactocentric radius
    """
    r_index = bisect.bisect(r_bins, r) - 1 # find radial bin index
    fig = plt.figure(figsize=(10,5))

    wcs = WCS(hdul_CO[0].header)
    wcs_2d = wcs[r_index, :, :]
    ax = plt.subplot(projection=wcs_2d)
    im = ax.imshow(density_data[r_index], cmap='magma', norm=matplotlib.colors.LogNorm())
    ax.set_xlabel('Galactic longitude', fontsize = 14)
    ax.set_ylabel('Galactic latitude', fontsize = 14)
    ax.tick_params(axis="both", labelsize=14) 
    
    X, Y = 0.01, 0.02
    cax = fig.add_axes([ax.get_position().x1 + X, ax.get_position().y0, Y, ax.get_position().height])
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label(r'Density [kg/cm$^3$]', size=14)
    cbar.ax.tick_params(labelsize=14)

    if save: 
        plt.savefig('plots/CO_density_r_{}kpc_bin_{}.pdf'.format(r, r_index), bbox_inches="tight")

# plot_CO_density(8.5, False)





# # rotation curve from Clemens (1985)

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


def plot_rotation_curves():
    """
    plots rotation curves from Clemens 1985, used for circular baryon velocity
    """
    plt.figure(figsize=(12, 8))
    plt.grid(which = 'both', linestyle = 'dotted')
    plt.xlim(-1, D_R[-1]); plt.ylim(0, 275)
    plt.xlabel('Radius from galactic centre [kpc]', fontsize=14)
    plt.ylabel('Rotation speed [km/s]', fontsize=14)
    plt.plot(A_R, theta_A)
    plt.plot(B_R, theta_B)
    plt.plot(C_R, theta_C)
    plt.plot(D_R, theta_D)
    plt.plot(r_rot, v_rot, color = 'black', linestyle = 'dashed')

# plot_rotation_curves()


# # radius & maximum line of sight distance


def plot_radius_vs_los(b, R_max):
    """
    plots galactic radius as a function of line of sight
    ****************
    b: [deg] galactic latitude
    R_max: [kpc] maximum galactic radius
    """
    # R_max = 50.0 # [kpc]
    plt.figure(figsize = (12, 8))
    plt.grid(which = 'both', linestyle = 'dotted')
    plt.xticks(size = 16); plt.yticks(size=16);
    plt.xlabel(r'Line of sight distance [kpc]', fontsize = 16); plt.ylabel(r'Radius [kpc]', fontsize = 16)
    plt.xlim(0, R_max)
    for l in [0, 5, 15, 30, 90]:
        psi = np.arccos( np.cos(np.radians(b)) * np.cos(np.radians(l)) ) # [radians]
        s_max = (np.sqrt(R_max * R_max - (R_odot) * (R_odot) * np.sin(psi) * np.sin(psi)) + (R_odot) * np.cos(psi)) # [kpc]
        esses = np.linspace(0, s_max, 400) # [kpc]
        # r = np.sqrt((R_odot/r_s) * (R_odot/r_s) + s**2 - 2 * (R_odot/r_s) * s * np.cos(psi)) # [unitless, in units of r_s]
        arrs = list(map(lambda s: np.sqrt((R_odot) * (R_odot) + s**2 - 2 * (R_odot) * s * np.cos(psi)), esses)) # [kpc]
        plt.plot(esses, arrs, label="(l, b) = ({}$^\circ$, {}$^\circ$)".format(l, b))
    plt.plot()
    plt.legend(fontsize=14)
    plt.title('radial coordinate as function of line of sight', fontsize = 20)

# plot_radius_vs_los(b = 0, R_max = 50.0)


# # density calculation & interpolation for DM and baryons


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
    rho_DM = rho_s / ( (r/r_s) * (1 + r/r_s)**2 ) # [GeV/cm^3]
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
    rho_DM = 1 / ( (r) * (1 + r)**2 ) # [unitless]
    # print(rho_DM * (rho_s * u.GeV * (1/u.cm)**3).to_value())
    if r >= r_unitless_bins[-1]: # if radius is beyond what is provided by GALPROP, return zero for baryon density
        rho_b = 0
        return(rho_b, rho_DM)

    rho_b = (nuc_dict[nucleus]['mass [unitless]'] / (nuc_dict['C12']['mass [unitless]'] + nuc_dict['O16']['mass [unitless]']) \
             * density_unitless_interpolator[r_unitless_index](np.array([b, l])))[0]
    return (rho_b, rho_DM)
        

def plot_DM_density_vs_los(nucleus, b):
    """
    plots DM density as function of line of sight
    ****************
    nucleus: 'C12' or 'O16'
    b: [deg] galactic latitude
    """
    R_max = 40.0
    plt.figure(figsize = (12, 8))
    plt.grid(which = 'both', linestyle = 'dotted')
    plt.xlabel(r'Line of sight distance [kpc]', fontsize = 16)
    plt.ylabel(r'DM density [GeV/cm$^3$]', fontsize = 16)
    plt.xlim(0, R_max)
    for l in [5.25, 15.25, 30.25, 90.25]:#, 45]:
        psi = np.arccos( np.cos(np.radians(b)) * np.cos(np.radians(l)) ) # [radians]
        s_max = (np.sqrt(R_max * R_max - (R_odot) * (R_odot) * np.sin(psi) * np.sin(psi)) + (R_odot) * np.cos(psi)) # [kpc]
        esses = np.linspace(0, s_max, 400) # [kpc]
        # r = np.sqrt((R_odot/r_s) * (R_odot/r_s) + s**2 - 2 * (R_odot/r_s) * s * np.cos(psi)) # [unitless, in units of r_s]
        rho_DM = list(map(lambda s: density(nucleus, np.sqrt((R_odot) * (R_odot) + s**2 - 2 * (R_odot) * s * np.cos(psi)), b, l)[1], esses))
        plt.plot(esses, rho_DM, label="(l, b) = ({}$^\circ$, {}$^\circ$)".format(l, b))
    plt.plot()
    plt.legend()
    plt.title('dm density as function of line of sight', fontsize = 24)

# plot_DM_density_vs_los(nucleus = 'C12', b = 0.25)


def plot_baryon_density_vs_los(nucleus, b):
    """
    plots baryon density as function of line of sight
    ****************
    nucleus: 'C12' or 'O16'
    b: [deg] galactic latitude
    """
    R_max = 50.0
    fig, ax1 = plt.subplots(figsize = (12, 8))
    ax2 = ax1.twinx()  # create a second axes that shares the same x-axis
    ax1.set_xlabel(r'Line of sight distance [kpc]', fontsize = 16)
    ax1.set_ylabel(r'Baryon density [kg/cm$^3$]', fontsize = 16)
    ax2.set_ylabel(r'Baryon density [GeV/cm$^3$]', fontsize = 16)  # we already handled the x-label with ax1
    ax2.set_xlim(0,35)
    for l in [5.25, 15.25, 30.25, 90.25]:#, 45]:
        psi = np.arccos( np.cos(np.radians(b)) * np.cos(np.radians(l)) ) # [radians]
        s_max = (np.sqrt(R_max * R_max - (R_odot) * (R_odot) * np.sin(psi) * np.sin(psi)) + (R_odot) * np.cos(psi)) # [kpc]
        esses = np.linspace(0, s_max, 400) # [kpc]
        # r = np.sqrt((R_odot/r_s) * (R_odot/r_s) + s**2 - 2 * (R_odot/r_s) * s * np.cos(psi)) # [unitless, in units of r_s]
        rho_b_GeV = list(map(lambda s: density(nucleus, np.sqrt((R_odot) * (R_odot) + s**2 - 2 * (R_odot) * s * np.cos(psi)), b, l)[0], esses))
        rho_b_kg = list(map(lambda s: density(nucleus, np.sqrt((R_odot) * (R_odot) + s**2 - 2 * (R_odot) * s * np.cos(psi)), b, l)[0] \
                            * 1/(u.kg * (1/u.cm)**3).to_value(u.GeV * (1/u.cm)**3, u.mass_energy()), esses))                     
        ax2.plot(esses, rho_b_GeV, label="(l, b) = ({}$^\circ$, {}$^\circ$)".format(l, b), linestyle='dashed', color='black')
        ax1.plot(esses, rho_b_kg, label="(l, b) = ({}$^\circ$, {}$^\circ$)".format(l, b))
    plt.plot()
    ax1.legend()
    plt.title('dm density as function of line of sight', fontsize = 24)

# plot_baryon_density_vs_los(nucleus = 'C12', b = 0)


def plot_baryon_density_vs_radius(nucleus, b):
    """
    plots baryon density as function of radius from galactic centre
    ****************
    nucleus: 'C12' or 'O16'
    b: [deg] galactic latitude
    """
    R_max = 50.0
    fig, ax1 = plt.subplots(figsize = (12, 8))
    ax2 = ax1.twinx()  # create a second axes that shares the same x-axis
    ax1.set_xlabel(r'Radius [kpc]', fontsize = 16)
    ax1.set_ylabel(r'Baryon density [kg/cm$^3$]', fontsize = 16)
    ax2.set_ylabel(r'Baryon density [GeV/cm$^3$]', fontsize = 16)  # we already handled the x-label with ax1
    ax2.set_xlim(0,35)
    
    for l in [5.25, 15.25, 30.25, 90.25]:#, 45]:
        arrs = np.linspace(1e-3, R_max, 400)
        rho_b_kg = list(map(lambda arr:density(nucleus, arr, b, l)[0] * 1/(u.kg * (1/u.cm)**3).to_value(u.GeV * (1/u.cm)**3, u.mass_energy()), arrs))
        rho_b_GeV = list(map(lambda arr:density(nucleus, arr, b, l)[0], arrs))
        ax1.plot(arrs, rho_b_kg, label="(l, b) = ({}$^\circ$, {}$^\circ$)".format(l, b))
        ax2.plot(arrs, rho_b_GeV, label="(l, b) = ({}$^\circ$, {}$^\circ$)".format(l, b), color='black', linestyle='')
        ax1.legend()
    
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title('baryon density as function of radius from GC', fontsize = 24);

# plot_baryon_density_vs_radius(nucleus = 'C12', b = 0.25)


# # velocity dispersions $\sigma(r, \psi)$

def velocity_dispersions_unitless(r):
    """
    returns: [unitless, in units of c] (baryon velocity dispersion, DM velocity dispersion)
    ****************
    r: [unitless, in units of r_s] radius from galactic centre 
    ****************
    DM dispersion function taken from 2111.03076 Figure 1 for r <~ 50 kpc using automeris.io (Distance 76, Delta X = 4 Px, Delta Y = 4 Px)
    """
    disp_b = 10 / (ct.c.to_value(u.km / u.s)) # [unitless, in units of c]
    disp_DM = (-42.9 * np.log(r) + 288.0) / (ct.c.to_value(u.km / u.s)) # [unitless, in units of c]

    return (disp_b, disp_DM)
        


def plot_escape_velocity():
    """
    plots escape velocity as function of radius from galactic centre
    ****************
    """
    plt.figure(figsize=(8, 6))
    plt.grid(which = 'both', linestyle = 'dotted')
    arrs = np.arange(r_rot[0], r_rot[-1], dr)
    plt.xlim(-1, r_rot[-1]); plt.ylim(np.sqrt(2) * v_circ_unitless(arrs[0]/r_s) * ct.c.to_value(u.km / u.s), np.sqrt(2) * 275)
    plt.xlabel('Radius from galactic centre [kpc]', fontsize=14)
    plt.ylabel('Escape velocity [km/s]', fontsize=14)
    plt.plot(arrs, list(map(lambda r: np.sqrt(2) * v_circ_unitless(r/r_s) * ct.c.to_value(u.km / u.s), arrs)), color='black', label='escape velocity')
    plt.plot(r_rot, v_rot, color='black', linestyle='dashed', label='rotation curve')
    plt.legend()
    
# plot_escape_velocity()


# # $v_\chi$ integral

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
    return norm


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

    return ( (1 / norm) * (1 / E_chi) * v_chi_plus * np.exp(-v_chi_plus**2 / (2 * sigma_chi**2)) ) # [unitless]

vchiIntegralFactor = (ct.c.to_value(u.km / u.s))/energyScale
# vchiIntegralFactor


# # $v_n$ integral

def v_n_integral_bounds(m_chi, m_n, dE, V_ESC):
    """
    returns: [unitless, in units of c] min and max bounds on v_n for baryon distribution function integral
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
    return norm


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
# vnIntegralFactor


# # line of sight integral

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
# losIntegralFactor


# # flux

def diff_flux_w_convolution_for_single_dE(g_chi, m_chi, nucleus, m_n, J_n, dE, GT, b, l, R_max, epsilon, E):
    """
    returns: [unitless] differential flux;
    to get unitful dimensions [cm^-2 s^-1 MeV^-1 sr^-1] multiply by fluxFactor:
        ( 1/energyScale * 1/energyScale \
         * losIntegralFactor \
         * 1/energyScale \
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
    epsilon: [unitless] energy resolution
    E: [unitless, in units of energyScale] observed photon energy
    """
    fluxes, flux_tot = [], 0
    
    # do some checks to save time:
    if m_chi > dE:
        print("m_chi > dE")
        # flux_tot = 0
        return flux_tot
    
    expTerm = np.exp(-(E - dE)**2 / (2 * epsilon**2 * dE**2))
    if expTerm == 0:
        print("expTerm == 0")
        # flux_tot = 0
        return flux_tot
    
    los_integral = line_of_sight_integral(m_chi, nucleus, m_n, dE, b, l, R_max) # [unitless] 
    # multiply los_integral by losIntegralFactor to get [cm^-5]
    if los_integral == 0:
        print("los == 0")
        # flux_tot = 0
        return flux_tot

    R = (np.sqrt(2 * np.pi) * epsilon * dE)**(-1) * expTerm # [unitless], needs to be multiplied by 1/energyScale to get [MeV^-1]
    flux = ((1/24 * g_chi * g_chi * g_A * g_A / (2*J_n + 1)) * (m_n + dE)/m_n * los_integral * GT * R) # [unitless]
    fluxes.append(flux)
    flux_tot += flux

    return flux_tot # [unitless]


fluxFactor = ( 1/energyScale * 1/energyScale \
              * losIntegralFactor \
              * 1/energyScale \
              * (ct.c.to_value(u.cm/u.s)) * (ct.hbar.to_value(u.MeV * u.s) * ct.c.to_value(u.cm/u.s))**2 )
# fluxFactor


def b_integral(g_chi, m_chi, nucleus, m_n, J_n, dE, GT, b_min, b_max, l, R_max, epsilon, E):
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
    l: [deg] galactic longitude
    R_max: [unitless, in units of r_s] maximum galactic radius
    epsilon: [unitless] energy resolution
    E: [unitless, in units of energyScale] observed photon energy
    """
    diffFlux = []

    def f(b):
        # b: [deg]
        #print("        b = {} deg".format(b))
        x = diff_flux_w_convolution_for_single_dE(g_chi, m_chi, nucleus, m_n, J_n, dE, GT, b, l, R_max, epsilon, E) # [unitless]
        #print("            differential flux [cm^-2 s^-1 MeV^-1 sr^-1]: {}".format(x * fluxFactor))
        diffFlux.append(x * fluxFactor)
        integrand = np.sin(b * np.pi/180 + np.pi/2) * x  # [unitless]
        # sine argument accounts for transformation from GALPROP convention to standard solid angle convention
        return integrand # [unitless]
    bees = np.linspace(b_min, b_max, 20) # 20; testing: use 3
    g = list(map(lambda bee: f(bee), bees))
    #tic = time.time()
    integral = cumtrapz(g, bees, initial=0) # [deg]
    #print("        time for b integral: {}".format(time.time() - tic))

    #### write differential flux to file for each value of b ####
    Path("data/differential_flux_data_by_dE/{}/dE_{}MeV/m_chi_{}MeV/E_obs_{}MeV".format(nucleus, dE*energyScale, m_chi*energyScale, E*energyScale)).mkdir(parents=True, exist_ok=True)
    filename = 'data/differential_flux_data_by_dE/{}/dE_{}MeV/m_chi_{}MeV/E_obs_{}MeV/l_{}deg_epsilon_{}.txt'.format(nucleus, dE*energyScale, m_chi*energyScale, E*energyScale, l, 100*epsilon)
    metadata = "# R_odot [kpc] = {}, \n\
# R_max [kpc] = {} kpc\n\
# rho_s [GeV cm^-3] = {}\n\
# r_s [kpc] = {}\n\
# g_chi [MeV^-1] = {}\n\
# g_A = {}\n\
# GT = {}\n#\n".format(R_odot, R_max*r_s, rho_s, r_s, g_chi/energyScale, g_A, GT)
    with open(filename, 'w') as fp:
        fp.write(metadata)
    dat = {'b [deg]': bees, 'Differential flux [cm^-2 s^-1 MeV^-1 sr^-1]': diffFlux}
    df = pd.DataFrame(dat)
    df.to_csv(filename, sep='\t', index=False, mode='a')
    ################
    
    return integral[-1] * np.pi/180 # [deg] -> [rad]


def l_integral(g_chi, m_chi, nucleus, m_n, J_n, dE, GT, b_min, b_max, l_min1, l_max1, l_min2, l_max2, R_max, epsilon, E):
    """
    returns: [unitless * rad * rad] (flux1, flux2) integrated over galactic latitude from b_min to b_max, and:
        flux1: galactic longitude from l_min1 to l_max1
        flux2: galactic longitude from l_min2 to l_max2 
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
    epsilon: [unitless] energy resolution
    E: [unitless, in units of energyScale] observed photon energy
    """
    bInt = []
    def f(l):
        # l: [deg]
        #print("    l = {} deg".format(l))
        integrand = b_integral(g_chi, m_chi, nucleus, m_n, J_n, dE, GT, b_min, b_max, l, R_max, epsilon, E) # [rad]
        bInt.append(integrand * fluxFactor)
        return integrand

    ells1 = np.linspace(l_min1, l_max1, 60) # 60; testing: use 3
    tic = time.time()
    g1 = list(map(lambda ell1: f(ell1), ells1))
    integral1 = cumtrapz(g1, ells1, initial=0) # [rad] * [deg]
    print("        time for first l integral: {}".format(time.time() - tic))

    ells2 = np.linspace(l_min2, l_max2, 60) # 60; testing: use 3
    tic = time.time()
    g2 = list(map(lambda ell2: f(ell2), ells2))
    integral2 = cumtrapz(g2, ells2, initial=0) # [rad] * [deg]
    print("        time for second l integral: {}".format(time.time() - tic))
    # integral = integral1[-1] + integral2[-1]

    #### write differential flux (integrated over b) to file for each value of l ####
    Path("data/b_integral_data_by_dE/{}/dE_{}MeV/m_chi_{}MeV/E_photon_{}MeV".format(nucleus, dE*energyScale, m_chi*energyScale, E*energyScale)).mkdir(parents=True, exist_ok=True)
    filename = 'data/b_integral_data_by_dE/{}/dE_{}MeV/m_chi_{}MeV/E_photon_{}MeV/epsilon_{}.txt'.format(nucleus, dE*energyScale, m_chi*energyScale, E*energyScale, 100*epsilon)
    metadata = "# b range [deg] = [{}, {}], \n\
# R_odot [kpc] = {}, \n\
# R_max [kpc] = {} kpc\n\
# rho_s [GeV cm^-3] = {}\n\
# r_s [kpc] = {}\n\
# g_chi [MeV^-1] = {}\n\
# g_A = {}\n\
# GT = {}\n#\n".format(b_min, b_max, R_odot, R_max*r_s, rho_s, r_s, g_chi/energyScale, g_A, GT)
    with open(filename, 'w') as fp:
        fp.write(metadata)
    dat = {'l [deg]': [*ells1, *ells2], 'Differential flux [cm^-2 s^-1 MeV^-1 rad^-1]': bInt}
    df = pd.DataFrame(dat)
    df.to_csv(filename, sep='\t', index=False, mode='a')
    ################
    
    return (integral1[-1] * np.pi/180, integral2[-1] * np.pi/180) # ([rad * rad], [rad * rad])


def solid_angle_integral(g_chi, m_chi, nucleus, m_n, J_n, dE, GT, b_min, b_max, l_min1, l_max1, l_min2, l_max2, integratedSolidAngle, R_max, epsilon):
    """
    returns: [unitless] solid-angle-averaged flux integrated over:
        galactic latitude from b_min to b_max
        galactic longitude from l_min1 to l_max1
        galactic longitude from l_min2 to l_max2 
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
    epsilon: [unitless] energy resolution
    """
    
    # observed photon energy spectrum; currently using 1 sigma:
    energy = np.linspace((dE - 1 * epsilon * dE), (dE + 1 * epsilon * dE), 11) # [unitless, in units of energyScale]
    # for testing:
    # [nuc_dict['C12']['dEs [MeV]'][0]/energyScale] # [unitless, in units of energyScale]
    energyMeV = [EE * energyScale for EE in energy] # [MeV] used for datafile
    
    integratedFlux1, integratedFlux2 = [], []
    integratedFlux = []
    
    for E in energy: # [unitless, in units of energyScale]
        tic = time.time()
        print("    observed photon energy = {} MeV".format(E*energyScale))
        intFlux1, intFlux2 = l_integral(g_chi, m_chi, nucleus, m_n, J_n, dE, GT, \
                                        b_min, b_max, l_min1, l_max1, l_min2, l_max2, R_max, epsilon, E) # [rad * rad], [rad * rad]
        integratedFlux1.append(intFlux1 * fluxFactor) # [rad * rad] * [cm^-2 s^-1 MeV^-1 sr^-1] = [cm^-2 s^-1 MeV^-1]
        integratedFlux2.append(intFlux2 * fluxFactor) # [rad * rad] * [cm^-2 s^-1 MeV^-1 sr^-1] = [cm^-2 s^-1 MeV^-1]
        integratedFlux.append((intFlux1 + intFlux2) * fluxFactor) # [cm^-2 s^-1 MeV^-1]
        print("    time for one obs photon energy: {}\n".format(time.time() - tic))

    #### calculate solid-angle-averaged-flux ####
    averagedFlux = [i / integratedSolidAngle for i in integratedFlux] # [cm^-2 s^-1 MeV^-1 sr^-1]
    ################
    
    #### write to file ####
    Path("data/solid_angle_integral_data_by_dE/{}/dE_{}MeV".format(nucleus, dE*energyScale)).mkdir(parents=True, exist_ok=True)
    filename = 'data/solid_angle_integral_data_by_dE/{}/dE_{}MeV/m_chi_{}MeV_epsilon_{}.txt'.format(nucleus, dE*energyScale, m_chi*energyScale, 100*epsilon)
    metadata = "# Integral 1 l bounds [deg] = [{}, {}], \n\
# Integral 2 l bounds [deg] = [{}, {}], \n\
# b range [deg] = [{}, {}], \n\
# R_odot [kpc] = {}, \n\
# R_max [kpc] = {} kpc\n\
# rho_s [GeV cm^-3] = {}\n\
# r_s [kpc] = {}\n\
# g_chi [MeV^-1] = {}\n\
# g_A = {}\n\
# GT = {}\n#\n".format(l_min1, l_max1, l_min2, l_max2, b_min, b_max, R_odot, R_max*r_s, rho_s, r_s, g_chi/energyScale, g_A, GT)
    with open(filename, 'w') as fp:
        fp.write(metadata)
    dat = {'Photon energy [MeV]': energyMeV, 'Solid-angle-averaged flux [cm^-2 s^-1 MeV^-1 sr^-1]': averagedFlux, 'Integrated flux [cm^-2 s^-1 MeV^-1]': integratedFlux, \
           'Integral 1 [cm^-2 s^-1 MeV^-1]': integratedFlux1, 'Integral 2 [cm^-2 s^-1 MeV^-1]': integratedFlux2}
    df = pd.DataFrame(dat)
    df.to_csv(filename, sep='\t', index=False, mode='a') 
    ################


if __name__=="__main__":
#def main():
    nucleus = 'C12'
    g_chi = 1 # [MeV^-1]
    R_max = 50.0 # [kpc]
    epsilon = 0.05 # [unitless]
    m_n, J_n = nuc_dict[nucleus]['mass [unitless]'], nuc_dict[nucleus]['spin'] # [unitless, in units of energyScale], [unitless]
    l_min1, l_max1 = 330.25, ls[-1] # [deg], [deg]; bounds for first galactic longitude integral
    l_min2, l_max2 = ls[0], 29.75 # [deg], [deg]; bounds for second galactic longitude integral
    b_min, b_max = -4.75, 4.75 # [deg], [deg]; bounds for galactic latitude integral

    #### integrate solid angle, for calculating solid-angle-averaged flux ####
    def f(b, l):
        integrand = np.sin(b * np.pi/180 + np.pi/2)
        return integrand # [unitless]
        
    integratedSolidAngle1 = dblquad(f, l_min1, l_max1, b_min, b_max)[0] * np.pi/180 * np.pi/180 # [sr]
    integratedSolidAngle2 = dblquad(f, l_min2, l_max2, b_min, b_max)[0] * np.pi/180 * np.pi/180 # [sr]
    integratedSolidAngle = integratedSolidAngle1 + integratedSolidAngle2 # [sr]
    print("integrated solid angle (flux denominator): {} sr".format(integratedSolidAngle))
    ################
    
    print(nucleus)
    for dE in nuc_dict[nucleus]['dEs [MeV]'][5:80:10]: # for now, only go to 80 for C12, 75 for O16
        i = list(nuc_dict[nucleus]['dEs [MeV]']).index(dE)
        GT = nuc_dict[nucleus]['GTs'][i] # [unitless]
        print("i = {}:    dE = {} MeV, GT = {}".format(i, dE, GT))

        # retrieve DM masses that maximize the v_n_integral for a given dE:
        mass_by_dE_data = 'optimal_m_chis_by_dE/{}/{}_optimal_m_chis_dE_{}_MeV.txt'.format(nucleus, nucleus, dE)
        mass_by_dE_df = pd.read_csv(mass_by_dE_data, sep='\t', names=['Radius from GC [kpc]', 'Optimal DM mass [MeV]'], skiprows=1)
        M_CHIS = mass_by_dE_df['Optimal DM mass [MeV]']
        dm = (np.max(M_CHIS) - np.min(M_CHIS))/2
        m_chis = np.linspace(np.min(M_CHIS) - dm, np.max(M_CHIS) + dm, 5)

        for m_chi in m_chis:
            print("    m_chi = {} MeV".format(m_chi))
            tic = time.time()
            solid_angle_integral(g_chi*energyScale, m_chi/energyScale, nucleus, m_n, J_n, dE/energyScale, GT, b_min, b_max, l_min1, l_max1, l_min2, l_max2, integratedSolidAngle, R_max/r_s, epsilon)
            print("    time for 1 mass: {}".format(time.time() - tic))

#main()
