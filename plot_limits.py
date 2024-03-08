import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle

colors = mcolors.TABLEAU_COLORS
color_tabs = list(colors.keys())
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rc('text', usetex=True)

##############################################
### Parameters
RECT_WIDTH = 0.1 # width of the rectangle
ALPHA = 0.4 # transparency of the rectangle
SELECT = 'all' # which experiments to plot: 'all', 'future', 'current'
EXPERIMENT_DICT = {
    # experiment: [limit filename, color]
    'AMEGO-X': ['AMEGO-X_g_chi_limits_epsilon_2.0', 'darkorange'],
    'GRAMS': ['GRAMS_g_chi_limits_epsilon_1.0', 'firebrick'],
    'e-ASTROGAM': ['e-ASTROGAM_g_chi_limits_epsilon_3.0', 'darkgreen'],
    'COMPTEL': ['COMPTEL_g_chi_limits_epsilon_2.0', 'mediumblue']
}
##############################################


future_experiments = ['AMEGO-X', 'GRAMS', 'e-ASTROGAM']
current_experiments = ['COMPTEL']
col_names = ['dE', 'm_chi', 'lower limit', 'upper limit']


def plot_rect(x: float, y_lower: float, y_upper: float, color: str, alpha=0.3, label=None):
    """
    Plot a rectangle with x-axis centered at x with width rect_width, and y-axis from y_lower to y_upper
    """
    # lower left (x, y), width, height
    rect = Rectangle((x-RECT_WIDTH/2, y_lower), RECT_WIDTH, y_upper-y_lower, color=color, alpha=alpha, label=label)
    return rect


def plot_one_limits(experiment: str, plot=False):
    """
    Plot the limit for one experiment
    """
    filename, color = EXPERIMENT_DICT[experiment]
    c12 = pd.read_csv(f'g_chi_limits/C12/{filename}.txt', delimiter="\t", skiprows=1, header=None, names=col_names)
    o16 = pd.read_csv(f'g_chi_limits/O16/{filename}.txt', delimiter="\t", skiprows=1, header=None, names=col_names)
    co = pd.concat([c12, o16], ignore_index=True).sort_values(by='m_chi')

    if plot: plt.figure(figsize=(8, 6))
    for idx, row in co.iterrows():
        legend_label = experiment if idx == 0 else None
        rect = plot_rect(row['m_chi'], row['lower limit'], row['upper limit'], color=color, alpha=ALPHA, label=legend_label)
        plt.gca().add_patch(rect)

    if plot:
        plt.xlabel(r'$m_{\chi}$ [MeV]', fontsize=18)
        plt.ylabel(r'$g_{\chi}$ [MeV$^{-1}$]', fontsize=18)
        plt.yscale('log')
        plt.xlim(9, 30)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.legend(fontsize=16)
        plt.tight_layout()
        plt.show()


def plot_limits(select: str):
    """
    Plot the limits for all experiments
    select: one of ('all', 'future', 'current'). which experiment(s) to plot
    """
    match select:
        case 'all':
            experiments = future_experiments + current_experiments
        case 'future':
            experiments = future_experiments
        case 'current':
            experiments = current_experiments

    plt.figure(figsize=(8, 6))
    for i, experiment in enumerate(experiments):
        plot_one_limits(experiment)

    plt.xlabel(r'$m_{\chi}$ [MeV]', fontsize=18)
    plt.ylabel(r'$g_{\chi}$ [MeV$^{-1}$]', fontsize=18)
    plt.yscale('log')
    plt.xlim(9, 30)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # plot_one_limits('GRAMS', 'blue', plot=True)
    plot_limits(SELECT)
