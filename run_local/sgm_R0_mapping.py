"""
Created on Wed May 30 14:19:32 2018
@author: John Holden

Run simulation in R0 finding mode, i.e. find the average number of secondary infected trees that result from a single
infected individual and plot the relationship between R0 simulated and R0 predicted over a range of infectivity/beta
values.
"""
import os
import sys
import numpy as np
from math import log
import sgm_test.sg_model_test as sg_model
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def R0_analytical(beta, rho, sigma, T):
    return (2 * np.pi * beta * rho * sigma ** 2) * ((1 - 2 / 9 * beta) ** T - 1) / log(1 - 2 / 9 * beta)


def R0_subgrid_mapping(arr, beta, ell):
    np.save('R0-beta_{}_ell_{}'.format(str(beta).replace('.', '_'), str(ell)), arr)


def lin_func(x, m, c):
    return m*x + c


# ------------------ SET settings ------------------  #
settings = {"out_path": os.getcwd() + '/animationsData/', "anim": False, "BCD3": False, "verbose": False,
            "save_figs": False, "plt_tseries": False, "run_local": True, "dyn_plots": [False, 1, False],
            "R0_mode": True}

# ------------------ SET parameters ------------------  #
params = {}
dispersal_ = 25  # average dispersal distance in (m)
params["alpha"] = 5  # Lattice constant (m)
params["l_time"] = 100  # Life time of disease
eff_dispersal = dispersal_ / params["alpha"]  # Convert the dispersal from m to computer units
params["eff_disp"] = np.round(eff_dispersal, 5)
params["domain_sz"] = [60, 60]  # Size of domain/Lattice boundaries (x\alpha for m^2)
if settings["R0_mode"]:
    params["time_horizon"] = params["l_time"]

# ------------------ SET arrays ------------------  #
if 1:  # Test rho array
    beta_Arr = np.array([0.0075])
    rho_Arr = np.linspace(0.001, 1, 10)
    test_ = "rho"

if 0:  # Test beta array
    beta_Arr = np.linspace(0.001, 0.20, 10)
    rho_Arr = np.array([0.05, 0.075, 0.10])
    test_ = "beta"

ensemble_size = 200
R0_predicted_arr = np.zeros(shape=[len(rho_Arr), len(beta_Arr)])
R0_simulated_arr = np.zeros(shape=[len(rho_Arr), len(beta_Arr), ensemble_size])
# ------------------ Begin simulation ------------------  #
for i, rho in enumerate(rho_Arr):
    params["rho"] = rho
    print('rho: {} / {}'.format(i, len(rho_Arr)))
    for j, beta in enumerate(beta_Arr):
        params["beta"] = beta
        R0_predicted_arr[i, j] = R0_analytical(beta, rho, sigma=eff_dispersal, T=params["l_time"])
        for r in range(ensemble_size):
            model = sg_model.SubGrid(params, settings)
            results = model.main_run()
            R0_simulated_arr[i, j, r] = results[0]  # Save mortality

R0_simulated_arr = R0_simulated_arr.sum(axis=2) / ensemble_size
if test_ == "rho":
    fig, ax = plt.subplots()
    for i in range(R0_predicted_arr.shape[1]):  # iterate over beta lines
        arr = R0_simulated_arr[:, i]
        popt, pcov = curve_fit(lin_func, rho_Arr, arr)
        R0_subgrid_mapping(arr, beta=beta_Arr[i], ell=dispersal_)
        ax.plot(rho_Arr, arr, label=r'$\beta = $ {}, $m = {}$'.format(beta_Arr[i], round(popt[0], 3)), c="C"+str(i))
        ax.scatter(rho_Arr, arr, c="C"+str(i))
        # ax.plot(rho_Arr, R0_predicted_arr[:, i], linestyle='--', c="C"+str(i), alpha=0.5)
        ax.plot(rho_Arr, lin_func(rho_Arr, m=popt[0], c=popt[1]), linestyle='--', c="C"+str(i), alpha=0.5)

    ax.plot([rho_Arr[0], rho_Arr[-1]], [1, 1], c='r', alpha=0.5)
    ax.set_ylabel(r'$R_0$', size=15)
    ax.set_xlabel(r'$\rho$', size=15)
    plt.legend()
    plt.show()

elif test_ == "beta":
    fig, ax = plt.subplots()
    for i in range(R0_predicted_arr.shape[0]):  # iterate over beta lines
        ax.plot(beta_Arr, R0_simulated_arr[i], label=r'$\rho = $ {}'.format(rho_Arr[i]), c="C"+str(i))
        ax.plot(beta_Arr, R0_predicted_arr[i], linestyle='--', c="C"+str(i), alpha=0.5)

    ax.set_ylabel(r'$R_0$', size=15)
    ax.set_xlabel(r'$\beta$', size=15)
    plt.legend()
    plt.show()

np.save('R0_rho_Arr', rho_Arr)
sys.exit("exiting...")


