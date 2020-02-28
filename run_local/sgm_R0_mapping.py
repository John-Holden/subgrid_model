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
import sgm_test.sg_model_test as model
import matplotlib.pyplot as plt

print("Running local mode...")
params = {}  # default simulation parameters
# ### -- SET settings -- ### #
settings = {"out_path": os.getcwd() + '/animationsData/', "anim": False, "BCD3": False, "verbose": False,
            "save_figs": False, "out_plots": False, "run_local": True, "dyn_plots": [False, 1, False], "R0_mode": True}

# ### -- SET parameters -- ### #
params["alpha"] = 5  # Lattice constant (m)
params["l_time"] = 100  # Life time of disease
dispersal_ = 25  # average dispersal distance in (m)
eff_dispersal = dispersal_ / params["alpha"]  # Convert the dispersal from m to computer units
params["eff_disp"] = np.round(eff_dispersal, 5)
params["domain_sz"] = [60, 60]  # Size of domain/Lattice boundaries (x\alpha for m^2)
if settings["R0_mode"]:
    params["time_horizon"] = params["l_time"]
# -- Define test array to measure R0 against -- #
test_ = ["beta", "rho"][1]
# Define beta array and single rho value
# todo iterate through both rho and beta values so you can see multiple lines!
# todo get multiple lines and sg mappings
if test_ == "beta":
    test_arr = np.linspace(0.001, 0.10, 30)
    params["rho"] = 0.035
# Define rho array & single beta value
elif test_ == "rho":
    test_arr = np.linspace(0.1, 1, 20)
    params["beta"] = 0.010

R0_Arr = np.zeros(shape=[2, test_arr.shape[0]])
ensemble_size = 10
plt_tseries, plt_pspace = [False, True]
# BEGIN Ensemble
for i, test_value in enumerate(test_arr):
    if test_arr.shape[0] > 1:  # Print if multiple values & small number of repeats
        print('i: {} / {}'.format(i, test_arr.shape[0]))
    # SET simulation parameters
    if test_ == "beta":
        params["beta"] = test_value
    elif test_ == "rho":
        params["rho"] = test_value
    # Define saving structs

    ens_mortality = np.zeros(ensemble_size)
    for j in range(ensemble_size):  # Iterate over repeats
        if test_arr.shape[0] < 2:  # Print if single value & large number of repeats
            print('run {} / {}'.format(j, ensemble_size))
        Results = model.main(settings, params)
        mortality_ = Results[0]
        ens_mortality[j] = mortality_

    # Finish
    rho_t = np.zeros(params["l_time"])
    rho, sigma, beta, T = params["rho"], params["eff_disp"], params["beta"], params["l_time"]
    R0_anal = (2 * np.pi * beta * rho * sigma ** 2) * ((1 - 2 / 9 * beta) ** T - 1) / log(1 - 2 / 9 * beta)
    R0_Arr[0][i], R0_Arr[1][i] = ens_mortality.mean(), R0_anal

plt.plot(test_arr, R0_Arr[0], label=r'simulated $R_0$ ')
plt.plot(test_arr, R0_Arr[1], label=r'analytical $R_0$ ', alpha=0.25)
plt.scatter(test_arr, R0_Arr[0], c='C0', s=10)
plt.scatter(test_arr, R0_Arr[1], c='C1', s=10)
if test_ == "beta":
    plt.xlabel(r'$\beta$', size=13)
    plt.title(r'$\rho = $ {}  $\ell = $ {}, T = {}'.format(rho, dispersal_, T))
elif test_ == "rho":
    plt.xlabel(r'$\rho$')
    plt.title(r'$\beta = $ {}  $\ell = $ {}, T = {}'.format(beta, dispersal_, T))

plt.ylabel(r'$R_0$', size=13)
plt.legend()
# plt.savefig('R0-beta')
plt.show()

sys.exit("exiting...")


