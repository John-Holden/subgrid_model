"""
Created on Wed May 30 14:19:32 2018
@author: John Holden
Compute either the epidemiological parameter space behaviuor over an ensemble or simulate an individual realisation.
This code can be run-HPC on the HPC or local machine for single or ensemble-averages. To run-HPC, execute ./run_SSTML.sh in
terminal directory.
"""
import os
import sys
import numpy as np
from math import log
import sgm_test.sg_model_test as model
import matplotlib.pyplot as plt

print("Running anim-single-debug...")
params = {}  # default simulation parameters
settings = {"out_path": os.getcwd()+'/animationsData/'}  # simulation settings
# SET simulation settings & boundary conditions
settings["anim"] = False
settings["BCD3"] = False  # Percolation condition : if False, simulations will run-HPC until pathogen dies
settings["verbose"] = False
settings["save_figs"] = False
settings["out_plots"] = False
settings["model_test"] = True
settings["dyn_plots"] = [False, 1, True]
alpha = 5  # Lattice constant in (m)
dispersal_ = 50  # average dispersal distance in (m)
params["rho"] = 0.100  # Typically \in [0.001, 0.100]
params["l_time"] = 100  # L# ife time of disease
params["alpha"] = alpha
eff_dispersal = dispersal_ / alpha  # Convert the dispersal distance from km to computer units
eff_dispersal = np.round(eff_dispersal, 5)
params["domain_sz"] = [100, 100]  # If channel [NxM] where N < M
params["eff_disp"] = eff_dispersal
# If model_test True then
params["time_horizon"] = params["l_time"]
# BEGIN Ensemble
ensemble_size = 100
plot_ = False
beta_Arr = np.linspace(0.0001, 0.99, 30)
R0_Arr = np.zeros(shape=[2, beta_Arr.shape[0]])

for i, beta in enumerate(beta_Arr):
    if beta_Arr.shape[0] > 1:
        print('i: {} / {}'.format(i, beta_Arr.shape[0]))
    # SET simulation parameters
    params["beta"] = beta
    ts_ens_removed = np.zeros(shape=[ensemble_size, params["time_horizon"]])
    ens_mortality = np.zeros(ensemble_size)
    print("Running: beta {}".format(round(beta, 6)))
    for j in range(ensemble_size):
        if beta_Arr.shape[0] < 2:
            print('run {} / {}'.format(j, ensemble_size))
        Results = model.main(settings, params)
        mortality_, velocity_, max_d_, run_time_, percolation_, population_sz, ts_max_d, ts_removed = Results
        ts_ens_removed[j] = ts_removed
        ens_mortality[j] = mortality_

    ts_ens_removed = ts_ens_removed.sum(axis=0) / ensemble_size
    time = np.array(range(params["l_time"]))
    rho_t = np.zeros(params["l_time"])
    rho, sigma, beta, T = params["rho"], params["eff_disp"], params["beta"], params["l_time"]
    R0_anal = (2*np.pi*beta*rho*sigma**2) * ((1 - 2/9 * beta)**T - 1) / log(1 - 2/9 * beta)
    R0_sim = np.gradient(ts_ens_removed).sum()
    R0_Arr[0][i], R0_Arr[1][i] = R0_sim, R0_anal
    if plot_:
        for k, t in enumerate(range(params["l_time"])):
            if k == 0:
                rho_t[k] = 1
            else:
                rho_t[k] = rho_t[k - 1] * (1 - 2 / 9 * beta)
        r0_t = 2 * np.pi * beta * rho * sigma ** 2 * rho_t
        plt.plot(time, r0_t, label=r'analytical $R(t)$:  $R_0 =$ {}'.format(round(R0_anal, 2)))
        plt.plot(time, np.gradient(ts_ens_removed),
                 label='simulated $R(t)$: $R_0 =$ {}'.format(round(R0_sim, 2)))
        plt.xlabel('t (days)', size=12)
        plt.ylabel(r'#expected infection $R(t)$', size=12)
        plt.title(r'$\rho = $ {}, $\beta $ = {}, $\ell = $ {}, T=100'.format(rho, round(beta, 6), sigma*alpha))
        plt.legend()
        plt.savefig('Rt-beta-{}-rho-{}-ell{}'.format(str(round(beta, 6)).replace('.', '_'), str(rho).replace('.', '_'),
                                                     str(sigma*alpha).replace('.', '_')))
        plt.show()
        print('analytical: R0 = {} '.format(round(R0_anal, 4)))
        print('simulated: R0 = {}'.format(round(R0_sim, 4)))


plt.title(r'$\rho = $ {}  $\ell = $ {}, T = {}'.format(rho, dispersal_, T))
plt.plot(beta_Arr, R0_Arr[0], label=r'simulated $R_0$')
plt.plot(beta_Arr, R0_Arr[1], label=r'analytical $R_0$')
plt.scatter(beta_Arr, R0_Arr[0], c='C0', s=10)
plt.scatter(beta_Arr, R0_Arr[1], c='C1', s=10)
plt.xlabel(r'$\beta$', size=13)
plt.ylabel(r'$R_0$', size=13)
plt.legend()
plt.savefig('R0-beta')
plt.show()

sys.exit("exiting...")


