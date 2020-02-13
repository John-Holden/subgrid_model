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

def R0_func(beta, rho, sigma, T):
    return (2 * np.pi * beta * rho * sigma ** 2) * ((1 - 2 / 9 * beta) ** T - 1) / log(1 - 2 / 9 * beta)


print("Running anim-single-debug...")
params = {}  # default simulation parameters
settings = {"out_path": os.getcwd()+'/animationsData/'}  # simulation settings
# SET simulation settings & boundary conditions
settings["anim"] = False
settings["BCD3"] = True  # Percolation condition : if False, simulations will run-HPC until pathogen dies
settings["model_test"] = False
settings["verbose"] = False
settings["save_figs"] = False
settings["out_plots"] = False
settings["dyn_plots"] = [False, 1, True]
alpha = 5  # Lattice constant in (m)
dispersal_ = 50  # average dispersal distance in (m)
params["rho"] = 0.0500  # Typically \in [0.001, 0.100]
params["l_time"] = 100  # L# ife time of disease
params["alpha"] = alpha
eff_dispersal = dispersal_ / alpha  # Convert the dispersal distance from km to computer units
eff_dispersal = np.round(eff_dispersal, 5)
params["domain_sz"] = [250, 250]  # If channel [NxM] where N < M
params["eff_disp"] = eff_dispersal
params["time_horizon"] = 3650
# Ensemble setup
ensemble_size = 10
beta_Arr = np.linspace(0.0001, 0.0010, 20)
R0_Arr = np.zeros(beta_Arr.shape[0])
perc_Arr = np.zeros(beta_Arr.shape[0])
ens_perc = np.zeros(shape=(beta_Arr.shape[0], ensemble_size))
# Run beta/R0 vs percolation Pr
for i, beta in enumerate(beta_Arr):
    R0_Arr[i] = R0_func(beta, params["rho"], params["eff_disp"], params["l_time"])
    print('i: {} / {}'.format(i, beta_Arr.shape[0]))
    # SET simulation parameters
    params["beta"], params["R0"] = beta, R0_Arr[i]
    print("Running: beta {}".format(round(beta, 6)))
    for j in range(ensemble_size):
        if beta_Arr.shape[0] < 2:
            print('run {} / {}'.format(j, ensemble_size))
        Results = model.main(settings, params)
        mortality_, velocity_, max_d_, run_time_, percolation_, population_sz, ts_max_d, ts_removed = Results
        ens_perc[i, j] = percolation_
    print('R0 :{}, Perc mean {}'.format(round(R0_Arr[i], 3), ens_perc[i].mean()))


print(R0_Arr, ' R0 array')
print(ens_perc, ' ens perc raw')
ens_perc = np.sum(ens_perc, axis=1) / ensemble_size
print('ens perc averaged', ens_perc)
fig, ax = plt.subplots()
label_ = r'$\rho = $ {}  $\ell = $ {}, T = {}'.format(params["rho"], params["eff_disp"]*alpha, params["l_time"])
ax.plot(beta_Arr, ens_perc, label=label_)
ax.set_xlabel(r'$\beta$', size=15, color='r')
ax.set_ylabel(r'Percolation probability', size=15)
ax.tick_params(axis='x', labelcolor='red', rotation=45)

ax2 = ax.twiny()  # instantiate a second axes that shares the same x-axis
ax2.set_xlabel(r'$R_0$ - analytical', color='blue', size=15)
ax2.set_xticks(R0_Arr.round(2))

ax2.tick_params(axis='x', labelcolor='blue', rotation=45)
plt.legend()

plt.grid(True, alpha=0.50)
plt.savefig('R0-v-perc')
plt.show()

sys.exit("exiting...")


