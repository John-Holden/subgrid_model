"""
Created on Wed May 30 14:19:32 2018
@author: John Holden
Compute either the epidemiological parameter space behaviuor over an ensemble or simulate an individual realisation.
This code can be run on the HPC or local machine for single or ensemble-averages. To run, execute ./run_SSTML.sh in
terminal directory.
"""
import os
import sys
import numpy as np
import sgm_test.sg_model_test as model
import matplotlib.pyplot as plt

print("Running debug...")
params = {}  # default simulation parameters
settings = {"out_path": os.getcwd()+'/animationsData/'}  # simulation settings
# SET simulation settings & boundary conditions
settings["save_figs"] = False
settings["out_plots"] = False
settings["anim"] = False
settings["BCD3"] = False  # Percolation condition : if False, simulations will run until pathogen dies
settings["verbose"] = False
settings["dyn_plots"] = [False, 1, True]
settings["test_R0"] = True
# SET simulation parameters
params["rho"] = 0.99  # Typically \in [0.001, 0.100]
params["beta"] = 0.99
params["l_time"] = 100  # L# ife time of disease
dispersal_ = 70  # average dispersal distance in (m)
alpha = 5  # Lattice constant in (m)
eff_dispersal = dispersal_ / alpha  # Convert the dispersal distance from km to computer units
eff_dispersal = np.round(eff_dispersal, 5)
params["eff_disp"] = eff_dispersal
params["alpha"] = alpha
params["time_horizon"] = params["l_time"]
params["domain_sz"] = [100, 100]  # If channel [NxM] where N < M

ensemble_size = 1000
ts_ens_removed = np.zeros(shape=[ensemble_size, params["time_horizon"]])
ens_mortality = np.zeros(ensemble_size)
# BEGIN
print("Running: ")
for i in range(ensemble_size):
    Results = model.main(settings, params)
    mortality_, velocity_, max_d_, run_time_, percolation_, population_sz, ts_max_d, ts_removed = Results
    ts_ens_removed[i] = ts_removed
    ens_mortality[i] = mortality_


ts_ens_removed = ts_ens_removed.sum(axis=0) / ensemble_size
time = range(params["l_time"])

rho_t = np.zeros(params["l_time"])
rho, sigma, beta = params["rho"], params["eff_disp"], params["beta"]
for i, t in enumerate(range(params["l_time"])):
    if i == 0:
        rho_t[i] = 1
    else:
        rho_t[i] = rho_t[i-1] * (1 - 0.4 * beta)

r0_t = 2*np.pi*beta*rho*sigma**2 * rho_t
plt.plot(time, r0_t, label='predicted')
plt.plot(time, np.gradient(ts_ens_removed), label='simulated')
plt.xlabel('Time days')
plt.ylabel(r'$R_0(t)$')
plt.title(r'$\rho = $ {}, $\beta $ = {}, $\ell = $ {}'.format(rho, beta, sigma*alpha))
plt.legend()
plt.show()

sys.exit()

plt.title('Mortality')
plt.plot(range(ensemble_size), ens_mortality, label='simulaitons')
plt.xlabel('days')
plt.ylabel('# dead')
plt.legend()
plt.show()

sys.exit("exiting...")


