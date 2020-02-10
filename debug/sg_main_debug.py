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
import sgm_debug.sgm_debug as model

print("Running debug...")
params = {}  # default simulation parameters
settings = {"out_path": os.getcwd()+'/animationsData/'}  # simulation settings
# SET simulation settings & boundary conditions
settings["save_figs"] = False
settings["out_plots"] = True
settings["anim"] = True
settings["BCD3"] = True  # Percolation condition : if False, simulations will run until pathogen dies
settings["verbose"] = True
settings["plt_tseries"] = True
settings["dyn_plots"] = [True, 1, True]
# SET simulation parameters
params["R0"] = 1  # number of secondary infections
params["rho"] = 0.100  # Typically \in [0.001, 0.100]
params["beta"] = 0.0010
params["l_time"] = 100  # L# ife time of disease
dispersal_ = 20  # average dispersal distance in (m)
alpha = 5  # Lattice constant in (m)
eff_dispersal = dispersal_ / alpha  # Convert the dispersal distance from km to computer units
eff_dispersal = np.round(eff_dispersal, 5)
params["eff_disp"] = eff_dispersal
params["alpha"] = alpha
params["time_horizon"] = 200
params["domain_sz"] = [50, 50]  # If channel [NxM] where N < M
# BEGIN
print("Running: ")
Results = model.main(settings, params)
mortality_, velocity_, max_d_, run_time_, percolation_, population_sz, ts_max_d = Results
print("__Finished__")
# END
print('percolation = {}'.format(percolation_))
print('Run time = {} (days)'.format(run_time_))
print('Mortality = {}'.format(mortality_))
print('Population size = {}'.format(population_sz))
print('Mortality ratio = {}'.format(mortality_ / population_sz))
print('Max distance reached = {} (km)'.format(max_d_))
print('velocity ensemble value = {} (km/yr)'.format(round(velocity_ * 365, 4)))
sys.exit("exiting...")


