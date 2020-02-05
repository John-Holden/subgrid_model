"""
Created on Wed May 30 14:19:32 2018
@author: John Holden
Compute either the epidemiological parameter space behaviuor over an ensemble or simulate an individual realisation.
This code can be run on the HPC or local machine for single or ensemble-averages. To run, execute ./run_SSTML.sh in
terminal directory.
"""
import os
import sys
import time
import pickle
import datetime
import numpy as np
import sgm_debug.sgm_debug as model


print("Running debug...")
params = {}  # default simulation parameters
settings = {"out_path": os.getcwd()+'/animationsData/'}  # simulation settings
# SET simulation settings & boundary conditions
settings["save_figs"] = False
settings["anim"] = True
settings["BCD3"] = True  # Percolation condition : if False, simulations will run until pathogen dies
settings["verbose"] = True
settings["plt_tseries"] = True
settings["dyn_plots"] = [True, 1, True]
# SET simulation parameters
params["R0"] = 10  # number of secondary infections
params["rho"] = 0.010  # Typically \in [0.001, 0.100]
params["l_time"] = 100  # Life time of disease
params["time_horizon"] = 3650  # Time before simulation ends
dispersal_ = 100  # average dispersal distance in (m)
alpha = 5  # Lattice constant in (m)
eff_dispersal = dispersal_ / alpha  # Convert the dispersal distance from km to computer units
eff_dispersal = np.round(eff_dispersal, 5)
params["eff_disp"] = eff_dispersal
params["alpha"] = 5
params["time_horizon"] = 600
params["domain_sz"] = [200, 200]  # If channel [NxM] where N < M
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
print('dist/run_time = {} (km/yr)'.format(round(max_d_ / run_time_ * 365, 4)))
print('velocity ensemble value = {} (km/yr)'.format(round(velocity_ * 365, 4)))
sys.exit("exiting...")


