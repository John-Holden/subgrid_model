"""
Created on Wed May 30 14:19:32 2018
@author: John Holden

The sub-grid model; call sg_model.sg_model
Compute either the epidemiological parameter space behavior over either dispersal (\ell), infectivity (\beta), or
density (\rho) over n repeats.

Run this extract to generate parameter space of a stochastic sub-grid sg_model of tree disease over 3D:
1. rho (tree density)
2. beta (infectivity) or R0
3. dispersal distance

This is done using the HPC **arc3@leeds.ac.uk** and the task_array feature. Each core gets the same set
of parameters to iterate over and metrics to record. Each core on the HPC will save results in the
output_path as 00**.npy, each core saves results in array form. Results are analysed in the 'processData/'
folder using 'data_process' file.

Settings[R0_mode]: set True, to get reproductive ratio of pathogen and tree species. Domain should be small and
                   R0 saved as mortality field
"""

import os
import sys
import time
import pickle
import datetime
import numpy as np
import sg_model.sg_model as model

job_id, date, c_time, mode, sim_type, sim_name = sys.argv[1:]  # input parameters from run_script.sh.
output_path = os.getcwd() + '/processData/' + date + '-' + mode + sim_type + sim_name  # saving data path
# default HPC model parameters
params = {"l_time": 100, "time_horizon": 3650}
# Default HPC simulation settings
settings = {"out_path": output_path, "date": date, "job_id": job_id, "plt_tseries": False, "save_figs": False,
            "dyn_plots": [False, None, False], "anim": False, "BCD3": True, "individual": False, "verbose": False,
            "HPC": mode, "debug_time": True, "R0_mode": False}

# Define parameters
alpha = 5  # lattice constant m^2
params["alpha"] = alpha
params["domain_sz"] = [200, 200]  # to convert to (m), multiply by alpha, If R0 then domain can be small ~3*ell
# save ID : unique to each core used
if int(job_id) < 10:
    save_id = '00' + str(job_id)
if 10 <= int(job_id) <= 100:
    save_id = '0' + str(job_id)
if 1:  # Sub-grid mapping format
    # RUN partial parameter space
    mode = "sg mapping 1D"
    # rho_Arr_hig = np.linspace(0.10, 0.400, 4) # rho_Arr = np.hstack([rho_Arr_low, rho_Arr_med, rho_Arr_hig])
    rho_Arr_med = np.linspace(0.051, 0.10, 10)
    rho_Arr_low = np.linspace(0.001, 0.050, 50)  # Tree density range stack at different resolutions
    rho_Arr = np.hstack([rho_Arr_low, rho_Arr_med])
    beta_Arr = np.array([0.005])  # Infectivity constant (or rate) \approx \in [0.005, 0.020]
    eff_sigma_Arr = np.array([25]) / alpha  # Dispersal distance in comp units (not physical)
if 0:  # Phase-plane format
    # RUN Full parameter space
    mode = "Phase plane 2D"
    rho_Arr = np.array([0.01, 0.02, 0.03, 0.04])
    beta_Arr = np.linspace(0.0, 0.0200, 100)
    eff_sigma_Arr = np.linspace(5, 30, len(beta_Arr)) / alpha

settings["R0_mode"] = False
repeats = 5  # ensemble size = repeats * # HCP_cores
dim_ = np.array([repeats, eff_sigma_Arr.shape[0], beta_Arr.shape[0], rho_Arr.shape[0]])  # parameter space dimension
# DEFINE data structures to save results
mortality = np.zeros(shape=dim_)
run_times = np.zeros(shape=dim_)
velocities = np.zeros(shape=dim_)
max_distances = np.zeros(shape=dim_)
percolation_pr = np.zeros(shape=dim_)
mortality_ratio = np.zeros(shape=dim_)  # (I + R) / Population_size
# START ensemble simulation
t0 = time.process_time()
print("Running; mode {}, R0 {}".format(mode, settings["R0_mode"]))
print("Start time: ", datetime.datetime.now(), ' |  sim : ', str(job_id))
for r in range(repeats):  # ITERATE repeats
    for i, eff_disp in enumerate(eff_sigma_Arr):  # ITERATE dispersal kernel
        print('    ell = {} : {} / {}'.format(round(eff_disp, 3), i+1, dim_[1]))
        params["eff_disp"] = eff_disp
        for j, beta in enumerate(beta_Arr):  # ITERATE infection rates
            params["beta"] = beta
            for k, rho in enumerate(rho_Arr):  # ITERATE through density values
                params["rho"] = rho
                subgrid = model.SubGrid(parameters=params, settings=settings)
                results = subgrid.main_run()
                mortality_, velocity_, max_d_, run_time_, percolation_, population_sz = results
                mortality[r, i, j, k] = mortality_
                velocities[r, i, j, k] = velocity_
                max_distances[r, i, j, k] = max_d_
                run_times[r, i, j, k] = run_time_
                percolation_pr[r, i, j, k] = percolation_
                mortality_ratio[r, i, j, k] = mortality_ / population_sz

    # save results as multi-dimensional arrays
    np.save(output_path + "/run_time/" + save_id, run_times)
    np.save(output_path + "/mortality/" + save_id, mortality)
    np.save(output_path + "/velocity/" + save_id, velocities)  # saved in km/day
    np.save(output_path + "/percolation/" + save_id, percolation_pr)
    np.save(output_path + "/max_distance_km/" + save_id, max_distances)  # saved in km
    np.save(output_path + "/mortality_ratio/" + save_id, mortality_ratio)

# #### End param sweep # ####
tf = time.process_time() - t0
tf = np.float64(tf / 60)
print('End time: {} |  sim : {} '.format(datetime.datetime.now(), str(job_id)))
print("Time taken: {} (mins)".format(tf.round(3)))
# WRITE parameters, settings and ensemble data to file
beta_str = str(beta_Arr)+' (m), # = '+str(len(beta_Arr))
rho_str = str(rho_Arr[0].round(4))+' -- '+str(rho_Arr[-1].round(4))+', # = '+str(len(rho_Arr))
ell_str = str(eff_sigma_Arr[0]*alpha)+' -- '+str(eff_sigma_Arr[-1]*alpha)+'(m), # = '+str(len(eff_sigma_Arr))
# Save arrays used
np.save(output_path + '/_sim-info/beta_Arr', beta_Arr)
np.save(output_path + '/_sim-info/rho_Arr', rho_Arr)
np.save(output_path + '/_sim-info/disp_Arr', eff_sigma_Arr)
# Save dictionaries
settings["repeats"] = repeats
with open(output_path + '/_sim-info/sim-settings.pickle', 'wb') as handle:
    pickle.dump(settings, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(output_path + '/_sim-info/sim-parameters.pickle', 'wb') as handle:
    pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)
# Save human-readable .txt
with open(output_path + "/_sim-info/parameter_and_settings_info.txt", "w+") as info_file:
    info_file.write("______Simulation Parameters_______" + "\n")
    for parameter in params:
        info_file.write(parameter + ' : ' + str(params[parameter]) + '\n')
    info_file.write("Beta values : {}".format(beta_str) + '\n')
    info_file.write("Dispersal values : {}".format(ell_str) + '\n')
    info_file.write("Density values : {}".format(rho_str) + '\n')
    info_file.write("# HPC core repeats : {}".format(repeats) + '\n')
    info_file.write("\n" + "______Simulation Settings_______" + "\n")
    for setting in settings:
        info_file.write(setting + ':' + str(settings[setting]) + '\n')

# ##### END HPC simulations # #####




