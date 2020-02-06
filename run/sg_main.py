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
sys.path.append(os.getcwd() + '/sgm_model')
import sg_model


job_id, date, c_time, mode, sim_type, sim_name = sys.argv[1:]  # input parameters from run_script.sh.

print("Running {} simulation, type {}".format(mode, sim_type))
output_path = os.getcwd() + '/processData/' + date + '-' + mode + sim_type + sim_name  # saving data path
params = {"l_time": 100, "time_horizon": 3650}  # default simulation parameters
settings = {"out_path": output_path, "date": date, "job_id": job_id, "plt_tseries": False,
            "save_figs": False, "dyn_plots": [False, 1, True], "anim": False, "BCD3": False, "individual": False,
            "verbose": False, "HPC": None, "local_type": "animation", "debug_time": True}  # simulation settings

# HPC MODE
# 1) FULL PARAM: full parameter mode, full parameter sweep
# 2) HRES: high resolution mode, partial parameter sweep
if mode == "HPC":
    """                             ----- HPC mode -----
    Run this extract to generate parameter space of a stochastic sub-grid sgm_model of tree disease over 3D: 
    1. rho (tree density)
    2. beta (infectivity) or R0
    3. dispersal distance
    This is done using the HPC **arc3@leeds.ac.uk** and the task_array feature. Each core gets the same set 
    of parameters to iterate over and metrics to record. Each core on the HPC will save results in the 
    output_path as 00**.npy, each core saves results in array form. Results are analysed in the 'processData/'
    folder using 'data_process' file.
    """
    settings["HPC"] = mode
    settings["BCD3"] = True  # IF false --> mortality sims i.e. run until all trees are dead
    settings["verbose"] = False
    # DEFINE parameters
    alpha = 5  # lattice constant m^2
    params["alpha"] = alpha
    params["domain_sz"] = [200, 200]  # to convert to (m), multiply by alpha
    # save ID : unique to each core used
    if int(job_id) < 10:
        save_id = '00' + str(job_id)
    if 10 <= int(job_id) <= 100:
        save_id = '0' + str(job_id)
    # RUN Full parameter space : beta | \ell | \rho
    # ---- Iterate indices as  ---> [i: dispersal, j:infectivity, k:tree density]
    repeats = 3  # ensemble size = repeats * # HCP_cores
    # Rho array has different resolutions for different different points through rho-space
    # rho_Arr_low = np.arange(0.0001, 0.050, 0.0001)  # Tree density range
    # rho_Arr_med = np.arange(0.051, 0.10, 0.010)
    # rho_Arr_hig = np.linspace(0.10, 0.400, 4)
    # rho_Arr = np.hstack([rho_Arr_low, rho_Arr_med, rho_Arr_hig])
    # rho_Arr = np.hstack([rho_Arr_low, rho_Arr_med]) # rho_Arr = rho_Arr_low
    rho_Arr = np.array([0.050])
    beta_Arr = np.arange(0.0, 1.0, 0.05)
    eff_sigma_Arr = np.linspace(5, 50, beta_Arr.shape[0]) / alpha  # Dispersal distance in comp units (not physical
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
    print("Start time: ", datetime.datetime.now(), ' |  sim : ', str(job_id))
    c = 0
    for r in range(repeats):
        c = 0
        for i, eff_disp in enumerate(eff_sigma_Arr):  # ITERATE dispersal kernel
            print('    ell = {} : {} / {}'.format(eff_disp, i+1, dim_[1]))
            params["eff_disp"] = eff_disp
            for j, beta in enumerate(beta_Arr):  # ITERATE infection rates
                params["beta"] = beta
                for k, rho in enumerate(rho_Arr):  # ITERATE through density values
                    params["rho"] = rho
                    results = sg_model.main(settings, params)
                    mortality_, velocity_, max_d_, run_time_, percolation_, population_sz, ts_max_d = results
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

    # #### END FULL PARAM SWEEP # ####
    tf = time.process_time() - t0
    tf = np.float64(tf / 60)
    print('End time: {} |  sim : {} '.format(datetime.datetime.now(), str(job_id)))
    print("Time taken: {} (mins)".format(tf.round(3)))
    # WRITE parameters, settings and ensemble data to file
    beta_str = str(beta_Arr)+' (m), # = '+str(len(beta_Arr))
    rho_str = str(rho_Arr[0].round(4))+' -- '+str(rho_Arr[-1].round(4))+', # = '+str(len(rho_Arr))
    ell_str = str(eff_sigma_Arr[0]*alpha)+' -- '+str(eff_sigma_Arr[-1]*alpha)+'(m), # = '+str(len(eff_sigma_Arr))
    output_path = settings["out_path"]
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


# LOCAL MACHINE MODE
# 1) ANIM: animation mode,
# 2) ENS: ensemble mode
elif mode == "LCL":
    # Individual simulation for animation
    if sim_type == "-anim":
        R0 = 10  # number of secondary infections
        dispersal_ = 100  # average dispersal distance in (m)
        alpha = 5  # Lattice constant in (m)
        eff_dispersal = dispersal_ / alpha  # Convert the dispersal distance from km to computer units
        eff_dispersal = np.round(eff_dispersal, 5)
        rho = 0.010  # Typically \in [0.001, 0.100]
        # SET simulation parameters
        params["R0"] = R0
        params["rho"] = rho
        params["alpha"] = alpha
        params["eff_disp"] = eff_dispersal
        params["time_horizon"] = 600
        params["domain_sz"] = [200, 200]  # If channel [NxM] where N < M
        # SET simulation settings & boundary conditions
        settings["anim"] = True
        settings["BCD3"] = True  # Percolation condition : if False, simulations will run until pathogen dies
        settings["verbose"] = True
        settings["plt_tseries"] = True
        settings["dyn_plots"] = [True, 1, True]
        # BEGIN
        print("Running: ")
        Results = sg_model.main(settings, params)
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

    elif sim_type == "-ens":
        """
        This code is designed to be changed, use this to generate single lines of how sgm_model behaves.
        Use for simple small lines through phase space.
        """
        R0 = 1  # R0 cases initial cases per day
        alpha = 5  # lattice constant in (m)
        sigma = 50  # dispersal distance in (m)
        eff_dispersal = sigma / alpha  # dispersal in computer units
        domain_sz = [30, 650]
        params["R0"] = R0
        params["alpha"] = alpha
        params["l_time"] = 100
        params["time_horizon"] = 1000
        params["eff_disp"] = eff_dispersal
        settings["BCD3"] = True  # False ==> no percolation BCD
        settings["verbose"] = False  # print time-step information
        params["domain_sz"] = domain_sz
        settings["plt_tseries"] = True
        params["area"] = domain_sz[0] * domain_sz[1] * alpha
        print("Running: LCL ENSEMBLE simulation")
        if 1:  # single plots
            rhos = np.array([0.005, 0.010, 0.015, 0.020, 0.025])
            ts_rt = np.zeros(rhos.shape[0])
            ts_max_d = np.zeros(shape=(rhos.shape[0], params["time_horizon"]))
            i = 0
            for rho in rhos:
                params["rho"] = rho
                Results = sg_model.main(settings, params)
                mortality_, velocity_, max_d_, run_time_, percolation_, population_sz, ts_max_d_ = Results
                ts_rt[i] = run_time_
                ts_max_d[i][:run_time_] = ts_max_d_
                print(i, '/', rhos.shape[0])
                print(' --percolation: {}'.format(percolation_))
                print(' --sim runtime {} (days): '.format(run_time_))
                i += 1
            ts_max_d = ts_max_d / 1000
            np.save('ens_dat', ts_max_d)
            np.save('ens_rt', ts_rt)


