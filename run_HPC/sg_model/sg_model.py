"""
Created on Mon Jan 15 15:30:51 2018
@author: b7068818

This file is to be called by main_SSTLM_phase.py in the parent directory.
"""
import numpy as np
import time
import os
import sys


class Plots(object):
    def __init__(self, beta, rho):
        self.beta = beta
        self.rho = rho

    def save_settings(self, parameters, settings, output_path):
        """
        write simulation details to file
        :param parameters: parameters used by physical sg_model
        :param settings: simulation setup and running options (different from physical values.)
        :param output_path: save txt location
        :return:
        """
        with open(os.path.join(output_path, "parameter_and_settings_info.txt"), "w+") as info_file:
            info_file.write("______Parameter settings_______" + "\n")
            for parameter in parameters:
                if parameter == "domain_sz":
                    # Save dimension on two separate lines
                    sz = parameters[parameter]
                    info_file.write('row_dim' + ':' + str(sz[0]) + '\n')
                    info_file.write('col_dim' + ':' + str(sz[1]) + '\n')

                else:
                    info_file.write(parameter + ':' + str(parameters[parameter]) + '\n')

            info_file.write("\n" + "______Simulation parameters_______" + "\n")
            for setting in settings:
                info_file.write(setting + ':' + str(settings[setting]) + '\n')

    def plot_tseries(self, metric, labels, fit, saves_):
        import matplotlib.pyplot as plt
        """
        Plot the time series of disease progression
        :param metric: distance or number-infected
        :param labels: axis labels dependent on input
        :param fit: if true, metrics will be fitted to a function.
        :return: None
        """
        save, save_name = saves_
        rho_str, beta_str = str(self.rho), str(round(self.beta, 3))
        fig, ax = plt.subplots()
        x = np.arange(0, len(metric), 1)
        ax.plot(x, metric, alpha=0.5, label=r'$\rho = $' + rho_str + r' $\beta = $' + beta_str)
        ax.scatter(x, metric, s=0.5)
        if fit:
            # Fit a function to the metric
            from scipy.optimize import curve_fit

            def exp(x, b):
                # use a simple exponential
                return np.exp(b * x)
            x = np.arange(metric.shape[0])
            try:
                print('fitting...')
                popt, pcov = curve_fit(exp, x, metric)
                r_exp, err = popt.round(3)[0], pcov.round(9)[0]
                label_ = r'Fitted: $r = {},\ = err = {}$'
                ax.plot(x, exp(x, r_exp), label=label_.format(r_exp, err))
            except:
                print('...parameters not found!')
        if "log" in labels["ylabel"]:
            ax.set_yscale('log')

        ax.set_xlabel(labels["xlabel"])
        ax.set_ylabel(labels["ylabel"])
        ax.set_title(labels["title"])
        plt.legend()
        plt.grid()
        if save:
            plt.savefig(save_name)
        plt.show()
        return


class SubGrid(object):
    def __init__(self, parameters, settings):
        """
        :param parameters: dictionary, keys are strings or parameter names, values are parameter values
        """
        np.random.seed()
        dim = parameters["domain_sz"]  # Dimension of domain
        tree_dist = np.zeros(shape=(dim[0], dim[1]))
        population_size = int(parameters["rho"] * dim[0] * dim[1])  # Theoretical tree population size
        p = 0
        # ------ Initialise tree population ------ #
        while p < population_size:
            # Seed population to exact population site-by-site
            rand_x = np.random.randint(0, dim[0])
            rand_y = np.random.randint(0, dim[1])
            if tree_dist[rand_x, rand_y] == 1:
                pass
            else:
                tree_dist[rand_x, rand_y] = 1
                p += 1
        try:  # Check seeded P size == density given P size
            assert len(np.where(tree_dist == 1)[0]) == population_size
        except:
            # Errors
            print("P actual = ", len(np.where(tree_dist == 1)[0]))
            print("P theory = ", population_size)
            sys.exit("error...")

        if dim[0] != dim[1]:  # Channel geometry
            try:
                assert dim[0] < dim[1]  # MxN where M < N
            except:
                print("Error dim[MxN] = {}, M !< N.".format(dim))
                sys.exit("Exiting...")

        # ------ Set simulation parameters & Data structures ------ #
        epi_cx, epi_cy = int(dim[0]/2), int(dim[1]/2)  # epicenter
        infected = np.zeros(dim)  # Infected field
        tree_dist[0], tree_dist[-1], tree_dist[:, 0], tree_dist[:, -1] = [0, 0, 0, 0]  # Boundary conditions
        self.dim = dim  # dimension of lattice
        self.epi_c = [epi_cx, epi_cy]  # epicenter locations
        self.population = population_size  # the number of trees in the population at T=0
        self.removed = np.zeros(dim)  # the removed field storing locations of all dead trees
        self.rho = parameters['rho']  # the Tree density
        self.eff_disp = parameters["eff_disp"]  # the 'effective' dispersal distance in computer unit
        self.time_f = parameters["time_horizon"]  # the time-epoch BCD, simulation stops if runtime exceeds this
        self.survival_times = parameters['l_time'] + 1 * np.ones(dim)  # the survival time of each lattice point
        self.pre_factor = 2 * np.pi * (parameters["eff_disp"]**2)  # the dispersal normalisation constant
        self.beta = parameters["beta"]  # The infectivity rate in units day^-1
        self.alpha = parameters["alpha"]  # Lattice parameter
        try:
            # Check beta satisfies a probability \in [0, 1]
            assert self.beta < 1
        except:
            print(self.eff_disp, ' disp factor')
            print(self.pre_factor, ' pre factor')
            print(self.beta, ' beta')
            print("Unphysical probability")
            sys.exit("error...")

        # ------ Get distance matrix for metrics ------ #
        x_rows, y_cols = np.arange(0, dim[0]), np.arange(0, dim[1])  # pathogen evolution in (m)
        x_arr, y_arr = np.meshgrid(x_rows, y_cols)
        latitude_ar = (x_arr - self.epi_c[0])
        longitude_ar = (y_arr - self.epi_c[1])
        dist_map = np.sqrt(np.square(longitude_ar) + np.square(latitude_ar)).T
        self.dist_map = dist_map  # Distance map of domain defined from the epicenter
        # Get metric structs
        self.percolation = 0  # Percolation status
        self.max_d_Arr = np.zeros(parameters["time_horizon"])   # Time-series 'max distance' array,
        self.t_debug_Arr = np.zeros(parameters["time_horizon"])  # Time-series physical time vs model time
        self.n_infected_Arr = np.zeros(parameters["time_horizon"] + 1)  # Time-series # infections/step
        self.parameters = parameters
        # Set mode for finding either reproductive ratio R0 or local spreading-dynamics
        if settings["R0_mode"]:
            self.time_f = parameters['l_time'] + 1  # Set sim run-time to be equal to life-time of infection
            settings["BCD3"] = False  # Set percolation BCD to False
            infected[epi_cx, epi_cy] = 1  # Set 1 epicenter to infected status
            tree_dist[epi_cx, epi_cy] = 0  # Remove tree at epicenter
        elif not settings["R0_mode"]:
            infected[epi_cx - 3:epi_cx + 3, epi_cy - 3:epi_cy + 3] = 1  # Set epicenters to infected status
            tree_dist[epi_cx - 3:epi_cx + 3, epi_cy - 3:epi_cy + 3] = 0  # Remove susceptible trees at epicenter

        self.infected = infected  # array containing the locations of infected trees
        self.susceptible = tree_dist  # the susceptible field containing information of all healthy trees
        self.population_init = len(np.where(tree_dist == 1)[0])  # Healthy tree # at t = 0
        # ------ Set simulation settings ------ #
        self.path_2_save = os.getcwd() + '/animationsData/raw_data/'
        self.BCD3 = settings["BCD3"]
        self.animate = settings["anim"]
        self.R0_mode = settings["R0_mode"]  # If testing for reproduction number
        self.verbose = settings["verbose"]  # control output print-to-screens
        self.dyn_plots = settings["dyn_plots"]  # control settings to 'dynamic-plots' .png files
        self.plt_tseries = settings["plt_tseries"]  # plot time-series results
        self.settings = settings

    def save_label(self, step):
        """
        Use this to save under in %4d format - to be used in animate.sh
        :param step: current time-step of the simulation
        :return:
        """
        if step < 10:
            return '000' + str(step)
        elif step < 100:
            return '00' + str(step)
        elif step < 1000:
            return '0' + str(step)
        elif step == 1000:
            return str(step)

    def d_metrics(self, inf_ind):
        """
        :param inf_ind: array-like, all the indicies of infected coordinates
        :return: mean_d: float, the mean distance of infected points
                 max_d: float, the maximum distance travelled by the pathogen
        """
        distances = self.dist_map[inf_ind]
        return distances.max()

    def get_new_infected(self, infected, susceptible):
        """
        The algorithm used to find the newly infected trees after each time-step.
        :param p_infected: array-like dtype=int, infected field, values taken  \in [1, T] where T = infectious lifetime
        :param susceptible: array-like dtype=int susceptible field, values taken \in [0, 1]
        :return: array-like, the NEW-INFECTED cells in the domain
        """
        from scipy.ndimage import gaussian_filter
        # GET All infected cells as 1's
        # -- infected field increases in time so have to reduce to a value of 1
        infected = np.array(infected > 0).astype(float)
        infected_ind = np.where(infected == 1)
        num_infected = len(infected_ind[0])
        pr_S_S = np.ones(infected.shape)  # Probability of remaining in S
        std3 = int(3*self.eff_disp)  # Truncation distance, set to 3 standard deviations
        subset = np.zeros(shape=[2*std3 + 1, 2*std3 + 1])  # init empty subset of 3 standard deviations
        ones = np.ones(subset.shape)
        subset[std3, std3] = 1  # set mid-point to infectious state
        # Blur sub-field with Gaussian kernel
        blurred_field = self.beta * self.pre_factor * gaussian_filter(subset,
                                                                      sigma=[self.eff_disp, self.eff_disp],
                                                                      truncate=3.0)
        # Iterate over each infected lattice position
        for inf in range(num_infected):
            # For each infected position apply gaussian filter and get Pr(S --> S)
            row, col = infected_ind[0][inf], infected_ind[1][inf]  # row, col of infected
            dim = pr_S_S[row - std3:row + std3 + 1, col - std3:col + std3 + 1].shape
            # If infected tree within boundary
            if 2*std3+1 == dim[0] and 2*std3+1 == dim[1]:
                pr_S_S[row-std3:row+std3+1, col-std3:col+std3+1] = \
                    pr_S_S[row-std3:row+std3+1, col-std3:col+std3+1] * (ones - blurred_field)
            # If infected tree not within 3 standard deviations of the boundary pass
            else:  # If infected tree is close to lattice boundary, then pass
                # todo test if this changes percolation result!
                pass

        pr_S_I = np.ones(pr_S_S.shape) - pr_S_S  # Find probability of transitioning to infected
        new_infected = np.where(pr_S_I > np.random.uniform(0, 1, size=infected.shape), 1, 0)  # Get new infected
        new_infected = new_infected * susceptible  # Take away non-empty cells
        return new_infected

    def main_run(self):
        """
        The main function which simulates the pathogen spreading. The sg_model comprises a monte carlo simulation
        of non-local dispersal between trees. First a while loop is triggered where the sg_model algorithm is implemented.
        After the simulation ends a series of time-series plots can be plotted to show disease progression.

        :param settings: dict, simulation settings controls what type of simulation is run_HPC and how
        :param parameters: dict, stores sg_model parameters
        :param domain: array-like, this is the field which is to be processed into an effect landscape
        :return: (1) float, num_removed: the number of trees killed in the simulation "mortality"
                 (2) float, max_distance: this is the maximum distance reached by the pathogen
                 (3) int, time-step: this is the time-elapsed (in computer units)
                 (4) binary, percolation: this is the status which informs the user if the pathogen has travelled to the
                     lattice boundary and triggered the simulation to end (a threshold type-behaviour).
        """
        np.random.seed()
        in_progress, time_step, num_infected = [True, 0, 1]  # Each time-step scaled as days
        if self.verbose:
            dist = round(self.max_d_Arr.max() * self.alpha, 3)
            print("...START...")
            print("max d = {} (m) | Infected = {}".format(dist, num_infected))
        # ________________Run Algorithm________________ #
        while in_progress:
            if self.verbose:
                t_0 = time.clock()
            # --- GET field + R0 @ t+1 --- #
            if self.R0_mode:
                # This indent prevents secondary infections from infecting other lattice positions
                new_infected = 2 * self.get_new_infected(infected=self.infected, susceptible=self.susceptible)
                new_removed = np.array((new_infected > 0),
                                       dtype=int)  # Transition newly infected trees straight to R class
                self.removed = (self.removed + new_removed) > 0  # Add new_removed cells to removed class
                self.susceptible = self.susceptible * (
                    np.logical_not(self.removed == 1))  # Remove infected from SUSCEPTIBLE class
                infected_ind = np.where(self.infected > 0)
                num_infected = len(infected_ind[0])
                num_removed = len(np.where(self.removed == 1)[0])

            # --- GET fields epidemic @ t+1 --- #
            elif not self.R0_mode:
                # This indent is the dynamics for a normal outbreak, secondary infected trees can infect others
                new_infected = 2 * self.get_new_infected(infected=self.infected, susceptible=self.susceptible)
                self.infected = self.infected + (self.infected > 0) + new_infected  # Transition to INFECTED class
                new_removed = np.array(self.infected == self.survival_times, dtype=int)  # Transition to REMOVED class
                self.removed = (self.removed + new_removed) > 0  # Add new_removed cells to removed class
                self.susceptible = self.susceptible * (
                    np.logical_not(self.infected > 1))  # Remove infected from SUSCEPTIBLE class
                self.infected = self.infected * (np.logical_not(new_removed == 1))  # remove dead trees from Infected class
                infected_ind = np.where(self.infected > 0)
                num_infected = len(infected_ind[0])

            # --- IF true get plots --- #
            if self.dyn_plots[0]:
                if time_step % self.dyn_plots[1] == 0:
                    T = SubGrid.save_label(step=time_step)
                    np.save(self.path_2_save + T, np.array([self.susceptible, self.infected, self.removed]))

            # ------ CHECK boundary conditions (BCDs) ------ #
            # BCD1 : disease dies, sim ends & percolation taken as negative percolation
            if num_infected == 0:
                self.percolation = 0
                break
            # BCD2: disease doesnt die but travels slowly & taken as neutral percolation
            if time_step == self.time_f:
                self.percolation = 0
                break
            # --- GET metrics --- #
            max_d = SubGrid.d_metrics(self, inf_ind=infected_ind)  # GET average and mean distance travelled by pathogen
            self.max_d_Arr[time_step] = max_d
            self.n_infected_Arr[time_step] = num_infected
            if self.verbose:  # Print out step max distance reached and #Infecteds
                dist = round(self.max_d_Arr.max() * self.alpha, 3)
                print("  Step: {}  | max d = {} (m) | #Infected = {}".format(time_step, dist, num_infected))
            # BCD3 If distance exceeds boundary then take as positive percolation
            if self.BCD3:
                if self.dim[0] == self.dim[1]:  # Square geometry
                    if max_d > (self.dim[0] / 2 - 10) or max_d > (self.dim[1] / 2 - 10):
                        # Vertical or Lateral percolation
                        self.percolation = 1
                        break
                else:  # Channel Geometry:
                    if max_d > (self.dim[1] / 2 - 25):
                        self.percolation = 1
                        break
            if self.verbose:
                t_f = time.clock()
                self.t_debug_Arr[time_step] = t_f - t_0
                print("  Time elapsed in loop: {}".format(round(t_f - t_0, 4)))
            time_step += 1
            # __________ITERATION COMPLETE_________ #

        # ________________END ALGORITHM________________ #
        # Get time series data
        max_d_reached = self.max_d_Arr.max() / 1000  # Maximum distance reached by the pathogen in (km)
        num_removed = len(np.where(self.removed == 1)[0])  # R (-1 from initially infected)
        velocity = max_d_reached / (time_step + 1)  # get velocity in (km/day)
        # GENERATE time series output plots over single simulation run_HPC
        if self.animate:
            t_debug = self.t_debug_Arr[:time_step]
            ts_num_infected = self.n_infected_Arr[: time_step + 1]
            ts_max_d = self.max_d_Arr[: time_step + 1] * self.alpha
            plot_cls = Plots(self.beta, self.rho)  # Plots module contains plotting functions
            plot_cls.save_settings(self.parameters, self.settings, self.path_2_save)
            if self.plt_tseries:
                max_pos = round(self.dist_map.max() * self.alpha / 1000, 4)
                max_d_reached = round(max_d_reached / 1000, 4)
                print('Step: {}, max d reached = {} (km), max d possible = {} (km)'.format(time_step, max_d_reached,
                                                                                           max_pos))
                # Plot max d metric
                label = {'title': "max d distance", 'xlabel': 'days', 'ylabel': 'distance (km)'}
                plot_cls.plot_tseries(metric=ts_max_d, labels=label, fit=False, saves_=[False, None])
                # Plot number of infected (SIR)
                label = {'title': "num infected", 'xlabel': 'days', 'ylabel': '# trees'}
                plot_cls.plot_tseries(metric=ts_num_infected, labels=label, fit=False, saves_=[False, 'num_inf_0'])
                # Plot log number of infected
                label = {'title': "(log) num infected", 'xlabel': 'days', 'ylabel': 'log # trees'}
                plot_cls.plot_tseries(metric=ts_num_infected, labels=label, fit=False, saves_=[False, 'num_inf_0'])
                # Plot physical computer runtime stats
                label = {'title': "computer runtime", 'xlabel': 'step', 'ylabel': 'physical runtime'}
                plot_cls.plot_tseries(metric=t_debug, labels=label, fit=False, saves_=[False, 'rt_0'])
                # IF True, save time-series data to file
                save_tseries = False
                if save_tseries:
                    beta = round(self.beta, 2)
                    name = 'b_' + str(beta).replace('.', '-') + '_r_' + str(self.rho).replace('.', '-') + \
                           '_L_' + str(self.eff_disp).replace('.', '-')

                    np.save('max_d_' + name, ts_max_d)

        # ##### COMPLETE: Individual realisation complete & metrics gathered # #####
        return num_removed, velocity, max_d_reached, time_step, self.percolation, self.population




