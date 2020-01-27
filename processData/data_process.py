import os, sys
import pickle
import numpy as np
import matplotlib.pyplot as plt


class Plots:
    def __init__(self, data_directory, field):
        with open(os.getcwd() + '/' + data_directory + '/_sim-info/sim-parameters.pickle', 'rb') as handle:
            self.params = pickle.load(handle)

        with open(data_directory + '/_sim-info/sim-settings.pickle', 'rb') as handle:
            self.settings = pickle.load(handle)

        self.field = field
        # Set axis
        if field == "percolation":
            self.set_plots = {"ylabel": "Percolation Probability"}
        if field == "velocity":
            self.set_plots = {"ylabel": r"Velocity $(km/year)$"}

        self.rho_Arr = np.load(data_directory + '/_sim-info/rho_Arr.npy')
        self.R0_Arr = np.load(data_directory + '/_sim-info/R0_Arr.npy')
        self.disp_Arr = np.load(data_directory + '/_sim-info/disp_Arr.npy') * self.params["alpha"]

        """
        rho_Arr_low = np.arange(0.001, 0.05, 0.001)  # Tree density range
        rho_Arr_med = np.arange(0.051, 0.10, 0.010)
        self.rho_Arr = np.hstack([rho_Arr_low, rho_Arr_med])
        self.disp_Arr = [50]
        self.R0_Arr = [1, 2, 10]"""
        return

    def get_ensemble(self, results_name, saveDat, show_individual):
        """
        :param results_name: directory to shape load and process into a npy file
        :param saveDat: bool, if True save to file
        :return: arr ensemble_Av, array of simulation data
        """
        data_path = os.getcwd() + '/' + results_name + '/' + self.field
        file_list = sorted(os.listdir(data_path))
        dim_ = np.load(data_path + '/' + file_list[0]).shape[1:4]  # drop extra dimension for repeats
        ensemble_data = np.zeros(dim_)
        hpc_core_repeat = np.load(data_path + '/' + file_list[0]).shape[0]
        for c, file in enumerate(file_list):  # iterate through files
            print('File: {} / {}'.format(c, len(file_list)))
            hpc_core_result = np.load(data_path + '/' + file)
            for repeat in hpc_core_result:  # iterate through repeated results get sum for each file
                try:
                    ensemble_data = ensemble_data + repeat
                except:
                    print("Faulty result : {}".format(c))
                    print('supposed shape : {} , shape of data {}'.format(dim_, hpc_core_result.shape))
                    continue

            # If True plot all individual data files
            if show_individual:
                for ell in hpc_core_result[0]:
                    i = 0
                    for R0_vs_rho in ell:
                        plt.plot(self.rho_Arr, R0_vs_rho, label=r'$R_0 = $ {}'.format(self.R0_Arr[i]))
                        i += 1

                plt.legend()
                plt.title('i = {}'.format(c))
                plt.show()

        ensemble_data = ensemble_data / (hpc_core_repeat * len(file_list))  # averaged
        if saveDat[0]:
            np.save(os.getcwd()+'/' + 'ens-' + self.field + saveDat[1], ensemble_data)
        return ensemble_data

    def plot_field(self, ensemble_data, title, saveFig):
        # plot R0 line
        fig, ax = plt.subplots(figsize=(7.5, 7))
        for ell in ensemble_data:
            i = 0
            for R0_vs_rho in ell:
                if self.field == "velocity":
                    R0_vs_rho = R0_vs_rho * 365  # km/year

                ax.plot(self.rho_Arr, R0_vs_rho, label=r'$R_0 = $ {}'.format(self.R0_Arr[i]))
                ax.scatter(self.rho_Arr, R0_vs_rho, s=15)
                i += 1

            plt.xticks(np.round(np.linspace(0, self.rho_Arr[-1], 21), 3), rotation=60)
            ax.set_ylabel(self.set_plots['ylabel'], size=14)
            ax.set_xlabel(r'Tree density $\rho$', size=14)
            ax.grid(True)
            ax.axvline(x=0.050, color='r', alpha=0.50, linestyle='--')
            ax.axvspan(0.001, 0.050, alpha=0.15, color='grey', label=r'Lower $\rho$ regime')
            ax.set_title(title + r"$\ell = $" + str(self.disp_Arr[0]) + ' (m)')
            plt.legend()
            if saveFig[0]:
                plt.savefig(os.getcwd() + '/' + 'ens-' + self.field + saveFig[1])
            plt.show()


if __name__ == '__main__':
    # Data fields saved
    fields = ['max_distance_km', 'mortality', 'mortality_ratio', 'percolation', 'run_time', 'velocity']
    # data_dir = '25-01-2020-HPC-debug-3R0-1ell-L250'
    data_dir = '26-01-2020-HPC-ell50m-part-rho-range'
    # data_dir = '26-01-2020-HPC-ell50m-full-rho-range'
    field_ = fields[5]
    # Plot data
    plots = Plots(data_dir, field_)
    ensemble_Av = plots.get_ensemble(results_name=data_dir, saveDat=[False, '-'], show_individual=False)
    plots.plot_field(ensemble_Av, title='Ensemble average: ', saveFig=[True, '-part-rho-lowR0'])
    # End
