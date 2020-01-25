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
            self.set_plots = {"xlabel": "Percolation Probability"}
        if field == "velocity":
            self.set_plots = {"xlabel": r"Velocity $(km/year)$"}
        """
        self.rho_Arr = np.load(data_directory + '/_sim-info/rho_Arr.npy')
        self.R0_Arr = np.load(data_directory + '/_sim-info/R0_Arr.npy')
        self.disp_Arr = np.load(data_directory + '/_sim-info/disp_Arr.npy') * self.params["alpha"]
        """
        self.rho_Arr = np.arange(0.001, 0.051, 0.001)
        self.disp_Arr = [100]
        self.R0_Arr = [0.25, 0.50, 1.00]
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
            data = np.load(data_path + '/' + file)
            for repeat in data:  # iterate through repeated results get sum for each file
                ensemble_data = ensemble_data + repeat
            # If True plot all individual data files
            if show_individual:
                for ell in data[0]:
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
        for ell in ensemble_data:
            i = 0
            for R0_vs_rho in ell:
                if self.field == "velocity":
                    R0_vs_rho = R0_vs_rho * 365  # km/year

                plt.plot(self.rho_Arr, R0_vs_rho, label=r'$R_0 = $ {}'.format(self.R0_Arr[i]))
                i += 1

            plt.xticks(self.rho_Arr[::2], rotation=60)
            plt.xlabel(r'Tree density $\rho$', size=12)
            plt.ylabel(self.set_plots["xlabel"], size=12)
            plt.grid(True)
            plt.axvline(x=0.050, color='r', alpha=0.50, linestyle='--')
            plt.axvspan(0.001, 0.050, alpha=0.15, color='grey', label=r'Lower $\rho$ regime')
            plt.title(title + r"$\ell = $" + str(self.disp_Arr[0]) + ' (m)')
            plt.legend()
            if saveFig[0]:
                plt.savefig(os.getcwd() + '/' + 'ens-' + self.field + saveFig[1])

            plt.show()


if __name__ == '__main__':
    # Data fields saved
    fields = ['max_distance_km', 'mortality', 'mortality_ratio', 'percolation', 'run_time', 'velocity']
    # data_dir = '24-01-2020-HPC-test-R0-vel-lines'
    data_dir = '24-01-2020-HPC-low-R0-lines'
    field_ = fields[5]
    # Plot data
    plots = Plots(data_dir, field_)
    ensemble_Av = plots.get_ensemble(results_name=data_dir, saveDat=[False, '-'], show_individual=False)
    plots.plot_field(ensemble_Av, title='Ensemble average: ', saveFig=[False, '-single line'])
    # End
