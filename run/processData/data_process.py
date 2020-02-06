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
            self.set_plots = {"ylabel": "Percolation Probability", "scale": 1}
        elif field == "velocity":
            self.set_plots = {"ylabel": r"Velocity $(km/year)$", "scale": 365}
        elif field == 'max_distance_km':
            self.set_plots = {"ylabel": "Distance reached km", "scale": 365}
        elif field == "mortality":
            self.set_plots = {"ylabel": "# dead", "scale": 365}
        elif field == "mortality_ratio":
            self.set_plots = {"ylabel": r"mortality/population - $\chi$", "scale": 365}
        elif field == 'run_time':
            self.set_plots = {"ylabel": "# days elapsed", "scale": 365}

        self.rho_Arr = np.load(data_directory + '/_sim-info/rho_Arr.npy')
        self.beta_Arr = np.load(data_directory + '/_sim-info/beta_Arr.npy')
        self.R0_Arr = np.load(data_directory + '/_sim-info/R0_Arr.npy')
        self.disp_Arr = np.load(data_directory + '/_sim-info/disp_Arr.npy') * self.params["alpha"]
        return

    def plot_line(self, ensemble_data, title, saveFig, saveData):
        # plot R0 line
        fig, ax = plt.subplots(figsize=(7.5, 7))
        for ell in ensemble_data:
            i = 0
            for R0_vs_rho in ell:
                if self.field == "velocity":
                    R0_vs_rho = R0_vs_rho * 365  # km/year

                ax.plot(self.rho_Arr, R0_vs_rho, label=r'$R_0 = $ {}'.format(self.R0_Arr[i]))
                # ax.scatter(self.rho_Arr, R0_vs_rho, s=10)
                i += 1
            plt.xticks(np.round(np.linspace(0, self.rho_Arr[-1], 21), 3), rotation=60)
            ax.set_ylabel(self.set_plots['ylabel'], size=14)
            ax.set_xlabel(r'Tree density $\rho$', size=14)
            ax.grid(True)
            ax.axvline(x=0.050, color='r', alpha=0.50, linestyle='--')
            ax.axvspan(0.001, 0.050, alpha=0.15, color='grey', label=r'Lower $\rho$ regime')
            ax.set_title(title + r"$\ell = $" + str(self.disp_Arr[0]) + ' (m)', size=15)
            plt.legend()
            if saveFig[0]:
                plt.savefig(os.getcwd() + '/' + 'ens-' + self.field + saveFig[1])
            plt.show()

        if saveData[0]:  # Save R0_lines...
            for i, ell_ in enumerate(ensemble_data):
                for j, R0_ in enumerate(ell_):
                    label = 'R0_{}_ell_{}'.format(str(self.R0_Arr[j]).replace('.', '_'), int(self.disp_Arr[i]))
                    np.save(os.getcwd() + '/{}-'.format(self.field) + label, R0_)
            np.save(os.getcwd() + '/' + 'rho-' + self.field + '-mapping' + saveData[1], ensemble_data)

    def plot_2D(self, ensemble_data, saveFig, saveData):
        fig, ax = plt.subplots()
        rho = self.rho_Arr[0]
        extent = [self.beta_Arr[0], self.beta_Arr[-1], self.disp_Arr[0], self.disp_Arr[-1]]
        ensemble_data = ensemble_data[:, :, 0] * self.set_plots["scale"]
        im = ax.imshow(ensemble_data, origin='lower', extent=extent)
        ax.set_ylabel(r"$\ell$", size=14)
        ax.set_xlabel(r"$\beta$", size=14)
        ax.set_aspect("auto")
        ax.set_title(r"$\rho = {}$".format(rho))
        cbar = plt.colorbar(im)
        cbar.set_label(self.set_plots["ylabel"])
        plt.show()


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
        null_count = 0
        for c, file in enumerate(file_list):  # iterate through files
            print('File: {} / {}'.format(c, len(file_list)))
            hpc_core_result = np.load(data_path + '/' + file)
            for repeat in hpc_core_result:  # iterate through repeated results get sum for each file
                ensemble_data = ensemble_data + repeat

        ensemble_data = ensemble_data / ((hpc_core_repeat * len(file_list)) - null_count)  # averaged
        return ensemble_data


if __name__ == '__main__':
    # Data fields saved
    fields = ['max_distance_km', 'mortality', 'mortality_ratio', 'percolation', 'run_time', 'velocity']
    # data_dir = '06-02-2020-HPC-param-sweep-ell-vs-beta'
    data_dir = '06-02-2020-HPC-param-sweep-ell-vs-constBeta'
    field_ = fields[5]
    # Plot data
    plots = Plots(data_dir, field_)
    ensemble_Av = plots.get_ensemble(results_name=data_dir, saveDat=[False, '-delMe'], show_individual=False)
    plots.plot_2D(ensemble_Av, saveFig=False, saveData=False)
    # End
