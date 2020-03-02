import os, sys
import pickle
import numpy as np
from math import log
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
            self.set_plots = {"ylabel": "Distance reached km", "scale": 1}
        elif field == "mortality":
            if self.settings["R0_mode"]:
                self.set_plots = {"ylabel": r"$R_0$", "scale": 1}
            else:
                self.set_plots = {"ylabel": "# dead", "scale": 1}
        elif field == "mortality_ratio":
            self.set_plots = {"ylabel": r"mortality/population - $\chi$",  "scale": 1}
        elif field == 'run_time':
            self.set_plots = {"ylabel": "# days elapsed",  "scale": 1}

        self.rho_Arr = np.load(data_directory + '/_sim-info/rho_Arr.npy')
        self.beta_Arr = np.load(data_directory + '/_sim-info/beta_Arr.npy')
        # self.R0_Arr = np.load(data_directory + '/_sim-info/R0_Arr.npy')
        self.disp_Arr = np.load(data_directory + '/_sim-info/disp_Arr.npy') * self.params["alpha"]
        return

    def get_R0_arr(self, rho, ell, T):
        """
        Get expected R0 as a function of rho, ell, T
        :param rho: tree density
        :param ell: disersal distance normally distributed
        :param T: infectious life-time T
        :return: 2D R0 array
        """
        R0_arr = np.zeros(self.beta_Arr.shape[0])
        for i in range(self.beta_Arr.shape[0]):
            beta = self.beta_Arr[i]
            R0_arr[i] = (2 * np.pi * beta * rho * ell ** 2) * ((1 - 2 / 9 * beta) ** T - 1) / log(1 - 2 / 9 * beta)
        return R0_arr

    def plot_chosen(self):
        """
        Plot selected data arrays
        :return:
        """
        dat1 = np.load(os.getcwd()+'/beta-perc-line_ell50_0_rho0_05.npy')
        dat2 = np.load(os.getcwd()+'/beta-perc-line_ell50_0_rho0_025.npy')
        plt.plot(self.beta_Arr, dat1, label=r'$\rho = 0.050$, $\ell = 50$')
        plt.plot(self.beta_Arr, dat2, label=r'$\beta = 0.025$, $\ell = 50$')
        plt.xlabel(r'Tree density $\rho$', size=12)
        plt.ylabel(self.set_plots["ylabel"], size=12)
        plt.grid(alpha=0.5)
        plt.legend()
        plt.savefig('sg-mapping-comparision-part-rho')
        plt.show()
        return

    def plot_rho_line(self, ensemble_data, title, saveFig, saveData):
        """
        Plot figures for different values of ell plot rho x-axis against field values for different beta values
        :param ensemble_data: HPC generated data
        :param title:
        :param saveFig: [Bool, '...']
        :param saveData: Bool, '...']
        :return:
        """
        if self.field == "velocity":
            ensemble_data = ensemble_data * 365  # km/year
        for i, ell in enumerate(ensemble_data):
            fig, ax = plt.subplots(figsize=(7.5, 7))
            for j, R0_vs_rho in enumerate(ell):
                ax.plot(self.rho_Arr, R0_vs_rho)
                ax.scatter(self.rho_Arr, R0_vs_rho, s=10, label=r'$\beta = $ {}'.format(self.beta_Arr[j]))
                # rho = self.rho_Arr[np.where(R0_vs_rho > 0.99)[0][0]]
                # ax.plot([rho, rho], [0, 1], c='r', label=r'@ 99% perc: $\rho = ${}'.format(round(rho, 3)))
                # rho = self.rho_Arr[np.where(R0_vs_rho > 0.50)[0][0]]
                # ax.plot([rho, rho], [0, 1], c='g', label=r'@ 50% perc: $\rho = ${}'.format(round(rho, 3)))
                rho = self.rho_Arr[np.where(R0_vs_rho > 0)[0][0]]
                ax.plot([rho, rho], [0, 1], c='C'+str(j), label=r'@ >0% perc: $\rho = ${}'.format(round(rho, 3)))

            plt.xticks(np.round(np.linspace(0, self.rho_Arr[-1], 21), 3), rotation=60)
            ax.set_ylabel(self.set_plots['ylabel'], size=14)
            ax.set_xlabel(r'Tree density $\rho$', size=14)
            ax.grid(True)
            ax.set_xlim(0, 0.10)
            ax.set_ylim(0, 1.1)
            # ax.axvspan(0.001, 0.050, alpha=0.15, color='grey', label=r'Lower $\rho$ regime')
            ax.set_title(title + r"$\ell = $" + str(self.disp_Arr[i]) + ' (m)', size=15)
            if saveFig[0]:
                plt.savefig(os.getcwd() + '/' + 'ens-' + self.field + saveFig[1])
            plt.legend()
            plt.show()

        if saveData[0]:  # Save R0_lines...
            for i, ell_ in enumerate(ensemble_data):
                for j, beta_ in enumerate(ell_):
                    label = 'beta_{}_ell_{}'.format(str(self.beta_Arr[j]).replace('.', '_'), int(self.disp_Arr[i]))
                    np.save(os.getcwd() + '/{}-'.format(self.field) + label, beta_)
            np.save(os.getcwd() + '/' + 'rho-' + self.field + '-mapping' + saveData[1], ensemble_data)
        return

    def plot_beta_line(self, ensemble_data, title, saveFig, saveData):
        """
        Plot figures for one value of rho, beta x-axis against the field values for different lines in ell
        :param ensemble_data:
        :param title:
        :param saveFig:
        :param saveData:
        :return:
        """
        i = 0
        colors = ['C0', 'C1', 'C2']
        for rho in range(ensemble_data.shape[2]):
            fig, ax = plt.subplots()
            rho_dat = ensemble_data[:, :, rho]
            j = 0
            for ell in rho_dat:
                # np.save('beta-perc-line_ell{}_rho{}'.format(str(self.disp_Arr[j]).replace('.', '_'),
                #                                            str(self.rho_Arr[i]).replace('.', '_')), ell)
                ax.plot(self.beta_Arr, ell, label=r"$\ell = $ {} (m)".format(self.disp_Arr[j]))
                R0_arr = self.get_R0_arr(rho=self.rho_Arr[i], ell=self.disp_Arr[j]/5, T=100)
                R0_1 = np.where(np.logical_and(R0_arr > 0.95, R0_arr < 1.05))
                R0_1 = R0_1[0][0]
                ax.plot([self.beta_Arr[R0_1], self.beta_Arr[R0_1]], [0, 1], c=colors[j], linestyle='--', alpha=0.5,
                        label=r"$R_0(\ell, \rho, \beta, T) \approx 1 $".format(self.disp_Arr[j]))
                j += 1
            ax.set_xlabel(r"Infectivity $\beta$",size=12)
            ax.set_ylabel(r"Percolation Pr", size=12)
            ax.set_title(r"$\rho = $ {}".format(self.rho_Arr[rho]))
            plt.legend()
            plt.savefig('perc-R-{}'.format(i))
            plt.show()
            i += 1

    def plot_2d(self, ensemble_data, saveFig, saveData):
        """
        Plot the 2D average i.e. field values z axis (shown as color) dispersal against beta as a
        function of tree density.
        :param ensemble_data:
        :param saveFig:
        :param saveData:
        :return:
        """
        for i in range(ensemble_data.shape[2]):
            fig, ax = plt.subplots()
            extent = [self.beta_Arr[0], self.beta_Arr[-1], self.disp_Arr[0], self.disp_Arr[-1]]
            data = ensemble_data[:, :, i] * self.set_plots["scale"]
            data = np.where(data == 0, np.nan, data)
            im = ax.imshow(data, origin='lower', extent=extent)
            ax.set_ylabel(r"$\ell$ (m)", size=14)
            ax.set_xlabel(r"Infectivity $\beta$", size=14)
            ax.set_aspect("auto")
            ax.set_title(r"$\rho = {}$".format(self.rho_Arr[i]))
            ax.tick_params(axis='x', rotation=70)
            cbar = plt.colorbar(im)
            cbar.set_label(self.set_plots["ylabel"], size=15)
            if saveFig[0]:
                plt.savefig(os.getcwd()+'/2D-pspace-' + self.field + '-' + str(i))
            plt.show()
        return

    def get_ensemble(self, results_name, saveDat, show_individual):
        """
        Get the ensemble average from HPC stored directory. Sum over different hpc cores and
        in-core number of repeats.
        :param saveDat: bool, if True save to file
        :param results_name: directory to shape load and process into a npy file
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
                ensemble_data = ensemble_data + repeat

        ensemble_data = ensemble_data / (hpc_core_repeat * len(file_list))  # averaged
        return ensemble_data


if __name__ == '__main__':
    # Data fields saved
    fields = ['max_distance_km', 'mortality', 'mortality_ratio', 'percolation', 'run_time', 'velocity']
    # data_dir = '07-02-2020-HPC-param-sweep-100ell-vs-100beta-ens-300-small'
    # data_dir = '28-02-2020-HPC-2D-phase-R0'
    data_dir = '02-03-2020-HPC-1D-sg-mapping'
    field_ = fields[3]
    # Plot data
    plots = Plots(data_dir, field_)
    ensemble_Av = plots.get_ensemble(results_name=data_dir, saveDat=[False, '-delMe'], show_individual=False)
    plots.plot_rho_line(ensemble_Av, title='', saveFig=[True, '2'], saveData=[False, ''])
    # plots.plot_2d(ensemble_Av, saveFig=[False, ''], saveData=False)

# End
