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
                self.set_plots = {"ylabel": r"$\langle R_0 \rangle$", "scale": 1, }
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

    def get_R0_arr(self, rho, ell, T, beta):
        """
        Get expected R0 as a function of rho, ell, T
        :param rho: tree density
        :param ell: disersal distance normally distributed
        :param T: infectious life-time T
        :return: 2D R0 array
        """
        return (2 * np.pi * beta * rho * ell ** 2) * ((1 - 2 / 9 * beta) ** T - 1) / log(1 - 2 / 9 * beta)

    def plot_rho_line(self, ensemble_data, title, saveFig, save_sg_map):
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
            fig, ax = plt.subplots(figsize=(6.5, 6))
            for j, R0_vs_rho in enumerate(ell):

                # Get simulated data
                ax.plot(self.rho_Arr, R0_vs_rho, c='C'+str(j), linewidth=1)
                ax.scatter(self.rho_Arr[::7], R0_vs_rho[::7], s=7.5, label=r'$\beta = $ {}'.format(self.beta_Arr[j]),
                           c='C'+str(j))

                # Get predicted R0 lines
                R0_arr = Plots.get_R0_arr(self, rho=self.rho_Arr, beta=self.beta_Arr[j], ell=self.disp_Arr[i]/5, T=100)
                # ax.scatter(self.rho_Arr[::5], R0_arr[::5], color='r', s=, alpha=0.5)
                ax.plot(self.rho_Arr[::5], R0_arr[::5], color='r', linewidth=0.35, alpha=1)

            plt.xticks(np.round(np.linspace(0, self.rho_Arr[-1], 11), 3), rotation=30)
            ax.set_ylabel(self.set_plots['ylabel'], size=25)
            ax.set_xlabel(r'$\rho$', size=20)
            # ax.grid(True)
            ax.set_xlim(0, 0.10)
            ax.set_title(title + r"$\ell = $ {}".format(self.disp_Arr[i]) + ' (m)', size=20)
            ax.tick_params(axis='both', size=3, labelsize=14)
            plt.tight_layout()
            plt.legend(prop={'size': 15})
            if saveFig[0]:
                plt.savefig(os.getcwd() + '/' + 'ens-' + self.field + saveFig[1])
            plt.show()

        if save_sg_map[0]:  # Save R0_lines: used in sub-grid mapping...
            for i, ell_ in enumerate(ensemble_data):
                for j, beta_ in enumerate(ell_):
                    label = 'beta_{}_ell_{}'.format(str(self.beta_Arr[j]).replace('.', '_'), int(self.disp_Arr[i]))
                    np.save(os.getcwd() + '/{}-'.format(self.field) + label, beta_)
            np.save(os.getcwd() + '/' + 'rho-' + self.field + '-mapping' + save_sg_map[1], ensemble_data)
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

            ax.set_xlabel(r'Infectivity $\beta$', size=12)
            ax.set_ylabel(self.set_plots["ylabel"], size=12)
            ax.set_title(r"$\rho = $ {}".format(self.rho_Arr[rho]))
            plt.legend()
            plt.savefig('perc-R-{}'.format(i))
            plt.show()
            i += 1

    def plot_2d_average(self, ensemble_data, saveFig, saveData):
        """
        Plot the 2D average i.e. field values z axis (shown as color) dispersal against beta as a
        function of tree density.
        :param ensemble_data:
        :param saveFig:
        :param saveData:
        :return:
        """
        for i in range(ensemble_data.shape[2]):
            fig, ax = plt.subplots(figsize=(12, 10))
            extent = [self.beta_Arr[0], self.beta_Arr[-1], self.disp_Arr[0], self.disp_Arr[-1]]
            data = ensemble_data[:, :, i] * self.set_plots["scale"]
            data = np.where(data < 0.01, np.nan, data)
            im = ax.imshow(data, origin='lower', extent=extent)
            ax.set_ylabel(r"Dispersal $\ell$ (m)", size=20)
            ax.set_xlabel(r"Infectivity $\beta$", size=20)
            ax.set_aspect("auto")
            ax.set_title(r"$\rho = {}$".format(self.rho_Arr[i]), size=20)
            ax.tick_params(axis='x', rotation=30, size=10)
            cbar = plt.colorbar(im)
            cbar.set_label(self.set_plots["ylabel"], size=30)
            ax.set_ylim(10, 30)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.tight_layout()
            if saveFig[0]:
                plt.savefig(os.getcwd()+'/2D-pspace-' + self.field + '-' + str(i))
            plt.show()
        return

    def plot_distribution(self, results_name, saveFig):
        import matplotlib.pyplot as plt
        """
        1) Plot a 2D distribution of R0, or other quantity.
        2) Plot a set of 1D distributions from the data 
        :param resuls_name:
        :return:
        """
        data_path = os.getcwd() + '/' + results_name + '/' + self.field
        file_list = sorted(os.listdir(data_path))
        rho_shape = np.load(data_path + '/' + file_list[0]).shape[3]
        max_R0 = 30  # Define maximum value of R0 for data-set
        distribution_arr = np.zeros(shape=[max_R0, rho_shape])
        for c, file in enumerate(file_list):  # iterate through files
            print('File: {} / {}'.format(c, len(file_list)))
            hpc_core_result = np.load(data_path + '/' + file)
            for rho_value in range(rho_shape):
                R0_dist = hpc_core_result[:, :, :, rho_value].T[0][0]
                if R0_dist.max() >= max_R0:
                    print("ERROR max R0")
                    print(R0_dist.max())
                    sys.exit()
                R0_dist = np.sort(R0_dist)
                R0_values = np.unique(R0_dist)
                R0_count = np.zeros(R0_values.shape)
                for i, R0 in enumerate(R0_values):
                    R0_count[i] = len(np.where(R0_dist == R0)[0])
                distribution_arr[:, rho_value][R0_values.astype(int)] += (R0_count / 1)

        for i in range(rho_shape):
            distribution_arr[:, i] = distribution_arr[:, i] / distribution_arr[:, i].sum()
        extent = [self.rho_Arr[0], self.rho_Arr[-1], 0, max_R0]
        fig, ax = plt.subplots()
        im = ax.imshow(np.where(distribution_arr < 0.01, np.nan, distribution_arr), origin="lower", cmap="jet",
                       aspect="auto", extent=extent, clim=[0, 0.50])
        ax.plot([0, self.rho_Arr[-1]], [1, 1], c='r')
        # ax.set_ylim(0., 30)
        ax.set_title(r'$\ell = $ {}, $\beta = {}$'.format(self.disp_Arr[0], self.beta_Arr[0]), size=18)
        ax.set_xlabel(r'tree density $\rho$', size=13)
        ax.set_ylabel(r'Reproduction number $R_0$', size=13)
        cbar = plt.colorbar(im)
        cbar.set_label(r'$Pr(R_0)$', size=15)
        if saveFig[0]:
            plt.savefig('2D-distribution-'+saveFig[1])
        plt.show()

        fig, ax = plt.subplots()
        for i in [98, 148, 198, 248]:
            ax.plot(distribution_arr[:, i], label=r'$\rho = {}$'.format(round(self.rho_Arr[i], 3)),
                     alpha=0.75)
            ax.scatter(range(max_R0), distribution_arr[:, i], s=10)

        ax.set_xlabel(r'$R_0$', size=15)
        ax.set_ylabel(r'$Pr(R_0)$', size=15)
        ax.set_title(r'$\ell = $ {}, $\beta = {}$'.format(self.disp_Arr[0], self.beta_Arr[0]), size=18)
        plt.legend(prop={'size': 15})
        if saveFig[0]:
            plt.tight_layout()
            plt.savefig('distribution-comparison-'+saveFig[1])
        plt.show()
        return

    def get_ensemble(self, results_name, mode):
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
        if mode == "mean":
            for c, file in enumerate(file_list):  # iterate through files
                print('File: {} / {}'.format(c, len(file_list)))
                hpc_core_result = np.load(data_path + '/' + file)
                for repeat in hpc_core_result:  # iterate through repeated results get sum for each file
                    ensemble_data = ensemble_data + repeat
            ens_size = hpc_core_repeat * len(file_list)
            print('Ensemble size', ens_size)
            ensemble_data = ensemble_data / (ens_size)  # averaged

        elif mode == 'dist':  # If distribution mode then get ALL core results
            ''

        return ensemble_data


def combine_ens(ens_av1, ens_av2):
    import matplotlib as mpl
    """
    Compare critical percolation vs R0 lines
    :param ensemble_Av1: R0 arr
    :param ensemble_Av2: percolation arr
    """
    # sort arr 1

    ens_av1[np.where(ens_av1 < 0.95)] = 0
    ens_av1[np.where(ens_av1 > 1.05)] = 0
    ens_av1[np.where(ens_av1 > 0)] = 1
    # Sort arr 2
    ens_av2[np.where(ens_av2 > 0.01)] = 0
    ens_av2[np.where(ens_av2 > 0)] = 2
    ens = ens_av1 + ens_av2
    ens = np.where(ens == 0, np.nan, ens)
    rho_Arr = [0.02, 0.03]
    cmap = mpl.colors.ListedColormap(["white", "blue", "red"])
    # norm = mpl.colors.BoundaryNorm(np.arange(0, 2), cmap.N)
    for i in range(ens_av1.shape[2]):
        fig, ax = plt.subplots(figsize=(12.5, 8.5))
        extent = [0, 0.02, 5, 30]
        data = ens[:, :, i]
        im = ax.imshow(data, origin='lower', extent=extent, cmap=cmap, clim=[0, 2])
        # data = ensemble_Av2[:, :, i]
        # ax.imshow(data, origin='lower', extent=extent, alpha=0.5)
        ax.set_xlabel(r"Infectivity $\beta$", size=18)
        ax.set_ylabel(r"Dispersal $\ell$", size=18)
        ax.set_aspect("auto")
        ax.set_title(r"$\rho = {}$".format(rho_Arr[i]), size=20)
        ax.tick_params(axis='x', rotation=70)
        cbar = fig.colorbar(im, ticks=np.linspace(0, 3, 4), orientation='vertical')
        cbar.ax.set_yticklabels([r'$\emptyset$', r'$R_0 \approx 1$', r'$\mathcal{P}\ \gtrsim \ 0$'], size=20)
        plt.setp(cbar.ax.get_yticklabels(), rotation=270)
        plt.savefig(os.getcwd() + '/2D-pspace-perc-vs-R0-' + str(i))
        plt.show()
    return


if __name__ == '__main__':
    # Data fields saved
    fields = ['max_distance_km', 'mortality', 'mortality_ratio', 'percolation', 'run_time', 'velocity']
    data_dir = '09-03-2020-HPC-1D-sg-mapping-r0-ensemble-v3'
    # data_dir = '02-03-2020-HPC-2D-phase-percolation'
    # data_dir = '05-03-2020-HPC-1D-sg-mapping-r0-ensemble-v2'
    field_ = fields[1]
    plots = Plots(data_dir, field_)
    # plots.plot_distribution(results_name=data_dir, saveFig=[True, ''])
    ensemble_Av = plots.get_ensemble(results_name=data_dir, mode="mean")
    plots.plot_rho_line(ensemble_Av, title='', saveFig=[True, '-R0'], save_sg_map=[False, ''])
    # plots.plot_2d_average(ensemble_Av, saveFig=[True, ''], saveData=False)

# End
