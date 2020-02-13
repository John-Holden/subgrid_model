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
        # self.R0_Arr = np.load(data_directory + '/_sim-info/R0_Arr.npy')
        self.disp_Arr = np.load(data_directory + '/_sim-info/disp_Arr.npy') * self.params["alpha"]
        return

    def plot_chosen(self):
        dat1 = np.load(os.getcwd()+'/velocity-beta_0_2_ell_25.npy')
        dat2 = np.load(os.getcwd()+'/velocity-beta_0_4_ell_20.npy')
        dat3 = np.load(os.getcwd()+'/velocity-beta_0_6_ell_19.npy')
        plt.plot(self.rho_Arr, dat1, label=r'$\beta = 0.20$, $\ell = 25$')
        plt.plot(self.rho_Arr, dat2, label=r'$\beta = 0.40$, $\ell = 20$')
        plt.plot(self.rho_Arr, dat3, label=r'$\beta = 0.60$, $\ell = 19$')
        plt.plot([0.025, 0.025], [0, 5], c='red', alpha=0.4, label=r'velocity $\approx 5.0\ km/yr$')
        plt.plot([0, 0.025], [5, 5], c='red', alpha=0.4)
        plt.xlabel(r'Tree density $\rho$', size=12)
        plt.ylabel(self.set_plots["ylabel"], size=12)
        plt.grid(alpha=0.5)
        plt.xlim(0, 0.050)
        plt.ylim(0, 10)
        plt.legend()
        plt.savefig('sg-mapping-comparision-part-rho')
        plt.show()
        return

    def plot_line(self, ensemble_data, title, saveFig, saveData):
        # plot R0 line
        if self.field == "velocity":
            ensemble_data = ensemble_data * 365  # km/year
        for i, ell in enumerate(ensemble_data):
            fig, ax = plt.subplots(figsize=(7.5, 7))
            for j, R0_vs_rho in enumerate(ell):
                ax.plot(self.rho_Arr, R0_vs_rho, label=r'$\beta = $ {}'.format(self.beta_Arr[j]))
                ax.scatter(self.rho_Arr, R0_vs_rho, s=10)

            plt.xticks(np.round(np.linspace(0, self.rho_Arr[-1], 21), 3), rotation=60)
            ax.set_ylabel(self.set_plots['ylabel'], size=14)
            ax.set_xlabel(r'Tree density $\rho$', size=14)
            ax.grid(True)
            ax.axvline(x=0.050, color='r', alpha=0.50, linestyle='--')
            ax.axvspan(0.001, 0.050, alpha=0.15, color='grey', label=r'Lower $\rho$ regime')
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

    def plot_2D(self, ensemble_data, saveFig, saveData):

        for i in range(ensemble_data.shape[2]):
            fig, ax = plt.subplots()
            rho = self.rho_Arr[i]
            extent = [self.beta_Arr[0], self.beta_Arr[-1], self.disp_Arr[0], self.disp_Arr[-1]]
            data = ensemble_data[:, :, i] * self.set_plots["scale"]
            # data = np.where(np.logical_and(data>4.5, data<5.5), data, np.nan)
            im = ax.imshow(data, origin='lower', extent=extent)
            ax.set_ylabel(r"$\ell$ (m)", size=14)
            ax.set_xlabel(r"Infectivity $\beta$", size=14)
            ax.set_aspect("auto")
            ax.set_title(r"$\rho = {}$".format(rho))
            cbar = plt.colorbar(im)
            cbar.set_label(self.set_plots["ylabel"], size=10)
            """
            plt.plot([0, 0.2], [25, 25], c='C0')
            plt.plot([0.2, 0.2], [0, 25], c='C0')
            plt.plot([0, 0.40], [20.75, 20.75], c='C1')
            plt.plot([0.4, 0.4], [0, 20.75], c='C1')
            plt.plot([0, 0.60], [19, 19], c='C2')
            plt.plot([0.60, 0.60], [0, 19], c='C2')
            """
            if saveFig[0]:
                plt.savefig(os.getcwd()+'/2D-pspace-'+self.field + '-' + str(i))
            if i ==0:
                plt.show()
            else:
                plt.close()
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
                ensemble_data = ensemble_data + repeat

        ensemble_data = ensemble_data / (hpc_core_repeat * len(file_list))  # averaged
        return ensemble_data


if __name__ == '__main__':
    # Data fields saved
    fields = ['max_distance_km', 'mortality', 'mortality_ratio', 'percolation', 'run_time', 'velocity']
    # data_dir = '06-02-2020-HPC-param-sweep-100ell-vs-100beta-ens-300'
    # data_dir = '07-02-2020-HPC-param-sweep-100ell-vs-100beta-ens-300-small'
    # data_dir = '10-02-2020-HPC-map-line-test'
    data_dir = '12-02-2020-HPC-HresLine-mapping'
    field_ = fields[3]
    # Plot data
    plots = Plots(data_dir, field_)
    ensemble_Av = plots.get_ensemble(results_name=data_dir, saveDat=[False, '-delMe'], show_individual=False)
    plots.plot_line(ensemble_Av, '', saveFig=[True, ''], saveData=[True, ''])
    # plots.plot_2D(ensemble_Av, saveFig=[True, ''], saveData=False)
    # End
