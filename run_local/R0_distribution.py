import os
import sys
import numpy as np
from math import log
import seaborn as sns
import sgm_test.sg_model_test as model
import matplotlib.pyplot as plt


def R0_analytical(beta, rho, sigma, T):
    return (2 * np.pi * beta * rho * sigma ** 2) * ((1 - 2 / 9 * beta) ** T - 1) / log(1 - 2 / 9 * beta)

#
# ------------------ SET settings ------------------  #
#
settings = {"out_path": os.getcwd() + '/animationsData/', "anim": False, "BCD3": False, "verbose": False,
            "save_figs": False, "out_plots": False, "dyn_plots": [False, 1, False],
            "R0_mode": False}
#
# ------------------ SET config parameters ------------------  #
#
params = {}
dispersal_ = 25  # average dispersal distance in (m)
params["alpha"] = 5  # Lattice constant (m)
params["l_time"] = 100  # Life time of disease
eff_dispersal = dispersal_ / params["alpha"]  # Convert the dispersal from m to computer units
params["eff_disp"] = np.round(eff_dispersal, 5)
params["domain_sz"] = [200, 200]  # Size of domain/Lattice boundaries (x\alpha for m^2)
if settings["R0_mode"]:  # Test R0
    params["time_horizon"] = params["l_time"]
elif not settings["R0_mode"]:  # Test normal model dynamics
    params["time_horizon"] = 3650
    settings["BCD3"] = True
    settings["dyn_plots"] = [False, 1, False]
    settings["out_plots"] = False
#
# ------------------ SET main params ------------------  #
#
params['rho'] = 0.02
params["beta"] = 0.005
ensemble_size = 100
ensemble_results = np.zeros(ensemble_size)
print('\n')

for i in range(ensemble_size):
    print('i : {} / {}'.format(i, ensemble_size))
    results = Results = model.main(settings, params)
    ensemble_results[i] = results[4]  # Save mortality

c = 0
for i in np.unique(ensemble_results):
    print('num {} = {}'.format(c, len(np.where(ensemble_results == c)[0])))
    c += 1
print(ensemble_results)

sys.exit()

fig, ax = plt.subplots()
sns.distplot(ensemble_results, hist=True, kde=True)
plt.plot([1, 1], [0, 1], c='b', label='R0 = 1')
mean = ensemble_results.mean()
med = np.sort(ensemble_results)[int(ensemble_size/2)]
plt.plot([mean, mean], [0, 1], c='r', label='ens mean = {}'.format(round(mean, 3)))
plt.plot([med, med], [0, 1], c='g', label='ens median = {}'.format(round(med, 3)))
R0_anal = R0_analytical(params["beta"], params["rho"], eff_dispersal, T=100)
plt.plot([R0_anal, R0_anal], [0, 1], c='brown', label='R0 pred = {}'.format(round(R0_anal, 3)))
plt.title(r'$\rho = {} $'.format(params["rho"]))
plt.legend()
plt.show()