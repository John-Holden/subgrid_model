import numpy as np
from math import log
import matplotlib.pyplot as plt
import sys

"""
Created on Wed May 30 14:19:32 2018
@author: John Holden

Run an analytically found equation of reproductive ratio for our tree model and compare that over different phase
plane values. I.e. R0(ell, beta, rho, T). The derived equation is only a function of time but properly it would also 
be a function of space too, taking into account the dispersal kernel.
"""

beta_Arr = np.linspace(0.0001, 0.0100, 1001)
# rho_Arr = np.linspace(0.0001, 0.10, beta_Arr.shape[0])
ell_Arr = np.linspace(5, 30, beta_Arr.shape[0])/5
rho = 0.0250
# ell = 50/5
alpha = 5
T = 100
R0_Arr = np.zeros(shape=[beta_Arr.shape[0], beta_Arr.shape[0]])

for i, ell in enumerate(ell_Arr):
    for j, beta in enumerate(beta_Arr):
        R0_Arr[i, j] = (2 * np.pi * beta * rho * ell ** 2) * ((1 - 2 / 9 * beta) ** T - 1) / log(1 - 2 / 9 * beta)

fig, ax = plt.subplots()
extent = [0, 0.01, ell_Arr[0]*5, ell_Arr[-1]*5]

im = ax.imshow(np.where(R0_Arr < 1, np.nan, R0_Arr), origin="lower", extent=extent, aspect="auto")


ind = np.where(np.logical_and(R0_Arr > 0.999, R0_Arr < 1.001))

ax.plot(beta_Arr[ind[1]], ell_Arr[ind[0]]*5, color='r', linestyle='--', linewidth=5, label=r"R_0 = 1")
cbar = plt.colorbar(im)
cbar.set_label(r'$R_0$', size=15)
ax.set_ylabel(r'$\ell\ (m)$', size=14)
ax.set_xticks(beta_Arr[::200].round(4))
# ax.tick_params(axis='x', rotation=45)
ax.set_xlabel(r'Infectivity $\beta$', size=14)
ax.set_title(r'$\rho = {}$'.format(rho))
plt.legend()
plt.savefig('phase-plane-')
plt.show()



