import numpy as np
from math import log
import matplotlib.pyplot as plt
import sys

beta_Arr = np.linspace(0.0001, 0.010, 100)
# rho_Arr = np.linspace(0.0001, 0.10, beta_Arr.shape[0])
ell_Arr = np.linspace(5, 30, beta_Arr.shape[0])/5
rho = 0.10
#ell = 50/5
T = 100
R0_Arr = np.zeros(shape=[beta_Arr.shape[0], beta_Arr.shape[0]])

for i, ell in enumerate(ell_Arr):
    for j, beta in enumerate(beta_Arr):
        R0_Arr[i, j] = (2 * np.pi * beta * rho * ell ** 2) * ((1 - 2 / 9 * beta) ** T - 1) / log(1 - 2 / 9 * beta)

fig, ax = plt.subplots()
extent = [beta_Arr[0], beta_Arr[-1], ell_Arr[0]*5, ell_Arr[-1]*5]
im = ax.imshow(np.where(R0_Arr < 1, np.nan, 1), origin="lower", extent=extent, aspect="auto")

cbar = plt.colorbar(im)
cbar.set_label(r'$R_0$', size=15)
ax.set_ylabel(r'$\rho$', size=14)
ax.set_xticks(beta_Arr[::10].round(4))
ax.tick_params(axis='x', rotation=45)
ax.set_xlabel(r'$\beta$', size=14)
ax.set_title(r'$\ell = {}$'.format(ell))
plt.show()



