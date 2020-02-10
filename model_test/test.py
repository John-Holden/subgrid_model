import numpy as np
import matplotlib.pyplot as plt
import sys

def rho_t_func(t_arr, sigma, beta, rho):
    rho_t = np.zeros(t_arr.shape[0])
    for i, t in enumerate(t_arr):
        if i == 0:
            rho_t[i] = rho * (1 - sigma**2/L**2 * 2*np.pi * beta)
        else:
            rho_t[i] = rho_t[i-1] * (1 - sigma**2/L**2 * 2*np.pi * beta)

    if 0:
        plt.plot(t_arr, rho_t)
        plt.title(r'$\beta =$ {}'.format(beta))
        plt.ylabel(r'$\rho$')
        plt.xlabel('t')
        plt.show()
    return rho_t


sigma = 15
L = 500
beta = 0.0100
rho = 0.100
t_arr = np.arange(0, 100, 1)

rho_t = rho_t_func(t_arr, sigma, beta, rho)
r0_t = 2*np.pi*beta*rho*sigma**2 * rho_t

plt.plot(t_arr, r0_t)
plt.title(r'$\beta =$ {}'.format(beta))
plt.ylabel(r'$R_0(t)$')
plt.xlabel('t')
plt.show()



