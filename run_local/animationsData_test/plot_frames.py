import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import os, sys
from scipy.ndimage import gaussian_filter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Circle

"""
This script is called by animate.sh and generates the figures which are animated via ffmpeg.
simulation settings area read in and displayed on each separate figure.
"""


def save_nm(step):
    """
    :param step: time in simulation, convert this to %000d format and save
    :return: string, save-label of step
    """
    if step < 10:
        return '000' + str(step)
    elif step < 100:
        return '00' + str(step)
    elif step < 1000:
        return '0' + str(step)


path = os.getcwd() + '/raw_data/'
save_name = 'frame-'
files = sorted(os.listdir(os.getcwd() + '/raw_data/'))
files = np.array(files)

choosen_indices = [100, 150]
files = files[choosen_indices]

extent = [0, 1000, 0, 1000]  # set domain boundaries in (m)
c = 0
for i, frame in enumerate(files):
    print('Step: ', i, ' file: ', frame)
    # make a color map of fixed colors
    cmap = colors.ListedColormap(['white', 'green'])  # white for empty states, green for susceptible trees
    bounds = [0, 1, 2]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    t_step = np.load(path + frame)
    S, I, R = t_step  # fields of simulation
    I2 = np.zeros(I.shape)
    ind_inf = np.where(I > 0)  # infected indicies
    ind_sus = np.where(S > 0)  # susceptible indicies
    ind_rem = np.where(R > 0)
    I2[ind_inf] = 1
    kernels = gaussian_filter(input=I2, sigma=5, truncate=3.0)  # overlay the visualisation of the dispersal kernel
    kernels_dead = gaussian_filter(input=R, sigma=3, truncate=3.0)  # overlay the visualisation of the dispersal kernel
    #  kernels = np.where(kernels > 0, 1, 0)
    fig, ax = plt.subplots(figsize=(7.5, 8.5))
    title = ax.set_title('Day = ' + str(choosen_indices[i]), size=15)
    title.set_y(1.02)
    max = 0.10
    ax.imshow(S, cmap=cmap, origin="lower", norm=norm, alpha=0.1)
    # if i == 0:
    if 0:
        from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
        from mpl_toolkits.axes_grid1.inset_locator import mark_inset
        axins = zoomed_inset_axes(ax, 2.3, loc=2)  # zoom-factor: 2.5, location: upper-left
        x1, x2, y1, y2 = 140, 220, 230, 305  # specify the limits
        axins.set_xlim(x1, x2)  # apply the x-limits
        axins.set_ylim(y1, y2)  # apply the y-limits
        axins.set_xticks([])
        axins.set_yticks([])
        axins.imshow(S, cmap=cmap, origin="lower", norm=norm)
        mark_inset(ax, axins, loc1=1, loc2=3, fc="None", ec='black', alpha=0.5)
        axins.imshow(kernels_dead, cmap=plt.get_cmap('gray_r'), vmin=0, vmax=0.105, alpha=0.9, origin="lower")
        axins.imshow(kernels/max, cmap=plt.get_cmap('Reds'), vmin=0, vmax=0.105, alpha=0.35, origin="lower")

    if 1:
        ax.imshow(kernels_dead, cmap=plt.get_cmap('gray_r'), vmin=0, vmax=0.105, alpha=0.9, origin="lower")
        ax.imshow(kernels / max, cmap=plt.get_cmap('Reds'), vmin=0, vmax=0.105, alpha=0.35, origin="lower")
        #  ax.imshow(S, cmap=cmap, norm=norm, alpha=0.75, origin="lower")
        for co in zip(ind_sus[0], ind_sus[1]):
            circ = Circle((co[1], co[0]), 0.5, color='g', alpha=1)
            ax.add_patch(circ)
            if i == 0:
                circ1 = Circle((co[1], co[0]), 0.3, color='g', alpha=1)
                #axins.add_patch(circ1)

        for co in zip(ind_rem[0], ind_rem[1]):
            circ = Circle((co[1], co[0]), 0.5, alpha=1, color='saddlebrown')
            ax.add_patch(circ)
            if i == 0:
                circ1 = Circle((co[1], co[0]), 0.3, alpha=1, color='saddlebrown')
                #axins.add_patch(circ1)

        for co in zip(ind_inf[0], ind_inf[1]):
            c_ = I[co[0], co[1]]  # yellow-red on color scale each step through infectious period
            c_ = (c_ - 1)/100
            circ = Circle((co[1], co[0]), 0.5, color=[1, 1-c_, 0], alpha=1)
            ax.add_patch(circ)
            if i == 0:
                circ1 = Circle((co[1], co[0]), 0.3, color=[1, 1-c_, 0], alpha=1)
                #axins.add_patch(circ1)


    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(save_name + save_nm(c))
    c+=1
    plt.show()
    plt.close()
