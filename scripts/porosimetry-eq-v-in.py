"""
This script plots the equivalent versus inscribed porosimetry curves

Created by: Mike McKague
Date: August 16, 2025
"""
import porespy as ps
import numpy as np
import matplotlib.pyplot as plt

ps.visualization.set_mpl_style()
np.random.seed(10)

name = "A1"

# load image-based data
data = np.loadtxt('../data/porosimetry-' + name + '.csv', delimiter=",")
pc_im = data[:, 0]
sw_im = data[:, 1]

# load equivalent diameter
data = np.loadtxt('../data/porosimetry-eq-' + name + '.csv', delimiter=",")
pc_eq = data[:, 0]
sw_eq = data[:, 1]

# make plot
plt.figure(3)
plt.semilogx(pc_im, sw_im, 'k-o', label=name, linewidth=4, markersize=12)
plt.semilogx(pc_eq, sw_eq, 'b-^', label="SNOW", linewidth=4, markersize=12)
plt.xlabel('Capillary Pressure (Pa)', fontsize="large")
plt.ylabel('Saturation', fontsize="large")
plt.legend(frameon=True)
plt.grid(True, which='both', linestyle='--', color='lightgrey', linewidth=0.7, alpha=0.7)
plt.minorticks_on()

# save
plt.savefig('../figures/porosimetry-eq-v-in-'+ name, dpi=500)
