"""
This script tests to see what network SIZE can resonably reproduce
the psd and tsd of a Berea sandstone sample 

Created by: Mike McKague
Date: December 8th 2025
"""

import numpy as np
import openpnm as op
import matplotlib.pyplot as plt
from scipy.stats import weibull_min

np.random.seed(1)

# import snow network
image = "Berea"
net = op.io.network_from_csv('../networks/' + image + '-snow' + '.csv')

# get sampled data
samples = net['pore.equivalent_diameter'] * 1e6

# if you want to force loc=0 (common):
c0, loc0, scale0 = weibull_min.fit(samples, floc=0)
print("shape k (loc=0) =", c0, "scale lambda =", scale0)

# select scale and shape
lam = scale0  # 1 scale
k = c0  # 1.5  # shape

# select number of samples to try
num_samples = [125, 512, 1000, 3375]

# select bins
# bins = np.arange(0, 5.1, 0.1)
bins = np.arange(0, np.max(samples), 4)


# get true pdf
# x = np.linspace(0, 5, 500)
x = np.linspace(0, np.max(samples), 500)
pdf = weibull_min.pdf(x, c=k, scale=lam)

# start plot
plt.figure(1)
fig, axes = plt.subplots(2, 2, dpi=600)
axes = axes.flatten()
axes[0].plot(x, pdf, color="r", linestyle='--', linewidth=2, label=image + " psd")
axes[1].plot(x, pdf, color="r", linestyle='--', linewidth=2, label=image + " psd")
axes[2].plot(x, pdf, color="r", linestyle='--', linewidth=2, label=image + " psd")
axes[3].plot(x, pdf, color="r", linestyle='--', linewidth=2, label=image + " psd")
# axes[0].set_xlim([-0.1, 5])
# axes[1].set_xlim([-0.1, 5])
# axes[2].set_xlim([-0.1, 5])
# axes[3].set_xlim([-0.1, 5])
# axes[0].set_ylim([0, 1])
# axes[1].set_ylim([0, 1])
# axes[2].set_ylim([0, 1])
# axes[3].set_ylim([0, 1])

# plot weibull distributions
letters = ["a", "b", "c", "d"]
for i, n in enumerate(num_samples):
    X = np.random.weibull(k, size=n)
    X = lam * X
    axes[i].hist(X, alpha=0.5, bins=bins, density=True)
    axes[i].set_title(f"{letters[i]}) No. of Samples: {n}")
    axes[i].legend(frameon=True, fontsize=10)
    # axes[i].set_xlabel("Pore Diameter (um)", fontsize=8)
    # axes[i].set_ylabel("Probability Density", fontsize=8)

# save figure
plt.savefig('../figures/review-network-size-' + image + '.png')


# show
plt.show()

