import numpy as np
import openpnm as op
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

op.visualization.set_mpl_style()

image = "Berea"

# set properties
sigma = 0.4791
theta = 140

# load networks
net1 = np.load('../networks/fitted-eq-BoT-Berea.npz')
net2 = np.load('../networks/fitted-review-BoT-Berea.npz')

# load network extraction
net_s = op.io.network_from_csv('../networks/Berea-snow' + '.csv')

# calculate spacing
data = np.loadtxt('../data/porosimetry-eq-Berea.csv', delimiter=',')
mask = ~np.isinf(data[:, 0])
sat_target = np.array(data[:, 1][mask]).astype(np.float32)
x_target = np.array(data[:, 0][mask]).astype(np.float32)
Dp = -4*sigma*np.cos(theta*np.pi/180)/x_target
spacing = Dp[0]

# retrieve original pore sizes
Dp1 = net1['pore.diameter'] * spacing * 1e6

# retrieve new pore sizes 
Dp2 = net2['pore.diameter'] * spacing * 1e6

# retrieve initial guess for throat size factor
tsf10 = net1['throat.initial_tsf']
tsf20 = net2['throat.initial_tsf']

# retrieve extracted pore amd throat diameters
Dp_snow = net_s['pore.equivalent_diameter'] * 1e6

# calculate psd from bundle of tubes model
# First, write interpolation function
x = Dp[::-1] * 1e6
y = sat_target[::-1]
interp_func = interp1d(x, y, kind='linear', bounds_error=False, fill_value=(y[0], y[-1]))
# evaluate at bins
bins = np.linspace(0, max(Dp_snow.max(), Dp1.max()), 1000)
sat_interp = interp_func(bins)
# get height and x for bar plot
height = sat_interp[:-1] - sat_interp[1:]
x = (bins[:-1] + bins[1:]) / 2
width = bins[1] - bins[0]
# scale height values so area under curve is 1
height = height/width

plt.figure(1, dpi=600)
fig, axes = plt.subplots(2, 2)
for spine in axes[0, 0].spines.values():
    spine.set_linewidth(2)
axes[1, 0].tick_params(direction='in', length=4, width=2)
bins = np.linspace(0, max(Dp_snow.max(), Dp1.max()), 50)
axes[1, 0].hist(Dp_snow, bins=bins, alpha=0.5, density=True, color='c', label='SNOW')
axes[1, 0].hist(Dp1, bins=bins, alpha=0.5, density=True, color='tab:purple', label='JAX')
axes[1, 0].set_title("c) Original PSD", fontsize=10, fontweight="normal")
# axes[1, 0].plot(x, height, color='tab:blue', label='Bundle of Tubes', linewidth=2.5)
axes[1, 0].legend(fontsize=8, frameon=True)
# axes[1, 0].set_title("a) Original PSD", fontsize=10)
axes[1, 0].set_xlabel("Pore Diameter (um)", fontsize=10)
for spine in axes[0, 1].spines.values():
    spine.set_linewidth(2)
axes[1, 1].tick_params(direction='in', length=4, width=2)
bins = np.linspace(0, max(Dp_snow.max(), Dp1.max()), 50)
axes[1, 1].hist(Dp_snow, bins=bins, alpha=0.5, density=True, color='c', label='SNOW')
axes[1, 1].hist(Dp2, bins=bins, alpha=0.5, density=True, color='tab:purple', label='JAX')
axes[1, 1].set_title("d) New PSD", fontsize=10, fontweight="normal")
# axes[1, 1].plot(x, height, color='tab:blue', label='Bundle of Tubes', linewidth=2.5)
axes[1, 1].legend(fontsize=8, frameon=True)
# axes[1, 1].set_title("b) New PSD", fontsize=10)
axes[1, 1].set_xlabel("Pore Diameter (um)", fontsize=10)
for spine in axes[1, 0].spines.values():
    spine.set_linewidth(2)
axes[0, 0].tick_params(direction='in', length=4, width=2)
axes[0, 0].hist(tsf10, bins=25, alpha=0.5, density=True, color='orange', label='Throat Aspect Ratio')
axes[0, 0].set_title("a) Original Initial Guess", fontsize=10, fontweight="normal")
axes[0, 0].set_xlim([0, 0.6])
axes[0, 0].legend(fontsize=8, loc="upper left", frameon=True)
for spine in axes[1, 1].spines.values():
    spine.set_linewidth(2)
axes[0, 1].tick_params(direction='in', length=4, width=2)
axes[0, 1].hist(tsf20, bins=25, alpha=0.5, density=True, color='orange', label='Throat Aspect Ratio')
axes[0, 1].set_title("b) New Initial Guess", fontsize=10, fontweight="normal")
axes[0, 1].legend(fontsize=8, loc="upper left", frameon=True)
plt.tight_layout()
plt.savefig("../figures/review-initial-guess.png")
plt.show()
