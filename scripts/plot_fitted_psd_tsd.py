import numpy as np
import openpnm as op
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import rv_discrete

image = 'S5'
BoT = 'BoT-'
equivalent = False

# set properties
sigma = 0.4791
theta = 140

# load fitted network
if equivalent is True:
    net_f = np.load('../networks/fitted-eq-' + BoT + image + '.npz')
    net_f = {key: net_f[key] for key in net_f.files}
else:
    net_f = np.load('../networks/fitted-' + BoT + image + '.npz')
    net_f = {key: net_f[key] for key in net_f.files}

# load network extraction
net_s = op.io.network_from_csv('../networks/' + image + '-snow' + '.csv')

# load porosimetry data
if equivalent is True:
    data = np.loadtxt('../data/porosimetry-eq-' + image + '.csv', delimiter=',')
    mask = ~np.isinf(data[:, 0])
    sat_target = np.array(data[:, 1][mask]).astype(np.float32)
    x_target = np.array(data[:, 0][mask]).astype(np.float32)
else:
    data = np.loadtxt('../data/porosimetry-' + image + '.csv', delimiter=',')
    mask = ~np.isinf(data[:, 0])
    sat_target = np.array(data[:, 1][mask]).astype(np.float32)
    x_target = np.array(data[:, 0][mask]).astype(np.float32)

# calculate spacing
Dp = -4*sigma*np.cos(theta*np.pi/180)/x_target
spacing = Dp[0]

# retrieve fitted pore and throat diameters
Dp_fitted = net_f['pore.diameter'] * spacing * 1e6
tsf = net_f['throat.tsf']
conns = net_f['throat.conns']
Dt_fitted = np.min(Dp_fitted[conns], axis=1) * tsf

# retrieve initial guess pore and throat diameters
Dp_guess = net_f['pore.initial_diameters'] * spacing * 1e6
tsf = net_f['throat.initial_tsf']
conns = net_f['throat.conns']
Dt_guess = np.min(Dp_guess[conns], axis=1) * tsf

# retrieve extracted pore amd throat diameters
Dp_snow = net_s['pore.equivalent_diameter'] * 1e6
Dt_snow = net_s['throat.inscribed_diameter'] * 1e6

# calculate psd from bundle of tubes model
# First, write interpolation function
x = Dp[::-1] * 1e6
y = sat_target[::-1]
interp_func = interp1d(x, y, kind='linear', bounds_error=False, fill_value=(y[0], y[-1]))
# evaluate at bins
bins = np.linspace(0, max(Dp_snow.max(), Dp_fitted.max()), 1000)
sat_interp = interp_func(bins)
# get height and x for bar plot
height = sat_interp[:-1] - sat_interp[1:]
x = (bins[:-1] + bins[1:]) / 2
width = bins[1] - bins[0]
# scale height values so area under curve is 1
height = height/width

# plot psd
plt.figure(1, dpi=500)
ax = plt.gca()  # Get current axes
# Make the bounding box bold
for spine in ax.spines.values():
    spine.set_linewidth(3)
ax.tick_params(direction='in', length=6, width=3)
bins = np.linspace(0, max(Dp_snow.max(), Dp_fitted.max()), 50)
plt.hist(Dp_snow, bins=bins, alpha=0.5, density=True, color='c', label='SNOW')
plt.hist(Dp_fitted, bins=bins, alpha=0.5, density=True, color='tab:purple', label='JAX')
# plt.bar(x=x, height=height, width=width, alpha=0.5, color='tab:blue', label='Bundle of Tubes')
plt.plot(x, height, color='tab:blue', label='Bundle of Tubes', linewidth=2.5)
plt.title(image + ' PSD', fontweight='semibold', fontsize=16)
plt.xlabel('Pore Diameter (\u03BCm)', fontsize=18)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(fontsize=18, frameon=True)
plt.tight_layout()
if equivalent is True:
    plt.savefig('../figures/fitted-psd-eq-' + image + '.png')
else:
    plt.savefig('../figures/fitted-psd-' + image + '.png')
plt.show()

# plt tsd
plt.figure(2, dpi=500)
ax = plt.gca()  # Get current axes
# Make the bounding box bold
for spine in ax.spines.values():
    spine.set_linewidth(3)
ax.tick_params(direction='in', length=6, width=3)
bins = np.linspace(0, max(Dt_snow.max(), Dt_fitted.max()), 50)
plt.hist(Dt_snow, bins=bins, alpha=0.5, density=True, color='c', label='SNOW')
plt.hist(Dt_fitted, bins=bins, alpha=0.5, density=True, color='tab:purple', label='JAX')
# plt.hist(Dp_guess, bins=bins, alpha=0.5, density=True, color='tab:blue', label='Initial Guess')
plt.title(image + ' TSD', fontweight='semibold', fontsize=16)
plt.xlabel('Throat Diameter (\u03BCm)', fontsize=18)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(fontsize=18)
plt.tight_layout()
if equivalent is True:
    plt.savefig('../figures/fitted-tsd-eq-' + image + '.png')
else:
    plt.savefig('../figures/fitted-tsd-' + image + '.png')
plt.show()
