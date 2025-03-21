import numpy as np
import jax.numpy as jnp
import jax.experimental.sparse as js
from _fit_cubic_network import FitCubicNetwork
import pandas as pd
import os
from jax import config
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

os.environ["JAX_PLATFORMS"] = "cpu"
config.update("jax_enable_x64", False)

# ps.visualization.set_mpl_style()

image = "A1"

# set properties
sigma = 0.4791
theta = 140

# load fitted network
net = np.load('../networks/fitted-' + image + '.npz')
net = {key: net[key] for key in net.files}

# get Np and Nt
Np = len(net['pore.coords'])
Nt = len(net['throat.conns'])

# load porosimetry data
data = np.loadtxt('../data/porosimetry-' + image + '.csv', delimiter=',')
mask = ~np.isinf(data[:, 0])
sat_target = jnp.array(data[:, 1][mask]).astype(jnp.float32)
x_target = jnp.array(data[:, 0][mask]).astype(jnp.float32)

# calculate spacing
Dp = -4*sigma*jnp.cos(theta*jnp.pi/180)/x_target
spacing = Dp[0]

# load porosimetry data
data = pd.read_csv('../data/K-' + image + '.csv', header=None).values.flatten()
K_target = jnp.array(data).astype(jnp.float32)

# reconstruct A matrix
data = net['A.data']
indices = net['A.indices']
net['A'] = js.BCOO((data, indices), shape=(Np, Np))

# reconstruct adjacency matrix
data = net['am.data']
indices = net['am.indices']
indptr = net['am.indptr']
net['adjacency_matrix'] = js.BCSR((data, indices, indptr), shape=(Np, Np))

# create fcn object
pressures = jnp.arange(jnp.min(x_target)*0.9, jnp.max(x_target)*1.1, 1e3)
fcn = FitCubicNetwork(net,
                      surface_tension=sigma,
                      contact_angle=theta,
                      sat_target=sat_target,
                      x_target=x_target,
                      K_target=K_target,
                      pressure=pressures,
                      spacing=1,
                      smoothing_factor=0.4)

# pre-process pressures (b/c D is btwn 0 and 1, spacing set as 1)
fcn.process_pressure(spacing=spacing, mode='pre')

# retrieve diameters
D = net['pore.diameter']
D0 = net['pore.initial_diameters']

# retrieve throat size factors
tsf = net['throat.tsf']
tsf0 = net['throat.initial_tsf']

# run invasion
sat = fcn.run_invasion(D, tsf)
sat0 = fcn.run_invasion(D0, tsf0)

# interpolate results
sat = jnp.interp(x_target, pressures, sat)
sat0 = jnp.interp(x_target, pressures, sat0)

# plot pc results
plt.figure(1, dpi=500)
ax = plt.gca()  # Get current axes
# Make the bounding box bold
for spine in ax.spines.values():
    spine.set_linewidth(2)
ax.tick_params(direction='out', length=5, width=2)
# Set x-axis to scientific notation
ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
ax.xaxis.get_offset_text().set_fontsize(14)  # Adjust offset text size
ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))  # Force scientific notation 
plt.plot(x_target, sat0, label='Initial Guess', color='tab:blue', marker='o', markerfacecolor='none', markersize=9, linewidth=2.5)
plt.plot(x_target, sat, label='JAX', color='tab:purple', marker='o', markerfacecolor='none', markersize=9, linewidth=2.5)
plt.plot(x_target, sat_target, label='Target', color='tab:green', marker='^', markerfacecolor='none', markersize=9, linewidth=2.5)
plt.xlabel('Pressure (Pa)', fontsize=14)
plt.ylabel('Saturation', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title(image, fontsize=16, fontweight='semibold')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(loc='best', fontsize=12)
plt.tight_layout()
plt.savefig('../figures/fitted-porosimetrty-' + image + '.png')
plt.show()

# run flow and calculate K for fitted D
px = fcn.flow(D, tsf, axis='x')
Kx = fcn.calc_K(px, axis='x')
py = fcn.flow(D, tsf, axis='y')
Ky = fcn.calc_K(py, axis='y')
pz = fcn.flow(D, tsf, axis='z')
Kz = fcn.calc_K(pz, axis='z')

# run flow and calculate K for initial D
p0x = fcn.flow(D0, tsf0, axis='x')
K0x = fcn.calc_K(px, axis='x')
p0y = fcn.flow(D0, tsf0, axis='y')
K0y = fcn.calc_K(py, axis='y')
p0z = fcn.flow(D0, tsf0, axis='z')
K0z = fcn.calc_K(pz, axis='z')

# calcualte averages
K_fitted_avg = jnp.average(jnp.array([Kx, Ky, Kz]))
K_initial_avg = jnp.average(jnp.array([K0x, K0y, K0z]))
K_target_avg = jnp.array([jnp.average(K_target)])

# gather Ks
K_fitted = jnp.array([Kx, Ky, Kz, K_fitted_avg]) * spacing ** 2  / 0.98e-12 * 1000
K_initial = jnp.array([K0x, K0y, K0z, K_initial_avg]) * spacing ** 2  / 0.98e-12 * 1000
K_target = jnp.concatenate((jnp.array(K_target), K_target_avg))  # CHECK!!
K = jnp.vstack((K_target, K_fitted, K_initial))

# plot K results
plt.figure(2, dpi=500)
ax = plt.gca()  # Get current axes
# Make the bounding box bold
for spine in ax.spines.values():
    spine.set_linewidth(2)
ax.tick_params(direction='out', length=5, width=2) 
x = jnp.arange(len(K_fitted))
bar_width = 0.25
plt.bar(x, K_target, width=bar_width, label='Target', color='tab:green')
plt.bar(x + bar_width, K_fitted, width=bar_width, label='JAX', color='tab:purple')
plt.bar(x + 2 * bar_width, K_initial, width=bar_width, label='Initial', color='tab:blue')
plt.ylabel('Permeability (mD)', fontsize=14, fontweight='normal')
plt.title(image, fontsize=16, fontweight='semibold')
plt.xticks(x + bar_width, ['X', 'Y', 'Z', 'Avg'], fontsize=12, fontweight='normal')
plt.yticks(fontsize=12, fontweight='normal')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.07), ncol=3, fontsize=12)
plt.tight_layout()
plt.savefig('../figures/fitted-permeability-' + image + '.png')
plt.show()