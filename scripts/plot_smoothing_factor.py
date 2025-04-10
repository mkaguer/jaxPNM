import os
from jax import config
import jax
import jax.numpy as jnp
import mypnmlib as pnm
import matplotlib.pyplot as plt
from _fit_cubic_network import FitCubicNetwork
import porespy as ps
import openpnm as op

ps.visualization.set_mpl_style()

os.environ["JAX_PLATFORMS"] = "cpu"
config.update("jax_enable_x64", True)

# create network
spacing = 1e-3
shape = [4, 1, 1]
net = pnm.network.make_cubic_network(shape=shape, spacing=spacing)

# get Nt and Np
Nt = len(net['throat.conns'])
Np = len(net['pore.coords'])

# get target diameters
key = jax.random.PRNGKey(0)
D = jnp.array([0.9, 0.7, 0.5, 0.3])

# add the adjacency matrix
weights = jnp.arange(1, Nt+1)
am = pnm.network.create_adjacency_matrix(net, weights=weights, fmt='csr')
net['adjacency_matrix'] = am

# update pore.boundary
net['pore.boundary'] = net['pore.left']

# create instance of FitCubicNetwork
pressure = jnp.arange(0.1, 1.8, 0.01) / spacing
fcn = FitCubicNetwork(net,
                      pressure=pressure,
                      spacing=spacing,
                      smoothing_factor=0.0001)

# add sat_target as attribute to fcn
sat_target = fcn.run_invasion(D)
fcn.sat_target = sat_target

# plot target saturation
plt.figure(1, dpi=500)
ax = plt.gca()  # Get current axes
# Make the bounding box bold
for spine in ax.spines.values():
    spine.set_linewidth(3)
ax.tick_params(direction='in', length=6, width=3)
plt.plot(pressure, sat_target, label=f'No Smoothing Factor', color='k')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.grid(axis='x', linestyle='--', alpha=0.7)
sfs = jnp.array([0.01, 0.03, 0.06])
# colors = ['tab:blue', 'tab:orange', 'tab:green']
colors = ['tab:blue', 'tab:purple', 'tab:cyan']
for i, sf in enumerate(sfs):
    # create instance of FitCubicNetwork
    pressure = jnp.arange(0.1, 1.8, 0.01) / spacing
    fcn = FitCubicNetwork(net,
                          pressure=pressure,
                          spacing=spacing,
                          smoothing_factor=sf)
    # add sat_target as attribute to fcn
    sat_target = fcn.run_invasion(D)
    fcn.sat_target = sat_target
    # plot target saturation
    plt.figure(1)
    plt.plot(pressure, sat_target, label=f'Smoothing Factor: {sf}', color=colors[i], linewidth=3)
plt.legend(frameon=True, loc='lower right', fontsize=18)
plt.xlabel('Pressure (Pa)', fontsize=18)
plt.ylabel('Saturation', fontsize=18)
plt.xticks(fontsize=18, fontweight='normal')
plt.yticks(fontsize=18, fontweight='normal')
plt.savefig('../figures/smoothing_factor.png')
plt.show()

# print network for visualization
pn = op.network.Cubic(shape, spacing)
pn['pore.diameter'] = net['pore.diameter']
pn['throat.diameter'] = net['throat.diameter']/2
op.io.project_to_vtk(project=pn.project, filename='../networks/net_sf')
