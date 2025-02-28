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
spacing = 1
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


# create instance of FitCubicNetwork
pressure = jnp.arange(0.1, 2, 0.01)
fcn = FitCubicNetwork(net,
                      pressure=pressure,
                      spacing=spacing,
                      smoothing_factor=0.0001)
# add sat_target as attribute to fcn
sat_target = fcn.run_invasion(D)
fcn.sat_target = sat_target
# plot target saturation
plt.figure(1)
plt.plot(pressure, sat_target, label=f'No Smoothing Factor')
plt.xlabel('Pressures')
plt.ylabel('Saturation')
plt.legend()

sfs = jnp.array([0.01, 0.03, 0.06])
for sf in sfs:
    # create instance of FitCubicNetwork
    pressure = jnp.arange(0.1, 2, 0.01)
    fcn = FitCubicNetwork(net,
                          pressure=pressure,
                          spacing=spacing,
                          smoothing_factor=sf)
    # add sat_target as attribute to fcn
    sat_target = fcn.run_invasion(D)
    fcn.sat_target = sat_target
    # plot target saturation
    plt.figure(1)
    plt.plot(pressure, sat_target, label=f'Smoothing Factor: {sf}')
    plt.legend()
plt.xlabel('Pressure', fontsize=16)
plt.ylabel('Saturation', fontsize=16)
plt.show()

# print network for visualization
pn = op.network.Cubic(shape, spacing)
pn['pore.diameter'] = net['pore.diameter']
pn['throat.diameter'] = net['throat.diameter']/2
op.io.project_to_vtk(project=pn.project, filename='../networks/net_sf')
