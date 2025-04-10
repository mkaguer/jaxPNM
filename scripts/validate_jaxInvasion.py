import os
from jax import config
import jax
import diffrax as dfx
import jax.numpy as jnp
import mypnmlib as pnm
import matplotlib.pyplot as plt
from _fit_cubic_network import FitCubicNetwork
import openpnm as op
import numpy as np
import os

os.environ["JAX_PLATFORMS"] = "cpu"
config.update("jax_enable_x64", True)

# create network
spacing = 1e-3
shape = [10, 10, 10]
net = pnm.network.make_cubic_network(shape=shape, spacing=spacing)
net['pore.boundary'] = net['pore.left']

# get Nt and Np
Nt = len(net['throat.conns'])
Np = len(net['pore.coords'])

# get diameters
key = jax.random.PRNGKey(3)
D = jax.random.uniform(key, shape=(Np,)) 

# add the adjacency matrix
weights = jnp.arange(1, Nt+1)
am = pnm.network.create_adjacency_matrix(net, weights=weights, fmt='csr')
net['adjacency_matrix'] = am

# create instance of FitCubicNetwork
pressure = jnp.arange(0.1, 2, 0.01)/spacing
fcn = FitCubicNetwork(net, pressure=pressure, spacing=spacing, smoothing_factor=0)

# add sat_target as attribute to fcn
sat = fcn.run_invasion(D)

# create openpnm network object
net_op = op.network.Cubic(shape=shape, spacing=spacing)

# add same geometry as jax network
net_op['pore.diameter'] = net['pore.diameter']
net_op['throat.diameter'] = net['throat.diameter']
net_op['throat.length'] = net['throat.length']
net_op['pore.volume'] = net['pore.volume'] 
net_op['throat.total_volume'] = net['throat.total_volume'] 
net_op['throat.lens_volume'] = net['throat.lens_volume']
props = ['throat.total_volume', 'throat.lens_volume']
net_op['throat.volume'] = net['throat.volume']

# create phase object
phase = op.phase.Phase(network=net_op)

# add entry pressure model
phase['throat.contact_angle'] = 120
phase['throat.surface_tension'] = 0.072
f = op.models.physics.capillary_pressure.washburn
phase.add_model(propname='throat.entry_pressure',
              model=f)
phase.regenerate_models()

# algorithm object
alg = op.algorithms.Drainage(phase=phase, network=net_op)
alg.set_inlet_BC(pores=net_op.pores('left'))
pressure = jnp.arange(0.1, 2, 0.01)/spacing
alg.run(pressures=pressure)

# get pc curve data
data = alg.pc_curve(pressures=pressure)

# plot target saturation
plt.figure(1, dpi=500)
ax = plt.gca()  # Get current axes
# Make the bounding box bold
for spine in ax.spines.values():
    spine.set_linewidth(3)
ax.tick_params(direction='in', length=6, width=3)
plt.semilogx(pressure, sat, label='JAX', color='tab:purple', linewidth=8)
plt.step(data.pc, data.snwp, label='OpenPNM', color='k', linestyle='solid', linewidth=3)
plt.xlabel('Pressure (Pa)', fontsize=18)
plt.ylabel('Saturation', fontsize=18)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(frameon=True, fontsize=18)
plt.savefig('../figures/validate_jaxPNM.png')
plt.show()
