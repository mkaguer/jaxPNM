print('importing')
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
# config.update("jax_disable_jit", True)

# create network
spacing = 1e-3
shape = [10, 10, 10]
print('creating jax network')
net = pnm.network.make_cubic_network(shape=shape, spacing=spacing)
net['pore.boundary'] = net['pore.left']

# get Nt and Np
Nt = len(net['throat.conns'])
Np = len(net['pore.coords'])

# get target diameters
print('assigning diameters')
key = jax.random.PRNGKey(3)
D_target = jax.random.uniform(key, shape=(Np,)) 
# D_target = jnp.array([0.8, 0.6, 0.4, 0.3])

# add the adjacency matrix
print('creating ad matrix')
weights = jnp.arange(1, Nt+1)
am = pnm.network.create_adjacency_matrix(net, weights=weights, fmt='csr')
net['adjacency_matrix'] = am

# create instance of FitCubicNetwork
print('creating jax object')
pressure = jnp.arange(0.1, 2/spacing, 0.01/spacing)
fcn = FitCubicNetwork(net, pressure=pressure, spacing=spacing, smoothing_factor=0)

# add sat_target as attribute to fcn
print('running jax invasion')
sat_target = fcn.run_invasion(D_target)
print('finished jax invasion')

fcn.sat_target = sat_target   


print('filling openpnm arrays from JAX arrays')
net_op = op.network.Cubic(shape=shape, spacing=spacing)
net_op['pore.diameter'] = net['pore.diameter'] 
# regenerate geometry models
net_op['throat.diameter'] = net['throat.diameter']
net_op['throat.length'] = net['throat.length']
net_op['pore.volume'] = net['pore.volume'] 
net_op['throat.total_volume'] = net['throat.total_volume'] 
net_op['throat.lens_volume'] = net['throat.lens_volume']
props = ['throat.total_volume', 'throat.lens_volume']
net_op['throat.volume'] = net['throat.volume']

phs = op.phase.Phase(network=net_op)
# add entry pressure model
phs['throat.contact_angle'] = 120
phs['throat.surface_tension'] = 0.072

print('creating phase object')
f = op.models.physics.capillary_pressure.washburn
phs.add_model(propname='throat.entry_pressure',
              model=f,
              )
phs.regenerate_models()
# print(phs['throat.entry_pressure'])

alg = op.algorithms.Drainage(phase=phs, network=net_op)
alg.set_inlet_BC(pores=net_op.pores('left'))
print('starting openpnm....')
pressure = np.arange(0.1, 2/spacing, 0.01/spacing)
alg.run(pressures=pressure)

data = alg.pc_curve(pressures=pressure)
print('plotting')
plt.figure(1)

# plot target saturation
plt.plot(pressure, sat_target, 'r', label='JAX')
plt.step(data.pc, data.snwp, 'b', label='openpnm')
plt.xlabel('Pressure [Pa]')
plt.ylabel('Saturation')
plt.title('shape=' + str(shape))
# plt.xlabel('Capillary Pressure')
# plt.ylabel('Non-Wetting Phase Saturation');
plt.legend()
plt.show()

net_op['throat.entry_pressure'] = phs['throat.entry_pressure']

current_directory = os.getcwd()
path_to_file = current_directory
op.io._vtk.project_to_vtk(net_op.project, filename=path_to_file+'/Paraview_net'+str(shape))








