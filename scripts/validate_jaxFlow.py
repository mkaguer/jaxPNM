import openpnm as op
import numpy as np
import porespy as ps
import jax.numpy as jnp
import mypnmlib as pnm
import os
from jax import config
from _fit_cubic_network import FitCubicNetwork

os.environ["JAX_PLATFORMS"] = "cpu"
config.update("jax_enable_x64", True)

# ps.visualization.set_mpl_style()
op.visualization.set_mpl_style()

np.random.seed(1)

# select spacing and shape
spacing = 1e-3
shape = [10, 10, 10]

# select pressure boundary conditions
Pin, Pout = 1, 0

# %% OpenPNM flow solution
# create openpnm network
net_o = op.network.Cubic(shape=shape, spacing=spacing)

# add geometry models to net_o
geo_mods = op.models.collections.geometry.spheres_and_cylinders.copy()
net_o.add_model_collection(geo_mods)
net_o.regenerate_models()

# create phase object
phase = op.phase.Water(network=net_o)

# add hydraulic conductance
phase['pore.viscosity'] = 1e-3
Gh_mod = op.models.physics.hydraulic_conductance._funcs.generic_hydraulic
phase.add_model(propname='throat.hydraulic_conductance',
                model=Gh_mod,
                throat_viscosity='throat.viscosity',
                size_factors='throat.hydraulic_size_factors')

# get viscosity
mu = phase['pore.viscosity'][0]

# run flow in x
flow_x = op.algorithms.StokesFlow(network=net_o, phase=phase)
flow_x.set_value_BC(pores=net_o.pores('xmin'), values=Pin)
flow_x.set_value_BC(pores=net_o.pores('xmax'), values=Pout)
flow_x.run()

# calculate permeability in x
Lx_o = shape[0] * spacing
Ax_o = shape[1] * shape[2] * spacing ** 2
Qx_o = flow_x.rate(pores=net_o.pores('xmin'), mode='group')[0]
Kx_o = Qx_o * Lx_o * mu / (Ax_o * (Pin - Pout)) / 0.98e-12 * 1000
print(f'Kx_openpnm is: {Kx_o:.5f} mD')

# %% JAX flow solution
net_j = pnm.network.make_cubic_network(shape=shape, spacing=spacing)

# get Nt and Np
Nt = len(net_j['throat.conns'])
Np = len(net_j['pore.coords'])

# add "constant" properties to network
net_j['pore.viscosity'] = jnp.ones(Np) * 1e-3
net_j['throat.viscosity'] = jnp.ones(Nt) * 1e-3

# set BCs
pores = jnp.where(net_j['pore.left'])[0]
pnm.simulations.set_BC(net_j,
                       pores=pores,
                       bctype='value',
                       bcvalues=Pin,
                       mode='overwrite')
pores = jnp.where(net_j['pore.right'])[0]
pnm.simulations.set_BC(net_j,
                       pores=pores,
                       bctype='value',
                       bcvalues=Pout,
                       mode='add')

# add pores to calculate rate
net_j['rate_pores'] = pores

# define FitCubicNetwork object
fcn = FitCubicNetwork(network=net_j, spacing=1)

# get Diameters
D = net_o['pore.diameter']

# run flow
x = fcn.flow(D)

# calcualte K
Lx_j = shape[0] * spacing
Ax_j = shape[1] * shape[2] * spacing ** 2
Qx_j = -1*pnm.simulations.rate(net_j, x, pores=pores)[0]
Kx_j = Qx_j * Lx_j * mu / (Ax_j * (Pin - Pout)) / 0.98e-12 * 1000
print(f'Kx_jax is: {Kx_j:.5f} mD')

# %% Check error
# print out the error
mask = net_j['pore.left'] + net_j['pore.right']
mask = ~mask  # mask out boundary when calculating error
error = jnp.average(jnp.abs(x[mask]-flow_x.x[mask])/flow_x.x[mask])
sse = jnp.sum((x[mask]-flow_x.x[mask])**2)
print(f'Avg Error: {error*100}%')
print(f'SSE: {sse}%')

# Reasons why they are not the same:
# 1) Rounding is different. For some reason, Ft is slightly rounded
#    differently! Handled differently in memory.
#    Ft = Lt / (_np.pi / 4 * Dt**2)**2
# 2) Permeability is calculated differently (e.g. L and A) when using calc_K.
#    Be careful! But this problem is not in this script!
