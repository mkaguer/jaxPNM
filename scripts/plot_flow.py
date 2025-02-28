import jax.numpy as jnp
import diffrax as dfx
import mypnmlib as pnm
import os
from jax import config
import jax.random as random
from _fit_cubic_network import FitCubicNetwork
import openpnm as op

os.environ["JAX_PLATFORMS"] = "cpu"
config.update("jax_enable_x64", True)

# create network
spacing = 1
net = pnm.network.make_cubic_network(shape=[10, 10, 10],
                                     spacing=spacing,
                                     connectivity=6)

# get Nt and Np
Nt = len(net['throat.conns'])
Np = len(net['pore.coords'])

# add "constant" properties to network
net['pore.viscosity'] = jnp.ones(Np) * 1e-3
net['throat.viscosity'] = jnp.ones(Nt) * 1e-3

# set BCs
P1, P2 = 1.0, 0.0
pores = jnp.where(net['pore.left'])[0]
pnm.simulations.set_BC(net,
                       pores=pores,
                       bctype='value',
                       bcvalues=P1,
                       mode='overwrite')
pores = jnp.where(net['pore.right'])[0]
pnm.simulations.set_BC(net,
                       pores=pores,
                       bctype='value',
                       bcvalues=P2,
                       mode='add')

# add pores to calculate rate
net['rate_pores'] = pores  # FIXME: cannot do jnp.where inside f!

# get target diameters
key = random.PRNGKey(0)
D_target = random.uniform(key, shape=(Np,))

# define FitCubicNetwork object
fcn = FitCubicNetwork(network=net, spacing=spacing)

# find K_target
x = fcn.flow(D_target)
K_target = fcn.calc_K(x)
fcn.K_target = K_target

pn = op.network.Cubic(shape=[10, 10, 10], spacing=1)
pn['pore.diameter'] = net['pore.diameter']
pn['throat.radius'] = net['throat.diameter']/2
pn['pore.pressure'] = x
op.io.project_to_vtk(project=pn.project, filename='../networks/net_flow')
