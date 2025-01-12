import jax.numpy as jnp
import diffrax as dfx
import mypnmlib as pnm
import os
from jax import config
import jax.random as random
from _fit_cubic_network import FitCubicNetwork

os.environ["JAX_PLATFORMS"] = "cpu"
config.update("jax_enable_x64", True)

# create network
spacing = 1
net = pnm.network.make_cubic_network(shape=[10, 10, 10],
                                     spacing=1,
                                     connectivity=6)

# get Nt and Np
Nt = len(net['throat.conns'])
Np = len(net['pore.coords'])

# add "constant" properties to network
net['pore.viscosity'] = jnp.ones(Np) * 1e-3
net['throat.viscosity'] = jnp.ones(Nt) * 1e-3

# set BCs
pores = jnp.where(net['pore.left'])[0]
pnm.simulations.set_BC(net,
                       pores=pores,
                       bctype='value',
                       bcvalues=1.0,
                       mode='overwrite')
pores = jnp.where(net['pore.right'])[0]
pnm.simulations.set_BC(net,
                       pores=pores,
                       bctype='value',
                       bcvalues=0.0,
                       mode='add')

# add pores to calculate rate
net['rate_pores'] = pores  # FIXME: cannot do jnp.where inside f!

# initial guess
key = random.PRNGKey(1)
D0 = random.uniform(key, shape=(Np,))

# set target permeability
K_target = 0.99425941

# Use JAX to fit cubic network
fcn = FitCubicNetwork(network=net, K_target=K_target)
D, loss = fcn.fit_K(D0, solver=dfx.Euler(), t_span=(0, 10), dt=1)

print(f"Avg D = {jnp.average(D)}")
print(f"Min D = {jnp.min(D)}")
print(f"Max D = {jnp.max(D)}")
print(f"Loss = {loss}")

# plot loss landscape
fcn.plot_loss(D0, index=0, N=100)
