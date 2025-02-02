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

# initial guess
key = random.PRNGKey(1)
D0 = random.uniform(key, shape=(Np,))

# set target permeability
K_target = 8.94833e-05

# Use JAX to fit cubic network
fcn = FitCubicNetwork(network=net, K_target=K_target)
D, loss = fcn.fit_K(D0, solver=dfx.Euler(), t_span=(0, 1e9), dt=1e8)

# check permeability
x = fcn.flow(D)
K = fcn.calc_K(x)
print(K)  # 8.948330993468651e-05

print(f"Avg D = {jnp.average(D)}")  # 0.5010305962353754
print(f"Min D = {jnp.min(D)}")  # 1.5224705660751351e-05
print(f"Max D = {jnp.max(D)}")  # 0.9993184311534421
print(f"Loss = {loss}")  # 9.869799604927808e-23

# plot loss landscape
fcn.plot_loss(D0, index=0, N=100)
