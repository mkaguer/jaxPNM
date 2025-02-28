device = "cpu"

import os
os.environ["JAX_PLATFORMS"] = device
from jax import config
config.update("jax_enable_x64", True)

import jax
import diffrax as dfx
import jax.numpy as jnp
import jax.random as random
import mypnmlib as pnm
import matplotlib.pyplot as plt
import time
from _fit_cubic_network import FitCubicNetwork

ts = []
loss0 = []
lossf = []
tspans = [(1e7, 1e8), (1e7, 1e9), (1e7, 1e10), (1e7, 1e11)]
for tspan in tspans:
    # get tf and dt
    tf = tspan[1]
    dt = tspan[0]
    print(f'Iterations: {tf/dt}')
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

    # get target D for target permeability
    key = random.PRNGKey(1)
    D_target = random.uniform(key, shape=(Np,))

    # creat instance of FitCubicNetwork class
    fcn = FitCubicNetwork(network=net)

    # get target permeability
    x = fcn.flow(D_target)
    K_target = fcn.calc_K(x)
    fcn.K_target = K_target

    # initial guess
    key = random.PRNGKey(2)
    D0 = random.uniform(key, shape=(Np,))

    # get the initial loss
    x = fcn.flow(D0)
    K = fcn.calc_K(x)
    loss0.append((K - K_target)**2)

    # Use JAX to fit cubic network
    start = time.time()
    D, loss = fcn.fit_K(D0,
                        solver=dfx.Euler(),
                        t_span=(0, tf),
                        dt=dt,
                        max_steps=10001)
    stop = time.time()

    # get the final loss
    x = fcn.flow(D)
    K = fcn.calc_K(x)
    lossf.append(loss)
    print(f'Loss: {loss}')

    # append time
    ts.append(stop - start)
    print(f'Time: {stop - start}s')

# save results
results = jnp.vstack((jnp.array(ts), jnp.array(loss0), jnp.array(lossf))).T
results = jnp.array(results)
jnp.save('../data/optimization-flow-speed-iter-' + device, results)
