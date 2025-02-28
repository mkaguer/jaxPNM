device = "cpu"

import os
os.environ["JAX_PLATFORMS"] = device
from jax import config
config.update("jax_enable_x64", True)

import jax
import diffrax as dfx
import jax.numpy as jnp
import mypnmlib as pnm
import matplotlib.pyplot as plt
import time
from _fit_cubic_network import FitCubicNetwork

ts = []
loss0 = []
lossf = []
shapes = jnp.array([[5, 2, 1],
                    [10, 10, 1],
                    [10, 10, 10]])
for shape in shapes:
    print(shape)
    # create network
    spacing = 1
    net = pnm.network.make_cubic_network(shape=shape, spacing=spacing)
    
    # get Nt and Np
    Nt = len(net['throat.conns'])
    Np = len(net['pore.coords'])
    
    # get target diameters
    key = jax.random.PRNGKey(0)
    D = jax.random.uniform(key, shape=(Np,)) * spacing
    print(D.device)

    # add the adjacency matrix
    weights = jnp.arange(1, Nt+1)
    am = pnm.network.create_adjacency_matrix(net, weights=weights, fmt='csr')
    net['adjacency_matrix'] = am
    
    # create instance of FitCubicNetwork
    pressure = jnp.arange(0.1, 2, 0.01)
    fcn = FitCubicNetwork(net, pressure=pressure)
    
    # add sat_target as attribute to fcn
    sat_target = fcn.run_invasion(D)
    fcn.sat_target = sat_target
    
    # get initial diameters
    key = jax.random.PRNGKey(1)
    D0 = jax.random.uniform(key, shape=(Np,)) * spacing

    loss = fcn.sat_loss(D0)
    print(f'Initial loss: {loss}')  # 3.8878519491468686
    loss0.append(loss)
    
    # fit porosimetry
    start = time.time()
    D, loss = fcn.fit_porosimetry(D0, solver=dfx.Euler(), t_span=(0, 1), dt=0.01)
    stop = time.time()

    loss = fcn.sat_loss(D)
    print(f'Final loss: {loss}')  # 0.008630249199476726
    lossf.append(loss)

    ts.append(stop - start)
    print(f'Time: {stop - start}s')

    
results = jnp.vstack((jnp.array(ts), jnp.array(loss0), jnp.array(lossf))).T
results = jnp.array(results)
jnp.save('../data/optimization-invasion-speed-' + device, results)