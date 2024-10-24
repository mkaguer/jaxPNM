import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCOO
from jax.scipy.sparse.linalg import cg, gmres, bicgstab

from scipy.sparse.linalg import cg as cgs, gmres as gmress, bicgstab as bicgstabs
import openpnm as op
import numpy as np
import time
import importlib

# importlib.reload(jax)
# importlib.reload(jnp)
# importlib.reload(op)
# importlib.reload(np)
# importlib.reload(BCOO)
# importlib.reload(cg)
# importlib.reload(gmres)
# importlib.reload(bicgstab)

# print(jax.devices())
# create network

def run(shape):
    net = op.network.Cubic(shape=shape, spacing=1e-4)
    # print(f'shape is {shape}')
    
    # add geometry models
    geo_mods = op.models.collections.geometry.spheres_and_cylinders.copy()
    del geo_mods['pore.diameter']
    net['pore.diameter'] = 5e-5
    net.add_model_collection(geo_mods)
    net.regenerate_models()
    
    # add phase
    phase = op.phase.Phase(network=net)
    phase['pore.viscosity'] = 1e-3
    phase['pore.diffusivity'] = 1e-9
    phase_mods = op.models.collections.physics.basic.copy()
    del phase_mods['throat.entry_pressure']
    phase.add_model_collection(phase_mods)
    phase.regenerate_models()
    
    # set up flow algorithm
    flow = op.algorithms.StokesFlow(network=net, phase=phase)
    flow.set_value_BC(pores=net.pores('left'), values=1)
    flow.set_value_BC(pores=net.pores('right'), values=0)
    
    # solve using OpenPNM
    start = time.time()
    flow.run()
    end = time.time()
    t_pnm = end - start
    print(f'OpenPNM CPU time: {end - start} s')
    # print(f'OpenPNM Solution: {flow.x}')
    
    # get scale
    scale = flow.A.max()  # scale for JAX
    # scale = 1  # don't need to scale for scipy
    
    # solve using JAX on GPU
    # A = flow.A  # for scipy
    start = time.time()
    A = BCOO.from_scipy_sparse(flow.A)  # for JAX
    end = time.time()
    print(f'From scipy sparse time: {end - start} s')
    b = flow.b
    
    # normalize A and b
    A = A/scale
    b = b/scale
    # print(A.todense())
    # print(b)
    
    # solve sparse arrays using jax
    # for i in range(5):
    start = time.time()
    x1, _ = cg(A, b)
    end = time.time()
    t_jax = end - start
    print(f'cg GPU time: {end - start} s')
    
    start = time.time()
    x2, _ = cgs(flow.A/scale, b)
    end = time.time()
    t_scipy = end - start
    print(f'cg GPU time: {end - start} s')
        
    # print(f'JAX cg Solution: {x}')
    # start = time.time()
    # x, _ = gmres(A, b)  # JAX does not currently support gmres for GPU!
    # end = time.time()
    # print(f'gmres GPU time: {end - start} s')
    # print(f'JAX gmres Solution: {x}')
    # start = time.time()
    # x, _ = bicgstab(A, b)
    # end = time.time()
    # print(f'bicgstab GPU time: {end - start} s')
    # print(f'JAX bicgstab Solution: {x}')
    
    # put A and b on cpu
    # A_cpu = jax.device_put(A, jax.devices('cpu')[0])
    # b_cpu = jax.device_put(b, jax.devices('cpu')[0])
    
    # start = time.time()
    # x2, _ = cg(A_cpu, b_cpu)
    # end = time.time()
    # print(f'cg CPU time: {end - start} s')
    # print(f'JAX cg Solution: {x}')
    
    # calculate error
    # print(jnp.mean(jnp.abs(flow.x - x2)))
    # print(abs(jnp.mean(x2) - np.mean(flow.x)))
    # use jax.device_put
    
    # run(shape=[1000,1000,1])
    
    return t_pnm, t_jax, t_scipy

shapes = [[10, 10, 1],
          [10, 10, 1],
          [100, 10, 1],
          [100, 100, 1],
          [1000, 100, 1], 
          [1000, 1000, 1],
          [1000, 5000, 1]]

times = np.zeros((len(shapes), 3))
for i, shape in enumerate(shapes):
    print(shape)
    t_pnm, t_jax, t_scipy = run(shape)
    times[i, 0] = t_pnm
    times[i, 1] = t_jax
    times[i, 2] = t_scipy

import matplotlib.pyplot as plt
plt.figure(1)
x = np.product(shapes, axis=1)
plt.plot(x[1:], times[1:, 0], label='OpenPNM')
plt.plot(x[1:], times[1:, 1], label='JAX_cg')
plt.plot(x[1:], times[1:, 2], label='Scipy_cg')
plt.xscale('log')
plt.yscale('log')
plt.legend()