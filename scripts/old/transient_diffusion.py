"""
Created to solve transient diffusion problem using jax

Created by: Mike McKague
"""

import numpy as np
import openpnm as op
import jax
import jax.numpy as jnp
from scipy.integrate import solve_ivp 
from diffrax import diffeqsolve, Dopri5, ODETerm, SaveAt, PIDController
from jax.experimental import sparse
import time

print(jax.devices())

# create network
net = op.network.Cubic(shape=[100, 100, 10], spacing=1e-4)

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

# set up diffusion algorithm
tfd = op.algorithms.TransientReactiveTransport(network=net, phase=phase)
tfd.settings['conductance'] = 'throat.diffusive_conductance'
tfd.settings['quantity'] = 'pore.concentration'

# get inlet and outlet pores
inlet = net.pores('xmin')
outlet = net.pores('xmax')

# apply BC
tfd.set_value_BC(pores=inlet, values=1)
tfd.set_value_BC(pores=outlet, values=0)

# run transient fickian diffusion
x0 = np.zeros(net.Np)
x0[inlet] = 1  # merge with BCs is important!
start = time.time()
tfd.run(x0, tspan=(0, 100), saveat=10)
stop = time.time()
print(f'{stop - start}s')

start = time.time()
tfd.run(x0, tspan=(0, 100), saveat=10)
stop = time.time()
print(f'{stop - start}s')

# update A and b
tfd._update_A_and_b()

# get A and b
A = tfd.A.tocsc()
b = tfd.b

# get volume
V = net['pore.volume']

def rhs(t, y):
    
    # FIXME: for now, A and b are fixed!
    
    return (-A.dot(y) + b) / V

x0 = np.zeros(net.Np)
x0[inlet] = 1  # merge with BCs is important!
start = time.time()
soln = solve_ivp(rhs, t_span=(0, 100), y0=x0, method='RK45', t_eval=np.arange(0, 101, 10), atol=1e-6, rtol=1e-6)
stop = time.time()
print(f'{stop - start}s')

# convert A and b to jax arrays
A = sparse.BCOO.from_scipy_sparse(A)
b = jnp.array(b)
V = jnp.array(V)
x0 = jnp.array(x0)

# now let's solve this in diffrax
rhs = lambda t, y, args: (-A @ y + b) / V
term = ODETerm(rhs)
solver = Dopri5()
saveat = SaveAt(ts=jnp.arange(0, 101, 10))
stepsize_controller = PIDController(rtol=1e-6, atol=1e-6)
start = time.time()
sol = diffeqsolve(term, solver, t0=0, t1=100, dt0=1e-3, y0=x0,
                  saveat=saveat, stepsize_controller=stepsize_controller)
stop = time.time()
print(f'{stop - start}s')

start = time.time()
sol = diffeqsolve(term, solver, t0=0, t1=100, dt0=1e-3, y0=x0,
                  saveat=saveat, stepsize_controller=stepsize_controller)
stop = time.time()
print(f'{stop - start}s')