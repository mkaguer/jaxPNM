"""
The goal is to write a script that fits a cubic network to have 
desired permeability using JAX and Diffrax.

Created by: Mike McKague
Date: November 18, 2024
"""

import jax
import jax.numpy as jnp
import jax.experimental.sparse as js
import matplotlib.pyplot as plt
import models as mods
import diffrax as dfx
import mypnmlib as pnm
import os
from jax import config
import jax.random as random


os.environ["JAX_PLATFORMS"] = "cpu"
config.update("jax_enable_x64", True)

# create network
spacing = 1
coords = jnp.array([[0, 0, 0],
                    [1, 0, 0],
                    [2, 0, 0],
                    [3, 0, 0]])
conns = jnp.array([[0, 1],
                   [1, 2],
                   [2, 3]])
net = {'pore.coords': coords, 'throat.conns': conns}

# add properties to network
Nt = conns.shape[0]
net['throat.length'] = jnp.ones(Nt) * spacing
net['throat.viscosity'] = jnp.ones(Nt) * 1e-3
net['throat.diameter'] = jnp.ones(Nt) * 0.5  # this will get overwritten

# set BCs
pnm.simulations.set_BC(net, pores=0, bctype='value', bcvalues=1.0, mode='overwrite')
pnm.simulations.set_BC(net, pores=3, bctype='value', bcvalues=0.0, mode='add')


# add target value
net['target'] = 0.00242071


def f(D, net):

    # update D
    net['throat.diameter'] = D
    # calculate conductance G
    G = pnm.models.calc_conductance(net)
    net['throat.conductance'] = G
    # build A and b
    A = pnm.simulations.build_A(net)
    b = pnm.simulations.build_b(net)
    net['A'] = A
    net['b'] = b
    # apply BCs
    A, b = pnm.simulations.apply_BC(net)
    # solve Ax = b
    A = js.BCSR.from_bcoo(A)  # need CSR format for linalg.spsolve!
    A.indptr = A.indptr.astype('int64')  # FIXME: can I remove this line?
    x = js.linalg.spsolve(A.data, A.indices, A.indptr, b, tol=1e-6)
    # calc flow rate
    Q = -G[-1] * (x[-1] - x[-2])
    # calc loss
    Q_target = net['target']
    loss = (Q - Q_target)**2

    return loss


# test f works
key = random.PRNGKey(0)  # make results reproducible
D = random.uniform(key, shape=(3,))
print(f(D, net))

# Define the gradient of f(x)
grad_f = jax.grad(f)
print(grad_f(D, net))

# Define the ODE system for gradient flow: dx/dt = -grad(f)
def dydt(t, y, net):
    return -grad_f(y, net)

# Initial condition
y0 = jnp.array([0.5, 0.3, 0.8])
t0, t1 = 0, 10  # Time span (we treat the optimization as a "time" evolution)
# Choose an ODE solver
solver = dfx.Tsit5()  # Tsit5 is a good general-purpose ODE solver
# Define the ODE problem
term = dfx.ODETerm(dydt)
# Solve the ODE, treating time as "iterations" for optimization
solution = dfx.diffeqsolve(term, solver, t0=t0, t1=t1, dt0=1e-2, y0=y0, args=net)
# The final x value after "evolving" it toward the minimum
x_min = solution.ys[-1]
print(f"Minimum found at x = {x_min}, f(x) = {f(x_min, net)}")
