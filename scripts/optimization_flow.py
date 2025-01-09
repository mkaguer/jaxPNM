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
import diffrax as dfx
import mypnmlib as pnm
import os
from jax import config
import jax.random as random


os.environ["JAX_PLATFORMS"] = "cpu"
config.update("jax_enable_x64", True)

# create network
spacing = 1
net = pnm.network.make_cubic_network(shape=[4, 1, 1], spacing=1, connectivity=6)

# get Nt and Np
Nt = len(net['throat.conns'])
Np = len(net['pore.coords'])

# add "constant" properties to network
net['pore.viscosity'] = jnp.ones(Np) * 1e-3
net['throat.viscosity'] = jnp.ones(Nt) * 1e-3

# set BCs
pores = jnp.where(net['pore.left'])[0]
pnm.simulations.set_BC(net, pores=pores, bctype='value', bcvalues=1.0, mode='overwrite')
pores = jnp.where(net['pore.right'])[0]
pnm.simulations.set_BC(net, pores=pores, bctype='value', bcvalues=0.0, mode='add')

# add target value
net['target'] = 0.01286327


def f(D, net):

    # update D
    net['pore.diameter'] = D
    # update models that depend on D
    net['throat.diameter'] = pnm.models.throat_diameter(net)
    # net['throat.conduit_length'] = pnm.models.spheres_and_cylinders(net)  # Nt by 3
    # conduit lengths updated in hydraulic size factor automatically
    net['throat.hydraulic_size_factors'] = pnm.models.hydraulic_size_factor(net)
    # calculate conductance G
    G = pnm.models.generic_hydraulic(net)
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
    # calc penalty
    lbd = jnp.maximum(0.0 - D, 0)
    ubd = jnp.maximum(D - 1.0, 0)
    penalty = jnp.sum(lbd**2 + ubd**2)
    # add penalty to loss
    loss += penalty

    return loss


# test f works
key = random.PRNGKey(0)  # make results reproducible
D = random.uniform(key, shape=(4,))
print(f(D, net))

# Define the gradient of f(x)
grad_f = jax.grad(f)
print(grad_f(D, net))

# Define the ODE system for gradient flow: dx/dt = -grad(f)
def dydt(t, y, net):
    return -grad_f(y, net)

# Initial condition
y0 = jnp.array([0.4, 0.3, 0.3, 0.4])
t0, t1 = 0, 1000 # Time span (we treat the optimization as a "time" evolution)
# Choose an ODE solver
solver = dfx.Tsit5()  # Tsit5 is a good general-purpose ODE solver
# solver = dfx.Euler()
# Define the ODE problem
term = dfx.ODETerm(dydt)
# Solve the ODE, treating time as "iterations" for optimization
solution = dfx.diffeqsolve(term, solver, t0=t0, t1=t1, dt0=1, y0=y0, args=net)
# The final x value after "evolving" it toward the minimum
x_min = solution.ys[-1]
print(f"Minimum found at x = {x_min}, f(x) = {f(x_min, net)}")
xx
# visualize loss as a function of one of the parameters!
D_vals = jnp.linspace(0.0, 1.0, 100)
losses = jnp.array([f(jnp.array([D, 0.3, 0.3, 0.8]), net) for D in D_vals])

plt.plot(D_vals, losses)
plt.xlabel("D[0]")
plt.ylabel("Loss")
plt.title("Loss Landscape Along D[0]")
plt.show()

