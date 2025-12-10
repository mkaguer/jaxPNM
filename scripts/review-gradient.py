"""
Wrote this script to check that JAX gradient is accurate

Created by: Mike McKague
Date: December 10, 2025
"""
import openpnm as op
import jax
import jax.numpy as jnp
import jax.experimental.sparse as js
import matplotlib.pyplot as plt
import mypnmlib as pnm
import os
from jax import config
import jax.random as random

op.visualization.set_mpl_style()

os.environ["JAX_PLATFORMS"] = "cpu"
config.update("jax_enable_x64", True)

# create network
spacing = 1
net = pnm.network.make_cubic_network(shape=[10, 10, 10], spacing=1, connectivity=6)

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

# add pores to calculate rate
# FIXME: did this b/c I cannot do jnp.where inside f!
net['rate_pores'] = pores

# add target value
# net['target'] = 0.01286327
# net['target'] = 0.00262833
net['target'] = 0.99425941


def f(D, net):

    # update D
    net['pore.diameter'] = D  # FIXME: not enforcine size of arrays!
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
    # FIXME: write function to calculate Q
    # calc flow rate
    # pores = jnp.where(net['pore.right'])[0]
    pores = net['rate_pores']
    Q = -1*pnm.simulations.rate(net, x, pores=pores)[0]
    # print(Q)
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
D = random.uniform(key, shape=(Np,))
print(f(D, net))

# Define the gradient of f(x)
grad_f = jax.grad(f)
print(grad_f(D, net))

# take gradient at x
grad_x = grad_f(D, net)

# check that gradient is accurate
D1 = D.copy()
D2 = D.copy()
deltas = jnp.logspace(-3, -10, 8)
dfdxs = []
grad_xs = []
for delta in deltas:
    D2 = D2.at[0].set(D[0] + delta)
    f1 = f(D1, net)
    f2 = f(D2, net)
    dfdx = (f2 - f1)/delta
    dfdxs.append(dfdx)
    grad_xs.append(grad_x[0])

plt.figure(1, dpi=500)
plt.loglog(deltas, jnp.array(dfdxs), color='k', linewidth=4, marker="o")
plt.loglog(deltas, jnp.array(grad_xs), color='r', linewidth=4)
plt.legend(["Numerical Gradient", "JAX Gradient"], frameon=True, fontsize=18)
plt.ylabel("Gradient, dfdx", fontsize=16)
plt.xlabel("Pertubation, dx", fontsize=16)
plt.show()
plt.savefig("../figures/review-gradient.png")
