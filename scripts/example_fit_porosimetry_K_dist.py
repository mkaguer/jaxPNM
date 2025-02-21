import os
from jax import config
import jax
import diffrax as dfx
import jax.numpy as jnp
import mypnmlib as pnm
import matplotlib.pyplot as plt
from _fit_cubic_network import FitCubicNetwork

os.environ["JAX_PLATFORMS"] = "cpu"
config.update("jax_enable_x64", True)

# create network with spacing of 1 for optimizer
spacing = 1
net = pnm.network.make_cubic_network(shape=[5, 5, 5], spacing=1)

# get Nt and Np
Nt = len(net['throat.conns'])
Np = len(net['pore.coords'])

# add the adjacency matrix
weights = jnp.arange(1, Nt+1)
am = pnm.network.create_adjacency_matrix(net, weights=weights, fmt='csr')
net['adjacency_matrix'] = am

# add "constant" Gh properties to network
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

# create instance of FitCubicNetwork
pressure = jnp.arange(0.1, 2/spacing, 0.01/spacing)
fcn = FitCubicNetwork(net, pressure=pressure, spacing=spacing)

# get target D and tsf
shape, scale = 2, 0.5
D_target = fcn.sample_weibull_jax(Np, shape, scale)
tsf_target = fcn.sample_weibull_jax(Nt, shape, scale)

# add sat_target as attribute to fcn
sat_target = fcn.run_invasion(D_target, tsf_target)
fcn.sat_target = sat_target

# add K_target as attribute to fcn
x = fcn.flow(D_target, tsf_target)
K_target = fcn.calc_K(x)
fcn.K_target = K_target
print(f'Target Permeability: {K_target}')  # 2.0834366271421264e-05

# plot target saturation
plt.figure(1)
plt.plot(pressure, sat_target, label='target')
plt.xlabel('Pressures')
plt.ylabel('Saturation')

# get initial diameters
D0 = fcn.bundle_of_tubes_rvs(Np, seed=1) * 2
shape0, scale0 = fcn.fit_weibull(D0)
w0 = jnp.array([scale0, shape0, scale0, shape0])
print(f'Initial loss: {fcn.loss_dist(w0)}')  # 0.8936073899721305

# get initial saturation
D0 = fcn.sample_weibull_jax(Np, shape0, scale0)
tsf0 = fcn.sample_weibull_jax(Nt, shape0, scale0)
sat0 = fcn.run_invasion(D0, tsf0)
plt.figure(1)
plt.plot(pressure, sat0, label='Initial Guess')

# fit porosimetry
w, loss = fcn.fit_porosimetry_K_dist(w0,
                                     solver=dfx.Euler(),
                                     t_span=(0, 0.1), dt=0.001, clip=(-10, 10))
print(f'Final loss: {loss}')  # 0.0037254820612184466

# sample D and tsf from weibull
lambda_p, k_p, lambda_t, k_t = w
D = fcn.sample_weibull_jax(Np, k_p, lambda_p)
tsf = fcn.sample_weibull_jax(Nt, k_t, lambda_t)

# plot AI porosimetry
sat = fcn.run_invasion(D, tsf)
plt.figure(1)
plt.plot(pressure, sat, label='AI')
plt.legend()
plt.show()

# check permeability
x = fcn.flow(D, tsf)
K = fcn.calc_K(x)
print(f'Fitted Permeability: {K}')  # 2.1105243467149116e-05

