
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

# create network
spacing = 1e-4
net = pnm.network.make_cubic_network(shape=[5, 5, 5], spacing=spacing)

# get Nt and Np
Nt = len(net['throat.conns'])
Np = len(net['pore.coords'])

# get target diameters
key = jax.random.PRNGKey(0)
D_target = jax.random.uniform(key, shape=(Np,))

# add the adjacency matrix
weights = jnp.arange(1, Nt+1)
am = pnm.network.create_adjacency_matrix(net, weights=weights, fmt='csr')
net['adjacency_matrix'] = am

# create instance of FitCubicNetwork
pressure = jnp.arange(0.1, 2/spacing, 0.01/spacing)
fcn = FitCubicNetwork(net, pressure=pressure, spacing=spacing)

# add sat_target as attribute to fcn
sat_target = fcn.run_invasion(D_target)
fcn.sat_target = sat_target

# plot target saturation
plt.figure(1)
plt.plot(pressure, sat_target, label='target')
plt.xlabel('Pressures')
plt.ylabel('Saturation')

# get initial diameters
key = jax.random.PRNGKey(1)
D0 = jax.random.uniform(key, shape=(Np,))
print(f'Initial loss: {fcn.sat_loss(D0)}')  # 6.320036460627565

# get initial saturation
sat0 = fcn.run_invasion(D0)
plt.figure(1)
plt.plot(pressure, sat0, label='Initial Guess')

# fit porosimetry
D, loss = fcn.fit_porosimetry(D0, solver=dfx.Euler(), t_span=(0, 1), dt=0.01)
print(f'Final loss: {fcn.sat_loss(D)}')  # 0.008630249199476726

# plot AI porosimetry
sat = fcn.run_invasion(D)
plt.figure(1)
plt.plot(pressure, sat, label='AI')
plt.legend()

# %% FLOW SIMULATION

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

# get target permeability
p = fcn.flow(D_target)
K_target = fcn.calc_K(p)
print(f'Target permeability: {K_target}')

# get initial permeabiity
p = fcn.flow(D0)
K0 = fcn.calc_K(p)
print(f'Initial permeability: {K0}')

# get previous solutin permeabiity
p = fcn.flow(D)
K = fcn.calc_K(p)
print(f'Old Solution permeability: {K}')

# Use JAX to fit cubic network
fcn.K_target = K_target
D, loss = fcn.fit_K(D, solver=dfx.Euler(), t_span=(0, 1), dt=0.1)

# get new solutin permeabiity or Q
p = fcn.flow(D)
K = fcn.calc_K(p)
print(f'New Solution permeability: {K}')

print(f"Avg D = {jnp.average(D)}")  # 0.4850915090568124
print(f"Min D = {jnp.min(D)}")  # 0.011316144272656644
print(f"Max D = {jnp.max(D)}")  # 0.9999999903538638
print(f"Loss = {loss}")  # 1.1088945252252821e-10


# Get porosimetry again
sat = fcn.run_invasion(D)
plt.figure(1)
plt.plot(pressure, sat, label='AI K')
plt.legend()
plt.show()
