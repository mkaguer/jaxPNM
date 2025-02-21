
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

# set spacing for data processing
spacing = 1e-4

# create network with spacing of 1 for optimizer
net = pnm.network.make_cubic_network(shape=[5, 5, 5], spacing=1)

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
pressure = jnp.arange(0.1, 2, 0.01)
fcn = FitCubicNetwork(net, pressure=pressure, spacing=1)

# add sat_target as attribute to fcn
sat_target = fcn.run_invasion(D_target)

# add "experimental" data
fcn.sat_target = sat_target
fcn.pressure = pressure/spacing
fcn.x_target = pressure/spacing

# pre-process experimental data
fcn.process_pressure(spacing=spacing, mode='pre')

# get initial diameters
key = jax.random.PRNGKey(1)
D0 = jax.random.uniform(key, shape=(Np,))
print(f'Initial loss: {fcn.sat_loss(D0)}')  # 6.320036460627565

# fit porosimetry
D, loss = fcn.fit_porosimetry(D0, solver=dfx.Euler(), t_span=(0, 1), dt=0.01)
print(f'Final loss: {fcn.sat_loss(D)}')  # 0.008630249196374291

# get initial saturation
sat = fcn.run_invasion(D)
sat0 = fcn.run_invasion(D0)

# plot target, initial, and fitted saturations
plt.figure(1)
plt.plot(pressure, sat0, label='Initial Guess')
plt.plot(pressure, sat, label='AI')
plt.plot(pressure, sat_target, label='target')
plt.xlabel('Pressures')
plt.ylabel('Saturation')
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

# add "experimental" K
fcn.K_target = K_target * spacing ** 2
print(f'Experimental permeability: {fcn.K_target}')  # 1.138793222360462e-12

# pre-process experimental data
fcn.process_K(spacing=spacing, mode='pre')
print(f'Target permeability: {K_target}')  # 0.0001138793222360462

# get initial permeabiity
p = fcn.flow(D0)
K0 = fcn.calc_K(p)
print(f'Initial permeability: {K0}')  # 4.470684518764306e-05

# get previous solutin permeabiity
p = fcn.flow(D)
K = fcn.calc_K(p)
print(f'Old Solution permeability: {K}')  # 8.65525279274762e-05

# Use JAX to fit cubic network
fcn.K_target = K_target
D, loss = fcn.fit_K(D, solver=dfx.Euler(), t_span=(0, 1), dt=0.1)

# get new solutin permeabiity or Q
p = fcn.flow(D)
K = fcn.calc_K(p)
print(f'New Solution permeability: {K}')  # 0.00011387811500839045

print(f"Avg D = {jnp.average(D)}")  # 0.4845949743541203
print(f"Min D = {jnp.min(D)}")  # 0.011315972655708953
print(f"Max D = {jnp.max(D)}")  # 0.9999162466969737
print(f"Loss = {loss}")  # 1.1237989043420169e-10

# Get porosimetry again
satK = fcn.run_invasion(D)
plt.figure(1)
plt.plot(pressure, satK, label='AI K')
plt.legend()
plt.show()
plt.title('Scaled Data')

# post process data
fcn.process_pressure(spacing=spacing, mode='post')
fcn.process_K(spacing=spacing, mode='post')

print(f'Fitted permeability: {K * spacing ** 2}')  # 1.1387811500839045e-12

# plot fitted saturations to "experimental data"
plt.figure(2)
plt.plot(fcn.pressure, sat0, label='Initial Guess')
plt.plot(fcn.pressure, sat, label='AI')
plt.plot(fcn.pressure, sat_target, label='target')
plt.plot(fcn.pressure, satK, label='AI K')
plt.xlabel('Pressures')
plt.ylabel('Saturation')
plt.legend()
plt.show()