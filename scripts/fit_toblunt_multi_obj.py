import os
from jax import config
import jax
import diffrax as dfx
import jax.numpy as jnp
import mypnmlib as pnm
import matplotlib.pyplot as plt
from _fit_cubic_network import FitCubicNetwork
import numpy as np

os.environ["JAX_PLATFORMS"] = "cpu"
config.update("jax_enable_x64", True)

# set spacing for data processing
spacing = 1e-4

# set contact angle and surface tension
sigma = 0.4791  # N/m
theta = 140  # radians!

# create network with spacing of 1 for optimizer
net = pnm.network.make_cubic_network(shape=[5, 5, 5], spacing=1)

# get Nt and Np
Nt = len(net['throat.conns'])
Np = len(net['pore.coords'])

# load experimental data
image = 'Berea'
data = jnp.array(np.loadtxt(image + '.csv', delimiter=','))
sat_target = data[:, 1]
pressures = data[:, 0]  # interpolate?


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
# pressure = jnp.arange(0.1, 2/spacing, 0.01/spacing)
pressure = pressures
fcn = FitCubicNetwork(net, surface_tension=sigma, contact_angle=theta, pressure=pressure, spacing=1, smoothing_factor=0.4)



# set target permeability
K_target =  17e2 * 0.98e-12 / 1000


# add experimental data
fcn.sat_target = sat_target
fcn.x_target = pressures
fcn.K_target = K_target

# get initial diameters
key = jax.random.PRNGKey(1)
D0 = jax.random.uniform(key, shape=(Np,))

# preprocess experimental data
fcn.process_pressure(spacing=spacing, mode='pre')
fcn.process_K(spacing=spacing, mode='pre')

print(f'Initial porosimetry loss: {fcn.sat_loss(D0)}')  # 0.16252246185541455
print(f'Initial flow loss: {fcn.K_loss(D0)}')  # 6.320036460627565

fcn.loss(D0)

# fit porosimetry
D, loss = fcn.fit_porosimetry_K(D0, solver=dfx.Euler(), t_span=(0, 30), dt=0.1)
print(f'Final loss: {fcn.sat_loss(D)}')  # 0.008630249196374291
print(f'Final flow loss: {fcn.K_loss(D)}')  # 6.320036460627565
p = fcn.flow(D)
print(f'Target permeability: {K_target} [m^2]')
print(f'final permeability: {fcn.calc_K(p)*spacing**2} [m^2]')
# get initial and final saturation
sat = fcn.run_invasion(D)
sat0 = fcn.run_invasion(D0)

# plot target, initial, and fitted saturations
plt.figure(1)
plt.plot(pressure, sat0, label='Initial Guess')
plt.plot(pressure, sat, label='Pc and K fitted')
plt.plot(pressures, sat_target, label='target')
plt.xlabel('Pressures')
plt.ylabel('Saturation')
plt.legend()
plt.show()

