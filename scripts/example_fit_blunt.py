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
data = jnp.array(np.loadtxt('../data/' + image + '.csv', delimiter=','))
sat_target = data[:, 1]
x_target = data[:, 0]  # interpolate?


# add the adjacency matrix
weights = jnp.arange(1, Nt+1)
am = pnm.network.create_adjacency_matrix(net, weights=weights, fmt='csr')
net['adjacency_matrix'] = am

# add "constant" Gh properties to network
net['pore.viscosity'] = jnp.ones(Np) * 1e-3
net['throat.viscosity'] = jnp.ones(Nt) * 1e-3

# assign boundary pores
net['pore.boundary'] = (
    net['pore.left'] +
    net['pore.right'] +
    net['pore.back'] +
    net['pore.front'] +
    net['pore.top'] +
    net['pore.bottom']
)


# set BCs in x direction
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
net['pore.bc.valuex'] = net['pore.bc.value']
net['pore.bc.maskx'] = net['pore.bc.mask']
net['boundary_poresx'] = net['boundary_pores']
net['rate_poresx'] = pores
del net['pore.bc.value'], net['pore.bc.mask'], net['boundary_pores']

# set BCs in y direction
pores = jnp.where(net['pore.front'])[0]
pnm.simulations.set_BC(net,
                       pores=pores,
                       bctype='value',
                       bcvalues=1.0,
                       mode='overwrite')
pores = jnp.where(net['pore.back'])[0]
pnm.simulations.set_BC(net,
                       pores=pores,
                       bctype='value',
                       bcvalues=0.0,
                       mode='add')
net['pore.bc.valuey'] = net['pore.bc.value']
net['pore.bc.masky'] = net['pore.bc.mask']
net['boundary_poresy'] = net['boundary_pores']
net['rate_poresy'] = pores
del net['pore.bc.value'], net['pore.bc.mask'], net['boundary_pores']

# set BCs in z direction
pores = jnp.where(net['pore.bottom'])[0]
pnm.simulations.set_BC(net,
                       pores=pores,
                       bctype='value',
                       bcvalues=1.0,
                       mode='overwrite')
pores = jnp.where(net['pore.top'])[0]
pnm.simulations.set_BC(net,
                       pores=pores,
                       bctype='value',
                       bcvalues=0.0,
                       mode='add')
net['pore.bc.valuez'] = net['pore.bc.value']
net['pore.bc.maskz'] = net['pore.bc.mask']
net['boundary_poresz'] = net['boundary_pores']
net['rate_poresz'] = pores
del net['pore.bc.value'], net['pore.bc.mask'], net['boundary_pores']

# create instance of FitCubicNetwork
# pressures = jnp.arange(0.1, 2/spacing, 0.01/spacing)
pressures = jnp.arange(1e3, 1.5e5, 1e3)
fcn = FitCubicNetwork(net,
                      surface_tension=sigma,
                      contact_angle=theta,
                      pressure=pressures,
                      spacing=1,
                      smoothing_factor=0.4)

# set target permeability
Kx_target = 1659 * 0.98e-12 / 1000
Ky_target = 1801 * 0.98e-12 / 1000
Kz_target = 1872 * 0.98e-12 / 1000
K_target = jnp.array([Kx_target, Ky_target, Kz_target])

# add experimental data
fcn.sat_target = sat_target
fcn.x_target = x_target
fcn.K_target = K_target

# get initial diameters
key = jax.random.PRNGKey(1)
D0 = jax.random.uniform(key, shape=(Np,))

# preprocess experimental data
fcn.process_pressure(spacing=spacing, mode='pre')
fcn.process_K(spacing=spacing, mode='pre')

print(f'Initial porosimetry loss: {fcn.sat_loss(D0)}')  # 0.1625886817705871

# fit porosimetry
D, loss = fcn.fit_porosimetry_Kxyz(D0, solver=dfx.Euler(), t_span=(0, 30), dt=0.1)
print(f'Final porosimetry loss: {fcn.sat_loss(D)}')  # 0.0021440948077985815

# calculate final permeabilities in each direction
px = fcn.flow(D, axis='x')
Kx = fcn.calc_K(px, axis='x')
py = fcn.flow(D, axis='y')
Ky = fcn.calc_K(py, axis='y')
pz = fcn.flow(D, axis='z')
Kz = fcn.calc_K(pz, axis='z')

print(f'Target permeability: {K_target} [m^2]')
print(f'final permeability: {jnp.array([Kx, Ky, Kz])*spacing**2} [m^2]')
# [1.62781178e-12 1.76469849e-12 1.83649728e-12]

# get initial and final saturation
sat = fcn.run_invasion(D)
sat0 = fcn.run_invasion(D0)
sat = jnp.interp(x_target, pressures, sat)
sat0 = jnp.interp(x_target, pressures, sat0)

# plot target, initial, and fitted saturations
plt.figure(1)
plt.plot(x_target, sat0, label='Initial Guess')
plt.plot(x_target, sat, label='Pc and K fitted')
plt.plot(x_target, sat_target, label='target')
plt.xlabel('Pressures')
plt.ylabel('Saturation')
plt.legend()
plt.show()

jnp.save('../data/ai-diameters-' + image, jnp.array([D0, D]).T)
