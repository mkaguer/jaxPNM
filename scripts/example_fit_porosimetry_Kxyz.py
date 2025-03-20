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
spacing = 1

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
pressure = jnp.arange(0.1, 2, 0.01)
fcn = FitCubicNetwork(net, pressure=pressure, spacing=1)

# get sat_target
sat_target = fcn.run_invasion(D_target)

# get target permeability
px = fcn.flow(D_target, axis='x')
Kx_target = fcn.calc_K(px, axis='x')
py = fcn.flow(D_target, axis='y')
Ky_target = fcn.calc_K(py, axis='y')
pz = fcn.flow(D_target, axis='z')
Kz_target = fcn.calc_K(pz, axis='z')

K_target = jnp.array([Kx_target, Ky_target, Kz_target])
print(f'Target permeability: {K_target}')  # [0.00011388 0.00010884 0.0001218 ]

# add experimental data
fcn.sat_target = sat_target
fcn.x_target = pressure
fcn.K_target = K_target

# get initial diameters
key = jax.random.PRNGKey(1)
D0 = jax.random.uniform(key, shape=(Np,))
print(f'Initial loss: {fcn.sat_loss(D0)}')  # 2.522691006452184

# fit porosimetry
D, loss = fcn.fit_porosimetry_Kxyz(D0, solver=dfx.Euler(), t_span=(0, 10), dt=0.1)
print(f'Final loss: {fcn.sat_loss(D)}')  # 0.014499245785742104

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
plt.show()

