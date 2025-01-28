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
spacing = 1
net = pnm.network.make_cubic_network(shape=[5, 5, 5], spacing=spacing)

# get Nt and Np
Nt = len(net['throat.conns'])
Np = len(net['pore.coords'])

# get target diameters
key = jax.random.PRNGKey(0)
D = jax.random.uniform(key, shape=(Np,)) * spacing
net['pore.diameter'] = D

# add geometry models
net['throat.diameter'] = pnm.models.throat_diameter(net)
net['throat.length'] = pnm.models.throat_length(net)
net['pore.volume'] = pnm.models.sphere(net)
net['throat.total_volume'] = pnm.models.cylinder(net)
net['throat.lens_volume'] = pnm.models.lens(network=net)
props = ['throat.total_volume', 'throat.lens_volume']
net['throat.volume'] = pnm.models.difference(network=net, props=props)

# add entry pressure model
net['throat.contact_angle'] = 120
net['throat.surface_tension'] = 0.072
Pc = pnm.models.washburn(network=net)
net['throat.entry_pressure'] = Pc

# add the adjacency matrix
weights = jnp.arange(1, Nt+1)
am = pnm.network.create_adjacency_matrix(net, weights=weights, fmt='csr')
net['adjacency_matrix'] = am

# create instance of FitCubicNetwork
pressure = jnp.arange(0.1, 2, 0.01)
fcn = FitCubicNetwork(net, pressure=pressure)

# add sat_target as attribute to fcn
sat_target = fcn.run_invasion()
fcn.sat_target = sat_target

# plot target saturation
plt.figure(1)
plt.plot(pressure, sat_target, label='target')
plt.xlabel('Pressures')
plt.ylabel('Saturation')

# get initial diameters
key = jax.random.PRNGKey(1)
D0 = jax.random.uniform(key, shape=(Np,)) * spacing
print(f'Initial loss: {fcn.sat_loss(D0)}')  # 3.8878519491468686

# get initial saturation
sat0 = fcn.run_invasion()
plt.figure(1)
plt.plot(pressure, sat0, label='Initial Guess')

# fit porosimetry
D, loss = fcn.fit_porosimetry(D0, solver=dfx.Euler(), t_span=(0, 1), dt=0.01)
print(f'Final loss: {fcn.sat_loss(D)}')  # 0.010063525072810366 

# plot AI porosimetry
sat = fcn.run_invasion()
plt.figure(1)
plt.plot(pressure, sat, label='AI')
plt.legend()
plt.show()
