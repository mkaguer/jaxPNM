# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 22:15:03 2025

@author: mehrn
"""

import os
from jax import config
import jax
import jax.numpy as jnp
import mypnmlib as pnm
import matplotlib.pyplot as plt
# import models


os.environ["JAX_PLATFORMS"] = "cpu"
config.update("jax_enable_x64", True)

# create network
spacing = 1e-4
net = pnm.network.make_cubic_network(shape=[25, 25, 1],spacing=spacing)


# coords = jnp.array([[5.0e-07, 5.0e-07, 5.0e-07],
#  [5.0e-07, 1.5e-06, 5.0e-07],
#  [5.0e-07, 2.5e-06, 5.0e-07],
#  [5.0e-07, 3.5e-06, 5.0e-07],
#  [1.5e-06, 5.0e-07, 5.0e-07],
#  [1.5e-06, 1.5e-06, 5.0e-07],
#  [1.5e-06, 2.5e-06, 5.0e-07],
#  [1.5e-06, 3.5e-06, 5.0e-07],
#  [2.5e-06, 5.0e-07, 5.0e-07],
#  [2.5e-06, 1.5e-06, 5.0e-07],
#  [2.5e-06, 2.5e-06, 5.0e-07],
#  [2.5e-06, 3.5e-06, 5.0e-07]])
# conns = jnp.array([[ 0,  1],
#  [ 1,  2],
#  [ 2,  3],
#  [ 4,  5],
#  [ 5,  6],
#  [ 6,  7],
#  [ 8,  9],
#  [ 9, 10],
#  [10, 11],
#  [ 0,  4],
#  [ 1,  5],
#  [ 2,  6],
#  [ 3,  7],
#  [ 4,  8],
#  [ 5,  9],
#  [ 6, 10],
#  [ 7, 11]])
# net = {'pore.coords': coords, 'throat.conns': conns}

# add properties to network
Nt = net['throat.conns'].shape[0]
Np = net['pore.coords'].shape[0]
net['throat.length'] = jnp.ones(Nt) * spacing
# net['throat.viscosity'] = jnp.ones(Nt) * 1e-3
key = jax.random.PRNGKey(0)
net['throat.diameter'] = jax.random.uniform(key, shape=(Nt,)) * spacing # Fix me: we need random diameters

net['pore.contact_angle'] = jnp.ones(Nt) * 120 # for air
net['pore.surface_tension'] = jnp.ones(Nt) * 0.072
pc = pnm.models.washburn(net)
net['throat.entry_pressure'] = pc
# th_ind = pnm.models.find_neighbor_throats(net, pores=[0,1])
inlet_pores = net['pore.left'] # these are the pores at the left boundary
inlet_pores = jnp.where(inlet_pores)[0]
## algorithm starts here
net['pore.sat'] = jnp.zeros(Np)
net['throat.sat'] = jnp.zeros(Nt)
invaded_pores = jnp.array(inlet_pores)
new_sat = jnp.sum(net['throat.sat'])
invading_pressure = 100 # Pa
pc_mat = []
pc_mat.append(invading_pressure)
sat_mat = [0]
while True:
    invading_pressure += 1000
    print(invading_pressure)
    while True: 
        thr_neighbor_ind = pnm.models.find_neighbor_throats(net, invaded_pores)
        old_sat = new_sat
        throat_invadable_ind = net['throat.entry_pressure'] < invading_pressure
        throat_newly_invaded_ind = jnp.logical_and(thr_neighbor_ind, throat_invadable_ind) 
        net['throat.sat'] = net['throat.sat'].at[throat_newly_invaded_ind].set(jnp.ones(jnp.sum(throat_newly_invaded_ind)))
        new_sat = jnp.sum(net['throat.sat'])
        if new_sat == old_sat:
            break
        new_pores_1d = net['throat.conns'][throat_newly_invaded_ind].ravel()
        all_invaded_pores = jnp.concatenate([new_pores_1d, invaded_pores])
        invaded_pores = jnp.unique(all_invaded_pores) # removing duplicate pores
    print(f'new sat is: {new_sat}')
    pc_mat.append(invading_pressure)
    sat_mat.append(new_sat / Nt)
    if new_sat > 0.95 * Nt:
        break

fig, ax = plt.subplots()

ax.plot(pc_mat, sat_mat)
ax.set_ylabel('saturation')
ax.set_xlabel('pressure [pa]')







