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
import diffrax as dfx

os.environ["JAX_PLATFORMS"] = "cpu"
config.update("jax_enable_x64", True)

# create network
spacing = 1e-4
net = pnm.network.make_cubic_network(shape=[4, 4, 1],spacing=spacing)



# add properties to network
Nt = 24# net['throat.conns'].shape[0]
Np = 16#net['pore.coords'].shape[0]

# net['throat.viscosity'] = jnp.ones(24) * 1e-3
key = jax.random.PRNGKey(0)

# net['throat.length'] = jnp.ones(24) * spacing
# net['throat.diameter'] = jax.random.uniform(key, shape=(24,)) *0.3 * spacing # FIXME: needs random diameter

# net['pore.contact_angle'] = jnp.ones(24) * 120 # for air
# net['pore.surface_tension'] = jnp.ones(24) * 0.072
# pc = pnm.models.washburn(net)
# net['throat.entry_pressure'] = pc
# # th_ind = pnm.models.find_neighbor_throats(net, pores=[0,1])
# inlet_pores = jnp.array([0, 1, 2, 3]) # these are the pores at the left boundary
# net['pore.inlet'] = inlet_pores
def R_squared(y, y_target):

    SSE = jnp.sum((y - y_target)**2)
    SST = jnp.sum((y_target - jnp.average(y_target))**2)
    R2 = 1 - SSE/SST

    return R2

def run_invasion(net, pressures):
   
    vp = net['pore.volume']
    vt = net['throat.volume']
    total_volume = jnp.sum(vp) + jnp.sum(vt)
    invaded_throats = jnp.zeros(24, bool)
    invaded_pores = net['pore.left']
    count = jnp.sum(invaded_throats)
    invading_pressure = 100 # Pa
    # pc_mat = []
    # pc_mat.append(invading_pressure)
    # sat_mat = [0]
    sat_array = jnp.zeros_like(pressures, dtype=float)
    # while True:
    for i, invading_pressure in enumerate(pressures):
        # invading_pressure += 3000
        print(invading_pressure)
        while True: 
            connected = pnm.models.find_neighbor_throats(net, invaded_pores)
            old_count = count
            invadable = net['throat.entry_pressure'] < invading_pressure
            invaded = jnp.logical_and(invadable, connected) 
            # net['throat.sat'] = net['throat.sat'].at[throat_newly_invaded_ind].set(jnp.ones(jnp.sum(throat_newly_invaded_ind)))
            invaded_throats += invaded
            count = jnp.sum(invaded_throats)
            if count == old_count:
                break
            invaded_pores = invaded_pores.at[net['throat.conns'][invaded_throats][:,0]].set(True)
            invaded_pores = invaded_pores.at[net['throat.conns'][invaded_throats][:,1]].set(True)
            # new_pores_1d = net['throat.conns'][throat_newly_invaded_ind]
            # all_invaded_pores = jnp.concatenate([new_pores_1d, invaded_pores])
            # invaded_pores = jnp.unique(all_invaded_pores) # removing duplicate pores
        print(f'new sat is: {count}')
        sat = (jnp.sum(vt[invaded_throats]) + jnp.sum(vp[invaded_pores])) / total_volume
         
        sat_array = sat_array.at[i].set(sat)
        # pc_mat.append(invading_pressure)
        # sat_mat.append(count / 24)
        # if count > 0.85 * Nt:
        #     break
    return sat_array




def f(D, net, sat_target, pressures):

    # set pore diameter
    net['pore.diameter'] = D
    # regenerate geometry models
    net['throat.diameter'] = pnm.models.throat_diameter(network=net)
    net['throat.length'] = pnm.models.throat_length(network=net)
    net['pore.volume'] = pnm.models.sphere(network=net)
    net['throat.total_volume'] = pnm.models.cylinder(network=net)
    net['throat.lens_volume'] = pnm.models.lens(network=net)
    props = ['throat.total_volume', 'throat.lens_volume']
    net['throat.volume'] = pnm.models.difference(network=net, props=props)
    # add entry pressure model
    net['throat.contact_angle'] = 120
    net['throat.surface_tension'] = 0.072
    Pc = pnm.models.washburn(network=net)
    net['throat.entry_pressure'] = Pc
    # run invasion simulation
    sat = run_invasion(net, pressures)
    # calculate R2
    R2 = R_squared(sat, sat_target)
    # calculate penalty
    penalty = ...  # FIXME: add penalty to loss to fix diameters between 0 and 1
    # calculate loss
    loss = R2 + penalty

    return loss

key = jax.random.PRNGKey(0)

D = jax.random.uniform(key, shape=(16,)) *0.3 * spacing
# D = jnp.ones(shape=(16,)) * 0.15 * spacing
net['pore.diameter'] = D
# regenerate geometry models
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
# set the range of pressures to be investigated
pressures = jnp.arange(1000,101000, 10000)
# run invasion simulation
sat = run_invasion(net, pressures)
# loss = calc_loss(testD, net)
# fig, ax = plt.subplots()

# ax.plot(pc_mat, sat_mat)
# ax.set_ylabel('saturation')
# ax.set_xlabel('pressure [pa]')

# # Define the gradient of f(x)
# grad_f = jax.grad(calc_loss)
# print(grad_f(testD, net))

# def dydt(t, y, net):
#     return -grad_f(y, net)

# # Initial condition
# y0 = jnp.ones(shape=(24,)) * 0.15 * spacing
# t0, t1 = 0, 1000  # Time span (we treat the optimization as a "time" evolution)

# # Choose an ODE solver
# solver = dfx.Tsit5()  # Tsit5 is a good general-purpose ODE solver

# # Define the ODE problem
# term = dfx.ODETerm(dydt)

# # Pass the net_static and net as a tuple to the args
# solution = dfx.diffeqsolve(
#     term,
#     solver,
#     t0=t0,
#     t1=t1,
#     dt0=1,
#     y0=y0,
#     args=net
# )
# # The final x value after "evolving" it toward the minimum
# x_min = solution.ys[-1]
# print(f"Minimum found at x = {x_min}, f(x) = {calc_loss(x_min, net)}")

# # visualize loss as a function of one of the parameters!
# D_vals = jnp.linspace(0.0, 1.0, 100)
# losses = jnp.array([calc_loss(jnp.array([D, 0.3, 0.8]), net) for D in D_vals])

# plt.plot(D_vals, losses)
# plt.xlabel("D[0]")
# plt.ylabel("Loss")
# plt.title("Loss Landscape Along D[0]")
# plt.show()







