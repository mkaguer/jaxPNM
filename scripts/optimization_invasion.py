
import os
from jax import config
import jax
import jax.numpy as jnp
import mypnmlib as pnm
import matplotlib.pyplot as plt
import diffrax as dfx

os.environ["JAX_PLATFORMS"] = "cpu"
config.update("jax_enable_x64", True)

def run_invasion(net, pressures):
   
    vp = net['pore.volume'] # pore volume
    vt = net['throat.volume'] # throat volume
    total_volume = jnp.sum(vp) + jnp.sum(vt)
    invaded_throats = jnp.zeros(24, bool) # boolean array that keeps track for invaded throats
    # FIXME: should update the line below so that it gets its value from boundary condition
    invaded_pores = net['pore.left'] # a boolean array to keep track of invaded pores
    count = jnp.sum(invaded_throats) # number of invaded throats
    sat_array = jnp.zeros_like(pressures, dtype=float) # an array that contains saturation for each given invading pressure

    for i, invading_pressure in enumerate(pressures):
        
        while True: 
            connected = pnm.models.find_neighbor_throats(net, invaded_pores)
            old_count = count
            invadable = net['throat.entry_pressure'] < invading_pressure
            invaded = jnp.logical_and(invadable, connected) # newly invaded throats
            invaded_throats += invaded # all the invaded throats
            count = jnp.sum(invaded_throats)
            if count == old_count:
                break
            invaded_pores = invaded_pores.at[net['throat.conns'][invaded_throats][:,0]].set(True)
            invaded_pores = invaded_pores.at[net['throat.conns'][invaded_throats][:,1]].set(True)

        sat = (jnp.sum(vt[invaded_throats]) + jnp.sum(vp[invaded_pores])) / total_volume
        sat_array = sat_array.at[i].set(sat)

    return sat_array

def R_squared(y, y_target):

    SSE = jnp.sum((y - y_target)**2)
    SST = jnp.sum((y_target - jnp.average(y_target))**2)
    R2 = 1 - SSE/SST

    return R2


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



# create network
spacing = 1e-4
net = pnm.network.make_cubic_network(shape=[4, 4, 1],spacing=spacing)
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
pressures = jnp.arange(1000, 51000, 3000)
# run invasion simulation
sat = run_invasion(net, pressures)

