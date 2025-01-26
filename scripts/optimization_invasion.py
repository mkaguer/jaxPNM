import os
from jax import config
import jax
import jax.numpy as jnp
from jax import lax
import mypnmlib as pnm
import matplotlib.pyplot as plt
import diffrax as dfx
import time

os.environ["JAX_PLATFORMS"] = "cpu"
config.update("jax_enable_x64", True)


def find_invasion_pressure(net):

    # get Nt and Np
    Nt = len(net['throat.conns'])
    Np = len(net['pore.volume'])
    # get capillary pressure
    pc = net['throat.entry_pressure']
    # get maximum entry pressure
    max_pc = jnp.max(pc)
    # initialize invaded_throats and invaded_pores
    invaded_throats = jnp.zeros(Nt, dtype=bool)
    invaded_pores = net['pore.left']  # FIXME: update so it gets from bc

    def body_fun(i, state):
        # breakout state
        invaded_throats, invaded_pores, pressure, invasion_pressure = state
        # find throats connected to set of invaded pores
        pores = jnp.where(invaded_pores, jnp.arange(Np), -1)
        connected = pnm.models.find_neighbor_throats(net, pores)
        # find invadable throats (set of connected but not invaded)
        invadable = jnp.logical_xor(connected, invaded_throats)
        # get index of invadable throat with minimum pressure
        index = jnp.argmin(jnp.where(invadable, pc, max_pc*2))
        # update pressure
        pressure = jnp.maximum(pressure, pc[index])
        # update invasion pressire for throat
        invasion_pressure = invasion_pressure.at[index].set(pressure)
        # add newly invaded throats
        invaded_throats = invaded_throats.at[index].set(True)
        # update invaded pores, find ALL pores neighbouring invaded throats!
        throats = jnp.where(invaded_throats, jnp.arange(Nt), -1)
        invaded_pores = pnm.models.find_neighbor_pores(net, throats)
        return invaded_throats, invaded_pores, pressure, invasion_pressure

    # initialize state for the for loop
    pressure = 0
    invasion_pressure = jnp.zeros(Nt)
    init_state = (invaded_throats, invaded_pores, pressure, invasion_pressure)
    # run for loop
    final_state = lax.fori_loop(0, Nt, body_fun, init_state)
    # get invasion pressure from state
    invasion_pressure = final_state[-1]

    return invasion_pressure


def run_invasion(net, pressure):

    # get invasion pressures (Nt,)
    invasion_pressure = find_invasion_pressure(net)
    # get volumes
    vp = net['pore.volume']
    vt = net['throat.volume']
    total_volume = jnp.sum(vp) + jnp.sum(vt)

    def body_func(i, state):
        # breakout state
        saturation = state
        # apply sigmoid function
        sf = 0.01
        throat_prob = jax.nn.sigmoid((pressure[i] - invasion_pressure)/sf)
        # find pore probability
        pore_prob = pnm.models.get_max_of_neighbor_throats(net, throat_prob)
        pore_prob = jnp.where(net['pore.left'], 1.0, pore_prob)
        # calculate invaded volume
        invaded_volume = jnp.sum(vp * pore_prob) + jnp.sum(vt * throat_prob)
        # calculate saturation
        sat = invaded_volume/total_volume
        saturation = saturation.at[i].set(sat)
        return saturation

    # initialize saturation
    saturation = jnp.zeros(len(pressure), dtype=float)  # FIXME: minus bp vol
    # run for loop
    state = saturation
    sat = lax.fori_loop(0, len(pressure), body_func, state)

    return sat


def R_squared(y, y_target):

    SSE = jnp.sum((y - y_target)**2)
    SST = jnp.sum((y_target - jnp.average(y_target))**2)
    R2 = 1 - SSE/SST

    return R2


def calc_sse(y, y_target):

    SSE = jnp.sum((y - y_target)**2)

    return SSE


def f(D, net):

    # retrieve sat_target and pressures from network
    sat_target = net['sat_target']
    pressures = net['pressures']
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
    # calculate SSE
    SSE = calc_sse(sat, sat_target)
    # calculate penalty
    lbd = jnp.maximum(0.0 - D, 0)
    ubd = jnp.maximum(D - 1.0, 0)
    penalty = jnp.sum(lbd**2 + ubd**2)
    # calculate loss
    loss = SSE + penalty

    return loss


if __name__ == "__main__":

    # create network
    spacing = 1
    net = pnm.network.make_cubic_network(shape=[4, 1, 1], spacing=spacing)

    # get Nt and Np
    Nt = len(net['throat.conns'])
    Np = len(net['pore.coords'])

    # get diameters
    # key = jax.random.PRNGKey(0)
    # D = jax.random.uniform(key, shape=(Np,)) * spacing
    D = jnp.array([0.9, 0.5, 0.4, 0.25]) * spacing
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
    pressures = jnp.arange(0.1, 2, 0.01)

    # add the adjacency matrix
    weights = jnp.arange(1, Nt+1)
    am = pnm.network.create_adjacency_matrix(net, weights=weights, fmt='csr')
    net['adjacency_matrix'] = am

    # run invasion simulation
    sat = run_invasion(net, pressures)

    # plot
    plt.figure(1)
    plt.plot(pressures, sat)
    plt.xlabel('Pressures')
    plt.ylabel('Saturation')
    plt.show()

    # try to take gradient
    grad_f = jax.grad(f)

    # add sat_target and pressures to net
    net['sat_target'] = sat
    net['pressures'] = pressures

    # test grad_f working
    D = jnp.array([0.8, 0.55, 0.35, 0.2]) * spacing
    print(f(D, net))

    start = time.time()
    print(grad_f(D, net))
    stop = time.time()
    print(f'{stop - start:.4f}s')

    # Define the ODE system for gradient flow: dx/dt = -grad(f)
    def dydt(t, y, net):
        return jnp.clip(-grad_f(y, net), -10.0, 10.0)

    # Initial condition
    y0 = D
    t0, t1 = 0, 10  # Time span (we treat the optimization as a "time" evolution)
    # Choose an ODE solver
    # solver = dfx.Tsit5()  # Tsit5 is a good general-purpose ODE solver
    solver = dfx.Euler()
    # Define the ODE problem
    term = dfx.ODETerm(dydt)
    # Solve the ODE, treating time as "iterations" for optimization
    start = time.time()
    solution = dfx.diffeqsolve(term, solver, t0=t0, t1=t1, dt0=0.01, y0=y0, args=net, max_steps=10000)
    stop = time.time()
    print(f'{stop - start:.4f}s')
    # The final x value after "evolving" it toward the minimum
    x_min = solution.ys[-1]
    loss = f(x_min, net)
    print(x_min)
    print(loss)

    # visualize loss as a function of one of the parameters!
    D_vals = jnp.linspace(0.01, 1.0, 100)
    losses = jnp.array([f(jnp.array([0.8, D, 0.35, 0.2]), net) for D in D_vals])

    plt.plot(D_vals, losses)
    plt.xlabel("D[0]")
    plt.ylabel("Loss")
    plt.title("Loss Landscape Along D[0]")
    plt.show()