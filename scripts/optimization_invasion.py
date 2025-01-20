
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

    # get Nt
    Nt = len(net['throat.conns'])  # FIXME: add helper method
    # get volumes
    vp = net['pore.volume']
    vt = net['throat.volume']
    total_volume = jnp.sum(vp) + jnp.sum(vt)
    # boolean array that keeps track for invaded throats
    invaded_throats = jnp.zeros(Nt, bool)
    # a boolean array to keep track of invaded pores
    # FIXME: update so it gets from boundary condition
    invaded_pores = net['pore.left']
    # initialize counter, number of invaded throats
    count = jnp.sum(invaded_throats)
    # create array to contain saturation for each invading pressure
    sat_array = jnp.zeros_like(pressures, dtype=float)
    # loop through invading pressures
    for i, invading_pressure in enumerate(pressures):
        # ensure old_count does not equal count
        old_count = -1
        while count != old_count:
            # reset old_count
            old_count = count
            # find ALL invadable throats
            invadable = net['throat.entry_pressure'] < invading_pressure
            # find throats connected to invaded pores
            pores = jnp.where(invaded_pores, jnp.arange(Np), -1)
            connected = pnm.models.find_neighbor_throats(net, pores)
            # find connected AND invaded throats, newly invaded
            invaded = jnp.logical_and(invadable, connected)
            # add newly invaded throats
            invaded_throats += invaded
            # update count
            count = jnp.sum(invaded_throats)
            # find pores neighbouring ALL invaded throats
            throats = jnp.where(invaded_throats, jnp.arange(Nt), -1)
            pores = pnm.models.find_neighbor_pores(net, throats)
            # update list of invaded pores
            invaded_pores += pores
        # calculate invaded volume
        invaded_pore_volume = jnp.sum(vp * invaded_pores)
        invaded_throat_volume = jnp.sum(vt * invaded_throats)
        invaded_volume = invaded_pore_volume + invaded_throat_volume
        # calculate and update saturation
        sat = invaded_volume / total_volume
        sat_array = sat_array.at[i].set(sat)

    return sat_array


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
    # calculate R2
    # R2 = R_squared(sat, sat_target)
    # calculate SSE
    SSE = calc_sse(sat, sat_target)
    # calculate penalty
    # penalty = ...  # FIXME: add penalty to loss to fix diameters between 0 and 1
    # calculate loss
    # loss = R2 + penalty
    loss = SSE

    return loss


if __name__ == "__main__":

    # create network
    spacing = 1e-4
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
    pressures = jnp.arange(500, 20000, 500)

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
    jit_f = jax.jit(f)

    # add sat_target and pressures to net
    net['sat_target'] = sat
    net['pressures'] = pressures

    # test grad_f working
    D = jnp.array([0.8, 0.55, 0.35, 0.2]) * spacing
    print(grad_f(D, net))
    print(jit_f(D, net))

    # Define the ODE system for gradient flow: dx/dt = -grad(f)
    def dydt(t, y, net):
        return -grad_f(y, net)

    # Initial condition
    # key = random.PRNGKey(1)  # make results reproducible
    # y0 = random.uniform(key, shape=(Np,))
    y0 = D
    t0, t1 = 0, 10 # Time span (we treat the optimization as a "time" evolution)
    # Choose an ODE solver
    # solver = dfx.Tsit5()  # Tsit5 is a good general-purpose ODE solver
    solver = dfx.Euler()
    # Define the ODE problem
    term = dfx.ODETerm(dydt)
    # Solve the ODE, treating time as "iterations" for optimization
    solution = dfx.diffeqsolve(term, solver, t0=t0, t1=t1, dt0=1, y0=y0, args=net)
    # The final x value after "evolving" it toward the minimum
    x_min = solution.ys[-1]
    loss = f(x_min, net)
