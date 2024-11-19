"""
The goal is to write a script that fits a cubic network to have 
desired permeability using JAX and Diffrax.

Created by: Mike McKague
Date: November 18, 2024
"""

import models as mods
import jax.experimental.sparse as js

def make_cubic_network(shape, spacing=1, connectivity=6):
    '''
    Create custom cubic network that returns dictionary of JAX arrays.
    '''
    network = mods.cubic(shape=shape,
                         spacing=spacing,
                         connectivity=connectivity,
                         node_prefix='pore',
                         edge_prefix='throat')
    return network


def calc_throat_length(net):

    conns = net['throat.conns']
    coords = net['pore.coords'][conns]
    diff = jnp.diff(coords, axis=1)
    throat_length = jnp.linalg.norm(diff, axis=2)[:, 0]

    return throat_length


def calc_conductance(net):
    
    R = net['throat.diameter'] / 2
    mu = net['throat.viscosity']
    L = net['throat.length']
    
    return jnp.pi * R ** 4 / 8 / mu / L


def build_A_and_b(net):
    
    g = net['throat.conductance']
    am = mods.create_adjacency_matrix(net, weights=g, fmt='coo')
    # laplacian
    
    return am


if __name__ == "__main__":
    
    import jax.numpy as jnp
    
    # Step 1: create JAX network
    net = make_cubic_network(shape=[4, 1, 1], spacing=1e-4)
    
    # get Np and Nt
    Np = len(net['pore.coords'])
    Nt = len(net['throat.conns'])
    
    # Step 2: add properties
    # a) add viscosity
    net['throat.viscosity'] = jnp.ones(Nt)*1e-3  # keep it constant
    # b) add diameter
    net['throat.diameter'] = jnp.ones(Nt)*5e-5  # keep it constant
    # c) calculate throat length and add!
    L = calc_throat_length(net)
    net['throat.length'] = L
    
    # Step 3: calculate conductance
    G = calc_conductance(net)
    net['throat.conductance'] = G
    
    # Step 4: build A
    am = build_A_and_b(net)
    print(am)
    
    
    
    
    
    
    


