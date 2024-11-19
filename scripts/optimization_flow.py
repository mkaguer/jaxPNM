"""
The goal is to write a script that fits a cubic network to have 
desired permeability using JAX and Diffrax.

Created by: Mike McKague
Date: November 18, 2024
"""

import models as mods

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


if __name__ == "__main__":
    
    # Step 1: create JAX network
    net = make_cubic_network(shape=[4, 1, 1], spacing=1e-4)
    
    # Step 2: add properties
    # a) 
    


