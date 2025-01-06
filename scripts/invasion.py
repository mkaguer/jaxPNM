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
import models


os.environ["JAX_PLATFORMS"] = "cpu"
config.update("jax_enable_x64", True)

# create network
spacing = 1
coords = jnp.array([[0, 0, 0],
                    [1, 0, 0],
                    [2, 0, 0],
                    [3, 0, 0]])
conns = jnp.array([[0, 1],
                   [1, 2],
                   [2, 3]])
net = {'pore.coords': coords, 'throat.conns': conns}

# add properties to network
Nt = conns.shape[0]
net['throat.length'] = jnp.ones(Nt) * spacing
# net['throat.viscosity'] = jnp.ones(Nt) * 1e-3
key = jax.random.PRNGKey(0)
net['throat.diameter'] = jax.random.uniform(key, shape=(Nt,))  # Fix me: we need random diameters
net['pore.contact_angle'] = jnp.ones(Nt) * 120 # for air
net['pore.surface_tension'] = jnp.ones(Nt) * 0.072
pc = pnm.models.washburn(net)





