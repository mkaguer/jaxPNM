# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 13:22:34 2025

@author: mehrn
"""

import openpnm as op
from jax import config
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

op.visualization.set_mpl_style()

config.update("jax_enable_x64", True)

# create network
spacing = 1e-4

np.random.seed(5)
spacing=1e-4
shape=[20, 20, 1]
pn = op.network.Cubic(shape=shape, spacing=spacing)
air = op.phase.Air(network=pn)
key = jax.random.PRNGKey(0)
D = jax.random.uniform(key, shape=(pn.Nt,)) * spacing
D = np.array(D)
pn['throat.diameter'] = D


air['pore.contact_angle'] = 120
air['pore.surface_tension'] = 0.072
f = op.models.physics.capillary_pressure.washburn
air.add_model(propname='throat.entry_pressure',
              model=f, 
              surface_tension='throat.surface_tension',
              contact_angle='throat.contact_angle',
              diameter='throat.diameter',)

Pc = air['throat.entry_pressure']

# FIXME: we need models here 
pn['pore.volume'] = 1 
pn['throat.volume'] = 1

mip = op.algorithms.Drainage(network=pn, phase=air)
mip.set_inlet_BC(pores=pn.pores('left'))  # invasions starts from the left side
mip.run()
pc_mat = np.arange(100, 30000,100 )
data = mip.pc_curve(pressures=pc_mat)
plt.plot(data.pc, data.snwp, 'b-o')
plt.xlabel('Capillary Pressure [Pa]')
plt.ylabel('Non-Wetting Phase Saturation');

         

