"""
OpenPNM porosimetry example for comparison to JAX

Created by: Mohammad Mehrnia
Date: January 14th, 2025
"""

import openpnm as op
from jax import config
import numpy as np
import matplotlib.pyplot as plt

op.visualization.set_mpl_style()
config.update("jax_enable_x64", True)

# pick a numpy seed
np.random.seed(5)

# create network
spacing = 1e-4
shape = [20, 20, 1]
pn = op.network.Cubic(shape=shape, spacing=spacing)

# add geometry models
geo_models = op.models.collections.geometry.spheres_and_cylinders
pn.add_model_collection(geo_models)
pn.regenerate_models()

# create air phase
air = op.phase.Air(network=pn)
air['pore.contact_angle'] = 120
air['pore.surface_tension'] = 0.072

# add entry pressure model
f = op.models.physics.capillary_pressure.washburn
air.add_model(propname='throat.entry_pressure',
              model=f,
              surface_tension='throat.surface_tension',
              contact_angle='throat.contact_angle',
              diameter='throat.diameter')

# drainage algorithm
mip = op.algorithms.Drainage(network=pn, phase=air)
mip.set_inlet_BC(pores=pn.pores('left'))  # invasions starts from the left side
mip.run()

# plot pc curve
pc_mat = np.arange(100, 30000, 100)
data = mip.pc_curve(pressures=pc_mat)
plt.plot(data.pc, data.snwp, 'b-o')
plt.xlabel('Capillary Pressure [Pa]')
plt.ylabel('Non-Wetting Phase Saturation');
plt.show()

# export pore and throat diameters for JAX
np.savetxt('throat_diameters', pn['throat.diameter'])
np.savetxt('pore_diameters', pn['pore.diameter'])

print(np.average(air['throat.entry_pressure']))