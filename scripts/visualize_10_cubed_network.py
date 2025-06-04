import openpnm as op
import numpy as np

np.random.seed(1)

# create network
net = op.network.Cubic(shape=[10, 10, 10], spacing=1)

# add geometry models
geo_mods = op.models.collections.geometry.spheres_and_cylinders.copy()
net.add_model_collection(geo_mods)
net.regenerate_models()

# create phase object
phase = op.phase.Mercury(network=net)
phase.add_model(propname='throat.entry_pressure',
                model=op.models.physics.capillary_pressure._funcs.washburn,
                surface_tension='throat.surface_tension',
                contact_angle='throat.contact_angle',
                diameter='throat.diameter')

# create drainage algorithm
drn = op.algorithms.Drainage(network=net, phase=phase)

# run drainage
inlet = net.pores(['xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax'])
drn.set_inlet_BC(pores=inlet)
pressures=np.arange(0, 20, 0.1)
drn.run(pressures=pressures)

# save sampled network
net['throat.radius'] = net['throat.diameter']/2
net['pore.invasion_sequence'] = drn['pore.invasion_sequence']
net['throat.invasion_sequence'] = drn['throat.invasion_sequence']
op.io.project_to_vtk(project=net.project, filename='../paraview/cubic.vtk')
