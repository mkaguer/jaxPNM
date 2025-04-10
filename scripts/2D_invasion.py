import openpnm as op
import numpy as np

np.random.seed(1)

# create network
spacing=1
net = op.network.Cubic([10, 10, 1], spacing=1)

# add geometry models
geo_mods = op.models.collections.geometry.spheres_and_cylinders.copy()
net.add_model_collection(geo_mods)
net.regenerate_models()

# create phase object
phase = op.phase.Mercury(network=net)

# add phase physics model
model = op.models.physics.capillary_pressure._funcs.washburn
phase.add_model(propname='throat.entry_pressure',
                model=model,
                surface_tension='throat.surface_tension',
                contact_angle='throat.contact_angle',
                diameter='throat.diameter')

# set up algorithm
alg = op.algorithms.Drainage(phase=phase, network=net)
alg.set_inlet_BC(pores=net.pores('surface'))

# run
pressure = np.arange(0, 2, 0.01)/spacing
alg.run()

# export network
project = net.project
net['throat.radius'] = net['throat.diameter']/2
net['throat.invasion_sequence'] = alg['throat.invasion_sequence']
net['pore.invasion_sequence'] = alg['pore.invasion_sequence']
op.io.project_to_vtk(project, filename='../paraview/2d_invasion.vtk')