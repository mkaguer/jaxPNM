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
phase = op.phase.Water(network=net)

# add phase physics model
model = op.models.physics.hydraulic_conductance._funcs.generic_hydraulic
phase.add_model(propname='throat.hydraulic_conductance',
                model=model,
                pore_viscosity='pore.viscosity',
                throat_viscosity='throat.viscosity',
                size_factors='throat.hydraulic_size_factors')

# set up algorithm
alg = op.algorithms.StokesFlow(phase=phase, network=net)
alg.set_BC(pores=net.pores('left'), bctype='value', bcvalues=1.0)
alg.set_BC(pores=net.pores('right'), bctype='value', bcvalues=0.0)

# run
alg.run()

# export network
project = net.project
net['throat.radius'] = net['throat.diameter']/2
net['pore.pressure'] = alg['pore.pressure']
op.io.project_to_vtk(project, filename='../paraview/2d_flow.vtk')