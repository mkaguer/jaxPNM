import openpnm as op
import numpy as np

# create network
net = op.network.Cubic(shape=[2, 1, 1], spacing=1)

# add geometry
net['pore.diameter'] = np.array([0.6, 0.45])
net['throat.diameter'] = np.array([0.225])
net['throat.radius'] = net['throat.diameter']/2

# export project to vtk
project = net.project
op.io.project_to_vtk(project,
                     filename='../paraview/sphere_and_cylinder')