import openpnm as op

net = op.network.Cubic(shape=[10, 10, 10], spacing=1)
geo_mods = op.models.collections.geometry.spheres_and_cylinders.copy()
net.add_model_collection(geo_mods)
net.regenerate_models()
op.io.project_to_vtk(project=net.project, filename='../paraview/cubic.vtk')