import openpnm as op

res = 4e-6
image = "S5"

net = op.io.network_from_csv(filename="../networks/" + image + "-snow.csv")
net['pore.all'] = net['pore.all'].astype(bool)
net['pore.boundary'] = net['pore.boundary'].astype(bool)
net['pore.xmin'] = net['pore.xmin'].astype(bool)
net['pore.xmax'] = net['pore.xmax'].astype(bool)
net['pore.ymin'] = net['pore.ymin'].astype(bool)
net['pore.ymax'] = net['pore.ymax'].astype(bool)
net['pore.zmin'] = net['pore.zmin'].astype(bool)
net['pore.zmax'] = net['pore.zmax'].astype(bool)

# trim boundary pores
pores = net['pore.boundary'].astype(bool)
op.topotools.trim(net, pores=pores)

net['throat.radius'] = net['throat.inscribed_diameter']/2
net['pore.coords'] = net['pore.coords'] / res + 7
net['pore.diameter'] = net['pore.equivalent_diameter'] / res
project = net.project
op.io.project_to_vtk(project, filename="../paraview/" + image + "-snow.vtk")