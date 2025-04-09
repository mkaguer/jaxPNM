import numpy as np
import openpnm as op

image = "Berea"

shape = [15, 15, 15]

# load sampled network
data = np.load('../networks/sampled-' + image + f'-{shape}' + '.npz')
data = {key: np.array(data[key]) for key in data.files}

# create a new clearn network from data
net_s = {}
net_s['pore.coords'] = data['pore.coords']
net_s['throat.conns'] = data['throat.conns']
net_s['pore.diameter'] = data['pore.diameter']
net_s['pore.xmin'] = data['pore.left']
net_s['pore.xmax'] = data['pore.right']
net_s['pore.ymin'] = data['pore.front']
net_s['pore.ymax'] = data['pore.back']
net_s['pore.zmin'] = data['pore.bottom']
net_s['pore.zmax'] = data['pore.top']

# calculate throat diameter from tsf
D = net_s['pore.diameter']
conns = net_s['throat.conns']
tsf = data['throat.tsf']
net_s['throat.diameter'] = tsf * np.min(D[conns], axis=1)

# load fitted network
data = np.load('../networks/fitted-' + image + '.npz')
data = {key: np.array(data[key]) for key in data.files}

# create a new clearn network from data
net_f = {}
net_f['pore.coords'] = data['pore.coords']
net_f['throat.conns'] = data['throat.conns']
net_f['pore.diameter'] = data['pore.diameter']
net_f['pore.xmin'] = data['pore.left']
net_f['pore.xmax'] = data['pore.right']
net_f['pore.ymin'] = data['pore.front']
net_f['pore.ymax'] = data['pore.back']
net_f['pore.zmin'] = data['pore.bottom']
net_f['pore.zmax'] = data['pore.top']

# calculate throat diameter from tsf
D = net_f['pore.diameter']
conns = net_f['throat.conns']
tsf = data['throat.tsf']
net_f['throat.diameter'] = tsf * np.min(D[conns], axis=1)

# create openpnm network 
pn_s = op.io.network_from_porespy(net_s)
pn_f = op.io.network_from_porespy(net_f)

# add geometry models to sampled network
pn_s.add_model(propname='throat.length',
               model=op.models.geometry.throat_length._funcs.spheres_and_cylinders,
               pore_diameter='pore.diameter',
               throat_diameter='throat.diameter')
pn_s.add_model(propname='throat.total_volume',
               model=op.models.geometry.throat_volume._funcs.cylinder,
               throat_diameter='throat.diameter',
               throat_length='throat.length')
pn_s.add_model(propname='throat.lens_volume',
               model=op.models.geometry.throat_volume._funcs.lens,
               throat_diameter='throat.diameter',
               pore_diameter='pore.diameter')
pn_s.add_model(propname='throat.volume',
               model=op.models.misc._basic_math.difference,
               props=['throat.total_volume', 'throat.lens_volume'])
pn_s.add_model(propname='pore.volume',
               model=op.models.geometry.pore_volume._funcs.sphere,
               pore_diameter='pore.diameter')

# add geometry models to fitted network
pn_f.add_model(propname='throat.length',
               model=op.models.geometry.throat_length._funcs.spheres_and_cylinders,
               pore_diameter='pore.diameter',
               throat_diameter='throat.diameter')
pn_f.add_model(propname='throat.total_volume',
               model=op.models.geometry.throat_volume._funcs.cylinder,
               throat_diameter='throat.diameter',
               throat_length='throat.length')
pn_f.add_model(propname='throat.lens_volume',
               model=op.models.geometry.throat_volume._funcs.lens,
               throat_diameter='throat.diameter',
               pore_diameter='pore.diameter')
pn_f.add_model(propname='throat.volume',
               model=op.models.misc._basic_math.difference,
               props=['throat.total_volume', 'throat.lens_volume'])
pn_f.add_model(propname='pore.volume',
               model=op.models.geometry.pore_volume._funcs.sphere,
               pore_diameter='pore.diameter')

# create phase objects for both networks
phase_s = op.phase.Mercury(network=pn_s)
phase_f = op.phase.Mercury(network=pn_f)

# add entry pressure model
phase_s.add_model(propname='throat.entry_pressure',
                  model=op.models.physics.capillary_pressure._funcs.washburn,
                  surface_tension='throat.surface_tension',
                  contact_angle='throat.contact_angle',
                  diameter='throat.diameter')
phase_f.add_model(propname='throat.entry_pressure',
                  model=op.models.physics.capillary_pressure._funcs.washburn,
                  surface_tension='throat.surface_tension',
                  contact_angle='throat.contact_angle',
                  diameter='throat.diameter')

# create drainage objects
drn_s = op.algorithms.Drainage(network=pn_s, phase=phase_s)
drn_f = op.algorithms.Drainage(network=pn_f, phase=phase_f)

# set BCs
inlet_s = pn_s.pores(['xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax'])
inlet_f = pn_f.pores(['xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax'])
drn_s.set_inlet_BC(pores=inlet_s)
drn_f.set_inlet_BC(pores=inlet_f)

# run drainage
pressures=np.arange(0, 20, 0.1)
drn_s.run(pressures=pressures)
drn_f.run(pressures=pressures)

# save sampled network
pn_s['throat.radius'] = pn_s['throat.diameter']/2
pn_s['pore.invasion_sequence'] = drn_s['pore.invasion_sequence']
pn_s['throat.invasion_sequence'] = drn_s['throat.invasion_sequence']
op.io.project_to_vtk(project=pn_s.project, filename='../paraview/network-' + image + '-sampled')

# save fitted network
pn_f['throat.radius'] = pn_f['throat.diameter']/2
pn_f['pore.invasion_sequence'] = drn_f['pore.invasion_sequence']
pn_f['throat.invasion_sequence'] = drn_f['throat.invasion_sequence']
op.io.project_to_vtk(project=pn_f.project, filename='../paraview/network-' + image + '-fitted')