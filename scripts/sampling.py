import openpnm as op
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.spatial import cKDTree
import time

op.visualization.set_mpl_style()

np.random.seed(1)

# choose data
image = "Berea"

# load experimental data
data = np.loadtxt('../data/porosimetry-' + image + '.csv', delimiter=',')
mask = ~np.isinf(data[:, 0])
sat_target = np.array(data[:, 1][mask])
x_target = np.array(data[:, 0][mask])

# transform experimental data via spacing!
sigma = 0.4791  # N/m
theta = 140  # radians!
Dp = -4*sigma*np.cos(theta*np.pi/180)/x_target
spacing = Dp[0]
x_target = x_target * spacing

# load fitted diameters
net = np.load('../networks/fitted-' + image + '.npz')
net = {key: net[key] for key in net.files}
Dp = net['pore.diameter']
tsf = net['throat.tsf']
conns = net['throat.conns']
Dt = np.min(Dp[conns], axis=1) * tsf

# create networks, one 10^3 and another N^3 for sampling test
shape_f = [10, 10, 10]
shape_s = [20, 20, 20]
net_f = op.network.Cubic(shape=shape_f, spacing=1)
net_s = op.network.Cubic(shape=shape_s, spacing=1)

# get Np and Nt of sampled network
Np = len(net_s['pore.coords'])
Nt = len(net_s['throat.conns'])

# get copy of network coords and scale to 0 and 1
coords_f = net_f['pore.coords'].copy()
coords_s = net_s['pore.coords'].copy()


def scale_coords(coords):

    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    x_scaled = (x - np.min(x))/(np.max(x) - np.min(x))
    y_scaled = (y - np.min(y))/(np.max(y) - np.min(y))
    z_scaled = (z - np.min(z))/(np.max(z) - np.min(z))

    return np.vstack((x_scaled, y_scaled, z_scaled)).T


coords_f = scale_coords(coords_f)
coords_s = scale_coords(coords_s)

# get throat coords, already scaled!
conns_f = net_f['throat.conns'].copy()
conns_s = net_s['throat.conns'].copy()
throat_coords_f = np.sum(coords_f[conns_f], axis=1)/2
throat_coords_s = np.sum(coords_s[conns_s], axis=1)/2

# fit gaussian kde to D and take sample
kde = gaussian_kde(Dp, bw_method=0.01)
Dp_kde = kde.resample(Np)[0]

# fit GP to D and coords_f
kernel = C(1.0) * RBF(length_scale=0.05)
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0)
start = time.time()
gp.fit(coords_f, Dp)
stop = time.time()
print(f"Finished GP Dp fit in {stop - start}s")

# sample from GP
tree = cKDTree(coords_f)
_, indices = tree.query(coords_s)
coords_s = coords_f[indices]
Dp_gp, _ = gp.predict(coords_s, return_std=True)

# sort to preserve spatial trends
indices = np.argsort(Dp_gp)
values = np.sort(Dp_kde)
# get sampled Ds
Dp_sampled = np.zeros_like(Dp_gp)
Dp_sampled[indices] = values
Dp_sampled = np.clip(Dp_sampled, 1e-2, 1.0)

# get sampled tsf
kde = gaussian_kde(tsf, bw_method=0.01)
tsf_kde = kde.resample(Nt)[0]

kernel = C(1.0) * RBF(length_scale=0.05)
X_train = np.hstack([coords_f[conns_f[:, 0]], coords_f[conns_f[:, 1]]]) 
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0)
start = time.time()
gp.fit(X_train, tsf)
stop = time.time()
print(f"Finished GP tsf fit in {stop - start}s")

X_pred = np.hstack([coords_s[conns_s[:, 0]], coords_s[conns_s[:, 1]]]) 
start = time.time()
tsf_gp, _ = gp.predict(X_pred, return_std=True)
stop = time.time()
print(f"Time to predict GP throat size: {stop - start}s")

# sort to preserve spatial trends
indices = np.argsort(tsf_gp)
values = np.sort(tsf_kde)
# get sampled Ds
tsf_sampled = np.zeros_like(tsf_gp)
tsf_sampled[indices] = values
tsf_sampled = np.clip(tsf_sampled, 1e-2, 1.0)

# sample from GP
Dt_sampled = tsf_sampled * np.min(Dp_sampled[conns_s], axis=1)

# add models to fitted network
net_f['pore.diameter'] = Dp
net_f['throat.diameter'] = Dt
net_f.add_model(propname='throat.length',
                model=op.models.geometry.throat_length._funcs.spheres_and_cylinders,
                pore_diameter='pore.diameter',
                throat_diameter='throat.diameter')
net_f.add_model(propname='throat.total_volume',
                model=op.models.geometry.throat_volume._funcs.cylinder,
                throat_diameter='throat.diameter',
                throat_length='throat.length')
net_f.add_model(propname='throat.lens_volume',
                model=op.models.geometry.throat_volume._funcs.lens,
                throat_diameter='throat.diameter',
                pore_diameter='pore.diameter')
net_f.add_model(propname='throat.volume',
                model=op.models.misc._basic_math.difference,
                props=['throat.total_volume', 'throat.lens_volume'])
net_f.add_model(propname='pore.volume',
                model=op.models.geometry.pore_volume._funcs.sphere,
                pore_diameter='pore.diameter')
net_f.add_model(propname='throat.hydraulic_size_factors',
                model=op.models.geometry.hydraulic_size_factors._funcs.spheres_and_cylinders,
                pore_diameter='pore.diameter',
                throat_diameter='throat.diameter')

# add models to sampled network
net_s['pore.diameter'] = Dp_sampled
net_s['throat.diameter'] = Dt_sampled
net_s.add_model(propname='throat.length',
                model=op.models.geometry.throat_length._funcs.spheres_and_cylinders,
                pore_diameter='pore.diameter',
                throat_diameter='throat.diameter')
net_s.add_model(propname='throat.total_volume',
                model=op.models.geometry.throat_volume._funcs.cylinder,
                throat_diameter='throat.diameter',
                throat_length='throat.length')
net_s.add_model(propname='throat.lens_volume',
                model=op.models.geometry.throat_volume._funcs.lens,
                throat_diameter='throat.diameter',
                pore_diameter='pore.diameter')
net_s.add_model(propname='throat.volume',
                model=op.models.misc._basic_math.difference,
                props=['throat.total_volume', 'throat.lens_volume'])
net_s.add_model(propname='pore.volume',
                model=op.models.geometry.pore_volume._funcs.sphere,
                pore_diameter='pore.diameter')
net_s.add_model(propname='throat.hydraulic_size_factors',
                model=op.models.geometry.hydraulic_size_factors._funcs.spheres_and_cylinders,
                pore_diameter='pore.diameter',
                throat_diameter='throat.diameter')

# create phase objects for both networks
phase_s = op.phase.Mercury(network=net_s)
phase_f = op.phase.Mercury(network=net_f)

# add moels to sampled phase
phase_s['pore.viscosity'] = 1e-3  # FIXME: is this right?
phase_s['throat.viscosity'] = 1e-3
phase_s.add_model(propname='throat.entry_pressure',
                  model=op.models.physics.capillary_pressure._funcs.washburn,
                  surface_tension='throat.surface_tension',
                  contact_angle='throat.contact_angle',
                  diameter='throat.diameter')
phase_s.add_model(propname='throat.hydraulic_conductance',
                  model=op.models.physics.hydraulic_conductance._funcs.generic_hydraulic,
                  throat_viscosity='throat.viscosity',
                  size_factors='throat.hydraulic_size_factors')

# add moels to fitted phase
phase_f['pore.viscosity'] = 1e-3
phase_f['throat.viscosity'] = 1e-3
phase_f.add_model(propname='throat.entry_pressure',
                  model=op.models.physics.capillary_pressure._funcs.washburn,
                  surface_tension='throat.surface_tension',
                  contact_angle='throat.contact_angle',
                  diameter='throat.diameter')
phase_f.add_model(propname='throat.hydraulic_conductance',
                  model=op.models.physics.hydraulic_conductance._funcs.generic_hydraulic,
                  throat_viscosity='throat.viscosity',
                  size_factors='throat.hydraulic_size_factors')

# create drainage objects
drn_s = op.algorithms.Drainage(network=net_s, phase=phase_s)
drn_f = op.algorithms.Drainage(network=net_f, phase=phase_f)

# set BCs
inlet_s = net_s.pores(['xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax'])
inlet_f = net_f.pores(['xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax'])
drn_s.set_inlet_BC(pores=inlet_s)
drn_f.set_inlet_BC(pores=inlet_f)

# run drainage
pressures=np.arange(0, 20, 0.1)
drn_s.run(pressures=pressures)
drn_f.run(pressures=pressures)

# get results
pc_s, sat_s = drn_s.pc_curve()
pc_f, sat_f = drn_f.pc_curve()

# create flow objects
flow_s = op.algorithms.StokesFlow(network=net_s, phase=phase_s)
flow_f = op.algorithms.StokesFlow(network=net_f, phase=phase_f)

# set BCs
flow_s.set_value_BC(pores=net_s.pores('xmin'), values=1.0)
flow_s.set_value_BC(pores=net_s.pores('xmax'), values=0.0)
flow_f.set_value_BC(pores=net_f.pores('xmin'), values=1.0)
flow_f.set_value_BC(pores=net_f.pores('xmax'), values=0.0)

# run
flow_s.run()
flow_f.run()

# updated phase objects with results
phase_s['pore.pressure'] = flow_s.x
phase_f['pore.pressure'] = flow_f.x

# calculate permeability for sampled network
mu = 1e-3
L_s = shape_s[0]
A_s = shape_s[1] * shape_s[2]
Q_s = flow_s.rate(pores=net_s.pores('xmin'), mode='group')[0]
K_s = Q_s * L_s * mu / (A_s * (1.0 - 0.0)) * spacing ** 2 / 0.98e-12 * 1000
print(f'K_sampled is: {K_s:.2f} mD')

# calculate permeability for fitted network
mu = 1e-3
L_f = shape_f[0]
A_f = shape_f[1] * shape_f[2]
Q_f = flow_f.rate(pores=net_f.pores('xmin'), mode='group')[0]
K_f = Q_f * L_f * mu / (A_f * (1.0 - 0.0)) * spacing ** 2 / 0.98e-12 * 1000
print(f'K_fitted is: {K_f:.2f} mD')

# calculate porosity
eps_s = (np.sum(net_s['pore.volume']) + np.sum(net_s['throat.volume']))/np.prod(shape_s)
eps_f = (np.sum(net_f['pore.volume']) + np.sum(net_f['throat.volume']))/np.prod(shape_f)
print(f'Sampled Porosity: {eps_s:.4f}')
print(f'Fitted Porosity: {eps_f:.4f}')

# plot results
plt.figure(1)
plt.plot(pc_f, sat_f, label=f'Fitted: {shape_f}')
plt.plot(pc_s, sat_s, label=f'Sampled: {shape_s}')
plt.xlabel('Pressure')
plt.ylabel('Saturation')
plt.legend()
plt.show()

# interpolate results
sat_s = np.interp(x_target, pc_s, sat_s)
sat_f = np.interp(x_target, pc_f, sat_f)

# plot results
plt.figure(2)
plt.plot(x_target, sat_f, label=f'Fitted: {shape_f}')
plt.plot(x_target, sat_s, label=f'Sampled: {shape_s}')
plt.plot(x_target, sat_target, label='Experimental Data')
plt.xlabel('Pressure')
plt.ylabel('Saturation')
plt.legend()
plt.show()

# plot psd distribution
plt.figure(3)
plt.hist(Dp, bins=30, alpha=0.5, density=True, label='Fitted')
plt.hist(Dp_sampled, bins=30, alpha=0.5, density=True, label='Sampled')
plt.title('PSD')
plt.legend()
plt.show()

# plot tsd distribution
bins = np.arange(0, 0.5, 0.02)
Dt_s = net_s['throat.diameter']
Dt_f = net_f['throat.diameter']
plt.figure(4)
plt.hist(Dt_f, bins=bins, alpha=0.5, density=True, label='Fitted')
plt.hist(Dt_s, bins=bins, alpha=0.5, density=True, label='Sampled')
plt.title('TSD')
plt.legend()
plt.show()

# plot Gh distribution
bins = np.arange(-4, 5, 0.1)
Gh_s = np.log10(phase_s['throat.hydraulic_conductance'])
Gh_f = np.log10(phase_f['throat.hydraulic_conductance'])
plt.figure(5)
plt.hist(Gh_f, bins=bins, alpha=0.5, density=True, label='Fitted')
plt.hist(Gh_s, bins=bins, alpha=0.5, density=True, label='Sampled')
plt.title('Gh distribution')
plt.legend()
plt.show()

plt.figure(6)
plt.hist(tsf_sampled, bins=50, alpha=0.5, density=True, label='Sampled')
plt.hist(tsf, bins=50, alpha=0.5, density=True, label='Fitted')
plt.title('tsf distribution')
plt.legend()
plt.show()

# export networks
net_f['throat.tsf'] = tsf
net_f['throat.radius'] = net_f['throat.diameter']/2
net_f['pore.invasion_sequence'] = drn_f['pore.invasion_sequence']
net_f['throat.invasion_sequence'] = drn_f['throat.invasion_sequence']
op.io.project_to_vtk(project=net_f.project, filename='../paraview/network-fitted')

net_s['throat.tsf'] = tsf_sampled
net_s['throat.radius'] = net_s['throat.diameter']/2
net_s['pore.invasion_sequence'] = drn_s['pore.invasion_sequence']
net_s['throat.invasion_sequence'] = drn_s['throat.invasion_sequence']
op.io.project_to_vtk(project=net_s.project, filename='../paraview/network-sampled')
