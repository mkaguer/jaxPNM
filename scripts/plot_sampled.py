import numpy as np
import jax.numpy as jnp
import jax.experimental.sparse as js
from _fit_cubic_network import FitCubicNetwork
import os
from jax import config
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import mypnmlib as pnm
from scipy.stats import gaussian_kde
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.spatial import cKDTree
import jax
import pandas as pd

np.random.seed(0)

os.environ["JAX_PLATFORMS"] = "cpu"
config.update("jax_enable_x64", False)

image = "Berea"
bw_method = 0.01

# set properties
sigma = 0.4791
theta = 140

# create sampled JAX network
shape = [15, 15, 15] # [20, 15, 15]
net_s = pnm.network.make_cubic_network(shape=shape, spacing=1)

# load fitted network
net_f = np.load('../networks/fitted-' + image + '.npz')
net_f = {key: jnp.array(net_f[key]) for key in net_f.files}

# get Np and Nt
Np_s = len(net_s['pore.coords'])
Nt_s = len(net_s['throat.conns'])
Np_f = len(net_f['pore.coords'])
Nt_f = len(net_f['throat.conns'])

# add adjacency network to net_s
weights = jnp.arange(1, Nt_s+1)
am = pnm.network.create_adjacency_matrix(net_s, weights=weights, fmt='csr')
net_s['adjacency_matrix'] = am

# assign boundary pores to sampled network
net_s['pore.boundary'] = (
    net_s['pore.left'] +
    net_s['pore.right'] +
    net_s['pore.back'] +
    net_s['pore.front'] +
    net_s['pore.top'] +
    net_s['pore.bottom']
)

# reconstruct adjacency matrix
data = net_f['am.data']
indices = net_f['am.indices']
indptr = net_f['am.indptr']
net_f['adjacency_matrix'] = js.BCSR((data, indices, indptr), shape=(Np_f, Np_f))

# load porosimetry data
data = np.loadtxt('../data/porosimetry-' + image + '.csv', delimiter=',')
mask = ~np.isinf(data[:, 0])
sat_target = jnp.array(data[:, 1][mask]).astype(jnp.float32)
x_target = jnp.array(data[:, 0][mask]).astype(jnp.float32)

# calculate spacing
Dp = -4*sigma*jnp.cos(theta*jnp.pi/180)/x_target
spacing = Dp[0]

# add "constant" Gh properties to sampled network
net_s['pore.viscosity'] = jnp.ones(Np_s) * 1e-3
net_s['throat.viscosity'] = jnp.ones(Nt_s) * 1e-3

# set BCs in x direction
pores = jnp.where(net_s['pore.left'])[0]
pnm.simulations.set_BC(net_s,
                       pores=pores,
                       bctype='value',
                       bcvalues=1.0,
                       mode='overwrite')
pores = jnp.where(net_s['pore.right'])[0]
pnm.simulations.set_BC(net_s,
                       pores=pores,
                       bctype='value',
                       bcvalues=0.0,
                       mode='add')
net_s['pore.bc.valuex'] = net_s['pore.bc.value']
net_s['pore.bc.maskx'] = net_s['pore.bc.mask']
net_s['boundary_poresx'] = net_s['boundary_pores']
net_s['rate_poresx'] = pores
del net_s['pore.bc.value'], net_s['pore.bc.mask'], net_s['boundary_pores']

# set BCs in y direction
pores = jnp.where(net_s['pore.front'])[0]
pnm.simulations.set_BC(net_s,
                       pores=pores,
                       bctype='value',
                       bcvalues=1.0,
                       mode='overwrite')
pores = jnp.where(net_s['pore.back'])[0]
pnm.simulations.set_BC(net_s,
                       pores=pores,
                       bctype='value',
                       bcvalues=0.0,
                       mode='add')
net_s['pore.bc.valuey'] = net_s['pore.bc.value']
net_s['pore.bc.masky'] = net_s['pore.bc.mask']
net_s['boundary_poresy'] = net_s['boundary_pores']
net_s['rate_poresy'] = pores
del net_s['pore.bc.value'], net_s['pore.bc.mask'], net_s['boundary_pores']

# set BCs in z direction
pores = jnp.where(net_s['pore.bottom'])[0]
pnm.simulations.set_BC(net_s,
                       pores=pores,
                       bctype='value',
                       bcvalues=1.0,
                       mode='overwrite')
pores = jnp.where(net_s['pore.top'])[0]
pnm.simulations.set_BC(net_s,
                       pores=pores,
                       bctype='value',
                       bcvalues=0.0,
                       mode='add')
net_s['pore.bc.valuez'] = net_s['pore.bc.value']
net_s['pore.bc.maskz'] = net_s['pore.bc.mask']
net_s['boundary_poresz'] = net_s['boundary_pores']
net_s['rate_poresz'] = pores
del net_s['pore.bc.value'], net_s['pore.bc.mask'], net_s['boundary_pores']

# create fcn objects
pressures = jnp.arange(jnp.min(x_target)*0.9, jnp.max(x_target)*1.1, 1e3)
fcn_s = FitCubicNetwork(net_s,
                        surface_tension=sigma,
                        contact_angle=theta,
                        sat_target=sat_target,
                        x_target=x_target,
                        pressure=pressures,
                        spacing=1,
                        smoothing_factor=0.4)
fcn_f = FitCubicNetwork(net_f,
                        surface_tension=sigma,
                        contact_angle=theta,
                        sat_target=sat_target,
                        x_target=x_target,
                        pressure=pressures,
                        spacing=1,
                        smoothing_factor=0.4)

# pre-process pressures (b/c D is btwn 0 and 1, spacing set as 1)
fcn_s.process_pressure(spacing=spacing, mode='pre')
fcn_f.process_pressure(spacing=spacing, mode='pre')

# retrieve fitted diameters and tsf
Dp_fitted = net_f['pore.diameter']
tsf_fitted = net_f['throat.tsf']
conns = net_f['throat.conns']
Dt_fitted = jnp.min(Dp_fitted[conns], axis=1) * tsf_fitted

# Now, do sampling to get D_sampled...
# get copy of coords and scale from 0 to 1
coords_f = net_f['pore.coords'].copy()
coords_s = net_s['pore.coords'].copy()

def scale_coords(coords):

    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    x_scaled = (x - jnp.min(x))/(jnp.max(x) - jnp.min(x))
    y_scaled = (y - jnp.min(y))/(jnp.max(y) - jnp.min(y))
    z_scaled = (z - jnp.min(z))/(jnp.max(z) - jnp.min(z))

    return jnp.vstack((x_scaled, y_scaled, z_scaled)).T

coords_f = scale_coords(coords_f)
coords_s = scale_coords(coords_s)

# get throat coords
conns_f = net_f['throat.conns'].copy()
conns_s = net_s['throat.conns'].copy()
throat_coords_f = jnp.sum(coords_f[conns_f], axis=1)/2
throat_coords_s = jnp.sum(coords_s[conns_s], axis=1)/2

# fit gaussian kde to Dp and sample
kde = gaussian_kde(Dp_fitted, bw_method=bw_method)
Dp_kde = kde.resample(Np_s)[0]  # this is a numpy array!

# fit GP to Dp and coords_f
kernel = C(1.0) * RBF(length_scale=0.05)
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0)
gp.fit(coords_f, Dp_fitted)
print('Finished GP Dp fit')

# sample from GP
tree = cKDTree(coords_f)
_, indices = tree.query(coords_s)
coords_s = coords_f[indices]
Dp_gp, _ = gp.predict(coords_s, return_std=True)  # this is a numpy array

# sort to preserve spatial trends
indices = np.argsort(Dp_gp)
values = np.sort(Dp_kde)
# get sampled Ds
Dp_sampled = np.zeros_like(Dp_gp)
Dp_sampled[indices] = values
Dp_sampled = np.clip(Dp_sampled, 1e-2, 0.99)
Dp_sampled = jnp.array(Dp_sampled)  # convert back to jax array

# fit gaussian kde to Dt and sample
kde = gaussian_kde(tsf_fitted, bw_method=bw_method)
tsf_kde = kde.resample(Nt_s)[0]  # this is a numpy array

# fit GP to tsf and sample
kernel = C(1.0) * RBF(length_scale=0.05)
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0)
X_train = np.hstack([coords_f[conns_f[:, 0]], coords_f[conns_f[:, 1]]]) 
gp.fit(X_train, tsf_fitted)
print('Finished GP Dt fit')

X_pred = np.hstack([coords_s[conns_s[:, 0]], coords_s[conns_s[:, 1]]]) 
tsf_gp, _ = gp.predict(X_pred, return_std=True)

# sort to preserve spatial trends
indices = np.argsort(tsf_gp)
values = np.sort(tsf_kde)
# get sampled Ds
tsf_sampled = np.zeros_like(tsf_gp)
tsf_sampled[indices] = values
tsf_sampled = np.clip(tsf_sampled, 1e-2, 0.95)  # FIXME: should this be 0.95?
tsf_sampled = jnp.array(tsf_sampled)  # convert back to jax array

# sample from GP
Dt_sampled = tsf_sampled * jnp.min(Dp_sampled[conns_s], axis=1)

# get sampled diameters without GP
Dp_sampled_n = jnp.clip(Dp_kde, 1e-2, 0.99)
tsf_sampled_n = jnp.clip(tsf_kde, 1e-2, 0.95)
Dt_sampled_n = tsf_sampled_n * jnp.min(Dp_sampled_n[conns_s], axis=1)

# create copy of network BEFORE jitting
net_copy = fcn_s.network.copy() 

# calculate saturations
jitted_run_invasion_s = jax.jit(fcn_s.run_invasion)
jitted_run_invasion_f = jax.jit(fcn_f.run_invasion)
sat_s = jitted_run_invasion_s(Dp_sampled, tsf_sampled)
sat_n = jitted_run_invasion_s(Dp_sampled_n, tsf_sampled_n)
sat_f = jitted_run_invasion_f(Dp_fitted, tsf_fitted)

# calculate permeabilities for fitted
px = fcn_f.flow(Dp_fitted, tsf_fitted, axis='x')
Kx = fcn_f.calc_K(px, axis='x')
py = fcn_f.flow(Dp_fitted, tsf_fitted, axis='y')
Ky = fcn_f.calc_K(py, axis='y')
pz = fcn_f.flow(Dp_fitted, tsf_fitted, axis='z')
Kz = fcn_f.calc_K(pz, axis='z')
K_avg = jnp.average(jnp.array([Kx, Ky, Kz]))
K_fitted = jnp.array([Kx, Ky, Kz, K_avg])*spacing**2/ 0.98e-12 * 1000

# calculate permeabilities for GP
px = fcn_s.flow(Dp_sampled, tsf_sampled, axis='x')
Kx = fcn_s.calc_K(px, axis='x')
py = fcn_s.flow(Dp_sampled, tsf_sampled, axis='y')
Ky = fcn_s.calc_K(py, axis='y')
pz = fcn_s.flow(Dp_sampled, tsf_sampled, axis='z')
Kz = fcn_s.calc_K(pz, axis='z')
K_avg = jnp.average(jnp.array([Kx, Ky, Kz]))
K_sampled = jnp.array([Kx, Ky, Kz, K_avg])*spacing**2/ 0.98e-12 * 1000

# calculate permeabilities for no GP
px = fcn_s.flow(Dp_sampled_n, tsf_sampled_n, axis='x')
Kx = fcn_s.calc_K(px, axis='x')
py = fcn_s.flow(Dp_sampled_n, tsf_sampled_n, axis='y')
Ky = fcn_s.calc_K(py, axis='y')
pz = fcn_s.flow(Dp_sampled_n, tsf_sampled_n, axis='z')
Kz = fcn_s.calc_K(pz, axis='z')
K_avg = jnp.average(jnp.array([Kx, Ky, Kz]))
K_sampled_n = jnp.array([Kx, Ky, Kz, K_avg])*spacing**2/ 0.98e-12 * 1000

# load permeability data
data = pd.read_csv('../data/K-' + image + '.csv', header=None).values.flatten()
K_target = jnp.array(data).astype(jnp.float32)
K_target = jnp.concatenate((K_target, jnp.array([jnp.average(K_target)])))

# Print permeability
print(f'Target permeability: {K_target} mD')
print(f'Final permeability: {K_sampled} mD')

plt.hist(jnp.log10(net_s['throat.conductance']), bins=50, density=True, alpha=0.5)
plt.hist(jnp.log10(net_f['throat.conductance']), bins=50, density=True, alpha=0.5)
plt.show()

# interpolate
sat_s = jnp.interp(x_target, pressures, sat_s)
sat_n = jnp.interp(x_target, pressures, sat_n)
sat_f = jnp.interp(x_target, pressures, sat_f)

# plot psd distribution
plt.figure(1, dpi=500)
ax = plt.gca()  # Get current axes
# Make the bounding box bold
for spine in ax.spines.values():
    spine.set_linewidth(3)
ax.tick_params(direction='in', length=6, width=3)
bins = np.linspace(0, max(Dp_sampled.max(), Dp_fitted.max()), 50) * spacing * 1e6
plt.hist(Dp_fitted * spacing * 1e6, bins=bins, alpha=0.5, density=True, color='tab:purple', label='JAX')
plt.hist(Dp_sampled * spacing * 1e6, bins=bins, alpha=0.5, density=True, color='tab:orange',  label='Sampled')
plt.title(image + ' PSD', fontweight='semibold', fontsize=18)
plt.xlabel('Pore Diameter (\u03BCm)', fontsize=18)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(fontsize=18, frameon=True)
plt.savefig('../figures/sampled-psd-' + image + f'-{shape}' + '.png')
plt.show()

# plot tsd distribution
bins = np.arange(0, 0.5, 0.02)
plt.figure(2, dpi=500)
ax = plt.gca()  # Get current axes
# Make the bounding box bold
for spine in ax.spines.values():
    spine.set_linewidth(3)
ax.tick_params(direction='in', length=6, width=3)
bins = np.linspace(0, max(Dt_sampled.max(), Dt_fitted.max()), 50) * spacing * 1e6
plt.hist(Dt_fitted * spacing * 1e6, bins=bins, alpha=0.5, density=True, color='tab:purple', label='JAX')
plt.hist(Dt_sampled * spacing * 1e6, bins=bins, alpha=0.5, density=True, color='tab:orange', label='Sampled')
plt.title(image + ' TSD', fontweight='semibold', fontsize=18)
plt.xlabel('Throat Diameter (\u03BCm)', fontsize=18)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(fontsize=18, frameon=True)
plt.savefig('../figures/sampled-tsd-' + image + f'-{shape}' + '.png')
plt.show()

# plot pc results
plt.figure(3, dpi=500)
ax = plt.gca()  # Get current axes
# Make the bounding box bold
for spine in ax.spines.values():
    spine.set_linewidth(3)
ax.tick_params(direction='in', length=6, width=3)
# Set x-axis to scientific notation
ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
ax.xaxis.get_offset_text().set_fontsize(18)  # Adjust offset text size
ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))  # Force scientific notation 
plt.plot(x_target, sat_s, label='GP', color='tab:orange', marker='o', markerfacecolor='none', markersize=9, linewidth=2.5)
plt.plot(x_target, sat_n, label='No GP', color='tab:pink', marker='o', markerfacecolor='none', markersize=9, linewidth=2.5)
plt.plot(x_target, sat_f, label='JAX', color='tab:purple', linestyle='--', marker='o', markerfacecolor='none', markersize=9, linewidth=2.5)
plt.plot(x_target, sat_target, label='Target', color='tab:green', linestyle='--', marker='^', markerfacecolor='none', markersize=9, linewidth=2.5)
plt.xlabel('Pressure (Pa)', fontsize=18)
plt.ylabel('Saturation', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.title(image, fontsize=18, fontweight='semibold')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(loc='best', fontsize=18, frameon=True)
plt.tight_layout()
plt.savefig('../figures/sampled-porosimetrty-' + image + f'-{shape}' + '.png')
plt.show()

# plot K results
plt.figure(4, dpi=500)
ax = plt.gca()  # Get current axes
# Make the bounding box bold
for spine in ax.spines.values():
    spine.set_linewidth(3)
ax.tick_params(direction='in', length=6, width=3) 
x = jnp.arange(len(K_fitted))
bar_width = 0.2
plt.bar(x, K_target, width=bar_width, label='Target', color='tab:green')
plt.bar(x + bar_width, K_fitted, width=bar_width, label='JAX', color='tab:purple')
plt.bar(x + 2 * bar_width, K_sampled, width=bar_width, label='GP', color='tab:orange')
plt.bar(x + 3 * bar_width, K_sampled_n, width=bar_width, label='No GP', color='tab:pink')
plt.ylabel('Permeability (mD)', fontsize=18, fontweight='normal')
plt.title(image, fontsize=18, fontweight='semibold')
plt.xticks(x + bar_width*3/2, ['X', 'Y', 'Z', 'Avg'], fontsize=18, fontweight='normal')
plt.yticks(fontsize=18, fontweight='normal')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.07), ncol=4, fontsize=14, frameon=True)
plt.tight_layout()
plt.savefig('../figures/sampled-permeability-' + image + f'-{shape}' + '.png')
plt.show()

# save sampled network
net_s = net_copy
net_s['am.data'] = net_s['adjacency_matrix'].data
net_s['am.indices'] = net_s['adjacency_matrix'].indices
net_s['am.indptr'] = net_s['adjacency_matrix'].indptr
del net_s['adjacency_matrix']

dic = {}
for key in net_s.keys():
    dic[key] = np.array(net_s[key])
dic['pore.diameter'] = Dp_sampled
dic['throat.tsf'] = tsf_sampled

np.savez_compressed("../networks/sampled-" + image + f'-{shape}' + ".npz", **dic)
