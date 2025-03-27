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

np.random.seed(0)

os.environ["JAX_PLATFORMS"] = "cpu"
config.update("jax_enable_x64", False)

image = "Berea"
bw_method = 0.01

# set properties
sigma = 0.4791
theta = 140

# create sampled JAX network
shape = [20, 15, 15]
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

# fit GP to D and coords_f
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
kde = gaussian_kde(Dt_fitted, bw_method=bw_method)
Dt_kde = kde.resample(Nt_s)[0]  # this is a numpy array

# fit GP to Dt and throat_coords_f
kernel = C(1.0) * RBF(length_scale=0.05)
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0)
gp.fit(throat_coords_f, Dt_fitted)
print('Finished GP Dt fit')

# sample from GP
tree = cKDTree(throat_coords_f)
_, indices = tree.query(throat_coords_s)
throat_coords_s = throat_coords_f[indices]
Dt_gp, _ = gp.predict(throat_coords_s, return_std=True) # this is a numpy array

# sort to preserve spatial trends
indices = np.argsort(Dt_gp)
values = np.sort(Dt_kde)
# get sampled Ds
Dt_sampled = np.zeros_like(Dt_gp)
Dt_sampled[indices] = values
Dt_sampled = np.clip(Dt_sampled, 1e-2, 0.99)
Dt_sampled = jnp.array(Dt_sampled)  # convert back to jax array

# ensure that Dt is not greater than Dp
max_throat_size = jnp.min(Dp_sampled[conns_s], axis=1)
mask = Dt_sampled >= max_throat_size
Dt_sampled = jnp.where(mask, max_throat_size * 0.99, Dt_sampled)

# convert Dt_sampled to tsf_sampled
tsf_sampled = Dt_sampled/max_throat_size

# get sampled diameters without GP
Dp_sampled_n = jnp.clip(Dp_kde, 1e-2, 0.99)
Dt_sampled_n = jnp.clip(Dt_kde, 1e-2, 0.99)
max_throat_size = jnp.min(Dp_sampled_n[conns_s], axis=1)
mask = Dt_sampled_n >= max_throat_size
Dt_sampled_n = jnp.where(mask, max_throat_size * 0.99, Dt_sampled_n)
tsf_sampled_n = Dt_sampled_n/max_throat_size

# create copy of network BEFORE jitting
net_copy = fcn_s.network.copy() 

# calculate saturations
jitted_run_invasion_s = jax.jit(fcn_s.run_invasion)
jitted_run_invasion_f = jax.jit(fcn_f.run_invasion)
sat_s = jitted_run_invasion_s(Dp_sampled, tsf_sampled)
sat_n = jitted_run_invasion_s(Dp_sampled_n, tsf_sampled_n)
sat_f = jitted_run_invasion_f(Dp_fitted, tsf_fitted)

# interpolate
sat_s = jnp.interp(x_target, pressures, sat_s)
sat_n = jnp.interp(x_target, pressures, sat_n)
sat_f = jnp.interp(x_target, pressures, sat_f)

# plot psd distribution
plt.figure(1, dpi=500)
ax = plt.gca()  # Get current axes
# Make the bounding box bold
for spine in ax.spines.values():
    spine.set_linewidth(2)
ax.tick_params(direction='in', length=5, width=2)
bins = np.linspace(0, max(Dp_sampled.max(), Dp_fitted.max()), 50) * spacing * 1e6
plt.hist(Dp_fitted * spacing * 1e6, bins=bins, alpha=0.5, density=True, color='tab:purple', label='JAX')
plt.hist(Dp_sampled * spacing * 1e6, bins=bins, alpha=0.5, density=True, color='tab:orange',  label='Sampled')
plt.title(image + ' PSD', fontweight='semibold', fontsize=16)
plt.xlabel('Pore Diameter (\u03BCm)', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.savefig('../figures/sampled-psd-' + image + f'-{bw_method}' + '.png')
plt.show()

# plot tsd distribution
bins = np.arange(0, 0.5, 0.02)
plt.figure(2, dpi=500)
ax = plt.gca()  # Get current axes
# Make the bounding box bold
for spine in ax.spines.values():
    spine.set_linewidth(2)
ax.tick_params(direction='in', length=5, width=2)
bins = np.linspace(0, max(Dt_sampled.max(), Dt_fitted.max()), 50) * spacing * 1e6
plt.hist(Dt_fitted * spacing * 1e6, bins=bins, alpha=0.5, density=True, color='tab:purple', label='JAX')
plt.hist(Dt_sampled * spacing * 1e6, bins=bins, alpha=0.5, density=True, color='tab:orange', label='Sampled')
plt.title(image + ' TSD', fontweight='semibold', fontsize=16)
plt.xlabel('Throat Diameter (\u03BCm)', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.savefig('../figures/sampled-tsd-' + image + f'-{bw_method}' + '.png')
plt.show()

# plot pc results
plt.figure(3, dpi=500)
ax = plt.gca()  # Get current axes
# Make the bounding box bold
for spine in ax.spines.values():
    spine.set_linewidth(2)
ax.tick_params(direction='in', length=5, width=2)
# Set x-axis to scientific notation
ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
ax.xaxis.get_offset_text().set_fontsize(14)  # Adjust offset text size
ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))  # Force scientific notation 
plt.plot(x_target, sat_s, label='GP', color='tab:orange', marker='o', markerfacecolor='none', markersize=9, linewidth=2.5)
plt.plot(x_target, sat_n, label='No GP', color='tab:pink', marker='o', markerfacecolor='none', markersize=9, linewidth=2.5)
plt.plot(x_target, sat_f, label='JAX', color='tab:purple', linestyle='--', marker='o', markerfacecolor='none', markersize=9, linewidth=2.5)
plt.plot(x_target, sat_target, label='Target', color='tab:green', linestyle='--', marker='^', markerfacecolor='none', markersize=9, linewidth=2.5)
plt.xlabel('Pressure (Pa)', fontsize=14)
plt.ylabel('Saturation', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title(image, fontsize=16, fontweight='semibold')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(loc='best', fontsize=12)
plt.tight_layout()
plt.savefig('../figures/sampled-porosimetrty-' + image + f'-{bw_method}' + '.png')
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

np.savez_compressed("../networks/sampled-" + image + ".npz", **dic)