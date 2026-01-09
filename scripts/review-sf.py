import numpy as np
import openpnm as op
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

# load network
image = "Berea"
data = np.load('../networks/fitted-eq-BoT-Berea.npz')
data = {key: np.array(data[key]) for key in data.files}

# take out trained weights
D = data["pore.diameter"]
tsf = data["throat.tsf"]

# load experimental data
data = np.loadtxt('../data/porosimetry-eq-Berea.csv', delimiter=',')
mask = ~np.isinf(data[:, 0])
sat_target = np.array(data[:, 1][mask]).astype(np.float32)
x_target = np.array(data[:, 0][mask]).astype(np.float32)

# properties of mercurcy invasion
sigma = 0.4791  # N/m
theta = 140  # degrees

# get spacing
Dp = -4*sigma*np.cos(theta*np.pi/180)/x_target
spacing = Dp[0]

# create a new network in openpnm
shape = [10, 10, 10]
net = op.network.Cubic(shape=shape, spacing=spacing)

# get coords and conns
coords = net["pore.coords"]
conns = net["throat.conns"]

# FIXME: use fitted geometry
# add pore and throat diameters to network
net["pore.diameter"] = D * spacing
net["throat.diameter"] = tsf * np.min(np.abs(D[conns]), axis=1) * spacing

# add geometry models 
geo_mods = op.models.collections.geometry.spheres_and_cylinders.copy()
del geo_mods["pore.seed"], geo_mods["pore.max_size"], geo_mods["pore.diameter"]
del geo_mods["throat.max_size"], geo_mods["throat.diameter"]
net.add_model_collection(models=geo_mods)
net.regenerate_models()


# create phase object
phase = op.phase.Phase(network=net)
phase["throat.contact_angle"] = theta
phase["throat.surface_tension"] = sigma
# phase["throat.viscosity"] = 1e-3 

# add physics models
phys_mods = op.models.collections.physics.basic.copy()
del phys_mods["throat.diffusive_conductance"]
del phys_mods["throat.hydraulic_conductance"]
phase.add_model_collection(models=phys_mods)
phase.regenerate_models()

# run drainage
alg = op.algorithms.Drainage(phase=phase, network=net)
alg.set_inlet_BC(pores=net.pores("surface"))
alg.run()

# get pc curve data
data = alg.pc_curve()
sat_tra = data.snwp
pc_tra = data.pc

# add zero
sat_tra = np.concatenate((np.array([0]), sat_tra))
pc_tra = np.concatenate((np.array([1e4]), pc_tra))

# plt.figure(1, dpi=600)
# plt.plot(x_target, sat_target)
# plt.plot(pc_tra, sat_tra)
# plt.show()

# interpolate to get saturation with no sf
# sat_no_sf = np.interp(x_target, pc_tra, sat_tra)

# plot pc results
plt.figure(1, dpi=500)
ax = plt.gca()  # Get current axes
# Make the bounding box bold
for spine in ax.spines.values():
    spine.set_linewidth(3)
ax.tick_params(direction='in', length=6, width=3)
# Set x-axis to scientific notation
ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
ax.xaxis.get_offset_text().set_fontsize(18)  # Adjust offset text size
ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))  # Force scientific notation
plt.plot(pc_tra[0:-6], sat_tra[0:-6], label='PNM without Smoothing Factor', color='tab:red', marker='o', markerfacecolor='none', markersize=9, linewidth=2.5)
plt.plot(x_target, sat_target, label='Target', color='tab:green', linestyle='--', marker='^', markerfacecolor='none', markersize=9, linewidth=2.5)
plt.xlabel('Pressure (Pa)', fontsize=18)
plt.ylabel('Saturation', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.title(image, fontsize=18, fontweight='semibold')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(loc='best', fontsize=14, frameon=True)
plt.tight_layout()
plt.savefig('../figures/review-no-sf-Berea.png')
plt.show()
