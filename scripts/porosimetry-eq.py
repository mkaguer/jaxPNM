"""
This script uses the equivalent diameter to get porosimetry data

Created by: Mike McKague
Date: August 16, 2025
"""
import openpnm as op
import porespy as ps
import numpy as np
import matplotlib.pyplot as plt

ps.visualization.set_mpl_style()
np.random.seed(10)

# geometry models
pore_volume = op.models.geometry.pore_volume.sphere  # cones and clyinders collection uses sphere
pore_volume_effective = op.models.geometry.pore_volume.effective
throat_volume = op.models.geometry.throat_volume.cylinder
throat_length = op.models.geometry.throat_length.cones_and_cylinders

# import network
name = 'A1'
net = op.io.network_from_csv('../networks/' + name + '-snow.gpickle')
net['throat.diameter'] = net['throat.equivalent_diameter']  # FIXME: change to equivalent!
net['pore.diameter'] = net['pore.equivalent_diameter']
for key in net.keys():
    if net[key].dtype == "O":
        net[key] = net[key].astype(bool)

net.add_model(propname='throat.length', model=throat_length)
net.add_model(propname='throat.volume', model=throat_volume)
net.add_model(propname='pore.volume', model=pore_volume)
net.add_model(propname='pore.volume_effective', model=pore_volume_effective)
net.regenerate_models()

h = op.utils.check_network_health(net)
print(h)

hg = op.phase.Mercury(network=net, name='mercury')
phys = op.models.collections.physics.basic.copy()
hg.add_model_collection(phys)
hg.regenerate_models()

inlet = net.pores(['xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax'])
mip = op.algorithms.Drainage(network=net, phase=hg)
mip.settings['pore_volume'] = 'pore.volume'
mip.set_inlet_BC(pores=inlet)
mip.run()

# get pc vs sat data
pc = mip.pc_curve()

# make plot
plt.figure(1)
ax = plt.gca()  # Get current axes
# Make the bounding box bold
for spine in ax.spines.values():
    spine.set_linewidth(3)
ax.tick_params(direction='in', length=6, width=3) 
plt.semilogx(*pc, 'k-o', label=name, linewidth=4, markersize=12)
plt.legend(fontsize=18)
plt.yticks(fontsize=18, fontweight='normal')
plt.xticks(fontsize=18, fontweight='normal')
# plt.xlabel('Capillary Pressure (Pa)', fontsize=18)
plt.title('Saturation vs. Pressure (Pa)', fontsize=24, fontweight='semibold')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# save
plt.savefig('../figures/porosimetry-eq-'+name, dpi=500)

# save data
pcs = np.array([pc[0]])
sat = np.array([pc[1]])
data = np.concatenate((pcs, sat), axis=0).T
np.savetxt('../data/porosimetry-eq-' + name + '.csv', data, delimiter=',')
