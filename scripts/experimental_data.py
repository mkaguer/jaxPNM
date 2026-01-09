import numpy as np
import matplotlib.pyplot as plt
# import porespy as ps

# ps.visualization.set_mpl_style()

# import data
data = np.loadtxt(fname="../data/churcher_fig15.csv", delimiter=",", skiprows=1)
data[:, 1] *= 1e3
data[:, 0] *= 1e-2
pc = data[2:16, 1]
sat = data[2:16, 0]

# scale sat data between 1 and 0
sat_max = np.max(sat)
sat = sat/sat_max

# convert from kPa to Pa
# pc *= 1e3

# plot data
plt.figure(1)
ax = plt.gca()  # Get current axes
# Make the bounding box bold
for spine in ax.spines.values():
    spine.set_linewidth(3)
ax.tick_params(direction='in', length=6, width=3) 
plt.semilogx(data[:, 1], data[:, 0], 'k-o', label="Churchel et al.", linewidth=4, markersize=12)
plt.semilogx(pc, sat, 'g-o', label="Target", linewidth=4, markersize=12, markerfacecolor="white")
plt.legend(fontsize=18, frameon=True)
plt.yticks(fontsize=18, fontweight='normal')
plt.xticks(fontsize=18, fontweight='normal')
plt.title('Experiment', fontsize=18, fontweight='semibold')
plt.xlabel('Pressure (Pa)', fontsize=18)
plt.ylabel('Saturation', fontsize=18)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('../figures/porosimetry-experiment-', dpi=500)
plt.show()

# save data
pc = np.array([pc])
sat = np.array([sat])
data = np.concatenate((pc, sat), axis=0).T
np.savetxt('../data/porosimetry-Experiment' + '.csv', data, delimiter=',')