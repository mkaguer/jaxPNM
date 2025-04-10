import porespy as ps
import numpy as np
import matplotlib.pyplot as plt

ps.visualization.set_mpl_style()
np.random.seed(10)

name = 'S9'
data = np.loadtxt('../data/porosimetry-' + name + '.csv', delimiter=",")

pc = data[:, 0]
sw = data[:, 1]

plt.figure(3)
ax = plt.gca()  # Get current axes
# Make the bounding box bold
for spine in ax.spines.values():
    spine.set_linewidth(3)
ax.tick_params(direction='in', length=6, width=3) 
plt.semilogx(pc, sw, 'k-o', label=name, linewidth=4, markersize=12)
plt.legend(fontsize=18)
plt.yticks(fontsize=18, fontweight='normal')
plt.xticks(fontsize=18, fontweight='normal')
# plt.xlabel('Capillary Pressure (Pa)', fontsize=18)
plt.title('Saturation vs. Pressure (Pa)', fontsize=24, fontweight='semibold')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('../figures/porosimetry-' + name, dpi=500)
plt.show()