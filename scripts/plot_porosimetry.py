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
plt.semilogx(pc, sw, 'k-o', label='Image-Based')
plt.legend(fontsize=16)
plt.title(name, fontsize=18)
plt.xlabel('Capillary Pressure (Pa)', fontsize=16)
plt.ylabel('Saturation', fontsize=16)
plt.savefig('../figures/porosimetry-' + name, dpi=500)
plt.show()