import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

image = "C2"

# load porosimetry data
data = pd.read_csv('../data/K-' + image + '.csv', header=None).values.flatten()
K_target = data
K_target_avg = np.array([np.average(K_target)])
K_target = np.concatenate((np.array(K_target), K_target_avg))  # CHECK!!

# plot K results
plt.figure(2, dpi=500)
ax = plt.gca()  # Get current axes
# Make the bounding box bold
for spine in ax.spines.values():
    spine.set_linewidth(3)
ax.tick_params(direction='in', length=6, width=3) 
x = np.arange(len(K_target))
bar_width=0.5
plt.bar(x, K_target, width=bar_width, label=image, color='tab:blue')
# plt.ylabel('Permeability (mD)', fontsize=14, fontweight='normal')
plt.title('Permeability (mD)', fontsize=24, fontweight='semibold')
plt.xticks(x, ['X', 'Y', 'Z', 'Avg'], fontsize=18, fontweight='normal')
plt.yticks(fontsize=18, fontweight='normal')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(loc='best', fontsize=18, frameon=True)
plt.tight_layout()
plt.savefig('../figures/target-permeability-' + image + '.png')
plt.show()