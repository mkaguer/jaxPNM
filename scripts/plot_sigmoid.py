import numpy as np
import matplotlib.pyplot as plt
import porespy as ps

ps.visualization.set_mpl_style()

# Define the sigmoid function with a smoothing factor
def sigmoid(x, k=1):
    return 1 / (1 + np.exp(-x/k))

# Generate x values
x = np.linspace(-10, 10, 400)

# Define different smoothing factors
smoothing_factors = [1, 2, 3]

# Create the plot
plt.figure(1, dpi=500, figsize=(8, 6))
ax = plt.gca()  # Get current axes
# Make the bounding box bold
for spine in ax.spines.values():
    spine.set_linewidth(4)
ax.tick_params(direction='in', length=7, width=4)
# Plot sigmoid curves for different smoothing factors
colors = ['tab:blue', 'tab:purple', 'tab:cyan']
for i, k in enumerate(smoothing_factors):
    plt.plot(x, sigmoid(x, k), label=f'Smoothing Factor={k}', color=colors[i], linewidth=4)
# Add labels and legend
# plt.ylabel('sigmoid(x)', fontsize=16)
# plt.xlabel('x', fontsize=18)
# plt.title('Sigmoid(x)', fontsize=20, fontweight='semibold')
plt.legend(fontsize=18, frameon=True)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlim([-10, 10])
plt.xticks(fontsize=20, fontweight='normal')
plt.yticks(fontsize=28, fontweight='normal')
# save figure
plt.savefig('../figures/sigmoid.png')
# Show the plot
plt.show()

