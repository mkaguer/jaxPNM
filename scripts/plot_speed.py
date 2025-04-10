import numpy as np
import matplotlib.pyplot as plt
import porespy as ps

ps.visualization.set_mpl_style()

# import flow data
flow_gpu = np.load('../data/optimization-flow-speed-gpu.npy')
flow_cpu = np.load('../data/optimization-flow-speed-cpu.npy')
flow_iter_gpu = np.load('../data/optimization-flow-speed-iter-gpu.npy')
flow_iter_cpu = np.load('../data/optimization-flow-speed-iter-cpu.npy')

# import porosimetry data
inva_gpu = np.load('../data/optimization-invasion-speed-gpu.npy')
inva_cpu = np.load('../data/optimization-invasion-speed-cpu.npy')
inva_iter_gpu = np.load('../data/invasion-speed-iter-10by10by10-gpu.npy')
inva_iter_cpu10 = np.load('../data/invasion-speed-iter-10by10by10-cpu.npy')
inva_iter_cpu5 = np.load('../data/invasion-speed-iter-5by5by5-cpu.npy')

# plot
plt.figure(1, dpi=500)
fig, ax = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False, figsize=(12, 10))

# primary y-axis
ax[0, 0].loglog(np.logspace(1, 3, 3), inva_gpu[:, 0], label='GPU', marker='o', color='tab:red', markersize=11, linewidth=3.5)
ax[0, 0].loglog(np.logspace(1, 3, 3), inva_cpu[:, 0], label='CPU', marker='^', color='tab:brown', markersize=11, linewidth=3.5)
ax[1, 0].loglog(np.logspace(1, 5, 5), flow_gpu[:, 0], label='GPU', marker='o', color='tab:red', markersize=11, linewidth=3.5)
ax[1, 0].loglog(np.logspace(1, 5, 5), flow_cpu[:, 0], label='CPU', marker='^', color='tab:brown', markersize=11, linewidth=3.5)
ax[0, 1].loglog(np.logspace(1, 3, 3), inva_iter_gpu[:, 0], label='GPU 10^3', marker='o', color='tab:red', markersize=11, linewidth=3.5)
ax[0, 1].loglog(np.logspace(1, 3, 3), inva_iter_cpu10[:, 0], label='CPU 10^3', marker='^', color='tab:brown', markersize=11, linewidth=3.5)
ax[0, 1].loglog(np.logspace(1, 3, 3), inva_iter_cpu5[:, 0], label='CPU 5^3', linestyle='--', marker='o', color='tab:red', markersize=11, linewidth=3.5)
ax[1, 1].loglog(np.logspace(1, 4, 4), flow_iter_gpu[:, 0], label='GPU', marker='o', color='tab:red', markersize=11, linewidth=3.5)
ax[1, 1].loglog(np.logspace(1, 4, 4), flow_iter_cpu[:, 0], label='CPU', marker='^', color='tab:brown', markersize=11, linewidth=3.5)

# add labels
ax[0, 0].set_ylabel('Invasion Optimization Time (s)', fontweight='normal', fontsize=18)
ax[1, 0].set_xlabel('No. of Pores', fontweight='normal', fontsize=18)
ax[1, 0].set_ylabel('Flow Optimization Time (s)', fontweight='normal', fontsize=18)
ax[1, 1].set_xlabel('No. of Iterations', fontweight='normal', fontsize=18)

# add legends
ax[0, 0].legend(frameon=True, fontsize=16)
ax[1, 0].legend(frameon=True, fontsize=16)
ax[0, 1].legend(frameon=True, fontsize=16)
ax[1, 1].legend(frameon=True, fontsize=16)

# add titles
ax[0, 0].set_title('(a)', fontweight='semibold', fontsize=18)
ax[1, 0].set_title('(c)', fontweight='semibold', fontsize=18)
ax[0, 1].set_title('(b)', fontweight='semibold', fontsize=18)
ax[1, 1].set_title('(d)', fontweight='semibold', fontsize=18)

# add grid lines
ax[0, 0].grid(axis='y', linestyle='--', alpha=0.7)
ax[1, 0].grid(axis='y', linestyle='--', alpha=0.7)
ax[0, 1].grid(axis='y', linestyle='--', alpha=0.7)
ax[1, 1].grid(axis='y', linestyle='--', alpha=0.7)
ax[0, 0].grid(axis='x', linestyle='--', alpha=0.7)
ax[1, 0].grid(axis='x', linestyle='--', alpha=0.7)
ax[0, 1].grid(axis='x', linestyle='--', alpha=0.7)
ax[1, 1].grid(axis='x', linestyle='--', alpha=0.7)

# change tick params
ax[0, 0].tick_params(axis='x', labelsize=16, direction='in', length=6, width=3)
ax[0, 1].tick_params(axis='x', labelsize=16, direction='in', length=6, width=3)
ax[1, 1].tick_params(axis='x', labelsize=16, direction='in', length=6, width=3)
ax[1, 0].tick_params(axis='x', labelsize=16, direction='in', length=6, width=3)
ax[0, 0].tick_params(axis='y', labelsize=16, direction='in', length=6, width=3)
ax[0, 1].tick_params(axis='y', labelsize=16, direction='in', length=6, width=3)
ax[1, 1].tick_params(axis='y', labelsize=16, direction='in', length=6, width=3)
ax[1, 0].tick_params(axis='y', labelsize=16, direction='in', length=6, width=3)


# Make the bounding box bold
for spine in ax[0, 0].spines.values():
    spine.set_linewidth(3)
for spine in ax[0, 1].spines.values():
    spine.set_linewidth(3)
for spine in ax[1, 1].spines.values():
    spine.set_linewidth(3)
for spine in ax[1, 0].spines.values():
    spine.set_linewidth(3)
    
# save
plt.savefig('../figures/speed_plot.png')

# show
plt.show()