import numpy as np
import matplotlib.pyplot as plt
import porespy as ps
import pandas as pd

ps.visualization.set_mpl_style()

# %% Plot porosimetry curves with line

name = "Berea"

# load image-based data
data = np.loadtxt('../data/porosimetry-' + name + '.csv', delimiter=",")
pc_im = data[:, 0]
sw_im = data[:, 1]

# get intercept with y=0 and y=1
m = (sw_im[3:] - sw_im[2:-1])/(pc_im[3:] - pc_im[2:-1])
idx = np.where(m == np.max(m))[0]
b = -m[idx] * pc_im[2:][idx] + sw_im[2:][idx]
x_im_at_0 = -b/(m[idx])  # when y = 0
x_im_at_1 = (1-b)/(m[idx])  # when y = 1
print(f"Inscribed Breakthrough Pressure: {x_im_at_0}")

# load equivalent diameter
data = np.loadtxt('../data/porosimetry-eq-' + name + '.csv', delimiter=",")
pc_eq = data[:, 0]
sw_eq = data[:, 1]

# get intercept with y=0 and y=1
m = (sw_eq[1:] - sw_eq[0:-1])/(pc_eq[1:] - pc_eq[0:-1])
idx = np.where(m == np.max(m))[0]
b = -m[idx] * pc_eq[idx] + sw_eq[idx]
x_eq_at_0 = -b/(m[idx])  # when y = 0
x_eq_at_1 = (1-b)/(m[idx])  # when y = 1
print(f"Equivalent Breakthrough Pressure: {x_eq_at_0}")

# make plot
plt.figure(1, dpi=500)
plt.plot(pc_im*1e-3, sw_im, 'k-o', label=name, linewidth=4, markersize=12)
plt.plot(pc_eq[0:-2]*1e-3, sw_eq[0:-2], 'b-^', label="SNOW", linewidth=4, markersize=12)
plt.plot([x_im_at_0*1e-3, x_im_at_1*1e-3], [0, 1], 'r--', linewidth=4, markersize=12)
plt.plot([x_eq_at_0*1e-3, x_eq_at_1*1e-3], [0, 1], 'r--', label="Tangent (Equivalent)", linewidth=4, markersize=12)
plt.xlabel('Capillary Pressure (kPa)', fontsize="large")
plt.ylabel('Saturation', fontsize="large")
plt.legend(labels=[name, "SNOW", "tanget"], frameon=True, fontsize=18)
plt.grid(True, which='both', linestyle='--', color='lightgrey', linewidth=0.7, alpha=0.7)
plt.minorticks_on()

# save
plt.savefig('../figures/porosimetry-churcher'+ name, dpi=500)

# %% Plot figure from churcher

data = np.loadtxt(fname="../data/churcher.csv", delimiter=",", skiprows=1)
Kexp = pd.read_csv('../data/K-' + name + '.csv', header=None).values.flatten()
K_avg = np.array([np.average(Kexp)])

# retrieve data
Pc = data[:, 0]  # break-through capillary pressure (kPa)
K = data[:, 1]  # permeability (mD)

# plot fitted curve from churcher et al. 1991
def fun(x):
    y = 1/(1.58e-5 * x + 1.25e-4) - 488.5
    return y
x = np.arange(13, 100, 1)
y = fun(x)

# plot K results
plt.figure(2, dpi=500)
ax = plt.gca()  # Get current axes
# Make the bounding box bold
for spine in ax.spines.values():
    spine.set_linewidth(3)
ax.tick_params(direction='in', length=6, width=3, labelsize=12) 
plt.title('Permeability vs. Breakthrough Pressure', fontsize=14, fontweight='semibold')
plt.ylabel("Permeability (mD)", fontsize=14)
plt.xlabel("Breakthrough Pressure (kPa)", fontsize=14)
plt.plot(x, y, color="grey", label="Churcher et al. Model")
plt.plot(Pc, K, "ko", label="Experiment", markerfacecolor="k")
plt.plot([x_eq_at_0/1000], K_avg, "ro", label=name, markerfacecolor="r")
plt.ylim([0, 3000])
plt.xlim([0, 120])
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(loc='best', fontsize=18, frameon=True)
plt.tight_layout()
plt.savefig('../figures/churcher' + '.png')
plt.show()