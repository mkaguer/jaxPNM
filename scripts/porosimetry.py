import numpy as np
from edt import edt
import porespy as ps
import matplotlib.pyplot as plt

ps.visualization.set_mpl_style()
np.random.seed(10)

name = 'A1'
res = 3.85e-6
raw = np.fromfile('../images/' + name + '.raw', dtype=np.uint8)
shape = np.ceil(len(raw)**(1/3)).astype('int')
im = (raw.reshape(shape, shape, shape))
im = im == 0

# take distance transform
dt = edt(im)

# capillary transform
pc = ps.filters.capillary_transform(im, sigma=0.4791, theta=140, voxel_size=res)

# get steps
vmax = pc[pc < np.inf].max()
vmin = pc[im][pc[im] > -np.inf].min()
steps = np.logspace(np.log10(vmin), np.log10(vmax)*1.1, 25)

# simulate drainage
inlets = np.zeros(np.asarray(im.shape) - 2, dtype=bool)
inlets = np.pad(inlets, pad_width=1, mode='constant', constant_values=1)
e = ps.simulations.drainage(im, pc=pc, dt=dt, inlets=inlets, steps=steps)

# plot
plt.figure(3)
plt.semilogx(e.pc, e.snwp, 'k-o', label='Image-Based')
plt.legend(fontsize=16)
plt.title(name, fontsize=18)
plt.xlabel('Capillary Pressure (Pa)', fontsize=16)
plt.ylabel('Saturation', fontsize=16)
plt.savefig('../figures/porosimetry-' + name, dpi=500)
plt.show()

# save data
pc = np.array([e.pc])
sat = np.array([e.snwp])
data = np.concatenate((pc, sat), axis=0).T
np.savetxt('../data/porosimetry-' + name + '.csv', data, delimiter=',')

