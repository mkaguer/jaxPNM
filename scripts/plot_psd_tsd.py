import openpnm as op
import matplotlib.pyplot as plt
import numpy as np
import porespy as ps

ps.visualization.set_mpl_style()

# select image
name = 'Berea'
res = 5.35e-6

# import image
raw = np.fromfile('../images/' + name + '.raw', dtype=np.uint8)
shape = np.ceil(len(raw)**(1/3)).astype('int')
im = (raw.reshape(shape, shape, shape))
im = im == 0
im = ps.filters.fill_blind_pores(im, conn=26, surface=True)
im = ps.filters.trim_floating_solid(im, conn=6, surface=False)

# calculate porosity
print(f'porosity: {np.sum(im)/np.prod(im.shape)*100}')

# perform local thickness transform on image!
thk = ps.filters.local_thickness(im, mode='dt')

# get psd from local thickness transform
bins = np.arange(0, 150, 10)/1e6
psd = ps.metrics.pore_size_distribution(thk, bins=bins, log=False, voxel_size=res)

# normalize freq by volume, this did not seem to help!
freq = psd.pdf
R = psd.R
V = 4/3 * np.pi * R ** 3
freq = freq / V
freq = freq / np.max(freq)

# load network
net_m = op.io.network_from_csv('../networks/' + name + '-magnet.csv')
net_s = op.io.network_from_csv('../networks/' + name + '-snow.csv')

fig, ax = plt.subplots(1, 2, sharey=True)
ax[0].hist(net_m['pore.inscribed_diameter']*1e6, bins=bins*1e6, label='MAGNET', color='m', alpha=0.5, density=True)
ax[0].hist(net_s['pore.inscribed_diameter']*1e6, bins=bins*1e6, label='SNOW', color='c', alpha=0.5, density=True)
# ax[0].bar(psd.bin_centers*1e6, freq, width=psd.bin_widths*1e6, align='center', label='Local Thickness', color='k', alpha=0.5)
ax[0].hist(thk[im]*res*1e6, bins=bins*1e6, label='Local Thickness', color='k', alpha=0.5, density=True)
ax[0].set_title('PSD', fontsize=12)
ax[0].set_xlabel('Pore Diameter (um)', fontsize=14)
ax[0].legend()
ax[1].hist(net_m['throat.inscribed_diameter']*1e6, bins=bins*1e6, label='MAGNET', color='m', alpha=0.5, density=True)
ax[1].hist(net_s['throat.inscribed_diameter']*1e6, bins=bins*1e6, label='SNOW', color='c', alpha=0.5, density=True)
ax[1].set_title('TSD', fontsize=12)
ax[1].set_xlabel('Throat Diameter (um)', fontsize=14)
ax[1].legend()
# fig.suptitle(name, fontsize=16)
# plt.tight_layout()
plt.show()