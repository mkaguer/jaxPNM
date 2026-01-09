import porespy as ps
import numpy as np
import matplotlib.pyplot as plt
from edt import edt

# %%
im1 = ps.generators.conical_capillary([50, 50, 50], [12, 7], axis=0)
im2 = ps.generators.conical_capillary([50, 50, 50], [7, 15], axis=0)
im3 = np.ones([50, 50, 50])
im3[..., :5] = False
im3[..., -5:] = False
im3[..., :5, :] = False
im3[..., -5:, :] = False
im3[-5:, :, :] = False
im = np.vstack((im1, im2, im3)).astype(bool)
# im = im[30:, :, :]
# plt.imshow(ps.visualization.xray(im, axis=1))

# %%
inlets = ps.generators.faces(im.shape, inlet=0)*im


pc = ps.filters.capillary_transform(
    im, sigma=0.01, theta=180, voxel_size=1e-6)

inj = ps.simulations.injection(pc=pc, im=im, inlets=inlets)
drn = ps.simulations.drainage(pc=pc, im=im, inlets=inlets)

data1 = ps.metrics.pc_map_to_pc_curve(pc=inj.im_pc, seq=inj.im_seq, im=im)
data2 = ps.metrics.pc_map_to_pc_curve(pc=drn.im_pc, seq=drn.im_seq, im=im)

fig, ax = plt.subplot_mosaic([['a', 'b'],
                              ['a', 'c']])
ax['a'].plot(data1.pc, data1.snwp)
ax['a'].step(data2.pc, data2.snwp, where='post')
ax['b'].imshow(inj.im_pc[..., 25].T) 
ax['c'].imshow(drn.im_pc[..., 25].T) 
 