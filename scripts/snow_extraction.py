import matplotlib.pyplot as plt
import porespy as ps
import openpnm as op
import numpy as np
from skimage.morphology import square, cube
import scipy.ndimage as spim
import tifffile
import time as time

np.random.seed(10)

# import image
name = 'S5'
res = 4e-6
raw = np.fromfile('../images/' + name + '.raw', dtype=np.uint8)
shape = np.ceil(len(raw)**(1/3)).astype('int')
im = (raw.reshape(shape, shape, shape))
im = im == 0
im = ps.filters.fill_blind_pores(im, conn=26, surface=True)
im = ps.filters.trim_floating_solid(im, conn=6, surface=False)

# calculate porosity
print(f'porosity: {np.sum(im)/np.prod(im.shape)*100}')

# SNOW extraction
bw = 3
r_max = 4
start = time.time()
snow = ps.networks.snow2(im,
                         boundary_width=bw,
                         voxel_size=res,
                         r_max=r_max)
stop = time.time()
print(f'Extraction time: {stop - start}s')  # time NOT measured for parallelization=None
net = op.io.network_from_porespy(snow.network)

# check network health
h = op.utils.check_network_health(net)
print(h)
op.topotools.trim(net, pores=np.append(h['isolated_pores'], h['disconnected_pores']))
h = op.utils.check_network_health(net)
print(h)

# save network
op.io.network_to_csv(net, filename='../networks/' + name + '-snow.csv')

# save exact image we performed extraction on
tifffile.imwrite('../images/' + name + '-snow-extracted.tif', im)

# export to paraview
op.topotools.trim(net, pores=net.pores('boundary'))
ps.io.to_stl(im, '../paraview/' + name + '-im')
net['pore.coords'] = net['pore.coords']/res
net['pore.diameter'] = net['pore.inscribed_diameter']/res
net['throat.radius'] = net['throat.inscribed_diameter']/res/2
net['pore.coords'] += 10  #*res
proj = net.project
op.io.project_to_xdmf(proj, filename= '../paraview/' + name + '-network-snow')