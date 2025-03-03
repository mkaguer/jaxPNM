import porespy as ps
import openpnm as op
import numpy as np
import scipy.ndimage as spim
from skimage.morphology import square, cube
import tifffile
import time


# import image
name = 'A1'
res = 3.85e-6
raw = np.fromfile('../images/' + name + '.raw', dtype=np.uint8)
shape = np.ceil(len(raw)**(1/3)).astype('int')
im = (raw.reshape(shape, shape, shape))
im = im == 0
im = im
im = ps.filters.fill_blind_pores(im, conn=26, surface=True)
im = ps.filters.trim_floating_solid(im, conn=6, surface=False)

# calculate porosity
print(f'porosity: {np.sum(im)/np.prod(im.shape)*100}')

# perform MAGNET on im
start = time.time()
net, sk, juncs, throat_area = ps.networks.magnet(im,
                                                 sk=None,
                                                 parallel=False,
                                                 surface=False,
                                                 voxel_size=res,
                                                 l_max=7,
                                                 throat_junctions="fast marching",
                                                 throat_area=True,
                                                 n_walkers=10,
                                                 max_n_steps=10)
stop = time.time()
print(f'Extraction Time: {stop - start}s')

# add diameter property
net = op.io.network_from_porespy(net)

# check network health
h = op.utils.check_network_health(net)
print(h)
op.topotools.trim(net, throats=np.append(h['looped_throats'], h['duplicate_throats']))
h = op.utils.check_network_health(net)
print(h)

# number of pore vs. skeleton clusters in network
from scipy.sparse import csgraph as csg
am = net.create_adjacency_matrix(fmt='coo', triu=True)
Np, cluster_num = csg.connected_components(am, directed=False)
print('Pore clusters:', Np)
# number of skeleton pieces
b = square(3) if im.ndim == 2 else cube(3)
_, Ns = spim.label(input=sk.astype('bool'), structure=b)
print('Skeleton clusters:', Ns)

# save network
op.io.network_to_csv(net, filename='../networks/' + name + '-magnet.csv')

# save exact image we performed extraction on
tifffile.imwrite('../images/' + name + '-magnet-extracted.tif', im)

# export to paraview
ps.io.to_stl(~sk, '../paraview/' + name + '-sk')
ps.io.to_stl(im, '../paraview/' + name + '-im')
ps.io.to_stl(im[:int(im.shape[0]/2), :, :], '../paraview/' + name + '-im-half')
net['pore.coords'] = net['pore.coords']/res
net['pore.diameter'] = net['pore.inscribed_diameter']/res
net['throat.radius'] = net['throat.min_diameter']/res/2
net['pore.coords'] += 10  #*res
proj = net.project
op.io.project_to_xdmf(proj, filename= '../paraview/' + name + '-network-magnet')