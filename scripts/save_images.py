import numpy as np
import porespy as ps 

name = 'Berea'
raw = np.fromfile('../images/' + name + '.raw', dtype=np.uint8)
shape = np.ceil(len(raw)**(1/3)).astype('int')
im = (raw.reshape(shape, shape, shape))
im = im == 0

print(f'Porosity: {np.sum(im)/np.prod(im.shape)}')

ps.io.to_stl(im, '../images/' + name + '.stl')
