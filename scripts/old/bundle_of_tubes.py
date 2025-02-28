import numpy as np
import jax
import jax.numpy as jnp
import mypnmlib as pnm
from _fit_cubic_network import FitCubicNetwork
import matplotlib.pyplot as plt
import diffrax as dfx
from scipy.stats import rv_discrete
import matplotlib.pyplot as plt

# import data
image = '../data/Berea'
data = jnp.array(np.loadtxt(image + '.csv', delimiter=','))
sat_target = data[:, 1]
pressures = data[:, 0]  # interpolate?

# calculate D from data
sigma = 0.4791  # N/m
theta = 140  # radians!
D = -4*sigma*jnp.cos(jnp.radians(theta))/pressures

# assume volume of a tube, bundles of tubes approzimation!
l = 1
V = jnp.pi*D**2/4*l
f = jnp.zeros(len(sat_target), dtype=float)
f = f.at[0].set(sat_target[0])
f = f.at[1:].set(sat_target[1:] - sat_target[0:-1])
f = f/V
f = f/f.sum()

# create discrtete random variable sampler
pmf = rv_discrete(values=(D, f), seed=1)

# take samples
diameters = pmf.rvs(size=1000)

# plot sample
plt.figure(1)
plt.hist(diameters, bins=20)
plt.show()
 