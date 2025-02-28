import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

np.random.seed(1)

# select number of samples
num_samples = 1000

# import diameters
image = 'Berea'
Ds = np.load('../data/ai-diameters-' + image + '.npy')
D = Ds[:, 1]

# fit data to weibull distribution
shape, loc, scale = sp.stats.weibull_min.fit(D, floc=0)  
w_dist = sp.stats.weibull_min(c=shape, scale=scale, loc=loc)

# find maximum seed
min_seed = w_dist.cdf(x=1e-3)
max_seed = w_dist.cdf(x=1)

# generate uniform distribution of seeds from 0 to max_seed
u_dist = sp.stats.uniform(loc=min_seed, scale=max_seed)
seeds = u_dist.rvs(num_samples)

# use ppf to get diameters from seed values
D_sampled = w_dist.ppf(seeds)

# get parent distribution
D_parent = w_dist.rvs(100000)

# plot
plt.hist(D_sampled, bins=10, alpha=0.5, density=True, label='Sample')
plt.hist(D_parent, bins=100, alpha=0.5, density=True, label='Parent Distribution')
plt.xlabel('Diameter')
plt.ylabel('Frequency')
plt.legend()
plt.show()
