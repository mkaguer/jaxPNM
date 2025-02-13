import numpy as np
import matplotlib.pyplot as plt
import jax

# load files
grad_5 = np.load('../data/gradient-5by5by5.npy')
grad_7 = np.load('../data/gradient-7by7by7.npy')
grad_9 = np.load('../data/gradient-9by9by9.npy')

# plot histogram
bins = np.linspace(-5, 2, 21)
plt.figure(1)
plt.hist(np.log10(np.abs(grad_5)), bins=bins, alpha=0.5, label='5by5by5', density=True)
plt.hist(np.log10(np.abs(grad_7)), bins=bins, alpha=0.5, label='7by7by7', density=True)
plt.hist(np.log10(np.abs(grad_9)), bins=bins, alpha=0.5, label='9by9by9', density=True)
plt.ylabel('Probability density')
plt.xlabel('log10(abs(grad))')
plt.title('Distribution of Initial Gradients')
plt.legend()
plt.show()

# get diameters
key = jax.random.PRNGKey(0)
D_5 = np.asarray(jax.random.uniform(key, shape=(5**3,)))
D_7 = np.asarray(jax.random.uniform(key, shape=(7**3,)))
D_9 = np.asarray(jax.random.uniform(key, shape=(9**3,)))

# plot histogram
bins = np.linspace(0, 1, 11)
plt.figure(2)
plt.hist(D_5, bins=bins, alpha=0.5, label='5by5by5', density=True)
plt.hist(D_7, bins=bins, alpha=0.5, label='7by7by7', density=True)
plt.hist(D_9, bins=bins, alpha=0.5, label='9by9by9', density=True)
plt.ylabel('Probability density')
plt.xlabel('Diameter, D (of spacing)')
plt.title('Distribution of Diameters')
plt.legend()
plt.show()

'''
# plot histogram
bins = np.linspace(-5, 2, 21)
plt.figure(3)
plt.hist(np.log10(np.abs(grad_5))*D_5, bins=bins, alpha=0.5, label='5by5by5', density=True)
plt.hist(np.log10(np.abs(grad_7))*D_7, bins=bins, alpha=0.5, label='7by7by7', density=True)
plt.hist(np.log10(np.abs(grad_9))*D_9, bins=bins, alpha=0.5, label='9by9by9', density=True)
plt.ylabel('Probability density')
plt.xlabel('log10(abs(grad))*D')
plt.title('Distribution of Initial Gradients*D')
plt.legend()
plt.show()

# plot gradient versus pore size
plt.figure(4)
plt.plot(D_5, np.log10(np.abs(grad_5)), marker='o', linewidth=0, alpha=0.5, label='5by5by5')
# plt.plot(D_7, np.log10(np.abs(grad_7)), marker='o', alpha=0.5, label='7by7by7')
# plt.plot(D_9, np.log10(np.abs(grad_9)), marker='o', alpha=0.5, label='9by9by9')
plt.ylabel('log10(abs(grad))')
plt.xlabel('Diamter')
plt.title('Distribution of Initial Gradients*D')
plt.legend()
plt.show()
'''