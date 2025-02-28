import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import porespy as ps

ps.visualization.set_mpl_style()

# %% Plot Weibull with different scale and shape!

def weibull_pdf(x, scale, shape):

    k = shape
    lam = scale
    f = k/lam * (x/lam) ** (k-1) * np.exp(-(x/lam) ** k)

    return f

scale = 1
shapes = np.array([0.5, 1.0, 1.5, 5.0])
x = np.arange(0, 2.5, 0.025)
for i in range(len(shapes)):
    f = weibull_pdf(x, scale=scale, shape=shapes[i])
    plt.figure(1)
    plt.plot(x, f, label=f'scale: {scale}, shape: {shapes[i]}')
plt.legend(fontsize=17)
plt.show()

scales = np.array([0.5, 1.0, 1.5])
shape = 5
x = np.arange(0, 2.5, 0.025)
for i in range(len(scales)):
    f = weibull_pdf(x, scale=scales[i], shape=shape)
    plt.figure(1)
    plt.plot(x, f, label=f'scale: {scales[i]}, shape: {shape}')
plt.legend(fontsize=17)
plt.show()


# %% Plot Sigmoid function

def sigmoid(x, sf=1):

    f = 1/(1 + np.exp(-x/sf))

    return f

colors = np.array(['skyblue', 'royalblue', 'navy'])
sfs = np.array([1, 2, 3])
x = np.arange(-10, 10, 0.1)
for i, sf in enumerate(sfs):
    f = sigmoid(x, sf)
    plt.figure(2)
    plt.plot(x, f, label=f'Smoothing Factor: {sf}', color=colors[i])
plt.axvline(x=0, color='gray', linestyle='--')
plt.axhline(y=0.5, color='gray', linestyle='--')
plt.grid()
plt.xlim([-10, 10])
plt.ylim([-0.1, 1.1])
plt.legend()
# plt.title('Sigmoid', fontsize=18)
plt.show()
