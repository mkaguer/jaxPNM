import numpy as np
import matplotlib.pyplot as plt
import openpnm as op

op.visualization.set_mpl_style()

# import loss data
l1 = np.loadtxt("../data/losses1.csv", delimiter=",")
l2 = np.loadtxt("../data/losses2.csv", delimiter=",")

# get size of data
n = len(l1)

# plot
plt.figure(1, dpi=600)
plt.semilogy(np.arange(0, n, 1), l1, color="k", linewidth=4, label="Original Initial Guess")
plt.semilogy(np.arange(0, n, 1), l2, color="r", linewidth=4, label="New Initial Guess")
plt.legend(frameon=True, fontsize=14)
plt.xlabel("Number of Iterations", fontsize=14)
plt.ylabel("Loss", fontsize=14)
plt.savefig("../figures/review-loss.png")
plt.show()