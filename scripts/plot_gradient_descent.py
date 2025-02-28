import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import ffmpeg
import porespy as ps

ps.visualization.set_mpl_style()

# Define the function f(x)
def f(x):
    return x**2 + 2 * jnp.sin(x) + 4 * jnp.cos(x)

# Define the gradient of f(x)
grad_f = jax.grad(f)

# Create x values for plotting the function
x = np.linspace(-4, 4, 100)
y = f(x)

# Set learning rate and initial point
lr = 1.0
x0 = [3.5]
fx = [f(x0[0])]

# Perform gradient descent for 10 iterations
for i in range(10):
    x_new = x0[i] - lr * grad_f(x0[i])
    x0.append(x_new)
    fx.append(f(x_new))

# Set up the figure and axis
fig, ax = plt.subplots()
ax.plot(x, y, label="f(x)", linewidth=3)
point, = ax.plot([], [], 'ro-', markersize=8, markerfacecolor='red') # Marker for the descent steps

# Animation function
def update(frame):
    point.set_data(x0[:frame], fx[:frame])
    return point,

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=len(x0), interval=500, blit=True)

# Show animation
plt.legend()
plt.show()

ani.save(f"../figures/gradient_descent_{lr}" + ".gif", writer="pillow", fps=5, dpi=500)
