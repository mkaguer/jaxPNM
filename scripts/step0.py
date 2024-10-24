import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import cg

# Example of a symmetric positive definite matrix A
def generate_spd_matrix(size):
    A = jnp.eye(size) + 0.01 * jax.random.normal(jax.random.PRNGKey(0), (size, size))
    A = (A + A.T) / 2  # Make the matrix symmetric
    return A

# Define the size of the matrix
n = 1000

# Create a random symmetric positive definite matrix A
A = generate_spd_matrix(n)

# Define a random vector b
b = jax.random.normal(jax.random.PRNGKey(1), (n,))

# Define the linear operator function for CG
def A_mv(x):
    return jnp.dot(A, x)

# Solve the linear system using conjugate gradient method
x0 = jnp.zeros_like(b)  # Initial guess for x

# Run the conjugate gradient algorithm
solution, _ = cg(A_mv, b, x0=x0)

# Output the solution
print("Conjugate Gradient Solution:", solution)
# print("Info:", info)
