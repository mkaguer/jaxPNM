import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.sparse.linalg import cg
from scipy.sparse.linalg import cg as cgs
import time

# Example of a symmetric positive definite matrix A
def generate_spd_matrix(size):
    A = jnp.eye(size) + 0.01 * jax.random.normal(jax.random.PRNGKey(0), (size, size))
    A = (A + A.T) / 2  # Make the matrix symmetric
    return A

# Define the size of the matrix
n = 10000
print(f'Size: {n*n}')


# Create a random symmetric positive definite matrix A
A = generate_spd_matrix(n)
A_numpy = np.array(A)

# Define a random vector b
b = jax.random.normal(jax.random.PRNGKey(1), (n,))
b_numpy  = np.array(b)

# Define the linear operator function for CG
# def A_mv(x):
#     return jnp.dot(A, x)

# Run the JAX conjugate gradient algorithm
start = time.time()
soln_jax, _ = cg(A, b)
stop = time.time()
print(f'JAX time: {stop - start}s')

# Run the Scipy conjugate gradient algorithm
start = time.time()
soln_scipy, _ = cgs(A_numpy, b_numpy)
stop = time.time()
print(f'Scipy time: {stop - start}s')

print(soln_jax.mean())
print(soln_scipy.mean())

# Output the solution
# print("Conjugate Gradient Solution:", solution)
# print("Info:", info)
