import jax.numpy as jnp
import jax.experimental.sparse as js


def cubic(shape, spacing=1, connectivity=6, node_prefix='node', edge_prefix='edge'):
    r"""
    Generate a simple cubic lattice

    Parameters
    ----------
    shape : array_like
        The number of unit cells in each direction.  A unit cell has 1 vertex
        at its center.
    spacing : array_like or float
        The size of a unit cell in each direction. If an scalar is given it is
        applied in all 3 directions.

    Returns
    -------
    network : dict
        A dictionary containing ``coords`` and ``conns`` of a cubic network with the
        specified spacing and connectivity.

    """
    shape = jnp.array(shape, ndmin=1)
    shape = jnp.concatenate((shape, jnp.ones(3 - shape.size, dtype=int))).astype(int)
    arr = jnp.empty(shape).reshape((shape[0], shape[1], shape[2]))
    
    spacing = jnp.asarray(spacing, dtype=jnp.float32)
    if spacing.size == 2:
        spacing = jnp.concatenate((spacing, jnp.ones(1)))
    spacing = jnp.ones(3, dtype=float) * jnp.array(spacing, ndmin=1)
    
    z = jnp.tile(jnp.arange(shape[2]), shape[0] * shape[1])
    y = jnp.tile(jnp.repeat(jnp.arange(shape[1]), shape[2]), shape[0])
    x = jnp.repeat(jnp.arange(shape[0]), shape[1] * shape[2])
    points = (jnp.vstack([x, y, z]).T).astype(float) + 0.5
    
    idx = jnp.arange(arr.size).reshape(arr.shape)
    
    face_joints = [(idx[:, :, :-1], idx[:, :, 1:]),
                   (idx[:, :-1], idx[:, 1:]),
                   (idx[:-1], idx[1:])]
    
    corner_joints = [(idx[:-1, :-1, :-1], idx[1:, 1:, 1:]),
                     (idx[:-1, :-1, 1:], idx[1:, 1:, :-1]),
                     (idx[:-1, 1:, :-1], idx[1:, :-1, 1:]),
                     (idx[1:, :-1, :-1], idx[:-1, 1:, 1:])]
    
    edge_joints = [(idx[:, :-1, :-1], idx[:, 1:, 1:]),
                   (idx[:, :-1, 1:], idx[:, 1:, :-1]),
                   (idx[:-1, :, :-1], idx[1:, :, 1:]),
                   (idx[1:, :, :-1], idx[:-1, :, 1:]),
                   (idx[1:, 1:, :], idx[:-1, :-1, :]),
                   (idx[1:, :-1, :], idx[:-1, 1:, :])]
    
    if connectivity == 6:
        joints = face_joints
    elif connectivity == 6 + 8:
        joints = face_joints + corner_joints
    elif connectivity == 6 + 12:
        joints = face_joints + edge_joints
    elif connectivity == 12 + 8:
        joints = edge_joints + corner_joints
    elif connectivity == 6 + 8 + 12:
        joints = face_joints + corner_joints + edge_joints
    else:
        raise Exception("Invalid connectivity. Must be 6, 14, 18, 20 or 26.")
    
    tails, heads = jnp.array([], dtype=int), jnp.array([], dtype=int)
    for T, H in joints:
        tails = jnp.concatenate((tails, T.flatten()))
        heads = jnp.concatenate((heads, H.flatten()))
    pairs = jnp.vstack([tails, heads]).T
    
    if connectivity != 6:
        pairs = jnp.sort(pairs, axis=1)
    
    d = {}
    d[f"{node_prefix}.coords"] = points * spacing
    d[f"{edge_prefix}.conns"] = pairs
    
    return d


def create_adjacency_matrix(net, weights=None, fmt='coo', triu=False, drop_zeros=False):
    """
    Generates a weighted adjacency matrix in the desired sparse format using JAX.

    Parameters
    ----------
    weights : array_like, optional
        An array containing the throat values to enter into the matrix
        (in graph theory these are known as the 'weights').
        - If `weights` is `None`, all connections will have weight 1.
        - If `weights` has length Nt, it is assumed the matrix is symmetric.
        - If `weights` has length 2*Nt, the first Nt values are for the upper
          triangle, and the second Nt for the lower triangle.
    fmt : str, optional
        The sparse storage format to return. Options are:
            - `'coo'`: Coordinate format (default).
            - `'csr'`: Compressed Sparse Row format.
    triu : bool, default is False
        If True, only the upper-triangular part is included (ignored if weights
        has length 2*Nt).
    drop_zeros : bool, default is False
        If True, entries with zero weights are removed.

    Returns
    -------
    A JAX sparse adjacency matrix in the specified format.
    """
    # get Nt and Np
    Nt = len(net['throat.conns'])
    Np = len(net['pore.coords'])
    
    allowed_weights = [(Nt,), (2 * Nt,), (Nt, 2)]

    # Handle weights
    if weights is None:
        weights = jnp.ones((Nt,), dtype=jnp.float32)
    elif weights.shape not in allowed_weights:
        raise ValueError("Received weights are of incorrect length")

    # Extract throat connections
    conn = net['throat.conns']
    row, col = conn[:, 0], conn[:, 1]

    # Modify rows and columns for symmetric or triangular matrices
    if weights.shape == (2 * Nt,):
        row = jnp.concatenate([row, conn[:, 1]])
        col = jnp.concatenate([col, conn[:, 0]])
    elif weights.shape == (Nt, 2):
        row = jnp.concatenate([row, conn[:, 1]])
        col = jnp.concatenate([col, conn[:, 0]])
        weights = weights.flatten(order='F')
    elif not triu:
        row = jnp.concatenate([row, conn[:, 1]])
        col = jnp.concatenate([col, conn[:, 0]])
        weights = jnp.concatenate([weights, weights])

    # Eliminate zeros if needed
    if drop_zeros:
        mask = weights != 0
        row, col, weights = row[mask], col[mask], weights[mask]

    # Shape of the adjacency matrix
    shape = (Np, Np)
    
    print(weights.shape)
    print(jnp.stack([row, col]).shape)
    # Create the adjacency matrix in COO format
    coo_matrix = js.BCOO((weights, jnp.stack([row, col]).T), shape=shape)

    # Convert to desired format
    if fmt == 'coo':
        return coo_matrix
    elif fmt == 'csr':
        return js.BCSR.from_bcoo(coo_matrix)
    else:
        raise ValueError(f"Format {fmt} is not supported for JAX sparse matrices")