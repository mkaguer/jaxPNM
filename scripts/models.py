import jax.numpy as jnp

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
