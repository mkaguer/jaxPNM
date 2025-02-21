import jax.numpy as jnp
import jax.experimental.sparse as js
from mypnmlib.network import create_adjacency_matrix, graph_laplacian

__all__ = ["build_A",
           "build_b",
           "update_A_and_b",
           "set_BC",
           "apply_BC",
           "rate"]

def build_A(net):

    g = net['throat.conductance']
    am = create_adjacency_matrix(net, weights=g, fmt='coo')
    # FIXME: watch out for assymetric physics!
    # laplacian
    A = graph_laplacian(am, fmt='coo')  # keep in coo until the end!

    return A


def build_b(net):

    Np = len(net['pore.coords'])
    b = jnp.zeros(Np, dtype=float)

    return b


def update_A_and_b(net):

    # build A and b
    pure_A = build_A(net)
    pure_b = build_b(net)
    # assign to network
    net['A'] = pure_A
    net['b'] = pure_b
    # apply BCs
    A, b = apply_BC(net)
    # FIXME: remover zeros here?
    # assign to dict again!
    net['A'] = A
    net['b'] = b

    return A, b


def set_BC(net, pores, bctype, bcvalues, mode='overwrite'):

    # FIXME: This method needs some work! Handling 'mode' specifically!
    Np = len(net['pore.coords'])
    if bctype == 'value':
        if mode == 'overwrite':
            # note zero is assigned by default as bc value, use mask to control
            net['pore.bc.value'] = jnp.zeros(Np, dtype=float)
            net['pore.bc.mask'] = jnp.zeros(Np, dtype=bool)
            net['pore.bc.value'] = net['pore.bc.value'].at[pores].set(bcvalues)
            net['pore.bc.mask'] = net['pore.bc.mask'].at[pores].set(True)
            net['boundary_pores'] = jnp.array(pores, dtype=int)
        elif mode == 'add':
            net['pore.bc.value'] = net['pore.bc.value'].at[pores].set(bcvalues)
            net['pore.bc.mask'] = net['pore.bc.mask'].at[pores].set(True)
            net['boundary_pores'] = jnp.concatenate((net['boundary_pores'], jnp.array(pores)), dtype=int)
        else:
            raise ValueError(f"{mode} is not a supported mode")
    else:
        raise ValueError(f"{bctype} is not a supported bctype")


def apply_BC(net, axis=''):

    """Applies specified boundary conditions by modifying A and b."""
    # get pure A and pure b from network
    A = net['A']
    b = net['b']
    # apply rate BC to b
    if 'pore.bc.rate' in net.keys():
        ind = jnp.isfinite(net['pore.bc.rate'])
        b = b.at[ind].set(-net['pore.bc.rate'][ind])  # negative for production
    # apply value BC to A and b
    if 'pore.bc.value' + axis in net.keys():
        Np = len(net['pore.coords'])
        # get average of diagonal, note this only works for 'coo' format
        diag = A.data[0:Np]  # this only works for 'coo' format
        f = diag.mean()
        # Update b (impose bc values)
        bc_values = net['pore.bc.value' + axis]
        bc_mask = net['pore.bc.mask' + axis]
        b = jnp.where(bc_mask, bc_values*f, b)
        # Update b (subtract quantities from b to keep A symmetric)
        x_BC = jnp.where(bc_mask, bc_values, 0.0)
        temp = b - A @ x_BC
        b = jnp.where(bc_mask, b, temp)
        # update A
        P_bc = net['boundary_pores' + axis]  # FIXME: Is there way aroung this?
        mask = jnp.isin(A.indices[:, 0], P_bc) | jnp.isin(A.indices[:, 1], P_bc)
        A_data = jnp.where(mask, 0, A.data)
        # Add diagonal entries back into A
        isdiag = A.indices[:, 0] == A.indices[:, 1]
        mask = isdiag * jnp.isin(A.indices[:, 0], P_bc)
        A_data = jnp.where(mask, f, A_data)
        A_indices = A.indices
        # Finally, update A in BCOO format
        A = js.BCOO((A_data, A_indices), shape=(Np, Np))

    return A, b


def rate(net, x, pores=[], throats=[], mode='group'):
    """
    Calculates the net rate of material moving into a given set of
    pores or throats

    Parameters
    ----------
    x : array_like
        The solved for quantity
    pores : array_like
        The pores for which the rate should be calculated
    throats : array_like
        The throats through which the rate should be calculated
    mode : str, optional
        Controls how to return the rate. The default value is 'group'.
        Options are:

        ===========  =====================================================
        mode         meaning
        ===========  =====================================================
        'group'      Returns the cumulative rate of material
        'single'     Calculates the rate for each pore individually
        ===========  =====================================================

    Returns
    -------
    If ``pores`` are specified, then the returned values indicate the
    net rate of material exiting the pore or pores.  Thus a positive
    rate indicates material is leaving the pores, and negative values
    mean material is entering.

    If ``throats`` are specified the rate is calculated in the
    direction of the gradient, thus is always positive.

    If ``mode`` is 'single' then the cumulative rate through the given
    pores (or throats) are returned as a vector, if ``mode`` is
    'group' then the individual rates are summed and returned as a
    scalar.

    """
    pores = jnp.array(pores)
    throats = jnp.array(throats)

    if throats.size > 0 and pores.size > 0:
        raise Exception('Must specify either pores or throats, not both')
    if (throats.size == 0) and (pores.size == 0):
        raise Exception('Must specify either pores or throats')

    # get Nt and Np
    Nt = len(net['throat.conns'])
    Np = len(net['pore.coords'])

    # get conductance
    g = net['throat.conductance']

    P12 = net['throat.conns']
    X12 = x[P12]
    if g.size == Nt:
        g = jnp.tile(g, (2, 1)).T    # Make conductance an Nt by 2 matrix
    # The next line is critical for rates to be correct
    # We could also do "g.T.flatten()" or "g.flatten('F')"
    g = jnp.flip(g, axis=1)
    Qt = jnp.diff(g*X12, axis=1).ravel()

    if throats.size:
        R = jnp.absolute(Qt[throats])
        if mode == 'group':
            R = jnp.sum(R)
    elif pores.size:
        Qp = jnp.zeros((Np, ))
        Qp = Qp.at[P12[:, 0]].add(-Qt)
        Qp = Qp.at[P12[:, 1]].add(Qt)
        R = Qp[pores]
        if mode == 'group':
            R = jnp.sum(R)

    return jnp.array(R, ndmin=1)