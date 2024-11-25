"""
The goal is to write a script that fits a cubic network to have 
desired permeability using JAX and Diffrax.

Created by: Mike McKague
Date: November 18, 2024
"""

import models as mods


def make_cubic_network(shape, spacing=1, connectivity=6):
    '''
    Create custom cubic network that returns dictionary of JAX arrays.
    '''
    network = mods.cubic(shape=shape,
                         spacing=spacing,
                         connectivity=connectivity,
                         node_prefix='pore',
                         edge_prefix='throat')
    return network


def calc_throat_length(net):

    conns = net['throat.conns']
    coords = net['pore.coords'][conns]
    diff = jnp.diff(coords, axis=1)
    throat_length = jnp.linalg.norm(diff, axis=2)[:, 0]

    return throat_length


def calc_conductance(net):

    R = net['throat.diameter'] / 2
    mu = net['throat.viscosity']
    L = net['throat.length']

    return jnp.pi * R ** 4 / 8 / mu / L


def build_A(net):

    g = net['throat.conductance']
    am = mods.create_adjacency_matrix(net, weights=g, fmt='coo')
    # FIXME: watch out for assymetric physics!
    # laplacian
    A = mods.graph_laplacian(am, fmt='coo')  # keep in coo until the end!

    return A


def build_b(net):

    Np = len(net['pore.coords'])
    b = jnp.zeros(Np, dtype=float)

    return b


def set_BC(net, pores, bctype, bcvalues, mode='add'):

    # FIXME: This method needs some work! Handling 'mode' specifically!
    Np = len(net['pore.coords'])
    if bctype == 'value':
        if mode == 'overwrite':
            net['pore.bc.value'] = jnp.ones(Np, dtype=float) * jnp.nan
            net['pore.bc.value'] = net['pore.bc.value'].at[pores].set(bcvalues)
        elif mode == 'add':
            net['pore.bc.value'] = net['pore.bc.value'].at[pores].set(bcvalues)
        else:
            raise ValueError(f"{mode} is not a supported mode")
    elif bctype == 'rate':
        if mode == 'overwrite':
            net['pore.bc.rate'] = jnp.ones(Np, dtype=float) * jnp.nan
            net['pore.bc.rate'] = net['pore.bc.rate'].at[pores].set(bcvalues)
        elif mode == 'add':
            net['pore.bc.rate'] = net['pore.bc.rate'].at[pores].set(bcvalues)
        else:
            raise ValueError(f"{mode} is not a supported mode")
    else:
        raise ValueError(f"{bctype} is not a supported bctype")


def apply_BC(net):

    """Applies specified boundary conditions by modifying A and b."""
    # get pure A and pure b from network
    A = net['A']
    b = net['b']
    # apply rate BC to b
    if 'pore.bc.rate' in net.keys():
        ind = jnp.isfinite(net['pore.bc.rate'])
        b = b.at[ind].set(-net['pore.bc.rate'][ind])  # negative for production
    # apply value BC to A and b
    if 'pore.bc.value' in net.keys():
        Np = len(net['pore.coords'])
        # get average of diagonal, note this only works for 'coo' format
        isdiag = A.indices[:, 0] == A.indices[:, 1]
        # diag = A.data[isdiag]
        diag = A.data[0:Np]  # this only works for 'coo' format
        f = diag.mean()
        # Update b (impose bc values)
        bc_values = net['pore.bc.value']
        # FIXME: do jnp.isnan(bc_values) once!!
        b = jnp.where(jnp.isnan(bc_values), b, bc_values*f)  # avoid boolean masks!
        # Update b (subtract quantities from b to keep A symmetric)
        x_BC = jnp.zeros_like(b)
        x_BC = jnp.where(jnp.isnan(bc_values), b, bc_values)
        temp = b - A @ x_BC
        b = jnp.where(jnp.isnan(bc_values), temp, b)
        # update A
        P_bc = jnp.where(jnp.isnan(bc_values), jnp.nan, jnp.arange(0, Np))
        mask = jnp.isin(A.indices[:, 0], P_bc) | jnp.isin(A.indices[:, 1], P_bc)
        # remove entries from A for all BC rows/cols
        A_data = jnp.where(mask, 0, A.data)
        # Add diagonal entries back into A
        mask = isdiag * jnp.isin(A.indices[:, 0], jnp.array(P_bc))
        A_data = jnp.where(mask, f, A_data)
        A_indices = A.indices
        # Cannot remove zeros here to jax transform this function
        # Finally, update A in BCOO format
        A = js.BCOO((A_data, A_indices), shape=(Np, Np))
        
    return A, b


def calc_rate(net, throats, quantity='pore.pressure'):
    
    # FIXME: I should provide pore not throat and use find_neighbour_throats
    conns = net['throat.conns']
    P1 = conns[throats, 0]
    P2 = conns[throats, 1]
    G = net['throat.conductance'][throats]
    X1 = net[quantity][P1]
    X2 = net[quantity][P2]
    rate = -G*(X2 - X1)
    
    return rate


def update_A_and_b(net):
    
    # build A and b
    pure_A = build_A(net)
    pure_b = build_b(net)
    # assign to network
    net['A'] = pure_A
    net['b'] = pure_b
    # apply BCs
    A, b = apply_BC(net)
    # assign to dict again!
    net['A'] = A
    net['b'] = b
    
    return A, b

if __name__ == "__main__":
    
    import jax
    import jax.numpy as jnp
    import jax.experimental.sparse as js
    import diffrax as dfx

    # Step 1: create JAX network
    shape = [4, 1, 1]
    spacing = 1e-4
    net = make_cubic_network(shape=shape, spacing=spacing)

    # get Np and Nt
    Np = len(net['pore.coords'])
    Nt = len(net['throat.conns'])

    # Step 2: add properties
    # a) add viscosity
    mu = 1e-3
    net['throat.viscosity'] = jnp.ones(Nt)*mu  # keep it constant
    # b) add diameter
    net['throat.diameter'] = jnp.ones(Nt)*5e-5  # keep it constant
    # c) calculate throat length and add!
    L = calc_throat_length(net)
    net['throat.length'] = L

    # Step 3: calculate conductance
    G = calc_conductance(net)
    net['throat.conductance'] = G

    # Step 4: build A and b
    # First, build pure A and pure b
    pure_A = build_A(net)
    pure_b = build_b(net)
    net['A'] = pure_A
    net['b'] = pure_b
    # Second, set BCs
    Pin, Pout = 1, 0
    set_BC(net, pores=0, bctype='value', bcvalues=Pin, mode='overwrite')
    set_BC(net, pores=3, bctype='value', bcvalues=Pout, mode='add')
    # Third, apply BCs
    A, b = apply_BC(net)
    net['A'] = A
    net['b'] = b

    # Step 5: Flow sim and get K
    A = js.BCSR.from_bcoo(A)  # need CSR format for linalg.spsolve!
    x = js.linalg.spsolve(A.data, A.indices, A.indptr, b, tol=1e-6)
    net['pore.pressure'] = x

    # calculate permeability
    throats = jnp.array([2])
    Q = calc_rate(net, throats)[0]  # FIXME: sum for total!
    L = (shape[0] - 1)*spacing
    mu = mu
    A = spacing**2
    K = Q * L * mu / (A * (Pin - Pout))/0.98e-12
    print(f'K is: {K:.2f} D')
    
    # Step 6: Wrap steps 3 to 5 into one function!
    def f(D):
        
        K_target = 16  # D
        # Update network with new throat diameter
        net['throat.diameter'] = D
        # Calculate hydraulic conductance
        G = calc_conductance(net)
        net['throat.conductance'] = G
        # update A and b
        A, b = update_A_and_b(net)
        # run flow simulation
        A = js.BCSR.from_bcoo(A)  # need CSR format for linalg.spsolve!
        x = js.linalg.spsolve(A.data, A.indices, A.indptr, b, tol=1e-6)
        net['pore.pressure'] = x
        # calculate permeability
        throats = jnp.array([2])
        Q = calc_rate(net, throats)[0]  # FIXME: sum for total!
        L = (shape[0] - 1)*spacing
        mu = net['throat.viscosity'][-1]
        A = spacing**2
        K = Q * L * mu / (A * (Pin - Pout))/0.98e-12
        # calculate loss
        loss = K - K_target
        
        return loss

    # Define the gradient of f(x)
    grad_f = jax.grad(f)

    # Define the ODE system for gradient flow: dx/dt = -grad(f)
    def dxdt(t, x, args):
        return -grad_f(x)
    
    # Initial condition
    D0 = net['throat.diameter'] # Starting point for x
    t0, t1 = 0, 10  # Time span (we treat the optimization as a "time" evolution)
    # Choose an ODE solver
    solver = dfx.Tsit5()  # Tsit5 is a good general-purpose ODE solver
    # Define the ODE problem
    term = dfx.ODETerm(dxdt)
    # Solve the ODE, treating time as "iterations" for optimization
    solution = dfx.diffeqsolve(term, solver, t0=t0, t1=t1, dt0=0.1, y0=D0)
    # The final x value after "evolving" it toward the minimum
    x_min = solution.ys[-1]
    print(f"Minimum found at x = {x_min}, f(x) = {f(x_min)}")
    
    

    
    
    
    
    
    
    
    


