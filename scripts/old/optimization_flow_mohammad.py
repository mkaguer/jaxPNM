"""
The goal is to write a script that fits a cubic network to have 
desired permeability using JAX and Diffrax.

Created by: Mike McKague and Mohammad Mehrnia
Date: November 18, 2024
"""

import models as mods
import jax.experimental.sparse as js
import jax.numpy as jnp

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



def build_A_and_b(net):
    
    g = net['throat.conductance']
    am = mods.create_adjacency_matrix(net, weights=g, fmt='coo')
    Np = len(net['pore.coords'])
    B = jnp.zeros(Np)
    A = am
   
    net['A'] = A
    net['B'] = B
    # laplacian
    
    # FIXME: delete this lines 
    new_ind = jnp.array([[0,0],
                        [1,1],
                        [3,3]])
    new_data = jnp.array([1,2,4])
    
    updated_ind = jnp.vstack((new_ind, A.indices))
    updateddata = jnp.concatenate([new_data, A.data])
    A = js.BCOO((updateddata, updated_ind), shape = (Np, Np))

    
    
    net['A'] = A
    net['B'] = B
    return am
'''
def get_no_bc(arr):
    no_bc = jnp.nan if arr.dtype in (float, int) else False
    return no_bc

def parse_mode(mode, allowed=None, single=False):
    r"""
    This private method is for checking the \'mode\' used in the calling
    method.

    Parameters
    ----------
    mode : str or List[str]
        The mode(s) to be parsed
    allowed : List[str]
        A list containing the allowed modes.  This list is defined by the
        calling method.  If any of the received modes are not in the
        allowed list an exception is raised.
    single : bool (default is False)
        Indicates if only a single mode is allowed.  If this argument is
        True than a string is returned rather than a list of strings, which
        makes it easier to work with in the caller method.

    Returns
    -------
    A list containing the received modes as strings, checked to ensure they
    are all within the allowed set (if provoided).  Also, if the ``single``
    argument was True, then a string is returned.

    """
    if isinstance(mode, str):
        mode = [mode]
    for item in mode:
        if (allowed is not None) and (item not in allowed):
            raise Exception('\'mode\' must be one of the following: '
                            + allowed.__str__())
    # Remove duplicates, if any
    _ = [mode.remove(L) for L in mode if mode.count(L) > 1]
    if single:
        if len(mode) > 1:
            raise Exception('Multiple modes received when only one mode '
                            + 'is allowed by this method')
        mode = mode[0]
    return mode 




def set_bc(net, pores=None, bctype=[], bcvalues=[], mode='add'):
    import numpy as np
    if not isinstance(mode, str):
        for item in mode:
            set_bc(net, pores=pores, bctype=bctype,
                        bcvalues=bcvalues, mode=item)
        return
    # If a list of bctypes was given, handle them each in order
    if len(bctype) == 0:
        bctype = net['pore.bc'].keys()
    if not isinstance(bctype, str):
        for item in bctype:
            set_bc(net, pores=pores, bctype=item,
                        bcvalues=bcvalues, mode=mode)
        return

    # Begin method
    bc_types = list(net['pore.bc'].keys())
    other_types = np.setdiff1d(bc_types, bctype).tolist()

    mode = parse_mode(
        mode,
        allowed=['overwrite', 'add', 'remove'],
        single=True)

    # Infer the value that indicates "no bc" based on array dtype
    no_bc = get_no_bc(net["pore.bc"][bctype])

    
    
    Np = len(net['pore.coords'])
    pores = jnp.array(pores)
    pores = jnp.isin(jnp.arange(Np) ,pores)

    # Deal with size of the given bcvalues
    values = jnp.array(bcvalues)
    if values.size == 1:
        values = jnp.ones_like(pores, dtype=values.dtype)*values
    # Ensure values and pores are the same size
    if values.size > 1 and values.size != pores.size:
        raise Exception('The number of values must match the number of locations')

    # Finally adjust the BCs according to mode
    if mode == 'add':
        mask = jnp.ones_like(pores, dtype=bool)  # Indices of pores to keep
        for item in net['pore.bc'].keys():  # Remove pores that are taken
            mask[jnp.isfinite(net["pore.bc"][bctype][pores])] = False
        if not jnp.all(mask):  # Raise exception if some conflicts found
            msg = "Some of the given locations already have BCs, " \
                + "either use mode='remove' first or " \
                + "use mode='overwrite' instead"
            raise Exception(msg)
        net["pore.bc"][bctype][pores[mask]] = values[mask]
    elif mode == 'overwrite':
        # Put given values in specified BC, sort out conflicts below
        net["pore.bc"][bctype][pores] = values
        # Collect indices that are present for other BCs for removal
        mask = jnp.ones_like(pores, dtype=bool)
        for item in other_types:
            net["pore.bc"][bctype][pores] = get_no_bc(net[f"pore.bc.{item}"])
            # Make a note of any BCs values of other types
            mask[jnp.isfinite(net[f'pore.bc.{item}'][pores])] = False
        
    elif mode == 'remove':
        net[f"pore.bc.{bctype}"][pores] = no_bc

'''

def apply_BC(net):
    
    """Applies specified boundary conditions by modifying A and b."""
    # rate: positive means enter or production/ a minus rate indicates consumption or exit
    
    A = net['A']
    B = net['B']
    
    if 'pore.bc.rate' in net.keys():
        
        ind = jnp.isfinite(net['pore.bc.rate'])
        B = B.at[ind].set(-net['pore.bc.rate'][ind]) # written for production/enter rate
        
    if 'pore.bc.value' in net.keys():
        Np = len(net['pore.coords'])
        x = jnp.zeros(Np)
        mask2 = A.indices[:,0] == A.indices[:,1]
        diag = A.data[mask2]
        f = diag.mean()
        # Update b (impose bc values)
        ind = jnp.isfinite(net['pore.bc.value'])
        B = B.at[ind].set(net['pore.bc.value'][ind] * f)
         
        # Update b (subtract quantities from b to keep A symmetric)
        x_BC = jnp.zeros_like(B)
        x_BC = x_BC.at[ind].set(net['pore.bc.value'][ind])
        
        temp = B[~ind] - (A @ x_BC)[~ind]
        B = B.at[~ind].set(temp)
        
        # Update A
        P_bc = jnp.where(ind)
        mask = jnp.isin(A.indices[:,0], jnp.array(P_bc)) | jnp.isin(A.indices[:,1], jnp.array(P_bc))
        # Remove entries from A for all BC rows/cols
        
        temp_A = A.data.at[mask].set(0)
        # Add diagonal entries back into A
        mask2 = A.indices[:,0] == A.indices[:,1] * jnp.isin(A.indices[:,0], jnp.array(P_bc))
        
        temp_A_2 = A.data.at[mask2].set(jnp.ones(sum(mask2), dtype=float) * f) 
        #concatenate all changes changes
        updated_data = temp_A + temp_A_2
        
        # update A
        A = js.BCOO((updated_data, A.indices), shape=(Np, Np))
        
        nonzero_mask = A.data != 0  # Identify non-zero elements
        
        A_updated = js.BCOO((A.data[nonzero_mask], A.indices[nonzero_mask]), shape=(Np, Np))
        net['A'] = A_updated
        
    net['B'] = B
    

if __name__ == "__main__":
    
    import jax.numpy as jnp
    
    # Step 1: create JAX network
    net = make_cubic_network(shape=[4, 1, 1], spacing=1e-4)
    
    # get Np and Nt
    Np = len(net['pore.coords'])
    Nt = len(net['throat.conns'])
    # net['pore.bc'] = {'rate': jnp.ones(Np) * jnp.nan,
    #                   'value': jnp.ones(Np) * jnp.nan,}
    # Step 2: add properties
    # a) add viscosity
    net['throat.viscosity'] = jnp.ones(Nt)*1e-3  # keep it constant
    # b) add diameter
    net['throat.diameter'] = jnp.ones(Nt)*5e-5  # keep it constant
    # c) calculate throat length and add!
    L = calc_throat_length(net)
    net['throat.length'] = L
    
    # Step 3: calculate conductance
    G = calc_conductance(net)
    net['throat.conductance'] = G
    
    # Step 4: build A
    am = build_A_and_b(net)
    print(am)
    net['pore.bc.value'] = jnp.ones(Np) * jnp.nan
    net['pore.bc.rate'] = jnp.ones(Np) * jnp.nan
    net['pore.bc.value'] = net['pore.bc.value'].at[0].set(1) 
    net['pore.bc.value'] = net['pore.bc.value'].at[3].set(0)
    
    # set_bc(net,pores=[1], bctype=['value'])
    apply_BC(net)
    
    
    
    
    
    


