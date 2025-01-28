import jax.numpy as jnp

__all__ = ["calc_throat_length",
           "calc_conductance",
           "throat_diameter",
           "hydraulic_size_factor", 
           "generic_hydraulic",
           "washburn",
           "find_neighbor_throats",
           "find_neighbor_pores",
           "throat_length",
           "sphere",
           "cylinder",
           "lens",
           "difference", 
           "get_max_of_neighbor_throats"]


def calc_throat_length(network):

    conns = network['throat.conns']
    coords = network['pore.coords'][conns]
    diff = jnp.diff(coords, axis=1)
    throat_length = jnp.linalg.norm(diff, axis=2)[:, 0]

    return throat_length


def calc_conductance(network):

    R = network['throat.diameter'] / 2
    mu = network['throat.viscosity']
    L = network['throat.length']

    return jnp.pi * R ** 4 / 8 / mu / L


def throat_diameter(network):

    D = network['pore.diameter']
    conns = network['throat.conns']

    return 0.5 * jnp.min(jnp.abs(D[conns]), axis=1)


def hydraulic_size_factor(network,
                          pore_diameter="pore.diameter",
                          throat_diameter="throat.diameter"):
    r"""
    Computes hydraulic size factors for conduits assuming pores are
    spheres and throats are cylinders.

    Parameters
    ----------
    %(networkwork)s
    %(Dp)s
    %(Dt)s

    Returns
    -------
    size_factors : ndarray
        Array (Nt by 3) containing conduit values for each element
        of the pore-throat-pore conduits. The array is formatted as
        ``[pore1, throat, pore2]``.

    Notes
    -----
    The hydraulic size factor is the geometrical part of the pre-factor in
    Stoke's flow:

    .. math::

        Q = \frac{A^2}{8 \pi \mu L} \Delta P
          = \frac{S_{hydraulic}}{\mu} \Delta P

    Thus :math:`S_{hydraulic}` represents the combined effect of the area
    and length of the *conduit*, which consists of a throat and 1/2 of the
    pores on each end.
    """
    D1, Dt, D2 = _get_conduit_data(network, pore_diameter.split('.', 1)[-1]).T
    L1, Lt, L2 = spheres_and_cylinders(network=network,
                                       pore_diameter=pore_diameter,
                                       throat_diameter=throat_diameter).T
    
    # Fi is the integral of (1/A^2) dx, x = [0, Li]
    a = 4 / (D1**3 * jnp.pi**2)
    b = 2 * D1 * L1 / (D1**2 - 4 * L1**2) + jnp.arctanh(2 * L1 / D1)
    F1 = a * b
    a = 4 / (D2**3 * jnp.pi**2)
    b = 2 * D2 * L2 / (D2**2 - 4 * L2**2) + jnp.arctanh(2 * L2 / D2)
    F2 = a * b
    Ft = Lt / (jnp.pi / 4 * Dt**2)**2

    # I is the integral of (y^2 + z^2) dA, divided by A^2
    I1 = I2 = It = 1 / (2 * jnp.pi)

    # S is 1 / (16 * pi^2 * I * F)
    S1 = 1 / (16 * jnp.pi**2 * I1 * F1)
    St = 1 / (16 * jnp.pi**2 * It * Ft)
    S2 = 1 / (16 * jnp.pi**2 * I2 * F2)
    
    return jnp.vstack([S1, St, S2]).T


def generic_hydraulic(
    network,
    pore_viscosity='pore.viscosity',
    throat_viscosity='throat.viscosity',
    size_factors='throat.hydraulic_size_factors'
):
    r"""
    Calculates the hydraulic conductance of conduits in network.

    Parameters
    ----------
    %(phase)s
    pore_viscosity : str
        %(dict_blurb)s pore viscosity
    throat_viscosity : str
        %(dict_blurb)s throat viscosity
    size_factors : str
        %(dict_blurb)s conduit hydraulic size factors

    Returns
    -------
    %(return_arr)s hydraulic conductance

    """
    conns = network['throat.conns']
    mu1, mu2 = network[pore_viscosity][conns].T
    mut = network[throat_viscosity]

    SF = network[size_factors]
    if isinstance(SF, dict):  # Legacy approach
        F1, Ft, F2 = SF.values()
    elif SF.ndim > 1:  # Nt-by-3 array
        F1, Ft, F2 = SF.T
    else:  # Nt array, like from network extraction predictions
        F1, Ft, F2 = jnp.inf, SF, jnp.inf

    g1 = F1 / mu1
    gt = Ft / mut
    g2 = F2 / mu2
    return 1 / (1/g1 + 1/gt + 1/g2)


def spheres_and_cylinders(
    network,
    pore_diameter="pore.diameter",
    throat_diameter="throat.diameter",
):
    r"""
    Calculates conduit lengths in the network assuming pores are spheres
    and throats are cylinders.

    A conduit is defined as ( 1/2 pore - full throat - 1/2 pore ).

    Parameters
    ----------
    %(network)s
    %(Dp)s
    %(Dt)s

    Returns
    -------
    lengths : ndarray
        Array (Nt by 3) containing conduit values for each element
        of the pore-throat-pore conduits. The array is formatted as
        ``[pore1, throat, pore2]``.

    """
    L_ctc = _get_L_ctc(network)
    D1, Dt, D2 = _get_conduit_data(network, pore_diameter.split(".", 1)[-1]).T

    # Handle the case where Dt > Dp
    # if (Dt > D1).any() or (Dt > D2).any():
    #     _raise_incompatible_data()

    # If spheres do not overlap:
    L1 = jnp.sqrt(D1**2 - Dt**2) / 2
    L2 = jnp.sqrt(D2**2 - Dt**2) / 2
    Lt = L_ctc - (L1 + L2)

    # if jnp.any(Lt < 0):  # Find pores that touch/overlap
    #     # Find diameter of overlap between spheres
    #     d = L_ctc
    #     R1 = D1/2
    #     R2 = D2/2
    #     # Check distance to the intersection
    #     L1_int = (d**2 - R2**2 + R1**2) / (2*d)
    #     if jnp.any(L1_int < 0) or jnp.any(L1_int > L_ctc):
    #         raise Exception('The pores overlap too much')
    #     D_int = 2/(2*d) * jnp.sqrt(4*d**2 * R1**2 - (d**2 - R2**2 + R1**2)**2)
    #     mask = D_int > Dt
    #     if jnp.any(mask):
    #         d = d[mask]
    #         R1 = R1[mask]
    #         R2 = R2[mask]
    #         L1[mask] = (d**2 - R2**2 + R1**2) / (2*d)
    #         L2[mask] = L_ctc[mask] - L1[mask]
    #         Lt[mask] = 1e-15

    return jnp.vstack((L1, Lt, L2)).T


def _get_conduit_data(network, propname):
    r"""
    Fetches an Nt-by-3 array of the requested property

    Parameters
    ----------
    propname : str
        The dictionary key of the property to fetch.

    Returns
    -------
    data : ndarray
        An Nt-by-3 array with each column containing the requrested data
        for pore1, throat, and pore2 respectively.

    """
    poreprop = 'pore.' + propname.split('.', 1)[-1]
    throatprop = 'throat.' + propname.split('.', 1)[-1]
    conns = network['throat.conns']
    Nt = len(network['throat.conns'])
    try:
        T = network[throatprop]
        if T.ndim > 1:
            raise Exception(f'{throatprop} must be a single column wide')
    except KeyError:
        T = jnp.ones([Nt, ], dtype=float)*jnp.nan
    try:
        P1, P2 = network[poreprop][conns.T]
    except KeyError:
        P1 = jnp.ones([Nt, ], dtype=float)*jnp.nan
        P2 = jnp.ones([Nt, ], dtype=float)*jnp.nan
    vals = jnp.vstack((P1, T, P2)).T

    return vals


# Dealing with errors and exceptions
def _raise_incompatible_data():
    msg = (
        "'spheres_and_cylinders' can only be applied when throat diameter is"
        " smaller than that of adjacent pores."
    )
    raise Exception(msg)


def _get_L_ctc(network):
    """Returns throat spacing if it exists, otherwise calculates it."""
    try:
        L_ctc = network["throat.spacing"]
    except KeyError:
        P12 = network["throat.conns"]
        C1 = network["pore.coords"][P12[:, 0]]
        C2 = network["pore.coords"][P12[:, 1]]
        L_ctc = jnp.linalg.norm(C1 - C2, axis=1)
    return L_ctc


def washburn(network):
    r"""
    Computes the capillary entry pressure assuming the throat in a
    cylindrical tube.

    Parameters
    ----------
    %(phase)s
    surface_tension : str
        %(dict_blurb)s surface tension. If a pore property is given, it is
        interpolated to a throat list.
    contact_angle : str
        %(dict_blurb)s contact angle. If a pore property is given, it is
        interpolated to a throat list.
    diameter : str
        %(dict_blurb)s throat diameter

    Returns
    -------
    %(return_arr)s capillary entry pressure

    Notes
    -----
    The Washburn equation is:

    .. math::
        P_c = -\frac{2\sigma(cos(\theta))}{r}

    This is the most basic approach to calculating entry pressure and is
    suitable for highly non-wetting invading phases in most materials.

    """
    sigma = network['pore.surface_tension']
    theta = network['pore.contact_angle']
    r = network['throat.diameter'] / 2
    value = -2 * sigma * jnp.cos(jnp.radians(theta)) / r
    # if diameter.split(".")[0] == "throat":
    #     pass
    # else:
    #     value = value[phase.pores()]
    # value[jnp.absolute(value) == jnp.inf] = 0 # fix this later
    return value


def find_neighbor_throats(network, pores):

    conns = network['throat.conns']
    throats = jnp.any(jnp.isin(conns, pores), axis=1)

    return throats


def find_neighbor_pores(network, throats):

    # retrieve adjacency matrix
    am = network['adjacency_matrix']  # FIXME: requires am with Nt + 1 as weights
    # get data
    data = am.data - 1  # subtract one to get back 0
    # get start/end indices in data for each row
    start = am.indptr[:-1]
    end = am.indptr[1:]
    # get boolean array of throats we want from data
    mask = jnp.isin(data, throats)
    # get booleam array of pores connected to any throats
    N = len(data)
    r = jnp.arange(N)
    pores = [jnp.any(((r >= s) & (r < e)) * mask) for s, e in zip(start, end)]
    pores = jnp.array(pores)

    return pores


def get_max_of_neighbor_throats(network, throat_prob):

    # retrieve adjacency matrix
    am = network['adjacency_matrix']  # FIXME: requires am with Nt + 1 as weights
    # get data
    data = am.data - 1  # subtract one to get back 0
    data = throat_prob[data]
    # get start/end indices in data for each row
    start = am.indptr[:-1]
    end = am.indptr[1:]
    # get booleam array of pores connected to any throats
    N = len(data)
    r = jnp.arange(N)
    mx = [jnp.max(((r >= s) & (r < e)) * data) for s, e in zip(start, end)]
    mx = jnp.array(mx)

    return mx


def throat_length(network,
                  pore_diameter='pore.diameter',
                  throat_diameter='throat.diameter'):
    r"""
    Finds throat length assuming pores are spheres and throats are
    cylinders.

    Parameters
    ----------
    network : dict
        The network dictionary
    pore_diameter : str
        The dictionary key used to fetch pore diameter
    throat_diameter : str
        The dictionary key used to fetch throat diameter

    Returns
    -------
    lengths : ndarray
        A numpy ndarray containing throat length values

    """
    out = spheres_and_cylinders(network=network,
                                pore_diameter=pore_diameter,
                                throat_diameter=throat_diameter)
    return out[:, 1]


def sphere(network, pore_diameter='pore.diameter'):
    r"""
    Calculate pore volume from diameter assuming a spherical pore body

    Parameters
    ----------
    network : dict
        The network dictionary
    pore_diameter : str
        The dictionary key used to fetch pore diameter

    Returns
    -------
    volumes : ndarray
        Numpy ndarray containing pore volume values

    """
    return 4/3*jnp.pi*(network[pore_diameter]/2)**3


def cylinder(network,
             throat_diameter='throat.diameter',
             throat_length='throat.length'):
    r"""
    Calculate throat volume assuing a cylindrical shape

    Parameters
    ----------
    %(network)s
    %(Dt)s
    %(Lt)s

    Returns
    -------
    volumes : ndarray
        A numpy ndarray containing throat volume values

    Notes
    -----
    This models does not account for the volume reprsented by the
    intersection of the throat with a spherical pore body.  Use the ``lens``
    or ``pendular_ring`` models in addition to this one to account for this
    volume.

    """
    leng = network[throat_length]
    diam = network[throat_diameter]
    value = jnp.pi/4*leng*diam**2
    return value


def lens(network,
         throat_diameter='throat.diameter',
         pore_diameter='pore.diameter'):
    r"""
    Calculates the volume residing the hemispherical caps formed by the
    intersection between cylindrical throats and spherical pores.

    This volume should be subtracted from throat volumes if the throat lengths
    were found using throat end points.

    Parameters
    ----------
    %(network)s
    %(Dt)s
    %(Dp)s

    Returns
    -------

    Notes
    -----
    This model does not consider the possibility that multiple throats might
    overlap in the same location which could happen if throats are large and
    connectivity is random.

    See Also
    --------
    pendular_ring
    """
    conns = network['throat.conns']
    Rp = network[pore_diameter]/2
    Rt = network[throat_diameter]/2
    a = jnp.atleast_2d(Rt).T
    q = jnp.arcsin(a/Rp[conns])
    b = Rp[conns]*jnp.cos(q)
    h = Rp[conns] - b
    V = 1/6*jnp.pi*h*(3*a**2 + h**2)
    return jnp.sum(V, axis=1)


def difference(network, props):
    r"""
    Subtracts elements 1:N in `props` from element 0

    Parameters
    ----------
    network : dict
        The network dictionary
    props : list
        A list of dict keys containing the values to operate on.  If the first
        element is A, and the next are B and C, then the results is A - B - C.
    """
    A = network[props[0]]
    for B in props[1:]:
        A = A - network[B]
    return A


def washburn(network,
             surface_tension="throat.surface_tension",
             contact_angle="throat.contact_angle",
             diameter="throat.diameter"):
    r"""
    Computes the capillary entry pressure assuming the throat in a
    cylindrical tube.

    Parameters
    ----------
    %(phase)s
    surface_tension : str
        %(dict_blurb)s surface tension. If a pore property is given, it is
        interpolated to a throat list.
    contact_angle : str
        %(dict_blurb)s contact angle. If a pore property is given, it is
        interpolated to a throat list.
    diameter : str
        %(dict_blurb)s throat diameter

    Returns
    -------
    %(return_arr)s capillary entry pressure

    Notes
    -----
    The Washburn equation is:

    .. math::
        P_c = -\frac{2\sigma(cos(\theta))}{r}

    This is the most basic approach to calculating entry pressure and is
    suitable for highly non-wetting invading phases in most materials.

    """
    sigma = network[surface_tension]
    theta = network[contact_angle]
    r = network[diameter] / 2
    value = -2 * sigma * jnp.cos(jnp.radians(theta)) / r
    if diameter.split(".")[0] == "throat":
        pass
    else:
        raise ValueError("Invalid input, use throat.diameter")
    # FIXME: commented out b/c r is not zero!
    # idx = jnp.absolute(value) == jnp.inf
    # value = value.at[idx].set(0)
    return value