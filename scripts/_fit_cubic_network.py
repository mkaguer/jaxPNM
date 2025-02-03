import jax
import jax.numpy as jnp
import jax.experimental.sparse as js
import matplotlib.pyplot as plt
import diffrax as dfx
import mypnmlib as pnm
from jax import lax


class FitCubicNetwork:
    """
    A class for fitting a cubic network to experimental data using automatic
    differentiation in JAX
    """

    def __init__(self,
                 network,
                 K_target=None,
                 sat_target=None,
                 pressure=None,
                 spacing=None):
        r"""
        Initializes class with network attribute

        Parameters
        ----------
        network : dict
            The network dictionary containing all important data
        K_target : float
            The target permeability that we are fitting our network to
        sat_target : ndarray
            The target saturation that we are fitting our network to
        pressure : ndarray
            The pressure corresponding to the target saturation data

        """
        self.network = network
        self.K_target = K_target
        self.sat_target = sat_target
        self.pressure = pressure
        self.spacing = spacing

    def flow(self, D):
        r"""
        Runs a flow simulation on the network using diameter D

        Parameters
        ----------
        D : ndarray
            The pore diameters to run the flow simulation with
        network : dict
            The network dictionary containing all important data

        Returns
        ---------
        x : ndarray
            An Np array of resulting pressure field

        """
        # get network
        net = self.network
        # update D
        net['pore.diameter'] = D  # FIXME: not enforcine size of arrays!
        # update models that depend on D
        net['throat.diameter'] = pnm.models.throat_diameter(net)
        net['throat.hydraulic_size_factors'] = pnm.models.hydraulic_size_factor(net)
        # calculate conductance G
        G = pnm.models.generic_hydraulic(net)
        net['throat.conductance'] = G
        # build A and b
        A = pnm.simulations.build_A(net)
        b = pnm.simulations.build_b(net)
        net['A'] = A
        net['b'] = b
        # apply BCs
        A, b = pnm.simulations.apply_BC(net)
        # solve Ax = b
        A = js.BCSR.from_bcoo(A)  # need CSR format for linalg.spsolve!
        A.indptr = A.indptr.astype('int64')  # FIXME: can I remove this line?
        x = js.linalg.spsolve(A.data, A.indices, A.indptr, b, tol=1e-6)

        return x

    def find_invasion_pressure(self):

        # get network
        net = self.network
        # get Nt and Np
        Nt = len(net['throat.conns'])
        Np = len(net['pore.volume'])
        # get capillary pressure
        pc = net['throat.entry_pressure']
        # get maximum entry pressure
        max_pc = jnp.max(pc)
        # initialize invaded_throats and invaded_pores
        invaded_throats = jnp.zeros(Nt, dtype=bool)
        invaded_pores = net['pore.left']  # FIXME: update so it gets from bc

        def body_fun(i, state):
            # breakout state
            invaded_throats, invaded_pores, pressure, invasion_pressure = state
            # find throats connected to set of invaded pores
            pores = jnp.where(invaded_pores, jnp.arange(Np), -1)
            connected = pnm.models.find_neighbor_throats(net, pores)
            # find invadable throats (set of connected but not invaded)
            invadable = jnp.logical_xor(connected, invaded_throats)
            # get index of invadable throat with minimum pressure
            index = jnp.argmin(jnp.where(invadable, pc, max_pc*2))
            # update pressure
            pressure = jnp.maximum(pressure, pc[index])
            # update invasion pressire for throat
            invasion_pressure = invasion_pressure.at[index].set(pressure)
            # add newly invaded throats
            invaded_throats = invaded_throats.at[index].set(True)
            # update invaded pores, find ALL pores neighboring invaded throats!
            throats = jnp.where(invaded_throats, jnp.arange(Nt), -1)
            invaded_pores = pnm.models.find_neighbor_pores(net, throats)
            return invaded_throats, invaded_pores, pressure, invasion_pressure

        # initialize state for the for loop
        pressure = 0
        invasion_pressure = jnp.zeros(Nt)
        state0 = (invaded_throats, invaded_pores, pressure, invasion_pressure)
        # run for loop
        final_state = lax.fori_loop(0, Nt, body_fun, state0)
        # get invasion pressure from state
        invasion_pressure = final_state[-1]

        return invasion_pressure

    def run_invasion(self, D):

        # get network
        net = self.network
        # set pore diameter
        net['pore.diameter'] = D
        # regenerate geometry models
        net['throat.diameter'] = pnm.models.throat_diameter(network=net)
        net['throat.length'] = pnm.models.throat_length(network=net)
        net['pore.volume'] = pnm.models.sphere(network=net)
        net['throat.total_volume'] = pnm.models.cylinder(network=net)
        net['throat.lens_volume'] = pnm.models.lens(network=net)
        props = ['throat.total_volume', 'throat.lens_volume']
        net['throat.volume'] = pnm.models.difference(network=net, props=props)
        # add entry pressure model
        net['throat.contact_angle'] = 120
        net['throat.surface_tension'] = 0.072
        Pc = pnm.models.washburn(network=net)
        net['throat.entry_pressure'] = Pc
        # get pressure 
        pressure = self.pressure
        # get invasion pressures (Nt,)
        invasion_pressure = self.find_invasion_pressure()
        # get volumes
        vp = jnp.where(net['pore.left'], 0.0, net['pore.volume'])
        vt = net['throat.volume']
        total_volume = jnp.sum(vp) + jnp.sum(vt)

        def body_func(i, state):
            # breakout state
            saturation = state
            # apply sigmoid function
            sf = 0.01
            throat_prob = jax.nn.sigmoid((pressure[i] - invasion_pressure)/sf)
            # find pore probability
            pore_prob = pnm.models.get_max_of_neighbor_throats(net, throat_prob)
            # calculate invaded volume
            invaded_volume = jnp.sum(vp * pore_prob) + jnp.sum(vt * throat_prob)
            # calculate saturation
            sat = invaded_volume/total_volume
            saturation = saturation.at[i].set(sat)
            return saturation

        # initialize saturation
        # FIXME: minus bp vol
        saturation = jnp.zeros(len(pressure), dtype=float)
        # run for loop
        state = saturation
        sat = lax.fori_loop(0, len(pressure), body_func, state)

        return sat

    def R_squared(self, y, y_target):

        SSE = jnp.sum((y - y_target)**2)
        SST = jnp.sum((y_target - jnp.average(y_target))**2)
        R2 = 1 - SSE/SST

        return R2

    def calc_sse(self, y, y_target):

        SSE = jnp.sum((y - y_target)**2)

        return SSE

    def sat_loss(self, D):

        # get network
        net = self.network
        # run invasion simulation
        sat = self.run_invasion(D)
        # calculate SSE
        SSE = self.calc_sse(sat, self.sat_target)
        # find spacing
        spacing = self.spacing  # FIXME: spacing as attribute for cubic net
        # cons = net['throat.conns']
        # dif = jnp.diff(net['pore.coords'][cons[0]],axis=0)
        # spacing = jnp.sqrt(jnp.sum(dif**2))
        # calculate penalty
        lbd = jnp.maximum(0.0 - D, 0)
        ubd = jnp.maximum(D - spacing, 0)
        penalty = jnp.sum(lbd**2 + ubd**2) * 1e3 / spacing ** 2
        # calculate loss
        loss = SSE + penalty

        return loss

    def fit_porosimetry(self, D0, solver=dfx.Euler(), t_span=(0, 10), dt=1):

        # retrieve loss function
        f = self.sat_loss
        # Define the gradient of f(x)
        grad_f = jax.grad(f)

        # Define the ODE system for gradient flow: dx/dt = -grad(f)
        def dydt(t, y, args):
            return jnp.clip(-grad_f(y), -10.0, 10.0)

        # Time span (we treat the optimization as a "time" evolution)
        t0, t1 = t_span
        # Define the ODE problem
        term = dfx.ODETerm(dydt)
        # Solve the ODE, treating time as "iterations" for optimization
        solution = dfx.diffeqsolve(term,
                                   solver,
                                   t0=t0, t1=t1, dt0=dt, y0=D0)
        # The final x value after "evolving" it toward the minimum
        D = solution.ys[-1]
        loss = f(D)

        return D, loss

    def calc_K(self, x):

        # get network
        net = self.network
        # get spacing
        spacing = 1.0  # FIXME: add as attribute to network
        # get coords
        coords = net['pore.coords']
        # calc flow rate
        pores = net['rate_pores']
        Q = -1*pnm.simulations.rate(net, x, pores=pores)[0]
        # FIXME: add shape as network attribute
        # get length, width, and height of network
        L = jnp.max(coords[:, 0]) - jnp.min(coords[:, 0])
        w = jnp.max(coords[:, 1]) - jnp.min(coords[:, 1]) + spacing
        h = jnp.max(coords[:, 2]) - jnp.min(coords[:, 2]) + spacing
        # calculate area perpendicular to flow, assumes flow in x-direction
        A = w * h
        # get deltaP
        P1 = jnp.max(net['pore.bc.value'][net['pore.bc.mask']])
        P2 = jnp.min(net['pore.bc.value'][net['pore.bc.mask']])
        deltaP = P1 - P2
        # viscosity
        mu = net['pore.viscosity'][0]
        # calculate K
        K = mu * Q * L / A / deltaP

        return K

    def mse_loss(self, measured, target):
        r"""
        Calculates the mean squared error loss for a given target value

        Parameters
        ----------
        measured : float or array-like
            The measured value(s) to compare to the target one(s).
        target : float or array-like
            The target value(s) to compare to.

        Returns
        -------
        loss : float
            The calculated MSE loss.
        """
        # Ensure measured and target are jax.numpy arrays for consistency
        measured = jnp.asarray(measured)
        target = jnp.asarray(target)
        # Calculate MSE loss (supports scalar or array inputs)
        mse = jnp.mean((measured - target) ** 2)

        return mse

    def K_loss(self, D, net):
        # FIXME: remove net from args?
        # run flow simulation
        x = self.flow(D)
        # calculate permeability
        K = self.calc_K(x)
        # calculate loss
        loss = self.mse_loss(K, self.K_target)
        # get spacing
        spacing = self.spacing
        # add penalty to loss
        lbd = jnp.maximum(0.0 - D, 0)  # keep D > 0
        ubd = jnp.maximum(D - spacing, 0)  # Keep D < 1
        penalty = jnp.sum(lbd**2 + ubd**2) / spacing ** 2
        # Add penalty to loss
        loss += penalty
        return loss

    def fit_K(self, D0, solver=dfx.Euler(), t_span=(0, 10), dt=1):
        r"""
        This method fits the diameters of a cubic network, assuming spheres
        and cylinders geometry, to a target permeability.

        Parameters
        ----------
        D0 : ndarray
            The array of initial diameters to start integration from
        solver : diffrax solver object
            The diffrax solver to use. Two common options are: dfx.Euler() and
            dfx.Tsit5(). The default is Euler which uses a fixed timestep!
        t_span : tuple
            A tuple of start and end times for integration
        dt : float
            Time step required for integration, depending on solver it may be
            just the initial step or the step used throughout.

        Returns
        -------
        D : ndarray
            The solved for diameters that
        loss : ndarray
            The resulting loss

        """
        # get network
        net = self.network
        # retrieve loss function
        f = self.K_loss
        # Define the gradient of f(x)
        grad_f = jax.grad(f)

        # Define the ODE system for gradient flow: dx/dt = -grad(f)
        def dydt(t, y, net):
            return -grad_f(y, net)

        # Time span (we treat the optimization as a "time" evolution)
        t0, t1 = t_span
        # Define the ODE problem
        term = dfx.ODETerm(dydt)
        # Solve the ODE, treating time as "iterations" for optimization
        solution = dfx.diffeqsolve(term,
                                   solver,
                                   t0=t0, t1=t1, dt0=dt, y0=D0, args=net)
        # The final x value after "evolving" it toward the minimum
        D = solution.ys[-1]
        loss = f(D, net)

        return D, loss

    def plot_loss(self, y0, index=0, N=100):
        r"""
        This function allows you to plot the loss of one parameter at a time!

        Parameters
        ----------
        y0 : ndarray
            The array of ALL parameters, normally diameters.
        K_target : float
            The target permeability that we are fitting our network to
        index : int
            The index of of the parameter to change (default is 0)
        N : int
            The number of different parameter values to try in a range from
            0 to 1 (default is 100).

        """
        # retrieve network
        net = self.network
        # retrieve loss function
        f = self.K_loss
        # calculate loss for each D
        D_vals = jnp.linspace(0.01, 1.0, N)
        losses = jnp.array([f(y0.at[index].set(D), net) for D in D_vals])
        # plot D_vals versus loss
        plt.plot(D_vals, losses)
        plt.xlabel(f"D[{index}]")
        plt.ylabel("Loss")
        plt.title(f"Loss Landscape Along D[{index}]")
        plt.show()
