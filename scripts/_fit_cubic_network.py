import jax
import jax.numpy as jnp
import jax.experimental.sparse as js
import matplotlib.pyplot as plt
import diffrax as dfx
import mypnmlib as pnm


class FitCubicNetwork:
    """
    A class for fitting a cubic network to experimental data using automatic
    differentiation in JAX
    """

    def __init__(self, network, K_target):
        r"""
        Initializes class with network attribute

        Parameters
        ----------
        network : dict
            The network dictionary containing all important data
        K_target : float
            The target permeability that we are fitting our network to

        """
        self.network = network
        self.K_target = K_target

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

    def porosimetry():
        ...
        return

    def fit_porosimetry():
        ...
        return

    def calc_K(self, x):

        # get network
        net = self.network
        # calc flow rate
        pores = net['rate_pores']
        Q = -1*pnm.simulations.rate(net, x, pores=pores)[0]
        K = Q  # FIXME: we need to actually calculate K!

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
        # run flow simulation
        x = self.flow(D)
        # calculate permeability
        K = self.calc_K(x)
        # calculate loss
        loss = self.mse_loss(K, self.K_target)
        # add penalty to loss
        lbd = jnp.maximum(0.0 - D, 0)  # keep D > 0
        ubd = jnp.maximum(D - 1.0, 0)  # Keep D < 1
        penalty = jnp.sum(lbd**2 + ubd**2)
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
