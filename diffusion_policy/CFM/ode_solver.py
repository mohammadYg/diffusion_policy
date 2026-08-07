
from typing import Callable, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor
from torchdiffeq import odeint

from diffusion_policy.CFM.solver import Solver
from diffusion_policy.CFM.utils import gradient, ModelWrapper


class ODESolver(Solver):
    """
    ODE solver for continuous Flow Matching policies.

    The model represents a velocity field

        dx/dt = v_theta(x, t, obs)

    where x is an action trajectory of shape (B, H, A).
    """

    def __init__(self, velocity_model: Union[ModelWrapper, Callable]):
        super().__init__()
        self.velocity_model = velocity_model

    # ---------------------------------------------------------
    # Sampling
    # ---------------------------------------------------------
    def sample(
        self,
        x_init: Tensor,
        time_grid: Tensor,
        method: str = "dopri5",
        step_size: Optional[float] = None,
        atol: float = 1e-5,
        rtol: float = 1e-5,
        return_intermediates: bool = False,
        enable_grad: bool = False,
        **model_extras,
    ) -> Union[Tensor, Sequence[Tensor]]:

        """
        Args:
            x_init (Tensor): initial conditions (e.g., source samples :math:`X_0 \sim p`). Shape: [batch_size, ...].
            step_size (Optional[float]): The step size. Must be None for adaptive step solvers.
            method (str): A method supported by torchdiffeq. Defaults to "euler". Other commonly used solvers are "dopri5", "midpoint" and "heun3". For a complete list, see torchdiffeq.
            atol (float): Absolute tolerance, used for adaptive step solvers.
            rtol (float): Relative tolerance, used for adaptive step solvers.
            time_grid (Tensor): The process is solved in the interval [min(time_grid, max(time_grid)] and if step_size is None then time discretization is set by the time grid. May specify a descending time_grid to solve in the reverse direction. Defaults to torch.tensor([0.0, 1.0]).
            return_intermediates (bool, optional): If True then return intermediate time steps according to time_grid. Defaults to False.
            enable_grad (bool, optional): Whether to compute gradients during sampling. Defaults to False.
            **model_extras: Additional input for the model.
        
        Returns:
            Union[Tensor, Sequence[Tensor]]: The last timestep when return_intermediates=False, otherwise all values specified in time_grid.
        """
        time_grid = time_grid.to(x_init.device)

        def ode_func(t, x):
            return self.velocity_model(x=x, t=t, **model_extras)

        ode_opts = {"step_size": step_size} if step_size is not None else {}

        with torch.set_grad_enabled(enable_grad):
            traj = odeint(
                ode_func,
                x_init,
                time_grid,
                method=method,
                options=ode_opts,
                atol=atol,
                rtol=rtol,
            )

        if return_intermediates:
            return traj
        else:
            return traj[-1]

    # ---------------------------------------------------------
    # Exact likelihood
    # ---------------------------------------------------------
    def compute_likelihood(
        self,
        x_1: Tensor,
        log_p0: Callable[[Tensor], Tensor],
        time_grid: Tensor = torch.tensor([1.0, 0.0]),
        method: str = "dopri5",
        step_size: Optional[float] = None,
        atol: float = 1e-5,
        rtol: float = 1e-5,
        exact_divergence: bool = False,
        return_intermediates: bool = False,
        enable_grad: bool = False,
        **model_extras,
    ) -> Union[Tuple[Tensor, Tensor], Tuple[Sequence[Tensor], Tensor]]:

        r"""Solve for log likelihood given a target sample at :math:`t=0`.

        Works similarly to sample, but solves the ODE in reverse to compute the log-likelihood. The velocity model must be differentiable with respect to x.
        The function assumes log_p0 is the log probability of the source distribution at :math:`t=0`.

        Args:
            x_1 (Tensor): target sample (e.g., samples :math:`X_1 \sim p_1`).
            log_p0 (Callable[[Tensor], Tensor]): Log probability function of the source distribution.
            step_size (Optional[float]): The step size. Must be None for adaptive step solvers.
            method (str): A method supported by torchdiffeq. Defaults to "euler". Other commonly used solvers are "dopri5", "midpoint" and "heun3". For a complete list, see torchdiffeq.
            atol (float): Absolute tolerance, used for adaptive step solvers.
            rtol (float): Relative tolerance, used for adaptive step solvers.
            time_grid (Tensor): If step_size is None then time discretization is set by the time grid. Must start at 1.0 and end at 0.0, otherwise the likelihood computation is not valid. Defaults to torch.tensor([1.0, 0.0]).
            return_intermediates (bool, optional): If True then return intermediate time steps according to time_grid. Otherwise only return the final sample. Defaults to False.
            exact_divergence (bool): Whether to compute the exact divergence or use the Hutchinson estimator.
            enable_grad (bool, optional): Whether to compute gradients during sampling. Defaults to False.
            **model_extras: Additional input for the model.

        Returns:
            Union[Tuple[Tensor, Tensor], Tuple[Sequence[Tensor], Tensor]]: Samples at time_grid and log likelihood values of given x_1.
        """

        assert (
            time_grid[0] == 1.0 and time_grid[-1] == 0.0
        ), "Likelihood requires reverse integration from t=1 to t=0."

        time_grid = time_grid.to(x_1.device)

        # Hutchinson noise for divergence estimation
        if not exact_divergence:
            z = torch.randint_like(x_1, low=0, high=2).float() * 2 - 1

        def ode_func(t, x):
            return self.velocity_model(x=x, t=t, **model_extras)

        def dynamics_func(t, states):

            xt, log_det = states

            with torch.set_grad_enabled(True):
                xt = xt.requires_grad_(True)
                vt = ode_func(t, xt)

                if exact_divergence:
                    flat_v = vt.flatten(start_dim=1)
                    div = torch.zeros(
                        xt.shape[0],
                        device=xt.device,
                    )
                    for i in range(flat_v.shape[1]):
                        grad_i = gradient(flat_v[:, i], xt, retain_graph=True)
                        div += grad_i.flatten(start_dim=1)[:, i]

                else:

                    vt_dot_z = (vt * z).flatten(start_dim=1).sum(dim=1)
                    grad_vt_dot_z = gradient(vt_dot_z, xt)
                    div = (grad_vt_dot_z * z).flatten(start_dim=1).sum(dim=1)

            if enable_grad:
                return vt, div
            else:
                # detach the outputs to save memory and avoid computing gradients
                return vt.detach(), div.detach()

        y0 = (x_1, torch.zeros(x_1.shape[0], device=x_1.device))
        ode_opts = {"step_size": step_size} if step_size is not None else {}

        #with torch.set_grad_enabled(enable_grad):
        traj, log_det = odeint(
            dynamics_func,
            y0,
            time_grid,
            method=method,
            options=ode_opts,
            atol=atol,
            rtol=rtol,
        )

        x0 = traj[-1]

        log_p0_val = log_p0(x0.flatten(start_dim=1))
        log_px = log_p0_val + log_det[-1]

        if return_intermediates:
            return traj, log_px
        else:
            return x0, log_px