from abc import ABC
import torch
from torch import nn, Tensor
from typing import Optional

def skewed_timestep_sample(num_samples: int, device: torch.device) -> torch.Tensor:
    P_mean = -1.2
    P_std = 1.2
    rnd_normal = torch.randn((num_samples,), device=device)
    sigma = (rnd_normal * P_std + P_mean).exp()
    time = 1 / (1 + sigma)
    time = torch.clip(time, min=0.0001, max=1.0)
    return time

class ModelWrapper(ABC, nn.Module):
    """
    This class is used to wrap around another model, adding custom forward pass logic.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: Tensor, t: Tensor, **extras) -> Tensor:
        r"""
        This method defines how inputs should be passed through the wrapped model.
        Here, we're assuming that the wrapped model takes both :math:`x` and :math:`t` as input,
        along with any additional keyword arguments.

        Optional things to do here:
            - check that t is in the dimensions that the model is expecting.
            - add a custom forward pass logic.
            - call the wrapped model.

        | given x, t
        | returns the model output for input x at time t, with extra information `extra`.

        Args:
            x (Tensor): input data to the model (batch_size, ...).
            t (Tensor): time (batch_size).
            **extras: additional information forwarded to the model, e.g., text condition.

        Returns:
            Tensor: model output.
        """
        return self.model(x=x, t=t, **extras)
    

def gradient(
    output: Tensor,
    x: Tensor,
    grad_outputs: Optional[Tensor] = None,
    create_graph: bool = False,
    retain_graph: bool = False,
) -> Tensor:
    """
    Compute the gradient of the inner product of output and grad_outputs w.r.t :math:`x`.

    Args:
        output (Tensor): [N, D] Output of the function.
        x (Tensor): [N, d_1, d_2, ... ] input
        grad_outputs (Optional[Tensor]): [N, D] Gradient of outputs, if `None`,
            then will use a tensor of ones
        create_graph (bool): If True, graph of the derivative will be constructed, allowing
            to compute higher order derivative products. Defaults to False.
    Returns:
        Tensor: [N, d_1, d_2, ... ]. the gradient w.r.t x.
    """

    if grad_outputs is None:
        grad_outputs = torch.ones_like(output).detach()
    grad = torch.autograd.grad(
        output,
        x,
        grad_outputs=grad_outputs,
        create_graph=create_graph,
        retain_graph=retain_graph,
    )[0]
    return grad