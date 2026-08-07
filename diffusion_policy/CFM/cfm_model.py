import torch
from diffusion_policy.CFM.utils import ModelWrapper
from torch.nn.modules import Module
from torch.nn.parallel import DistributedDataParallel


class CFMVectorField(ModelWrapper):
    def __init__(self, model: Module):
        super().__init__(model)
        self.nfe_counter = 0
        self.grad_enabled = False

    def forward(
        self, x: torch.Tensor, t: torch.Tensor,  **model_extras,
    ):
        """
        x : (B, Ta, Da)
        t : scalar or (B,)
        global_cond : (B, To, Do)
            ...
        """
        if t.ndim == 0:
            t = t.expand(x.shape[0])
        t = t.to(device=x.device, dtype=x.dtype)

        result = self.model(sample=x, timestep=t, **model_extras)

        self.nfe_counter += 1
        return result.float()

    def reset_nfe_counter(self) -> None:
        self.nfe_counter = 0

    def get_nfe(self) -> int:
        return self.nfe_counter