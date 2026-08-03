import torch
import torch.nn as nn

class CFMVectorField(nn.Module):

    def __init__(self, model, global_cond=None):
        super().__init__()
        self.model = model
        self.global_cond = global_cond

    def forward(self, t, x):

        batch = x.shape[0]

        #! make sure this is compatible with the model's expected input shape
        t = torch.full(
            (batch,),
            t,
            device=x.device,
            dtype=x.dtype
        )

        return self.model(
            sample=x,
            timestep=t,
            global_cond=self.global_cond
        )