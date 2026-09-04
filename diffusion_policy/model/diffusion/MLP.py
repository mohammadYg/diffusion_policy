import torch
import torch.nn as nn

from diffusion_policy.model.diffusion.positional_embedding import (
    SinusoidalPosEmb
)


class MLPVectorField(nn.Module):
    """
    Fully-connected MLP alternative to ConditionalUnet1D.

    The external interface is intentionally identical to ConditionalUnet1D.

    Input:
        sample:
            [B, H, D]

        timestep:
            [B]

        global_cond:
            [B, C]

    Output:
        [B, H, D]

    Where:
        B = batch size
        H = horizon
        D = input/action dimension
        C = global conditioning dimension

    Unlike a position-wise MLP, this model flattens the entire
    trajectory [H, D] so that every output position can depend
    on every input position.
    """

    def __init__(
        self,
        input_dim,
        horizon,
        global_cond_dim=None,
        diffusion_step_embed_dim=256,
        hidden_dims=[512, 512, 512],
    ):
        super().__init__()

        self.input_dim = input_dim
        self.horizon = horizon
        self.global_cond_dim = global_cond_dim
        self.diffusion_step_embed_dim = diffusion_step_embed_dim

        # Timestep embedding
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(diffusion_step_embed_dim),
            nn.Linear(
                diffusion_step_embed_dim,
                diffusion_step_embed_dim * 4,
            ),
            nn.SiLU(),
            #nn.Mish(),
            nn.Linear(
                diffusion_step_embed_dim * 4,
                diffusion_step_embed_dim,
            ),
        )

        self.diffusion_step_encoder = diffusion_step_encoder

        # ------------------------------------------------------------
        # Input dimension
        #
        # x:
        #     [B, H, D] -> [B, H*D]
        #
        # timestep:
        #     [B, diffusion_step_embed_dim]
        #
        # global_cond:
        #     [B, global_cond_dim]
        # ------------------------------------------------------------

        cond_dim = diffusion_step_embed_dim

        if global_cond_dim is not None:
            cond_dim += global_cond_dim

        mlp_input_dim = horizon * input_dim + cond_dim

        # ------------------------------------------------------------
        # MLP
        # ------------------------------------------------------------

        layers = []
        in_dim = mlp_input_dim
        for hidden_dim in hidden_dims:
            layers.append(
                nn.Linear(in_dim, hidden_dim)
            )
            layers.append(
                nn.SiLU(),
                #nn.Mish()
            )
            in_dim = hidden_dim

        # Predict the complete trajectory
        layers.append(
            nn.Linear(
                in_dim,
                horizon * input_dim,
            )
        )
        self.net = nn.Sequential(*layers)

        # ------------------------------------------------------------
        # Parameter count
        # ------------------------------------------------------------

        num_params = sum(
            p.numel()
            for p in self.parameters()
        )

        print(
            f"MLPVectorField parameters: {num_params:e}"
        )

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        global_cond=None,
        **kwargs,
    ):
        """
        Args:
            sample:
                [B, H, D]

            timestep:
                [B]

            global_cond:
                [B, C]

        Returns:
            [B, H, D]
        """

        # ------------------------------------------------------------
        # Check input shape
        # ------------------------------------------------------------

        if sample.ndim != 3:
            raise ValueError(
                f"Expected sample to have shape [B, H, D], "
                f"got {sample.shape}"
            )

        B, H, D = sample.shape

        if H != self.horizon:
            raise ValueError(
                f"Expected horizon={self.horizon}, "
                f"got H={H}"
            )

        if D != self.input_dim:
            raise ValueError(
                f"Expected input_dim={self.input_dim}, "
                f"got D={D}"
            )

        # ------------------------------------------------------------
        # Timestep
        #
        # timestep:
        #     [B]
        #
        # diffusion_step_encoder:
        #     [B, diffusion_step_embed_dim]
        # ------------------------------------------------------------

        if not torch.is_tensor(timestep):
            raise ValueError(
                f"timestep must be a tensor, "
                f"got {type(timestep)}"
            )

        if timestep.ndim == 0:
            timestep = timestep[None]

        timestep = timestep.to(
            device=sample.device
        )

        timestep = timestep.expand(B)

        global_feature = self.diffusion_step_encoder(
            timestep
        )

        # ------------------------------------------------------------
        # Global conditioning
        # ------------------------------------------------------------

        if self.global_cond_dim is not None:

            if global_cond is None:
                global_cond = torch.zeros(
                    B,
                    self.global_cond_dim,
                    device=sample.device,
                    dtype=sample.dtype,
                )
            else:
                if global_cond.shape != (
                    B,
                    self.global_cond_dim,
                ):
                    raise ValueError(
                        f"Expected global_cond shape "
                        f"[{B}, {self.global_cond_dim}], "
                        f"got {global_cond.shape}"
                    )

            global_feature = torch.cat(
                [
                    global_feature,
                    global_cond,
                ],
                dim=-1,
            )

        # ------------------------------------------------------------
        # Flatten trajectory
        #
        # [B, H, D]
        #      ↓
        # [B, H*D]
        #
        # This means the MLP can model interactions between
        # different positions in the horizon.
        # ------------------------------------------------------------

        x = sample.reshape(
            B,
            H * D,
        )

        # ------------------------------------------------------------
        # Concatenate x_t + timestep embedding + global conditioning
        #
        # [B, H*D]
        # [B, cond_dim]
        #
        #      ↓
        #
        # [B, H*D + cond_dim]
        # ------------------------------------------------------------

        h = torch.cat(
            [
                x,
                global_feature,
            ],
            dim=-1,
        )

        # ------------------------------------------------------------
        # MLP
        # ------------------------------------------------------------

        out = self.net(h)

        # ------------------------------------------------------------
        # Restore trajectory shape
        #
        # [B, H*D]
        #      ↓
        # [B, H, D]
        # ------------------------------------------------------------

        out = out.reshape(
            B,
            H,
            D,
        )

        return out