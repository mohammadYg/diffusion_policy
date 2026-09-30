"""
Variant 11: unlike prob1-10 (which couple "is this block Bayesian" for conv
weights and FiLM conditioning together), here they're decoupled.

- Down/up-sampling backbone (down_modules/up_modules conv blocks, residual_conv,
  Downsample1d/Upsample1d): deterministic. These carry generic temporal
  features and the residual identity path - kept clean/cheap.
- FiLM conditioning (every residual block, down/mid/up): Bayesian. This is
  where the observation enters the network - state-conditional epistemic
  uncertainty should live here.
- Mid (bottleneck) modules: fully Bayesian (conv + FiLM) - most abstract,
  task-specific representation.
- final_conv: fully Bayesian (last layer matters most for output uncertainty).
- Timestep encoder: deterministic (no epistemic uncertainty over a known,
  uniformly-sampled index).
"""
import torch
import torch.nn as nn

from einops.layers.torch import Rearrange
from typing import Union

import einops
from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb
from diffusion_policy.model.diffusion.bayes_conv1d_components import (
    ProbConv1d, ProbConv1dBlock, ProbLinear
)
from diffusion_policy.model.diffusion.conv1d_components import (
    Downsample1d, Upsample1d, Conv1dBlock)


class ProbConditionalResidualBlock1D(nn.Module):
    """Fully Bayesian residual block (conv + FiLM) - used for mid_modules."""

    def __init__(
        self,
        in_channels,
        out_channels,
        cond_dim,
        kernel_size=3,
        n_groups=8,
        cond_predict_scale=False,
        rho_post=-3.0,
        rho_prior=-3.0,
        prior_dist='gaussian',
        init_post='random',
        init_prior='random',
        post_sigma_scale=None,
        prior_sigma_scale=None,
        local_reparam=True
    ):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                ProbConv1dBlock(
                    in_channels, out_channels, kernel_size,
                    n_groups=n_groups, rho_post=rho_post, rho_prior=rho_prior,
                    prior_dist=prior_dist, init_post=init_post, init_prior=init_prior,
                    post_sigma_scale=post_sigma_scale, prior_sigma_scale=prior_sigma_scale,
                    local_reparam=local_reparam
                ),
                ProbConv1dBlock(
                    out_channels, out_channels, kernel_size,
                    n_groups=n_groups, rho_post=rho_post, rho_prior=rho_prior,
                    prior_dist=prior_dist, init_post=init_post, init_prior=init_prior,
                    post_sigma_scale=post_sigma_scale, prior_sigma_scale=prior_sigma_scale,
                    local_reparam=local_reparam
                ),
            ]
        )

        # FiLM modulation https://arxiv.org/abs/1709.07871
        cond_channels = out_channels
        if cond_predict_scale:
            cond_channels = out_channels * 2
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels

        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            ProbLinear(
                cond_dim, cond_channels,
                rho_post=rho_post, rho_prior=rho_prior, prior_dist=prior_dist,
                init_post=init_post, init_prior=init_prior,
                post_sigma_scale=post_sigma_scale, prior_sigma_scale=prior_sigma_scale,
                local_reparam=local_reparam
            ),
            Rearrange("batch t -> batch t 1"),
        )

        if in_channels != out_channels:
            self.residual_conv = ProbConv1d(
                in_channels, out_channels, kernel_size=1, rho_post=rho_post,
                rho_prior=rho_prior, prior_dist=prior_dist, init_post=init_post,
                init_prior=init_prior, padding=0,
                post_sigma_scale=post_sigma_scale, prior_sigma_scale=prior_sigma_scale,
                local_reparam=local_reparam
            )
        else:
            self.residual_conv = nn.Identity()

    def sample_weights(self):
        self.blocks[0].sample_weights()
        self.blocks[1].sample_weights()
        self.cond_encoder[1].sample_weights()
        self.residual_conv.sample_weights() if not isinstance(self.residual_conv, nn.Identity) else None

    def clear_sample(self):
        self.blocks[0].clear_sample()
        self.blocks[1].clear_sample()
        self.cond_encoder[1].clear_sample()
        self.residual_conv.clear_sample() if not isinstance(self.residual_conv, nn.Identity) else None

    def forward(self, x, cond, stochastic=False):
        out = self.blocks[0](x, stochastic=stochastic)

        embed = self.cond_encoder[0](cond)
        embed = self.cond_encoder[1](embed, stochastic=stochastic)
        embed = self.cond_encoder[2](embed)

        if self.cond_predict_scale:
            embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
            scale = embed[:, 0, ...]
            bias = embed[:, 1, ...]
            out = scale * out + bias
        else:
            out = out + embed

        out = self.blocks[1](out, stochastic=stochastic)

        if isinstance(self.residual_conv, nn.Identity):
            residual = self.residual_conv(x)
        else:
            residual = self.residual_conv(x, stochastic=stochastic)
        return out + residual

    def compute_kl(self):
        kl_div = self.blocks[0].block[0].kl_div + self.blocks[1].block[0].kl_div
        kl_div += self.cond_encoder[1].kl_div
        if not isinstance(self.residual_conv, nn.Identity):
            kl_div += self.residual_conv.kl_div
        return kl_div


class MixedConditionalResidualBlock1D(nn.Module):
    """Deterministic conv + residual, Bayesian FiLM - used for down/up modules."""

    def __init__(
        self,
        in_channels,
        out_channels,
        cond_dim,
        kernel_size=3,
        n_groups=8,
        cond_predict_scale=False,
        rho_post=-3.0,
        rho_prior=-3.0,
        prior_dist='gaussian',
        init_post='random',
        init_prior='random',
        post_sigma_scale=None,
        prior_sigma_scale=None,
        local_reparam=True
    ):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        cond_channels = out_channels
        if cond_predict_scale:
            cond_channels = out_channels * 2
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels

        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            ProbLinear(
                cond_dim, cond_channels,
                rho_post=rho_post, rho_prior=rho_prior, prior_dist=prior_dist,
                init_post=init_post, init_prior=init_prior,
                post_sigma_scale=post_sigma_scale, prior_sigma_scale=prior_sigma_scale,
                local_reparam=local_reparam
            ),
            Rearrange("batch t -> batch t 1"),
        )

        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def sample_weights(self):
        self.cond_encoder[1].sample_weights()

    def clear_sample(self):
        self.cond_encoder[1].clear_sample()

    def forward(self, x, cond, stochastic=False):
        out = self.blocks[0](x)

        embed = self.cond_encoder[0](cond)
        embed = self.cond_encoder[1](embed, stochastic=stochastic)
        embed = self.cond_encoder[2](embed)

        if self.cond_predict_scale:
            embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
            scale = embed[:, 0, ...]
            bias = embed[:, 1, ...]
            out = scale * out + bias
        else:
            out = out + embed

        out = self.blocks[1](out)
        return out + self.residual_conv(x)

    def compute_kl(self):
        return self.cond_encoder[1].kl_div


class BayesianConditionalUnet1D(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim=None,
        global_cond_dim=None,
        diffusion_step_embed_dim=256,
        down_dims=[256, 512, 1024],
        kernel_size=3,
        n_groups=8,
        cond_predict_scale=False,
        rho_post=-3.0,
        rho_prior=-3.0,
        prior_dist='gaussian',
        init_post='random',
        init_prior='zeros',
        post_sigma_scale=None,
        prior_sigma_scale=None,
        local_reparam=True,
    ):
        super().__init__()

        if output_dim is None:
            output_dim = input_dim

        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )

        cond_dim = dsed
        if global_cond_dim is not None:
            cond_dim += global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]

        bayes_kwargs = dict(
            rho_post=rho_post, rho_prior=rho_prior, prior_dist=prior_dist,
            init_post=init_post, init_prior=init_prior,
            post_sigma_scale=post_sigma_scale, prior_sigma_scale=prior_sigma_scale,
            local_reparam=local_reparam,
        )

        self.mid_modules = nn.ModuleList([
            ProbConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim, kernel_size=kernel_size,
                n_groups=n_groups, cond_predict_scale=cond_predict_scale, **bayes_kwargs
            ),
            ProbConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim, kernel_size=kernel_size,
                n_groups=n_groups, cond_predict_scale=cond_predict_scale, **bayes_kwargs
            ),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                MixedConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, kernel_size=kernel_size,
                    n_groups=n_groups, cond_predict_scale=cond_predict_scale, **bayes_kwargs
                ),
                MixedConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim, kernel_size=kernel_size,
                    n_groups=n_groups, cond_predict_scale=cond_predict_scale, **bayes_kwargs
                ),
                Downsample1d(dim_out) if not is_last else nn.Identity(),
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                MixedConditionalResidualBlock1D(
                    dim_out * 2, dim_in, cond_dim=cond_dim, kernel_size=kernel_size,
                    n_groups=n_groups, cond_predict_scale=cond_predict_scale, **bayes_kwargs
                ),
                MixedConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim, kernel_size=kernel_size,
                    n_groups=n_groups, cond_predict_scale=cond_predict_scale, **bayes_kwargs
                ),
                Upsample1d(dim_in) if not is_last else nn.Identity(),
            ]))

        final_conv = nn.Sequential(
            ProbConv1dBlock(
                start_dim, start_dim, kernel_size=kernel_size, n_groups=n_groups,
                **bayes_kwargs
            ),
            ProbConv1d(
                start_dim, output_dim, kernel_size=1, padding=0,
                **bayes_kwargs
            ),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        self.rho_prior = rho_prior
        self.rho_post = rho_post
        self.prior_dist = prior_dist
        self.local_reparam = local_reparam

    def sample_weights(self):
        for layer in self.mid_modules:
            if hasattr(layer, "sample_weights"):
                layer.sample_weights()
        for module_list in self.down_modules:
            for layer in module_list:
                if hasattr(layer, "sample_weights"):
                    layer.sample_weights()
        for module_list in self.up_modules:
            for layer in module_list:
                if hasattr(layer, "sample_weights"):
                    layer.sample_weights()
        if hasattr(self.final_conv[0], "sample_weights"): self.final_conv[0].sample_weights()
        if hasattr(self.final_conv[1], "sample_weights"): self.final_conv[1].sample_weights()

    def clear_sampled_weights(self):
        for layer in self.mid_modules:
            if hasattr(layer, "clear_sample"):
                layer.clear_sample()
        for module_list in self.down_modules:
            for layer in module_list:
                if hasattr(layer, "clear_sample"):
                    layer.clear_sample()
        for module_list in self.up_modules:
            for layer in module_list:
                if hasattr(layer, "clear_sample"):
                    layer.clear_sample()
        if hasattr(self.final_conv[0], "clear_sample"): self.final_conv[0].clear_sample()
        if hasattr(self.final_conv[1], "clear_sample"): self.final_conv[1].clear_sample()

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int] = None,
        global_cond=None,
        stochastic=False,
        **kwargs,
    ):
        sample = einops.rearrange(sample, "b h t -> b t h")

        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)
        if global_cond is not None:
            global_feature = torch.cat([global_feature, global_cond], axis=-1)

        x = sample
        h = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, global_feature, stochastic=stochastic)
            x = resnet2(x, global_feature, stochastic=stochastic)
            h.append(x)
            x = downsample(x)  # deterministic: Identity or plain Downsample1d

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature, stochastic=stochastic)

        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature, stochastic=stochastic)
            x = resnet2(x, global_feature, stochastic=stochastic)
            x = upsample(x)  # deterministic: Identity or plain Upsample1d

        x = self.final_conv[0](x, stochastic=stochastic)
        x = self.final_conv[1](x, stochastic=stochastic)
        x = einops.rearrange(x, "b t h -> b h t")
        return x

    def compute_kl(self):
        kl_div = 0
        for layer in self.mid_modules:
            kl_div += layer.compute_kl()
        for module_list in self.down_modules:
            for layer in module_list:
                if hasattr(layer, 'compute_kl'):
                    kl_div += layer.compute_kl()
        for module_list in self.up_modules:
            for layer in module_list:
                if hasattr(layer, 'compute_kl'):
                    kl_div += layer.compute_kl()
        kl_div += self.final_conv[0].compute_kl()
        kl_div += self.final_conv[1].kl_div
        return kl_div
