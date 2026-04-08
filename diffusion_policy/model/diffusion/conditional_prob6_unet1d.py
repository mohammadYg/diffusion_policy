import torch
import torch.nn as nn

import torch.nn.functional as F
from einops.layers.torch import Rearrange
from typing import Union
  
import einops
from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb
from diffusion_policy.model.diffusion.bayes_conv1d_components import (
    ProbConv1d, ProbConv1dBlock,
    ProbDownsample1d, ProbUpsample1d, ProbLinear
) 
from diffusion_policy.model.diffusion.conv1d_components import (
    Downsample1d, Upsample1d, Conv1dBlock)

class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, 
            in_channels, 
            out_channels, 
            cond_dim,
            kernel_size=3,
            n_groups=8,
            cond_predict_scale=False):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels
        if cond_predict_scale:
            cond_channels = out_channels * 2
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            Rearrange('batch t -> batch t 1'),
        )

        # make sure dimensions compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        '''
            x : [ batch_size x in_channels x horizon ]
            cond : [ batch_size x cond_dim]

            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)
        if self.cond_predict_scale:
            embed = embed.reshape(
                embed.shape[0], 2, self.out_channels, 1)
            scale = embed[:,0,...]
            bias = embed[:,1,...]
            out = scale * out + bias
        else:
            out = out + embed
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out

class ProbConditionalResidualBlock1D(nn.Module):
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
        init_prior='random'
    ):
        super().__init__()
        
        self.blocks = nn.ModuleList(
            [
                ProbConv1dBlock(
                    in_channels, out_channels, kernel_size, 
                    n_groups=n_groups, rho_post=rho_post, rho_prior=rho_prior,
                    prior_dist=prior_dist, init_post=init_post, init_prior=init_prior
                ),
                ProbConv1dBlock(
                    out_channels, out_channels, kernel_size,
                    n_groups=n_groups, rho_post=rho_post, rho_prior=rho_prior,
                    prior_dist=prior_dist, init_post=init_post, init_prior=init_prior
                ),
            ]
        )

        # FiLM modulation https://arxiv.org/abs/1709.07871
        cond_channels = out_channels
        if cond_predict_scale:
            cond_channels = out_channels * 2
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels
        
        # Probabilistic conditioning pathway
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            ProbLinear(
                cond_dim, cond_channels, 
                rho_post=rho_post,
                rho_prior=rho_prior, 
                prior_dist=prior_dist,
                init_post = init_post,
                init_prior=init_prior
            ),
            Rearrange("batch t -> batch t 1"),
        )

        # # Deterministic conditioning pathway
        # self.cond_encoder = nn.Sequential(
        #     nn.Mish(),
        #     nn.Linear(cond_dim, cond_channels),
        #     Rearrange("batch t -> batch t 1"),
        # )

        # make sure dimensions compatible. residual layer is deterministic
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

        # if in_channels != out_channels:
        #     self.residual_conv = ProbConv1d(
        #         in_channels, out_channels, kernel_size=1, rho_post=rho_post,
        #         rho_prior=rho_prior, prior_dist=prior_dist, init_post=init_post, init_prior=init_prior, padding=0
        #     )
        # else:
        #     self.residual_conv = nn.Identity()

    def sample_weights(self):
        self.blocks[0].sample_weights()
        self.blocks[1].sample_weights()
        self.cond_encoder[1].sample_weights()
        #self.residual_conv.sample_weights() if not isinstance(self.residual_conv, nn.Identity) else None

    def clear_sample(self):
        self.blocks[0].clear_sample()
        self.blocks[1].clear_sample()
        self.cond_encoder[1].clear_sample()
        #self.residual_conv.clear_sample() if not isinstance(self.residual_conv, nn.Identity) else None

    def forward(self, x, cond, stochastic=False):
        """
        x : [ batch_size x in_channels x horizon ]
        cond : [ batch_size x cond_dim]

        returns:
        out : [ batch_size x out_channels x horizon ]
        """
        # First convolutional block
        out = self.blocks[0](x, stochastic=stochastic)
        
        # FiLM conditioning with probabilistic linear layer
        embed = self.cond_encoder[0](cond)  # Mish activation
        embed = self.cond_encoder[1](embed, stochastic=stochastic)  # ProbLinear
        embed = self.cond_encoder[2](embed)  # Rearrange
        
        # # FiLM conditioning with deterministic linear layer
        # embed = self.cond_encoder(cond)
        
        if self.cond_predict_scale:
            embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
            scale = embed[:, 0, ...]
            bias = embed[:, 1, ...]
            out = scale * out + bias
        else:
            out = out + embed
        
        # Second convolutional block
        out = self.blocks[1](out, stochastic=stochastic)
        
        # Residual connection
        out = out + self.residual_conv(x)

        # # Residual connection
        # if isinstance(self.residual_conv, nn.Identity):
        #     residual = self.residual_conv(x)
        # else:
        #     residual = self.residual_conv(x, stochastic=stochastic)
        # out = out + residual
       
        return out
    
    def compute_kl(self):  # Renamed for consistency
        """Get total KL divergence from all probabilistic components"""
        kl_div = 0
        
        # KL from convolutional blocks
        kl_div += self.blocks[0].block[0].kl_div
        kl_div += self.blocks[1].block[0].kl_div
        
        # KL from conditioning linear layer
        kl_div += self.cond_encoder[1].kl_div
        
        # # KL from residual convolution (if present)
        # if not isinstance(self.residual_conv, nn.Identity):
        #     kl_div += self.residual_conv.kl_div
            
        return kl_div

class BayesianConditionalUnet1D(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim=None,
        local_cond_dim=None,
        global_cond_dim=None,
        diffusion_step_embed_dim=256,
        down_dims=[256, 512, 1024],
        kernel_size=3,
        n_groups=8,
        cond_predict_scale=False,
        use_dropout=False,
        # Bayesian parameters
        rho_post=-3.0,
        rho_prior=-3.0,
        prior_dist='gaussian',
        init_post='random',
        init_prior='random'
    ):
        super().__init__()
        
        if output_dim is None:
            output_dim = input_dim
        #print("output_dim", output_dim)

        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]
        
        # Deterministic timestep encoding 
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

        # Probabilistic local conditioning
        local_cond_encoder = None
        if local_cond_dim is not None:
            _, dim_out = in_out[0]
            dim_in = local_cond_dim
            local_cond_encoder = nn.ModuleList([
                # down encoder
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                # up encoder
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale)
            ])

        # mid_dim = all_dims[-1]
        # self.mid_modules = nn.ModuleList([
        #     ConditionalResidualBlock1D(
        #         mid_dim, mid_dim, cond_dim=cond_dim,
        #         kernel_size=kernel_size, n_groups=n_groups,
        #         cond_predict_scale=cond_predict_scale
        #     ),
        #     ConditionalResidualBlock1D(
        #         mid_dim, mid_dim, cond_dim=cond_dim,
        #         kernel_size=kernel_size, n_groups=n_groups,
        #         cond_predict_scale=cond_predict_scale
        #     ),
        # ])

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList(
            [
                ProbConditionalResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale,
                    rho_post=rho_post,
                    rho_prior=rho_prior,
                    prior_dist=prior_dist,
                    init_post=init_post,
                    init_prior=init_prior
                ),
                ProbConditionalResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale,
                    rho_post=rho_post,
                    rho_prior=rho_prior,
                    prior_dist=prior_dist,
                    init_post=init_post,
                    init_prior=init_prior
                ),
            ]
        )

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            if ind==0:
                down_modules.append(
                nn.ModuleList(
                    [
                        ProbConditionalResidualBlock1D(
                            dim_in,
                            dim_out,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            cond_predict_scale=cond_predict_scale,
                            rho_post=rho_post,
                            rho_prior=rho_prior,
                            prior_dist=prior_dist,
                            init_post=init_post
                        ),
                        ProbConditionalResidualBlock1D(
                            dim_out,
                            dim_out,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            cond_predict_scale=cond_predict_scale,
                            rho_post=rho_post,
                            rho_prior=rho_prior,
                            prior_dist=prior_dist,
                            init_post=init_post
                        ),
                        #Downsample1d(dim_out) if not is_last else nn.Identity()
                        ProbDownsample1d(
                            dim_out, 
                            rho_post=rho_post,
                            rho_prior=rho_prior,
                            prior_dist=prior_dist
                        ) if not is_last else nn.Identity()
                    ]
                )
            )
            else:
                down_modules.append(nn.ModuleList([
                    ConditionalResidualBlock1D(
                        dim_in, dim_out, cond_dim=cond_dim, 
                        kernel_size=kernel_size, n_groups=n_groups,
                        cond_predict_scale=cond_predict_scale),
                    ConditionalResidualBlock1D(
                        dim_out, dim_out, cond_dim=cond_dim, 
                        kernel_size=kernel_size, n_groups=n_groups,
                        cond_predict_scale=cond_predict_scale),
                    Downsample1d(dim_out) if not is_last else nn.Identity()
                ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            if ind==5 or ind==6:
                up_modules.append(
                nn.ModuleList(
                    [
                        ProbConditionalResidualBlock1D(
                            dim_out * 2,
                            dim_in,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            cond_predict_scale=cond_predict_scale,
                            rho_post=rho_post,
                            rho_prior=rho_prior,
                            prior_dist=prior_dist,
                            init_post=init_post,
                            init_prior=init_prior
                        ),
                        ProbConditionalResidualBlock1D(
                            dim_in,
                            dim_in,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            cond_predict_scale=cond_predict_scale,
                            rho_post=rho_post,
                            rho_prior=rho_prior,
                            prior_dist=prior_dist,
                            init_post=init_post,
                            init_prior=init_prior
                        ),
                        # ProbUpsample1d(
                        #     dim_in,
                        #     rho_post=rho_post,
                        #     rho_prior=rho_prior,
                        #     prior_dist=prior_dist,
                        #     init_post=init_post,
                        #     init_prior=init_prior
                        # ) if not is_last else nn.Identity(),
                        Upsample1d(dim_in) if not is_last else nn.Identity()
                    ]
                )
            )
            else:
                up_modules.append(nn.ModuleList([
                    ConditionalResidualBlock1D(
                        dim_out*2, dim_in, cond_dim=cond_dim,
                        kernel_size=kernel_size, n_groups=n_groups,
                        cond_predict_scale=cond_predict_scale),
                    ConditionalResidualBlock1D(
                        dim_in, dim_in, cond_dim=cond_dim,
                        kernel_size=kernel_size, n_groups=n_groups,
                        cond_predict_scale=cond_predict_scale),
                    Upsample1d(dim_in) if not is_last else nn.Identity()
                ]))

        # final_conv = nn.Sequential(
        #     Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
        #     nn.Conv1d(start_dim, input_dim, 1),
        # )

        final_conv = nn.Sequential(
            ProbConv1dBlock(
                start_dim, start_dim, kernel_size=kernel_size,
                n_groups=n_groups, rho_post=rho_post, rho_prior=rho_prior,
                prior_dist=prior_dist,
                init_post=init_post, init_prior=init_prior
            ),
            #nn.Conv1d(start_dim, input_dim, 1)
            ProbConv1d(
                start_dim, input_dim, kernel_size=1,
                rho_post=rho_post,
                rho_prior=rho_prior, prior_dist=prior_dist, init_post=init_post, init_prior=init_prior
            ),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.local_cond_encoder = local_cond_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        # Store Bayesian parameters for reference
        self.rho_prior = rho_prior
        self.rho_post = rho_post
        self.prior_dist = prior_dist

    def sample_weights(self):
        # Sample weights for all probabilistic layers in the model
        # for layer in self.diffusion_step_encoder:
        #     if hasattr(layer, "sample_weights"):
        #         layer.sample_weights()
        
        # if self.local_cond_encoder is not None:
        #     for layer in self.local_cond_encoder:
        #         if hasattr(layer, "sample_weights"):
        #             layer.sample_weights()

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
        #Clear sampled weights for all probabilistic layers in the model
        # for layer in self.diffusion_step_encoder:
        #     if hasattr(layer, "clear_sample"):
        #         layer.clear_sample()

        # if self.local_cond_encoder is not None:
        #     for layer in self.local_cond_encoder:
        #         if hasattr(layer, "clear_sample"):
        #             layer.clear_sample()

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
        local_cond=None,
        global_cond=None,
        stochastic=False,  # Added stochastic parameter
        **kwargs,
    ):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        local_cond: (B,T,local_cond_dim)
        global_cond: (B,global_cond_dim)
        output: (B,T,output_dim)
        """
        sample = einops.rearrange(sample, "b h t -> b t h")

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor(
                [timesteps], dtype=torch.long, device=sample.device
            )
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])

        # use deterministic encoding for timesteps
        global_feature = self.diffusion_step_encoder(timesteps)

        if global_cond is not None:
            global_feature = torch.cat([global_feature, global_cond], axis=-1)
        
        # encode local features
        h_local = list()
        if local_cond is not None:
            local_cond = einops.rearrange(local_cond, 'b h t -> b t h')
            resnet, resnet2 = self.local_cond_encoder
            x = resnet(local_cond, global_feature)
            h_local.append(x)
            x = resnet2(local_cond, global_feature)
            h_local.append(x)
        
        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            if idx==0:
                x = resnet(x, global_feature, stochastic=stochastic)
                if idx == 0 and len(h_local) > 0:
                    x = x + h_local[0]
                x = resnet2(x, global_feature, stochastic=stochastic)
                h.append(x)
                #x = downsample(x)
                if not isinstance(downsample, nn.Identity):
                    x = downsample(x, stochastic=stochastic)
                else:
                    x = downsample(x)
            else:
                x = resnet(x, global_feature)
                if idx == 0 and len(h_local) > 0:
                    x = x + h_local[0]
                x = resnet2(x, global_feature)
                h.append(x)                     # skip connection
                x = downsample(x)

        for mid_module in self.mid_modules:
            #x = mid_module(x, global_feature)
            x = mid_module(x, global_feature, stochastic=stochastic)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            if idx==5 or idx==6:
                x = resnet(x, global_feature, stochastic=stochastic)
                if idx == (len(self.up_modules) - 1) and len(h_local) > 0:
                    x = x + h_local[1]
                x = resnet2(x, global_feature, stochastic=stochastic)
                x = upsample(x)
                # if not isinstance(upsample, nn.Identity):
                #     x = upsample(x, stochastic=stochastic)
                # else:
                #     x = upsample(x)
            else:
                x = resnet(x, global_feature)
                # The correct condition should be:
                if idx == (len(self.up_modules)-1) and len(h_local) > 0:
                # However this change will break compatibility with published checkpoints.
                # Therefore it is left as a comment.
                # if idx == len(self.up_modules) and len(h_local) > 0:
                    x = x + h_local[1]
                x = resnet2(x, global_feature)
                x = upsample(x)
        
        # Apply final convolution with stochastic sampling
        #x = self.final_conv(x)
        x = self.final_conv[0](x, stochastic=stochastic)
        x = self.final_conv[1](x, stochastic=stochastic)
        x = einops.rearrange(x, "b t h -> b h t")
        return x
    
    def compute_kl(self):
        """Compute total KL divergence from all probabilistic components"""
        kl_div = 0
        
        # # KL from local condition encoder
        # if self.local_cond_encoder is not None:
        #     for layer in self.local_cond_encoder:
        #         if hasattr(layer, 'compute_kl'):
        #             kl_div += layer.compute_kl()

        # KL from mid modules
        for layer in self.mid_modules:
            if hasattr(layer, 'compute_kl'):
                kl_div += layer.compute_kl()

        # KL from down modules
        for module_list in self.down_modules:
            for layer in module_list:
                if hasattr(layer, 'compute_kl'):
                    kl_div += layer.compute_kl()

        # KL from up modules
        for module_list in self.up_modules:
            for layer in module_list:
                if hasattr(layer, 'compute_kl'):
                    kl_div += layer.compute_kl()

        # KL from final convolution
        kl_div += self.final_conv[0].compute_kl()
        kl_div += self.final_conv[1].kl_div
        
        return kl_div

   