from typing import Dict
import torch
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.drift.drift_util import drift_loss

class DriftUnetLowdimPolicy(BaseLowdimPolicy):
    def __init__(self,
            model: ConditionalUnet1D,
            horizon,
            obs_dim,
            action_dim,
            n_action_steps,
            n_obs_steps,
            obs_as_global_cond=True,
            temperatures=[0.02, 0.05, 0.2],
            per_timestep_loss=False,
            gen_per_label=8,
            **kwargs):
        super().__init__()
        assert obs_as_global_cond
        self.model = model
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.temperatures = temperatures
        # Doubles as a mixing probability p in [0,1] (bool True/False already
        # coerce to 1.0/0.0, so existing configs are unaffected): compute_loss
        # draws a fresh Bernoulli(p) choice every call, using the per-timestep
        # branch with probability p and the flattened branch with probability
        # 1-p. p=1.0 (True) always uses per-timestep; p=0.0 (False) always
        # uses flattened - identical to the old bool-only behavior in both
        # cases; any p in between stochastically mixes the two per call.
        self.per_timestep_loss = per_timestep_loss
        self.gen_per_label = gen_per_label
        self.kwargs = kwargs

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert 'obs' in obs_dict
        nobs = self.normalizer['obs'].normalize(obs_dict['obs'])
        B, _, Do = nobs.shape
        To = self.n_obs_steps
        assert Do == self.obs_dim
        T = self.horizon
        Da = self.action_dim
        device = self.device
        dtype = self.dtype

        global_cond = nobs[:,:To].reshape(B, -1)

        noise = torch.randn(size=(B, T, Da), device=device, dtype=dtype)
        timesteps = torch.zeros((B,), device=device, dtype=torch.long)
        naction_pred = self.model(noise, timesteps, global_cond=global_cond)

        action_pred = self.normalizer['action'].unnormalize(naction_pred)
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]

        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        nbatch = self.normalizer.normalize(batch)
        nobs = nbatch['obs']
        nactions = nbatch['action']
        batch_size = nactions.shape[0]
        G = self.gen_per_label

        global_cond = nobs[:,:self.n_obs_steps].reshape(batch_size, -1)

        # Generate G samples per observation (official: gen_per_label)
        global_cond_rep = global_cond.repeat_interleave(G, dim=0)  # [B*G, cond_dim]
        noise = torch.randn(batch_size * G, nactions.shape[1], nactions.shape[2], device=nactions.device, dtype=nactions.dtype)
        timesteps = torch.zeros(batch_size * G, device=nactions.device, dtype=torch.long)
        pred_all = self.model(noise, timesteps, global_cond=global_cond_rep)  # [B*G, T, D]
        pred_actions = pred_all.reshape(batch_size, G, nactions.shape[1], nactions.shape[2])  # [B, G, T, D]

        R_list = tuple(self.temperatures)

        # Bernoulli(p) draw, fresh every call - see the comment on
        # self.per_timestep_loss in __init__ for the p=0/p=1 boundary cases.
        use_per_timestep = torch.rand((), device=nactions.device).item() < float(self.per_timestep_loss)

        if use_per_timestep:
            T_horizon = nactions.shape[1]
            total_loss = 0
            # Collect every timestep's value per metric key instead of only
            # accumulating their mean, so we can also see how much a metric
            # varies ACROSS the horizon (e.g. whether error concentrates at a
            # few late/contact-critical timesteps rather than being uniform) -
            # that resolution was previously discarded by averaging in-place.
            per_t_values = {}
            for t in range(T_horizon):
                gen_t = pred_actions[:, :, t, :]           # [B, G, D]
                pos_t = nactions[:, t, :].unsqueeze(1)     # [B, 1, D]
                loss_t, info_t = drift_loss(gen_t, pos_t, R_list=R_list)
                total_loss = total_loss + loss_t.mean()
                for k, v in info_t.items():
                    per_t_values.setdefault(k, []).append(v.item())
            loss = total_loss / T_horizon
            all_metrics = {}
            for k, vals in per_t_values.items():
                vals_t = torch.tensor(vals)
                all_metrics[k] = vals_t.mean().item()
                all_metrics[f"{k}_min_t"] = vals_t.min().item()
                all_metrics[f"{k}_max_t"] = vals_t.max().item()
                all_metrics[f"{k}_std_t"] = vals_t.std().item() if len(vals) > 1 else 0.0
        else:
            gen = pred_actions.reshape(batch_size, G, -1)          # [B, G, T*D]
            pos = nactions.reshape(batch_size, 1, -1)              # [B, 1, T*D] //pne demonstrated action trajectory per observation
            loss, info = drift_loss(gen, pos, R_list=R_list)
            loss = loss.mean()
            all_metrics = {k: v.item() for k, v in info.items()}

        return loss, all_metrics