from typing import Dict
import torch
import torch.nn as nn
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.model.common.normalizer import LinearNormalizer

class BaseLowdimPacPolicy(ModuleAttrMixin):  
    # ========= inference  ============
    # also as self.device and self.dtype for inference device transfer
    def predict_action(self, obs_dict: Dict[str, torch.Tensor], stochastic=False, clamping=True) -> Dict[str, torch.Tensor]:
        """
        obs_dict:
            obs: B,To,Do
            stochastic: False           stochastic sampling of network's weights and biases 
            claming: True               clamp the output of the model
        return: 
            action: B,Ta,Da
        To = 3
        Ta = 4
        T = 6
        |o|o|o|
        | | |a|a|a|a|
        |o|o|
        | |a|a|a|a|a|
        | | | | |a|a|
        """
        raise NotImplementedError()

    # reset state for stateful policies
    def reset(self):
        pass

    # ========== training ===========
    # no standard training interface except setting normalizer
    def set_normalizer(self, normalizer: LinearNormalizer):
        raise NotImplementedError()

    # ========== PAC-Bayes bound support ===========
    # Shared by PacDiffusionUnetLowdimPolicy/PacDriftUnetLowdimPolicy/
    # PacFlowUnetLowdimPolicy's compute_bound(): the fquad/classic/friendly
    # bound formulas are only mathematically valid for an empirical risk
    # bounded in [0,1] (they're derived from concentration bounds for
    # bounded/Bernoulli-like random variables - see e.g. Dziugaite & Roy 2017,
    # Pérez-Ortiz et al. 2021). No rescaling is applied anywhere below - these
    # losses are already ~O(1) by construction (MSE against a unit-variance
    # noise/velocity target) - only *how* the raw loss_emp is mapped into
    # [0,1] is a choice, made via `transform`. compute_bound's own `bounded`
    # flag makes calling this at all optional.
    def _bound_empirical_risk(self, loss_emp: torch.Tensor, transform: str = "clamp") -> torch.Tensor:
        """
        transform (no scaling in any of these - operates on the raw loss_emp):
          - "clamp" (default): min(loss_emp, 1) via torch.clamp(loss_emp,
            max=1.0) (a lower clamp at 0 isn't needed since MSE is already
            >= 0). Exactly bounded, so the PAC-Bayes bound computed from it
            is valid. Caveat: torch.clamp has ZERO gradient once loss_emp > 1
            - backprop gets no gradient from the empirical-risk term at all
            on a step where that happens (only the KL term still gets one),
            which can stall training exactly when the network most needs to
            reduce the loss. Safe to use once you've empirically confirmed
            (as here) that loss_emp reliably drops below 1 early on.
          - "exp": 1 - exp(-loss_emp). Smooth, monotonic, exactly bounded in
            [0, 1), with gradient exp(-loss_emp) that is strictly positive
            everywhere - it never stalls, no matter how large loss_emp is,
            though the gradient shrinks (like a Huber-style robust loss) the
            larger loss_emp gets. Standard "soft-clip"/saturating-loss
            construction used to keep an a-priori-unbounded loss provably
            bounded without a training-time gradient cliff.
          - "tanh": tanh(loss_emp). Same guarantees/shape as "exp" (smooth,
            strictly positive gradient, bounded in [0,1)) with a slightly
            different saturation curve (approaches 1 faster).
        """
        if transform == "clamp":
            return torch.clamp(loss_emp, max=1.0)
        elif transform == "exp":
            return 1.0 - torch.exp(-loss_emp)
        elif transform == "tanh":
            return torch.tanh(loss_emp)
        else:
            raise ValueError(f"Unknown bound transform {transform!r} (expected 'clamp', 'exp', or 'tanh')")

