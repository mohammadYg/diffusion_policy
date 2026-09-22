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
    def _bound_empirical_risk(self, loss_emp: torch.Tensor, transform: str = "clamp", scale: float = 1.0) -> torch.Tensor:
        """
        scale (M): loss_emp is divided by this BEFORE the transform below.
        The fquad/classic/friendly bounds need loss in [0,1], but the raw
        loss (e.g. drift_loss, summed over several temperatures) is often
        O(few) rather than O(1) - with scale=1.0 (the old default), every
        transform below ends up deep in its saturating tail, where its
        gradient is ~0 (tanh'(7) ~ 3e-6), silently killing the empirical-risk
        gradient while the (untouched) KL gradient keeps dominating. Pick
        `scale` as a calibrated estimate of loss_emp's typical scale so
        loss_emp/scale sits near the transform's near-linear region.
        `scale` is only an empirical estimate, not a provable ceiling on
        loss_emp - the transform is still required (not optional) even with
        a well-chosen scale, since it's the only part that GUARANTEES the
        [0,1] bound holds on every batch, including rare ones where
        loss_emp/scale > 1.

        transform (operates on loss_emp/scale):
          - "clamp" (default): min(loss_emp/scale, 1) via torch.clamp(...,
            max=1.0) (a lower clamp at 0 isn't needed since MSE is already
            >= 0). Exactly bounded, so the PAC-Bayes bound computed from it
            is valid, and - unlike "exp"/"tanh" - exactly linear (gradient
            1/scale, undiminished) whenever loss_emp/scale < 1. Caveat:
            torch.clamp has ZERO gradient once loss_emp/scale > 1 - backprop
            gets no gradient from the empirical-risk term at all on a step
            where that happens (only the KL term still gets one). With a
            well-chosen `scale` this should only bite on rare outlier
            batches instead of being the typical case.
          - "exp": 1 - exp(-loss_emp/scale). Smooth, monotonic, exactly
            bounded in [0, 1), with gradient exp(-loss_emp/scale) that is
            strictly positive everywhere - it never stalls, though the
            gradient shrinks (like a Huber-style robust loss) the larger
            loss_emp/scale gets.
          - "tanh": tanh(loss_emp/scale). Same guarantees/shape as "exp"
            with a slightly different saturation curve (approaches 1
            faster) - meaning it needs a larger `scale` than "exp"/"clamp"
            to avoid saturating at the same raw loss_emp value.
        """
        x = loss_emp / scale
        if transform == "clamp":
            return torch.clamp(x, max=1.0)
        elif transform == "exp":
            return 1.0 - torch.exp(-x)
        elif transform == "tanh":
            return torch.tanh(x)
        else:
            raise ValueError(f"Unknown bound transform {transform!r} (expected 'clamp', 'exp', or 'tanh')")

