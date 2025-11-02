from typing import Dict
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.policy.base_lowdim_prob_policy import BaseLowdimProbPolicy

class BaseLowdimRunner:
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def run(self, policy: BaseLowdimPolicy) -> Dict:
        raise NotImplementedError()
    
    def run_prob(self, policy: BaseLowdimProbPolicy, stochastic=False, clamping=False) -> Dict:
        raise NotImplementedError()
