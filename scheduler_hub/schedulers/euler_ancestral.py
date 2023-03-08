from __future__ import annotations

from torch import Tensor

from ..scheduler import Scheduler1
from ..utils import get_ancestral_step


class EulerAncestral(Scheduler1):
    def __init__(
        self,
        sigmas: Tensor,
        eta: float = 1,
        s_noise: float = 1,
    ) -> None:
        sigmas, steps = self.clone_sigmas(sigmas)
        super().__init__(steps)

        cs, ns = sigmas[:-1], sigmas[1:]

        sigma_down, sigma_up = get_ancestral_step(cs, ns, eta)
        noise_mask = ns > 0

        self.timestep[:] = cs

        self.prep_x[:] = 1

        self.transform_x[:] = sigma_down / cs
        self.transform_denoised[:] = 1 - sigma_down / cs
        self.transform_noise[:] = s_noise * sigma_up * noise_mask
