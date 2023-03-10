from __future__ import annotations

from torch import Tensor

from ..scheduler import Scheduler1
from ..utils import get_gamma_sigma_hat


class Euler(Scheduler1):
    def __init__(
        self,
        sigmas: Tensor,
        s_churn: float = 0,
        s_tmin: float = 0,
        s_tmax: float = float("inf"),
        s_noise: float = 1,
    ) -> None:
        sigmas, steps = self.clone_sigmas(sigmas)
        super().__init__(steps)

        cs, ns = sigmas[:-1], sigmas[1:]

        gamma, sigma_hat = get_gamma_sigma_hat(cs, steps, s_churn, s_tmin, s_tmax)
        noise_mask = gamma > 0

        self.timestep[:] = sigma_hat

        self.prep_x[:] = 1
        self.prep_noise[:] = s_noise * (sigma_hat**2 - cs**2).sqrt() * noise_mask

        self.transform_x[:] = ns / sigma_hat
        self.transform_denoised[:] = 1 - ns / sigma_hat
