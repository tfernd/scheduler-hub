from __future__ import annotations

import math

import torch
from torch import Tensor

from ..scheduler import Scheduler1


TRAINED_STEPS = 1_000
BETA_BEGIN = 0.00085
BETA_END = 0.012
POWER = 2


class DDIM(Scheduler1):
    """Denoising Diffusion Implicit Models scheduler."""

    # https://arxiv.org/abs/2010.02502

    def __init__(
        self,
        steps: int,
        # sigmas: Tensor,
        eta: float = 1,
        approx: bool = False,
    ) -> None:
        # sigmas, steps = self.sigma_config(sigmas)
        super().__init__(steps)

        # trimmed timesteps for selection
        self.timesteps = torch.linspace(TRAINED_STEPS - 1, 0, steps + 1).ceil().long()

        if approx:
            # better approximation
            t = torch.linspace(1, 0, steps + 1)

            ᾱ = torch.exp(0.242816 * t - 2.28274 * t**2 - 2.78543 * t**3)
            ᾱ *= 1 - 1.11443 * t + 0.688069 * t**2
        else:
            # scheduler betas and alphas
            β_begin = math.pow(BETA_BEGIN, 1 / POWER)
            β_end = math.pow(BETA_END, 1 / POWER)
            β = torch.linspace(β_begin, β_end, TRAINED_STEPS).pow(POWER)
            print(β)

    #         # cummulative ᾱ trimmed
    #         α = 1 - β
    #         ᾱ = α.cumprod(dim=0)
    #         ᾱ /= ᾱ.max()  # makes last-value=1
    #         ᾱ = ᾱ[self.timesteps]

    #     ϖ = 1 - ᾱ

    #     # standard deviation, eq (16)
    #     σ = torch.sqrt(ϖ[1:] / ϖ[:-1] * (1 - ᾱ[:-1] / ᾱ[1:]))
    #     σ *= self.eta

    #     ## coefficients
    #     # eq (4) [remove last-step, not needed]
    #     self.noise_latents = ᾱ[:-1].sqrt()
    #     self.noise_noise = ϖ[:-1].sqrt()

    #     # eq (12)
    #     self.step_latents = torch.sqrt(ᾱ[1:] / ᾱ[:-1])
    #     self.step_pred_noise = -ϖ[:-1].sqrt() * self.step_latents
    #     self.step_pred_noise += (ϖ[1:] - σ**2).clamp(0).sqrt()
    #     self.step_noise = σ
