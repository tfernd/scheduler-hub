from __future__ import annotations

import math

import torch
from torch import Tensor


def get_sigmas_karras(steps: int, sigma_min: float, sigma_max: float, rho: float = 7) -> Tensor:
    t = torch.linspace(0, 1, steps)
    a = sigma_min ** (1 / rho)
    b = sigma_max ** (1 / rho)

    sigmas = (b + t * (a - b)) ** rho

    return torch.cat([sigmas, sigmas.new_zeros([1])])


def get_gamma_sigma_hat(
    cs: Tensor,
    steps: int,
    s_churn: float,
    s_tmin: float,
    s_tmax: float,
) -> tuple[Tensor, Tensor]:
    range_mask = (s_tmin <= cs) & (cs <= s_tmax)
    gamma = min(s_churn / steps, math.sqrt(2) - 1) * range_mask
    sigma_hat = cs * (gamma + 1)

    return gamma, sigma_hat


def get_ancestral_step(
    sigma_from: Tensor,
    sigma_to: Tensor,
    eta: float,
) -> tuple[Tensor, Tensor]:
    sigma_up = eta * sigma_to * torch.sqrt(1 - (sigma_to / sigma_from) ** 2)
    sigma_up = torch.min(sigma_to, sigma_up)
    sigma_down = torch.sqrt(sigma_to**2 - sigma_up**2)

    return sigma_down, sigma_up
