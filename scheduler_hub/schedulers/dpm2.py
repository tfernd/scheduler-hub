from __future__ import annotations

import torch
from torch import Tensor

from ..scheduler import Scheduler2
from ..utils import get_gamma_sigma_hat, lerp


class DPM2(Scheduler2):
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
        sigma_mid = lerp(sigma_hat, ns, 1 / 2)
        mask_noise = gamma > 0
        order_mask = ns != 0
        gn = torch.where(order_mask, sigma_mid, ns)

        self.timestep[:] = sigma_hat
        self.timestep2[:] = sigma_mid

        self.prep_x[:] = 1
        self.prep_noise[:] = s_noise * (sigma_hat**2 - cs**2).sqrt() * mask_noise

        self.transform_x[:] = gn / sigma_hat
        self.transform_denoised[:] = 1 - gn / sigma_hat

        self.order_mask[:] = order_mask

        self.transform2_x[:] = (ns - sigma_hat) / sigma_mid
        self.transform2_denoised[:] = (sigma_hat - ns) / sigma_mid
        self.transform2_prev_x[:] = 1

        self.zero_transform2_by_order_mask()
