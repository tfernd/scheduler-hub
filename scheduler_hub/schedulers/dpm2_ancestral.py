from __future__ import annotations

import torch
from torch import Tensor

from ..scheduler import Scheduler2
from ..utils import get_ancestral_step, lerp


class DPM2Ancestral(Scheduler2):
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
        sigma_mid = lerp(cs, sigma_down, 1 / 2)
        order_mask = sigma_down != 0

        self.timestep[:] = cs
        self.timestep2[:] = sigma_mid

        self.prep_x[:] = 1

        gs = torch.where(order_mask, sigma_mid, sigma_down)
        self.transform_x[:] = gs / cs
        self.transform_denoised[:] = 1 - gs / cs

        self.order_mask[:] = order_mask

        self.transform2_x[:] = (sigma_down - cs) / sigma_mid
        self.transform2_denoised[:] = (cs - sigma_down) / sigma_mid
        self.transform2_prev_x[:] = 1
        self.transform2_noise[:] = s_noise * sigma_up

        self.zero_transform2_by_order_mask()
