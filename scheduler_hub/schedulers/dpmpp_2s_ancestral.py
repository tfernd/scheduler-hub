from __future__ import annotations

import torch
from torch import Tensor

from ..scheduler import Scheduler2
from ..utils import get_ancestral_step


class DPMpp2SAncestral(Scheduler2):
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
        mask_noise = ns > 0
        order_mask = sigma_down != 0

        A = sigma_down / cs
        A[order_mask] = A[order_mask].sqrt()

        B = s_noise * sigma_up * mask_noise

        self.timestep[:] = cs
        self.timestep2[:] = torch.sqrt(cs * sigma_down)

        self.prep_x[:] = 1

        self.transform_x[:] = A
        self.transform_denoised[:] = 1 - A
        self.transform_noise[~order_mask] = B[~order_mask]

        self.order_mask[:] = order_mask

        self.transform2_denoised[:] = 1 - sigma_down / cs
        self.transform2_prev_x[:] = sigma_down / cs
        self.transform2_noise[order_mask] = B[order_mask]

        self.zero_transform2_by_order_mask()
