from __future__ import annotations

import torch
from torch import Tensor

from ..scheduler import Scheduler1


class DPMpp2M(Scheduler1):
    def __init__(self, sigmas: Tensor) -> None:
        sigmas, steps = self.sigma_config(sigmas)
        super().__init__(steps)

        i = torch.arange(steps)
        ps = sigmas.roll(1)[:-1]
        cs, ns = sigmas[:-1], sigmas[1:]

        mask = (i == 0) | (ns == 0)

        A = 1 + torch.log(cs / ns) / (2 * torch.log(ps / cs))
        B = torch.log(ns / cs) / (2 * torch.log(ps / cs))
        C = 1 - ns / cs

        A[mask] = 1
        B[mask] = 0

        A *= C
        B *= C

        self.timestep = cs

        self.prep_x[:] = 1

        self.transform_x[:] = ns / cs
        self.transform_denoised = A
        self.transform_prev_denoised = B
