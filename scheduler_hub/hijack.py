from __future__ import annotations
from typing import Optional

import torch
from torch import Tensor

from .utils import orthonormal_factory


class HijackRandom:
    # Hijack torch random number generation and save Model parameters

    def __init__(
        self,
        latents_history: list[Tensor],
        noise_pred_history: list[Tensor],
        noise_history: list[Tensor],
        max_size: int,
        /,
        rand: bool = True,
    ) -> None:
        self.latents_history = latents_history
        self.noise_pred_history = noise_pred_history
        self.noise_history = noise_history

        self.sample_noise = orthonormal_factory(max_size, rand)

    def model(
        self,
        latents: Tensor,
        timestep: Tensor,
        /,
        guidance_scale: Optional[float | Tensor] = None,
    ) -> Tensor:
        self.latents_history.append(latents)

        noise_pred = self.sample_noise()
        self.noise_pred_history.append(noise_pred)

        return noise_pred

    def __enter__(self):
        self.randn_backup = torch.randn
        self.randn_like_backup = torch.randn_like

        def randn(*args, **kwargs):
            eps = self.sample_noise()
            self.noise_history.append(eps)

            return eps

        def randn_like(*args):
            return randn()

        torch.randn = randn
        torch.randn_like = randn_like

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        torch.randn = self.randn_backup
        torch.randn_like = self.randn_like_backup
