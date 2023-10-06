from __future__ import annotations
from typing import Optional

from sympy import Symbol

import torch
from torch import Tensor


class TorchSymbol(Symbol):
    @property
    def device(self) -> torch.device:
        return torch.device('cpu')
    
    @property
    def dtype(self) -> torch.dtype:
        return torch.float32

    @property
    def shape(self) -> tuple:
        return (1, 4, 32, 32)
    
    def to(self, *args, **kwargs):
        return self

class HijackRandom:
    def __init__(
        self,
        noise_pred_history: list,
        noise_history: list,
        /,
    ) -> None:
        self.noise_pred_history = noise_pred_history
        self.noise_history = noise_history

        self.count_model = 0
        self.count_random = 0

    def model(
        self,
        latents: Tensor,
        timestep: Tensor,
        /,
        guidance_scale: Optional[float | Tensor] = None,
    ):
        noise_pred = TorchSymbol(f'noise_pred({self.count_model})')
        self.noise_pred_history.append(noise_pred)
        self.count_model += 1

        return noise_pred

    def __enter__(self):
        self.randn_backup = torch.randn
        self.randn_like_backup = torch.randn_like

        def randn(*args, **kwargs):
            eps = TorchSymbol(f'noise({self.count_random})')
            self.noise_history.append(eps)
            self.count_random += 1

            return eps

        def randn_like(*args):
            return randn()

        torch.randn = randn
        torch.randn_like = randn_like

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        torch.randn = self.randn_backup
        torch.randn_like = self.randn_like_backup
