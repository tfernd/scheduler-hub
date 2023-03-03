from __future__ import annotations
from typing import Protocol
from typing_extensions import TypedDict

from torch import Tensor


class Model(Protocol):
    def __call__(
        self,
        latents: Tensor,
        timestep: float | Tensor,
        **extra_kwargs,
    ) -> Tensor:
        ...


class ExtraArgs(TypedDict):
    i: int | float
    x: Tensor
    denoised: Tensor


class Callback(Protocol):
    def __call__(self, kwargs: ExtraArgs) -> Tensor:
        ...
