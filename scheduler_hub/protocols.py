from __future__ import annotations
from typing import Protocol
from typing_extensions import TypedDict

from torch import Tensor


class Model(Protocol):
    def __call__(
        self,
        x: Tensor,
        timestep: Tensor,
        **extra_kwargs,
    ) -> Tensor:
        ...


class CallbackArgs(TypedDict):
    i: int | float
    x: Tensor
    denoised: Tensor


class Callback(Protocol):
    def __call__(self, kwargs: CallbackArgs) -> Tensor:
        ...
