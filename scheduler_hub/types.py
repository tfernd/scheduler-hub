from __future__ import annotations
from typing import Optional, Protocol, Union

from torch import Tensor

from diffusers.schedulers import (
    DDIMScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    HeunDiscreteScheduler,
    DDPMScheduler,
    DEISMultistepScheduler,
    PNDMScheduler,
    LMSDiscreteScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    KDPM2DiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    DEISMultistepScheduler,
    UniPCMultistepScheduler,
    DPMSolverSDEScheduler,
    DDPMWuerstchenScheduler,
    IPNDMScheduler,
    KarrasVeScheduler,
    RePaintScheduler,
    ScoreSdeVeScheduler,
    ScoreSdeVpScheduler,
    UnCLIPScheduler,
)

# TODO add more as more are validated
Schedulers = Union[
    DDIMScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    HeunDiscreteScheduler,
    # DDPMScheduler,
    # DEISMultistepScheduler,
    # PNDMScheduler,
    # LMSDiscreteScheduler,
    # DPMSolverMultistepScheduler,
    # DPMSolverSinglestepScheduler,
    # KDPM2DiscreteScheduler,
    # KDPM2AncestralDiscreteScheduler,
    # DEISMultistepScheduler,
    # UniPCMultistepScheduler,
    # DPMSolverSDEScheduler,
    # DDPMWuerstchenScheduler,
    # IPNDMScheduler,
    # KarrasVeScheduler,
    # RePaintScheduler,
    # ScoreSdeVeScheduler,
    # ScoreSdeVpScheduler,
    # UnCLIPScheduler,
]


class Model(Protocol):
    def __call__(
        self,
        latents: Tensor,
        timestep: Tensor,
        /,
        guidance_scale: Optional[float | Tensor] = None,
        # extra_args # TODO
    ) -> Tensor:
        ...


class Sampler(Protocol):
    def __call__(
        self,
        model: Model,
        latents: Tensor,
        sigmas: Tensor,
        /,
        extra_args: Optional[dict] = None,
        disable: bool = True,
    ) -> Tensor:
        ...
