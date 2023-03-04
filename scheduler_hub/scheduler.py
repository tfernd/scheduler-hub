from __future__ import annotations
from typing import Optional
from typing_extensions import Self

from tqdm.auto import trange

import matplotlib.pyplot as plt

import torch
from torch import Tensor

from .protocols import Model, Callback


class BaseScheduler:
    steps: int

    device: torch.device = torch.device("cpu")
    dtype: torch.dtype = torch.float32

    # TODO rename
    @staticmethod
    def sigma_config(sigmas: Tensor) -> tuple[Tensor, int]:
        steps = len(sigmas) - 1
        sigmas = sigmas.detach().cpu().float().clone()

        return sigmas, steps

    def to(
        self,
        device: Optional[str | torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Self:
        if device is not None:
            self.device = torch.device(device)
        if dtype is not None:
            self.dtype = dtype

        for name, tensor in self.__dict__.items():
            if isinstance(tensor, Tensor):
                tensor.data = tensor.to(device=device)

                if tensor.dtype not in (torch.long, torch.bool):
                    tensor.data = tensor.to(dtype=dtype)

        return self

    def cuda(self) -> Self:
        return self.to(device="cuda")

    def cpu(self) -> Self:
        return self.to(device="cpu")

    def half(self) -> Self:
        return self.to(dtype=torch.half)

    def float(self) -> Self:
        return self.to(dtype=torch.float32)

    @property
    def name(self) -> str:
        return self.__class__.__qualname__

    def plot(self, show: bool = True) -> None:
        for name, value in self.__dict__.items():
            if not isinstance(value, Tensor):
                continue

            if value.eq(0).all():
                continue

            if name.startswith("timestep"):
                continue

            plt.plot(value.cpu(), label=name, marker=".")

        plt.title(f"{self.name}")
        plt.legend()
        plt.xlabel("steps")
        if show:
            plt.show()

    def smooth(self, iterations: int = 1, weight: float = 0.95) -> Self:
        if iterations == 0:
            return self

        for name, vec in self.__dict__.items():
            if not isinstance(vec, Tensor):
                continue

            if vec.dtype in (torch.long, torch.bool):
                continue

            for _ in range(iterations):
                vec[1:-1] = vec[1:-1] * weight + (1 - weight) * (vec[2:] + vec[:-2]) / 2

            if isinstance(self, Scheduler2):
                self.zero_transform2_by_order_mask()

        return self


class Scheduler1(BaseScheduler):
    # First order scheduler

    timestep: Tensor

    prep_x: Tensor
    prep_noise: Tensor

    transform_x: Tensor
    transform_denoised: Tensor
    transform_noise: Tensor
    transform_prev_x: Tensor
    transform_prev_denoised: Tensor

    def __init__(self, steps: int) -> None:
        self.steps = steps

        self.timestep = torch.zeros(steps)

        self.prep_x = torch.zeros(steps)
        self.prep_noise = torch.zeros(steps)

        self.transform_x = torch.zeros(steps)
        self.transform_denoised = torch.zeros(steps)
        self.transform_noise = torch.zeros(steps)
        self.transform_prev_x = torch.zeros(steps)
        self.transform_prev_denoised = torch.zeros(steps)

    @torch.no_grad()
    def sample(
        self,
        model: Model,
        x: Tensor,
        extra_args: Optional[dict] = None,
        callback: Optional[Callback] = None,
        disable: bool = False,
    ) -> Tensor:
        extra_args = extra_args or {}
        _callback = callback or (lambda obj: ...)

        self.to(self.device, self.dtype)

        s_in = x.new_ones([x.shape[0]])

        # memory
        prev_x = prev_denoised = 0
        for index in trange(self.steps, disable=disable):
            # generate noise
            noise = torch.randn_like(x)  # TODO add generator from seeds?

            # prepare latents # TODO add function for this
            x = self.prep_x[index] * x
            x = x + self.prep_noise[index] * noise

            # denoise
            denoised = model(x, self.timestep[index] * s_in, **extra_args)

            # callback
            _callback({"x": x, "i": index, "denoised": denoised})

            # transform # TODO add function for this
            x = self.transform_x[index] * x
            x = x + self.transform_denoised[index] * denoised
            x = x + self.transform_noise[index] * noise
            x = x + self.transform_prev_denoised[index] * prev_denoised
            x = x + self.transform_prev_x[index] * prev_x

            prev_x, prev_denoised = x, denoised

        return x


class Scheduler2(BaseScheduler):
    # Second order scheduler

    timestep: Tensor
    timestep2: Tensor

    prep_x: Tensor
    prep_noise: Tensor

    transform_x: Tensor
    transform_denoised: Tensor
    transform_noise: Tensor

    order_mask: Tensor

    transform2_x: Tensor
    transform2_denoised: Tensor
    transform2_noise: Tensor
    transform2_prev_x: Tensor
    transform2_prev_denoised: Tensor

    def __init__(self, steps: int) -> None:
        self.steps = steps

        self.timestep = torch.zeros(steps)
        self.timestep2 = torch.zeros(steps)

        self.prep_x = torch.zeros(steps)
        self.prep_noise = torch.zeros(steps)

        self.transform_x = torch.zeros(steps)
        self.transform_denoised = torch.zeros(steps)
        self.transform_noise = torch.zeros(steps)

        self.order_mask = torch.zeros(steps).bool()

        self.transform2_x = torch.zeros(steps)
        self.transform2_denoised = torch.zeros(steps)
        self.transform2_noise = torch.zeros(steps)
        self.transform2_prev_x = torch.zeros(steps)
        self.transform2_prev_denoised = torch.zeros(steps)

    @torch.no_grad()
    def sample(
        self,
        model: Model,
        x: Tensor,
        extra_args: Optional[dict] = None,
        callback: Optional[Callback] = None,
        disable: bool = False,
    ) -> Tensor:
        extra_args = extra_args if extra_args is not None else {}
        _callback = callback or (lambda obj: ...)

        self.to(self.device, self.dtype)

        s_in = x.new_ones([x.shape[0]])

        for index in trange(self.steps, disable=disable):
            # generate noise
            noise = torch.randn_like(x)  # TODO add generator from seeds?

            # prepare latents
            x = self.prep_x[index] * x  # TODO add function for this
            x = x + self.prep_noise[index] * noise

            # denoise
            denoised = model(x, self.timestep[index] * s_in, **extra_args)

            # memory
            prev_x, prev_denoised = x, denoised

            # callback
            _callback({"x": x, "i": index, "denoised": denoised})

            # transform # TODO add function for this
            x = self.transform_x[index] * x
            x = x + self.transform_denoised[index] * denoised
            x = x + self.transform_noise[index] * noise

            if self.order_mask[index]:
                # re-denoise
                denoised = model(x, self.timestep2[index] * s_in)

                # re-callback at half-steps # ! addition
                _callback({"x": x, "i": index + 1 / 2, "denoised": denoised})

                # re-transform # TODO add function for this
                x = self.transform2_x[index] * x
                x = x + self.transform2_denoised[index] * denoised
                x = x + self.transform2_noise[index] * noise
                x = x + self.transform2_prev_x[index] * prev_x
                x = x + self.transform2_prev_denoised[index] * prev_denoised

        return x

    def zero_transform2_by_order_mask(self) -> None:
        # these values should never be used since order_mask prohibits it

        self.timestep2[~self.order_mask] = 0
        self.transform2_x[~self.order_mask] = 0
        self.transform2_denoised[~self.order_mask] = 0
        self.transform2_noise[~self.order_mask] = 0
        self.transform2_prev_x[~self.order_mask] = 0
        self.transform2_prev_denoised[~self.order_mask] = 0
