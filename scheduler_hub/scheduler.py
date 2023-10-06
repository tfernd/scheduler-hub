from __future__ import annotations
from typing import Optional
from typing_extensions import Self

from abc import abstractmethod

from tqdm.auto import trange
from pathlib import Path


import random
import math
import torch
import torch.nn as nn
from torch import Tensor

import matplotlib.pyplot as plt

from .types import Model, Schedulers
from .hijack import HijackRandom, TorchSymbol


def find_scale_and_residual(out, X):
    symbols = X.free_symbols

    Xvec = torch.tensor([float(X.coeff(symbol)) for symbol in symbols])
    outvec = torch.tensor([float(out.coeff(symbol)) for symbol in symbols])

    num, den = Xvec.mul(outvec).sum().item(), Xvec.square().sum().item()
    if num != 0 and den != 0:
        alpha = num / den
    else:
        alpha = 0
    
    residual = out - alpha * X if alpha != 0 else out

    return alpha, residual

class BaseModule(nn.Module):
    device: torch.device = torch.device("cpu")
    dtype: torch.dtype = torch.float32

    def train(self, mode: bool = True, /) -> Self:
        self.training = mode

        for module in self.children():
            module.train(mode)

        self.requires_grad_(mode)

        return self

    def eval(self) -> Self:
        return self.train(False)

    def named_tensors(self):
        for name, param in self.named_parameters():
            yield name, param
        for name, param in self.named_buffers():
            yield name, param

    def to(
        self,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Self:
        if device is not None:
            self.device = device

            for name, param in self.named_tensors():
                param.data = param.to(self.device)

        if dtype is not None:
            self.dtype = dtype

            for name, param in self.named_tensors():
                param.data = param.to(self.dtype)

        return self


class BaseScheduler(BaseModule):
    timesteps: Tensor

    scale: Tensor
    noise: Tensor
    scale_pred: Tensor

    noisy_scale: Tensor
    noisy_noise_scale: Tensor

    guidance_scale: Tensor

    seed: int = -1

    @abstractmethod
    def __init__(
        self,
        steps: int,
        order: int = 1,
        noise_order: int = 1,
    ) -> None:
        super().__init__()

    def extra_repr(self) -> str:
        return f"steps={self.steps}, order={self.order}, noise_order={self.noise_order}, seed={self.seed}"  # ! SEED

    @property
    def steps(self) -> int:
        return len(self.timesteps)

    @property
    def order(self) -> int:
        return self.scale_pred.size(1)

    @property
    def noise_order(self) -> int:
        return self.noise.size(1)

    def trim_order(self, min_error: float = 1e-8, /) -> Self:
        for name, value in self.named_parameters():
            if value.ndim == 2:
                mask = value.abs().lt(min_error).all(0)
                idx = mask.nonzero()
                if len(idx) >= 1:
                    order = idx[0]
                    value.data = value.data[:, :order]

        return self

    def set_seed(self, seed: int = -1) -> Self:
        self.seed = random.randint(0, 2**32 - 1) if seed < 0 else seed

        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(self.seed)

        return self

    def randn(self, x: Tensor | tuple[int, ...], /) -> Tensor:
        shape = x.shape if isinstance(x, Tensor) else x

        return torch.randn(*shape, device=self.device, dtype=self.dtype, generator=self.generator)

    def clone(self) -> Self:
        new = self.__class__(self.steps, self.order, self.noise_order)
        new.to(device=self.device, dtype=self.dtype)
        new.seed = self.seed

        for name, value in self.named_tensors():
            getattr(new, name).data = value.data.clone()

        return new

    def inverse(self) -> Self:
        self.timesteps.data = self.timesteps.flip(0)

        self.noisy_scale.data = self.noisy_scale.flip(0)
        self.noisy_noise_scale.data = self.noisy_noise_scale.flip(0)

        alpha = self.scale.data.clone()
        beta = self.scale_pred.data.clone()
        gamma = self.noise.data.clone()

        self.scale.data = 1 / alpha.roll(-1).flip(0)
        self.scale_pred.data = -(beta / alpha[1:, None]).flip(0).flip(1)
        if self.noise_order > 0:
            self.noise.data = -(gamma / alpha[1:, None]).flip(0)

        self.trim_order()

        return self

    def early_stop(self, step: int, /) -> Self:
        steps = self.steps
        for name, value in self.named_tensors():
            if value.ndim in (1, 2):
                if value.size(0) == steps:
                    value.data = value[:step]
                else:
                    value.data = value[:step + 1]

        return self
    
    # TODO check
    def __getitem__(self, idx: slice, /) -> Self:
        new = self.clone()

        start = idx.start or 0
        stop = idx.stop or self.steps
        step = idx.step or 1

        assert step == 1

        if not isinstance(start, int):
            start = round(self.steps * start)
        if not isinstance(stop, int):
            stop = round(self.steps * stop)

        if start < 0:
            start = self.steps + start
        if stop < 0:
            stop = self.steps + stop

        start = max(0, start)
        stop = min(self.steps, stop)

        new.early_stop(stop)
        new.inverse().early_stop(new.steps - start).inverse()

        return new


class ConvertScheduler(BaseScheduler, BaseModule):
    @classmethod
    def from_diffusers(
        cls,
        scheduler: Schedulers,
        *,
        steps: int,
        order: int = 1,
        extra_noise: int = 1,  # ! needed, to account for brownian noise
    ) -> Self:
        # Configure the scheduler
        initial_steps = steps
        scheduler.set_timesteps(steps)  # type: ignore
        assert scheduler.timesteps is not None
        steps = len(scheduler.timesteps)  # recompute steps

        # Memory
        noise_pred_history = []
        noise_history = []

        self = cls(steps, order, extra_noise)
        self.timesteps.data = scheduler.timesteps.clone().float()

        # pre-scale latents
        init_latents = TorchSymbol('x(0)')
        latents = init_latents * scheduler.init_noise_sigma

        # Sample
        with HijackRandom(noise_pred_history, noise_history) as hijacked:
            for step, timestep in enumerate(scheduler.timesteps):
                input_latents = scheduler.scale_model_input(latents, timestep)  # type: ignore

                if step == 0:
                    alpha, residual = find_scale_and_residual(input_latents, init_latents)
                    self.scale.data[step] = alpha

                pred_noise = hijacked.model(input_latents, timestep)
                latents = scheduler.step(pred_noise, timestep, latents).prev_sample  # type: ignore

                # Get latents coefficient
                alpha, residual = find_scale_and_residual(latents, input_latents)
                self.scale.data[step+1] = alpha

                # Get pred-noise coefficients
                for o, pred_noise in enumerate(noise_pred_history[-order:]):
                    alpha, residual = find_scale_and_residual(residual, pred_noise)
                    self.scale_pred.data[step, o] = alpha

            # # Try all noises...
            # gammas: list[Tensor] = []
            # ids_to_remove: list[int] = []
            # for i, noise in enumerate(noise_history):
            #     gamma, residual = find_scale_and_residual(residual, noise)
            # #     if abs(gamma) > 1e-8:  # !
            # #         gammas.append(gamma)
            # #         ids_to_remove.append(i)
            # # for i in ids_to_remove:
            # #     del noise_history[i]
            # if len(gammas) > 0:
            #     self.noise.data[step, : len(gammas)] = torch.tensor(gammas).sort(descending=True).values

        # # TODO re-implement.
        # scheduler.set_timesteps(initial_steps)  # type: ignore
        # assert scheduler.timesteps is not None

        # for step, timestep in enumerate(scheduler.timesteps):
        #     latents = torch.randn(1, 4, 32, 32)
        #     noise = torch.randn_like(latents)

        #     noisy_latents = scheduler.add_noise(latents, noise, timestep.view(-1))  # type: ignore

        #     alpha, out = find_scale_and_residual(noisy_latents, latents)
        #     self.noisy_scale.data[step] = alpha

        #     beta, out = find_scale_and_residual(out, noise)
        #     self.noisy_noise_scale.data[step] = beta

        self.trim_order()

        return self


class Guidance(BaseScheduler):
    def set_guidance_scale(
        self,
        initial_scale: float,
        /,
        *,
        final_scale: Optional[float] = None,
        early_flat: int | float = 0,
        gamma: float = 1,
    ) -> Self:
        if final_scale is None:
            self.guidance_scale.data.fill_(initial_scale)
        else:
            if not isinstance(early_flat, int):
                early_flat = max(0, min(self.steps - 1, early_flat))
                early_flat = round(early_flat * self.steps)

            t = torch.arange(self.steps).div(self.steps - early_flat)
            t = t.to(device=self.device, dtype=self.dtype)

            X = torch.cos(t.pow(gamma) * math.pi / 2).square()

            out = (initial_scale - final_scale) * X + final_scale
            out.masked_fill_(t > 1, final_scale)

            self.guidance_scale.data = out.clamp(1)

        return self


class Plotting(BaseScheduler):
    def plot(self):
        fig, axes = plt.subplots(1, 3, figsize=(12, 3))

        norm = lambda x: x.detach().float().cpu()

        idx = torch.arange(-1, self.steps)

        axes[0].plot(idx, norm(self.scale), c="k", label="Latents scale")
        axes[0].plot(norm(self.scale_pred), label="Noise pred. scale")
        axes[0].plot(norm(self.noise), c="g", label="Noise")

        axes[1].plot(norm(self.timesteps), label="Timesteps", c="k")
        axes1t = axes[1].twinx()
        axes1t.plot(norm(self.guidance_scale), label="Guidance scale", c="r")

        axes[2].plot(norm(self.noisy_scale), label="Noisify latents scale")
        axes[2].plot(norm(self.noisy_noise_scale), label="Noisify noise scale")

        for ax in axes:
            ax.set_xlabel("Steps (1)")
            ax.legend()

        axes[1].legend(loc="lower left")
        axes1t.legend(loc="upper right")

        plt.tight_layout()
        plt.show()


class Scheduler(ConvertScheduler, Guidance, Plotting, BaseModule):
    def __init__(
        self,
        steps: int,
        order: int = 1,
        noise_order: int = 1,
    ) -> None:
        super().__init__(steps, order, noise_order)

        order = min(order, steps)

        # step=0 scale and add noise to the latents prior to calling the model, hence steps + 1
        self.scale = nn.Parameter(torch.zeros(steps + 1))
        self.noise = nn.Parameter(torch.zeros(steps + 1, noise_order))
        self.scale_pred = nn.Parameter(torch.zeros(steps, order))

        self.noisy_scale = nn.Parameter(torch.zeros(steps))
        self.noisy_noise_scale = nn.Parameter(torch.zeros(steps))

        self.guidance_scale = nn.Parameter(torch.ones(steps))

        self.timesteps = nn.Parameter(torch.zeros(steps))

        self.eval()

    def add_noise(
        self,
        latents: Tensor,
        /,
        step: int,
        *,
        seed: int = -1,
    ) -> Tensor:
        self.set_seed(seed)

        return self.noisy_scale[step] * latents + self.noisy_noise_scale[step] * self.randn(latents)

    def __call__(
        self,
        model: Model,
        x: Tensor | tuple[int, int, int, int],
        /,
        *,
        seed: int = -1,
        leave: bool = True,
        disable: bool = False,
        # ! extra args for model?
    ) -> Tensor:
        self.set_seed(seed)

        if isinstance(x, Tensor):
            self.to(device=x.device, dtype=x.dtype)
        else:
            x = self.randn(x)

        prev_np: list[Tensor] = []
        with trange(self.steps + 1, leave=leave, disable=disable) as pbar:
            for step in pbar:
                if step > 0:
                    noise_pred = model(x, self.timesteps[step - 1], self.guidance_scale[step - 1])
                    prev_np.append(noise_pred)
                    prev_np = prev_np[-self.order :]

                x = x * self.scale[step]
                for order in range(self.noise_order):
                    x = x + self.randn(x) * self.noise[step, order]
                for order, noise_pred in enumerate(reversed(prev_np)):
                    x = x + noise_pred * self.scale_pred[step - 1, order]

        return x

    def save(self, path: str | Path, /) -> Self:
        path = Path(path)
        assert path.suffix == ".pt"
        path.parent.mkdir(exist_ok=True, parents=True)

        state = self.state_dict()
        state.update(steps=self.steps, order=self.order, noise_order=self.noise_order)

        torch.save(state, path)

        return self

    @classmethod
    def load(cls, path: str | Path, /) -> Self:
        path = Path(path)
        assert path.suffix == ".pt"

        state = torch.load(path)

        steps = state.pop("steps")
        order = state.pop("order")
        noise_order = state.pop("noise_order")

        self = cls(steps, order, noise_order)
        self.load_state_dict(state)

        return self
