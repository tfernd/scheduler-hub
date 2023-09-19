from __future__ import annotations
from typing import Optional
from typing_extensions import Self

from tqdm.auto import trange
from pathlib import Path

import random
import math
import torch
import torch.nn as nn
from torch import Tensor

from k_diffusion.sampling import get_sigmas_karras

import matplotlib.pyplot as plt

from .types import Model, Schedulers, Sampler
from .utils import find_scale_and_residual, sigmas_to_timesteps
from .hijack import HijackRandom


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
    guidance_scale: Tensor

    seed: int = -1

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

    def make_generator(self, /, seed: int = 1) -> tuple[int, torch.Generator]:
        self.seed = seed = random.randint(0, 2**32 - 1) if seed < 0 else seed

        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.seed)

        return self.seed, generator


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
        scheduler.set_timesteps(steps)  # type: ignore
        assert scheduler.timesteps is not None
        steps = len(scheduler.timesteps)  # recompute steps

        # Memory
        latents_history: list[Tensor] = []
        noise_pred_history: list[Tensor] = []
        noise_history: list[Tensor] = []

        self = cls(steps, order, extra_noise)
        self.timesteps.data = scheduler.timesteps.clone().float()

        # Random perp base
        max_size = 3 * steps * extra_noise
        with HijackRandom(latents_history, noise_pred_history, noise_history, max_size) as hijacked:
            # pre-scale latents
            latents = hijacked.sample_noise()
            latents_history.append(latents)

            latents = latents * scheduler.init_noise_sigma

            # Sample
            for timestep in scheduler.timesteps:
                input_latents = scheduler.scale_model_input(latents, timestep)  # type: ignore
                pred_noise = hijacked.model(input_latents, timestep)
                latents = scheduler.step(pred_noise, timestep, latents).prev_sample  # type: ignore
            latents_history.append(latents)

        # Get all the coefficients
        for step in range(steps + 1):
            out = latents_history[step + 1]

            # Coeff. previous latents (out = alpha * X + res)
            X = latents_history[step]
            alpha, out = find_scale_and_residual(out, X)
            self.scale.data[step] = alpha

            if step == 0:
                continue

            # Coeff. previous noise-pred (out = beta * Y + res)
            Ys = noise_pred_history[:step][-order:]
            for o, Y in enumerate(reversed(Ys)):
                beta, out = find_scale_and_residual(out, Y)
                self.scale_pred.data[step - 1, o] = beta

            # Try all noises...
            gammas: list[Tensor] = []
            ids_to_remove: list[int] = []
            for i, noise in enumerate(noise_history):
                gamma, out = find_scale_and_residual(out, noise)
                if abs(gamma) > 1e-8:  # !
                    gammas.append(gamma)
                    ids_to_remove.append(i)
            for i in ids_to_remove:
                del noise_history[i]
            if len(gammas) > 0:
                self.noise.data[step, : len(gammas)] = torch.tensor(gammas).sort(descending=True).values

        self.trim_order()

        return self

    # TODO BUGGY
    # @classmethod
    # def __from_sampler(
    #     cls,
    #     sampler: Sampler,
    #     /,
    #     steps: int,
    #     order: int = 1,
    # ) -> Self:
    #     # Memory
    #     sigma_history: list[Tensor] = []
    #     latents_history: list[Tensor] = []
    #     noise_pred_history: list[Tensor] = []
    #     noise_history: list[Tensor] = []

    #     sigmas = get_sigmas_karras(steps, 0.0292, 14.6146)  # TODO !!!

    #     max_size = 4 * steps * 50
    #     with HijackRandom(sigma_history, latents_history, noise_pred_history, noise_history, max_size) as ctx:
    #         latents = ctx.sample_noise()
    #         latents_history.append(latents)

    #         latents = sampler(ctx.model, latents, sigmas)
    #         latents_history.append(latents)

    #     self = cls(steps, order, steps * 50)  # ! needed, to account for brownian noise

    #     for step in range(steps + 1):
    #         out = latents_history[step + 1]

    #         # Coeff. previous latents (out = alpha * X + res)
    #         X = latents_history[step]
    #         alpha, out = find_scale_and_residual(out, X)
    #         self.scale.data[step] = alpha

    #         if step == 0:
    #             continue

    #         # Coeff. previous noise-pred (out = beta * Y + res)
    #         Ys = noise_pred_history[:step][-order:]
    #         for o, Y in enumerate(reversed(Ys)):
    #             beta, out = find_scale_and_residual(out, Y)
    #             self.scale_pred.data[step - 1, o] = beta

    #         # Try all noises...
    #         gammas: list[float] = []
    #         ids_to_remove: list[int] = []
    #         for i, noise in enumerate(noise_history):
    #             gamma, out = find_scale_and_residual(out, noise)
    #             if abs(gamma) > 1e-8:  # !
    #                 gammas.append(gamma)
    #                 ids_to_remove.append(i)
    #         for i in ids_to_remove:
    #             del noise_history[i]
    #         if len(gammas) > 0:
    #             self.noise.data[step, : len(gammas)] = torch.tensor(gammas).sort(descending=True).values

    #     sigmas = torch.stack(sigma_history)
    #     self.timesteps.data = sigmas_to_timesteps(sigmas)

    #     self.trim_order()

    #     return self


class GuidanceScheduler(BaseScheduler):
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


class PlotScheduler(BaseScheduler):
    def plot(self):
        fig, axes = plt.subplots(1, 3, figsize=(12, 3))

        norm = lambda x: x.detach().float().cpu()

        axes[0].plot(norm(self.scale), c="k", label="Latents scale")
        axes[0].plot(norm(self.scale_pred), label="Noise pred. scale")
        axes[0].plot(norm(self.noise), c="g", label="Noise")

        axes[1].plot(norm(self.timesteps), label="Timesteps")

        axes[2].plot(norm(self.guidance_scale), label="Guidance scale")

        for ax in axes:
            ax.set_xlabel("Steps (1)")
            ax.legend()

        plt.tight_layout()
        plt.show()


class Scheduler(ConvertScheduler, GuidanceScheduler, PlotScheduler, BaseModule):
    def __init__(
        self,
        steps: int,
        order: int = 1,
        noise_order: int = 1,
    ) -> None:
        super().__init__()

        order = min(order, steps)

        # step=0 scale and add noise to the latents prior to calling the model, jhence steps + 1
        self.scale = nn.Parameter(torch.zeros(steps + 1))
        self.noise = nn.Parameter(torch.zeros(steps + 1, noise_order))
        self.scale_pred = nn.Parameter(torch.zeros(steps, order))
        self.guidance_scale = nn.Parameter(torch.ones(steps))
        self.timesteps = nn.Parameter(torch.zeros(steps))

        self.eval()

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
        if isinstance(x, Tensor):
            self.to(device=x.device, dtype=x.dtype)
            shape = x.shape
        else:
            shape = x

        self.seed, generator = self.make_generator(seed)
        eps = lambda: torch.randn(*shape, device=self.device, dtype=self.dtype, generator=generator)
        if not isinstance(x, Tensor):
            x = eps()

        prev_np: list[Tensor] = []
        for step in trange(self.steps + 1, leave=leave, disable=disable):
            if step > 0:
                noise_pred = model(x, self.timesteps[step - 1], self.guidance_scale[step - 1])
                prev_np.append(noise_pred)
                prev_np = prev_np[-self.order :]

            x = x * self.scale[step]
            for order in range(self.noise_order):
                x = x + eps() * self.noise[step, order]
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
