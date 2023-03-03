from __future__ import annotations
from typing import Optional

from torch import Tensor

from .protocols import Model, Callback, ExtraArgs
from .schedulers import Euler, EulerAncestral, Heun, DPM2, DPM2Ancestral, DPMpp2SAncestral, DPMpp2M


def sample_euler(
    model: Model,
    x: Tensor,
    sigmas: Tensor,
    extra_args: Optional[ExtraArgs] = None,
    callback: Optional[Callback] = None,
    disable: bool = False,
    s_churn: float = 0,
    s_tmin: float = 0,
    s_tmax: float = float("inf"),
    s_noise: float = 1,
    *,
    smooth_steps: int = 0,
    smooth_weight: float = 0.95
) -> Tensor:
    scheduler = Euler(sigmas, s_churn, s_tmin, s_tmax, s_noise).smooth(smooth_steps, smooth_weight)
    return scheduler.sample(model, x, extra_args, callback, disable)


def sample_euler_ancestral(
    model: Model,
    x: Tensor,
    sigmas: Tensor,
    extra_args: Optional[ExtraArgs] = None,
    callback: Optional[Callback] = None,
    disable: bool = False,
    eta: float = 1,
    s_noise: float = 1,
    *,
    smooth_steps: int = 0,
    smooth_weight: float = 0.95
) -> Tensor:
    scheduler = EulerAncestral(sigmas, eta, s_noise).smooth(smooth_steps, smooth_weight)
    return scheduler.sample(model, x, extra_args, callback, disable)


def sample_heun(
    model: Model,
    x: Tensor,
    sigmas: Tensor,
    extra_args: Optional[ExtraArgs] = None,
    callback: Optional[Callback] = None,
    disable: bool = False,
    s_churn: float = 0,
    s_tmin: float = 0,
    s_tmax: float = float("inf"),
    s_noise: float = 1,
    *,
    smooth_steps: int = 0,
    smooth_weight: float = 0.95
) -> Tensor:
    scheduler = Heun(sigmas, s_churn, s_tmin, s_tmax, s_noise).smooth(smooth_steps, smooth_weight)
    return scheduler.sample(model, x, extra_args, callback, disable)


def sample_dpm_2(
    model: Model,
    x: Tensor,
    sigmas: Tensor,
    extra_args: Optional[ExtraArgs] = None,
    callback: Optional[Callback] = None,
    disable: bool = False,
    s_churn: float = 0,
    s_tmin: float = 0,
    s_tmax: float = float("inf"),
    s_noise: float = 1,
    *,
    smooth_steps: int = 0,
    smooth_weight: float = 0.95
) -> Tensor:
    scheduler = DPM2(sigmas, s_churn, s_tmin, s_tmax, s_noise).smooth(smooth_steps, smooth_weight)
    return scheduler.sample(model, x, extra_args, callback, disable)


def sample_dpm_2_ancestral(
    model: Model,
    x: Tensor,
    sigmas: Tensor,
    extra_args: Optional[ExtraArgs] = None,
    callback: Optional[Callback] = None,
    disable: bool = False,
    eta: float = 1,
    s_noise: float = 1,
    *,
    smooth_steps: int = 0,
    smooth_weight: float = 0.95
) -> Tensor:
    scheduler = DPM2Ancestral(sigmas, eta, s_noise).smooth(smooth_steps, smooth_weight)
    return scheduler.sample(model, x, extra_args, callback, disable)


def sample_dpmpp_2s_ancestral(
    model: Model,
    x: Tensor,
    sigmas: Tensor,
    extra_args: Optional[ExtraArgs] = None,
    callback: Optional[Callback] = None,
    disable: bool = False,
    eta: float = 1,
    s_noise: float = 1,
    *,
    smooth_steps: int = 0,
    smooth_weight: float = 0.95
) -> Tensor:
    scheduler = DPMpp2SAncestral(sigmas, eta, s_noise).smooth(smooth_steps, smooth_weight)
    return scheduler.sample(model, x, extra_args, callback, disable)


def sample_dpmpp_2m(
    model: Model,
    x: Tensor,
    sigmas: Tensor,
    extra_args: Optional[ExtraArgs] = None,
    callback: Optional[Callback] = None,
    disable: bool = False,
    *,
    smooth_steps: int = 0,
    smooth_weight: float = 0.95
) -> Tensor:
    scheduler = DPMpp2M(sigmas).smooth(smooth_steps, smooth_weight)
    return scheduler.sample(model, x, extra_args, callback, disable)
