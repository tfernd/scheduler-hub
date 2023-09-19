from __future__ import annotations

import gc

import math
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


def flush():
    torch.cuda.empty_cache()
    gc.collect()


def orthonormal_factory(
    max_size: int,
    /,
    rand: bool = True,
):
    if rand:
        # TODO test and remove
        # eps = np.random.randn(max_size, max_size)
        # eps = torch.from_numpy(eps).to(dtype=torch.float64)

        eps = torch.randn(max_size, max_size, dtype=torch.float64)
        nn.init.orthogonal_(eps)
    else:
        eps = torch.eye(max_size, dtype=torch.float64)

    count = -1

    def sample() -> Tensor:
        nonlocal count
        count += 1

        assert count < max_size

        return eps[count].view(-1, 1, 1, 1)

    return sample


def find_scale_and_residual(x: Tensor, y: Tensor, /) -> tuple[Tensor, Tensor]:
    # Solves x == a y + res

    # (x-res).y == a y.y -> a = (x-res).y / y.y

    # res initialized as 0
    alpha = y.mul(x - 0).mean() / y.square().mean()
    res = x - alpha * y

    return alpha, res


# TODO Check this out
def sigmas_to_timesteps(sigmas_k: Tensor, /) -> Tensor:
    betas = torch.linspace(math.sqrt(0.00085), math.sqrt(0.012), 1000).square()
    alphas = 1 - betas
    alphas_cumprod = alphas.cumprod(0)
    sigmas = ((1 - alphas_cumprod) / alphas_cumprod).sqrt()

    dist = sigmas_k - sigmas[:, None]

    il, ih = dist.abs().argsort(0)[:2]
    low = sigmas[il].log()
    high = sigmas[ih].log()
    w = (low - sigmas_k.log()).div(low - high).clamp(0, 1)

    timesteps = il * (1 - w) + ih * w

    return timesteps
