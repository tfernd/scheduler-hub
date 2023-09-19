from __future__ import annotations
from typing import Optional

from tqdm.auto import tqdm

import torch
from torch import Tensor

from diffusers import UNet2DConditionModel

from .types import Schedulers, Model


def predict_noise(
    unet: UNet2DConditionModel,
    /,
    latents: Tensor,
    embeddings: list[Tensor],
    timestep: float | Tensor,
    guidance_scale: Optional[float | Tensor],
) -> Tensor:
    if len(embeddings) == 2 and guidance_scale is not None and guidance_scale == 1:
        # unconditional is not needed (it would be removed otherwise: u + (p-u) = p!)
        guidance_scale = None
        embeddings = [embeddings[1]]

    num_emb = len(embeddings)
    assert num_emb in (1, 2)

    batch_size = latents.size(0)
    latents = torch.cat([latents] * num_emb)
    emb = torch.cat(embeddings).repeat_interleave(batch_size, dim=0)

    pred_noise = unet(latents, timestep, emb).sample

    if num_emb == 1:  # Positive only
        assert guidance_scale is None
        return pred_noise

    assert guidance_scale is not None

    pred_negative, pred_positive = pred_noise.chunk(2)

    return pred_negative + (pred_positive - pred_negative) * guidance_scale


def make_noise_predictor(
    unet: UNet2DConditionModel,
    /,
    embeddings: list[Tensor],
) -> Model:
    def model(
        latents: Tensor,
        timestep: Tensor,
        /,
        guidance_scale: Optional[float | Tensor] = None,
    ) -> Tensor:
        return predict_noise(unet, latents, embeddings, timestep, guidance_scale)

    return model


def diffuse(
    model: Model,
    latents: Tensor,
    scheduler: Schedulers,
    /,
    *,
    guidance_scale: float,
    steps: int,
) -> Tensor:
    # Standard schedulers diffusion
    scheduler.set_timesteps(steps, device=latents.device)
    assert scheduler.timesteps is not None

    for step, timestep in enumerate(tqdm(scheduler.timesteps)):
        if step == 0:
            latents = latents * scheduler.init_noise_sigma

        input_latents = scheduler.scale_model_input(latents, timestep)  # type: ignore
        pred_noise = model(input_latents, timestep, guidance_scale)
        latents = scheduler.step(pred_noise, timestep, latents).prev_sample  # type: ignore

    return latents
