from __future__ import annotations
from typing import Tuple

from PIL import Image

import numpy as np
import torch
from torch import Tensor
from einops import rearrange

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, StableDiffusionPipeline


def load_model(
    pretrained_path: str,
    /,
    *,
    dtype: torch.dtype,
    device: torch.device,
    clip_skip: int = 1,
) -> Tuple[CLIPTokenizer, CLIPTextModel, UNet2DConditionModel, AutoencoderKL]:
    pipe = StableDiffusionPipeline.from_single_file(pretrained_path, local_files_only=True, use_safetensors=True)

    send = lambda model: model.requires_grad_(False).eval().to(dtype=dtype, device=device)

    tokenizer = pipe.tokenizer
    text_encoder = send(pipe.text_encoder)
    vae = send(pipe.vae)
    unet = send(pipe.unet)

    del pipe

    text_encoder.config.num_hidden_layers -= clip_skip - 1

    return tokenizer, text_encoder, unet, vae


def make_text_encoder(
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    /,
):
    cache: dict[str, Tensor] = {}  # local cache

    def encode_text(texts: str | list[str], /) -> list[Tensor]:
        texts = [texts] if isinstance(texts, str) else texts

        out: list[Tensor] = []
        for text in texts:
            if text not in cache:
                ids = tokenizer(
                    text,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                ).input_ids

                cache[text] = text_encoder(ids.to(text_encoder.device)).last_hidden_state

            out.append(cache[text])

        return out

    # ? How to add types to this? Useful?
    encode_text.clear = lambda: cache.clear()

    return encode_text


@torch.no_grad()
def decode_latents(
    vae: AutoencoderKL,
    latents: Tensor,
    /,
) -> list[Image.Image]:
    data = vae.decode(latents / vae.config.scaling_factor).sample  # type: ignore
    data = data.add(1).mul(255 / 2).round().clamp(0, 255).byte().cpu().numpy()
    data = rearrange(data, "b c h w -> b h w c")

    return [Image.fromarray(d) for d in data]


@torch.no_grad()
def encode_image(
    vae: AutoencoderKL,
    img: Image.Image,
    /,
) -> Tensor:
    img = img.convert("RGB")
    data = np.asarray(img)
    data = torch.from_numpy(data)  # ? clone first?
    data = data.to(dtype=vae.dtype, device=vae.device)
    data = data.div(255 / 2).sub(1)
    data = rearrange(data, "h w c -> 1 c h w")

    return vae.encode(data).latent_dist.sample() * vae.config.scaling_factor  # type: ignore
