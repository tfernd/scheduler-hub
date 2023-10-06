from __future__ import annotations

import gc

import torch


def flush():
    torch.cuda.empty_cache()
    gc.collect()
