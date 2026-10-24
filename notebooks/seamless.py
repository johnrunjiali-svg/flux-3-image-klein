"""Seamless tiling hacks, shared by the notebooks: periodic RoPE for the DiT and wrap-around padding for the VAE decoder.

`seamless_tile_rope.ipynb` walks through the same code with explanations; this file is what the batch notebooks import.
Device selection is up to the caller (CUDA_VISIBLE_DEVICES before importing torch).
"""

import math
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor, nn

from flux3.autoencoder import latent_ids, latent_shape
from flux3.dit import AXES_DIM, IN_CHANNELS, THETA, DiT
from flux3.pipeline import IMAGE_ROPE_SCALE, MODELS, Pipeline, pack, scaled_ids, to_pil
from flux3.sampling import denoise, denoise_flow_map, timesteps

TWO_PI = 2 * math.pi
Tile = tuple[str, ...]  # subset of ("h", "w"): the axes made periodic


# ---------------------------------------------------------------- periodic RoPE


def axis_freqs(dim: int, theta: float = THETA) -> Tensor:
    # the original per-plane frequencies of one axis, as in flux3.dit.rope
    scale = torch.arange(0, dim, 2, dtype=torch.float64) / dim
    return 1.0 / (theta**scale)


def quantize_freqs(omega: Tensor, period: float, rule: str = "round") -> Tensor:
    # each frequency -> the nearest one whose rotation repeats after `period` id units: omega' = 2*pi*k / period
    cycles = omega * period / TWO_PI
    if rule == "round":
        k = torch.floor(cycles + 0.5)
    elif rule == "fundamental":
        k = torch.clamp(torch.floor(cycles + 0.5), min=1.0)
    elif rule == "keep":
        k = cycles
    else:
        raise ValueError(f"unknown rule {rule!r}")
    return k * TWO_PI / period


def rope_from_freqs(pos: Tensor, omega: Tensor) -> Tensor:
    # same as flux3.dit.rope, with an explicit frequency vector instead of (dim, theta)
    out = torch.einsum("...n,d->...nd", pos.to(omega.dtype), omega.to(pos.device))
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    return out.reshape(*out.shape[:-1], 2, 2).float()


def make_rope_embed(freqs: list[Tensor | None]):
    # ids (B, L, 4) -> (B, 1, L, head_dim // 2, 2, 2); freqs[i] is the frequency vector of axis i, None = original
    assert len(freqs) == len(AXES_DIM)
    omegas = [axis_freqs(dim) if f is None else f for f, dim in zip(freqs, AXES_DIM)]

    def rope_embed(ids: Tensor) -> Tensor:
        with torch.autocast(device_type=ids.device.type, enabled=False):
            emb = torch.cat([rope_from_freqs(ids[..., i], omegas[i]) for i in range(len(AXES_DIM))], dim=-3)
        return emb.unsqueeze(1)

    return rope_embed


def quantized_rope_embed(h: int, w: int, tile: Tile = ("h", "w"), rule: str = "round"):
    # h, w: latent grid size; tile: which of the "h" / "w" axes are made periodic; t and l axes untouched
    freqs: list[Tensor | None] = [None] * len(AXES_DIM)
    if "h" in tile:
        freqs[1] = quantize_freqs(axis_freqs(AXES_DIM[1]), h * IMAGE_ROPE_SCALE, rule)
    if "w" in tile:
        freqs[2] = quantize_freqs(axis_freqs(AXES_DIM[2]), w * IMAGE_ROPE_SCALE, rule)
    return make_rope_embed(freqs)


# ---------------------------------------------------------------- seamless VAE decoder


def pad_periodic(x: Tensor, tile: Tile) -> Tensor:
    # x (B, C, H, W) -> (B, C, H+2, W+2): one pixel on each side, wrap-around on the tiled axes, zero on the others
    x = F.pad(x, (1, 1, 0, 0), mode="circular" if "w" in tile else "constant")
    x = F.pad(x, (0, 0, 1, 1), mode="circular" if "h" in tile else "constant")
    return x


def conv3x3_periodic(conv: nn.Conv2d, x: Tensor, tile: Tile) -> Tensor:
    # same weights as `conv` (3x3, stride 1, padding 1), applied to the hand-padded input instead of conv's zero padding
    assert conv.kernel_size == (3, 3) and conv.stride == (1, 1) and conv.padding == (1, 1)
    return F.conv2d(pad_periodic(x, tile), conv.weight, conv.bias)


@contextmanager
def seamless_decoder(decoder: nn.Module, tile: Tile = ("h", "w")):
    # inside the block every 3x3 conv of `decoder` wraps around on the axes in `tile`; () leaves the decoder untouched
    convs = [m for m in decoder.modules() if isinstance(m, nn.Conv2d) and m.kernel_size == (3, 3)] if tile else []
    saved = [conv.forward for conv in convs]
    for conv in convs:
        conv.forward = lambda x, conv=conv: conv3x3_periodic(conv, x, tile)
    try:
        yield
    finally:
        for conv, forward in zip(convs, saved):
            conv.forward = forward


# ---------------------------------------------------------------- text-to-image with a chosen rope / decoder


def sample(pipe: Pipeline, dit: DiT, model: str, rope_embed, prompt: str, width: int, height: int,
           steps: int | None = None, guidance: float | None = None, seed: int = 1) -> tuple[Tensor, Tensor]:
    # mirrors Pipeline.generate for t2i (no references) -> latent tokens (h*w, 128) and ids (h*w, 4)
    dit.rope_embed = rope_embed
    steps = steps or MODELS[model]["steps"]
    guidance = MODELS[model]["guidance"] if guidance is None else guidance
    h, w = latent_shape(height, width)
    noise = torch.randn((h * w, IN_CHANNELS["image"]), generator=torch.Generator().manual_seed(seed))
    noise = noise.to(torch.bfloat16).float().to(pipe.device)
    ids = latent_ids(h, w, pipe.device)
    image = (noise, scaled_ids(ids))
    schedule = timesteps(steps, 16.0)
    if dit.flow_map:
        packed = pack([[pipe.text_encoder.encode(prompt), None, image]])
        tokens = denoise_flow_map(dit, packed, schedule)
    else:
        text = [pipe.text_encoder.encode(""), pipe.text_encoder.encode(prompt)]
        packed = pack([[text[0], None, image], [text[1], None, image]])
        tokens = denoise(dit, packed, schedule, guidance)
    return tokens, ids


def decode(pipe: Pipeline, tokens: Tensor, ids: Tensor, tile: Tile = ()) -> Image.Image:
    # tile: axes on which the VAE decoder wraps around; () is the stock decoder
    with seamless_decoder(pipe.autoencoder.decoder, tile):
        return to_pil(pipe.autoencoder.decode(tokens, ids))


# ---------------------------------------------------------------- prompt files and evaluation


def read_prompts(path: str | Path) -> dict[str, str]:
    # one "name: prompt" per line; blank lines and "#" comments ignored
    prompts = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        name, _, prompt = line.partition(":")
        assert prompt, f"no ': ' in line {line!r}"
        prompts[name.strip()] = prompt.strip()
    return prompts


def write_prompts(path: str | Path, prompts: dict[str, str]) -> None:
    # inverse of read_prompts; newlines inside a prompt are collapsed so one prompt stays one line
    lines = [f"{name}: {' '.join(prompt.split())}" for name, prompt in prompts.items()]
    Path(path).write_text("\n".join(lines) + "\n")


def tiled(image: Image.Image, reps: int = 2) -> Image.Image:
    canvas = Image.new("RGB", (image.width * reps, image.height * reps))
    for i in range(reps):
        for j in range(reps):
            canvas.paste(image, (j * image.width, i * image.height))
    return canvas


def seam_score(image: Image.Image) -> dict[str, float]:
    # mean |difference| across the wrap-around seams, against the mean over interior neighbours (1.0 = no visible seam)
    x = torch.from_numpy(np.asarray(image)).float()
    interior_w = (x[:, 1:] - x[:, :-1]).abs().mean().item()
    interior_h = (x[1:] - x[:-1]).abs().mean().item()
    wrap_w = (x[:, 0] - x[:, -1]).abs().mean().item()
    wrap_h = (x[0] - x[-1]).abs().mean().item()
    return {"w": wrap_w / interior_w, "h": wrap_h / interior_h}
