"""Load a torch distributed checkpoint or a bf16 safetensors file into a module."""

from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from torch import nn
from torch.distributed.checkpoint import FileSystemReader, load

FLOW_MAP_KEY = "ref_time_in.in_layer.weight"


def safetensors_file(path: str) -> Path | None:
    path = Path(path)
    if path.suffix == ".safetensors":
        return path
    return path / "model.safetensors" if (path / "model.safetensors").exists() else None


def checkpoint_keys(path: str) -> set[str]:
    if file := safetensors_file(path):
        with safe_open(file, "pt") as f:
            return set(f.keys())
    return {k.removeprefix("model.") for k in FileSystemReader(path).read_metadata().state_dict_metadata}


def is_flow_map(path: str) -> bool:
    return FLOW_MAP_KEY in checkpoint_keys(path)


def load_checkpoint(module: nn.Module, path: str, device: str, dtype: torch.dtype) -> nn.Module:
    """Reads only the keys the module has. A safetensors file must match them exactly."""
    if file := safetensors_file(path):
        return _load_safetensors(module, file, device, dtype)
    state = {
        f"model.{name}": torch.empty(tensor.shape, device=device, dtype=dtype if tensor.is_floating_point() else tensor.dtype)
        for name, tensor in module.state_dict().items()
    }
    load(state, checkpoint_id=path)
    module.load_state_dict({name[len("model."):]: tensor for name, tensor in state.items()}, assign=True)
    return module.to(device).eval()


def _load_safetensors(module: nn.Module, file: Path, device: str, dtype: torch.dtype) -> nn.Module:
    state = load_file(str(file), device=str(device))
    expected = module.state_dict()
    missing, unexpected = sorted(set(expected) - set(state)), sorted(set(state) - set(expected))
    if missing or unexpected:
        raise KeyError(f"{file}: missing {missing[:5]}, unexpected {unexpected[:5]}")
    state = {k: v.to(dtype) if v.is_floating_point() else v for k, v in state.items()}
    module.load_state_dict(state, strict=True, assign=True)
    return module.to(device).eval()
