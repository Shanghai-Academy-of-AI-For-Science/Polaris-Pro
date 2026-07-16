"""Checkpoint loading utilities."""
from __future__ import annotations

import json
import os
from glob import glob
from pathlib import Path

import torch


def load_state_dict_from_ckpt_dir(ckpt_dir):
    """Read a flat CPU ``state_dict`` from a HuggingFace-style checkpoint dir.

    Supports safetensors (sharded or single-file) and ``pytorch_model.bin``.
    """
    ckpt_dir = str(Path(ckpt_dir))

    index_file = os.path.join(ckpt_dir, "model.safetensors.index.json")
    if os.path.isfile(index_file):
        from safetensors.torch import load_file
        with open(index_file, "r") as f:
            index = json.load(f)
        shard_files = sorted(set(index["weight_map"].values()))
        state = {}
        for shard in shard_files:
            state.update(load_file(os.path.join(ckpt_dir, shard), device="cpu"))
        return state

    single_st = os.path.join(ckpt_dir, "model.safetensors")
    if os.path.isfile(single_st):
        from safetensors.torch import load_file
        return load_file(single_st, device="cpu")

    bin_index = os.path.join(ckpt_dir, "pytorch_model.bin.index.json")
    if os.path.isfile(bin_index):
        with open(bin_index, "r") as f:
            index = json.load(f)
        shard_files = sorted(set(index["weight_map"].values()))
        state = {}
        for shard in shard_files:
            state.update(torch.load(os.path.join(ckpt_dir, shard), map_location="cpu"))
        return state

    single_bin = os.path.join(ckpt_dir, "pytorch_model.bin")
    if os.path.isfile(single_bin):
        return torch.load(single_bin, map_location="cpu")

    candidates = sorted(glob(os.path.join(ckpt_dir, "*.safetensors")))
    if candidates:
        from safetensors.torch import load_file
        state = {}
        for c in candidates:
            state.update(load_file(c, device="cpu"))
        return state

    raise FileNotFoundError(
        f"No model weights found in {ckpt_dir} "
        f"(looked for model.safetensors[.index.json] and pytorch_model.bin[.index.json])"
    )
