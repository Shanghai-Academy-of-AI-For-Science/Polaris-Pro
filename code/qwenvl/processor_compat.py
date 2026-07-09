"""Compatibility helpers for loading Qwen processors across transformers versions."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from transformers import AutoProcessor


_FIX_MISTRAL_REGEX_FRAGMENT = "fix_mistral_regex"
_HEAVY_FILE_SUFFIXES = (
    ".bin",
    ".ckpt",
    ".h5",
    ".msgpack",
    ".onnx",
    ".pt",
    ".pth",
    ".safetensors",
)
_HEAVY_FILE_NAMES = {
    "optimizer.pt",
    "rng_state.pth",
    "scheduler.pt",
    "training_args.bin",
}


def _symlink_or_copy(src: Path, dst: Path) -> None:
    try:
        os.symlink(src, dst, target_is_directory=src.is_dir())
        return
    except OSError:
        pass
    if src.is_dir():
        shutil.copytree(src, dst, symlinks=True)
    else:
        shutil.copy2(src, dst)


def _should_stage_processor_entry(path: Path) -> bool:
    if path.is_dir():
        return True
    name = path.name.lower()
    if name in _HEAVY_FILE_NAMES:
        return False
    return not name.endswith(_HEAVY_FILE_SUFFIXES)


def _stage_processor_dir_without_fix_mistral_regex(src_dir: Path, dst_dir: Path) -> bool:
    removed = False
    for src in src_dir.iterdir():
        if not _should_stage_processor_entry(src):
            continue
        dst = dst_dir / src.name
        if src.name == "tokenizer_config.json":
            cfg = json.loads(src.read_text())
            removed = _FIX_MISTRAL_REGEX_FRAGMENT in cfg
            cfg.pop(_FIX_MISTRAL_REGEX_FRAGMENT, None)
            dst.write_text(json.dumps(cfg, ensure_ascii=False, indent=2))
        else:
            _symlink_or_copy(src, dst)
    return removed


def load_auto_processor_compat(pretrained_model_name_or_path: str | os.PathLike[str], **kwargs: Any):
    """Load ``AutoProcessor`` with a fallback for a transformers 5 tokenizer bug.

    Some saved Qwen tokenizer configs contain ``fix_mistral_regex``. In affected
    transformers builds that key is forwarded twice and processor loading raises
    ``TypeError: ... got multiple values for keyword argument 'fix_mistral_regex'``.
    The fallback stages a temporary processor-only view of the checkpoint with
    that single key removed, leaving the checkpoint itself untouched.
    """

    try:
        return AutoProcessor.from_pretrained(pretrained_model_name_or_path, **kwargs)
    except TypeError as exc:
        msg = str(exc)
        if _FIX_MISTRAL_REGEX_FRAGMENT not in msg or "multiple values" not in msg:
            raise

        src_dir = Path(pretrained_model_name_or_path)
        if not src_dir.is_dir():
            raise

        tmp = tempfile.TemporaryDirectory(prefix="qwen_processor_compat_")
        try:
            removed = _stage_processor_dir_without_fix_mistral_regex(src_dir, Path(tmp.name))
            processor = AutoProcessor.from_pretrained(tmp.name, **kwargs)
        except Exception:
            tmp.cleanup()
            raise

        # Keep the staged files alive for any lazy tokenizer/processor access.
        processor._processor_compat_tmpdir = tmp
        if removed:
            print(
                "[processor] Retried with a temporary tokenizer_config.json "
                "without fix_mistral_regex"
            )
        return processor
