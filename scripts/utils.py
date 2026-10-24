"""Shared utilities for the LocalLLM pipeline."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import unicodedata
from collections.abc import Iterable
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def prompt_fingerprint(text: str) -> str:
    """Conservative identity for deduplication and cross-split leakage checks."""
    canonical = " ".join(unicodedata.normalize("NFC", text).casefold().split())
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def setup_logging(name: str, log_file: str | None = None) -> logging.Logger:
    """Return an idempotently configured logger."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger
    formatter = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_file:
        path = PROJECT_ROOT / "logs" / log_file
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def load_config(config_path: str = "configs/project_config.yaml") -> dict[str, Any]:
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError("PyYAML is required to load configuration. Install dependencies with `pip install -r requirements.txt`.") from error
    with (PROJECT_ROOT / config_path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save_jsonl(data: Iterable[dict[str, Any]], output_path: str | Path) -> int:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for item in data:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            count += 1
    return count


def load_jsonl(input_path: str | Path) -> list[dict[str, Any]]:
    path = Path(input_path)
    data: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path}:{line_number}") from error
    return data


def require_file(path: Path, message: str = "") -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}. {message}".strip())
    return path


def get_gpu_info() -> dict[str, Any]:
    configure_nix_cuda_runtime()
    try:
        import torch
        if not torch.cuda.is_available():
            return {"available": False}
        return {"available": True, "device_count": torch.cuda.device_count(),
                "device_name": torch.cuda.get_device_name(0),
                "total_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)}
    except ImportError as error:
        return {"available": False, "reason": str(error)}


def configure_nix_cuda_runtime() -> None:
    """Expose NVIDIA and C++ runtime libraries to pip wheels on Nix hosts.

    This is a no-op on conventional Linux distributions.
    """
    paths = [Path("/run/opengl-driver/lib")]
    if shutil.which("g++"):
        try:
            paths.append(Path(subprocess.check_output(["g++", "-print-file-name=libstdc++.so.6"], text=True).strip()).parent)
        except (OSError, subprocess.SubprocessError):
            pass
    usable = [str(path) for path in paths if path.is_dir()]
    if usable:
        os.environ["LD_LIBRARY_PATH"] = ":".join(usable + [os.environ.get("LD_LIBRARY_PATH", "")])


def environment(name: str, default: str | None = None) -> str | None:
    """Read an environment variable without printing its value."""
    return os.getenv(name, default)
