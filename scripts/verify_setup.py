"""Validate the LocalLLM runtime without requiring a GPU."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

from utils import PROJECT_ROOT, get_gpu_info


def main() -> int:
    errors = []
    print(f"Python: {sys.version.split()[0]}")
    if sys.version_info < (3, 10):
        errors.append("Python 3.10+ is required")
    required_dirs = ["data", "configs", "scripts", "src/gui", "outputs", "results", "memory"]
    for directory in required_dirs:
        ok = (PROJECT_ROOT / directory).is_dir()
        print(f"{'OK' if ok else 'MISSING'} directory: {directory}")
        if not ok:
            errors.append(f"Missing directory: {directory}")
    for module in ("yaml", "requests"):
        try:
            imported = importlib.import_module(module)
            print(f"OK dependency: {module} {getattr(imported, '__version__', '')}".strip())
        except ImportError:
            errors.append(f"Missing dependency: {module}")
    print(f"GPU: {get_gpu_info()}")
    if errors:
        print("\nSetup needs attention:\n- " + "\n- ".join(errors))
        return 1
    print("\nSetup structure is valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
