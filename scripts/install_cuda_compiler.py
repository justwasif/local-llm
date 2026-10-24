"""Install NVIDIA's published CUDA 12.8.1 compiler archive with checksum verification."""

import hashlib
import json
import tarfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
runtime = Path("D:/local-llm-runtime")
base = "https://developer.download.nvidia.com/compute/cuda/redist/"
response = requests.get(base + "redistrib_12.8.1.json", timeout=30)
response.raise_for_status()
info = response.json()["cuda_nvcc"]["linux-x86_64"]
archive_path = runtime / "cuda-nvcc-12.8.1.tar.xz"
if not archive_path.exists():
    response = requests.get(base + info["relative_path"], timeout=120)
    response.raise_for_status()
    archive_path.write_bytes(response.content)
if hashlib.sha256(archive_path.read_bytes()).hexdigest() != info["sha256"]:
    raise ValueError("NVIDIA compiler archive checksum mismatch")
target = runtime / "cuda-12.8"
target.mkdir(exist_ok=True)
with tarfile.open(archive_path) as archive:
    for member in archive.getmembers():
        if not member.isfile():
            continue
        relative = Path(*Path(member.name).parts[1:])
        if ".." in relative.parts:
            raise ValueError("Unexpected compiler archive path")
        output = target / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(archive.extractfile(member).read())
report = {
    "cuda_toolkit_release": "12.8.1",
    "compiler_archive": info,
    "path": "/mnt/d/local-llm-runtime/cuda-12.8",
}
(ROOT / "results/reports/cuda_compiler.json").write_text(
    json.dumps(report, indent=2), encoding="utf-8"
)
print(json.dumps(report, indent=2))
