"""Fetch pinned model files into the configured shared Hugging Face cache."""

import concurrent.futures
import json
import os
from pathlib import Path

import requests

model = "Qwen/Qwen2.5-1.5B-Instruct"
response = requests.get(f"https://huggingface.co/api/models/{model}", timeout=30)
response.raise_for_status()
info = response.json()
revision = info["sha"]
runtime = Path(
    os.environ.get(
        "LOCALLLM_RUNTIME",
        "D:/local-llm-runtime" if os.name == "nt" else "/mnt/d/local-llm-runtime",
    )
)
path = runtime / "models/Qwen2.5-1.5B-Instruct" / revision
path.mkdir(parents=True, exist_ok=True)
files = [
    r["rfilename"]
    for r in info["siblings"]
    if "/" not in r["rfilename"]
    and (
        r["rfilename"].endswith((".json", ".safetensors", ".txt", ".jinja"))
        or r["rfilename"] in ["LICENSE", "README.md"]
    )
]


def download(name):
    target = path / name
    if target.exists():
        return
    partial = target.with_suffix(target.suffix + ".part")
    with requests.get(
        f"https://huggingface.co/{model}/resolve/{revision}/{name}",
        stream=True,
        timeout=(30, 120),
    ) as response:
        response.raise_for_status()
        with partial.open("wb") as output:
            for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                output.write(chunk)
    partial.replace(target)
    print(f"Downloaded {name}: {target.stat().st_size} bytes", flush=True)


with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(download, files))
root = Path(__file__).resolve().parents[1]
linux_path = (
    str(path).replace("D:\\", "/mnt/d/").replace("\\", "/")
    if os.name == "nt"
    else str(path)
)
report = {
    "model": model,
    "revision": revision,
    "path": linux_path,
    "files": {name: (path / name).stat().st_size for name in files},
}
(root / "results/reports/model_download.json").write_text(
    json.dumps(report, indent=2), encoding="utf-8"
)
print(json.dumps(report, indent=2))
