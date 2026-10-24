"""Cache Python-heavy GPU packages on native WSL storage; keep large libraries on D:."""

import argparse
import json
import os
import subprocess
import zipfile
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("environment", choices=["train", "serve"])
parser.add_argument("--extract", action="store_true")
args = parser.parse_args()

if args.extract:
    source = (
        Path("/mnt/d/local-llm-runtime")
        / args.environment
        / "lib/python3.12/site-packages"
    )
    target = Path.home() / ".cache/local-llm/native-packages" / args.environment / "site-packages"
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        f"/mnt/d/local-llm-runtime/{args.environment}-python.zip"
    ) as archive:
        archive.extractall(target)
    links = json.loads((target / "_localllm_links.json").read_text())
    for relative, is_directory in links:
        link = target / relative
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.is_symlink() and not link.exists():
            link.symlink_to(source / relative, target_is_directory=is_directory)
    (target / ".ready").write_text("complete\n")
    print(f"Native Python overlay: {target}", flush=True)
else:
    source = (
        Path("D:/local-llm-runtime") / args.environment / "lib/python3.12/site-packages"
    )
    archive_path = Path(f"D:/local-llm-runtime/{args.environment}-python.zip")
    count, links = 0, []
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_STORED) as archive:
        for directory, dirs, files in os.walk(source):
            kept = []
            for name in dirs:
                if name == "__pycache__":
                    continue
                if name in ["nvidia", "lib", "include", "bin"] or name.endswith(
                    ".libs"
                ):
                    links.append(
                        (
                            str((Path(directory) / name).relative_to(source)).replace(
                                "\\", "/"
                            ),
                            True,
                        )
                    )
                else:
                    kept.append(name)
            dirs[:] = kept
            for name in files:
                file = Path(directory) / name
                relative = str(file.relative_to(source)).replace("\\", "/")
                if file.suffix == ".pyc":
                    continue
                text_extensions = {
                    ".py",
                    ".pyi",
                    ".json",
                    ".yaml",
                    ".yml",
                    ".txt",
                    ".jinja",
                    ".jinja2",
                    ".md",
                    ".cfg",
                    ".ini",
                    ".toml",
                }
                is_metadata = ".dist-info/" in relative
                if (
                    not (file.suffix in text_extensions or is_metadata)
                    or file.stat().st_size > 5 * 1024 * 1024
                ):
                    links.append((relative, False))
                else:
                    archive.write(file, relative)
                    count += 1
        archive.writestr("_localllm_links.json", json.dumps(links))
    print(f"Archived {count} files, {archive_path.stat().st_size} bytes", flush=True)
    script = "/mnt/c/Users/Krish/local-llm/scripts/prepare_native_overlay.py"
    subprocess.run(
        ["wsl", "-d", "Ubuntu", "--", "python3", script, args.environment, "--extract"],
        check=True,
    )
