"""Generate standard top_level metadata to avoid DrvFS RECORD scans at TRL import."""

import csv
from pathlib import Path

for environment in ["train", "serve"]:
    root = Path("D:/local-llm-runtime") / environment / "lib/python3.12/site-packages"
    count = 0
    for directory in root.glob("*.dist-info"):
        target = directory / "top_level.txt"
        record = directory / "RECORD"
        if target.exists() or not record.exists():
            continue
        names = set()
        with record.open(encoding="utf-8", newline="") as stream:
            for row in csv.reader(stream):
                first = row[0].split("/")[0]
                if first.endswith((".dist-info", ".data")) or first.startswith("."):
                    continue
                name = first.split(".")[0]
                if name.isidentifier():
                    names.add(name)
        if names:
            target.write_text("\n".join(sorted(names)) + "\n", encoding="utf-8")
            count += 1
    print(f"{environment}: generated top_level.txt for {count} distributions")
