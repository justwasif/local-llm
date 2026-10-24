"""Fetch published wheels for offline API inspection without importing CUDA."""
import io
import json
import urllib.request
import zipfile
from pathlib import Path

root = Path(__file__).resolve().parents[1] / "tmp/toolkit-sources"
for package, version in [("aligntune", "0.1.28"), ("curatorkit", "0.1.22"), ("auditkit", "1.1.3")]:
    with urllib.request.urlopen(f"https://pypi.org/pypi/{package}/{version}/json") as response:
        metadata = json.load(response)
    wheels = [f for f in metadata["urls"] if f["filename"].endswith(".whl")]
    wheel = next((f for f in wheels if "none-any" in f["filename"]), wheels[0])
    with urllib.request.urlopen(wheel["url"]) as response:
        data = response.read()
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for name in archive.namelist():
            if name.endswith((".py", ".yaml", ".json", "METADATA")):
                archive.extract(name, root / package)
    print(package, wheel["filename"], len(data))
