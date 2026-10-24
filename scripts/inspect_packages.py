"""Inspect published package constraints before selecting compatible environments."""
import json
import urllib.request

for package, version in [("aligntune", "0.1.28"), ("sglang", "0.5.9"), ("curatorkit", "0.1.22"), ("auditkit", "1.1.3")]:
    with urllib.request.urlopen(f"https://pypi.org/pypi/{package}/{version}/json") as response:
        info = json.load(response)["info"]
    print(package, version)
    for requirement in info.get("requires_dist") or []:
        print(" ", requirement)
