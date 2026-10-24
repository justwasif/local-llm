"""Freeze a small evaluation-only slice of the original, public FLORES-200 devtest."""
import hashlib
import json
import tarfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"
LANGUAGES = {"hin_Deva": "Hindi", "ben_Beng": "Bengali", "tam_Taml": "Tamil", "tel_Telu": "Telugu", "mar_Deva": "Marathi", "guj_Gujr": "Gujarati", "kan_Knda": "Kannada", "mal_Mlym": "Malayalam", "pan_Guru": "Punjabi", "ory_Orya": "Odia"}
archive_path = ROOT / "tmp/flores200_dataset.tar.gz"
archive_path.parent.mkdir(exist_ok=True)
if not archive_path.exists():
    response = requests.get(URL, timeout=120)
    response.raise_for_status()
    archive_path.write_bytes(response.content)
with tarfile.open(archive_path) as archive:
    def lines(code):
        member = next(m for m in archive.getmembers() if m.isfile() and m.name.endswith(f"/devtest/{code}.devtest"))
        return archive.extractfile(member).read().decode("utf-8").splitlines()
    english = lines("eng_Latn")
    rows = []
    for code, language in LANGUAGES.items():
        target = lines(code)
        for index in range(5):
            rows.append({"id": f"flores200-devtest-{code}-{index}", "instruction": f"Translate the following English sentence into {language}. Return only the translation.\n\n{english[index]}", "output": target[index], "source_text": english[index], "language": code, "source": URL, "split": "devtest", "source_row": index, "license": "CC-BY-SA-4.0", "usage": "evaluation_only"})
output = ROOT / "data/eval/flores200.jsonl"
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
report = {"dataset": "original FLORES-200 (not FLORES+)", "url": URL, "archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(), "license": "CC-BY-SA-4.0", "split": "devtest", "count": len(rows), "per_language": 5, "selection": "First five aligned devtest rows, fixed before model evaluation", "limitations": "Small pilot only; not full-benchmark performance. Never used for training."}
(ROOT / "results/reports/flores_provenance.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
