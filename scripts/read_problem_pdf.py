"""Extract the original problem statement and render every page for review."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tmp/pdf-tools"))
import pymupdf
from pypdf import PdfReader

root = Path(__file__).resolve().parents[1]
source = root / "LocalLLM PS.pdf"
out = root / "tmp/pdfs"
out.mkdir(parents=True, exist_ok=True)
reader = PdfReader(source)
text = "\n\n".join(f"PAGE {i + 1}\n{page.extract_text()}" for i, page in enumerate(reader.pages))
(out / "problem-statement.txt").write_text(text, encoding="utf-8")
with pymupdf.open(source) as doc:
    for i, page in enumerate(doc):
        page.get_pixmap(matrix=pymupdf.Matrix(1, 1)).save(str(out / f"page-{i+1}.png"))
print(text)
