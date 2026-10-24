"""Run AuditKIT benchmark and safety reports against the SGLang endpoint."""
from __future__ import annotations
import shutil, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import PROJECT_ROOT
def main() -> int:
    if not shutil.which("auditkit"): print("AuditKIT is not installed.", file=sys.stderr); return 2
    benchmarks = PROJECT_ROOT / "results/benchmarks/eval_report.json"; safety = PROJECT_ROOT / "results/safety/safety_report.json"; benchmarks.parent.mkdir(parents=True, exist_ok=True); safety.parent.mkdir(parents=True, exist_ok=True)
    commands = [["auditkit", "eval", "--model", "outputs/merged_model", "--backend", "sglang", "--config", "configs/auditkit/eval_config.yaml", "--output", str(benchmarks)], ["auditkit", "safety", "--model", "outputs/merged_model", "--backend", "sglang", "--tests", "toxicity,bias,hallucination", "--output", str(safety)]]
    return max(subprocess.run(command, cwd=PROJECT_ROOT).returncode for command in commands)
if __name__ == "__main__": raise SystemExit(main())
