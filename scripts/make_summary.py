import json
import os
import re
import sys
from pathlib import Path

OVARE = re.compile(r"OVA_AVG_ACCURACY=([0-9]*\.?[0-9]+)")
BEST_VAL_RE = re.compile(r"Best Val Acc.*?:\s*([0-9]*\.?[0-9]+)")
BEST_TEST_RE = re.compile(r"Test Acc(?:\s*@ Best Val)?:\s*([0-9]*\.?[0-9]+)")
SMOKE_RE = re.compile(r"CPU_SMOKE_OK=1.*?VAL_ACC=([0-9]*\.?[0-9]+).*?TEST_ACC=([0-9]*\.?[0-9]+)")


def parse_log(path: Path):
    res = {"log": str(path)}
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return res

    m = OVARE.search(text)
    if m:
        try:
            res["ova_avg_accuracy"] = float(m.group(1))
        except Exception:
            pass

    m = SMOKE_RE.search(text)
    if m:
        try:
            res["cpu_smoke_val_acc"] = float(m.group(1))
            res["cpu_smoke_test_acc"] = float(m.group(2))
        except Exception:
            pass

    # Pick the last occurrence as final best
    best_vals = BEST_VAL_RE.findall(text)
    if best_vals:
        try:
            res["best_val_acc"] = float(best_vals[-1])
        except Exception:
            pass
    best_tests = BEST_TEST_RE.findall(text)
    if best_tests:
        try:
            res["best_test_acc"] = float(best_tests[-1])
        except Exception:
            pass

    return res


def main():
    repo_root = Path(__file__).resolve().parents[1]
    logs_dir = repo_root / "results" / "logs"
    out_path = repo_root / "results" / "summary.json"

    # Allow overriding input/out via args
    if len(sys.argv) > 1:
        logs_dir = Path(sys.argv[1])
    if len(sys.argv) > 2:
        out_path = Path(sys.argv[2])

    logs_dir.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    runs = []
    if logs_dir.exists():
        for p in sorted(logs_dir.glob("*.txt")):
            runs.append(parse_log(p))
    else:
        print(f"Logs directory not found: {logs_dir}")

    payload = {"runs": runs}
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote summary to {out_path} with {len(runs)} runs")


if __name__ == "__main__":
    main()
