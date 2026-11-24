"""LeukemiPrediag — run the project locally (macOS, Linux, Windows).

Run from the project folder with the virtual environment's Python:

    python run_local.py check        # verify files against the Drive snapshot
    python run_local.py notebooks    # (re)create notebooks/*.local.ipynb for Jupyter
    python run_local.py v27          # v2.7 feature-robustness pipeline + unit tests
    python run_local.py tests        # every test suite present on this branch
    python run_local.py v28 [--embeddings]   # v2.8 diagram pipeline (v2.8 branch only)

Research use only. Not a diagnostic device.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
V2 = ROOT / "data" / "processed" / "v2"
PY = sys.executable


def run(cmd, env=None) -> None:
    print("\n$ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], cwd=ROOT, check=True, env=env)


def src_env() -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    return env


def cmd_check(_):
    run([PY, "scripts/check_data.py"])


def cmd_notebooks(_):
    run([PY, "scripts/make_local_notebooks.py"])
    print("\nStart Jupyter with:  python -m jupyter lab notebooks")
    print("Open the *.local.ipynb copies. Active flow: 12 -> 13 -> 14 -> 15 -> 16 -> 17 -> 18 -> 19 -> 20.")
    print("Do not run 10/11 (fusion training/inference) until outcome-linked data exist (CLAUDE.md §11).")


def cmd_v27(args):
    out = ROOT / "outputs" / "v27_local"
    out.mkdir(parents=True, exist_ok=True)
    run([PY, "src/feature_robustness_v27.py", "--filtered", V2 / "real_serial_filtered.csv",
         "--unfiltered", V2 / "real_serial_unfiltered.csv", "--outdir", out,
         "--draws", str(args.draws), "--seed", str(args.seed)])
    run([PY, "src/finalize_v27.py", out])
    run([PY, "-m", "unittest", "-v", "src/test_feature_robustness_v27.py"], env=src_env())
    print(f"\nDONE. Outputs: {out}")


def cmd_tests(_):
    run([PY, "-m", "unittest", "-v", "src/test_feature_robustness_v27.py"], env=src_env())
    if (ROOT / "tests").is_dir():
        run([PY, "-m", "pytest", "-q", "-rs", "tests"])
    else:
        print("\n(no tests/ folder on this branch; switch to review/v2.7.1 or v2.8-diagram-pipeline for the pytest suites)")


def cmd_v28(args):
    if not (ROOT / "src" / "pipeline_v28.py").exists():
        sys.exit("pipeline_v28.py is not on this branch. Run:  git switch v2.8-diagram-pipeline")
    out = ROOT / "outputs" / ("v28_with_embeddings" if args.embeddings else "v28_local")
    cmd = [PY, "src/pipeline_v28.py", "--filtered", V2 / "real_serial_filtered.csv",
           "--unfiltered", V2 / "real_serial_unfiltered.csv", "--outdir", out]
    if args.embeddings:
        table = ROOT / "data" / "processed" / "v2_molecular" / "esm2_delta_table.pkl"
        if not table.exists():
            run([PY, "scripts/make_esm2_delta_table.py"])
        cmd += ["--embeddings", table]
    run(cmd)
    print(f"\nDONE. Outputs: {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run LeukemiPrediag locally")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check").set_defaults(fn=cmd_check)
    sub.add_parser("notebooks").set_defaults(fn=cmd_notebooks)
    p = sub.add_parser("v27"); p.add_argument("--draws", type=int, default=500)
    p.add_argument("--seed", type=int, default=17); p.set_defaults(fn=cmd_v27)
    sub.add_parser("tests").set_defaults(fn=cmd_tests)
    p = sub.add_parser("v28"); p.add_argument("--embeddings", action="store_true"); p.set_defaults(fn=cmd_v28)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
