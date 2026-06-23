"""
Parse a benchmark experiment root and produce a CSV table matching the
Benchmark-1 quantitative results format:

  Scene | Model | T | R | SSIM | PSNR | LPIPS | FID | Δ PSNR vs O/O

Usage:
    python benchmark_table.py --root /path/to/exp_ds_bm_01 --out results.csv
    python benchmark_table.py --root /path/to/exp_ds_bm_01  # prints to stdout

Expected folder layout:
    {root}/{model}/{scene}_{pose_suffix}/eval/results.json

Recognised pose suffixes and their T/R mapping:
    optRT        → T=O  R=O
    optR_measT   → T=M  R=O
    optT_measR   → T=O  R=M
    measRT       → T=M  R=M
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


POSE_SUFFIXES = {
    "optRT":       ("O", "O"),
    "optR_measT":  ("M", "O"),
    "optT_measR":  ("O", "M"),
    "measRT":      ("M", "M"),
}

ROW_ORDER = ["measRT", "optR_measT", "optT_measR", "optRT"]

BASELINE_SUFFIX = "optRT"


def parse_scene_and_suffix(folder_name: str) -> tuple[str, str] | None:
    for suffix in POSE_SUFFIXES:
        if folder_name.endswith("_" + suffix):
            scene = folder_name[: -(len(suffix) + 1)]
            return scene, suffix
    return None


def load_metrics(exp_dir: Path) -> dict | None:
    results_file = exp_dir / "eval" / "results.json"
    if not results_file.exists():
        return None
    with open(results_file) as f:
        data = json.load(f)
    return data.get("avrg")


def collect_results(root: Path) -> dict:
    """
    Returns nested dict: results[scene][model][suffix] = metrics_dict
    """
    results: dict = {}

    for model_dir in sorted(root.iterdir()):
        if not model_dir.is_dir():
            continue
        model = model_dir.name

        for exp_dir in sorted(model_dir.iterdir()):
            if not exp_dir.is_dir():
                continue
            parsed = parse_scene_and_suffix(exp_dir.name)
            if parsed is None:
                print(f"  Warning: unrecognised folder name '{exp_dir.name}', skipping", file=sys.stderr)
                continue
            scene, suffix = parsed

            metrics = load_metrics(exp_dir)
            if metrics is None:
                print(f"  Warning: no eval/results.json in {exp_dir}, skipping", file=sys.stderr)
                continue

            results.setdefault(scene, {}).setdefault(model, {})[suffix] = metrics

    return results


def build_rows(results: dict) -> list[dict]:
    rows = []

    for scene in sorted(results):
        first_scene_row = True
        for model in sorted(results[scene]):
            first_model_row = True
            model_data = results[scene][model]

            baseline_psnr = None
            if BASELINE_SUFFIX in model_data:
                baseline_psnr = model_data[BASELINE_SUFFIX].get("psnr")

            for suffix in ROW_ORDER:
                if suffix not in model_data:
                    continue
                m = model_data[suffix]
                t_val, r_val = POSE_SUFFIXES[suffix]

                delta_psnr = ""
                if baseline_psnr is not None and m.get("psnr") is not None:
                    delta = m["psnr"] - baseline_psnr
                    delta_psnr = f"{delta:+.3f}"

                rows.append({
                    "scene":      scene if first_scene_row else "",
                    "model":      model if first_model_row else "",
                    "T":          t_val,
                    "R":          r_val,
                    "ssim":       f"{m['ssim']:.4f}"  if m.get("ssim")  is not None else "",
                    "psnr":       f"{m['psnr']:.3f}"  if m.get("psnr")  is not None else "",
                    "lpips":      f"{m['lpips']:.4f}" if m.get("lpips") is not None else "",
                    "fid":        f"{m['fid']:.3f}"   if m.get("fid")   is not None else "",
                    "delta_psnr": delta_psnr,
                })

                first_scene_row = False
                first_model_row = False

    return rows


def write_csv(rows: list[dict], out_path: Path | None) -> None:
    fieldnames = ["scene", "model", "T", "R", "ssim", "psnr", "lpips", "fid", "delta_psnr"]
    headers    = ["Scene name", "Model", "T", "R", "SSIM↑", "PSNR↑", "LPIPS↓", "FID↓", "Δ PSNR vs O/O"]

    f = open(out_path, "w", newline="") if out_path else sys.stdout
    try:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(dict(zip(fieldnames, headers)))
        writer.writerows(rows)
    finally:
        if out_path:
            f.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate benchmark results table as CSV.")
    parser.add_argument("--root", required=True, help="Root directory of the benchmark experiment")
    parser.add_argument("--out",  default=None,  help="Output CSV path (default: print to stdout)")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"Error: root path does not exist: {root}", file=sys.stderr)
        sys.exit(1)

    results = collect_results(root)
    if not results:
        print("No results found.", file=sys.stderr)
        sys.exit(1)

    rows = build_rows(results)
    out_path = Path(args.out) if args.out else None
    write_csv(rows, out_path)

    if out_path:
        print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
