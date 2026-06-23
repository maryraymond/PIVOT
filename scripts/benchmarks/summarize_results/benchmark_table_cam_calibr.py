"""
Parse BM-CalibR experiment root and produce a CSV table.

BM-CalibR: Camera Intrinsics — Pre-calibrated vs NeRF-optimised

Training:   orbit_inward_low/mid/high (90% train / 10% eval each)
            Optimised poses (OO condition), camera-optimizer OFF
Two conditions:
    opt_camera   — intrinsics optimised by NeRF during training
    calib_camera — pre-calibrated intrinsics fixed during training

Table layout:
    Scene | Model | Intrinsics | SSIM↑ | PSNR↑ | LPIPS↓ | FID↓ | Δ PSNR vs opt

Δ PSNR = PSNR(row) − PSNR(opt_camera, same scene + model)

Folder layout:
    {root}/{base_model}/{scene}_opt_camera/eval/results.json
    {root}/{base_model}/{scene}_calib_camera/eval/results.json

Usage:
    python benchmark_table_bm_cam_calibr.py --root /path/to/exp_ds_bm_cam_calibr --out results.csv
    python benchmark_table_bm_cam_calibr.py --root /path/to/exp_ds_bm_cam_calibr  # stdout
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


CONDITION_SUFFIXES: dict[str, str] = {
    "_opt_camera":   "opt_camera",
    "_calib_camera": "calib_camera",
}

CONDITION_ORDER = ["opt_camera", "calib_camera"]
CONDITION_LABELS = {
    "opt_camera":   "NeRF-optimised",
    "calib_camera": "Pre-calibrated",
}

BASELINE_CONDITION = "opt_camera"

BASE_MODEL_ORDER = ["nerfacto", "splatfacto"]
BASE_MODEL_LABELS = {
    "nerfacto":   "Nerfacto w pose opt.",
    "splatfacto": "Splatfacto w pose opt.",
}


def parse_scene_condition(folder_name: str) -> tuple[str, str] | None:
    for suffix, condition in CONDITION_SUFFIXES.items():
        if folder_name.endswith(suffix):
            return folder_name[: -len(suffix)], condition
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
    results[scene][base_model][condition] = metrics_dict (from results.json "avrg")
    """
    results: dict = {}

    for model_dir in sorted(root.iterdir()):
        if not model_dir.is_dir():
            continue
        base_model = model_dir.name

        for exp_dir in sorted(model_dir.iterdir()):
            if not exp_dir.is_dir():
                continue
            parsed = parse_scene_condition(exp_dir.name)
            if parsed is None:
                print(f"  Warning: unrecognised folder '{exp_dir.name}', skipping",
                      file=sys.stderr)
                continue
            scene, condition = parsed

            metrics = load_metrics(exp_dir)
            if metrics is None:
                print(f"  Warning: no eval/results.json in {exp_dir}", file=sys.stderr)
                continue

            (results
                .setdefault(scene, {})
                .setdefault(base_model, {})
            )[condition] = metrics

    return results


def build_rows(results: dict) -> list[dict]:
    rows = []

    for scene in sorted(results):
        first_scene_row = True

        for base_model in [m for m in BASE_MODEL_ORDER if m in results[scene]]:
            model_data      = results[scene][base_model]
            first_model_row = True

            baseline_psnr = model_data.get(BASELINE_CONDITION, {}).get("psnr")

            for condition in CONDITION_ORDER:
                if condition not in model_data:
                    continue
                m = model_data[condition]

                delta_psnr = ""
                if baseline_psnr is not None and m.get("psnr") is not None:
                    delta_psnr = f"{m['psnr'] - baseline_psnr:+.3f}"

                rows.append({
                    "scene":      scene if first_scene_row else "",
                    "model":      BASE_MODEL_LABELS.get(base_model, base_model)
                                  if first_model_row else "",
                    "intrinsics": CONDITION_LABELS.get(condition, condition),
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
    fieldnames = ["scene", "model", "intrinsics",
                  "ssim", "psnr", "lpips", "fid", "delta_psnr"]
    headers    = ["Scene", "Model", "Intrinsics",
                  "SSIM↑", "PSNR↑", "LPIPS↓", "FID↓", "Δ PSNR vs opt"]

    f = open(out_path, "w", newline="") if out_path else sys.stdout
    try:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(dict(zip(fieldnames, headers)))
        writer.writerows(rows)
    finally:
        if out_path:
            f.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate BM-CalibR intrinsics comparison results table as CSV.")
    parser.add_argument("--root", required=True,
                        help="BM-CalibR experiment root directory")
    parser.add_argument("--out",  default=None,
                        help="Output CSV path (default: stdout)")
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
    write_csv(rows, Path(args.out) if args.out else None)

    if args.out:
        print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
