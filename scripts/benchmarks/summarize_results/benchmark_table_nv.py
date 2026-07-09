# PIVOT: Pose, Intrinsics and Viewpoint Oriented Testbed
# Copyright (c) 2026 Mary Raymond
# Date: 2026-07-07
# MIT License — see LICENSE in the project root for details.

"""
Parse BM-NV experiment root and produce a CSV table.

BM-NV: Novel View Generalisation Benchmark

Training:   orbit_inward_low/mid/high + traversal_forward/left/right_low
            (90% train / 10% eval each, optimised poses, optimised intrinsics,
             camera-optimizer off)

Eval-only trajectories are organised into difficulty tiers based on how far
they diverge from the training distribution:

  Tier 0 — In-distribution     : held-out 10% of the 6 training trajectories
  Tier 1 — Unseen direction     : traversal_backward_low
  Tier 2 — Different topology   : traverse_loop_low
  Tier 3 — Different camera dir : orbit_outward_low
  Tier 4 — Different motion type: rocket_upward, bev_orbit_area
  Tier 5 — Different capture mode: panorama_360_station_a/b/c, scattered_low

Δ PSNR column = PSNR(row) − mean(PSNR of Tier-0 trajectories, same scene+model)
Shows quality degradation as viewpoints diverge from the training distribution.

Folder layout:
    {root}/{base_model}/{scene}_bm_nv/eval/results.json

Usage:
    python benchmark_table_bm_nv.py --root /path/to/exp_ds_bm_nv --out results.csv
    python benchmark_table_bm_nv.py --root /path/to/exp_ds_bm_nv           # stdout
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean


EVAL_FOLDER_SUFFIX = "_bm_nv"

BASE_MODEL_ORDER = ["nerfacto", "splatfacto"]
BASE_MODEL_LABELS = {
    "nerfacto":   "Nerfacto w pose opt.",
    "splatfacto": "Splatfacto w pose opt.",
}

# Difficulty tiers — ordered from easiest to hardest
TIERS: dict[int, dict] = {
    0: {
        "label": "In-distribution (10% held-out)",
        "trajectories": [
            "orbit_inward_low",
            "orbit_inward_mid",
            "orbit_inward_high",
            "traversal_forward_low",
            "traversal_left_low",
            "traversal_right_low",
        ],
    },
    1: {
        "label": "Unseen direction",
        "trajectories": ["traversal_backward_low"],
    },
    2: {
        "label": "Different path topology",
        "trajectories": ["traverse_loop_low"],
    },
    3: {
        "label": "Different camera direction",
        "trajectories": ["orbit_outward_low"],
    },
    4: {
        "label": "Different motion type",
        "trajectories": ["rocket_upward", "bev_orbit_area"],
    },
    5: {
        "label": "Different capture mode",
        "trajectories": [
            "panorama_360_station_a",
            "panorama_360_station_b",
            "panorama_360_station_c",
            "scattered_low",
        ],
    },
}

# Convenience lookups
TRAJ_ORDER  = [t for tier in TIERS.values() for t in tier["trajectories"]]
TIER_OF     = {t: tid for tid, tier in TIERS.items() for t in tier["trajectories"]}
IN_DIST     = set(TIERS[0]["trajectories"])


def parse_scene(folder_name: str) -> str | None:
    """Return scene name by stripping the _bm_nv suffix, or None if unrecognised."""
    if folder_name.endswith(EVAL_FOLDER_SUFFIX):
        return folder_name[: -len(EVAL_FOLDER_SUFFIX)]
    return None


def load_metrics(exp_dir: Path) -> dict | None:
    results_file = exp_dir / "eval" / "results.json"
    if not results_file.exists():
        return None
    with open(results_file) as f:
        return json.load(f)


def collect_results(root: Path) -> dict:
    """
    results[scene][base_model] = full results.json content
    """
    results: dict = {}

    for model_dir in sorted(root.iterdir()):
        if not model_dir.is_dir():
            continue
        base_model = model_dir.name

        for exp_dir in sorted(model_dir.iterdir()):
            if not exp_dir.is_dir():
                continue
            scene = parse_scene(exp_dir.name)
            if scene is None:
                print(f"  Warning: unrecognised folder '{exp_dir.name}', skipping",
                      file=sys.stderr)
                continue

            metrics = load_metrics(exp_dir)
            if metrics is None:
                print(f"  Warning: no eval/results.json in {exp_dir}", file=sys.stderr)
                continue

            results.setdefault(scene, {})[base_model] = metrics

    return results


def build_rows(results: dict) -> list[dict]:
    rows = []

    for scene in sorted(results):
        first_scene_row = True

        for base_model in [m for m in BASE_MODEL_ORDER if m in results[scene]]:
            m = results[scene][base_model]
            first_model_row = True

            # Δ PSNR baseline = mean PSNR of available Tier-0 trajectories
            in_dist_psnrs = [
                m[t]["psnr"] for t in TIERS[0]["trajectories"]
                if t in m and m[t].get("psnr") is not None
            ]
            baseline_psnr = mean(in_dist_psnrs) if in_dist_psnrs else None

            current_tier = -1

            for traj in TRAJ_ORDER:
                if traj not in m:
                    continue

                tier_id    = TIER_OF[traj]
                t_metrics  = m[traj]
                tier_label = TIERS[tier_id]["label"] if tier_id != current_tier else ""
                current_tier = tier_id

                psnr = t_metrics.get("psnr")
                delta_psnr = (
                    f"{psnr - baseline_psnr:+.3f}"
                    if psnr is not None and baseline_psnr is not None
                    else ""
                )

                rows.append({
                    "scene":      scene if first_scene_row else "",
                    "model":      BASE_MODEL_LABELS.get(base_model, base_model)
                                  if first_model_row else "",
                    "tier":       tier_label,
                    "trajectory": traj,
                    "ssim":       f"{t_metrics['ssim']:.4f}"  if t_metrics.get("ssim")  is not None else "",
                    "psnr":       f"{psnr:.3f}"               if psnr                   is not None else "",
                    "lpips":      f"{t_metrics['lpips']:.4f}" if t_metrics.get("lpips") is not None else "",
                    "fid":        f"{t_metrics['fid']:.3f}"   if t_metrics.get("fid")   is not None else "",
                    "delta_psnr": delta_psnr,
                })
                first_scene_row = False
                first_model_row = False

            # In-distribution average row
            in_dist_metrics = [m[t] for t in TIERS[0]["trajectories"] if t in m]
            if in_dist_metrics:
                rows.append(_avg_row("In-dist average", in_dist_metrics))

            # OOD average row (tiers 1-5)
            ood_trajs = [t for tid, tier in TIERS.items()
                         if tid > 0 for t in tier["trajectories"]]
            ood_metrics = [m[t] for t in ood_trajs if t in m]
            if ood_metrics:
                rows.append(_avg_row("OOD average", ood_metrics))

            # Overall average from results.json "avrg" key
            avrg = m.get("avrg", {})
            rows.append({
                "scene": "", "model": "", "tier": "",
                "trajectory": "Overall average",
                "ssim":       f"{avrg['ssim']:.4f}"  if avrg.get("ssim")  is not None else "",
                "psnr":       f"{avrg['psnr']:.3f}"  if avrg.get("psnr")  is not None else "",
                "lpips":      f"{avrg['lpips']:.4f}" if avrg.get("lpips") is not None else "",
                "fid":        f"{avrg['fid']:.3f}"   if avrg.get("fid")   is not None else "",
                "delta_psnr": "",
            })

    return rows


def _avg_row(label: str, metrics_list: list[dict]) -> dict:
    """Build a summary row from a list of per-trajectory metric dicts."""
    def _avg(key: str, fmt: str) -> str:
        vals = [m[key] for m in metrics_list if m.get(key) is not None]
        return format(mean(vals), fmt) if vals else ""

    return {
        "scene": "", "model": "", "tier": "",
        "trajectory": label,
        "ssim":       _avg("ssim",  ".4f"),
        "psnr":       _avg("psnr",  ".3f"),
        "lpips":      _avg("lpips", ".4f"),
        "fid":        _avg("fid",   ".3f"),
        "delta_psnr": "",
    }


def write_csv(rows: list[dict], out_path: Path | None) -> None:
    fieldnames = ["scene", "model", "tier", "trajectory",
                  "ssim", "psnr", "lpips", "fid", "delta_psnr"]
    headers    = ["Scene", "Model", "Tier", "Eval trajectory",
                  "SSIM↑", "PSNR↑", "LPIPS↓", "FID↓", "Δ PSNR vs in-dist"]

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
        description="Generate BM-NV novel view generalisation results table as CSV.")
    parser.add_argument("--root", required=True,
                        help="BM-NV experiment root directory")
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
