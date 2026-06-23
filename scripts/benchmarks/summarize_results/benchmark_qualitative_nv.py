"""
Generate BM-NV qualitative comparison plots.

Each output plot shows the novel-view difficulty-tier degradation gradient:
  rows    = one per difficulty tier (one representative trajectory per tier)
  columns = GT | Nerfacto Generated | Splatfacto Generated

One plot is saved per frame index. Frames are proportionally sampled so that
trajectories with more frames are sub-sampled to match the trajectory with the
fewest frames across all tiers.

Folder layout:
    {root}/{base_model}/{scene}_bm_nv/eval/{traj}/gt/{frame}.JPG
    {root}/{base_model}/{scene}_bm_nv/eval/{traj}/gen/{frame}.JPG

Usage:
    conda run -n nerfstudio_1.1 python3 benchmark_qualitative_bm_nv.py \\
        --root /path/to/exp_ds_bm_nv \\
        --out-dir /path/to/output_plots
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import numpy as np


EVAL_FOLDER_SUFFIX = "_bm_nv"

# One representative trajectory per tier (first available is used if multiple listed)
TIERS: list[dict] = [
    {
        "id": 0,
        "label": "Tier 0\nIn-distribution",
        "candidates": ["orbit_inward_low"],
    },
    {
        "id": 1,
        "label": "Tier 1\nUnseen direction",
        "candidates": ["traversal_backward_low"],
    },
    {
        "id": 2,
        "label": "Tier 2\nDifferent topology",
        "candidates": ["traverse_loop_low"],
    },
    {
        "id": 3,
        "label": "Tier 3\nDifferent camera dir.",
        "candidates": ["orbit_outward_low"],
    },
    {
        "id": 4,
        "label": "Tier 4\nDifferent motion type",
        "candidates": ["rocket_upward", "bev_orbit_area"],
    },
    {
        "id": 5,
        "label": "Tier 5\nDifferent capture mode",
        "candidates": ["panorama_360_station_a", "scattered_low"],
    },
]

BASE_MODEL_ORDER  = ["nerfacto", "splatfacto"]
BASE_MODEL_LABELS = {
    "nerfacto":   "Nerfacto\nw pose opt.",
    "splatfacto": "Splatfacto\nw pose opt.",
}

LABEL_BG  = "#D4A0C0"
HEADER_BG = "#C8D8F0"
FIG_BG    = "#F5F5F5"


def parse_scene(folder_name: str) -> str | None:
    if folder_name.endswith(EVAL_FOLDER_SUFFIX):
        return folder_name[: -len(EVAL_FOLDER_SUFFIX)]
    return None


def collect_structure(root: Path) -> dict:
    """
    data[scene][base_model][traj] = sorted list of {"gt": Path, "gen": Path, "name": str}
    """
    data: dict = {}

    for model_dir in sorted(root.iterdir()):
        if not model_dir.is_dir():
            continue
        base_model = model_dir.name

        for exp_dir in sorted(model_dir.iterdir()):
            if not exp_dir.is_dir():
                continue
            scene = parse_scene(exp_dir.name)
            if scene is None:
                continue

            eval_dir = exp_dir / "eval"
            if not eval_dir.exists():
                continue

            for traj_dir in sorted(eval_dir.iterdir()):
                if not traj_dir.is_dir():
                    continue
                traj    = traj_dir.name
                gt_dir  = traj_dir / "gt"
                gen_dir = traj_dir / "gen"
                if not gt_dir.exists() or not gen_dir.exists():
                    continue

                frames = []
                for gt_file in sorted(gt_dir.iterdir()):
                    gen_file = gen_dir / gt_file.name
                    if gen_file.exists():
                        frames.append({"gt": gt_file, "gen": gen_file,
                                       "name": gt_file.name})
                if frames:
                    (data
                        .setdefault(scene, {})
                        .setdefault(base_model, {})
                    )[traj] = frames

    return data


def resolve_tier_traj(tier: dict, scene_data: dict, models: list[str]) -> str | None:
    """Return the first candidate trajectory that has data for at least one model."""
    for traj in tier["candidates"]:
        if any(traj in scene_data.get(m, {}) for m in models):
            return traj
    return None


def sample_frame(frames: list[dict], plot_idx: int, n_plots: int) -> dict:
    n = len(frames)
    idx = 0 if n_plots == 1 else round(plot_idx * (n - 1) / (n_plots - 1))
    return frames[idx]


def load_image(path: Path, max_width: int = 640) -> np.ndarray:
    img = mpimg.imread(str(path))
    h, w = img.shape[:2]
    if w > max_width:
        img = img[::max(1, h // int(h * max_width / w)),
                  ::max(1, w // max_width)]
    return img


def make_plot(scene: str, plot_idx: int, n_plots: int,
              data: dict, models: list[str],
              tier_trajs: list[tuple[dict, str]],
              max_width: int) -> plt.Figure | None:
    """
    tier_trajs: list of (tier_dict, resolved_trajectory_name) for each active tier.
    rows = one per active tier, cols = GT | model_0 gen | model_1 gen ...
    """
    n_rows     = len(tier_trajs)
    n_img_cols = 1 + len(models)   # GT + one per model

    label_scene_w = 1.5
    label_tier_w  = 2.8
    img_col_w     = 6.0

    col_widths  = [label_scene_w, label_tier_w] + [img_col_w] * n_img_cols
    row_heights = [0.8] + [3.0] * n_rows

    fig = plt.figure(figsize=(sum(col_widths) * 1.8, sum(row_heights) * 1.8),
                     facecolor=FIG_BG)

    gs = gridspec.GridSpec(
        nrows=1 + n_rows,
        ncols=len(col_widths),
        figure=fig,
        width_ratios=col_widths,
        height_ratios=row_heights,
        wspace=0.02,
        hspace=0.02,
    )

    # ── header row ──────────────────────────────────────────────────────────
    for col in [0, 1]:
        ax = fig.add_subplot(gs[0, col])
        ax.set_facecolor(LABEL_BG)
        ax.axis("off")

    ax_gt = fig.add_subplot(gs[0, 2])
    ax_gt.set_facecolor(HEADER_BG)
    ax_gt.text(0.5, 0.5, "GT", ha="center", va="center",
               fontsize=20, fontweight="bold", transform=ax_gt.transAxes)
    ax_gt.axis("off")

    for j, base_model in enumerate(models):
        ax = fig.add_subplot(gs[0, 3 + j])
        ax.set_facecolor(HEADER_BG)
        ax.text(0.5, 0.5, BASE_MODEL_LABELS.get(base_model, base_model),
                ha="center", va="center",
                fontsize=16, fontweight="bold", transform=ax.transAxes)
        ax.axis("off")

    # ── image rows (one per tier) ────────────────────────────────────────────
    any_data = False

    for row_i, (tier, traj) in enumerate(tier_trajs):
        img_row = row_i + 1

        ax_scene = fig.add_subplot(gs[img_row, 0])
        ax_scene.set_facecolor(LABEL_BG)
        ax_scene.axis("off")

        tier_label = f"{tier['label']}\n{traj}"
        ax_tier = fig.add_subplot(gs[img_row, 1])
        ax_tier.set_facecolor(LABEL_BG)
        ax_tier.text(0.5, 0.5, tier_label, ha="center", va="center",
                     fontsize=11, transform=ax_tier.transAxes)
        ax_tier.axis("off")

        # GT — use first available model as source (GT is identical across models)
        gt_frame = None
        for base_model in models:
            frames = data.get(scene, {}).get(base_model, {}).get(traj, [])
            if frames:
                gt_frame = sample_frame(frames, plot_idx, n_plots)
                break

        ax_gt = fig.add_subplot(gs[img_row, 2])
        if gt_frame is not None:
            try:
                ax_gt.imshow(load_image(gt_frame["gt"], max_width))
                any_data = True
            except Exception:
                ax_gt.set_facecolor("#CCCCCC")
                ax_gt.text(0.5, 0.5, "Error", ha="center", va="center",
                           color="gray", transform=ax_gt.transAxes)
        else:
            ax_gt.set_facecolor("#CCCCCC")
            ax_gt.text(0.5, 0.5, "N/A", ha="center", va="center",
                       color="gray", transform=ax_gt.transAxes)
        ax_gt.axis("off")

        # Generated image per model
        for j, base_model in enumerate(models):
            ax = fig.add_subplot(gs[img_row, 3 + j])
            frames = data.get(scene, {}).get(base_model, {}).get(traj, [])
            if frames:
                frame = sample_frame(frames, plot_idx, n_plots)
                try:
                    ax.imshow(load_image(frame["gen"], max_width))
                    any_data = True
                except Exception:
                    ax.set_facecolor("#CCCCCC")
                    ax.text(0.5, 0.5, "Error", ha="center", va="center",
                            color="gray", transform=ax.transAxes)
            else:
                ax.set_facecolor("#CCCCCC")
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        color="gray", transform=ax.transAxes)
            ax.axis("off")

    if not any_data:
        plt.close(fig)
        return None

    scene_label = scene.replace("_", " ").title()
    fig.text(
        0.5 * label_scene_w / sum(col_widths),
        0.5,
        scene_label,
        ha="center", va="center",
        fontsize=18, fontweight="bold",
        rotation=90,
    )

    fig.suptitle(f"{scene_label}  ·  plot {plot_idx + 1}/{n_plots}",
                 fontsize=14, y=0.98, color="#444444")

    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate BM-NV qualitative degradation plots.")
    parser.add_argument("--root",      required=True,
                        help="BM-NV experiment root directory")
    parser.add_argument("--out-dir",   required=True,
                        help="Directory to write output PNG files")
    parser.add_argument("--dpi",       type=int, default=120)
    parser.add_argument("--max-width", type=int, default=640,
                        help="Max pixel width per image tile (default: 640)")
    args = parser.parse_args()

    root    = Path(args.root)
    out_dir = Path(args.out_dir)

    if not root.exists():
        print(f"Error: root path does not exist: {root}", file=sys.stderr)
        sys.exit(1)

    print("Scanning experiment folder...")
    data = collect_structure(root)

    if not data:
        print("No data found.", file=sys.stderr)
        sys.exit(1)

    total = 0
    for scene, scene_data in sorted(data.items()):

        models = [m for m in BASE_MODEL_ORDER if m in scene_data]

        # Resolve one trajectory per tier (skip tiers with no data)
        tier_trajs: list[tuple[dict, str]] = []
        for tier in TIERS:
            traj = resolve_tier_traj(tier, scene_data, models)
            if traj is not None:
                tier_trajs.append((tier, traj))

        if not tier_trajs:
            print(f"  {scene}: no tier data found, skipping")
            continue

        # n_plots = min frame count across all resolved representative trajectories
        counts = [
            len(scene_data[m][traj])
            for _, traj in tier_trajs
            for m in models
            if traj in scene_data.get(m, {})
        ]
        if not counts:
            continue
        n_plots = min(counts)

        tier_names = [t for _, t in tier_trajs]
        print(f"  {scene}: {n_plots} plots | tiers: {tier_names}")

        save_dir = out_dir / scene
        save_dir.mkdir(parents=True, exist_ok=True)

        for plot_idx in range(n_plots):
            fig = make_plot(scene, plot_idx, n_plots, data, models,
                            tier_trajs, args.max_width)
            if fig is None:
                continue

            out_path = save_dir / f"plot_{plot_idx:03d}.png"
            fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight",
                        facecolor=FIG_BG)
            plt.close(fig)
            print(f"    plot_{plot_idx:03d}.png")
            total += 1

    print(f"\nDone. {total} plots written to {out_dir}")


if __name__ == "__main__":
    main()
