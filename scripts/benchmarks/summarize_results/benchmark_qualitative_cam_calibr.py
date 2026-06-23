"""
Generate BM-CalibR qualitative comparison plots.

Compares two camera intrinsic conditions:
  opt_camera   — NeRF-optimised intrinsics (camera_intrinsics_optimized: true)
  calib_camera — Pre-calibrated intrinsics  (camera_intrinsics_optimized: false)

Layout per plot:
  rows    = models (Nerfacto, Splatfacto)
  columns = GT | NeRF-opt. intrinsics | Pre-calibrated intrinsics

One plot is saved per (scene, eval trajectory, frame index).
Frames are proportionally sampled when trajectories differ in length.

Folder layout:
    {root}/{base_model}/{scene}_opt_camera/eval/{traj}/gt/{frame}.JPG
    {root}/{base_model}/{scene}_opt_camera/eval/{traj}/gen/{frame}.JPG
    {root}/{base_model}/{scene}_calib_camera/eval/{traj}/gt/{frame}.JPG
    {root}/{base_model}/{scene}_calib_camera/eval/{traj}/gen/{frame}.JPG

Usage:
    conda run -n nerfstudio_1.1 python3 benchmark_qualitative_bm_cam_calibr.py \\
        --root /path/to/exp_ds_bm_cam_calibr \\
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


# Folder-name suffixes that identify each condition
CONDITION_SUFFIXES: dict[str, str] = {
    "_opt_camera":   "opt_camera",
    "_calib_camera": "calib_camera",
}

CONDITIONS = ["opt_camera", "calib_camera"]
CONDITION_LABELS = {
    "opt_camera":   "NeRF-opt.\nintrinsics",
    "calib_camera": "Pre-calibrated\nintrinsics",
}

BASE_MODEL_ORDER = ["nerfacto", "splatfacto"]
BASE_MODEL_LABELS = {
    "nerfacto":   "Nerfacto\nw pose opt.",
    "splatfacto": "Splatfacto\nw pose opt.",
}

LABEL_BG  = "#D4A0C0"
HEADER_BG = "#C8D8F0"
FIG_BG    = "#F5F5F5"


def parse_scene_condition(folder_name: str) -> tuple[str, str] | None:
    """Strip a known condition suffix; return (scene, condition) or None."""
    for suffix, condition in CONDITION_SUFFIXES.items():
        if folder_name.endswith(suffix):
            return folder_name[: -len(suffix)], condition
    return None


def collect_structure(root: Path) -> dict:
    """
    data[scene][base_model][condition][traj] = sorted list of
        {"gt": Path, "gen": Path, "name": str}
    """
    data: dict = {}

    for model_dir in sorted(root.iterdir()):
        if not model_dir.is_dir():
            continue
        base_model = model_dir.name

        for exp_dir in sorted(model_dir.iterdir()):
            if not exp_dir.is_dir():
                continue
            parsed = parse_scene_condition(exp_dir.name)
            if parsed is None:
                continue
            scene, condition = parsed

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
                        .setdefault(condition, {})
                    )[traj] = frames

    return data


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


def make_plot(scene: str, traj: str, plot_idx: int, n_plots: int,
              data: dict, models: list[str],
              conditions: list[str], max_width: int) -> plt.Figure | None:
    """
    rows = models (Nerfacto, Splatfacto)
    cols = GT | opt_camera gen | calib_camera gen
    """
    n_img_cols = 1 + len(conditions)   # GT + one per condition

    label_scene_w = 1.5
    label_model_w = 2.5
    img_col_w     = 6.0

    col_widths  = [label_scene_w, label_model_w] + [img_col_w] * n_img_cols
    row_heights = [0.8] + [3.0] * len(models)

    fig = plt.figure(figsize=(sum(col_widths) * 1.8, sum(row_heights) * 1.8),
                     facecolor=FIG_BG)

    gs = gridspec.GridSpec(
        nrows=1 + len(models),
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

    ax_gt_hdr = fig.add_subplot(gs[0, 2])
    ax_gt_hdr.set_facecolor(HEADER_BG)
    ax_gt_hdr.text(0.5, 0.5, "GT", ha="center", va="center",
                   fontsize=20, fontweight="bold", transform=ax_gt_hdr.transAxes)
    ax_gt_hdr.axis("off")

    for j, condition in enumerate(conditions):
        ax = fig.add_subplot(gs[0, 3 + j])
        ax.set_facecolor(HEADER_BG)
        ax.text(0.5, 0.5, CONDITION_LABELS.get(condition, condition),
                ha="center", va="center",
                fontsize=16, fontweight="bold", transform=ax.transAxes)
        ax.axis("off")

    # ── image rows (one per model) ───────────────────────────────────────────
    any_data = False

    for row_i, base_model in enumerate(models):
        img_row = row_i + 1

        ax_scene = fig.add_subplot(gs[img_row, 0])
        ax_scene.set_facecolor(LABEL_BG)
        ax_scene.axis("off")

        ax_model = fig.add_subplot(gs[img_row, 1])
        ax_model.set_facecolor(LABEL_BG)
        ax_model.text(0.5, 0.5, BASE_MODEL_LABELS.get(base_model, base_model),
                      ha="center", va="center",
                      fontsize=12, transform=ax_model.transAxes)
        ax_model.axis("off")

        # GT — use first available condition as GT source (GT is identical across conditions)
        gt_frame = None
        for cond in conditions:
            frames = (data.get(scene, {})
                         .get(base_model, {})
                         .get(cond, {})
                         .get(traj, []))
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

        # Generated image per condition
        for j, condition in enumerate(conditions):
            ax = fig.add_subplot(gs[img_row, 3 + j])
            frames = (data.get(scene, {})
                         .get(base_model, {})
                         .get(condition, {})
                         .get(traj, []))
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

    traj_label = traj.replace("_", " ")
    fig.suptitle(
        f"{scene_label}  ·  {traj_label}  ·  plot {plot_idx + 1}/{n_plots}",
        fontsize=14, y=0.98, color="#444444",
    )

    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate BM-CalibR qualitative intrinsics comparison plots.")
    parser.add_argument("--root",      required=True,
                        help="BM-CalibR experiment root directory")
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
        models     = [m for m in BASE_MODEL_ORDER if m in scene_data]
        conditions = [c for c in CONDITIONS
                      if any(c in scene_data.get(m, {}) for m in models)]

        all_trajs: set[str] = set()
        for m in models:
            for c in conditions:
                all_trajs.update(scene_data.get(m, {}).get(c, {}).keys())

        for traj in sorted(all_trajs):
            counts = [
                len(scene_data[m][c][traj])
                for m in models
                for c in conditions
                if traj in scene_data.get(m, {}).get(c, {})
            ]
            if not counts:
                continue
            n_plots = min(counts)

            print(f"  {scene}/{traj}: {n_plots} plots")

            save_dir = out_dir / scene / traj
            save_dir.mkdir(parents=True, exist_ok=True)

            for plot_idx in range(n_plots):
                fig = make_plot(scene, traj, plot_idx, n_plots,
                                data, models, conditions, args.max_width)
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
