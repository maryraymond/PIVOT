"""
Generate qualitative comparison plots for benchmark results.

Each output image shows a grid:
  rows    = one per model (e.g. Nerfacto, Splatfacto)
  columns = GT | T(M) R(M) | T(O) R(M) | T(M) R(O) | T(O) R(O)

One plot is saved per (scene, trajectory, frame).

Usage:
    conda run -n nerfstudio_1.1 python3 benchmark_qualitative.py \\
        --root /path/to/exp_ds_bm_01 \\
        --out-dir /path/to/output_plots

Expected folder layout (same as benchmark_table.py):
    {root}/{model}/{scene}_{pose_suffix}/eval/{traj}/gt/{frame}.JPG
    {root}/{model}/{scene}_{pose_suffix}/eval/{traj}/gen/{frame}.JPG
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


POSE_SUFFIXES = {
    "optRT":       ("O", "O"),
    "optR_measT":  ("M", "O"),
    "optT_measR":  ("O", "M"),
    "measRT":      ("M", "M"),
}

# Column order matching the reference image
COL_ORDER = ["measRT", "optT_measR", "optR_measT", "optRT"]

COL_LABELS = {
    "measRT":      "T(M) R(M)",
    "optT_measR":  "T(O) R(M)",
    "optR_measT":  "T(M) R(O)",
    "optRT":       "T(O) R(O)",
}

LABEL_BG   = "#D4A0C0"   # pinkish-purple matching reference image
HEADER_BG  = "#C8D8F0"   # light blue for column headers
FIG_BG     = "#F5F5F5"

MODEL_ORDER = ["nerfacto", "splatfacto"]
MODEL_LABELS = {"nerfacto": "Nerfacto", "splatfacto": "Splatfacto"}


def parse_scene_and_suffix(folder_name: str) -> tuple[str, str] | None:
    for suffix in POSE_SUFFIXES:
        if folder_name.endswith("_" + suffix):
            scene = folder_name[: -(len(suffix) + 1)]
            return scene, suffix
    return None


def collect_structure(root: Path) -> dict:
    """
    Returns:
        data[scene][model][suffix][traj] = {frame_name: {"gt": Path, "gen": Path}}
    """
    data: dict = {}

    for model_dir in sorted(root.iterdir()):
        if not model_dir.is_dir():
            continue
        model = model_dir.name

        for exp_dir in sorted(model_dir.iterdir()):
            if not exp_dir.is_dir():
                continue
            parsed = parse_scene_and_suffix(exp_dir.name)
            if parsed is None:
                continue
            scene, suffix = parsed

            eval_dir = exp_dir / "eval"
            if not eval_dir.exists():
                continue

            for traj_dir in sorted(eval_dir.iterdir()):
                if not traj_dir.is_dir():
                    continue
                traj = traj_dir.name
                gt_dir  = traj_dir / "gt"
                gen_dir = traj_dir / "gen"
                if not gt_dir.exists() or not gen_dir.exists():
                    continue

                for gt_file in sorted(gt_dir.iterdir()):
                    frame_name = gt_file.name
                    gen_file = gen_dir / frame_name
                    if not gen_file.exists():
                        continue

                    (data
                        .setdefault(scene, {})
                        .setdefault(model, {})
                        .setdefault(suffix, {})
                        .setdefault(traj, {})
                    )[frame_name] = {"gt": gt_file, "gen": gen_file}

    return data


def load_image(path: Path, max_width: int = 960) -> np.ndarray:
    img = mpimg.imread(str(path))
    h, w = img.shape[:2]
    if w > max_width:
        scale = max_width / w
        new_w = max_width
        new_h = int(h * scale)
        # simple box-filter downsample
        img = img[::max(1, h // new_h), ::max(1, w // new_w)]
    return img


def make_plot(scene: str, traj: str, frame_name: str,
              data: dict, models: list[str]) -> plt.Figure | None:
    """
    Build one comparison figure for a single (scene, traj, frame).
    Rows = models, cols = GT + 4 pose variants.
    """
    n_models  = len(models)
    n_img_cols = 1 + len(COL_ORDER)   # GT + 4 pose variants

    # label column widths (relative)
    label_scene_w = 1.5
    label_model_w = 2.0
    img_col_w     = 5.0

    col_widths = [label_scene_w, label_model_w] + [img_col_w] * n_img_cols
    row_heights = [0.8] + [3.0] * n_models   # header row + image rows

    fig = plt.figure(figsize=(sum(col_widths) * 1.8, sum(row_heights) * 1.8),
                     facecolor=FIG_BG)

    gs = gridspec.GridSpec(
        nrows=1 + n_models,
        ncols=2 + n_img_cols,
        figure=fig,
        width_ratios=col_widths,
        height_ratios=row_heights,
        wspace=0.02,
        hspace=0.02,
    )

    # ── header row ──────────────────────────────────────────────────────────
    for spine_ax in [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]:
        spine_ax.set_facecolor(LABEL_BG)
        spine_ax.axis("off")

    gt_header = fig.add_subplot(gs[0, 2])
    gt_header.set_facecolor(HEADER_BG)
    gt_header.text(0.5, 0.5, "GT", ha="center", va="center",
                   fontsize=20, fontweight="bold", transform=gt_header.transAxes)
    gt_header.axis("off")

    for j, suffix in enumerate(COL_ORDER):
        ax = fig.add_subplot(gs[0, 3 + j])
        ax.set_facecolor(HEADER_BG)
        ax.text(0.5, 0.5, COL_LABELS[suffix], ha="center", va="center",
                fontsize=20, fontweight="bold", transform=ax.transAxes)
        ax.axis("off")

    # ── image rows ──────────────────────────────────────────────────────────
    any_data = False

    for row_i, model in enumerate(models):
        img_row = row_i + 1

        # scene label (only on first model row, spans all model rows via
        # a separate axis covering the full column height — approximated by
        # placing text with model-count scaling)
        scene_ax = fig.add_subplot(gs[img_row, 0])
        scene_ax.set_facecolor(LABEL_BG)
        scene_ax.axis("off")
        if row_i == 0:
            # extend the scene label visually by drawing text centered for
            # the full group; matplotlib gridspec doesn't natively merge cells
            # so we use figure-level text placed at the midpoint
            pass  # handled after loop

        model_ax = fig.add_subplot(gs[img_row, 1])
        model_ax.set_facecolor(LABEL_BG)
        model_ax.text(0.5, 0.5, MODEL_LABELS.get(model, model),
                      ha="center", va="center", fontsize=18, fontweight="bold",
                      rotation=90 if len(MODEL_LABELS.get(model, model)) > 6 else 0,
                      transform=model_ax.transAxes)
        model_ax.axis("off")

        # GT image — use optRT gt as canonical (all variants share same GT)
        gt_path = None
        for suffix in COL_ORDER:
            try:
                gt_path = data[scene][model][suffix][traj][frame_name]["gt"]
                break
            except KeyError:
                continue

        gt_ax = fig.add_subplot(gs[img_row, 2])
        if gt_path and gt_path.exists():
            gt_ax.imshow(load_image(gt_path))
            any_data = True
        gt_ax.axis("off")

        # Generated images for each pose variant
        for j, suffix in enumerate(COL_ORDER):
            ax = fig.add_subplot(gs[img_row, 3 + j])
            try:
                gen_path = data[scene][model][suffix][traj][frame_name]["gen"]
                ax.imshow(load_image(gen_path))
                any_data = True
            except KeyError:
                ax.set_facecolor("#CCCCCC")
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        transform=ax.transAxes, color="gray")
            ax.axis("off")

    if not any_data:
        plt.close(fig)
        return None

    # scene label centered across all model rows using figure-level text
    scene_label = scene.replace("_", " ").title()
    fig.text(
        0.5 * label_scene_w / sum(col_widths),
        0.5,
        scene_label,
        ha="center", va="center",
        fontsize=18, fontweight="bold",
        rotation=90,
    )

    fig.suptitle(f"{scene_label}  ·  {traj}  ·  {frame_name}",
                 fontsize=14, y=0.98, color="#444444")

    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate qualitative benchmark comparison plots.")
    parser.add_argument("--root",    required=True, help="Benchmark experiment root directory")
    parser.add_argument("--out-dir", required=True, help="Directory to write output PNG files")
    parser.add_argument("--dpi",     type=int, default=120, help="Output DPI (default: 120)")
    parser.add_argument("--max-width", type=int, default=640,
                        help="Max pixel width per image tile before downsampling (default: 640)")
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

    # determine which models are present (in preferred order)
    all_models = set()
    for scene_data in data.values():
        all_models.update(scene_data.keys())
    models = [m for m in MODEL_ORDER if m in all_models] + \
             sorted(all_models - set(MODEL_ORDER))

    total = 0
    for scene, scene_data in sorted(data.items()):
        # collect all (traj, frame_name) pairs present in any model/suffix
        all_frames: set[tuple[str, str]] = set()
        for model_data in scene_data.values():
            for suffix_data in model_data.values():
                for traj, traj_frames in suffix_data.items():
                    for fn in traj_frames:
                        all_frames.add((traj, fn))

        scene_models = [m for m in models if m in scene_data]

        for traj, frame_name in sorted(all_frames):
            fig = make_plot(scene, traj, frame_name, data, scene_models)
            if fig is None:
                continue

            stem = Path(frame_name).stem
            save_dir = out_dir / scene / traj
            save_dir.mkdir(parents=True, exist_ok=True)
            out_path = save_dir / f"{stem}.png"

            fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight",
                        facecolor=FIG_BG)
            plt.close(fig)
            print(f"  Saved: {out_path.relative_to(out_dir)}")
            total += 1

    print(f"\nDone. {total} plots written to {out_dir}")


if __name__ == "__main__":
    main()
