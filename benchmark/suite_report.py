"""Build the splax vs gsplat benchmark suite report as a multi-page PDF.

Takes the results of ``bench_forward.py`` and ``bench_backward.py`` and generates a multi-page PDF
with a cover page, one page per scene, and a summary page. Generate the report with:

    pixi run -e tests python benchmark/suite_report.py
"""

from __future__ import annotations

import argparse
import json
import logging
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

if TYPE_CHECKING:
    from matplotlib.axes import Axes

logger = logging.getLogger(__name__)

SPLAX_COLOR = "#1b9e77"
GSPLAT_COLOR = "#d95f02"
OUT_DIR = Path(__file__).resolve().parents[1] / "reports"


def cover_page(pdf: PdfPages, data: dict) -> None:
    """Cover page with the run metadata and the scenes benchmarked."""
    meta = data["meta"]
    fig = plt.figure(figsize=(11, 8.5))
    text = "splax vs gsplat benchmark suite"
    fig.text(0.5, 0.88, text, ha="center", fontsize=26, fontweight="bold")
    text = "Forward and backward pass throughput, batch scaling, and peak GPU memory"
    fig.text(0.5, 0.83, text, ha="center", fontsize=13, color="#555")
    lines = [
        f"GPU: {meta['gpu']}",
        "   ".join(f"{fw} {meta[f'{fw}_version']}" for fw in ("jax", "gsplat", "torch")),
        f"warmup {meta['warmup']}, iters {meta['iters']}, best of {meta['repeat']}",
        f"batches: {', '.join(str(b) for b in meta['batches'])}",
        f"metric: {meta['metric']}",
        f"memory: {meta['memory_note']}",
        f"generated: {meta['generated']}",
    ]
    y = 0.70
    for line in lines:
        fig.text(0.12, y, line, ha="left", fontsize=12)
        y -= 0.035
    y -= 0.02
    fig.text(0.12, y, "Scenes", ha="left", fontsize=14, fontweight="bold")
    y -= 0.04
    for sc in data["scenes"]:
        text = f"- {sc['name']}: {sc['description']}"
        for line in textwrap.wrap(text, width=100, subsequent_indent="  "):
            fig.text(0.12, y, line, ha="left", fontsize=11)
            y -= 0.032
    pdf.savefig(fig)
    plt.close(fig)


def scene_page(pdf: PdfPages, base: Path, sc: dict) -> None:
    """One page per scene: sample render plus time, throughput, and memory curves."""
    rows = sc["rows"]
    batches = np.array([r["batch"] for r in rows])
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    text = f"{sc['name']}  -  {sc['n_gaussians']:,} gaussians, "
    text += f"{sc['img_shape'][0]}x{sc['img_shape'][1]}, {sc['cameras']} cameras"
    fig.suptitle(text, fontsize=14, fontweight="bold")

    ax_img = axes[0, 0]
    ax_img.axis("off")
    ax_img.imshow(iio.imread(base / sc["sample_render"]))
    ax_img.set_title("splax sample render (view 0)", fontsize=10)

    ax_t = axes[0, 1]
    t = np.array([r["splax"]["time_ms"] for r in rows], float)
    ax_t.plot(batches, t, "o-", color=SPLAX_COLOR, label="splax")
    t = np.array([r["gsplat"]["time_ms"] for r in rows], float)
    ax_t.plot(batches, t, "s-", color=GSPLAT_COLOR, label="gsplat")
    ax_t.set_xscale("log", base=2)
    ax_t.set_yscale("log")
    ax_t.set_xlabel("batch size (cameras)")
    ax_t.set_ylabel("render time per call (ms)")
    ax_t.set_title("Render time vs batch")

    ax_thru = axes[1, 0]
    fps = batches / np.array([r["splax"]["time_ms"] for r in rows], float) * 1e3
    ax_thru.plot(batches, fps, "o-", color=SPLAX_COLOR, label="splax")
    fps = batches / np.array([r["gsplat"]["time_ms"] for r in rows], float) * 1e3
    ax_thru.plot(batches, fps, "s-", color=GSPLAT_COLOR, label="gsplat")
    ax_thru.set_xscale("log", base=2)
    ax_thru.set_xlabel("batch size (cameras)")
    ax_thru.set_ylabel("throughput (images / s)")
    ax_thru.set_title("Throughput vs batch")

    ax_m = axes[1, 1]
    mem = np.array([r["splax"]["peak_bytes"] for r in rows], float)
    ax_m.plot(batches, mem / 1e6, "o-", color=SPLAX_COLOR, label="splax (jax)")
    mem = np.array([r["gsplat"]["peak_bytes"] for r in rows], float)
    ax_m.plot(batches, mem / 1e6, "s-", color=GSPLAT_COLOR, label="gsplat (torch)")
    ax_m.set_xscale("log", base=2)
    ax_m.set_xlabel("batch size (cameras)")
    ax_m.set_ylabel("peak allocator memory (MB)")
    ax_m.set_title("Peak GPU memory vs batch")

    for ax in (ax_t, ax_thru, ax_m):
        ax.set_xticks(batches)
        ax.set_xticklabels([str(b) for b in batches])
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    pdf.savefig(fig)
    plt.close(fig)


def backward_page(pdf: PdfPages, sc: dict) -> None:
    """One page per scene for the training step: time, throughput, and memory curves."""
    rows = sc["rows"]
    batches = np.array([r["batch"] for r in rows])
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    text = f"{sc['name']}  -  training step  -  {sc['n_gaussians']:,} gaussians, "
    text += f"{sc['img_shape'][0]}x{sc['img_shape'][1]}"
    fig.suptitle(text, fontsize=14, fontweight="bold")

    ax_t = axes[0, 0]
    t = np.array([r["splax"]["time_ms"] for r in rows], float)
    ax_t.plot(batches, t, "o-", color=SPLAX_COLOR, label="splax")
    t = np.array([r["gsplat"]["time_ms"] for r in rows], float)
    ax_t.plot(batches, t, "s-", color=GSPLAT_COLOR, label="gsplat")
    ax_t.set_xscale("log", base=2)
    ax_t.set_yscale("log")
    ax_t.set_xlabel("batch size (cameras)")
    ax_t.set_ylabel("step time per call (ms)")
    ax_t.set_title("Training step time vs batch")

    ax_thru = axes[0, 1]
    fps = batches / np.array([r["splax"]["time_ms"] for r in rows], float) * 1e3
    ax_thru.plot(batches, fps, "o-", color=SPLAX_COLOR, label="splax")
    fps = batches / np.array([r["gsplat"]["time_ms"] for r in rows], float) * 1e3
    ax_thru.plot(batches, fps, "s-", color=GSPLAT_COLOR, label="gsplat")
    ax_thru.set_xscale("log", base=2)
    ax_thru.set_xlabel("batch size (cameras)")
    ax_thru.set_ylabel("throughput (images / s)")
    ax_thru.set_title("Throughput vs batch")

    ax_m = axes[1, 0]
    mem = np.array([r["splax"]["peak_bytes"] for r in rows], float)
    ax_m.plot(batches, mem / 1e6, "o-", color=SPLAX_COLOR, label="splax (jax)")
    mem = np.array([r["gsplat"]["peak_bytes"] for r in rows], float)
    ax_m.plot(batches, mem / 1e6, "s-", color=GSPLAT_COLOR, label="gsplat (torch)")
    ax_m.set_xscale("log", base=2)
    ax_m.set_xlabel("batch size (cameras)")
    ax_m.set_ylabel("peak allocator memory (MB)")
    ax_m.set_title("Peak GPU memory vs batch")

    axes[1, 1].axis("off")

    for ax in (ax_t, ax_thru, ax_m):
        ax.set_xticks(batches)
        ax.set_xticklabels([str(b) for b in batches])
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    pdf.savefig(fig)
    plt.close(fig)


def summary_page(pdf: PdfPages, data: dict, title: str) -> None:
    """Table of gsplat/splax time speedup and splax/gsplat memory ratio per batch."""
    scenes = data["scenes"]
    batches = data["meta"]["batches"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8.5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    def table(ax: Axes, key: str, num: str, den: str, title: str) -> None:
        ax.axis("off")
        ax.set_title(title, fontsize=11, pad=12)
        col_labels = ["scene"] + [f"b={b}" for b in batches]
        cells = []
        for sc in scenes:
            row = [sc["name"]]
            for r in sc["rows"]:
                v = r[num][key] / r[den][key]
                row.append("-" if np.isnan(v) else f"{v:.2f}")
            cells.append(row)
        tbl = ax.table(cellText=cells, colLabels=col_labels, loc="center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1, 1.6)

    text = "Time speedup  (gsplat / splax (ms). >1 means splax faster)"
    table(ax1, "time_ms", "gsplat", "splax", text)
    text = "Peak memory ratio  (splax / gsplat (bytes). <1 means splax leaner)"
    table(ax2, "peak_bytes", "splax", "gsplat", text)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    pdf.savefig(fig)
    plt.close(fig)


def build_report(data: dict, bwd: dict | None, out: Path) -> None:
    """Render the full PDF report from the loaded benchmark data."""
    out.parent.mkdir(parents=True, exist_ok=True)
    bwd_scenes = {sc["name"]: sc for sc in bwd["scenes"]} if bwd else {}
    with PdfPages(out) as pdf:
        cover_page(pdf, data)
        for sc in data["scenes"]:
            scene_page(pdf, out.parent, sc)
            if sc["name"] in bwd_scenes:
                backward_page(pdf, bwd_scenes[sc["name"]])
        summary_page(pdf, data, "Summary, forward render: splax relative to gsplat")
        if bwd:
            summary_page(pdf, bwd, "Summary, training step: splax relative to gsplat")


def main() -> None:
    """Build the PDF from an existing benchmark JSON."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, default=OUT_DIR / "benchmark_suite.json")
    ap.add_argument("--backward-json", type=Path, default=OUT_DIR / "bench_backward.json")
    ap.add_argument("--out", type=Path, default=OUT_DIR / "benchmark_suite.pdf")
    args = ap.parse_args()
    data = json.loads(args.json.read_text())
    bwd = json.loads(args.backward_json.read_text()) if args.backward_json.exists() else None
    if bwd is None:
        logger.info(f"no backward results at {args.backward_json}, forward only")
    build_report(data, bwd, args.out)
    logger.info(f"wrote {args.out}")


if __name__ == "__main__":
    main()
