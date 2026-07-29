"""Build the per-derivative old-vs-new benchmark report as a multi-page PDF.

Reads ``reports/bench_derivatives_old.json`` and ``reports/bench_derivatives_new.json`` and writes
``reports/bench_derivatives_report.pdf`` with a cover, one page per scene comparing the gaussian,
camera, and transform backward cost, and a summary of the per-derivative change. Generate with:

    pixi run -e tests python benchmark/report_derivatives.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

logger = logging.getLogger(__name__)

OUT_DIR = Path(__file__).resolve().parents[1] / "reports"
DERIVS = ("gaussians", "cameras", "transforms")
OLD_COLOR = "#d95f02"
NEW_COLOR = "#1b9e77"
STEADY = 32  # batches at or above this are the amortised regime used for the headline change


def _load(impl: str) -> dict:
    return json.loads((OUT_DIR / f"bench_derivatives_{impl}.json").read_text())


def _series(scene: dict, deriv: str) -> tuple[np.ndarray, np.ndarray]:
    """Batches and times for one derivative, NaN rows dropped."""
    b, t = [], []
    for row in scene["rows"]:
        ms = row[deriv]["time_ms"]
        if ms == ms:  # not NaN
            b.append(row["batch"])
            t.append(ms)
    return np.array(b), np.array(t)


def _pct_change(old: dict, new: dict, deriv: str) -> float:
    """Median percent change new vs old over the amortised batches, positive means new is slower."""
    bo, to = _series(old, deriv)
    bn, tn = _series(new, deriv)
    common = sorted(set(bo[bo >= STEADY]) & set(bn[bn >= STEADY]))
    if not common:
        return float("nan")
    do = {int(b): v for b, v in zip(bo, to)}
    dn = {int(b): v for b, v in zip(bn, tn)}
    return float(np.median([100.0 * (dn[b] - do[b]) / do[b] for b in common]))


def cover_page(pdf: PdfPages, old: dict, new: dict):
    """Title, run metadata, and the headline change per derivative."""
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.5, 0.9, "splax projection backward", ha="center", fontsize=26, fontweight="bold")
    fig.text(
        0.5,
        0.85,
        "Four specialized kernels (old) vs one unified kernel (new), per differentiation argument",
        ha="center",
        fontsize=13,
        color="#555",
    )
    m_old, m_new = old["meta"], new["meta"]
    lines = [
        f"old: {m_old['impl']}    new: {m_new['impl']}",
        f"jax {m_new['jax_version']}    warp {m_new['warp_version']}",
        f"metric: {m_new['metric']}",
        f"warmup {m_new['warmup']}, iters {m_new['iters']}, best of {m_old.get('repeat', 3)}",
        f"transform grad uses {m_new['n_slices']} slices covering all gaussians",
    ]
    fig.text(0.5, 0.7, "\n".join(lines), ha="center", fontsize=11, family="monospace")

    # headline change table
    scenes_old = {s["name"]: s for s in old["scenes"]}
    rows = []
    for sn in new["scenes"]:
        so = scenes_old.get(sn["name"])
        if so is None:
            continue
        rows.append([sn["name"]] + [f"{_pct_change(so, sn, d):+.1f}%" for d in DERIVS])
    ax = fig.add_axes((0.15, 0.28, 0.7, 0.28))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=["scene", *DERIVS], loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)
    fig.text(
        0.5,
        0.2,
        f"median new-vs-old change over batches >= {STEADY} (positive = new slower)",
        ha="center",
        fontsize=10,
        color="#555",
    )
    pdf.savefig(fig)
    plt.close(fig)


def scene_page(pdf: PdfPages, old: dict, new: dict):
    """One page: old vs new time per derivative across the batch sweep."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 8.5 * 0.55))
    fig.suptitle(
        f"{new['name']}  -  {new['n_gaussians']:,} gaussians, {new['res']}x{new['res']}",
        fontsize=15,
        fontweight="bold",
    )
    for ax, deriv in zip(axes, DERIVS):
        bo, to = _series(old, deriv)
        bn, tn = _series(new, deriv)
        ax.plot(bo, to, "-o", ms=3, color=OLD_COLOR, label="old")
        ax.plot(bn, tn, "-o", ms=3, color=NEW_COLOR, label="new")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(f"{deriv}   ({_pct_change(old, new, deriv):+.1f}%)", fontsize=12)
        ax.set_xlabel("batch")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=9)
    axes[0].set_ylabel("forward + backward step (ms)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    pdf.savefig(fig)
    plt.close(fig)


def summary_page(pdf: PdfPages, old: dict, new: dict):
    """Grouped bars of the median new-vs-old change per scene and derivative."""
    fig, ax = plt.subplots(figsize=(11, 8.5 * 0.6))
    scenes_old = {s["name"]: s for s in old["scenes"]}
    names = [s["name"] for s in new["scenes"] if s["name"] in scenes_old]
    x = np.arange(len(names))
    width = 0.25
    for i, deriv in enumerate(DERIVS):
        vals = [
            _pct_change(scenes_old[n], next(s for s in new["scenes"] if s["name"] == n), deriv)
            for n in names
        ]
        ax.bar(x + (i - 1) * width, vals, width, label=deriv)
    ax.axhline(0, color="#333", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel(f"median change new vs old, batches >= {STEADY} (%)")
    ax.set_title(
        "Per-derivative cost of unifying the backward kernel", fontsize=14, fontweight="bold"
    )
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def main():
    """Assemble the multi-page PDF from the two benchmark JSON files."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    old, new = _load("old"), _load("new")
    scenes_old = {s["name"]: s for s in old["scenes"]}
    out = OUT_DIR / "bench_derivatives_report.pdf"
    with PdfPages(out) as pdf:
        cover_page(pdf, old, new)
        for sn in new["scenes"]:
            if sn["name"] in scenes_old:
                scene_page(pdf, scenes_old[sn["name"]], sn)
        summary_page(pdf, old, new)
    logger.info("wrote %s", out)


if __name__ == "__main__":
    main()
