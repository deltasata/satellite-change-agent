"""Draw docs/figures/results.png: every learned model against the rule it had to beat.

The numbers are the measured results (runs/b1/report.json, runs/b2/report_A.json, runs/b2/report_C.json,
runs/b3/report_A.json, runs/b4/report_A.json);
they are written here so that the figure can be redrawn without the 10 GB of intermediate files.

    python scripts/make_figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures" / "results.png"
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
RULE, MODEL = "#a3a29c", "#2a78d6"

PANELS = [
    ("Water in optical images\nwater IoU, 10 floods each held out", [
        ("NDWI > 0 (standard rule)", 0.690, "rule"),
        ("NDWI > t, t fitted", 0.621, "rule"),
        ("MNDWI > t, t fitted", 0.777, "rule"),
        ("TerraMind-1.0-base, fine-tuned", 0.698, "model"),
        ("Prithvi-EO-2.0-300M, fine-tuned", 0.712, "model"),
        ("U-Net, trained from scratch", 0.833, "adopted"),
    ], "+0.143 [+0.083, +0.219]\nover NDWI > 0"),
    ("Water in radar images\nwater IoU, 10 floods each held out", [
        ("Otsu threshold per tile (VH)", 0.199, "rule"),
        ("VV < t, t fitted", 0.427, "rule"),
        ("VH < t, t fitted", 0.477, "rule"),
        ("U-Net, trained from scratch", 0.558, "model"),
        ("TerraMind-1.0-base, fine-tuned", 0.611, "adopted"),
    ], "+0.134 [+0.090, +0.192]\nover VH < t"),
    ("Flood questions answered correctly\n72 benchmark questions, end to end", [
        ("optical, NDWI > 0", 0.639, "rule"),
        ("optical, U-Net", 0.722, "model"),
        ("optical U-Net + radar rule", 0.875, "model"),
        ("optical U-Net + radar TerraMind", 0.889, "adopted"),
    ], "+0.167 [+0.031, +0.346]\nover optical U-Net"),
    ("Which pixels are cloud or shadow\nIoU of unusable, 953 expert-labelled images", [
        ("ESA scene classification", 0.644, "rule"),
        ("same, more classes, widened", 0.681, "rule"),
        ("U-Net, trained from scratch", 0.844, "model"),
        ("TerraMind-1.0-base, fine-tuned", 0.878, "model"),
        ("OmniCloudMask, as published", 0.885, "adopted"),
    ], "+0.241 [+0.221, +0.261]\nover ESA's classification"),
    ("Which pixels changed through construction\nIoU of changed, 24 OSCD cities, groups held out", [
        ("change / scatter > 4", 0.188, "rule"),
        ("same, threshold fitted", 0.188, "rule"),
        ("FC-Siam-diff, from scratch", 0.225, "model"),
        ("TerraMind-1.0-base, Siamese", 0.301, "adopted"),
    ], "+0.114 [+0.052, +0.180]\nover the rule"),
]


def main() -> None:
    plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": INK, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(2, 3, figsize=(18, 9.6), facecolor=SURFACE)
    axes = axes.ravel()
    axes[-1].axis("off")
    for ax, (title, rows, note) in zip(axes, PANELS):
        ax.set_facecolor(SURFACE)
        names = [r[0] for r in rows]
        vals = [r[1] for r in rows]
        cols = [RULE if r[2] == "rule" else MODEL for r in rows]
        y = range(len(rows))
        ax.barh(y, vals, color=cols, height=0.62, edgecolor=SURFACE, linewidth=2)
        for i, (v, r) in enumerate(zip(vals, rows)):
            ax.text(v + 0.012, i, f"{v:.3f}" + ("  adopted" if r[2] == "adopted" else ""), va="center",
                    fontsize=9.5, color=INK, fontweight="bold" if r[2] == "adopted" else "normal")
        ax.set_yticks(list(y), names)
        ax.set_xlim(0, 1.32)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(axis="y", length=0)
        ax.set_title(title, loc="left", fontsize=11, color=INK, pad=30)
        ax.text(0, 1.015, "adopted model: " + note.replace("\n", " "), transform=ax.transAxes,
                fontsize=8.5, color=MUTED, va="bottom")
    handles = [plt.Rectangle((0, 0), 1, 1, color=RULE), plt.Rectangle((0, 0), 1, 1, color=MODEL)]
    fig.legend(handles, ["rule (threshold)", "learned model"], loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.text(0.01, -0.02, "Interval: 95 %, resampling whole flood events, places or groups of cities. Protocols fixed before any score: "
             "docs/METHODS.md", fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=110, facecolor=SURFACE, bbox_inches="tight")
    print(OUT)


if __name__ == "__main__":
    main()
