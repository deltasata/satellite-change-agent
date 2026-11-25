"""Evidence figure for a time-series change result: what the agent shows the user.

Panels: before composite (true colour) | after composite | change map over the
after image | clear-fraction timeline with the two windows shaded.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

SURFACE, INK, INK_2, MUTED, SERIES_1, SERIES_2 = "#fcfcfb", "#0b0b0b", "#52514e", "#b9b8b3", "#2a78d6", "#eb6834"


def _rgb(red, green, blue, lo=0.0, hi=0.25):
    """True-colour image from reflectance. Callers pass the SAME lo/hi for before and
    after, so the two panels compare honestly."""
    x = np.stack([red, green, blue], -1)
    return np.clip((np.nan_to_num(x, nan=0.0) - lo) / (hi - lo), 0, 1) ** (1 / 1.4)


def _common_stretch(comp) -> tuple[float, float]:
    """One stretch for both dates: 1st-99th percentile of all RGB values of before and after.
    A fixed 0-0.25 stretch saturated bright deserts (found on the Lusail benchmark site)."""
    v = np.concatenate([comp[b][k].ravel() for b in ("red", "green", "blue") for k in (0, 1)])
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0, 0.25
    lo, hi = np.percentile(v, [1, 99])
    return float(min(lo, 0.0)), float(max(hi, 0.05))


def render_evidence(
    ds: xr.Dataset,
    result: dict,
    before: tuple[date, date],
    after: tuple[date, date],
    out_path: str | Path,
    title: str = "",
    water: str = "ndwi",
    change: str = "rule",
    cloud: str = "scl",
) -> Path:
    """Write the evidence PNG and return its path. `result` comes from
    core.models.baseline.detect_change_timeseries; `water`, `change` and `cloud` name the detectors that
    made it, so that the third panel and the footnote show what the answer was actually based on: new water
    for a flood question (a trained water detector), otherwise the changed pixels of the change detector."""
    import re

    from core.models.change import NAMES as CHANGE_NAMES
    from core.models.water import NAMES as WATER_NAMES

    def plain(name: str) -> str:        # visitors see the method, not our milestone labels
        return re.sub(r"\s*\((milestones?|B\d)[^)]*\)", "", name)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK_2,
                         "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK})
    fig = plt.figure(figsize=(13, 6.2), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 3, height_ratios=[3, 1.1], hspace=0.28, wspace=0.05)

    if result.get("status") == "ok":
        comp = result["_composites"]
        lo, hi = _common_stretch(comp)
        rb = _rgb(comp["red"][0], comp["green"][0], comp["blue"][0], lo, hi)
        ra = _rgb(comp["red"][1], comp["green"][1], comp["blue"][1], lo, hi)
        changed = result["_changed"]
        panels = [(rb, f"before: median of clear obs\n{before[0]} → {before[1]}"),
                  (ra, f"after: median of clear obs\n{after[0]} → {after[1]}")]
        for j, (img, ttl) in enumerate(panels):
            ax = fig.add_subplot(gs[0, j])
            ax.imshow(img, interpolation="nearest")
            ax.set_title(ttl, loc="left", fontsize=9)
            ax.axis("off")
        ax = fig.add_subplot(gs[0, 2])
        ax.imshow(ra * 0.55 + 0.45 * ra.mean(-1, keepdims=True), interpolation="nearest")
        if water != "ndwi":                     # a flood question: the answer is the new water
            mask, colour = result["_new_water"], (0.16, 0.47, 0.84, 0.85)
            frac = result["water_fraction_after"] - result["water_fraction_before"]
            ttl = (f"new water (blue): {frac:.1%} of usable area\n"
                   f"water in the after image, not in the before ({plain(WATER_NAMES.get(water.split('-without-')[0], water))})")
        else:
            mask, colour = changed, (0.92, 0.41, 0.20, 0.85)
            how = (f"change > {result['z_threshold']}× the date-to-date scatter (before window)" if change == "rule"
                   else plain(CHANGE_NAMES.get(change.split("-without-")[0], change)))
            ttl = f"changed pixels (orange): {result['changed_fraction']:.1%} of usable area\n{how}"
        ov = np.zeros(mask.shape + (4,))
        ov[mask] = colour
        ax.imshow(ov, interpolation="nearest")
        ax.set_title(ttl, loc="left", fontsize=9)
        ax.axis("off")
    else:
        ax = fig.add_subplot(gs[0, :])
        ax.axis("off")
        ax.text(0.5, 0.5, "CANNOT TELL\n" + result.get("reason", ""), ha="center", va="center",
                fontsize=13, color=INK)

    ax = fig.add_subplot(gs[1, :])
    t = ds.time.values.astype("datetime64[s]").astype(object)
    vf = ds["valid_fraction"].values
    ax.vlines(t, 0, vf, color=SERIES_1, lw=2)
    for (d0, d1), lab in ((before, "before"), (after, "after")):
        ax.axvspan(np.datetime64(d0), np.datetime64(d1) + np.timedelta64(1, "D"), color=MUTED, alpha=0.35, lw=0)
        ax.text(np.datetime64(d0), 1.05, f" {lab}", fontsize=8, color=INK_2, va="bottom")
    ax.set_ylim(0, 1.25)
    ax.set_ylabel("clear fraction")
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
    ax.set_title(f"every acquisition in the search period ({len(t)}); "
                 f"shaded = windows compared", loc="left", fontsize=9)

    fig.suptitle(title, x=0.01, ha="left", fontsize=11)
    clouds = {"scl": "ESA scene classification", "ocm": "OmniCloudMask"}.get(cloud, cloud)
    fig.text(0.01, 0.005, f"Sentinel-2 L2A (Element84 Earth Search); clouds and shadows: {clouds}; "
             f"bbox {ds.attrs.get('bbox')}",
             fontsize=7, color=INK_2)
    fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return out_path
