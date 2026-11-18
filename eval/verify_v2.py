"""Visual verification of the new curated v2 sites: contact sheets of the reference run's evidence
figures, 6 sites per sheet, titled with expected answer and reference outcome.

  python -m eval.verify_v2      -> runs/verify_v2/sheet_<k>.png + a summary table on stdout
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from eval.curated_v2 import CANDIDATES, CANNOT_TELL  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "runs" / "eval_v2" / "reference"
OUT = ROOT / "runs" / "verify_v2"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, figs = [], []
    for c in CANDIDATES + CANNOT_TELL:
        f = RES / f"{c['id']}.json"
        if not f.exists():
            rows.append((c["id"], c.get("expected", "cannot_tell"), "MISSING", "", ""))
            continue
        r = json.loads(f.read_text())
        try:
            ch = json.loads(r["explanation"])
        except Exception:
            ch = {}
        rows.append((c["id"], r["expected"], r["answer"], ch.get("changed_fraction"),
                     (r.get("explanation") or "")[:70] if not ch else ""))
        if r.get("evidence"):
            figs.append((c["id"], r["expected"], r["answer"], ch.get("changed_fraction"), r["evidence"]))
    for k in range(0, len(figs), 6):
        chunk = figs[k:k + 6]
        fig, axs = plt.subplots(len(chunk), 1, figsize=(12, 4.6 * len(chunk)))
        axs = axs if len(chunk) > 1 else [axs]
        for ax, (i, e, a, cf, p) in zip(axs, chunk):
            ax.imshow(mpimg.imread(p))
            ax.set_title(f"{i}: expected {e}, reference {a}, changed {cf}", loc="left", fontsize=11)
            ax.axis("off")
        fig.savefig(OUT / f"sheet_{k // 6}.png", dpi=60, bbox_inches="tight")
        plt.close(fig)
    for row in rows:
        ok = "✓" if row[1] == row[2] else "✗"
        print(ok, *row)


if __name__ == "__main__":
    main()
