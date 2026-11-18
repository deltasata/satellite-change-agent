"""Turn ESA's Scene Classification Layer (SCL) into a valid-pixel mask.

Every Sentinel-2 L2A scene comes with SCL: one integer class per 20 m pixel,
produced by ESA's processor. We trust it as-is (a known systematic: it misses
some thin cloud and some cloud shadow, and occasionally flags bright roofs as
cloud). A better detector can replace this function later without changing
anything that calls it.
"""

from __future__ import annotations

import numpy as np

SCL_CLASSES: dict[int, str] = {
    0: "no_data",
    1: "saturated_or_defective",
    2: "dark_area_pixels",
    3: "cloud_shadows",
    4: "vegetation",
    5: "not_vegetated",
    6: "water",
    7: "unclassified",
    8: "cloud_medium_probability",
    9: "cloud_high_probability",
    10: "thin_cirrus",
    11: "snow",
}

# Pixels we refuse to use. "dark_area_pixels" (2) and "unclassified" (7) are kept:
# they are often real surfaces (dark roofs, fresh asphalt, wet soil), and change
# detection cares exactly about such surfaces.
INVALID_SCL: frozenset[int] = frozenset({0, 1, 3, 8, 9, 10})


def valid_mask(scl: np.ndarray, invalid: frozenset[int] = INVALID_SCL) -> np.ndarray:
    """True where the pixel is a usable surface observation.

    Works on any array shape; NaN (e.g. outside the loaded area) counts as invalid.
    """
    scl = np.asarray(scl)
    finite = np.isfinite(scl) if np.issubdtype(scl.dtype, np.floating) else np.ones(scl.shape, bool)
    codes = np.where(finite, scl, 0).astype(np.int64)
    return finite & ~np.isin(codes, list(invalid))
