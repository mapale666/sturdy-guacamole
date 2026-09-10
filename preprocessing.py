"""
preprocessing.py
=================
Shared preprocessing utilities for the seal numeral recognition task.

FIX #17 (this revision -- component-cluster row extraction + degenerate
binarization retry, replacing FIX #16's band-gating approach):

  Diagnosis from debug_visualize.py --dump-components on real failures:

  - 01309.png: the digit-row y-cluster in the tag crop had only 6 of 7
    expected digit-shaped components (one digit -- likely worn/faint --
    was dropped by the aspect/fill shape filter, visible as a ~2x
    spacing gap between two neighboring x-positions). _best_row_group
    requires len(group) >= n_digits, so it returned None for a cluster
    that was one component short. FIX #16's _row_is_plausible guard then
    rejected the only detected text-row band (which had merged the
    brand-name text and the numeral row into one near-full-height band,
    since this photo is blurrier and the ink-density gap between the two
    text rows didn't register) *before* the valley splitter ever got a
    properly tight crop to work with -- so it fell through to the whole-
    image fallback and produced boxes spanning nearly the whole tag.
    FIXED: row location is no longer derived from the coarse ink-density
    band. Instead, _find_row_extent() clusters digit-shaped components
    directly by y-proximity and accepts a cluster that is missing UP TO
    HALF of n_digits (a worn/faint digit failing the shape filter is far
    more likely than the row being in the wrong place), then re-crops
    TIGHTLY to that cluster's own y-range before running _best_row_group
    or the valley splitter on it. A tight crop means the column ink
    profile used by the valley splitter is no longer contaminated by
    brand-name text sitting above it.

  - 00064.png: only ONE deduped component was found per polarity,
    covering nearly the entire tag crop (fill ~0.17-0.29). This is not a
    segmentation-logic bug -- Otsu/adaptive thresholding fused the whole
    digit row (or more) into a single low-fill blob on this low-contrast
    photo, so there was nothing for component-based logic to work with
    regardless of how it's structured. FIXED: _binarize_tag_crop() now
    detects when the first binarization pass produces a single
    connected component covering more than half the tag crop area (a
    strong signal of a degenerate, too-permissive threshold) and retries
    with a finer adaptive threshold (smaller block size, tuned for
    stroke-level detail rather than region-level blobs) before falling
    through to component analysis.

Prior fix history (FIX #1-16) retained: the two-stage tag-boundary
detection (Stage 1, multi-attempt Canny thresholds with morphological
closing), ordering fix in _best_subset (always returns components sorted
by x), and the component-shape filters (aspect ratio, fill ratio, nested-
component removal, height/spacing regularity, combinatorial best-subset
selection) originally developed against whole-image noise before being
narrowed to operate on small, clean tag crops.

Public entry points typically called from main.py / model.py:

load_image(path) -> BGR np.ndarray
locate_digit_row_multi_orientation(bgr) -> (oriented_gray, boxes|None)
extract_digit_crops(gray, boxes, ...) -> list[np.ndarray] (binary, numeral=255, per-crop thresholded)
prepare_isolated_crop(gray) -> binary np.ndarray (for pre-cropped training images AND per-digit inference crops)
to_classical_canvas(crop, size=32) -> binary np.ndarray, numeral=1
to_cnn_tensor_array(crop, size=28) -> float32 np.ndarray in [0,1], shape (size,size)
to_vlm_bytes(bgr, boxes=None, ...) -> PNG bytes (optionally cropped/enhanced, NOT binarized)
preprocess_seal(path, mode, n_digits, img_size) -> convenience wrapper, see bottom of file
"""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass
from itertools import combinations
from typing import List, Optional, Tuple

# --------------------------------------------------------------------------- #
# Basic I/O
# --------------------------------------------------------------------------- #

def load_image(path: str) -> np.ndarray:
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return im


def to_gray(bgr: np.ndarray) -> np.ndarray:
    if bgr.ndim == 2:
        return bgr
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


# --------------------------------------------------------------------------- #
# Illumination / exposure correction (FULL SEAL IMAGES ONLY)
# --------------------------------------------------------------------------- #

def correct_illumination(gray: np.ndarray, blur_ksize: int = 51) -> np.ndarray:
    """
    Flattens vignetting / uneven exposure by dividing out a heavily blurred
    version of the image, then re-normalizes contrast with CLAHE. DO NOT
    use this on small, already-cropped individual digit images -- see
    prepare_isolated_crop() instead.
    """
    h, w = gray.shape[:2]
    ksize = min(blur_ksize, max(3, min(h, w) - 1))
    ksize = ksize | 1
    background = cv2.GaussianBlur(gray, (ksize, ksize), 0)
    background = np.where(background == 0, 1, background)
    normalized = (gray.astype(np.float32) / background.astype(np.float32)) * 128.0
    normalized = np.clip(normalized, 0, 255).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    return clahe.apply(normalized)


def denoise(gray: np.ndarray, ksize: int = 3) -> np.ndarray:
    h, w = gray.shape[:2]
    ksize = min(ksize, max(1, min(h, w) - 1))
    ksize = ksize | 1
    return cv2.GaussianBlur(gray, (ksize, ksize), 0)


# --------------------------------------------------------------------------- #
# Binarization with polarity normalization
# --------------------------------------------------------------------------- #

def robust_binarize(gray: np.ndarray, polarity: str = "area") -> np.ndarray:
    """
    Produces a binary image with numeral pixels = 255, background = 0.
    Tries Otsu first, falls back to adaptive thresholding if degenerate.
    """
    blurred = denoise(gray, 3)
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    fg_ratio = np.count_nonzero(otsu) / otsu.size
    if fg_ratio < 0.02 or fg_ratio > 0.60:
        block = max(3, (min(blurred.shape) // 4) | 1)
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, block, 5,
        )
    else:
        binary = otsu

    if polarity == "border":
        binary = normalize_polarity_by_border(binary)
    else:
        binary = normalize_polarity(binary)
    return binary


def normalize_polarity(binary: np.ndarray) -> np.ndarray:
    """Forces numeral = foreground = 255, assuming numeral occupies less area."""
    fg_ratio = np.count_nonzero(binary) / binary.size
    if fg_ratio > 0.5:
        binary = cv2.bitwise_not(binary)
    return binary


def normalize_polarity_by_border(binary: np.ndarray, border_frac: float = 0.08) -> np.ndarray:
    h, w = binary.shape[:2]
    bx = max(1, int(round(w * border_frac)))
    by = max(1, int(round(h * border_frac)))
    border_pixels = np.concatenate([
        binary[:by, :].ravel(),
        binary[-by:, :].ravel(),
        binary[:, :bx].ravel(),
        binary[:, -bx:].ravel(),
    ])
    if border_pixels.size == 0:
        return normalize_polarity(binary)

    border_white_ratio = np.count_nonzero(border_pixels) / border_pixels.size
    if border_white_ratio > 0.5:
        binary = cv2.bitwise_not(binary)
    return binary


# --------------------------------------------------------------------------- #
# Morphological cleanup
# --------------------------------------------------------------------------- #

def clean_mask(binary: np.ndarray, ksize: int = 3) -> np.ndarray:
    h, w = binary.shape[:2]
    ksize = min(ksize, max(1, min(h, w) - 1))
    if ksize < 2:
        return binary
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    return closed


# --------------------------------------------------------------------------- #
# Isolated-crop normalization (ground_truth_chars_balanced -- TRAINING DATA,
# AND every individual digit crop pulled out of a full seal photo)
# --------------------------------------------------------------------------- #

def prepare_isolated_crop(gray: np.ndarray) -> np.ndarray:
    """
    Binarizes an already-isolated digit crop WITHOUT full-image illumination
    correction, using its own LOCAL Otsu threshold and border-based
    polarity.
    """
    h, w = gray.shape[:2]
    denoised = denoise(gray, 3)
    binary = robust_binarize(denoised, polarity="border")
    if min(h, w) >= 15:
        binary = clean_mask(binary, ksize=2)
    return binary


# --------------------------------------------------------------------------- #
# Rotation handling
# --------------------------------------------------------------------------- #

def estimate_rotation_angle(binary: np.ndarray) -> float:
    coords = cv2.findNonZero(binary)
    if coords is None or len(coords) < 10:
        return 0.0
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    if angle < -45:
        angle += 90
    if angle > 45:
        angle -= 90
    return angle


def rotate_image(bgr_or_gray: np.ndarray, angle: float) -> np.ndarray:
    if abs(angle) < 0.5:
        return bgr_or_gray
    h, w = bgr_or_gray.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        bgr_or_gray, M, (w, h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )


def deskew_digit(binary_crop: np.ndarray) -> np.ndarray:
    m = cv2.moments(binary_crop)
    if abs(m["mu02"]) < 1e-2:
        return binary_crop
    skew = m["mu11"] / m["mu02"]
    h, w = binary_crop.shape
    M = np.float32([[1, skew, -0.5 * skew * h], [0, 1, 0]])
    return cv2.warpAffine(
        binary_crop, M, (w, h),
        flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR,
        borderValue=0,
    )


def try_all_orientations(gray: np.ndarray) -> List[np.ndarray]:
    return [
        gray,
        cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE),
        cv2.rotate(gray, cv2.ROTATE_180),
        cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE),
    ]


# --------------------------------------------------------------------------- #
# Component-shape helpers (used by Stage 3, reused from prior fix history)
# --------------------------------------------------------------------------- #

@dataclass
class Component:
    x: int
    y: int
    w: int
    h: int
    area: int
    cx: float
    cy: float


def _get_components(binary: np.ndarray, min_area_frac: float = 0.0008,
                     max_area_frac: float = 0.35) -> List[Component]:
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    img_area = binary.shape[0] * binary.shape[1]
    min_area = img_area * min_area_frac
    max_area = img_area * max_area_frac

    comps = []
    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area or area > max_area:
            continue
        x, y, w, h = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                      stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
        cx, cy = centroids[i]
        comps.append(Component(x, y, w, h, int(area), cx, cy))
    return comps


def _looks_like_digit(c: Component, min_aspect: float = 1.15, max_aspect: float = 6.0,
                       min_fill_ratio: float = 0.15) -> bool:
    if c.w <= 0 or c.h <= 0:
        return False
    aspect = c.h / c.w
    if not (min_aspect <= aspect <= max_aspect):
        return False
    fill_ratio = c.area / float(c.w * c.h)
    return fill_ratio >= min_fill_ratio


def _bbox_overlap_frac(inner: Component, outer: Component) -> float:
    ix0, iy0, ix1, iy1 = inner.x, inner.y, inner.x + inner.w, inner.y + inner.h
    ox0, oy0, ox1, oy1 = outer.x, outer.y, outer.x + outer.w, outer.y + outer.h
    left, top = max(ix0, ox0), max(iy0, oy0)
    right, bottom = min(ix1, ox1), min(iy1, oy1)
    if right <= left or bottom <= top:
        return 0.0
    overlap_area = (right - left) * (bottom - top)
    inner_area = inner.w * inner.h
    return overlap_area / inner_area if inner_area > 0 else 0.0


def _remove_nested_components(comps: List[Component], overlap_thresh: float = 0.75) -> List[Component]:
    comps_sorted = sorted(comps, key=lambda c: c.w * c.h, reverse=True)
    kept: List[Component] = []
    for c in comps_sorted:
        if any(_bbox_overlap_frac(c, k) >= overlap_thresh for k in kept):
            continue
        kept.append(c)
    return kept


def _coeff_variation(values: np.ndarray) -> float:
    if len(values) == 0 or values.mean() <= 0:
        return float("inf")
    return float(values.std() / values.mean())


def _group_score(group: List[Component]) -> float:
    widths = np.array([c.w for c in group], dtype=np.float32)
    heights = np.array([c.h for c in group], dtype=np.float32)
    width_cv = _coeff_variation(widths)
    height_cv = _coeff_variation(heights)
    xs = np.sort(np.array([c.x for c in group], dtype=np.float32))
    spacing = np.diff(xs)
    spacing_cv = float(spacing.std() / spacing.mean()) if len(spacing) > 0 and spacing.mean() > 0 else 0.0
    return width_cv + height_cv + spacing_cv


def _group_is_regular(group: List[Component], tol: float = 0.35) -> bool:
    if len(group) < 2:
        return True
    widths = np.array([c.w for c in group], dtype=np.float32)
    heights = np.array([c.h for c in group], dtype=np.float32)
    width_cv = _coeff_variation(widths)
    height_cv = _coeff_variation(heights)
    xs = np.sort(np.array([c.x for c in group], dtype=np.float32))
    spacing = np.diff(xs)
    spacing_cv = float(spacing.std() / spacing.mean()) if len(spacing) > 0 and spacing.mean() > 0 else 0.0
    return width_cv <= tol and height_cv <= tol and spacing_cv <= tol


def _best_subset(group: List[Component], n_digits: int, max_search_extra: int = 6) -> List[Component]:
    # FIX #16.1: always return components sorted by x. Connected-component
    # labeling order is a raster scan, not left-to-right reading order, so
    # returning `group` as-is here (the previous bug) scrambled downstream
    # box indices even though each box's own position was correct.
    if len(group) <= n_digits:
        return sorted(group, key=lambda c: c.x)
    if len(group) - n_digits <= max_search_extra:
        best = None
        best_score = float("inf")
        for combo in combinations(group, n_digits):
            combo_sorted = sorted(combo, key=lambda c: c.x)
            score = _group_score(combo_sorted)
            if score < best_score:
                best_score = score
                best = combo_sorted
        return list(best)

    heights = sorted(c.h for c in group)
    ref_h = heights[len(heights) // 2]
    trimmed = sorted(group, key=lambda c: abs(c.h - ref_h))[:n_digits]
    trimmed.sort(key=lambda c: c.x)
    return trimmed


def _uniform_slice_boxes(extent: Tuple[int, int, int, int],
                          n_digits: int) -> List[Tuple[int, int, int, int]]:
    x0, y0, x1, y1 = extent
    seg_w = (x1 - x0) / n_digits
    boxes = []
    for i in range(n_digits):
        bx0 = int(round(x0 + i * seg_w))
        bx1 = int(round(x0 + (i + 1) * seg_w))
        boxes.append((bx0, y0, max(1, bx1 - bx0), y1 - y0))
    return boxes


def _best_row_group(binary: np.ndarray, n_digits: int,
                     row_tolerance_frac: float = 0.4) -> Optional[List[Component]]:
    """
    Given a binary mask (already restricted to a small, clean region --
    e.g. a tightly-cropped digit row within a tag crop), finds the best-
    scoring, regular group of n_digits digit-shaped components.
    """
    comps = _get_components(binary)
    comps = [c for c in comps if _looks_like_digit(c)]
    comps = _remove_nested_components(comps)
    if len(comps) < n_digits:
        return None

    heights = sorted(c.h for c in comps)
    ref_h = heights[len(heights) // 2]
    row_tol = ref_h * row_tolerance_frac

    comps_sorted_by_y = sorted(comps, key=lambda c: c.cy)
    best_group, best_score = None, float("inf")
    seen_cy = set()
    for base in comps_sorted_by_y:
        key = round(base.cy)
        if key in seen_cy:
            continue
        seen_cy.add(key)
        group = [c for c in comps if abs(c.cy - base.cy) <= row_tol]
        if len(group) < n_digits:
            continue
        trimmed = _best_subset(group, n_digits)
        if not _group_is_regular(trimmed):
            continue
        score = _group_score(trimmed)
        if score < best_score:
            best_score = score
            best_group = trimmed
    return best_group


# --------------------------------------------------------------------------- #
# FIX #17.1: component-cluster row extraction. Locates the digit row's
# y-extent directly from digit-shaped components' positions, tolerant of
# up to half of n_digits being missing (a worn/faint/blurred digit
# failing the shape filter is far more likely than "the row is
# elsewhere"). This replaces relying on the coarse ink-density band from
# _find_text_row_bands, which can merge brand text and the numeral row
# together on lower-contrast photos.
# --------------------------------------------------------------------------- #

def _find_row_extent(comps: List[Component], n_digits: int,
                      row_tolerance_frac: float = 0.4,
                      min_count_frac: float = 0.5) -> Optional[Tuple[int, int]]:
    """
    Finds the (y0, y1) extent of the most plausible digit row among
    shape-filtered components. Prefers the LOWEST sufficiently-populated,
    height-consistent cluster (digits sit below any brand text), even if
    its component count is a bit short of n_digits.
    """
    if not comps:
        return None
    min_count = max(2, int(round(n_digits * min_count_frac)))
    heights = sorted(c.h for c in comps)
    ref_h = heights[len(heights) // 2]
    row_tol = ref_h * row_tolerance_frac

    comps_sorted_by_y = sorted(comps, key=lambda c: c.cy)
    best = None  # (mean_cy, group)
    seen = set()
    for base in comps_sorted_by_y:
        key = round(base.cy)
        if key in seen:
            continue
        seen.add(key)
        group = [c for c in comps if abs(c.cy - base.cy) <= row_tol]
        if len(group) < min_count:
            continue
        heights_arr = np.array([c.h for c in group], dtype=np.float32)
        if _coeff_variation(heights_arr) > 0.35:
            continue
        mean_cy = float(np.mean([c.cy for c in group]))
        if best is None or mean_cy > best[0]:
            best = (mean_cy, group)

    if best is None:
        return None
    group = best[1]
    y0 = min(c.y for c in group)
    y1 = max(c.y + c.h for c in group)
    return y0, y1


# --------------------------------------------------------------------------- #
# FIX #17.2: projection-profile valley splitter (Stage 3, second-choice
# method). Used when the row is clean enough that a digit row exists, but
# connected-component analysis can't cleanly separate n_digits blobs --
# typically because two digits are touching/merged, or one digit's stroke
# is broken/faint. Requires a TIGHTLY-cropped row (see _find_row_extent
# above) -- if brand text is included in the crop its ink contaminates
# the column profile and produces a bad split.
# --------------------------------------------------------------------------- #

def _valley_split_row(row_binary: np.ndarray, n_digits: int) -> Optional[List[Tuple[int, int, int, int]]]:
    """
    Splits a binary text-row into n_digits boxes using local minima of the
    column ink-density profile, searched near evenly-spaced expected cut
    positions. Returns boxes (x, y, w, h) in row_binary's own coordinate
    space, or None if the split looks degenerate.
    """
    col_profile = row_binary.sum(axis=0).astype(np.float32) / 255.0
    nz = np.where(col_profile > 0)[0]
    if len(nz) == 0:
        return None
    x0, x1 = int(nz.min()), int(nz.max()) + 1
    profile = col_profile[x0:x1]
    width = x1 - x0
    if width < n_digits * 4:
        return None

    kernel = np.ones(3, dtype=np.float32) / 3
    smoothed = np.convolve(profile, kernel, mode="same")

    expected = [width * i / n_digits for i in range(1, n_digits)]
    search_radius = max(3, int(width / n_digits * 0.4))

    cuts = []
    for e in expected:
        lo, hi = max(1, int(e - search_radius)), min(width - 1, int(e + search_radius))
        if hi <= lo:
            cuts.append(int(e))
            continue
        window = smoothed[lo:hi]
        cuts.append(lo + int(np.argmin(window)))

    cuts = sorted(set(cuts))
    if len(cuts) != n_digits - 1:
        return None  # ambiguous split -- let the caller fall back

    bounds = [0] + cuts + [width]
    boxes = []
    for i in range(n_digits):
        bx0, bx1 = bounds[i], bounds[i + 1]
        seg = row_binary[:, x0 + bx0: x0 + bx1]
        rows_nz = np.where(seg.sum(axis=1) > 0)[0]
        if len(rows_nz) == 0:
            return None  # an empty segment means the split is wrong
        by0, by1 = int(rows_nz.min()), int(rows_nz.max()) + 1
        boxes.append((x0 + bx0, by0, bx1 - bx0, by1 - by0))
    return boxes


# --------------------------------------------------------------------------- #
# STAGE 1: tag boundary detection
# --------------------------------------------------------------------------- #

def _find_tag_bbox_attempt(blurred: np.ndarray, img_area: float,
                            low: int, high: int) -> Optional[Tuple[int, int, int, int]]:
    edges = cv2.Canny(blurred, low, high)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    best_bbox = None
    best_area = 0.0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < img_area * 0.03 or area > img_area * 0.75:
            continue
        x, y, cw, ch = cv2.boundingRect(cnt)
        rect_area = cw * ch
        if rect_area <= 0:
            continue
        fill = area / rect_area
        if fill < 0.30:
            continue
        aspect = ch / cw if cw > 0 else 0
        if not (0.4 <= aspect <= 3.0):
            continue
        if area > best_area:
            best_area = area
            best_bbox = (x, y, x + cw, y + ch)

    return best_bbox


def _find_tag_bbox(gray: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Locates the raised/molded tag plaque via its embossed border contour,
    which is present regardless of whether the tag's interior is light-
    on-dark or dark-on-light. Returns (x0, y0, x1, y1) in image
    coordinates, or None if no suitable contour is found.

    FIX #16.2: retries with a short list of Canny threshold pairs -- the
    original fixed pair first (so already-working cases don't change),
    then progressively looser ones, including one auto-derived from the
    image's own median intensity -- with a morphological CLOSE to bridge
    small gaps in a faint/blurry embossed border.
    """
    h, w = gray.shape[:2]
    img_area = h * w
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    median_val = float(np.median(blurred))
    auto_low = max(10, int(median_val * 0.5))
    auto_high = min(250, int(median_val * 1.5))

    for low, high in [(40, 120), (auto_low, auto_high), (20, 80), (60, 180)]:
        bbox = _find_tag_bbox_attempt(blurred, img_area, low, high)
        if bbox is not None:
            return bbox
    return None


# --------------------------------------------------------------------------- #
# STAGE 2 (legacy): text row-band detection within the tag crop.
# Retained for reference / potential reuse, but no longer used as a hard
# gate by _locate_within_tag -- see FIX #17.1 above for why.
# --------------------------------------------------------------------------- #

def _find_text_row_bands(binary: np.ndarray, min_frac: float = 0.12,
                          min_band_height_frac: float = 0.04,
                          gap_tol_frac: float = 0.015) -> List[Tuple[int, int]]:
    """
    Finds contiguous rows of significant ink density within `binary`
    (expected to already be restricted to the tag interior). Returns a
    list of (y0, y1) bands, in ascending y order. Small gaps are merged
    so a single row of text with minor internal gaps isn't split into
    multiple fragments.
    """
    h = binary.shape[0]
    row_profile = binary.sum(axis=1).astype(np.float32) / 255.0
    if row_profile.max() <= 0:
        return []
    threshold = row_profile.max() * min_frac
    mask = row_profile > threshold

    raw_bands = []
    in_band, start = False, 0
    for i, v in enumerate(mask):
        if v and not in_band:
            start, in_band = i, True
        elif not v and in_band:
            raw_bands.append((start, i))
            in_band = False
    if in_band:
        raw_bands.append((start, len(mask)))

    gap_tol = max(3, int(h * gap_tol_frac))
    merged: List[List[int]] = []
    for b in raw_bands:
        if merged and b[0] - merged[-1][1] <= gap_tol:
            merged[-1][1] = b[1]
        else:
            merged.append([b[0], b[1]])

    min_band_height = max(8, int(h * min_band_height_frac))
    return [(b[0], b[1]) for b in merged if (b[1] - b[0]) >= min_band_height]


# --------------------------------------------------------------------------- #
# FIX #17.3: degenerate-blob detection + finer-threshold retry.
# --------------------------------------------------------------------------- #

def _largest_component_area_frac(binary: np.ndarray) -> float:
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n_labels <= 1:
        return 0.0
    areas = stats[1:, cv2.CC_STAT_AREA]
    return float(areas.max()) / float(binary.shape[0] * binary.shape[1])


def _binarize_tag_crop(corrected: np.ndarray, polarity: str = "area") -> np.ndarray:
    """
    Binarizes a tag crop, detecting the degenerate case where the first
    (Otsu/adaptive-block) pass produces a single connected component
    covering more than half the crop -- a strong sign the threshold was
    too permissive for a low-contrast photo and fused the whole digit
    row (or more) into one blob. In that case, retries with a finer
    adaptive threshold tuned for stroke-level detail.
    """
    binary = clean_mask(robust_binarize(corrected, polarity=polarity))
    if _largest_component_area_frac(binary) > 0.5:
        block = max(3, (min(corrected.shape[:2]) // 6) | 1)
        denoised = denoise(corrected, 3)
        alt = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, block, 5,
        )
        if polarity == "border":
            alt = normalize_polarity_by_border(alt)
        else:
            alt = normalize_polarity(alt)
        alt = clean_mask(alt, ksize=2)
        if _largest_component_area_frac(alt) < _largest_component_area_frac(binary):
            binary = alt
    return binary


# --------------------------------------------------------------------------- #
# Full localization pipeline
# --------------------------------------------------------------------------- #

def _locate_within_tag(gray: np.ndarray, n_digits: int) -> Optional[List[Tuple[int, int, int, int]]]:
    """
    Stage 1+2+3: find the tag, locate the digit row via component
    clustering (FIX #17.1), and segment it into n_digits digit boxes.
    Returns boxes in ORIGINAL (full-image) coordinates, or None.
    """
    tag_bbox = _find_tag_bbox(gray)
    if tag_bbox is None:
        return None
    tx0, ty0, tx1, ty1 = tag_bbox
    tag_crop = gray[ty0:ty1, tx0:tx1]
    if tag_crop.size == 0:
        return None

    corrected = correct_illumination(tag_crop)

    for polarity_first in ("area", "inverted"):
        if polarity_first == "area":
            binary = _binarize_tag_crop(corrected, polarity="area")
        else:
            binary = cv2.bitwise_not(_binarize_tag_crop(corrected, polarity="area"))
            binary = clean_mask(binary)

        comps = _get_components(binary)
        shape_comps = [c for c in comps if _looks_like_digit(c)]
        shape_comps = _remove_nested_components(shape_comps)

        extent = _find_row_extent(shape_comps, n_digits)
        if extent is None:
            continue
        y0, y1 = extent
        pad_y = max(4, int(0.25 * (y1 - y0)))
        row_y0, row_y1 = max(0, y0 - pad_y), min(binary.shape[0], y1 + pad_y)
        row_binary = binary[row_y0:row_y1, :]

        group = _best_row_group(row_binary, n_digits)
        if group is not None:
            return [(tx0 + c.x, ty0 + row_y0 + c.y, c.w, c.h) for c in group]

        # CC-based grouping couldn't cleanly separate n_digits blobs
        # (e.g. a touching/merged pair, or one component still short) --
        # try the projection-profile valley splitter on this TIGHT crop.
        valley_boxes = _valley_split_row(row_binary, n_digits)
        if valley_boxes is not None:
            return [(tx0 + bx, ty0 + row_y0 + by, bw, bh)
                    for (bx, by, bw, bh) in valley_boxes]

        # Last resort for this row: equal-width slicing of its own
        # ink-column extent (still tightly y-cropped, unlike the old
        # whole-band fallback).
        col_profile = row_binary.sum(axis=0).astype(np.float32) / 255.0
        if col_profile.max() <= 0:
            continue
        col_threshold = col_profile.max() * 0.08
        cols = np.where(col_profile > col_threshold)[0]
        if len(cols) == 0:
            continue
        x0, x1 = int(cols.min()), int(cols.max()) + 1
        if x1 - x0 < 10:
            continue
        sliced = _uniform_slice_boxes((x0, row_y0, x1, row_y1), n_digits)
        return [(tx0 + bx, ty0 + by, bw, bh) for (bx, by, bw, bh) in sliced]

    return None


def _locate_whole_image_fallback(gray: np.ndarray, n_digits: int,
                                  row_tolerance_frac: float = 0.4) -> Optional[List[Tuple[int, int, int, int]]]:
    """
    Previous whole-image dual-polarity method, kept as a fallback for
    photos where tag-boundary detection (Stage 1) fails.
    """
    corrected = correct_illumination(gray)
    binary = clean_mask(robust_binarize(corrected))
    binary_inv = clean_mask(cv2.bitwise_not(binary))

    comps = _get_components(binary) + _get_components(binary_inv)
    comps = [c for c in comps if _looks_like_digit(c)]
    comps = _remove_nested_components(comps)

    regular_candidates: List[Tuple[float, float, List[Component]]] = []
    best_group_fallback = None
    best_score_fallback = float("inf")

    if len(comps) >= n_digits:
        heights = sorted(c.h for c in comps)
        ref_h = heights[len(heights) // 2]
        row_tol = ref_h * row_tolerance_frac

        comps_sorted_by_y = sorted(comps, key=lambda c: c.cy)
        seen_cy = set()
        for base in comps_sorted_by_y:
            key = round(base.cy)
            if key in seen_cy:
                continue
            seen_cy.add(key)
            group = [c for c in comps if abs(c.cy - base.cy) <= row_tol]
            if len(group) < n_digits:
                continue
            trimmed = _best_subset(group, n_digits)
            score = _group_score(trimmed)
            if score < best_score_fallback:
                best_score_fallback = score
                best_group_fallback = trimmed
            if _group_is_regular(trimmed):
                mean_cy = float(np.mean([c.cy for c in trimmed]))
                regular_candidates.append((mean_cy, score, trimmed))

    if regular_candidates:
        regular_candidates.sort(key=lambda item: item[0])
        best_group = regular_candidates[-1][2]
        return [(c.x, c.y, c.w, c.h) for c in best_group]

    if best_group_fallback is not None:
        xs0 = min(c.x for c in best_group_fallback)
        xs1 = max(c.x + c.w for c in best_group_fallback)
        ys0 = min(c.y for c in best_group_fallback)
        ys1 = max(c.y + c.h for c in best_group_fallback)
        pad_x = max(4, int(0.05 * (xs1 - xs0)))
        pad_y = max(4, int(0.15 * (ys1 - ys0)))
        extent = (
            max(0, xs0 - pad_x), max(0, ys0 - pad_y),
            min(binary.shape[1], xs1 + pad_x), min(binary.shape[0], ys1 + pad_y),
        )
        return _uniform_slice_boxes(extent, n_digits)

    return None


def find_digit_row(gray: np.ndarray, n_digits: int = 7,
                    row_tolerance_frac: float = 0.4) -> Optional[List[Tuple[int, int, int, int]]]:
    boxes = _locate_within_tag(gray, n_digits)
    if boxes is not None:
        return boxes
    return _locate_whole_image_fallback(gray, n_digits, row_tolerance_frac=row_tolerance_frac)


def locate_digit_row_multi_orientation(bgr: np.ndarray,
                                        n_digits: int = 7) -> Tuple[np.ndarray, Optional[List[Tuple[int, int, int, int]]]]:
    gray = to_gray(bgr)
    for candidate in try_all_orientations(gray):
        angle = estimate_rotation_angle(robust_binarize(correct_illumination(candidate)))
        leveled = rotate_image(candidate, angle)
        boxes = find_digit_row(leveled, n_digits=n_digits)
        if boxes is not None:
            return leveled, boxes
    return gray, None


# --------------------------------------------------------------------------- #
# Per-digit crop extraction and canvas normalization
# --------------------------------------------------------------------------- #

def extract_digit_crops(gray: np.ndarray, boxes: List[Tuple[int, int, int, int]],
                         pad_frac: float = 0.15) -> List[np.ndarray]:
    crops = []
    h_img, w_img = gray.shape[:2]
    for (x, y, w, h) in boxes:
        pad_x, pad_y = int(w * pad_frac), int(h * pad_frac)
        x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
        x1, y1 = min(w_img, x + w + pad_x), min(h_img, y + h + pad_y)
        crop_gray = gray[y0:y1, x0:x1]
        crops.append(prepare_isolated_crop(crop_gray))
    return crops


def to_classical_canvas(crop: np.ndarray, size: int = 32) -> np.ndarray:
    crop = deskew_digit(crop)
    ys, xs = np.nonzero(crop)
    if len(xs) == 0:
        return np.zeros((size, size), dtype=np.uint8)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    tight = crop[y0:y1, x0:x1]

    h, w = tight.shape
    scale = (size - 4) / max(h, w)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(tight, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((size, size), dtype=np.uint8)
    off_x, off_y = (size - new_w) // 2, (size - new_h) // 2
    canvas[off_y:off_y + new_h, off_x:off_x + new_w] = resized
    return (canvas > 0).astype(np.uint8)


def to_cnn_tensor_array(crop: np.ndarray, size: int = 28) -> np.ndarray:
    canvas_binary = to_classical_canvas(crop, size=size)
    return canvas_binary.astype(np.float32)


# --------------------------------------------------------------------------- #
# VLM path
# --------------------------------------------------------------------------- #

def to_vlm_bytes(bgr: np.ndarray, boxes: Optional[List[Tuple[int, int, int, int]]] = None,
                  pad_frac: float = 0.5, enhance: bool = True) -> bytes:
    out = bgr
    if boxes:
        xs = [b[0] for b in boxes] + [b[0] + b[2] for b in boxes]
        ys = [b[1] for b in boxes] + [b[1] + b[3] for b in boxes]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        w, h = x1 - x0, y1 - y0
        pad_x, pad_y = int(w * pad_frac), int(h * pad_frac)
        h_img, w_img = bgr.shape[:2]
        x0, y0 = max(0, x0 - pad_x), max(0, y0 - pad_y)
        x1, y1 = min(w_img, x1 + pad_x), min(h_img, y1 + pad_y)
        out = bgr[y0:y1, x0:x1]

    if enhance:
        gray = to_gray(out)
        gray = correct_illumination(gray, blur_ksize=41)
        out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if max(out.shape[:2]) < 400:
        scale = 400 / max(out.shape[:2])
        out = cv2.resize(out, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    ok, buf = cv2.imencode(".png", out)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return buf.tobytes()


# --------------------------------------------------------------------------- #
# Convenience wrapper used from main.py
# --------------------------------------------------------------------------- #

def preprocess_seal(path: str, mode: str = "cnn", n_digits: int = 7, img_size: int = 28):
    bgr = load_image(path)
    leveled_gray, boxes = locate_digit_row_multi_orientation(bgr, n_digits=n_digits)

    if mode == "vlm":
        leveled_bgr = cv2.cvtColor(leveled_gray, cv2.COLOR_GRAY2BGR)
        return to_vlm_bytes(leveled_bgr, boxes=boxes, enhance=True)

    if boxes is None:
        return None

    crops = extract_digit_crops(leveled_gray, boxes)
    if mode == "classical":
        return [to_classical_canvas(c, size=img_size) for c in crops]
    elif mode == "cnn":
        return [to_cnn_tensor_array(c, size=img_size) for c in crops]
    else:
        raise ValueError(f"Unknown mode: {mode!r}. Use 'classical', 'cnn', or 'vlm'.")
