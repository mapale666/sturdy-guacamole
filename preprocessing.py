"""
preprocessing.py
=================
Shared preprocessing utilities for the seal numeral recognition task.

See prior revisions' fix history (kept brief here; the important recent
ones are below).

FIX #1-3: training/inference crop preprocessing mismatches (illumination
kernel on tiny crops, area-based vs border-based polarity, global vs
per-crop thresholding). All fixed in CharsDataset (model.py) and
prepare_isolated_crop()/extract_digit_crops() here.

FIX #4-7: find_digit_row() localization. Added width/spacing regularity
scoring across ALL row candidates (not just the first match), and an
aspect-ratio filter so wide/short structural features (molded tag edges,
lettering baselines) aren't mistaken for digits.

FIX #8 (this revision -- single global polarity assumption): find_digit_row()
only ever binarized the photo ONE way (numeral assumed to be the area-
minority across the whole image). That works for a dark-digits-on-light-
tag photo, but fails completely for a light-digits-on-dark-tag photo (e.g.
a black "TESCO" tag with white lettering) -- there, the true digit strokes
are literally the WRONG polarity relative to what the connected-component
search is looking for, so zero valid digit-shaped components are ever
found and the code falls back to the crude last-resort full-image slice.
find_digit_row() now searches components in BOTH the binarized mask and
its inverse, pooling candidates from whichever polarity actually contains
the digit strokes for this particular photo.

FIX #9 (this revision -- phantom component vs. missed real digit): when a
row candidate has slightly MORE digit-shaped components than n_digits
(e.g. a stray dust speck or shadow mark happens to be digit-sized/shaped),
the previous "keep the n_digits closest to median height" heuristic could
keep the phantom and drop a genuine digit whose height happened to differ
slightly (e.g. a "4" with an open top). find_digit_row() now searches
over subsets (small combinatorial search, cheap when only a few
components are extra) and keeps whichever n_digits subset scores most
regular overall, rather than a single greedy per-component heuristic.

Public entry points typically called from main.py / model.py:

load_image(path) -> BGR np.ndarray
locate_digit_row_multi_orientation(bgr) -> (oriented_gray, boxes|None)
extract_digit_crops(gray, boxes, ...) -> list[np.ndarray] (binary, numeral=255, per-crop thresholded)
prepare_isolated_crop(gray) -> binary np.ndarray (for pre-cropped training images AND per-digit inference crops)
to_classical_canvas(crop, size=32) -> binary np.ndarray, numeral=1
to_cnn_tensor_array(crop, size=28) -> float32 np.ndarray in [0,1], shape (size,size)
to_vlm_bytes(bgr, boxes=None, ...) -> PNG bytes (optionally cropped/enhanced, NOT binarized)
preprocess_seal(path, mode, n_digits) -> convenience wrapper, see bottom of file
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
    version of the image, then re-normalizes contrast with CLAHE. Intended
    for full seal photos. DO NOT use this on small, already-cropped
    individual digit images -- see prepare_isolated_crop() instead.
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

    polarity:
    - "area" (default): assumes the numeral is a minority of the total
      image area. Correct for full seal photos on average, but see FIX #8
      -- a single global choice can still be locally wrong for a specific
      sub-region (e.g. a dark tag with light lettering).
    - "border": samples the image border instead. Use for already-cropped,
      tightly-framed single-digit images.
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
    polarity. This is polarity-agnostic per crop, so it correctly handles
    BOTH dark-on-light and light-on-dark digit crops regardless of which
    global polarity find_digit_row() used to locate the box.
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
# Digit-row localization (full seal images)
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


def _get_components(binary: np.ndarray, min_area_frac: float = 0.0005,
                     max_area_frac: float = 0.2) -> List[Component]:
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


def _looks_like_digit(c: Component, min_aspect: float = 0.8, max_aspect: float = 6.0) -> bool:
    """Real digits are reliably taller than wide (h/w roughly 1-6)."""
    if c.w <= 0 or c.h <= 0:
        return False
    aspect = c.h / c.w
    return min_aspect <= aspect <= max_aspect


def _group_score(group: List[Component]) -> float:
    """Lower is more regular (consistent widths, even spacing)."""
    widths = np.array([c.w for c in group], dtype=np.float32)
    if widths.mean() <= 0:
        return float("inf")
    width_cv = float(widths.std() / widths.mean())

    xs = np.sort(np.array([c.x for c in group], dtype=np.float32))
    spacing = np.diff(xs)
    spacing_cv = float(spacing.std() / spacing.mean()) if len(spacing) > 0 and spacing.mean() > 0 else 0.0

    return width_cv + spacing_cv


def _group_is_regular(group: List[Component], tol: float = 0.35) -> bool:
    if len(group) < 2:
        return True
    widths = np.array([c.w for c in group], dtype=np.float32)
    if widths.mean() <= 0:
        return False
    width_cv = float(widths.std() / widths.mean())
    xs = np.sort(np.array([c.x for c in group], dtype=np.float32))
    spacing = np.diff(xs)
    spacing_cv = float(spacing.std() / spacing.mean()) if len(spacing) > 0 and spacing.mean() > 0 else 0.0
    return width_cv <= tol and spacing_cv <= tol


def _best_subset(group: List[Component], n_digits: int, max_search_extra: int = 6) -> List[Component]:
    """
    Picks the n_digits-sized subset of `group` that scores most regular.
    Guards against a spurious extra component (dust speck, shadow) being
    kept over a genuine digit just because it happens to be closer to the
    group's median height -- see FIX #9. Falls back to a height-closeness
    heuristic if there are too many extra candidates for exhaustive search
    to stay cheap.
    """
    if len(group) <= n_digits:
        return group
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


def _row_band_and_extent(binary: np.ndarray, y0: Optional[int] = None,
                          y1: Optional[int] = None) -> Optional[Tuple[int, int, int, int]]:
    """Last-resort localization used only when no digit-like component group is found at all."""
    h, w = binary.shape[:2]

    if y0 is None or y1 is None:
        row_profile = binary.sum(axis=1).astype(np.float32) / 255.0
        if row_profile.max() <= 0:
            return None
        row_threshold = row_profile.max() * 0.15
        rows_with_ink = np.where(row_profile > row_threshold)[0]
        if len(rows_with_ink) == 0:
            return None
        y0, y1 = int(rows_with_ink.min()), int(rows_with_ink.max()) + 1
    else:
        y0, y1 = max(0, int(y0)), min(h, int(y1))

    if y1 <= y0:
        return None

    band = binary[y0:y1, :]
    col_profile = band.sum(axis=0).astype(np.float32) / 255.0
    if col_profile.max() <= 0:
        return None
    col_threshold = col_profile.max() * 0.05
    cols_with_ink = np.where(col_profile > col_threshold)[0]
    if len(cols_with_ink) == 0:
        return None
    x0, x1 = int(cols_with_ink.min()), int(cols_with_ink.max()) + 1

    if x1 - x0 < 5 or y1 - y0 < 5:
        return None
    return x0, y0, x1, y1


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


def find_digit_row(gray: np.ndarray, n_digits: int = 7,
                    row_tolerance_frac: float = 0.4) -> Optional[List[Tuple[int, int, int, int]]]:
    corrected = correct_illumination(gray)
    binary = clean_mask(robust_binarize(corrected))
    binary_inv = clean_mask(cv2.bitwise_not(binary))

    # Pool digit-shaped candidates from BOTH polarities -- the true digit
    # strokes might be the area-majority "foreground" OR "background" of
    # the whole photo depending on whether the tag they sit on is light-
    # on-dark or dark-on-light (see FIX #8).
    comps = _get_components(binary) + _get_components(binary_inv)
    comps = [c for c in comps if _looks_like_digit(c)]

    best_group = None
    best_score = float("inf")
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
            if score < best_score:
                best_score = score
                best_group = trimmed

    if best_group is not None and _group_is_regular(best_group):
        return [(c.x, c.y, c.w, c.h) for c in best_group]

    if best_group is not None:
        xs0 = min(c.x for c in best_group)
        xs1 = max(c.x + c.w for c in best_group)
        ys0 = min(c.y for c in best_group)
        ys1 = max(c.y + c.h for c in best_group)
        pad_x = max(4, int(0.05 * (xs1 - xs0)))
        pad_y = max(4, int(0.15 * (ys1 - ys0)))
        extent = (
            max(0, xs0 - pad_x), max(0, ys0 - pad_y),
            min(binary.shape[1], xs1 + pad_x), min(binary.shape[0], ys1 + pad_y),
        )
    else:
        extent = _row_band_and_extent(binary)

    if extent is None:
        return None
    return _uniform_slice_boxes(extent, n_digits)


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

def preprocess_seal(path: str, mode: str = "cnn", n_digits: int = 7):
    bgr = load_image(path)
    leveled_gray, boxes = locate_digit_row_multi_orientation(bgr, n_digits=n_digits)

    if mode == "vlm":
        leveled_bgr = cv2.cvtColor(leveled_gray, cv2.COLOR_GRAY2BGR)
        return to_vlm_bytes(leveled_bgr, boxes=boxes, enhance=True)

    if boxes is None:
        return None

    crops = extract_digit_crops(leveled_gray, boxes)
    if mode == "classical":
        return [to_classical_canvas(c) for c in crops]
    elif mode == "cnn":
        return [to_cnn_tensor_array(c) for c in crops]
    else:
        raise ValueError(f"Unknown mode: {mode!r}. Use 'classical', 'cnn', or 'vlm'.")
