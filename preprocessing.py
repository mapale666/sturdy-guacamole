"""
preprocessing.py
=================
Shared preprocessing utilities for the seal numeral recognition task.

This is the BEST-KNOWN-CONFIRMED configuration (0.5644 exact / 0.7430
per-digit on the full validation set) plus two additive changes: (a)
preprocess_seal() accepts an img_size parameter so model.py's IMG_SIZE
stays in sync between training and inference, and (b) FIX #12 below,
addressing two confirmed localization failure modes found via direct
visual inspection of box overlays. Do not re-attempt disabling
clean_mask()'s opening step or adding despeckle/median-blur logic without
a controlled --sample-size A/B first; three separate attempts at that all
regressed full-pipeline accuracy despite looking reasonable in isolation.

Fix history (FIX #1-9): see prior revisions -- illumination-kernel
mismatch, border-based polarity for isolated crops, per-crop vs. global
thresholding, component-group regularity scoring, fallback extent
estimation, aspect-ratio filtering, dual-polarity component search, and
combinatorial best-subset selection.

FIX #12 (this revision -- confirmed via visual inspection of box overlays
on 3 real error cases):

(a) Letters vs. digits: on tags with a brand name printed above the
numeric code (e.g. "TESCO" above "1583404"), find_digit_row() could lock
onto the BRAND LETTERING instead of the digit row, since tall uppercase
letters pass the same aspect-ratio/regularity checks as digits. Every
observed example has the numeric code positioned BELOW any brand
lettering, so find_digit_row() now collects ALL row candidates that pass
the regularity check (not just the single best-scoring one) and, among
those, prefers whichever sits LOWEST in the image (largest mean y).

(b) Screw/bolt components: a screw head's circular threading or knurling
pattern can fragment into components that are near-circular (aspect
ratio close to 1.0, which the old min_aspect=0.8 threshold let through)
or thin arcs. This produced a completely wrong localization (boxing the
screw assembly instead of the tag) that collapsed classification to a
single repeated digit on at least one persistent error case. Two changes
to _looks_like_digit(): min_aspect tightened from 0.8 to 1.15 (real
digits in this dataset are consistently noticeably taller than wide,
unlike a circular screw cap), and a new fill-ratio check (ink area /
bounding-box area) rejects thin/hollow shapes like rings or arcs, which a
solid printed digit stroke would never produce.

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
      image area. Correct for full seal photos on average.
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
    """Forces numeral = foreground = 255, assuming numeral occupies less area."""
    fg_ratio = np.count_nonzero(binary) / binary.size
    if fg_ratio > 0.5:
        binary = cv2.bitwise_not(binary)
    return binary


def normalize_polarity_by_border(binary: np.ndarray, border_frac: float = 0.08) -> np.ndarray:
    """
    Forces numeral = foreground = 255 by sampling the image BORDER as the
    background estimate, instead of assuming the foreground is a minority
    of total pixels.
    """
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
    polarity. Use for ground_truth_chars_balanced training images AND for
    each individual digit crop sliced out of a full seal photo at
    inference time.
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


def _looks_like_digit(c: Component, min_aspect: float = 1.15, max_aspect: float = 6.0,
                       min_fill_ratio: float = 0.15) -> bool:
    """
    Real digits in this dataset are consistently noticeably taller than
    wide (h/w >= ~1.15) and are solid printed/embossed strokes with a
    moderate ink-fill ratio within their bounding box. This rejects two
    confirmed false-positive sources: near-circular screw-head/knurling
    fragments (aspect ratio near 1.0, which a looser 0.8 threshold let
    through) and thin arcs/rings from screw threading (very low fill
    ratio, since a thin ring occupies little area relative to its
    bounding square) -- both were observed causing find_digit_row() to
    box a screw assembly instead of the actual digit row.
    """
    if c.w <= 0 or c.h <= 0:
        return False
    aspect = c.h / c.w
    if not (min_aspect <= aspect <= max_aspect):
        return False
    fill_ratio = c.area / float(c.w * c.h)
    return fill_ratio >= min_fill_ratio


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
    kept over a genuine digit. Falls back to a height-closeness heuristic
    if there are too many extra candidates for exhaustive search.
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

    comps = _get_components(binary) + _get_components(binary_inv)
    comps = [c for c in comps if _looks_like_digit(c)]

    # Collect EVERY row candidate that passes the regularity check (not
    # just the single best-scoring one), so we can choose among them by
    # position -- see FIX #12a. best_group_fallback/best_score_fallback
    # track the best-scoring candidate regardless of regularity, used
    # only if nothing passes the regularity check at all.
    regular_candidates: List[Tuple[float, float, List[Component]]] = []  # (mean_cy, score, group)
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
        # Prefer the row positioned LOWEST in the image (largest mean cy):
        # the numeric seal code consistently sits below any brand
        # lettering / other text on these tags.
        regular_candidates.sort(key=lambda item: item[0])
        best_group = regular_candidates[-1][2]
        return [(c.x, c.y, c.w, c.h) for c in best_group]

    best_group = best_group_fallback
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
    """
    Slices each digit's box out of the RAW GRAYSCALE full seal image (with
    padding) and binarizes each crop INDIVIDUALLY via prepare_isolated_crop()
    -- the same locally-thresholded, border-polarity path used for the
    ground_truth_chars_balanced training images.
    """
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
    """
    img_size controls the final canvas size for mode="classical"/"cnn"
    (passed to to_classical_canvas()/to_cnn_tensor_array()). Callers that
    train/load a CNN with a non-default IMG_SIZE (see model.py) should
    pass it explicitly here so inference matches training exactly.
    """
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
