"""
preprocessing.py
=================
Shared preprocessing utilities for the seal numeral recognition task.

Two distinct normalization paths are provided, and using the wrong one is
a common source of silent failure:

- correct_illumination() / robust_binarize() / clean_mask() -- designed
  for FULL seal photos, where there's real background/vignetting context
  for a large-kernel blur to estimate. Used by find_digit_row() to LOCATE
  the digit row (connected-component analysis needs one consistent binary
  mask of the whole photo). It is NOT used to produce the crops that are
  fed to the classifier -- see extract_digit_crops() below.

- prepare_isolated_crop() -- designed for ALREADY-CROPPED individual
  digit images (e.g. ground_truth_chars_balanced/100016_0.tif). These
  crops are small and tightly framed around a single digit, so running
  the full-image illumination-correction pipeline on them (with a 51px
  background-blur kernel) can be larger than the crop itself, flattening
  the digit into near-uniform gray and destroying the shape before
  thresholding ever sees it. Use this path for training data, AND for
  every individual digit crop pulled out of a full seal photo at
  inference time (see extract_digit_crops()) -- both need the exact same
  per-crop, locally-thresholded treatment to stay in the same visual
  domain the CNN was trained on.

FIX #1 (illumination kernel): CharsDataset in model.py used to call
correct_illumination()+robust_binarize()+clean_mask() on the tiny,
tightly-cropped training images -- the full-photo pipeline, which
flattens tiny crops. Training now uses prepare_isolated_crop() instead.

FIX #2 (polarity): robust_binarize()'s default polarity normalization
(normalize_polarity()) assumes the numeral is a minority of the image's
area (<50%). That's safe for full seal photos, but breaks for tightly-
framed single-digit crops where a bold digit can cover more than half
the crop. normalize_polarity_by_border() fixes this by sampling the
crop's border (reliably background even in the tightest crop) instead.
prepare_isolated_crop() uses this via polarity="border".

FIX #3 (per-crop vs. global thresholding): extract_digit_crops() used to
binarize the ENTIRE seal photo ONCE with a single global Otsu threshold,
then just slice sub-rectangles out of that one global binary mask. That
produced crops with systematically different-looking strokes than the
per-crop, locally-thresholded training images. extract_digit_crops() now
slices the RAW GRAYSCALE region for each box and binarizes each digit
crop individually with prepare_isolated_crop(), matching training exactly.

FIX #4 (unreliable component-based localization): find_digit_row() picked
whichever n_digits connected components had matching row height (cy),
with NO check on whether their widths/spacing actually look like n_digits
separate, evenly-spaced characters. Fragmented/merged strokes could pass
that check anyway. find_digit_row() now validates the selected group's
width/spacing regularity (_group_is_regular) and falls back to profile-
based row slicing when it fails.

FIX #5 (fallback used the wrong y-band): the first fallback version
searched for the row's vertical position by scanning ink density across
the *entire* photo height, which picked up unrelated ink elsewhere and
sliced the whole photo. Fixed by reusing the (even horizontally-
irregular) component group's y-range as the row-band hint.

FIX #6 (this revision -- x-extent still leaked to the full image width):
even restricted to the correct narrow y-band, a 5%-of-max column-density
threshold was still too permissive -- something spans near the full
width within that band (reflections/embossing texture/a border line),
so the "column profile" fallback kept picking x0=0, x1=full_width instead
of the true digit region. Fix: stop re-deriving the x-extent from a pixel
density profile at all when a component group already exists. Even
though the group's *internal* 7-way split was wrong (fragmented/merged
digits), its overall bounding box (leftmost to rightmost detected
component) is still a reliable signal, because spurious fragmentation/
merging happens *inside* the digit row, not outside it. The pixel-
density-profile approach (_row_band_and_extent) is now only used as a
last resort when NO component group was found at all.

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
    version of the image (an estimate of the local background), then
    re-normalizes contrast with CLAHE. Intended for full seal photos where
    the blur kernel is small relative to the image. DO NOT use this on
    small, already-cropped individual digit images -- see
    prepare_isolated_crop() instead, which caps the kernel to the crop size.
    """
    h, w = gray.shape[:2]
    ksize = min(blur_ksize, max(3, min(h, w) - 1))
    ksize = ksize | 1  # must be odd
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
    Produces a binary image with numeral pixels = 255, background = 0,
    regardless of source polarity or exposure. Tries Otsu first, falls
    back to adaptive thresholding if the result looks degenerate.

    polarity controls how foreground vs. background is decided:
    - "area" (default): normalize_polarity() -- assumes the numeral is a
      minority of the total image area. Correct for full seal photos
      (find_digit_row), where there's plenty of background margin around
      the digit row.
    - "border": normalize_polarity_by_border() -- samples the image border
      instead. Use this for already-cropped, tightly-framed single-digit
      images (prepare_isolated_crop), where a bold digit can cover more
      than half the crop's area and break the "area" assumption.
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
    of total pixels. Even in a tightly-framed digit crop, the outermost
    border strip reliably belongs to the background -- unlike total area,
    which a bold/thick digit can easily exceed 50% of. Use this for
    prepare_isolated_crop(); use normalize_polarity() for full seal photos.
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
        # Border is mostly white -> white is background -> flip so numeral is 255.
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
    correction, using its own LOCAL Otsu threshold. Use this for
    ground_truth_chars_balanced training images AND for each individual
    digit crop sliced out of a full seal photo at inference time -- NOT
    correct_illumination()+robust_binarize()+clean_mask() on the whole
    photo, since (a) that combo's background-blur kernel is tuned for
    whole seal photos and can exceed the size of a tight digit crop,
    flattening the digit before thresholding ever runs, and (b) a single
    threshold computed from the WHOLE photo produces different-looking
    strokes than one computed locally per digit, creating a domain shift
    versus training.

    Uses border-based polarity normalization (polarity="border") rather
    than the area-based default, because a bold digit can fill more than
    half of a tightly-framed crop and break the "numeral is a minority of
    the area" assumption that normalize_polarity() relies on.
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


def _group_is_regular(group: List[Component], tol: float = 0.35) -> bool:
    """
    Sanity-checks a candidate group of n_digits components: real digit
    characters in this fixed-width seal font should have similar widths
    and roughly even horizontal spacing. Rejects groups where widths vary
    wildly (merged/split strokes) or spacing is wildly uneven (missed or
    duplicated digits), even though the group technically satisfied the
    row-height/count criteria.
    """
    if len(group) < 2:
        return True
    widths = np.array([c.w for c in group], dtype=np.float32)
    if widths.mean() <= 0:
        return False
    width_cv = float(widths.std() / widths.mean())

    xs = np.sort(np.array([c.x for c in group], dtype=np.float32))
    spacing = np.diff(xs)
    if len(spacing) == 0 or spacing.mean() <= 0:
        spacing_cv = 0.0
    else:
        spacing_cv = float(spacing.std() / spacing.mean())

    return width_cv <= tol and spacing_cv <= tol


def _row_band_and_extent(binary: np.ndarray, y0: Optional[int] = None,
                          y1: Optional[int] = None) -> Optional[Tuple[int, int, int, int]]:
    """
    Last-resort localization used only when NO component group was found
    at all: finds a row band via a row ink-density profile across the
    whole image, then its horizontal extent via a column ink-density
    profile within that band. (When a component group DOES exist, even an
    irregular one, find_digit_row() uses its bounding box directly instead
    of this -- see FIX #6 above -- because pixel-density profiles can be
    thrown off by reflections/texture/border lines spanning much of the
    photo's width.)
    Returns (x0, y0, x1, y1), or None.
    """
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
    """Splits a (x0, y0, x1, y1) row extent into n_digits equal-width boxes."""
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
    binary = robust_binarize(corrected)
    binary = clean_mask(binary)

    comps = _get_components(binary)
    best_group = None
    if len(comps) >= n_digits:
        heights = sorted(c.h for c in comps)
        ref_h = heights[len(heights) // 2]
        row_tol = ref_h * row_tolerance_frac

        comps_sorted_by_y = sorted(comps, key=lambda c: c.cy)
        for base in comps_sorted_by_y:
            group = [c for c in comps if abs(c.cy - base.cy) <= row_tol]
            if len(group) >= n_digits:
                group.sort(key=lambda c: c.x)
                group = sorted(group, key=lambda c: abs(c.h - ref_h))[:max(n_digits, len(group))]
                group.sort(key=lambda c: c.x)
                if best_group is None or len(group) < len(best_group) or len(group) == n_digits:
                    best_group = group
                if len(group) == n_digits:
                    break

        if best_group is not None and len(best_group) > n_digits:
            best_group = sorted(best_group, key=lambda c: c.area, reverse=True)[:n_digits]
            best_group.sort(key=lambda c: c.x)

    if best_group is not None and _group_is_regular(best_group):
        return [(c.x, c.y, c.w, c.h) for c in best_group]

    if best_group is not None:
        # The group's internal 7-way split was unreliable (fragmented or
        # merged digit strokes), but its overall bounding box is still a
        # good estimate of the true digit row's extent -- fragmentation/
        # merging happens INSIDE the row, not outside it. Re-slice that
        # bounding box evenly instead of trusting the individual boxes.
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
        # No component group at all -- last resort, scan the whole image.
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
    ground_truth_chars_balanced training images. `gray` here is the leveled
    (rotation-corrected) grayscale image returned by
    locate_digit_row_multi_orientation(), i.e. NOT yet binarized.
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
