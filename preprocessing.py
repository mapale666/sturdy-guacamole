"""
preprocessing.py
=================
Shared preprocessing utilities for the seal numeral recognition task.

Designed to back all three recognition approaches (classical CV, CNN,
vision-language model) from a single, testable pipeline. main.py should
only worry about file paths / iterating the input directory; all image
manipulation lives here.

Public entry points you'll typically call from main.py:

    load_image(path)                       -> BGR np.ndarray
    locate_digit_row(bgr, n_digits=7)      -> list[(x, y, w, h)] or None
    extract_digit_crops(bgr, boxes, ...)   -> list[np.ndarray] (grayscale, normalized polarity)
    to_classical_canvas(crop, size=32)     -> binary np.ndarray, numeral=1
    to_cnn_tensor_array(crop, size=28)     -> float32 np.ndarray in [0,1], shape (size,size)
    to_vlm_bytes(bgr, boxes=None, ...)     -> PNG bytes (optionally cropped/enhanced, NOT binarized)
    preprocess_seal(path, mode, n_digits)  -> convenience wrapper, see bottom of file

NOTE: seal codes in this dataset are 7 digits long (e.g. "1584143"),
NOT 6 as an earlier draft assumed from the task doc's example text.
n_digits defaults to 7 everywhere below -- if you ever see codes of a
different length in your data, override n_digits at the call site rather
than editing these defaults blindly.

The test set is intentionally degraded: blur, exposure extremes, polarity
flips (white-on-black vs black-on-white), shifts, rotation, and zoom.
Every function below is written defensively with that in mind rather than
assuming clean, canonical inputs.
"""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple


DEFAULT_N_DIGITS = 7


# --------------------------------------------------------------------------- #
# Basic I/O
# --------------------------------------------------------------------------- #

def load_image(path: str) -> np.ndarray:
    """Load an image as BGR uint8. Raises if the file can't be read."""
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return im


def to_gray(bgr: np.ndarray) -> np.ndarray:
    if bgr.ndim == 2:
        return bgr
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


# --------------------------------------------------------------------------- #
# Illumination / exposure correction
# --------------------------------------------------------------------------- #

def correct_illumination(gray: np.ndarray, blur_ksize: int = 51) -> np.ndarray:
    """
    Flattens vignetting / uneven exposure by dividing out a heavily blurred
    version of the image (an estimate of the local background), then
    re-normalizes contrast with CLAHE. Robust to both over- and under-exposed
    inputs.
    """
    ksize = blur_ksize | 1  # must be odd
    background = cv2.GaussianBlur(gray, (ksize, ksize), 0)
    background = np.where(background == 0, 1, background)  # avoid div-by-zero
    normalized = (gray.astype(np.float32) / background.astype(np.float32)) * 128.0
    normalized = np.clip(normalized, 0, 255).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    return clahe.apply(normalized)


def denoise(gray: np.ndarray, ksize: int = 3) -> np.ndarray:
    ksize = ksize | 1
    return cv2.GaussianBlur(gray, (ksize, ksize), 0)


# --------------------------------------------------------------------------- #
# Binarization with polarity normalization
# --------------------------------------------------------------------------- #

def robust_binarize(gray: np.ndarray) -> np.ndarray:
    """
    Produces a binary image with numeral pixels = 255, background = 0,
    regardless of whether the source was originally light-on-dark or
    dark-on-light, and regardless of global exposure.

    Strategy: try Otsu first (fast, works on well-behaved histograms), and
    fall back to adaptive thresholding if the result looks degenerate
    (e.g., almost all foreground or almost all background, which happens
    on badly exposed or low-contrast crops).
    """
    blurred = denoise(gray, 3)
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    fg_ratio = np.count_nonzero(otsu) / otsu.size
    if fg_ratio < 0.02 or fg_ratio > 0.60:
        block = max(15, (min(blurred.shape) // 8) | 1)
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, block, 5,
        )
    else:
        binary = otsu

    binary = normalize_polarity(binary)
    return binary


def normalize_polarity(binary: np.ndarray) -> np.ndarray:
    """
    Forces numeral = foreground = 255. Assumes the numeral occupies less
    area than the background, so if the "white" class is the majority, invert.
    """
    fg_ratio = np.count_nonzero(binary) / binary.size
    if fg_ratio > 0.5:
        binary = cv2.bitwise_not(binary)
    return binary


# --------------------------------------------------------------------------- #
# Morphological cleanup
# --------------------------------------------------------------------------- #

def clean_mask(binary: np.ndarray, ksize: int = 3) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    return closed


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
    """
    Returns the image rotated at 0/90/180/270 degrees. Use this when the
    seal might genuinely be upside-down or sideways rather than merely
    tilted a few degrees.
    """
    return [
        gray,
        cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE),
        cv2.rotate(gray, cv2.ROTATE_180),
        cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE),
    ]


# --------------------------------------------------------------------------- #
# Digit-row localization (classical pipeline / shared panel finder)
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


def find_digit_row(gray: np.ndarray, n_digits: int = DEFAULT_N_DIGITS,
                    row_tolerance_frac: float = 0.4) -> Optional[List[Tuple[int, int, int, int]]]:
    """
    Locates n_digits components that lie approximately on the same
    horizontal line and returns their bounding boxes sorted left-to-right.
    Area thresholds are expressed as a fraction of total image area so the
    search remains valid whether the test image is zoomed in or out.
    Returns None if no plausible row is found.
    """
    corrected = correct_illumination(gray)
    binary = robust_binarize(corrected)
    binary = clean_mask(binary)

    comps = _get_components(binary)
    if len(comps) < n_digits:
        return None

    heights = sorted(c.h for c in comps)
    ref_h = heights[len(heights) // 2]
    row_tol = ref_h * row_tolerance_frac

    comps_sorted_by_y = sorted(comps, key=lambda c: c.cy)
    best_group = None
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

    if best_group is None:
        return None

    if len(best_group) > n_digits:
        best_group = sorted(best_group, key=lambda c: c.area, reverse=True)[:n_digits]
        best_group.sort(key=lambda c: c.x)

    return [(c.x, c.y, c.w, c.h) for c in best_group]


def locate_digit_row_multi_orientation(bgr: np.ndarray,
                                        n_digits: int = DEFAULT_N_DIGITS
                                        ) -> Tuple[np.ndarray, Optional[List[Tuple[int, int, int, int]]]]:
    """
    Handles the "turned around" case: tries 0/90/180/270 rotations and,
    within the best candidate, also applies fine-angle deskewing before
    localization. Returns (oriented_gray_image, boxes_or_None).
    """
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
    corrected = correct_illumination(gray)
    binary = robust_binarize(corrected)
    binary = clean_mask(binary)

    crops = []
    h_img, w_img = binary.shape
    for (x, y, w, h) in boxes:
        pad_x, pad_y = int(w * pad_frac), int(h * pad_frac)
        x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
        x1, y1 = min(w_img, x + w + pad_x), min(h_img, y + h + pad_y)
        crops.append(binary[y0:y1, x0:x1])
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
# VLM path: keep the image photographic, just crop/enhance, don't binarize
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

def preprocess_seal(path: str, mode: str = "cnn", n_digits: int = DEFAULT_N_DIGITS):
    """
    One-call entry point for main.py.

    mode="classical" -> returns list[np.ndarray] canvases (size x size, 0/1)
    mode="cnn"        -> returns list[np.ndarray] float32 arrays (size x size, [0,1])
    mode="vlm"        -> returns PNG bytes of the cropped/enhanced seal region
    Returns None (classical/cnn) if no digit row could be located.
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
        return [to_classical_canvas(c) for c in crops]
    elif mode == "cnn":
        return [to_cnn_tensor_array(c) for c in crops]
    else:
        raise ValueError(f"Unknown mode: {mode!r}. Use 'classical', 'cnn', or 'vlm'.")
