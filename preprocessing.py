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


def _iou(a: Component, b: Component) -> float:
    ax0, ay0, ax1, ay1 = a.x, a.y, a.x + a.w, a.y + a.h
    bx0, by0, bx1, by1 = b.x, b.y, b.x + b.w, b.y + b.h
    left, top = max(ax0, bx0), max(ay0, by0)
    right, bottom = min(ax1, bx1), min(ay1, by1)
    if right <= left or bottom <= top:
        return 0.0
    inter = (right - left) * (bottom - top)
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union > 0 else 0.0


def _dedupe_cross_polarity(comps: List[Component], overlap_thresh: float = 0.35) -> List[Component]:

    def fill_ratio(c: Component) -> float:
        return c.area / float(c.w * c.h) if c.w > 0 and c.h > 0 else 0.0

    comps_sorted = sorted(comps, key=fill_ratio, reverse=True)
    kept: List[Component] = []
    for c in comps_sorted:
        if any(_iou(c, k) >= overlap_thresh for k in kept):
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
    # Early-return branch sorts by x -- connected-component labeling
    # order is a raster scan, not left-to-right reading order, and
    # downstream code assumes boxes[i] is the i-th digit from the left.
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


def _reinforce_weak_group_members(group: List[Component], fill_ratio_drop: float = 0.5,
                                   size_drop: float = 0.55) -> List[Component]:

    if len(group) < 2:
        return group
    group_sorted = sorted(group, key=lambda c: c.x)
    n = len(group_sorted)
    fills = [c.area / float(c.w * c.h) if c.w > 0 and c.h > 0 else 0.0 for c in group_sorted]
    med_fill = float(np.median(fills)) if fills else 0.0
    med_w = float(np.median([c.w for c in group_sorted]))
    med_h = float(np.median([c.h for c in group_sorted]))
    med_y = float(np.median([c.y for c in group_sorted]))

    weak = [
        (c.w < med_w * size_drop or c.h < med_h * size_drop or fills[i] < med_fill * fill_ratio_drop)
        for i, c in enumerate(group_sorted)
    ]

    good_idx = [i for i in range(n) if not weak[i]]
    pitch = med_w * 1.3
    if len(good_idx) >= 2:
        pitches = [
            (group_sorted[b].x - group_sorted[a].x) / (b - a)
            for a, b in zip(good_idx[:-1], good_idx[1:])
        ]
        pitch = float(np.median(pitches))

    result = list(group_sorted)
    for i in range(n):
        if not weak[i]:
            continue
        c = result[i]
        if i > 0 and not weak[i - 1]:
            anchor_x = result[i - 1].x + pitch
        elif i < n - 1 and not weak[i + 1]:
            anchor_x = result[i + 1].x - pitch
        else:
            anchor_x = c.x + c.w / 2.0 - med_w / 2.0  # no good neighbor -- fall back to recentering
        new_x = int(round(anchor_x))
        new_y = int(round(med_y))
        result[i] = Component(new_x, new_y, int(med_w), int(med_h),
                               int(med_w * med_h * 0.4), new_x + med_w / 2.0, new_y + med_h / 2.0)
    return result

def _has_weak_member(group: List[Component], size_drop: float = 0.55,
                      fill_ratio_drop: float = 0.5) -> bool:
    if len(group) < 2:
        return False
    fills = [c.area / float(c.w * c.h) if c.w > 0 and c.h > 0 else 0.0 for c in group]
    med_fill = float(np.median(fills))
    med_w = float(np.median([c.w for c in group]))
    med_h = float(np.median([c.h for c in group]))
    for i, c in enumerate(group):
        if c.w < med_w * size_drop or c.h < med_h * size_drop or fills[i] < med_fill * fill_ratio_drop:
            return True
    return False


def _best_row_group(comps, n_digits, row_tolerance_frac=0.4, prefer_lowest=True,
                     tag_height=None, min_y_frac=0.60, allow_irregular_fallback=True,
                     reject_weak=False):

    if len(comps) < n_digits:
        return None

    heights = sorted(c.h for c in comps)
    ref_h = heights[len(heights) // 2]
    row_tol = ref_h * row_tolerance_frac

    comps_sorted_by_y = sorted(comps, key=lambda c: c.cy)
    regular_candidates: List[Tuple[float, List[Component]]] = []
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
        mean_cy = float(np.mean([c.cy for c in trimmed]))
        if tag_height and mean_cy < min_y_frac * tag_height:
            continue
        score = _group_score(trimmed)
        if score < best_score:
            best_score = score
            best_group = trimmed
        if _group_is_regular(trimmed):
            if not (reject_weak and _has_weak_member(trimmed)):
                regular_candidates.append((mean_cy, trimmed))

    if regular_candidates:
        regular_candidates.sort(key=lambda item: item[0])
        chosen = regular_candidates[-1][1] if prefer_lowest else regular_candidates[0][1]
        return _reinforce_weak_group_members(chosen)
    if allow_irregular_fallback and best_group is not None:
        return _reinforce_weak_group_members(best_group)
    return None


# --------------------------------------------------------------------------- #
# Gap interpolation for partial rows (Stage 3, secondary method, tried
# before falling back to noisy whole-image heuristics)
# --------------------------------------------------------------------------- #

def _cluster_candidates_by_row(comps: List[Component], min_count: int,
                                row_tolerance_frac: float = 0.4,
                                tag_height: Optional[int] = None,
                                min_y_frac: float = 0.60) -> List[List[Component]]:
    if len(comps) < min_count:
        return []
    heights = sorted(c.h for c in comps)
    ref_h = heights[len(heights) // 2]
    row_tol = ref_h * row_tolerance_frac
    sorted_by_y = sorted(comps, key=lambda c: c.cy)
    clusters: List[List[Component]] = []
    seen = set()
    for base in sorted_by_y:
        key = round(base.cy)
        if key in seen:
            continue
        seen.add(key)
        group = [c for c in comps if abs(c.cy - base.cy) <= row_tol]
        if len(group) < min_count:
            continue
        mean_cy = float(np.mean([c.cy for c in group]))
        if tag_height and mean_cy < min_y_frac * tag_height:
            continue
        clusters.append(group)
    return clusters


def _ink_density_in_window(binary_masks: Tuple[np.ndarray, np.ndarray],
                            x0: float, x1: float, y0: float, y1: float) -> float:

    best = 0.0
    for binary in binary_masks:
        x0c, x1c = max(0, int(x0)), min(binary.shape[1], int(x1))
        y0c, y1c = max(0, int(y0)), min(binary.shape[0], int(y1))
        if x1c <= x0c or y1c <= y0c:
            continue
        region = binary[y0c:y1c, x0c:x1c]
        if region.size == 0:
            continue
        best = max(best, float(np.count_nonzero(region)) / region.size)
    return best


def _try_interpolated_group(group: List[Component], n_digits: int,
                             binary_masks: Optional[Tuple[np.ndarray, np.ndarray]] = None
                             ) -> Optional[List[Component]]:

    group_sorted = sorted(group, key=lambda c: c.x)
    if len(group_sorted) < max(3, n_digits - 2):
        return None
    xs = np.array([c.x for c in group_sorted], dtype=np.float32)
    diffs = np.diff(xs)
    if len(diffs) == 0 or np.any(diffs <= 0):
        return None
    med_spacing = float(np.median(diffs))
    if med_spacing <= 0:
        return None
    multiples = np.round(diffs / med_spacing)
    if np.any(multiples < 1):
        return None
    residual = np.abs(diffs - multiples * med_spacing) / med_spacing
    if np.max(residual) > 0.35:
        return None  # not a clean arithmetic progression -- unsafe to interpolate

    med_w = float(np.median([c.w for c in group_sorted]))
    med_h = float(np.median([c.h for c in group_sorted]))
    med_y = float(np.median([c.y for c in group_sorted]))

    result = [group_sorted[0]]
    for i in range(1, len(group_sorted)):
        n_slots = int(round((group_sorted[i].x - result[-1].x) / med_spacing))
        for _ in range(1, n_slots):
            ix = int(round(result[-1].x + med_spacing))
            result.append(Component(ix, int(round(med_y)), int(med_w), int(med_h),
                                     int(med_w * med_h * 0.4), ix + med_w / 2.0, med_y + med_h / 2.0))
        result.append(group_sorted[i])

    # Any remaining shortfall means the missing digit is at one of the
    # two ENDS, not internal -- decide which side by checking actual ink
    # density in the extension window instead of always guessing right.
    while len(result) < n_digits:
        extend_right_x = result[-1].x + med_spacing
        extend_left_x = result[0].x - med_spacing
        choose_right = True
        if binary_masks is not None and extend_left_x >= 0:
            right_density = _ink_density_in_window(
                binary_masks, extend_right_x, extend_right_x + med_w, med_y, med_y + med_h)
            left_density = _ink_density_in_window(
                binary_masks, extend_left_x, extend_left_x + med_w, med_y, med_y + med_h)
            if left_density > right_density * 1.15:
                choose_right = False
        elif extend_left_x < 0:
            choose_right = True

        if choose_right:
            ix = int(round(extend_right_x))
            result.append(Component(ix, int(round(med_y)), int(med_w), int(med_h),
                                     int(med_w * med_h * 0.4), ix + med_w / 2.0, med_y + med_h / 2.0))
        else:
            ix = int(round(extend_left_x))
            result.insert(0, Component(ix, int(round(med_y)), int(med_w), int(med_h),
                                        int(med_w * med_h * 0.4), ix + med_w / 2.0, med_y + med_h / 2.0))

    return result if len(result) == n_digits else None


# --------------------------------------------------------------------------- #
# Cross-polarity component pooling (Stage 3, primary method)
# --------------------------------------------------------------------------- #

def _collect_tag_components(tag_crop: np.ndarray) -> Tuple[List[Component], np.ndarray, np.ndarray]:

    corrected = correct_illumination(tag_crop)
    binary_normal = clean_mask(robust_binarize(corrected, polarity="area"))
    binary_inverted = clean_mask(cv2.bitwise_not(robust_binarize(corrected, polarity="area")))

    pool: List[Component] = []
    for binary in (binary_normal, binary_inverted):
        comps = _get_components(binary)
        comps = [c for c in comps if _looks_like_digit(c)]
        comps = _remove_nested_components(comps)
        pool.extend(comps)
    return _dedupe_cross_polarity(pool), binary_normal, binary_inverted


# --------------------------------------------------------------------------- #
# Projection-profile valley splitter (Stage 3, tertiary fallback). Used
# when even the merged cross-polarity pool can't cleanly separate
# n_digits blobs -- typically because two digits are touching/merged.
# --------------------------------------------------------------------------- #

def _valley_split_row(row_binary: np.ndarray, n_digits: int) -> Optional[List[Tuple[int, int, int, int]]]:

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
    # Bridge small gaps in a faint/blurry embossed border before dilating.
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
        # A bounding box covering most of the whole photo is the outer
        # frame/vignette, never the actual raised tag plaque -- reject
        # by AREA ratio (robust to aspect skew), not per-dimension.
        if (cw * ch) >= 0.65 * img_area:
            continue
        rect_area = cw * ch
        if rect_area <= 0:
            continue
        fill = area / rect_area
        if fill < 0.35:
            continue
        aspect = ch / cw if cw > 0 else 0
        if not (0.4 <= aspect <= 3.0):
            continue
        if area > best_area:
            best_area = area
            best_bbox = (x, y, x + cw, y + ch)

    return best_bbox


def _find_tag_bbox(gray: np.ndarray) -> Optional[Tuple[int, int, int, int]]:

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
# STAGE 2: text row-band detection within the tag crop (fallback path only
# -- the primary path is the cross-polarity component pool in Stage 3)
# --------------------------------------------------------------------------- #

def _find_text_row_bands(binary: np.ndarray, min_frac: float = 0.25,
                          min_band_height_frac: float = 0.04,
                          gap_tol_frac: float = 0.008) -> List[Tuple[int, int]]:

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


def _row_is_plausible(row_y0: int, row_y1: int, tag_height: int, max_frac: float = 0.45) -> bool:

    if tag_height <= 0:
        return False
    return (row_y1 - row_y0) <= tag_height * max_frac


# --------------------------------------------------------------------------- #
# Full localization pipeline
# --------------------------------------------------------------------------- #

def _locate_within_tag(gray: np.ndarray, n_digits: int) -> Optional[List[Tuple[int, int, int, int]]]:

    tag_bbox = _find_tag_bbox(gray)
    if tag_bbox is None:
        return None
    tx0, ty0, tx1, ty1 = tag_bbox
    img_h, img_w = gray.shape[:2]
    pad_x = max(2, int(0.06 * (tx1 - tx0)))
    pad_y = max(2, int(0.06 * (ty1 - ty0)))
    tx0, ty0 = max(0, tx0 - pad_x), max(0, ty0 - pad_y)
    tx1, ty1 = min(img_w, tx1 + pad_x), min(img_h, ty1 + pad_y)
    tag_crop = gray[ty0:ty1, tx0:tx1]
    if tag_crop.size == 0:
        return None
    tag_height = tag_crop.shape[0]

    pool, binary_normal, binary_inverted = _collect_tag_components(tag_crop)
    group = _best_row_group(pool, n_digits, prefer_lowest=True, tag_height=tag_height,
                         allow_irregular_fallback=False)
    if group is not None:
        return [(tx0 + c.x, ty0 + c.y, c.w, c.h) for c in group]

    binary_masks = (binary_normal, binary_inverted)
    for min_count in (n_digits - 1, n_digits - 2):
        clusters = _cluster_candidates_by_row(pool, min_count, tag_height=tag_height)
        interpolated_candidates: List[Tuple[float, List[Component]]] = []
        for cl in clusters:
            filled = _try_interpolated_group(cl, n_digits, binary_masks=binary_masks)
            if filled is not None:
                mean_cy = float(np.mean([c.cy for c in cl]))
                interpolated_candidates.append((mean_cy, filled))
        if interpolated_candidates:
            interpolated_candidates.sort(key=lambda item: item[0])
            chosen = interpolated_candidates[-1][1]
            return [(tx0 + c.x, ty0 + c.y, c.w, c.h) for c in chosen]

    # --- Tertiary fallback: single-polarity band + valley split ---
    corrected = correct_illumination(tag_crop)
    for polarity_first in ("area", "inverted"):
        if polarity_first == "area":
            binary = clean_mask(robust_binarize(corrected, polarity="area"))
        else:
            binary = clean_mask(cv2.bitwise_not(robust_binarize(corrected, polarity="area")))

        bands = _find_text_row_bands(binary)
        if not bands:
            continue

        for y0, y1 in reversed(bands):
            pad_y2 = max(3, int(0.1 * (y1 - y0)))
            row_y0, row_y1 = max(0, y0 - pad_y2), min(binary.shape[0], y1 + pad_y2)
            if not _row_is_plausible(row_y0, row_y1, tag_height):
                continue
            row_binary = binary[row_y0:row_y1, :]

            row_comps = [c for c in _get_components(row_binary) if _looks_like_digit(c)]
            row_comps = _remove_nested_components(row_comps)
            row_group = _best_row_group(row_comps, n_digits, prefer_lowest=True)
            if row_group is not None:
                return [(tx0 + c.x, ty0 + row_y0 + c.y, c.w, c.h) for c in row_group]

            valley_boxes = _valley_split_row(row_binary, n_digits)
            if valley_boxes is not None:
                return [(tx0 + bx, ty0 + row_y0 + by, bw, bh)
                        for (bx, by, bw, bh) in valley_boxes]

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

    corrected = correct_illumination(gray)
    binary = clean_mask(robust_binarize(corrected))
    binary_inv = clean_mask(cv2.bitwise_not(binary))

    comps = _get_components(binary) + _get_components(binary_inv)
    comps = [c for c in comps if _looks_like_digit(c)]
    comps = _dedupe_cross_polarity(_remove_nested_components(comps))

    group = _best_row_group(comps, n_digits, row_tolerance_frac=row_tolerance_frac, prefer_lowest=True)
    if group is not None:
        return [(c.x, c.y, c.w, c.h) for c in group]

    if len(comps) >= n_digits:
        heights = sorted(c.h for c in comps)
        ref_h = heights[len(heights) // 2]
        row_tol = ref_h * row_tolerance_frac
        comps_sorted_by_y = sorted(comps, key=lambda c: c.cy)
        best_group_fallback, best_score_fallback = None, float("inf")
        seen_cy = set()
        for base in comps_sorted_by_y:
            key = round(base.cy)
            if key in seen_cy:
                continue
            seen_cy.add(key)
            grp = [c for c in comps if abs(c.cy - base.cy) <= row_tol]
            if len(grp) < n_digits:
                continue
            trimmed = _best_subset(grp, n_digits)
            score = _group_score(trimmed)
            if score < best_score_fallback:
                best_score_fallback = score
                best_group_fallback = trimmed

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
