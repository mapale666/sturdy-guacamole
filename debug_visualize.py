"""
debug_visualize.py
===================
Diagnostic tool: for a given seal image, draws the detected TAG BOUNDARY
(Stage 1) and the final digit boxes (Stage 3) on top of the original
photo, saves each extracted per-digit crop, and (--dump-components)
prints candidate component stats for threshold tuning.

Usage:
    python debug_visualize.py --image "C:\\path\\to\\val\\00001.png" --out debug_out --n-digits 7 --dump-components
"""

import argparse
import os

import cv2
import numpy as np

import preprocessing as pp


def dump_candidate_components(gray: np.ndarray, n_digits: int = 7):
    tag_bbox = pp._find_tag_bbox(gray)
    print(f"\n--- tag bbox: {tag_bbox} ---")
    if tag_bbox is None:
        print("Tag boundary NOT found -- pipeline will use the whole-image fallback.")
        return

    tx0, ty0, tx1, ty1 = tag_bbox
    tag_crop = gray[ty0:ty1, tx0:tx1]
    corrected = pp.correct_illumination(tag_crop)

    for polarity_first in ("area", "inverted"):
        label = "normal polarity" if polarity_first == "area" else "inverted polarity"
        if polarity_first == "area":
            binary = pp.clean_mask(pp.robust_binarize(corrected, polarity="area"))
        else:
            binary = pp.clean_mask(cv2.bitwise_not(pp.robust_binarize(corrected, polarity="area")))

        bands = pp._find_text_row_bands(binary)
        print(f"\n[{label}] text row-bands found (y0,y1) within tag crop: {bands}")
        if not bands:
            continue

        y0, y1 = bands[-1]
        pad_y = max(3, int(0.1 * (y1 - y0)))
        row_y0, row_y1 = max(0, y0 - pad_y), min(binary.shape[0], y1 + pad_y)
        row_binary = binary[row_y0:row_y1, :]

        comps = pp._get_components(row_binary)
        shape_filtered = [c for c in comps if pp._looks_like_digit(c)]
        deduped = pp._remove_nested_components(shape_filtered)
        print(f"[{label}] lowest band -> {len(comps)} raw -> {len(shape_filtered)} shape-filtered "
              f"-> {len(deduped)} deduped components")
        for c in sorted(deduped, key=lambda c: c.x):
            aspect = c.h / c.w if c.w > 0 else 0
            fill = c.area / float(c.w * c.h) if c.w > 0 and c.h > 0 else 0
            print(f"  x={c.x:>5} y={c.y:>5} w={c.w:>5} h={c.h:>5} aspect={aspect:>5.2f} fill={fill:>5.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", default="debug_out")
    parser.add_argument("--n-digits", type=int, default=7)
    parser.add_argument("--dump-components", action="store_true",
                         help="Print candidate component stats for threshold tuning.")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.image))[0]

    bgr = pp.load_image(args.image)

    if args.dump_components:
        gray = pp.to_gray(bgr)
        dump_candidate_components(gray, n_digits=args.n_digits)

    leveled_gray, boxes = pp.locate_digit_row_multi_orientation(bgr, n_digits=args.n_digits)

    print(f"\nboxes found: {boxes}")

    overlay = cv2.cvtColor(leveled_gray, cv2.COLOR_GRAY2BGR)

    tag_bbox = pp._find_tag_bbox(leveled_gray)
    if tag_bbox is not None:
        tx0, ty0, tx1, ty1 = tag_bbox
        cv2.rectangle(overlay, (tx0, ty0), (tx1, ty1), (255, 0, 0), 3)

    if boxes:
        for i, (x, y, w, h) in enumerate(boxes):
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(overlay, str(i), (x, max(0, y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    overlay_path = os.path.join(args.out, f"{stem}_boxes.png")
    cv2.imwrite(overlay_path, overlay)
    print(f"wrote {overlay_path} (blue = detected tag boundary, red = digit boxes)")

    if boxes is None:
        print("No boxes found -- localization failed entirely for this image.")
        return

    crops = pp.extract_digit_crops(leveled_gray, boxes)
    for i, crop in enumerate(crops):
        crop_path = os.path.join(args.out, f"{stem}_digit{i}_binary.png")
        cv2.imwrite(crop_path, crop)

        canvas = pp.to_classical_canvas(crop, size=112)
        canvas_vis = (canvas * 255).astype(np.uint8)
        canvas_path = os.path.join(args.out, f"{stem}_digit{i}_canvas.png")
        cv2.imwrite(canvas_path, canvas_vis)

    print(f"wrote {len(crops)} crop pairs to {args.out}/")


if __name__ == "__main__":
    main()
