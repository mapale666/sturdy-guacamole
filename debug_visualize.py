"""
debug_visualize.py
===================
Diagnostic tool: for a given seal image, draws the boxes find_digit_row()
actually selected on top of the original photo, saves each extracted
per-digit crop, AND (new) prints every candidate connected component's
shape stats (width, height, aspect ratio, fill ratio, position) so
filter thresholds in preprocessing.py can be tuned against real numbers
instead of guesswork.

Usage:
    python debug_visualize.py --image "C:\\path\\to\\val\\00001.png" --out debug_out --n-digits 7
"""

import argparse
import os

import cv2
import numpy as np

import preprocessing as pp


def dump_candidate_components(gray: np.ndarray, n_digits: int = 7):
    """
    Re-runs the first part of find_digit_row()'s logic (component
    extraction, BEFORE the aspect/fill-ratio filter is applied) and
    prints stats for every raw candidate component, flagging which ones
    the current _looks_like_digit() filter would keep or reject. This is
    for calibrating thresholds, not for production use.
    """
    corrected = pp.correct_illumination(gray)
    binary = pp.clean_mask(pp.robust_binarize(corrected))
    binary_inv = pp.clean_mask(cv2.bitwise_not(binary))

    all_comps = pp._get_components(binary) + pp._get_components(binary_inv)
    print(f"\n--- {len(all_comps)} raw candidate components (before shape filter) ---")
    print(f"{'x':>5} {'y':>5} {'w':>5} {'h':>5} {'aspect':>7} {'fill':>6} {'area':>7} {'keep?':>6}")
    for c in sorted(all_comps, key=lambda c: (round(c.cy / 20), c.x)):
        aspect = c.h / c.w if c.w > 0 else 0
        fill = c.area / float(c.w * c.h) if c.w > 0 and c.h > 0 else 0
        keep = pp._looks_like_digit(c)
        print(f"{c.x:>5} {c.y:>5} {c.w:>5} {c.h:>5} {aspect:>7.2f} {fill:>6.2f} {c.area:>7d} {str(keep):>6}")

    kept = [c for c in all_comps if pp._looks_like_digit(c)]
    print(f"\n{len(kept)} components pass the current shape filter.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", default="debug_out")
    parser.add_argument("--n-digits", type=int, default=7)
    parser.add_argument("--dump-components", action="store_true",
                         help="Print raw candidate component stats for threshold tuning.")
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
    if boxes:
        for i, (x, y, w, h) in enumerate(boxes):
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(overlay, str(i), (x, max(0, y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    overlay_path = os.path.join(args.out, f"{stem}_boxes.png")
    cv2.imwrite(overlay_path, overlay)
    print(f"wrote {overlay_path}")

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
