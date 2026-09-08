import argparse
import os

import cv2
import numpy as np

import preprocessing as pp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", default="debug_out")
    parser.add_argument("--n-digits", type=int, default=7)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.image))[0]

    bgr = pp.load_image(args.image)
    leveled_gray, boxes = pp.locate_digit_row_multi_orientation(bgr, n_digits=args.n_digits)

    print(f"boxes found: {boxes}")

    # Overlay boxes on the leveled grayscale image.
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

        canvas = pp.to_classical_canvas(crop, size=112)  # upscaled for visibility
        canvas_vis = (canvas * 255).astype(np.uint8)
        canvas_path = os.path.join(args.out, f"{stem}_digit{i}_canvas.png")
        cv2.imwrite(canvas_path, canvas_vis)

    print(f"wrote {len(crops)} crop pairs to {args.out}/")


if __name__ == "__main__":
    main()
