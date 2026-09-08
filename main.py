"""
main.py
=======
Entry point for the competition evaluation. Loads the trained digit CNN,
processes every PNG in --input-dir through preprocessing.py + model.py, and
writes a CSV of (filename, predicted_code) to --output-dir.

Usage:
    python main.py --input-dir /path/to/input --output-dir /path/to/output

Required alongside this file: preprocessing.py, model.py, and the trained
checkpoint (digit_cnn.pt) in the same directory (or pass --checkpoint).
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import time

import torch

import preprocessing as pp
from model import load_model, recognize_seal, resolve_device, N_DIGITS_DEFAULT


TEAM_NAME = "eval:squared(mat).csv"

N_DIGITS = N_DIGITS_DEFAULT  # seal codes are 7 digits, e.g. "1584143"
CHECKPOINT_DEFAULT = "digit_cnn.pt"


def parse_args():
    parser = argparse.ArgumentParser(description="Seal numeral recognition - main entry point")
    parser.add_argument("--input-dir", required=True, help="Directory containing input PNG images")
    parser.add_argument("--output-dir", required=True, help="Directory to write the result CSV into")
    parser.add_argument("--checkpoint", default=CHECKPOINT_DEFAULT, help="Path to trained model weights")
    parser.add_argument("--team-name", default=TEAM_NAME, help="Overrides the output CSV filename stem")
    parser.add_argument("--device", choices=["cuda", "cpu"], default=None)
    return parser.parse_args()


def find_png_images(input_dir: str):
    paths = sorted(glob.glob(os.path.join(input_dir, "*.png")))
    paths += sorted(glob.glob(os.path.join(input_dir, "*.PNG")))
    return paths


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = resolve_device(args.device)

    model = load_model(args.checkpoint, device=device)

    image_paths = find_png_images(args.input_dir)
    print(f"Found {len(image_paths)} PNG images in {args.input_dir}")

    results = []
    t_start = time.time()

    for path in image_paths:
        filename = os.path.basename(path)
        try:
            code = recognize_seal(model, path, n_digits=N_DIGITS, device=device)
        except Exception as exc:
            print(f"[WARN] failed on {filename}: {exc}")
            code = None

        if code is None:
            code = ""
            print(f"[WARN] could not recognize digits in {filename}")

        results.append((filename, code))

    elapsed = time.time() - t_start
    print(f"Processed {len(image_paths)} images in {elapsed:.2f}s "
          f"({elapsed / max(1, len(image_paths)):.3f}s/image)")

    out_path = os.path.join(args.output_dir, f"{args.team_name}.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "code"])
        writer.writerows(results)

    print(f"Wrote results to {out_path}")


if __name__ == "__main__":
    main()
