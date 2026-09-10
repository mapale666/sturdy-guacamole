"""
build_real_digit_dataset.py
============================
Harvests real, correctly-labeled per-digit crops from labeled seal photos
using the (now-reliable) segmentation pipeline in preprocessing.py, to
close the domain gap between synthetic-augmented isolated-char training
data and what the CNN actually sees at inference time.

For each labeled seal image:
  1. Runs the full localization pipeline (same code path as inference).
  2. Skips the image if localization fails or returns the wrong digit count.
  3. Optionally cross-checks the CURRENT model's prediction against the
     true label: if it's an exact cyclic rotation of the true code (a
     box-ordering error, not a classification error), skips the image.
  4. Converts each box to the exact CNN input tensor and saves it as a
     .npy file, paired with its true digit label in a manifest CSV.

The manifest is written INCREMENTALLY. --resume scans the output
directory ONCE at startup (O(1) lookups thereafter, not a fresh listing
per image). Each image now prints BEFORE processing starts (with elapsed
time from the previous one), so a hang is immediately visible by
filename instead of silently blocking progress output. A single image
that raises an exception or exceeds --warn-after-seconds is logged and
does not stop the batch.

Usage:
  python build_real_digit_dataset.py \\
      --images-dir "...\\train" --labels-csv "...\\train.csv" \\
      --out-dir real_digit_crops --checkpoint digit_cnn.pt --resume
"""

from __future__ import annotations

import argparse
import csv
import os
import time
import traceback
from collections import defaultdict
from typing import Dict, Set

import numpy as np

import preprocessing as pp
from model import load_model, predict_digits, resolve_device, N_DIGITS_DEFAULT


def load_labels(csv_path: str, filename_col: str, code_col: str, delimiter: str) -> Dict[str, str]:
    labels = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            fname = row[filename_col].strip()
            code = str(row[code_col]).strip()
            labels[fname] = code
    return labels


def cyclic_rotation_shift(pred: str, true: str):
    if len(pred) != len(true) or len(true) < 2:
        return None
    doubled = true + true
    idx = doubled.find(pred)
    if idx == -1 or idx == 0:
        return None
    return idx


def build_done_set(out_dir: str, n_digits: int) -> Set[str]:
    counts = defaultdict(set)
    if not os.path.isdir(out_dir):
        return set()
    for fname in os.listdir(out_dir):
        if not fname.endswith(".npy"):
            continue
        base = fname[:-4]
        parts = base.rsplit("_", 2)
        if len(parts) != 3:
            continue
        stem, digit_idx_str, _label_str = parts
        if digit_idx_str.isdigit():
            counts[stem].add(int(digit_idx_str))
    return {stem for stem, idxs in counts.items() if len(idxs) >= n_digits}


def main():
    parser = argparse.ArgumentParser(description="Harvest real, labeled digit crops from seal photos")
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--out-dir", default="real_digit_crops")
    parser.add_argument("--checkpoint", default="digit_cnn.pt")
    parser.add_argument("--filename-col", default="filename")
    parser.add_argument("--code-col", default="number")
    parser.add_argument("--delimiter", default=";")
    parser.add_argument("--n-digits", type=int, default=N_DIGITS_DEFAULT)
    parser.add_argument("--img-size", type=int, default=28)
    parser.add_argument("--skip-rotation-check", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--print-every", type=int, default=20)
    parser.add_argument("--warn-after-seconds", type=float, default=3.0,
                         help="Print a warning if a single image takes longer than this.")
    parser.add_argument("--device", choices=["cuda", "cpu"], default=None)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    manifest_path = os.path.join(args.out_dir, "manifest.csv")
    manifest_is_new = not os.path.exists(manifest_path)

    manifest_file = open(manifest_path, "a", newline="")
    manifest_writer = csv.writer(manifest_file)
    if manifest_is_new:
        manifest_writer.writerow(["filename", "label"])
        manifest_file.flush()

    device = resolve_device(args.device)
    model = None
    if not args.skip_rotation_check:
        model = load_model(args.checkpoint, device=device)

    labels = load_labels(args.labels_csv, args.filename_col, args.code_col, args.delimiter)
    print(f"[INFO] {len(labels)} labeled images found in {args.labels_csv}", flush=True)

    done_stems: Set[str] = set()
    if args.resume:
        print("[INFO] scanning output directory once to build resume set...", flush=True)
        done_stems = build_done_set(args.out_dir, args.n_digits)
        print(f"[INFO] {len(done_stems)} source images already fully processed -- will skip those.", flush=True)

    n_saved = 0
    n_localization_failed = 0
    n_rotation_skipped = 0
    n_resumed_skip = 0
    n_processed = 0
    n_errors = 0
    per_class_count = {str(d): 0 for d in range(10)}

    try:
        for i, (fname, true_code) in enumerate(labels.items()):
            path = os.path.join(args.images_dir, fname)
            if not os.path.exists(path):
                continue
            if len(true_code) != args.n_digits or not true_code.isdigit():
                continue

            stem = os.path.splitext(fname)[0]
            if args.resume and stem in done_stems:
                n_resumed_skip += 1
                continue

            t0 = time.time()
            try:
                bgr = pp.load_image(path)
                leveled_gray, boxes = pp.locate_digit_row_multi_orientation(bgr, n_digits=args.n_digits)

                if boxes is None or len(boxes) != args.n_digits:
                    n_localization_failed += 1
                    n_processed += 1
                else:
                    crops = pp.extract_digit_crops(leveled_gray, boxes)
                    tensors = [pp.to_cnn_tensor_array(c, size=args.img_size) for c in crops]

                    skip = False
                    if model is not None:
                        pred_code = "".join(str(d) for d in predict_digits(model, tensors, device=device))
                        if pred_code != true_code and cyclic_rotation_shift(pred_code, true_code) is not None:
                            n_rotation_skipped += 1
                            n_processed += 1
                            skip = True

                    if not skip:
                        for digit_idx, (tensor, true_digit) in enumerate(zip(tensors, true_code)):
                            out_name = f"{stem}_{digit_idx}_{true_digit}.npy"
                            out_path = os.path.join(args.out_dir, out_name)
                            np.save(out_path, tensor.astype(np.float32))
                            manifest_writer.writerow([out_name, true_digit])
                            per_class_count[true_digit] += 1
                            n_saved += 1
                        manifest_file.flush()
                        n_processed += 1
            except Exception as exc:
                n_errors += 1
                n_processed += 1
                print(f"[ERROR] {fname} raised {type(exc).__name__}: {exc} -- skipping. Traceback:", flush=True)
                traceback.print_exc()

            elapsed = time.time() - t0
            if elapsed > args.warn_after_seconds:
                print(f"[SLOW] {fname} took {elapsed:.1f}s", flush=True)

            if n_processed % args.print_every == 0:
                print(f"[INFO] processed {n_processed} new images "
                      f"({n_resumed_skip} skipped via resume, {n_errors} errors), "
                      f"{n_saved} crops saved so far", flush=True)
    finally:
        manifest_file.close()

    print(f"\n[DONE] saved {n_saved} real digit crops to {args.out_dir}")
    print(f"       manifest written to {manifest_path}")
    print(f"       localization failures skipped: {n_localization_failed}")
    print(f"       rotation-mismatch skipped:      {n_rotation_skipped}")
    print(f"       errors (exceptions) skipped:    {n_errors}")
    if args.resume:
        print(f"       already-done images skipped:    {n_resumed_skip}")
    print(f"       per-class crop counts: {per_class_count}")


if __name__ == "__main__":
    main()
