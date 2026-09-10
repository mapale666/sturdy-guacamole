"""
evaluate_pipeline.py
=====================
End-to-end evaluation of the FULL seal recognition pipeline (localization +
digit classification combined) against the train/val/test seal-image
folders, using the ground-truth codes in splits/split_seals/*.csv.

CSV format confirmed: semicolon-delimited, columns "filename;number".
Seal codes are 7 digits (e.g. "1584143") -- n_digits defaults to 7.

INSTRUMENTATION ADDED: a per-digit-class 10x10 confusion matrix (built
from position-aligned true/pred digits across every evaluated image) and
a cyclic-rotation detector. These two things look similar in the raw
error list -- "true=1584795 pred=5158479" looks like scattered digit
errors, but it's actually the WHOLE sequence rotated by one position
(a leftover localization/box-ordering issue), not a classifier confusion
between specific digits. Mixing rotation errors into a "which digits does
the classifier confuse" analysis would give misleading answers -- e.g. it
would look like every digit is confused with its neighbor, when the
actual cause is unrelated to per-digit classification at all. Separating
these lets you tell whether further effort should go into the CNN/crop
preprocessing (fix specific digit-pair confusions) or into localization
(fix the remaining box-ordering edge cases).

Usage:
python evaluate_pipeline.py \
    --images-dir val \
    --labels-csv splits/split_seals/val.csv \
    --checkpoint digit_cnn.pt
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from typing import Dict, List, Tuple

import numpy as np
import torch

from model import load_model, recognize_seal, resolve_device, N_DIGITS_DEFAULT


def load_labels(csv_path: str, filename_col: str, code_col: str, delimiter: str) -> Dict[str, str]:
    labels = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            fname = row[filename_col].strip()
            code = str(row[code_col]).strip()
            labels[fname] = code
    return labels


def per_digit_accuracy(pred: str, true: str) -> Tuple[int, int]:
    n = min(len(pred), len(true))
    correct = sum(1 for i in range(n) if pred[i] == true[i])
    return correct, len(true)


def cyclic_rotation_shift(pred: str, true: str) -> "int | None":
    """
    Returns the shift k (1 <= k < len(true)) such that
    true[k:] + true[:k] == pred, or None if pred isn't an exact cyclic
    rotation of true. Used to separate localization/box-ordering errors
    (whole sequence shifted) from genuine per-digit classification errors.
    """
    if len(pred) != len(true) or len(true) < 2:
        return None
    doubled = true + true
    idx = doubled.find(pred)
    if idx == -1 or idx == 0:
        return None
    return idx


def evaluate(images_dir: str, labels_csv: str, checkpoint: str,
             filename_col: str = "filename", code_col: str = "number",
             delimiter: str = ";", n_digits: int = N_DIGITS_DEFAULT,
             device: str = None):
    device = resolve_device(device)
    model = load_model(checkpoint, device=device)
    labels = load_labels(labels_csv, filename_col, code_col, delimiter)

    total, exact_correct = 0, 0
    total_digits, correct_digits = 0, 0
    localization_failures = 0
    mismatched_length = 0
    rotation_errors = 0
    other_errors = 0
    errors: List[Tuple[str, str, str]] = []
    confusion = np.zeros((10, 10), dtype=int)  # rows=true digit, cols=predicted digit

    t_start = time.time()
    for fname, true_code in labels.items():
        path = os.path.join(images_dir, fname)
        if not os.path.exists(path):
            print(f"[WARN] listed in {labels_csv} but missing on disk: {fname}")
            continue

        pred_code = recognize_seal(model, path, n_digits=n_digits, device=device)
        total += 1

        if pred_code is None:
            localization_failures += 1
            errors.append((fname, true_code, ""))
            continue

        if pred_code == true_code:
            exact_correct += 1
        else:
            errors.append((fname, true_code, pred_code))
            if len(pred_code) != len(true_code):
                mismatched_length += 1
            elif cyclic_rotation_shift(pred_code, true_code) is not None:
                rotation_errors += 1
            else:
                other_errors += 1

        c, n = per_digit_accuracy(pred_code, true_code)
        correct_digits += c
        total_digits += n

        if len(pred_code) == len(true_code):
            for t_ch, p_ch in zip(true_code, pred_code):
                if t_ch.isdigit() and p_ch.isdigit():
                    confusion[int(t_ch), int(p_ch)] += 1

    elapsed = time.time() - t_start

    exact_acc = exact_correct / max(1, total)
    digit_acc = correct_digits / max(1, total_digits)

    print(f"\n=== Full pipeline evaluation: {images_dir} ===")
    print(f"n_digits assumed:        {n_digits}")
    print(f"images evaluated:        {total}")
    print(f"exact full-code accuracy: {exact_acc:.4f} ({exact_correct}/{total})")
    print(f"per-digit accuracy:       {digit_acc:.4f} ({correct_digits}/{total_digits})")
    print(f"localization failures:   {localization_failures}")
    print(f"length-mismatched preds: {mismatched_length}")
    print(f"whole-sequence rotation errors (localization/box-order, NOT classifier): {rotation_errors}")
    print(f"other (genuine digit-level) errors: {other_errors}")
    print(f"elapsed: {elapsed:.2f}s ({elapsed/max(1,total):.3f}s/image)")

    print("\n=== Per-digit confusion matrix (rows=true, cols=predicted) ===")
    header = "     " + " ".join(f"{d:5d}" for d in range(10))
    print(header)
    for t in range(10):
        row = " ".join(f"{confusion[t, p]:5d}" for p in range(10))
        print(f"true {t}: {row}")

    print("\n=== Top confused digit pairs (true -> pred), excluding correct matches ===")
    pairs = []
    for t in range(10):
        for p in range(10):
            if t != p and confusion[t, p] > 0:
                pairs.append((confusion[t, p], t, p))
    pairs.sort(reverse=True)
    for count, t, p in pairs[:20]:
        row_total = confusion[t, :].sum()
        rate = count / row_total if row_total else 0.0
        print(f"  true={t} -> pred={p}: {count} times ({rate:.1%} of all true-{t} occurrences)")

    if errors:
        print(f"\nFirst {min(20, len(errors))} errors (filename, true, predicted):")
        for fname, true_code, pred_code in errors[:20]:
            tag = ""
            if pred_code and len(pred_code) == len(true_code):
                shift = cyclic_rotation_shift(pred_code, true_code)
                if shift is not None:
                    tag = f"  [ROTATION shift={shift}]"
            print(f"  {fname}: true={true_code} pred={pred_code}{tag}")

    return exact_acc, digit_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the full seal recognition pipeline")
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--checkpoint", default="digit_cnn.pt")
    parser.add_argument("--filename-col", default="filename")
    parser.add_argument("--code-col", default="number")
    parser.add_argument("--delimiter", default=";")
    parser.add_argument("--n-digits", type=int, default=N_DIGITS_DEFAULT)
    parser.add_argument("--device", choices=["cuda", "cpu"], default=None)
    args = parser.parse_args()

    evaluate(args.images_dir, args.labels_csv, args.checkpoint,
             filename_col=args.filename_col, code_col=args.code_col,
             delimiter=args.delimiter, n_digits=args.n_digits, device=args.device)
