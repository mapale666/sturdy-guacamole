from __future__ import annotations

import argparse
import csv
import os
import random
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
    rotation of true. Separates localization/box-ordering errors (whole
    sequence shifted) from genuine per-digit classification errors.
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
             device: str = None, sample_size: int = None, seed: int = 42):
    device = resolve_device(device)
    model = load_model(checkpoint, device=device)
    labels = load_labels(labels_csv, filename_col, code_col, delimiter)

    items = list(labels.items())
    if sample_size is not None and sample_size < len(items):
        rng = random.Random(seed)
        items = rng.sample(items, sample_size)
        print(f"[INFO] evaluating a random sample of {sample_size}/{len(labels)} images (seed={seed})")

    total, exact_correct = 0, 0
    total_digits, correct_digits = 0, 0
    localization_failures = 0
    mismatched_length = 0
    rotation_errors = 0
    other_errors = 0
    errors: List[Tuple[str, str, str]] = []
    confusion = np.zeros((10, 10), dtype=int)  # rows=true digit, cols=predicted digit

    t_start = time.time()
    for fname, true_code in items:
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
    print(f"images evaluated:        {total}" + (f" (sampled from {len(labels)})" if sample_size else ""))
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
    parser.add_argument("--sample-size", type=int, default=None,
                         help="Evaluate a random subset of this many images instead of the full folder (fast iteration).")
    parser.add_argument("--seed", type=int, default=42,
                         help="Random seed for --sample-size, so repeated runs sample the same subset.")
    args = parser.parse_args()

    evaluate(args.images_dir, args.labels_csv, args.checkpoint,
             filename_col=args.filename_col, code_col=args.code_col,
             delimiter=args.delimiter, n_digits=args.n_digits, device=args.device,
             sample_size=args.sample_size, seed=args.seed)
