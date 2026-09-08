from __future__ import annotations

import argparse
import csv
import os
import time
from typing import Dict, List, Tuple

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
    errors: List[Tuple[str, str, str]] = []

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
            errors.append((fname, true_code, "<localization failed>"))
            continue

        if pred_code == true_code:
            exact_correct += 1
        else:
            errors.append((fname, true_code, pred_code))
            if len(pred_code) != len(true_code):
                mismatched_length += 1

        c, n = per_digit_accuracy(pred_code, true_code)
        correct_digits += c
        total_digits += n

    elapsed = time.time() - t_start

    exact_acc = exact_correct / max(1, total)
    digit_acc = correct_digits / max(1, total_digits)

    print(f"\n=== Full pipeline evaluation: {images_dir} ===")
    print(f"n_digits assumed:        {n_digits}")
    print(f"images evaluated:        {total}")
    print(f"exact full-code accuracy: {exact_acc:.4f}  ({exact_correct}/{total})")
    print(f"per-digit accuracy:       {digit_acc:.4f}  ({correct_digits}/{total_digits})")
    print(f"localization failures:    {localization_failures}")
    print(f"length-mismatched preds:  {mismatched_length}")
    print(f"elapsed:                  {elapsed:.2f}s  ({elapsed/max(1,total):.3f}s/image)")

    if errors:
        print(f"\nFirst {min(20, len(errors))} errors (filename, true, predicted):")
        for fname, true_code, pred_code in errors[:20]:
            print(f"  {fname}: true={true_code}  pred={pred_code}")

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
