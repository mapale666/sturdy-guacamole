"""
rebuild_manifest.py
====================
Reconstructs manifest.csv for a real_digit_crops/ directory from the
.npy filenames themselves (format: "{stem}_{digit_idx}_{digit}.npy"),
for cases where build_real_digit_dataset.py got interrupted before its
final manifest write but the .npy files were already saved.

Usage:
  python rebuild_manifest.py --dir real_digit_crops
"""

import argparse
import csv
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    args = parser.parse_args()

    rows = []
    skipped = 0
    for fname in sorted(os.listdir(args.dir)):
        if not fname.endswith(".npy"):
            continue
        stem = fname[:-4]  # strip ".npy"
        label = stem.rsplit("_", 1)[-1]
        if not label.isdigit() or not (0 <= int(label) <= 9):
            skipped += 1
            continue
        rows.append((fname, label))

    out_path = os.path.join(args.dir, "manifest.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label"])
        writer.writerows(rows)

    per_class = {str(d): 0 for d in range(10)}
    for _, label in rows:
        per_class[label] += 1

    print(f"[DONE] rebuilt manifest with {len(rows)} entries -> {out_path}")
    if skipped:
        print(f"[WARN] skipped {skipped} .npy files with unparseable names")
    print(f"per-class counts: {per_class}")


if __name__ == "__main__":
    main()
