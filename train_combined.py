from __future__ import annotations

import argparse
import csv
import os
import random
from collections import Counter
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler

import model as m  # reuse DigitCNN, CharsDataset, augment_digit, get_or_create_splits, IMG_SIZE


class RealCharsDataset(Dataset):
    def __init__(self, real_dir: str, train: bool, seed: int = 42):
        self.real_dir = real_dir
        self.train = train
        self.rng = random.Random(seed)
        manifest_path = os.path.join(real_dir, "manifest.csv")
        with open(manifest_path, newline="") as f:
            reader = csv.DictReader(f)
            self.samples: List[Tuple[str, int]] = [(row["filename"], int(row["label"])) for row in reader]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        fname, label = self.samples[idx]
        canvas = np.load(os.path.join(self.real_dir, fname)).astype(np.float32)
        if self.train:
            canvas = m.augment_digit(canvas, self.rng)
        tensor = torch.from_numpy(canvas).unsqueeze(0)
        return tensor, label


def stratified_split_real(samples: List[Tuple[str, int]], seed: int = 42,
                           train_frac: float = 0.9) -> Tuple[List, List]:
    by_class = {}
    for fname, label in samples:
        by_class.setdefault(label, []).append((fname, label))
    rng = random.Random(seed)
    train, val = [], []
    for label, items in sorted(by_class.items()):
        rng.shuffle(items)
        n_train = int(len(items) * train_frac)
        train += items[:n_train]
        val += items[n_train:]
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def build_class_balanced_sampler(all_labels: List[int]) -> WeightedRandomSampler:
    counts = Counter(all_labels)
    print(f"[INFO] combined train class counts: {dict(sorted(counts.items()))}")
    weights = [1.0 / counts[label] for label in all_labels]
    return WeightedRandomSampler(weights, num_samples=len(all_labels), replacement=True)


def train_combined(chars_dir: str, real_dir: str, out_path: str,
                    manifest_dir: Optional[str] = None, epochs: int = 25,
                    batch_size: int = 128, lr: float = 1e-3, seed: int = 42,
                    device: Optional[str] = None, weight_power: float = 0.5,
                    use_sampler: bool = True):
    device = m.resolve_device(device)

    train_s, val_s, _ = m.get_or_create_splits(chars_dir, manifest_dir, seed=seed)
    synthetic_train = m.CharsDataset(train_s, chars_dir, train=True, seed=seed)
    synthetic_val = m.CharsDataset(val_s, chars_dir, train=False)

    real_manifest_path = os.path.join(real_dir, "manifest.csv")
    with open(real_manifest_path, newline="") as f:
        reader = csv.DictReader(f)
        real_samples = [(row["filename"], int(row["label"])) for row in reader]
    real_train_s, real_val_s = stratified_split_real(real_samples, seed=seed)

    real_train_ds = RealCharsDataset(real_dir, train=True, seed=seed)
    real_train_ds.samples = real_train_s
    real_val_ds = RealCharsDataset(real_dir, train=False)
    real_val_ds.samples = real_val_s

    print(f"[INFO] synthetic: {len(synthetic_train)} train / {len(synthetic_val)} val")
    print(f"[INFO] real:      {len(real_train_ds)} train / {len(real_val_ds)} val")

    train_ds = ConcatDataset([synthetic_train, real_train_ds])
    val_ds = ConcatDataset([synthetic_val, real_val_ds])

    all_train_labels = [label for _, label in train_s] + [label for _, label in real_train_s]

    # SOFTENED class weighting: weight_power=0.5 takes the sqrt of the
    # inverse-frequency weight, so classes are still corrected for but
    # much less aggressively than the previous run (weight_power=1.0),
    # which over-corrected and pushed everything toward "4".
    num_classes = 10
    counts = Counter(all_train_labels)
    print(f"[INFO] combined train class counts: {dict(sorted(counts.items()))}")
    total = sum(counts.get(c, 0) for c in range(num_classes))
    raw_weights = [total / (num_classes * max(1, counts.get(c, 0))) for c in range(num_classes)]
    class_weights = torch.tensor(
        [w ** weight_power for w in raw_weights], dtype=torch.float32
    ).to(device)
    print(f"[INFO] class weights (power={weight_power}): {[round(w, 3) for w in class_weights.tolist()]}")

    if use_sampler:
        sampler = build_class_balanced_sampler(all_train_labels)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                                   num_workers=2, pin_memory=(device == "cuda"))
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                   num_workers=2, pin_memory=(device == "cuda"))

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=2, pin_memory=(device == "cuda"))

    model = m.DigitCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    best_val_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=(device == "cuda")), y.to(device, non_blocking=(device == "cuda"))
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
            train_correct += (out.argmax(1) == y).sum().item()
            train_total += x.size(0)

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                val_loss += loss.item() * x.size(0)
                val_correct += (out.argmax(1) == y).sum().item()
                val_total += x.size(0)

        train_acc = train_correct / max(1, train_total)
        val_acc = val_correct / max(1, val_total)
        print(f"epoch {epoch:03d} train_loss={train_loss/train_total:.4f} "
              f"train_acc={train_acc:.4f} val_loss={val_loss/val_total:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model_state": model.state_dict(), "val_acc": val_acc, "img_size": m.IMG_SIZE}, out_path)

    print(f"best val_acc={best_val_acc:.4f}, saved to {out_path}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DigitCNN on synthetic + real auto-labeled crops")
    parser.add_argument("--chars-dir", required=True)
    parser.add_argument("--real-dir", required=True)
    parser.add_argument("--manifest-dir", default=None)
    parser.add_argument("--out", default="digit_cnn_v4.pt")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cuda", "cpu"], default=None)
    parser.add_argument("--weight-power", type=float, default=0.5,
                         help="Exponent applied to inverse-frequency class weights. 1.0=full correction (v3), 0.5=softened, 0.0=disabled.")
    parser.add_argument("--no-sampler", action="store_true",
                         help="Disable WeightedRandomSampler, use plain shuffle instead (combine with --weight-power 1.0 to test loss-only weighting).")
    args = parser.parse_args()

    train_combined(args.chars_dir, args.real_dir, args.out,
                    manifest_dir=args.manifest_dir, epochs=args.epochs,
                    batch_size=args.batch_size, lr=args.lr, seed=args.seed,
                    device=args.device, weight_power=args.weight_power,
                    use_sampler=not args.no_sampler)