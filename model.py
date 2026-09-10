from __future__ import annotations

import argparse
import csv
import glob
import os
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import cv2

import preprocessing as pp

IMG_SIZE = 28
N_CLASSES = 10
N_DIGITS_DEFAULT = 7
VALID_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


def resolve_device(requested: Optional[str] = None) -> str:
    if requested == "cuda":
        if not torch.cuda.is_available():
            print("[WARN] --device cuda requested but torch.cuda.is_available() is False. "
                  "Falling back to CPU.")
            return "cpu"
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        return "cuda"
    if requested == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        print(f"Auto-detected GPU: {torch.cuda.get_device_name(0)}")
        return "cuda"
    print("[INFO] No CUDA GPU detected, using CPU.")
    return "cpu"


# --------------------------------------------------------------------------- #
# Label parsing: "<id>_<label>.<ext>" or "<id>_<label>-<position>.<ext>" -> label
# --------------------------------------------------------------------------- #

def parse_label_from_filename(filename: str) -> Optional[int]:
    stem = os.path.splitext(filename)[0]
    if "_" not in stem:
        return None
    tail = stem.rsplit("_", 1)[-1]
    label_part = tail.split("-", 1)[0]
    if not label_part.isdigit():
        return None
    value = int(label_part)
    if 0 <= value <= 9:
        return value
    return None


def scan_chars_dir(chars_dir: str) -> List[Tuple[str, int]]:
    samples = []
    skipped = 0
    for path in sorted(glob.glob(os.path.join(chars_dir, "*"))):
        fname = os.path.basename(path)
        if os.path.splitext(fname)[1].lower() not in VALID_EXTS:
            continue
        label = parse_label_from_filename(fname)
        if label is None:
            skipped += 1
            continue
        samples.append((fname, label))
    if skipped:
        print(f"[WARN] skipped {skipped} files in {chars_dir} with unparseable filenames")
    return samples


# --------------------------------------------------------------------------- #
# Stratified split (80/10/10 per class), reproducible via fixed seed
# --------------------------------------------------------------------------- #

def stratified_split(samples: List[Tuple[str, int]], seed: int = 42,
                      train_frac: float = 0.8, val_frac: float = 0.1
                      ) -> Tuple[List, List, List]:
    by_class: Dict[int, List[Tuple[str, int]]] = defaultdict(list)
    for fname, label in samples:
        by_class[label].append((fname, label))

    rng = random.Random(seed)
    train, val, test = [], [], []
    for label, items in sorted(by_class.items()):
        rng.shuffle(items)
        n = len(items)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        train += items[:n_train]
        val += items[n_train:n_train + n_val]
        test += items[n_train + n_val:]
        print(f"class {label}: total={n} train={n_train} val={n_val} test={n - n_train - n_val}")

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def write_manifest(path: str, samples: List[Tuple[str, int]]):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label"])
        writer.writerows(samples)


def load_manifest(path: str) -> List[Tuple[str, int]]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return [(row["filename"], int(row["label"])) for row in reader]


def get_or_create_splits(chars_dir: str, manifest_dir: Optional[str] = None,
                          seed: int = 42, force: bool = False
                          ) -> Tuple[List, List, List]:
    manifest_dir = manifest_dir or chars_dir
    base = os.path.basename(os.path.normpath(chars_dir))
    paths = {
        split: os.path.join(manifest_dir, f"{base}_manifest_{split}.csv")
        for split in ("train", "val", "test")
    }

    if not force and all(os.path.exists(p) for p in paths.values()):
        print("Reusing cached digit split manifests.")
        return (load_manifest(paths["train"]), load_manifest(paths["val"]), load_manifest(paths["test"]))

    print(f"No cached split found -- scanning {chars_dir} and creating a fresh stratified 80/10/10 split.")
    samples = scan_chars_dir(chars_dir)
    if not samples:
        raise RuntimeError(f"No parseable digit images found in {chars_dir}")
    train, val, test = stratified_split(samples, seed=seed)

    os.makedirs(manifest_dir, exist_ok=True)
    write_manifest(paths["train"], train)
    write_manifest(paths["val"], val)
    write_manifest(paths["test"], test)
    print(f"Wrote split manifests to {manifest_dir}")
    return train, val, test


# --------------------------------------------------------------------------- #
# Augmentation (train-time only)
# --------------------------------------------------------------------------- #

def add_speckle_noise(img: np.ndarray, rng: random.Random, max_specks: int = 6) -> np.ndarray:

    if rng.random() >= 0.4:
        return img
    out = img.copy()
    h, w = out.shape
    n_specks = rng.randint(1, max_specks)
    for _ in range(n_specks):
        cx, cy = rng.randint(0, w - 1), rng.randint(0, h - 1)
        r = rng.choice([0, 1])
        val = rng.uniform(0.6, 1.0)
        y0, y1 = max(0, cy - r), min(h, cy + r + 1)
        x0, x1 = max(0, cx - r), min(w, cx + r + 1)
        out[y0:y1, x0:x1] = np.maximum(out[y0:y1, x0:x1], val)
    return out


def random_morph(img: np.ndarray, rng: random.Random) -> np.ndarray:
    if rng.random() >= 0.4:
        return img
    binary_u8 = (img > 0.5).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    if rng.random() < 0.5:
        morphed = cv2.erode(binary_u8, kernel, iterations=1)
    else:
        morphed = cv2.dilate(binary_u8, kernel, iterations=1)
    return morphed.astype(np.float32) / 255.0


def random_erase(img: np.ndarray, rng: random.Random) -> np.ndarray:
    if rng.random() >= 0.25:
        return img
    out = img.copy()
    h, w = out.shape
    ew = rng.randint(2, max(2, w // 4))
    eh = rng.randint(2, max(2, h // 4))
    ex = rng.randint(0, max(0, w - ew))
    ey = rng.randint(0, max(0, h - eh))
    out[ey:ey + eh, ex:ex + ew] = 0.0
    return out


def augment_digit(canvas: np.ndarray, rng: random.Random) -> np.ndarray:
    img = canvas.copy()

    if rng.random() < 0.5:
        k = rng.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)

    if rng.random() < 0.7:
        angle = rng.uniform(-20, 20)
        M = cv2.getRotationMatrix2D((IMG_SIZE / 2, IMG_SIZE / 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (IMG_SIZE, IMG_SIZE), borderValue=0)

    if rng.random() < 0.6:
        tx, ty = rng.uniform(-3, 3), rng.uniform(-3, 3)
        scale = rng.uniform(0.85, 1.15)
        M = cv2.getRotationMatrix2D((IMG_SIZE / 2, IMG_SIZE / 2), 0, scale)
        M[0, 2] += tx
        M[1, 2] += ty
        img = cv2.warpAffine(img, M, (IMG_SIZE, IMG_SIZE), borderValue=0)

    # Domain-randomization additions: mimic real-photo-crop degradation
    # that the clean isolated training scans don't naturally have.
    img = random_morph(img, rng)
    img = add_speckle_noise(img, rng)
    img = random_erase(img, rng)

    if rng.random() < 0.6:
        gain = rng.uniform(0.6, 1.4)
        bias = rng.uniform(-0.15, 0.15)
        img = np.clip(img * gain + bias, 0, 1)

    if rng.random() < 0.3:
        noise = np.random.normal(0, 0.03, img.shape).astype(np.float32)
        img = np.clip(img + noise, 0, 1)

    return img.astype(np.float32)


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #

class CharsDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, int]], chars_dir: str, train: bool, seed: int = 42):
        self.samples = samples
        self.chars_dir = chars_dir
        self.train = train
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        fname, label = self.samples[idx]
        path = os.path.join(self.chars_dir, fname)
        gray = pp.to_gray(pp.load_image(path))

        binary = pp.prepare_isolated_crop(gray)
        canvas = pp.to_classical_canvas(binary, size=IMG_SIZE).astype(np.float32)

        if self.train:
            canvas = augment_digit(canvas, self.rng)

        tensor = torch.from_numpy(canvas).unsqueeze(0)
        return tensor, label


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

class DigitCNN(nn.Module):
    def __init__(self, n_classes: int = N_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(3),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 3 * 3, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# --------------------------------------------------------------------------- #
# Training / evaluation (digit-level)
# --------------------------------------------------------------------------- #

def train_model(chars_dir: str, out_path: str, manifest_dir: Optional[str] = None,
                 epochs: int = 25, batch_size: int = 128, lr: float = 1e-3,
                 seed: int = 42, device: Optional[str] = None) -> DigitCNN:
    device = resolve_device(device)

    train_s, val_s, test_s = get_or_create_splits(chars_dir, manifest_dir, seed=seed)

    train_ds = CharsDataset(train_s, chars_dir, train=True, seed=seed)
    val_ds = CharsDataset(val_s, chars_dir, train=False)

    pin_memory = device == "cuda"
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=2, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=2, pin_memory=pin_memory)

    model = DigitCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=pin_memory), y.to(device, non_blocking=pin_memory)
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
            torch.save({"model_state": model.state_dict(), "val_acc": val_acc, "img_size": IMG_SIZE}, out_path)

    print(f"best val_acc={best_val_acc:.4f}, saved to {out_path}")
    return model


def evaluate_digit_model(checkpoint_path: str, chars_dir: str, manifest_dir: Optional[str] = None,
                          split: str = "test", device: Optional[str] = None) -> float:
    device = resolve_device(device)
    model = load_model(checkpoint_path, device=device)

    train_s, val_s, test_s = get_or_create_splits(chars_dir, manifest_dir)
    samples = {"train": train_s, "val": val_s, "test": test_s}[split]
    ds = CharsDataset(samples, chars_dir, train=False)
    loader = DataLoader(ds, batch_size=128, shuffle=False)

    confusion = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            preds = model(x).argmax(1).cpu().numpy()
            for p, t in zip(preds, y.numpy()):
                confusion[t, p] += 1
            correct += (preds == y.numpy()).sum()
            total += len(y)

    acc = correct / max(1, total)
    print(f"[{split}] digit_acc={acc:.4f}")
    print("confusion matrix (rows=true, cols=pred):")
    print(confusion)
    return acc


# --------------------------------------------------------------------------- #
# Inference glue (used by main.py)
# --------------------------------------------------------------------------- #

def load_model(checkpoint_path: str, device: Optional[str] = None) -> DigitCNN:
    device = resolve_device(device) if device in (None, "cuda", "cpu") else device
    model = DigitCNN().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


@torch.no_grad()
def predict_digits(model: DigitCNN, crops: List[np.ndarray], device: Optional[str] = None) -> List[int]:
    device = device or next(model.parameters()).device
    if not crops:
        return []
    batch = torch.from_numpy(np.stack(crops)).unsqueeze(1).to(device)
    logits = model(batch)
    return logits.argmax(1).cpu().tolist()


def recognize_seal(model: DigitCNN, image_path: str, n_digits: int = N_DIGITS_DEFAULT,
                    device: Optional[str] = None) -> Optional[str]:
    crops = pp.preprocess_seal(image_path, mode="cnn", n_digits=n_digits, img_size=IMG_SIZE)
    if crops is None or len(crops) != n_digits:
        return None
    digits = predict_digits(model, crops, device=device)
    return "".join(str(d) for d in digits)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the digit CNN on ground_truth_chars_balanced")
    parser.add_argument("--chars-dir", required=True)
    parser.add_argument("--manifest-dir", default=None)
    parser.add_argument("--out", default="digit_cnn.pt")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-test", action="store_true")
    parser.add_argument("--device", choices=["cuda", "cpu"], default=None)
    args = parser.parse_args()

    train_model(args.chars_dir, args.out, manifest_dir=args.manifest_dir,
                epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                seed=args.seed, device=args.device)

    if args.eval_test:
        evaluate_digit_model(args.out, args.chars_dir, manifest_dir=args.manifest_dir,
                              split="test", device=args.device)
