"""
Prepare dataset for YOLOv8 classification (rachadura vs sem rachadura).

Converts grayscale images from Pos/Neg (or processed dirs) to RGB
and splits into train/val/test with classes crack/no_crack.

Usage:
    python prepare_dataset.py --flow 1 --split 0.7 0.2 0.1
    python prepare_dataset.py --flow 2 --split 0.8 0.2 0.0
    python prepare_dataset.py --flow raw --split 0.7 0.2 0.1

Args:
    --flow: which processed images to use (1, 2, or raw for originals)
    --split: train/val/test proportions (default 0.7 0.2 0.1)
"""

import argparse
import cv2
import glob
import os
import shutil
import random


def gray_to_rgb(img_gray):
    """Convert grayscale to fake RGB by duplicating the channel."""
    return cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)


def collect_images(paths, label):
    """Read images, convert to RGB, return list of (filename, img_rgb)."""
    results = []
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Falha ao ler: {p}")
            continue
        img_rgb = gray_to_rgb(img)
        results.append((os.path.basename(p), img_rgb))
    return results


def split_and_copy(images, base_dir, label, train_ratio, val_ratio, test_ratio):
    """Split images into train/val/test and copy to dataset structure."""
    random.shuffle(images)
    n = len(images)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    splits = [
        (images[:n_train], "train"),
        (images[n_train:n_train + n_val], "val"),
        (images[n_train + n_val:], "test"),
    ]

    for split_images, split_name in splits:
        if not split_images:
            continue
        class_dir = os.path.join(base_dir, split_name, label)
        os.makedirs(class_dir, exist_ok=True)
        for fname, img in split_images:
            out_path = os.path.join(class_dir, fname)
            cv2.imwrite(out_path, img)


def main():
    parser = argparse.ArgumentParser(description="Prepare YOLOv8 classification dataset")
    parser.add_argument("--flow", type=str, default="1",
                        help="Which flow: 1, 2, or raw (original images)")
    parser.add_argument("--split", type=float, nargs=3, default=[0.7, 0.2, 0.1],
                        help="Train/val/test split ratios")
    args = parser.parse_args()

    train_r, val_r, test_r = args.split
    assert abs(train_r + val_r + test_r - 1.0) < 1e-6, "Split ratios must sum to 1.0"

    # Determine source directories
    if args.flow == "1":
        pos_dir = "./Positiva1"
        neg_dir = "./Negativa1"
        dataset_name = "dataset_cracks_flow1"
    elif args.flow == "2":
        pos_dir = "./Positiva2"
        neg_dir = "./Negativa2"
        dataset_name = "dataset_cracks_flow2"
    else:
        pos_dir = "./Pos"
        neg_dir = "./Neg"
        dataset_name = "dataset_cracks_raw"

    # Collect images
    pos_paths = glob.glob(os.path.join(pos_dir, "*.jpg")) + glob.glob(os.path.join(pos_dir, "*.png"))
    neg_paths = glob.glob(os.path.join(neg_dir, "*.jpg")) + glob.glob(os.path.join(neg_dir, "*.png"))

    pos_images = collect_images(pos_paths, "crack")
    neg_images = collect_images(neg_paths, "no_crack")

    print(f"Positivas (crack): {len(pos_images)} imagens")
    print(f"Negativas (no_crack): {len(neg_images)} imagens")

    # Build dataset structure
    base_dir = dataset_name
    for subdir in ["train", "val", "test"]:
        os.makedirs(os.path.join(base_dir, subdir, "crack"), exist_ok=True)
        os.makedirs(os.path.join(base_dir, subdir, "no_crack"), exist_ok=True)

    # Split each class separately
    split_and_copy(pos_images, base_dir, "crack", train_r, val_r, test_r)
    split_and_copy(neg_images, base_dir, "no_crack", train_r, val_r, test_r)

    # Count results
    for split in ["train", "val", "test"]:
        crack_count = len(glob.glob(os.path.join(base_dir, split, "crack", "*")))
        no_crack_count = len(glob.glob(os.path.join(base_dir, split, "no_crack", "*")))
        print(f"  {split}: crack={crack_count}, no_crack={no_crack_count}")

    print(f"\nDataset pronto: {dataset_name}/")
    print("Estrutura:")
    print(f"  {dataset_name}/")
    print(f"    train/crack/")
    print(f"    train/no_crack/")
    print(f"    val/crack/")
    print(f"    val/no_crack/")
    print(f"    test/crack/")
    print(f"    test/no_crack/")


if __name__ == "__main__":
    main()
