"""
Train YOLOv8 classification model for crack detection.

Usage:
    python train_crack_classifier.py --dataset dataset_cracks_flow1 --model yolov8n-cls --epochs 50
    python train_crack_classifier.py --dataset dataset_cracks_flow2 --model yolov8s-cls --epochs 30

Args:
    --dataset: path to prepared dataset (train/ val/ test/ with crack/ no_crack/)
    --model: pretrained cls model (yolov8n-cls.pt, yolov8s-cls.pt, yolov8m-cls.pt)
    --epochs: number of training epochs
    --imgsz: image size for training
    --batch: batch size
"""

import argparse
import os
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 crack classifier")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset directory with train/ val/ test/")
    parser.add_argument("--model", type=str, default="yolov8n-cls.pt",
                        help="Pretrained model: yolov8n-cls.pt, yolov8s-cls.pt, yolov8m-cls.pt")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=256,
                        help="Image size for training")
    parser.add_argument("--batch", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--name", type=str, default=None,
                        help="Run name (default: auto-generated)")
    args = parser.parse_args()

    # Verify dataset exists
    for subdir in ["train", "val"]:
        for cls in ["crack", "no_crack"]:
            cls_path = os.path.join(args.dataset, subdir, cls)
            if not os.path.isdir(cls_path):
                print(f"ERRO: Pasta não encontrada: {cls_path}")
                print("Execute prepare_dataset.py primeiro.")
                return

    # Count images
    train_crack = len([f for f in os.listdir(os.path.join(args.dataset, "train", "crack"))
                       if f.endswith(('.jpg', '.png'))])
    train_no_crack = len([f for f in os.listdir(os.path.join(args.dataset, "train", "no_crack"))
                          if f.endswith(('.jpg', '.png'))])
    print(f"Dataset: {args.dataset}")
    print(f"  Train: crack={train_crack}, no_crack={train_no_crack}")
    print(f"  Modelo: {args.model}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Image size: {args.imgsz}")
    print(f"  Batch: {args.batch}")
    print()

    # Load pretrained classification model
    model = YOLO(args.model)

    # Train
    results = model.train(
        data=args.dataset,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project="runs_crack",
        name=args.name or f"{os.path.basename(args.dataset)}_{args.model.replace('.pt', '')}",
        patience=10,
        amp=True,
    )

    print("\nTreino concluído!")
    print(f"Melhor modelo: runs_crack/{results.name}/weights/best.pt")


if __name__ == "__main__":
    main()
