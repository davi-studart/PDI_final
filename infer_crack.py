"""
Inference script for trained crack classifier.

Usage:
    python infer_crack.py --model runs_crack/dataset_cracks_flow1_yolov8n-cls/weights/best.pt --image teste.jpg
    python infer_crack.py --model runs_crack/.../best.pt --dir test_images/

Args:
    --model: path to best.pt weights
    --image: single image path
    --dir: directory of images to classify
"""

import argparse
import os
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Inference for crack classifier")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to best.pt weights")
    parser.add_argument("--image", type=str, default=None,
                        help="Single image to classify")
    parser.add_argument("--dir", type=str, default=None,
                        help="Directory of images to classify")
    args = parser.parse_args()

    if not os.path.isfile(args.model):
        print(f"ERRO: Modelo não encontrado: {args.model}")
        return

    model = YOLO(args.model)
    print(f"Modelo carregado: {args.model}")
    print(f"Nomes das classes: {model.names}")
    print()

    if args.image:
        results = model(args.image, verbose=True)
        for r in results:
            probs = r.probs
            print(f"Imagem: {args.image}")
            print(f"  Classe: {probs.names.get(probs.top1, 'desconhecida')}")
            print(f"  Confiança: {probs.top1conf:.4f}")
            for idx, conf in zip(probs.top5, probs.top5conf):
                cls_name = probs.names.get(idx, "desconhecida")
                print(f"    {cls_name}: {conf:.4f}")

    elif args.dir:
        if not os.path.isdir(args.dir):
            print(f"ERRO: Diretório não encontrado: {args.dir}")
            return
        img_files = [f for f in os.listdir(args.dir)
                     if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        if not img_files:
            print(f"Nenhuma imagem encontrada em {args.dir}")
            return
        print(f"{len(img_files)} imagens em {args.dir}:\n")
        for fname in sorted(img_files):
            img_path = os.path.join(args.dir, fname)
            results = model(img_path, verbose=False)
            for r in results:
                probs = r.probs
                pred = probs.names.get(probs.top1, "?")
                conf = probs.top1conf
                label = "RACHADURA" if pred == "crack" else "SEM RACHADURA"
                print(f"  {fname}: {label} ({conf:.4f})")
    else:
        print("ERRO: Especifique --image ou --dir")


if __name__ == "__main__":
    main()
