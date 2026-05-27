"""
Evaluate and compare trained YOLOv8 crack classifiers.

Tests each model on its own dataset (intra) and on the other datasets (cross).
Generates a detailed comparison report.

Usage:
    python evaluate_models.py
"""

import os
from ultralytics import YOLO


MODELS = {
    "flow1_run1": "runs/classify/runs_crack/dataset_cracks_flow1_yolov8n-cls/weights/best.pt",
    "flow1_run2": "runs/classify/runs_crack/dataset_cracks_flow1_yolov8n-cls-2/weights/best.pt",
    "flow2_run1": "runs/classify/runs_crack/dataset_cracks_flow2_yolov8n-cls/weights/best.pt",
    "flow2_run2": "runs/classify/runs_crack/dataset_cracks_flow2_yolov8n-cls-2/weights/best.pt",
    "raw_run1":   "runs/classify/runs_crack/dataset_cracks_raw_yolov8n-cls/weights/best.pt",
}

DATASETS = {
    "flow1": "dataset_cracks_flow1",
    "flow2": "dataset_cracks_flow2",
    "raw":   "dataset_cracks_raw",
}


def evaluate_model(model_path, datasets):
    """Evaluate a model on a list of datasets, return dict of results."""
    if not os.path.isfile(model_path):
        print(f"AVISO: Modelo não encontrado: {model_path}")
        return {}

    model = YOLO(model_path)
    results = {}

    for name, path in datasets.items():
        val_path = os.path.join(path, "val")
        if not os.path.isdir(val_path):
            continue
        res = model.val(data=path, split="val")
        results[name] = {
            "top1": res.top1,
            "top5": res.top5,
            "top1_acc": res.top1,
        }

    return results


def main():
    print("=" * 70)
    print("  AVALIAÇÃO E COMPARAÇÃO — MODELOS DE CLASSIFICAÇÃO DE RACHADURAS")
    print("=" * 70)
    print()

    # Evaluate each model
    all_results = {}
    for model_name, model_path in MODELS.items():
        print(f"Avaliando modelo: {model_name}...")
        results = evaluate_model(model_path, DATASETS)
        all_results[model_name] = results
        print()

    # Print detailed table
    print("=" * 70)
    print("  Acurácia Top-1 por modelo e dataset")
    print("=" * 70)
    print()

    # Header
    header = f"{'Modelo':<15}"
    for ds_name in DATASETS:
        header += f"  {ds_name:<12}"
    print(header)
    print("-" * len(header))

    # Rows
    for model_name in MODELS:
        row = f"{model_name:<15}"
        for ds_name in DATASETS:
            acc = all_results[model_name].get(ds_name, {}).get("top1_acc", None)
            if acc is not None:
                row += f"  {acc:.4f}  ({acc*100:.1f}%)"
            else:
                row += f"  {'N/A':<19}"
        print(row)

    print()

    # Best per dataset
    print("=" * 70)
    print("  Melhor modelo por dataset")
    print("=" * 70)
    print()

    for ds_name in DATASETS:
        best_model = None
        best_acc = -1
        for model_name in MODELS:
            acc = all_results[model_name].get(ds_name, {}).get("top1_acc", None)
            if acc is not None and acc > best_acc:
                best_acc = acc
                best_model = model_name

        if best_model:
            print(f"  {ds_name:<10} -> {best_model:<15}  top1={best_acc:.4f} ({best_acc*100:.1f}%)")

    print()

    # Cross-dataset analysis
    print("=" * 70)
    print("  Análise cruzada (modelo treinado em X, testado em Y)")
    print("=" * 70)
    print()

    ds_names = list(DATASETS.keys())
    for train_ds in ds_names:
        print(f"  Treinado em: {train_ds}")
        for test_ds in ds_names:
            if train_ds == test_ds:
                continue
            for model_name in MODELS:
                if train_ds not in model_name:
                    continue
                acc = all_results[model_name].get(test_ds, {}).get("top1_acc", None)
                if acc is not None:
                    marker = " *" if acc >= 0.9 else ""
                    print(f"    {model_name:<15} -> {test_ds:<10}  top1={acc:.4f} ({acc*100:.1f}%){marker}")
        print()

    # Overall best
    print("=" * 70)
    print("  Resumo geral")
    print("=" * 70)
    print()

    overall_best_model = None
    overall_best_acc = -1
    for model_name in MODELS:
        accs = [v["top1_acc"] for v in all_results[model_name].values() if v.get("top1_acc") is not None]
        if accs:
            avg_acc = sum(accs) / len(accs)
            print(f"  {model_name:<15}  média_across_datasets={avg_acc:.4f} ({avg_acc*100:.1f}%)")
            if avg_acc > overall_best_acc:
                overall_best_acc = avg_acc
                overall_best_model = model_name

    print()
    if overall_best_model:
        print(f"  >>> MELHOR MODEL: {overall_best_model}")
        print(f"      Acumédia across all datasets: {overall_best_acc:.4f} ({overall_best_acc*100:.1f}%)")

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
