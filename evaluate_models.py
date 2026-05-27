"""
Evaluate and compare trained YOLOv8 crack classifiers.

Tests each model on its own dataset (intra) and on the other datasets (cross).
Saves a PDF report with all results.

Usage:
    python evaluate_models.py
"""

import os
from ultralytics import YOLO
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


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

DS_NAMES = list(DATASETS.keys())
MODEL_NAMES = list(MODELS.keys())


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


def build_table_data(all_results):
    """Build a matrix top1[model_idx][ds_idx] and find best per dataset and overall."""
    n_models = len(MODEL_NAMES)
    n_ds = len(DS_NAMES)
    top1 = [[None] * n_ds for _ in range(n_models)]
    for i, m in enumerate(MODEL_NAMES):
        for j, d in enumerate(DS_NAMES):
            top1[i][j] = all_results[m].get(d, {}).get("top1_acc", None)

    best_per_ds = []
    for j in range(n_ds):
        best_val = -1
        best_idx = -1
        for i in range(n_models):
            v = top1[i][j]
            if v is not None and v > best_val:
                best_val = v
                best_idx = i
        best_per_ds.append((best_idx, best_val))

    averages = []
    for i in range(n_models):
        vals = [top1[i][j] for j in range(n_ds) if top1[i][j] is not None]
        averages.append(sum(vals) / len(vals) if vals else 0)

    best_overall = max(range(n_models), key=lambda i: averages[i])

    return top1, best_per_ds, averages, best_overall


def save_pdf(all_results):
    """Generate and save a PDF report with all evaluation results."""
    top1, best_per_ds, averages, best_overall = build_table_data(all_results)

    n_models = len(MODEL_NAMES)
    n_ds = len(DS_NAMES)

    with PdfPages("relatorio_modelos.pdf") as pdf:
        # --- Page 1: Acurácia Top-1 ---
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis("off")
        title = "Relatório de Avaliação — Classificação de Rachaduras\n"
        ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

        cell_text = []
        row_labels = []
        for i, m in enumerate(MODEL_NAMES):
            row = []
            for j, d in enumerate(DS_NAMES):
                v = top1[i][j]
                if v is not None:
                    row.append(f"{v*100:.1f}%")
                else:
                    row.append("N/A")
            cell_text.append(row)
            row_labels.append(m)

        col_labels = [f"Dataset {d}" for d in DS_NAMES]

        table = ax.table(
            cellText=cell_text,
            rowLabels=row_labels,
            colLabels=col_labels,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.6)

        for key, cell in table.get_celld().items():
            r, c = key
            if r == 0:
                cell.set_facecolor("#40466e")
                cell.set_text_props(color="white", fontweight="bold")
            elif c == -1:
                cell.set_facecolor("#e0e0e0")
                cell.set_text_props(fontweight="bold")
            else:
                cell.set_facecolor("#f5f5f5")

        # Highlight best per column
        for j, (best_i, _) in enumerate(best_per_ds):
            cell = table[(best_i + 1, j)]
            cell.set_facecolor("#a3e4d7")
            cell.set_text_props(fontweight="bold")

        ax.text(
            0.5, -0.08,
            "Células em verde: melhor modelo para aquele dataset",
            transform=ax.transAxes,
            ha="center", fontsize=9, style="italic",
        )

        pdf.savefig(fig, bbox_inches="tight")
        plt.close()

        # --- Page 2: Best per dataset + Overall ---
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.axis("off")
        ax.set_title("Melhor Modelo por Dataset", fontsize=14, fontweight="bold", pad=20)

        lines = []
        for j, (best_i, best_val) in enumerate(best_per_ds):
            lines.append(f"Dataset {DS_NAMES[j]:<10} →  {MODEL_NAMES[best_i]:<15}  "
                         f"top-1 = {best_val*100:.1f}%")

        lines.append("")
        lines.append("Média por modelo (across all datasets):")
        for i in range(n_models):
            lines.append(f"  {MODEL_NAMES[i]:<15}  média = {averages[i]*100:.1f}%")

        lines.append("")
        lines.append(f">>> MELHOR MODELO GERAL: {MODEL_NAMES[best_overall]}  "
                     f"(média {averages[best_overall]*100:.1f}%)")

        y = 0.85
        for line in lines:
            if line.startswith(">>>"):
                ax.text(0.1, y, line, fontsize=11, fontweight="bold", color="#c0392b")
            elif "Média" in line or "Melhor" in line:
                ax.text(0.1, y, line, fontsize=10, fontweight="bold")
            else:
                ax.text(0.1, y, line, fontsize=10)
            y -= 0.055

        pdf.savefig(fig, bbox_inches="tight")
        plt.close()

        # --- Page 3: Cross-dataset analysis ---
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.axis("off")
        ax.set_title("Análise Cruzada", fontsize=14, fontweight="bold", pad=20)

        lines = []
        for train_ds in DS_NAMES:
            lines.append(f"Treinado em: {train_ds}")
            for test_ds in DS_NAMES:
                if train_ds == test_ds:
                    continue
                for model_name in MODEL_NAMES:
                    if train_ds not in model_name:
                        continue
                    v = all_results[model_name].get(test_ds, {}).get("top1_acc", None)
                    if v is not None:
                        star = " ★" if v >= 0.9 else ""
                        lines.append(f"  {model_name:<15}  →  {test_ds:<10}  "
                                     f"top-1 = {v*100:.1f}%{star}")
            lines.append("")

        lines.append("★ = boa generalização (≥90%)")

        y = 0.92
        for line in lines:
            if "★" in line and "boa" not in line:
                ax.text(0.05, y, line, fontsize=9, color="#1a5276")
            elif "Treinado" in line:
                ax.text(0.05, y, line, fontsize=10, fontweight="bold")
            else:
                ax.text(0.05, y, line, fontsize=9)
            y -= 0.035

        pdf.savefig(fig, bbox_inches="tight")
        plt.close()

    print(f"\nRelatório salvo: relatorio_modelos.pdf")


def print_report(all_results):
    """Print a text summary to stdout."""
    top1, best_per_ds, averages, best_overall = build_table_data(all_results)

    print("=" * 70)
    print("  AVALIAÇÃO E COMPARAÇÃO — MODELOS DE CLASSIFICAÇÃO DE RACHADURAS")
    print("=" * 70)
    print()

    print(f"{'Modelo':<15}", end="")
    for d in DS_NAMES:
        print(f"  {d:<12}", end="")
    print()
    print("-" * (15 + 14 * len(DS_NAMES)))

    for i, m in enumerate(MODEL_NAMES):
        print(f"{m:<15}", end="")
        for j in range(len(DS_NAMES)):
            v = top1[i][j]
            if v is not None:
                print(f"  {v:.4f}  ({v*100:.1f}%)", end="")
            else:
                print(f"  {'N/A':<19}", end="")
        print()
    print()

    print("=" * 70)
    print("  Melhor modelo por dataset")
    print("=" * 70)
    for j, (best_i, best_val) in enumerate(best_per_ds):
        print(f"  {DS_NAMES[j]:<10} -> {MODEL_NAMES[best_i]:<15}  "
              f"top1={best_val:.4f} ({best_val*100:.1f}%)")
    print()

    print("=" * 70)
    print("  Resumo geral")
    print("=" * 70)
    for i in range(len(MODEL_NAMES)):
        print(f"  {MODEL_NAMES[i]:<15}  média={averages[i]:.4f} ({averages[i]*100:.1f}%)")
    print()
    print(f"  >>> MELHOR MODELO: {MODEL_NAMES[best_overall]}  "
          f"(média {averages[best_overall]*100:.1f}%)")
    print("=" * 70)


def main():
    all_results = {}
    for model_name, model_path in MODELS.items():
        print(f"Avaliando modelo: {model_name}...")
        results = evaluate_model(model_path, DATASETS)
        all_results[model_name] = results
        print()

    print_report(all_results)
    save_pdf(all_results)


if __name__ == "__main__":
    main()
