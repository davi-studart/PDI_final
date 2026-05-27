# PDI Final — Classificação de Rachaduras em Pisos

Processamento Digital de Imagens com pipelines de filtros em frequência e classificação via YOLOv8.

## Estrutura

```
main.ipynb          — Notebook com pipelines de processamento (Fluxo 1 e 2)
prepare_dataset.py  — Prepara dataset para YOLOv8 (converte grayscale → RGB, split train/val/test)
train_crack_classifier.py  — Treina classificador YOLOv8-cls
infer_crack.py             — Inferência em imagens individuais ou diretórios
evaluate_models.py         — Avalia e compara todos os modelos treinados
```

### Pastas de dados

| Pasta | Descrição |
|---|---|
| `Pos/` | 500 imagens originais COM rachadura |
| `Neg/` | 500 imagens originais SEM rachadura |
| `Positiva1/` | Imagens Pos processadas pelo Fluxo 1 |
| `Negativa1/` | Imagens Neg processadas pelo Fluxo 1 |
| `Positiva2/` | Imagens Pos processadas pelo Fluxo 2 |
| `Negativa2/` | Imagens Neg processadas pelo Fluxo 2 |
| `dataset_cracks_*` | Datasets preparados para treino (gerado) |
| `runs/classify/runs_crack/` | Modelos treinados (gerado) |

## Setup

```bash
source .venv/bin/activate   # Python 3.11, criado com uv 0.11.15
pip install ultralytics     # Dependência adicional para YOLOv8
```

Dependências principais: `numpy`, `matplotlib`, `opencv-python`, `ultralytics` (torch, torchvision, pyyaml, scipy).

## Pipeline de Processamento (main.ipynb)

Abrir o notebook e executar as células de cima para baixo.

### Fluxo 1 — Low-pass + Canny
```
blur(img) → Canny(100, 200) → morphological close (3x3, 7 iterações)
```

### Fluxo 2 — High-pass + Otsu
```
high_pass(img) → Otsu threshold → morphological close (3x3, 1 iteração)
```

Cada fluxo lê `Pos/*.jpg` + `Neg/*.jpg` e salva nas pastas `Positiva*` / `Negativa*`.

## YOLOv8 Classification

### 1. Preparar dataset

```bash
python prepare_dataset.py --flow 1 --split 0.7 0.2 0.1   # usa Positiva1/Negativa1
python prepare_dataset.py --flow 2 --split 0.7 0.2 0.1   # usa Positiva2/Negativa2
python prepare_dataset.py --flow raw --split 0.7 0.2 0.1  # usa Pos/Neg originais
```

Cria `dataset_cracks_flow1/`, `dataset_cracks_flow2/` ou `dataset_cracks_raw/`:
```
<dataset>/
  train/crack/    train/no_crack/
  val/crack/      val/no_crack/
  test/crack/     test/no_crack/
```

### 2. Treinar classificador

```bash
python train_crack_classifier.py --dataset dataset_cracks_flow1 --model yolov8n-cls.pt --epochs 50
```

Opções:
- `--model`: `yolov8n-cls.pt` (nano), `yolov8s-cls.pt` (small), `yolov8m-cls.pt` (medium)
- `--epochs`: número de épocas (default 50)
- `--imgsz`: tamanho da imagem (default 256)
- `--batch`: batch size (default 32)
- `--name`: nome personalizado para a execução

Saída: `runs/classify/runs_crack/<run_name>/weights/best.pt`

### 3. Inferência

```bash
# Imagem única
python infer_crack.py --model runs/classify/runs_crack/<run_name>/weights/best.pt --image teste.jpg

# Diretório inteiro
python infer_crack.py --model runs/classify/runs_crack/<run_name>/weights/best.pt --dir test_images/
```

Classes: `crack` (rachadura) e `no_crack` (sem rachadura).

### 4. Avaliar e comparar modelos

```bash
python evaluate_models.py
```

Testa cada modelo no próprio dataset (intra) e nos outros (cross), reporta top-1 accuracy e declara o melhor modelo.

## Notas Técnicas

- Variáveis e nomes de células em português (`fechamento`, `pasta_positiva`).
- Todas as imagens lidas em grayscale (`cv2.IMREAD_GRAYSCALE`).
- Filtros DFT: convertem para `float32`, aplicam máscara no domínio da frequência, convertem de volta para `uint8`.
- `prepare_dataset.py` converte grayscale → 3 canais RGB via `cv2.cvtColor(gray, COLOR_GRAY2BGR)`.
- Classes do YOLO: `0: crack`, `1: no_crack`.
- No detection mode — bounding box annotations seriam necessárias para detecção.

## Resultados (modelo atual)

Melhor modelo: `flow2_run1` — média 82.4% across all datasets (intra + cross).

Intra-dataset: ~99% em todos os modelos. Cross-domain: modelo `raw` generaliza melhor para flow1 (90.8%).
