# PDI_final

Single-file Jupyter notebook project (`main.ipynb`) — digital image processing (PDI) with two pipelines.

## Setup

```bash
source .venv/bin/activate   # Python 3.11, created with uv 0.11.15
```

Dependencies: `numpy`, `matplotlib`, `opencv-python` (no `requirements.txt` — seed venv).

## Architecture

Everything lives in `main.ipynb`. There is no package structure, no tests, no CI.

| Cell section | Function | Pipeline |
|---|---|---|
| Low pass filter | `blur(img)` | Gaussian low-pass in DFT domain → edge detection |
| High pass filter | `high_pass(img)` | High-pass in DFT domain → Otsu thresholding |
| Fluxo 1 | `process_image1(img)` | `blur` → Canny(100,200) → morphological close (3x3, 7 iters) |
| Fluxo 2 | `process_image2(img)` | `high_pass` → Otsu binary → morphological close (3x3, 1 iter) |

## Running

Open `main.ipynb` and run cells top-to-bottom. The last two code cells process all JPGs:

- **Fluxo 1:** reads `Pos/*.jpg` + `Neg/*.jpg` → writes `Positiva1/` + `Negativa1/`
- **Fluxo 2:** reads `Pos/*.jpg` + `Neg/*.jpg` → writes `Positiva2/` + `Negativa2/`

Each input dir has 500 images. Output dirs are gitignored.

## Conventions

- Portuguese variable/cell names throughout (e.g. `fechamento`, `pasta_positiva`).
- All images read in grayscale (`cv2.IMREAD_GRAYSCALE`).
- DFT-based filters convert to `float32`, apply mask in frequency domain, then back to `uint8`.

## YOLOv8 Classification

Dependencies: `ultralytics` (installed via `pip install ultralytics` in venv). Adds `torch`, `torchvision`, `pyyaml`, `scipy`.

### Workflow

1. **Prepare dataset** — converts grayscale processed images to RGB and splits into train/val/test:
   ```bash
   python prepare_dataset.py --flow 1 --split 0.7 0.2 0.1   # uses Positiva1/Negativa1
   python prepare_dataset.py --flow 2 --split 0.7 0.2 0.1   # uses Positiva2/Negativa2
   python prepare_dataset.py --flow raw --split 0.7 0.2 0.1  # uses Pos/Neg originals
   ```
   Creates `dataset_cracks_flow1/`, `dataset_cracks_flow2/`, or `dataset_cracks_raw/` with:
   ```
   <dataset>/
     train/crack/  train/no_crack/
     val/crack/    val/no_crack/
     test/crack/   test/no_crack/
   ```

2. **Train classifier** — fine-tunes pretrained ImageNet cls model:
   ```bash
   python train_crack_classifier.py --dataset dataset_cracks_flow1 --model yolov8n-cls.pt --epochs 50
   ```
   Outputs to `runs/classify/runs_crack/<run_name>/weights/best.pt`.

3. **Inference** — classify single image or directory:
   ```bash
   python infer_crack.py --model runs/classify/runs_crack/<run_name>/weights/best.pt --image teste.jpg
   python infer_crack.py --model runs/classify/runs_crack/<run_name>/weights/best.pt --dir test_images/
   ```

### Notes

- Images from the PDI pipeline are grayscale; `prepare_dataset.py` converts them to 3-channel RGB via `cv2.cvtColor(gray, COLOR_GRAY2BGR)`.
- Classes: `0: crack`, `1: no_crack`.
- Default model is `yolov8n-cls.pt` (nano). Use `yolov8s-cls.pt` or `yolov8m-cls.pt` for larger models.
- Output dirs (`dataset_cracks_*`, `runs/classify/runs_crack/`) are gitignored.
- No detection mode yet — bounding box annotations would be needed for that.

### Evaluation

Compare all trained models on all datasets:
```bash
python evaluate_models.py
```
Tests each model on its own dataset (intra) and the other datasets (cross), reports top-1 accuracy and declares the best model.
